"""Core logic for the project-sustainability diagnosis GPT action.

This module is intentionally Streamlit-free so the sustainability rubric can be
used by the Custom GPT action backend without importing the large Streamlit app.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import pandas as pd

RUBRIC_FILENAME = "Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx"
SHEET_NAME = "rubric"
MODEL = "gpt-5-mini"
MAX_WORKERS = 8
MAX_COMPLETION_TOKENS = 6500
MAX_DOCUMENT_CHARS = 420_000

DIMENSION_ORDER = {"Diseño": 0, "Implementación": 1, "Evaluación": 2}

TECHNICAL_COLUMNS = [
    "Dimensión",
    "Criterio",
    "Indicador",
    "Nivel 0",
    "Nivel 1",
    "Nivel 2",
    "Nivel 3",
    "Score",
    "Análisis",
    "Evidencia",
    "Status",
    "Error",
]

PUBLIC_COLUMNS = [
    "Dimensión",
    "Criterio",
    "Indicador",
    "Puntuación (0-3)",
    "Lectura rápida",
    "Principal oportunidad de mejora",
    "Evidencia clave",
    "Revisión humana recomendada",
]

SUSTAINABILITY_RESPONSE_SCHEMA = {
    "name": "sustainability_rubric_eval",
    "schema": {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "score": {"type": "integer", "enum": [0, 1, 2, 3]},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["analysis", "score", "evidence"],
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM_PROMPT = """Eres un evaluador experto de documentos de proyecto de la OIT.

Evalúas un indicador de sostenibilidad contra el texto del documento usando una escala estricta de 0 a 3. La rúbrica enviada en el prompt es la fuente de verdad.

Reglas:
- Usa únicamente evidencia presente en el documento. No infieras cumplimiento por intención general.
- El puntaje debe ser exactamente uno de: 0, 1, 2, 3.
- Selecciona el nivel más alto cuya descripción esté sustentada por evidencia clara.
- Si la evidencia es insuficiente, asigna 0 y explica qué falta.
- Cita evidencia textual breve y localizable. Si no hay evidencia, indícalo.
- No presentes el resultado como una determinación oficial de la OIT; es una valoración asistida que requiere validación experta.
- Responde siempre en español, incluso si el documento está en inglés."""

COUNTING_INSTRUCTIONS = """Este indicador puede depender de conteos o umbrales de participación de mandantes/actores.

Aplica estas reglas adicionales:
- Cuenta por separado Gobierno, empleadores y trabajadores cuando corresponda.
- Cuenta solo participación factual, pasada o presente: participó, asistió, validó, revisó, comentó, aportó información, forma parte de.
- No cuentes participación futura, planificada, esperada, prometida o hipotética: participará, se consultará, se espera, se prevé, se invitará, podría participar.
- Para cada actor, enumera las formas de participación verificables y excluye explícitamente las menciones futuras o planificadas.
- Aplica los umbrales de cada nivel tal como están escritos en la rúbrica."""

USER_PROMPT_TEMPLATE = """Indicador a evaluar:
Dimensión: {dimension}
Criterio: {criterio}
Indicador: {indicador}

Niveles de puntuación (escala 0-3):
Nivel 0: {nivel_0}
Nivel 1: {nivel_1}
Nivel 2: {nivel_2}
Nivel 3: {nivel_3}

{counting_instructions}

Documento a evaluar:
{document_text}

Devuelve JSON estricto con:
{{
  "analysis": "2-4 párrafos en español explicando el puntaje y las brechas",
  "score": 0|1|2|3,
  "evidence": ["cita breve 1", "cita breve 2"]
}}"""


def _resolve_rubric_path(rubric_path: str | Path = RUBRIC_FILENAME) -> Path:
    path = Path(rubric_path)
    if path.exists():
        return path

    matches = list(Path(".").glob("Evaluaci*n de sostenibilidad del proyecto_rubric_9feb26.xlsx"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find sustainability rubric: {rubric_path}")


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _indicator_id(indicador: Any) -> str:
    match = re.match(r"\s*\(?(\d+(?:\.\d+)*)", str(indicador or ""))
    return match.group(1) if match else ""


def _sort_key(indicador: Any) -> tuple[int, ...]:
    indicator_id = _indicator_id(indicador)
    if not indicator_id:
        return (999, 999, 999)
    return tuple(int(part) for part in indicator_id.split(".") if part.isdigit())


def _truncate_document_text(document_text: str) -> str:
    text = document_text or ""
    if len(text) <= MAX_DOCUMENT_CHARS:
        return text
    return text[:MAX_DOCUMENT_CHARS] + "\n\n[Documento truncado por límite de contexto.]"


def extract_docx_text_from_path(docx_path: str) -> str:
    """Extract full text from a DOCX path."""
    try:
        from docx2python import docx2python
    except ImportError as exc:
        raise RuntimeError(
            "docx2python is required for DOCX extraction. Install project requirements before running evaluations."
        ) from exc

    result = docx2python(docx_path)
    return result.text or ""


def extract_docx_text_from_bytes(docx_bytes: bytes) -> str:
    """Extract full text from DOCX bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(docx_bytes)
        tmp_path = tmp.name
    try:
        return extract_docx_text_from_path(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def load_sustainability_rubric(rubric_path: str | Path = RUBRIC_FILENAME) -> pd.DataFrame:
    """Load the sustainability rubric and return one row per indicator."""
    path = _resolve_rubric_path(rubric_path)
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    required = ["Dimensión", "Criterio", "Indicador", "Nivel 0", "Nivel 1", "Nivel 2", "Nivel 3"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Sustainability rubric is missing required columns: {missing}")

    df = df.dropna(subset=["Indicador"]).copy()
    for col in required:
        df[col] = df[col].map(_clean_text)
    df["_indicator_id"] = df["Indicador"].map(_indicator_id)
    df["_dim_order"] = df["Dimensión"].map(lambda x: DIMENSION_ORDER.get(x, 99))
    df["_sort_key"] = df["Indicador"].map(_sort_key)
    return df.reset_index(drop=True)


def filter_rubric(
    df_rubric: pd.DataFrame,
    dimensions: list[str] | None = None,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    """Filter rubric rows by dimension and/or indicator id."""
    df = df_rubric.copy()
    if dimensions:
        wanted = {str(dim).strip().casefold() for dim in dimensions if str(dim).strip()}
        df = df[df["Dimensión"].str.casefold().isin(wanted)]
    if indicators:
        wanted_ids = {str(ind).strip().rstrip(".") for ind in indicators if str(ind).strip()}
        df = df[df["_indicator_id"].isin(wanted_ids)]
    return (
        df.sort_values(by=["_dim_order", "_sort_key"])
        .reset_index(drop=True)
    )


def _is_counting_indicator(indicador: str) -> bool:
    lower = indicador.lower()
    return any(
        keyword in lower
        for keyword in [
            "mandante",
            "participan",
            "participación",
            "actores",
            "gobierno",
            "empleadores",
            "trabajadores",
        ]
    )


def build_user_prompt(row: pd.Series, document_text: str) -> str:
    """Build the single-indicator prompt."""
    indicador = _clean_text(row.get("Indicador", ""))
    counting_instructions = COUNTING_INSTRUCTIONS if _is_counting_indicator(indicador) else ""
    return USER_PROMPT_TEMPLATE.format(
        dimension=_clean_text(row.get("Dimensión", "")),
        criterio=_clean_text(row.get("Criterio", "")),
        indicador=indicador,
        nivel_0=_clean_text(row.get("Nivel 0", "")),
        nivel_1=_clean_text(row.get("Nivel 1", "")),
        nivel_2=_clean_text(row.get("Nivel 2", "")),
        nivel_3=_clean_text(row.get("Nivel 3", "")),
        counting_instructions=counting_instructions,
        document_text=_truncate_document_text(document_text),
    )


def evaluate_indicator(
    client: Any,
    row: pd.Series,
    document_text: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Evaluate one sustainability indicator."""
    base = {
        "Dimensión": _clean_text(row.get("Dimensión", "")),
        "Criterio": _clean_text(row.get("Criterio", "")),
        "Indicador": _clean_text(row.get("Indicador", "")),
        "Nivel 0": _clean_text(row.get("Nivel 0", "")),
        "Nivel 1": _clean_text(row.get("Nivel 1", "")),
        "Nivel 2": _clean_text(row.get("Nivel 2", "")),
        "Nivel 3": _clean_text(row.get("Nivel 3", "")),
    }

    if not document_text or not document_text.strip():
        return {
            **base,
            "Score": 0,
            "Análisis": "Documento vacío o sin texto extraíble.",
            "Evidencia": "",
            "Status": "Success",
            "Error": "",
        }

    last_error = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row, document_text)},
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort="minimal",
                timeout=120,
                response_format={
                    "type": "json_schema",
                    "json_schema": SUSTAINABILITY_RESPONSE_SCHEMA,
                },
            )
            content = (response.choices[0].message.content or "").strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                content = content.rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            evidence = parsed.get("evidence", [])
            if isinstance(evidence, list):
                evidence_text = "\n".join(str(item).strip() for item in evidence if str(item).strip())
            else:
                evidence_text = str(evidence or "")

            score = int(parsed.get("score", 0))
            score = max(0, min(score, 3))
            return {
                **base,
                "Score": score,
                "Análisis": str(parsed.get("analysis", "")),
                "Evidencia": evidence_text,
                "Status": "Success",
                "Error": "",
            }
        except Exception as exc:  # noqa: BLE001 - contain failures per indicator
            last_error = str(exc)
            if ("rate_limit" in last_error.lower() or "429" in last_error) and attempt < max_retries - 1:
                time.sleep((2 ** attempt) * 2)
                continue
            break

    return {
        **base,
        "Score": 0,
        "Análisis": f"Error durante la evaluación: {last_error}",
        "Evidencia": "",
        "Status": "Error",
        "Error": last_error,
    }


def evaluate_indicators(
    client: Any,
    df_indicators: pd.DataFrame,
    document_text: str,
    max_workers: int = MAX_WORKERS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all selected indicators."""
    total = len(df_indicators)
    if total == 0:
        return []

    workers = max(1, min(int(max_workers), total, MAX_WORKERS))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_indicator, client, row, document_text): idx
            for idx, row in df_indicators.iterrows()
        }
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if progress_callback:
                progress_callback(done, total)

    results.sort(key=lambda r: (DIMENSION_ORDER.get(r.get("Dimensión", ""), 99), _sort_key(r.get("Indicador", ""))))
    return results


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Return the technical audit dataframe."""
    if not results:
        return pd.DataFrame(columns=TECHNICAL_COLUMNS)
    df = pd.DataFrame(results)
    for col in TECHNICAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[TECHNICAL_COLUMNS]


def _compact_text(value: Any, max_chars: int = 700) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _score_reading(score: Any) -> str:
    try:
        value = int(score)
    except (TypeError, ValueError):
        value = 0
    mapping = {
        0: "No hay evidencia suficiente o no cumple el nivel mínimo.",
        1: "Cumplimiento inicial; requiere refuerzo sustantivo.",
        2: "Cumplimiento sustantivo, con brechas todavía relevantes.",
        3: "Cumplimiento completo según la rúbrica y la evidencia encontrada.",
    }
    return mapping.get(value, "Revisar la auditoría técnica.")


def _improvement_hint(score: Any) -> str:
    try:
        value = int(score)
    except (TypeError, ValueError):
        value = 0
    mapping = {
        0: "Incorporar evidencia específica para los elementos exigidos por la rúbrica.",
        1: "Completar los componentes faltantes y hacer explícita la evidencia en el documento.",
        2: "Cerrar las brechas señaladas para alcanzar el nivel máximo.",
        3: "Mantener la evidencia y verificar que las citas sean localizables.",
    }
    return mapping.get(value, "Revisar la auditoría técnica.")


def _review_flag(row: pd.Series) -> str:
    if str(row.get("Status", "")) == "Error":
        return "Sí - falla técnica"
    try:
        score = int(row.get("Score", 0))
    except (TypeError, ValueError):
        score = 0
    return "Sí - puntaje menor a 3" if score < 3 else "No - revisión muestral"


def results_to_public_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Return the user-facing dataframe."""
    technical = results_to_dataframe(results)
    if technical.empty:
        return pd.DataFrame(columns=PUBLIC_COLUMNS)

    public = technical.copy()
    public["Puntuación (0-3)"] = public["Score"]
    public["Lectura rápida"] = public["Score"].map(_score_reading)
    public["Principal oportunidad de mejora"] = public["Score"].map(_improvement_hint)
    public["Evidencia clave"] = public["Evidencia"].map(_compact_text)
    public["Revisión humana recomendada"] = public.apply(_review_flag, axis=1)
    return public[PUBLIC_COLUMNS]


def results_to_xlsx_bytes(results: list[dict[str, Any]]) -> bytes:
    """Serialize sustainability results to an XLSX workbook."""
    public_df = results_to_public_dataframe(results)
    technical_df = results_to_dataframe(results)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        public_df.to_excel(writer, index=False, sheet_name="Lectura amigable")
        technical_df.to_excel(writer, index=False, sheet_name="Auditoria tecnica")

        workbook = writer.book
        header_fmt = workbook.add_format(
            {"bold": True, "text_wrap": True, "valign": "top", "fg_color": "#D9EAF7", "border": 1}
        )
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

        sheet_specs = {
            "Lectura amigable": (
                public_df,
                {
                    "A:A": 16,
                    "B:C": 52,
                    "D:D": 16,
                    "E:G": 42,
                    "H:H": 28,
                },
            ),
            "Auditoria tecnica": (
                technical_df,
                {
                    "A:A": 16,
                    "B:C": 52,
                    "D:G": 56,
                    "H:H": 12,
                    "I:J": 60,
                    "K:L": 16,
                },
            ),
        }
        for sheet_name, (df_sheet, widths) in sheet_specs.items():
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, max(len(df_sheet), 1), max(len(df_sheet.columns) - 1, 0))
            for col_num, value in enumerate(df_sheet.columns):
                worksheet.write(0, col_num, value, header_fmt)
            for col_range, width in widths.items():
                worksheet.set_column(col_range, width, wrap_fmt)
    return buf.getvalue()


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a compact summary for GPT responses."""
    scores = [int(r.get("Score", 0)) for r in results if str(r.get("Status", "")) == "Success"]
    status_counts = Counter(r.get("Status", "Unknown") for r in results)
    score_counts = Counter(str(r.get("Score", 0)) for r in results)
    by_dimension: dict[str, dict[str, Any]] = {}

    for result in results:
        dim = str(result.get("Dimensión", "No especificada"))
        by_dimension.setdefault(dim, {"count": 0, "average_score": 0.0, "_scores": []})
        by_dimension[dim]["count"] += 1
        by_dimension[dim]["_scores"].append(int(result.get("Score", 0)))

    for dim, payload in by_dimension.items():
        dim_scores = payload.pop("_scores")
        payload["average_score"] = round(sum(dim_scores) / len(dim_scores), 2) if dim_scores else 0.0

    return {
        "total_indicators": len(results),
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "score_counts": dict(score_counts),
        "statuses": dict(status_counts),
        "by_dimension": by_dimension,
    }
