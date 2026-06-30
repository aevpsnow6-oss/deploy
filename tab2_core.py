"""Core logic for the Tab 2 specific-attributes diagnosis GPT action.

Streamlit-free port of the Tab 2 rubric evaluation in oli_v6_deploy.py, with the
5-run stability scheme added (see stability.py). The caller chooses which
rubric(s) to evaluate: participatory methods, gender, or just transition.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import stability

RUBRIC_FILENAME = "Rubricas_6ago2025.xlsx"
MODEL = "gpt-5-mini"
MAX_WORKERS = 8
MAX_COMPLETION_TOKENS = 6500
MAX_DOCUMENT_CHARS = 420_000

STABILITY_REPEATS = 5
STABILITY_THRESHOLD_PCT = 80.0
SCORE_ORDER = [1, 2, 3, 4, 5, "Error"]

# Selectable rubrics. Keys are the GPT-facing values; sheet is the xlsx tab.
RUBRICS: dict[str, dict[str, str]] = {
    "participatory": {
        "sheet": "rubric_parteval",
        "name": "Metodologías con enfoque participativo",
    },
    "gender": {
        "sheet": "rubric_gender_",
        "name": "Integración del enfoque de género",
    },
    "just_transition": {
        "sheet": "rubric_TJ_TJ",
        "name": "Transición Justa: enfoque moderno",
    },
}

TECHNICAL_COLUMNS = [
    "Rúbrica",
    "Dimensión",
    "Criterio",
    "Score",
    *stability.STABILITY_COLUMNS,
    "Análisis",
    "Evidencia",
    "Status",
    "Error",
]

PUBLIC_COLUMNS = [
    "Rúbrica",
    "Dimensión",
    "Criterio",
    "Puntuación (1-5)",
    "Estabilidad (%)",
    "Deriva principal (si inestable)",
    "Lectura rápida",
    "Principal oportunidad de mejora",
    "Evidencia clave",
    "Revisión humana recomendada",
]

RESPONSE_SCHEMA = {
    "name": "attribute_rubric_eval",
    "schema": {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["analysis", "score", "evidence"],
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM_PROMPT = (
    "Eres un evaluador experto de documentos. Siempre debes responder en español, "
    "incluso si el documento está en inglés."
)

USER_PROMPT_TEMPLATE = """Evalúa este documento contra el criterio: {criterion}

Niveles de puntuación (escala 1-5, en orden ascendente): {descriptions}

Documento a evaluar:
{document_text}

IMPORTANTE: Proporciona tu respuesta SIEMPRE en español, incluso si el documento está en inglés.

Devuelve JSON estricto con:
{{"analysis": "2-3 párrafos en español justificando el puntaje", "score": 1-5, "evidence": ["cita 1", "cita 2", "... 5-8 citas textuales del documento"]}}"""


def _resolve_rubric_path(rubric_path: str | Path = RUBRIC_FILENAME) -> Path:
    path = Path(rubric_path)
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not find attributes rubric: {rubric_path}")


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


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


def resolve_rubric_keys(rubrics: list[str] | None) -> list[str]:
    """Return the rubric keys to evaluate; default is all of them."""
    if not rubrics:
        return list(RUBRICS)
    keys = []
    for raw in rubrics:
        key = str(raw).strip().lower()
        if key not in RUBRICS:
            raise ValueError(
                f"Unknown rubric '{raw}'. Valid options: {', '.join(RUBRICS)}."
            )
        if key not in keys:
            keys.append(key)
    return keys


# Columns that are metadata, not scoring-level descriptions.
_META_COLS = {"Indicador", "Dimensión", "Criterio", "crit_short"}


def load_rubric(rubric_key: str, rubric_path: str | Path = RUBRIC_FILENAME) -> list[dict[str, Any]]:
    """Load one rubric sheet into a list of {criterio, descriptions, dimension}.

    Every column that is not metadata (see _META_COLS / "Unnamed*") is treated as
    a scoring-level description.
    """
    spec = RUBRICS[rubric_key]
    path = _resolve_rubric_path(rubric_path)
    df = pd.read_excel(path, sheet_name=spec["sheet"])

    criteria = []
    for _, row in df.iterrows():
        indicador = _clean_text(row.get("Indicador", ""))
        if not indicador:
            continue
        dimension = _clean_text(row.get("Dimensión", "")) or "No especificada"
        valores = [
            _clean_text(v)
            for col, v in row.items()
            if col not in _META_COLS and not str(col).startswith("Unnamed") and _clean_text(v)
        ]
        criteria.append(
            {"criterio": indicador, "descriptions": valores, "dimension": dimension}
        )
    return criteria


def build_user_prompt(criterion: str, descriptions: list[str], document_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        criterion=criterion,
        descriptions=json.dumps(descriptions, ensure_ascii=False),
        document_text=_truncate_document_text(document_text),
    )


def evaluate_criterion(client: Any, payload: dict[str, Any], document_text: str) -> dict[str, Any]:
    """Evaluate one criterion once. payload = {criterio, descriptions, dimension, rubrica}."""
    base = {
        "Rúbrica": payload.get("rubrica", ""),
        "Dimensión": payload.get("dimension", "No especificada"),
        "Criterio": payload.get("criterio", ""),
    }

    if not document_text or not document_text.strip():
        return {**base, "Score": "", "Análisis": "Documento vacío o sin texto extraíble.",
                "Evidencia": "", "Status": "Error", "Error": "Documento vacío."}

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(
                    payload["criterio"], payload["descriptions"], document_text
                )},
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            reasoning_effort="minimal",
            timeout=120,
            response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
        )
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            content = content.rsplit("```", 1)[0].strip()
        if not content:
            return {**base, "Score": "", "Análisis": "Respuesta vacía del modelo.",
                    "Evidencia": "", "Status": "Error", "Error": "Respuesta vacía."}

        parsed = json.loads(content)
        evidence = parsed.get("evidence", [])
        if isinstance(evidence, list):
            evidence_text = "\n".join(str(item).strip() for item in evidence if str(item).strip())
        else:
            evidence_text = str(evidence or "")
        score = max(1, min(int(parsed.get("score", 1)), 5))
        return {**base, "Score": score, "Análisis": str(parsed.get("analysis", "")),
                "Evidencia": evidence_text, "Status": "Success", "Error": ""}
    except Exception as exc:  # noqa: BLE001 - contain failures per criterion
        return {**base, "Score": "", "Análisis": f"Error durante la evaluación: {exc}",
                "Evidencia": "", "Status": "Error", "Error": str(exc)}


def _aggregate_criterion_runs(repeated_results: list[dict[str, Any]]) -> dict[str, Any]:
    agg = stability.aggregate_runs(
        repeated_results,
        value_key="Score",
        value_order=SCORE_ORDER,
        threshold_pct=STABILITY_THRESHOLD_PCT,
        evidence_key="Evidencia",
    )
    rep = agg["representative"]
    is_error = agg["modal"] == "Error"
    analysis = (
        f"{agg['reasoning_prefix']}\n\n"
        "Análisis representativo de una corrida con el resultado modal:\n"
        f"{rep.get('Análisis', '')}"
    ).strip()
    return {
        "Rúbrica": rep.get("Rúbrica", ""),
        "Dimensión": rep.get("Dimensión", "No especificada"),
        "Criterio": rep.get("Criterio", ""),
        "Score": "" if is_error else int(agg["modal"]),
        **agg["columns"],
        "Análisis": analysis,
        "Evidencia": rep.get("Evidencia", ""),
        "Status": "Error" if is_error else "Success",
        "Error": rep.get("Error", "") if is_error else "",
    }


def evaluate_rubrics(
    client: Any,
    rubric_keys: list[str],
    document_text: str,
    max_workers: int = MAX_WORKERS,
    progress_callback: Callable[[int, int], None] | None = None,
    rubric_path: str | Path = RUBRIC_FILENAME,
) -> list[dict[str, Any]]:
    """Evaluate every criterion in the selected rubrics with the 5-run scheme."""
    items: list[tuple[tuple[str, int], dict[str, Any]]] = []
    for rubric_key in rubric_keys:
        name = RUBRICS[rubric_key]["name"]
        for idx, crit in enumerate(load_rubric(rubric_key, rubric_path)):
            items.append((
                (rubric_key, idx),
                {
                    "criterio": crit["criterio"],
                    "descriptions": crit["descriptions"],
                    "dimension": crit["dimension"],
                    "rubrica": name,
                },
            ))

    return stability.evaluate_with_stability(
        items,
        lambda payload: evaluate_criterion(client, payload, document_text),
        lambda _key, runs: _aggregate_criterion_runs(runs),
        repeats=STABILITY_REPEATS,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )


def count_criteria(rubric_keys: list[str], rubric_path: str | Path = RUBRIC_FILENAME) -> int:
    """Total number of criteria across the selected rubrics (for progress total)."""
    return sum(len(load_rubric(key, rubric_path)) for key in rubric_keys)


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=TECHNICAL_COLUMNS)
    df = pd.DataFrame(results)
    for col in TECHNICAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[TECHNICAL_COLUMNS]


def _compact_text(value: Any, max_chars: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _score_reading(score: Any) -> str:
    try:
        value = int(score)
    except (TypeError, ValueError):
        return "Falla técnica; revisar o reintentar la evaluación."
    mapping = {
        1: "Cumplimiento muy bajo o ausente según la evidencia.",
        2: "Cumplimiento bajo; faltan elementos sustantivos.",
        3: "Cumplimiento medio; hay brechas relevantes.",
        4: "Cumplimiento alto; brechas menores.",
        5: "Cumplimiento completo según la rúbrica y la evidencia.",
    }
    return mapping.get(value, "Revisar la auditoría técnica.")


def _improvement_hint(score: Any) -> str:
    try:
        value = int(score)
    except (TypeError, ValueError):
        return "Reintentar la evaluación o revisar el documento/rúbrica."
    mapping = {
        1: "Incorporar evidencia específica para los elementos exigidos por la rúbrica.",
        2: "Desarrollar los componentes faltantes y hacerlos explícitos en el documento.",
        3: "Cerrar las brechas señaladas para subir de nivel.",
        4: "Afinar los detalles pendientes para alcanzar el nivel máximo.",
        5: "Mantener la evidencia y verificar que las citas sean localizables.",
    }
    return mapping.get(value, "Revisar la auditoría técnica.")


def _review_flag(row: pd.Series) -> str:
    reasons = []
    if str(row.get("Status", "")) == "Error":
        reasons.append("falla técnica")
    stability_pct = row.get("Estabilidad (%)")
    if stability_pct is not None and not pd.isna(stability_pct):
        try:
            if float(stability_pct) < STABILITY_THRESHOLD_PCT:
                reasons.append(f"estabilidad <{STABILITY_THRESHOLD_PCT:.0f}%")
        except (TypeError, ValueError):
            pass
    try:
        if int(row.get("Score", 0)) < 4:
            reasons.append("puntaje menor a 4")
    except (TypeError, ValueError):
        pass
    return "Sí - " + "; ".join(reasons) if reasons else "No - revisión muestral"


def results_to_public_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    technical = results_to_dataframe(results)
    if technical.empty:
        return pd.DataFrame(columns=PUBLIC_COLUMNS)
    public = technical.copy()
    public["Puntuación (1-5)"] = public["Score"]
    public["Lectura rápida"] = public["Score"].map(_score_reading)
    public["Principal oportunidad de mejora"] = public["Score"].map(_improvement_hint)
    public["Evidencia clave"] = public["Evidencia"].map(_compact_text)
    public["Revisión humana recomendada"] = public.apply(_review_flag, axis=1)
    return public[PUBLIC_COLUMNS]


def results_to_xlsx_bytes(results: list[dict[str, Any]]) -> bytes:
    """Serialize Tab 2 results to a friendly + technical two-sheet XLSX."""
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
            "Lectura amigable": (public_df, {"A:B": 30, "C:C": 52, "D:F": 18, "G:I": 42, "J:J": 28}),
            "Auditoria tecnica": (technical_df, {"A:B": 28, "C:C": 52, "D:K": 16, "L:M": 60, "N:O": 16}),
        }
        for sheet_name, (df_sheet, widths) in sheet_specs.items():
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, max(len(df_sheet), 1), max(len(df_sheet.columns) - 1, 0))
            for col_num, value in enumerate(df_sheet.columns):
                worksheet.write(0, col_num, value, header_fmt)
            for col_range, width in widths.items():
                worksheet.set_column(col_range, width, wrap_fmt)
            if "Estabilidad (%)" in df_sheet.columns:
                col_idx = df_sheet.columns.get_loc("Estabilidad (%)")
                number_fmt = workbook.add_format({"num_format": "0.0", "valign": "top"})
                worksheet.set_column(col_idx, col_idx, 16, number_fmt)
    return buf.getvalue()


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact deterministic summary for GPT responses."""
    status_counts = Counter(r.get("Status", "Unknown") for r in results)
    score_counts = Counter(str(r.get("Score", "")) for r in results)

    by_rubric: dict[str, dict[str, Any]] = {}
    for result in results:
        name = str(result.get("Rúbrica", "No especificada"))
        bucket = by_rubric.setdefault(name, {"count": 0, "_scores": []})
        bucket["count"] += 1
        if str(result.get("Status")) == "Success":
            try:
                bucket["_scores"].append(int(result.get("Score")))
            except (TypeError, ValueError):
                pass
    for bucket in by_rubric.values():
        scores = bucket.pop("_scores")
        bucket["average_score"] = round(sum(scores) / len(scores), 2) if scores else None

    stability_values = []
    unstable = []
    repeat_error_count = 0
    for result in results:
        try:
            stab = float(result.get("Estabilidad (%)"))
        except (TypeError, ValueError):
            continue
        stability_values.append(stab)
        if stab < STABILITY_THRESHOLD_PCT:
            unstable.append(result)
        try:
            repeat_error_count += int(result.get("Corridas con error", 0) or 0)
        except (TypeError, ValueError):
            pass

    return {
        "total_criteria": len(results),
        "statuses": dict(status_counts),
        "score_counts": dict(score_counts),
        "by_rubric": by_rubric,
        "stability_repeats": STABILITY_REPEATS,
        "stable_threshold_pct": STABILITY_THRESHOLD_PCT,
        "average_stability_pct": (
            round(sum(stability_values) / len(stability_values), 1) if stability_values else None
        ),
        "stable_count": len(results) - len(unstable),
        "unstable_count": len(unstable),
        "unstable": [
            {
                "rubrica": r.get("Rúbrica"),
                "criterio": r.get("Criterio"),
                "drift": r.get("Deriva principal (si inestable)", ""),
            }
            for r in unstable
        ],
        "repeat_error_count": repeat_error_count,
    }
