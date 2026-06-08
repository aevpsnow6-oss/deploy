"""Core logic for Tab 1 v3 project-quality appraisal.

This module is intentionally Streamlit-free so the same v3 rubric logic can be
used by the Streamlit experimental tab and by the Custom GPT action backend.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd

RUBRICA_V3_PATH = "./Rubrica_Tab1_Detallada_Full_v3.xlsx"
SHEET_NAME = "Rúbrica Tab 1"
HEADER_ROW_INDEX = 1
MODEL = "gpt-5-mini"
MAX_WORKERS = 48
MAX_COMPLETION_TOKENS = 4000

RESULT_COLUMNS = [
    "ID",
    "Subsección",
    "Pregunta orientadora (no evaluada)",
    "Criterio",
    "Tipo",
    "Subjetividad",
    "Transversales",
    "Respuesta",
    "Razonamiento",
    "Evidencia",
    "Status",
]

V3_SYSTEM_PROMPT = """Eres un analista experto en evaluación de documentos de proyecto (PRODOC) de la OIT.

Tu tarea: evaluar UN criterio específico contra el documento provisto, aplicando una rúbrica mecanizada.

ESTRUCTURA DE LA RÚBRICA QUE RECIBES:
- CRITERIO: el enunciado a evaluar.
- PREGUNTA ORIENTADORA: contexto general (NO se evalúa, solo enmarca).
- TIPO: forma del criterio (binario, lista, transversal, etc.).
- APLICABILIDAD: si el criterio aplica siempre o bajo condición.
- ASPECTOS TRANSVERSALES: indica si aplica filtro DEDICADO vs MARCO.
- ELEMENTOS A VERIFICAR: componentes atómicos del criterio.
- RÚBRICA Sí/Parcial/No/No aplica: TESTS atómicos (T1, T2, ...) y DECISIÓN booleana.
- ANCLAS VERIFICABLES: patrones de texto, códigos, nombres concretos a buscar.

PROCESO OBLIGATORIO (en este orden):
1. Si APLICABILIDAD es condicional: primero determina si la condición se cumple. Si no, veredicto = "N/A".
2. Localiza en el DOCUMENTO la(s) sección(es) que tratan del criterio. Usa las ANCLAS como guía de búsqueda.
3. Para cada TEST listado en la rúbrica Sí (T1, T2, T3, ...): evalúa explícitamente si se cumple (verdadero/falso) con base en el documento.
4. Aplica la DECISIÓN de Sí primero. Si no se cumple, aplica la DECISIÓN de Parcial. Si tampoco, aplica No.
5. Veredicto final: Yes / Partial / No / Not Found / N/A.

DEFINICIÓN CANÓNICA DE «DEDICADO vs MARCO» (idéntica a la usada en Tab 1):

Una mención del sujeto (género, discapacidad, pueblos indígenas, etc.) cuenta como
MARCO (NO cuenta para cumplimiento) cuando es cualquiera de:
  - Mención en el objetivo general / declaración de impacto que enumera varios grupos
  - Listas de partes interesadas / consulta / participantes de investigación
  - Enumeraciones de alcance de seguimiento («…entre otros», «…incluyendo X, Y, Z»)
  - Lenguaje boilerplate de inclusión
  - Cualquier pasaje donde el sujeto aparece en una lista de ≥3 grupos sin seguimiento dedicado

Una mención cuenta como DEDICADO si es cualquiera de:
  A. Sub-objetivo, resultado o producto cuyo título/propósito nombra al sujeto
  B. Indicador desagregado por el sujeto o que lo mide específicamente
  C. Actividad cuyo propósito principal aborda al sujeto
  D. Partida presupuestaria o asignación de recursos para el sujeto
  E. Meta cuantificable relativa al sujeto

Si toda la evidencia citable es MARCO, el veredicto DEBE ser "No" o "Not Found",
sin importar cuántas veces se nombre al sujeto.

REGLAS CRÍTICAS:
- NO inventes elementos. Si la rúbrica lista T1/T2/T3, evalúa exactamente esos — no añadas T4 propio.
- El filtro DEDICADO vs MARCO definido arriba aplica SOLO cuando la rúbrica del criterio lo invoque explícitamente (criterios marcados con ASPECTOS TRANSVERSALES ≠ «Ninguno»). Para criterios sin transversales, ignóralo.
- Cita evidencia textual entre comillas. Si la evidencia es ausencia, dilo: «No se encontró sección X».
- Si el documento carece de información para evaluar el criterio, veredicto = "Not Found".
- "N/A" solo cuando la APLICABILIDAD condicional no se satisface.
- El Razonamiento DEBE enumerar el resultado de cada TEST seguido de la DECISIÓN aplicada.

FORMATO DEL Razonamiento (estricto):
  T1: <verdadero/falso> — <una línea de justificación>
  T2: <verdadero/falso> — <una línea de justificación>
  ...
  DECISIÓN: <regla booleana evaluada con los resultados, p.ej. T1 ∧ T2 ∧ ¬T3 = falso → Parcial>
  VEREDICTO: <Yes/No/Partial/Not Found/N/A>

Devuelve SIEMPRE JSON con: {"Respuesta", "Razonamiento", "Evidencia"}. Idioma: español."""


V3_USER_PROMPT_TEMPLATE = """═════ CRITERIO A EVALUAR ═════
ID del criterio: {id}
CRITERIO: {criterio}

PREGUNTA ORIENTADORA (solo contexto, NO evaluar): {head}

═════ METADATA ═════
TIPO: {tipo}
APLICABILIDAD: {aplicabilidad}
ASPECTOS TRANSVERSALES: {transv}

═════ ELEMENTOS A VERIFICAR ═════
{elementos}

═════ RÚBRICA — Sí ═════
{si}

═════ RÚBRICA — Parcial ═════
{par}

═════ RÚBRICA — No ═════
{no}

═════ RÚBRICA — No aplica ═════
{na}

═════ ANCLAS VERIFICABLES ═════
{anclas}

═════ DOCUMENTO (PRODOC) ═════
{document_text}

═════ TU TAREA ═════
1. Verifica APLICABILIDAD. Si no aplica → Respuesta = "N/A".
2. Evalúa cada TEST de la rúbrica de Sí con base en el DOCUMENTO.
3. Aplica DECISIÓN Sí → Parcial → No en orden.
4. Devuelve JSON con Respuesta + Razonamiento (con todos los TESTS enumerados + DECISIÓN + VEREDICTO) + Evidencia (citas textuales)."""


V3_RESPONSE_SCHEMA = {
    "name": "rubric_eval_v3",
    "schema": {
        "type": "object",
        "properties": {
            "Respuesta": {
                "type": "string",
                "enum": ["Yes", "No", "Partial", "Not Found", "N/A"],
            },
            "Razonamiento": {"type": "string"},
            "Evidencia": {"type": "string"},
        },
        "required": ["Respuesta", "Razonamiento", "Evidencia"],
        "additionalProperties": False,
    },
    "strict": True,
}


def load_rubrica_v3(rubric_path: str = RUBRICA_V3_PATH) -> pd.DataFrame:
    """Load the v3 rubric xlsx and return one row per criterion."""
    df = pd.read_excel(rubric_path, sheet_name=SHEET_NAME, header=HEADER_ROW_INDEX)
    df = df.dropna(subset=["ID"]).reset_index(drop=True)
    df["ID"] = df["ID"].astype(str)
    return df


def _id_sort_key(id_str: str) -> tuple[int, ...]:
    return tuple(int(p) for p in str(id_str).split(".") if p.isdigit())


def _extract_section(id_str: str) -> int:
    m = re.match(r"(\d+)", str(id_str))
    return int(m.group(1)) if m else 0


def _extract_subsection(id_str: str) -> str:
    m = re.match(r"(\d+\.\d+)", str(id_str))
    return m.group(1) if m else ""


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


def filter_rubric(
    df_rubric: pd.DataFrame,
    sections: list[int] | None = None,
    subsections: list[str] | None = None,
) -> pd.DataFrame:
    """Filter rubric rows by section and/or subsection."""
    df = df_rubric.copy()
    df["_section"] = df["ID"].apply(_extract_section)
    df["_subsection"] = df["ID"].apply(_extract_subsection)
    if sections:
        df = df[df["_section"].isin([int(s) for s in sections])]
    if subsections:
        df = df[df["_subsection"].isin([str(s) for s in subsections])]
    return df.reset_index(drop=True)


def build_user_prompt(row: pd.Series, document_text: str) -> str:
    """Inject one criterion's rubric into the user prompt template."""

    def get(col: str, default: str = "") -> str:
        v = row.get(col, default)
        if pd.isna(v) or v is None:
            return default
        s = str(v).strip()
        return s if s else default

    na_text = get("Rúbrica — No aplica", "")
    if not na_text:
        na_text = "(esta rúbrica no contempla la categoría «No aplica»)"

    return V3_USER_PROMPT_TEMPLATE.format(
        id=get("ID"),
        criterio=get("Criterio a evaluar"),
        head=get("Pregunta orientadora (CONTEXTO — no evaluar)"),
        tipo=get("Tipo de criterio"),
        aplicabilidad=get("Aplicabilidad"),
        transv=get("Aspectos transversales", "Ninguno"),
        elementos=get("Elementos a verificar"),
        si=get("Rúbrica — Sí"),
        par=get("Rúbrica — Parcial"),
        no=get("Rúbrica — No"),
        na=na_text,
        anclas=get("Anclas verificables (v3)"),
        document_text=document_text,
    )


def evaluate_criterion(client: Any, row: pd.Series, document_text: str) -> dict[str, Any]:
    """Run a single criterion through the v3 prompt and return a result dict."""
    crit_id = str(row.get("ID", ""))
    subj = str(row.get("Subjetividad residual (v3)", "Media")).strip()
    effort = "medium" if subj == "Alta" else "minimal"

    base = {
        "ID": crit_id,
        "Subsección": _extract_subsection(crit_id),
        "Pregunta orientadora (no evaluada)": str(
            row.get("Pregunta orientadora (CONTEXTO — no evaluar)", "")
        ),
        "Criterio": str(row.get("Criterio a evaluar", "")),
        "Tipo": str(row.get("Tipo de criterio", "")),
        "Subjetividad": subj,
        "Transversales": str(row.get("Aspectos transversales", "Ninguno")),
    }

    if not document_text or not document_text.strip():
        return {
            **base,
            "Respuesta": "Not Found",
            "Razonamiento": "Documento vacío.",
            "Evidencia": "",
            "Status": "Success",
        }

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": V3_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row, document_text)},
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            reasoning_effort=effort,
            response_format={"type": "json_schema", "json_schema": V3_RESPONSE_SCHEMA},
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return {
                **base,
                "Respuesta": "Error",
                "Razonamiento": "Respuesta vacía del modelo.",
                "Evidencia": "",
                "Status": "Error",
            }
        result = json.loads(content)
        return {
            **base,
            "Respuesta": result.get("Respuesta", "Not Found"),
            "Razonamiento": result.get("Razonamiento", ""),
            "Evidencia": result.get("Evidencia", ""),
            "Status": "Success",
        }
    except Exception as e:  # noqa: BLE001 - contain failures per criterion
        return {
            **base,
            "Respuesta": "Error",
            "Razonamiento": f"Error en evaluación v3: {e}",
            "Evidencia": "",
            "Status": "Error",
        }


def evaluate_criteria(
    client: Any,
    df_criteria: pd.DataFrame,
    document_text: str,
    max_workers: int = MAX_WORKERS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all rows in a rubric dataframe."""
    total = len(df_criteria)
    results: list[dict[str, Any]] = []
    if total == 0:
        return results

    workers = max(1, min(int(max_workers), total))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(evaluate_criterion, client, row, document_text): row["ID"]
            for _, row in df_criteria.iterrows()
        }
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if progress_callback:
                progress_callback(done, total)

    results.sort(key=lambda r: _id_sort_key(r["ID"]))
    return results


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a sorted dataframe with the public result columns."""
    if not results:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.DataFrame(results)
    available = [c for c in RESULT_COLUMNS if c in df.columns]
    df_sorted = (
        df.assign(_sk=df["ID"].map(_id_sort_key))
        .sort_values("_sk")
        .drop(columns="_sk")
        .reset_index(drop=True)
    )
    return df_sorted[available]


def results_to_xlsx_bytes(results: list[dict[str, Any]]) -> bytes:
    """Serialize v3 results to an XLSX workbook."""
    df = results_to_dataframe(results)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados v3")
    return buf.getvalue()


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a compact deterministic summary for GPT responses."""
    answer_counts = Counter(r.get("Respuesta", "Unknown") for r in results)
    status_counts = Counter(r.get("Status", "Unknown") for r in results)
    high_subjectivity = [
        r for r in results if str(r.get("Subjetividad", "")).strip() == "Alta"
    ]
    by_subjectivity: dict[str, dict[str, int]] = {}
    for result in results:
        subj = str(result.get("Subjetividad", "Unknown"))
        ans = str(result.get("Respuesta", "Unknown"))
        by_subjectivity.setdefault(subj, {})
        by_subjectivity[subj][ans] = by_subjectivity[subj].get(ans, 0) + 1

    return {
        "total_criteria": len(results),
        "answers": dict(answer_counts),
        "statuses": dict(status_counts),
        "by_subjectivity": by_subjectivity,
        "high_subjectivity_count": len(high_subjectivity),
        "high_subjectivity_ids": [r.get("ID") for r in high_subjectivity],
    }
