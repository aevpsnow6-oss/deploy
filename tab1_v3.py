"""Tab 1 v3 (experimental) — evaluación con rúbrica mecanizada.

Módulo autocontenido y aislado del Tab 1 actual.
No modifica oli_v6_deploy.py más allá de ~5 líneas:
  1. `import tab1_v3` arriba
  2. Añadir `tab7` al tuple de `st.tabs([...])`
  3. Bloque `with tab7: tab1_v3.render(client)` al final

Para retirar:
  rm tab1_v3.py
  revertir las ~5 líneas en oli_v6_deploy.py

Sin estado compartido con Tab 1 actual (claves de session_state con prefijo "v3_").
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
import streamlit as st
from docx2python import docx2python

# ---------- Configuration ----------

RUBRICA_V3_PATH = "./Rubrica_Tab1_Detallada_Full_v3.xlsx"
SHEET_NAME = "Rúbrica Tab 1"
HEADER_ROW_INDEX = 1  # row index 0 = instructions; index 1 = column headers
MODEL = "gpt-5-mini"
MAX_WORKERS = 48
MAX_COMPLETION_TOKENS = 4000

# ---------- Prompt v3 (mecanizado, TESTS + DECISIÓN) ----------

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

REGLAS CRÍTICAS:
- NO inventes elementos. Si la rúbrica lista T1/T2/T3, evalúa exactamente esos — no añadas T4 propio.
- NO apliques el filtro DEDICADO vs MARCO globalmente. Aplícalo SOLO cuando la rúbrica del criterio lo invoque explícitamente (texto «filtro DEDICADO vs MARCO» en la rúbrica).
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


# ---------- Loaders & helpers ----------

@st.cache_data
def load_rubrica_v3() -> pd.DataFrame:
    """Load the v3 rubric xlsx and return one row per criterion."""
    df = pd.read_excel(RUBRICA_V3_PATH, sheet_name=SHEET_NAME, header=HEADER_ROW_INDEX)
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


def extract_docx_text(uploaded_file) -> str:
    """Extract full text from an uploaded .docx file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        result = docx2python(tmp_path)
        return result.text or ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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
    # Higher reasoning effort for high-subjectivity criteria where mechanical rules
    # still leave interpretive room; minimal effort elsewhere keeps cost in check.
    effort = "medium" if subj == "Alta" else "minimal"

    base = {
        "ID": crit_id,
        "Subsección": _extract_subsection(crit_id),
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

    user_prompt = build_user_prompt(row, document_text)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": V3_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
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
    except Exception as e:  # noqa: BLE001 — we want all failures contained per criterion
        return {
            **base,
            "Respuesta": "Error",
            "Razonamiento": f"Error en evaluación v3: {e}",
            "Evidencia": "",
            "Status": "Error",
        }


def _truncate_text_for_display(text: str, words: int = 60) -> str:
    parts = text.split()
    if len(parts) <= words:
        return text
    return " ".join(parts[:words]) + " […]"


# ---------- Streamlit UI ----------

def render(client: Any) -> None:
    """Render the v3 experimental tab. Self-contained; uses 'v3_*' session keys."""

    st.header("🧪 Valoración Preliminar de Calidad — Versión v3 (Experimental)")

    st.warning(
        "**Tab experimental.** Usa una rúbrica mecanizada con TESTS booleanos y "
        "reglas de decisión explícitas. Los resultados pueden diferir del Tab 1 actual. "
        "No usar para evaluaciones finales sin contrastar."
    )

    with st.expander("¿Qué cambia en v3 vs Tab 1 actual?"):
        st.markdown(
            """
- **Rúbrica por criterio**: cada uno de los 76 criterios tiene TESTS atómicos (T1, T2, T3...) con
  una regla de decisión booleana explícita.
- **Decisión mecanizada**: el modelo aplica la regla, no interpreta prosa.
- **Anclas verificables**: patrones de texto concretos a buscar (códigos CPO, convenios, regex de
  indicadores ODS, etc.).
- **Filtro DEDICADO vs MARCO selectivo**: se aplica solo donde la rúbrica del criterio lo invoca,
  no globalmente a los 76.
- **Razonamiento auditable**: el modelo enumera el resultado de cada TEST antes del veredicto.
- **Subjetividad declarada**: 12 criterios marcados como Alta (esperable: más variabilidad).
            """
        )

    # ---- Load rubric ----
    try:
        df_rub = load_rubrica_v3()
    except FileNotFoundError:
        st.error(f"No se encontró {RUBRICA_V3_PATH}. Verifica que el archivo esté en el directorio.")
        return
    except Exception as e:
        st.error(f"Error cargando la rúbrica v3: {e}")
        return

    st.success(f"Rúbrica v3 cargada: {len(df_rub)} criterios.")

    # ---- Upload PRODOC ----
    uploaded = st.file_uploader(
        "Sube el PRODOC (.docx)", type=["docx"], key="v3_upload"
    )
    if uploaded is None:
        st.info("Sube un PRODOC en formato .docx para evaluar.")
        return

    # Cache text extraction per file
    if (
        "v3_doc_text" not in st.session_state
        or st.session_state.get("v3_doc_name") != uploaded.name
    ):
        with st.spinner("Extrayendo texto del documento..."):
            try:
                text = extract_docx_text(uploaded)
            except Exception as e:
                st.error(f"Error extrayendo texto: {e}")
                return
        st.session_state["v3_doc_text"] = text
        st.session_state["v3_doc_name"] = uploaded.name
        # Clear previous results when document changes
        st.session_state.pop("v3_results", None)

    text = st.session_state["v3_doc_text"]
    word_count = len(text.split())
    st.info(f"Documento cargado: **{uploaded.name}** — {word_count:,} palabras")

    # ---- Filters ----
    df_rub_view = df_rub.copy()
    df_rub_view["_section"] = df_rub_view["ID"].apply(_extract_section)
    df_rub_view["_subsection"] = df_rub_view["ID"].apply(_extract_subsection)

    col_a, col_b = st.columns(2)
    with col_a:
        sections = sorted(df_rub_view["_section"].unique().tolist())
        selected_sections = st.multiselect(
            "Filtrar por sección (vacío = todas)", sections, key="v3_secs"
        )
    with col_b:
        if selected_sections:
            sub_pool = sorted(
                df_rub_view[df_rub_view["_section"].isin(selected_sections)][
                    "_subsection"
                ].unique().tolist()
            )
        else:
            sub_pool = sorted(df_rub_view["_subsection"].unique().tolist())
        selected_subs = st.multiselect(
            "Filtrar por subsección (vacío = todas)", sub_pool, key="v3_subs"
        )

    df_filtered = df_rub_view
    if selected_sections:
        df_filtered = df_filtered[df_filtered["_section"].isin(selected_sections)]
    if selected_subs:
        df_filtered = df_filtered[df_filtered["_subsection"].isin(selected_subs)]

    st.info(f"📌 {len(df_filtered)} criterios seleccionados para evaluación.")

    # ---- Run ----
    if st.button("▶️ Evaluar con v3", key="v3_run", type="primary"):
        if df_filtered.empty:
            st.warning("No hay criterios para evaluar con los filtros actuales.")
            return

        progress = st.progress(0.0)
        status = st.empty()
        results: list[dict[str, Any]] = []
        total = len(df_filtered)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(evaluate_criterion, client, row, text): row["ID"]
                for _, row in df_filtered.iterrows()
            }
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                progress.progress(done / total)
                status.text(f"Evaluado {done}/{total} criterios")

        results.sort(key=lambda r: _id_sort_key(r["ID"]))
        st.session_state["v3_results"] = results
        progress.empty()
        status.empty()
        st.success(f"✅ Evaluación v3 completa: {len(results)} criterios.")

    # ---- Results ----
    if "v3_results" not in st.session_state:
        return

    results = st.session_state["v3_results"]
    df_res = pd.DataFrame(results)

    # Summary metrics
    st.markdown("### Resumen")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", len(df_res))
    c2.metric("Yes", int((df_res["Respuesta"] == "Yes").sum()))
    c3.metric("Partial", int((df_res["Respuesta"] == "Partial").sum()))
    c4.metric("No", int((df_res["Respuesta"] == "No").sum()))
    c5.metric(
        "Not Found / N/A / Error",
        int(df_res["Respuesta"].isin(["Not Found", "N/A", "Error"]).sum()),
    )

    # Distribution by subjectivity (helpful to see where v3 may be unstable)
    if "Subjetividad" in df_res.columns:
        st.markdown("**Distribución por subjetividad residual:**")
        st.dataframe(
            df_res.groupby("Subjetividad")["Respuesta"].value_counts().unstack(fill_value=0),
            use_container_width=True,
        )

    # Interpretation guide — placed right before the detailed table so it's
    # accessible exactly where the user needs it.
    with st.expander("📖 Cómo leer e interpretar los resultados", expanded=False):
        st.markdown(
            """
### 1. Veredictos (columna **Respuesta**)

| Veredicto | Significado |
|---|---|
| **Yes** | El criterio se cumple. La DECISIÓN de la rúbrica Sí se satisfizo (todos los TESTS obligatorios = verdadero). |
| **Partial** | Cumplimiento parcial. La DECISIÓN de la rúbrica Sí falló pero la de Parcial se satisfizo. |
| **No** | El criterio no se cumple. La DECISIÓN de No se satisfizo. |
| **Not Found** | No se encontró información en el documento para evaluar. Distinto de **No** — aquí el modelo no halló sección/contenido relacionado. |
| **N/A** | El criterio tiene aplicabilidad condicional y la condición no se cumple (p.ej. 4.4.3 N/A cuando presupuesto ≤ 5M USD). |
| **Error** | Falla técnica en la evaluación. Revisar columna Razonamiento. |

### 2. Razonamiento — cómo está estructurado

El modelo enumera el resultado de cada TEST antes del veredicto. Formato esperado:

```
T1: verdadero — Cita explícita de P&B en sección 1.2
T2: falso — No se identifica DWCP del país
T3: verdadero — CPO ABC-101 mencionado en anexo
T4: verdadero — Verbo «contribuye a» presente
DECISIÓN: T1 ∧ ¬T2 ∧ T3 ∧ T4 = falso (regla Sí requiere los 4) → Parcial
VEREDICTO: Partial
```

**Si el razonamiento NO sigue este formato**, es señal de alerta: el modelo no aplicó la rúbrica mecánicamente. Leer con cuidado y verificar manualmente.

### 3. Evidencia

Citas textuales del PRODOC que respaldan el veredicto. Si dice *«No se encontró sección X»* o *«Documento no contiene…»*, es **ausencia documentada** — evidencia legítima para No / Not Found.

### 4. Subjetividad residual (columna **Subjetividad**)

Indica cuánto juicio cualitativo irreducible queda después de mecanizar la rúbrica:

| Nivel | # criterios | Cómo tratar el veredicto |
|---|---|---|
| 🟢 **Baja** | 27 | Reproducible. Confiar; muestrear ~10% para auditoría. |
| 🟡 **Media** | 37 | Pequeña variabilidad esperable. Confiar pero verificar bordes (Partial / Not Found). |
| 🟠 **Alta** | 12 | Tratar como hipótesis. Revisar razonamiento y evidencia manualmente antes de aceptar. |

Los 12 de Subjetividad Alta son los candidatos a calibrar con ejemplos reales si más adelante se hace el bootstrap.

### 5. Aspectos transversales — filtro DEDICADO vs MARCO

Cuando un criterio tiene aspectos transversales (Género, Discapacidad, NIT, EAS, Pueblos indígenas, etc.) se aplica el filtro:

- ✅ **DEDICADO**: el documento tiene **un sub-objetivo, indicador, actividad, partida presupuestaria o meta específica** para el sujeto. Cuenta.
- ❌ **MARCO**: el documento solo lista al sujeto entre otros grupos (*«mujeres, indígenas, jóvenes, entre otros»*) o usa lenguaje inclusivo genérico. **NO cuenta** para cumplimiento.

Una propuesta puede mencionar género 50 veces y aun así sacar **No** si todas las menciones son MARCO.

#### Diferencias respecto a Tab 1 actual

El filtro existe en ambos tabs, pero **se expone de forma diferente**:

| Dimensión | Tab 1 (actual) | Tab 7 (v3) |
|---|---|---|
| **Alcance de aplicación** | Global: el system prompt lo impone a TODOS los criterios | Selectivo: se aplica solo donde la rúbrica del criterio lo invoca explícitamente |
| **Elementos contables** | Fijos: A/B/C/D/E (sub-objetivo, indicador, actividad, presupuesto, meta) — todos peso igual | Variables por criterio; algunos elementos marcados como OPCIONALES según naturaleza del proyecto (resultado de la revisión cliente) |
| **Regla de decisión** | Escalonada: 0→No/NF, 1–2→Partial, 3–4→Partial-o-Yes, 5→Yes | Per-criterio: regla booleana explícita (p.ej. 1.5.1 = ≥2 de 3 obligatorios; 3.3.1 = condicional según si la inclusión es explícita o implícita) |
| **Trazabilidad** | El modelo etiqueta cada cita como [DEDICATED]/[FRAMING] dentro de un razonamiento en prosa | El modelo enumera T1/T2/T3 con verdadero/falso y luego aplica DECISIÓN — más auditable |
| **Criterios no transversales** | El filtro igual se inyecta como sistema (puede sesgar veredictos de criterios donde no aplica) | El filtro NO se aplica en criterios no transversales (no hay sesgo cruzado) |

**Consecuencia práctica**: el mismo PRODOC puede recibir un veredicto distinto entre Tab 1 y v3 en criterios transversales. La razón más común es que v3 admite Sí con menos elementos DEDICADOS si la rúbrica así lo define para ese criterio.

### 6. Cómo actuar sobre los resultados

| Combinación | Acción recomendada |
|---|---|
| Yes + Subjetividad Baja/Media | Aceptar; muestreo aleatorio para auditoría |
| Yes + Subjetividad Alta | Revisar razonamiento y evidencia antes de aceptar |
| Partial | Leer TESTS para identificar qué elementos faltan — guía directa para feedback al equipo de proyecto |
| No | Verificar que la evidencia documenta ausencia, no que el modelo no encontró |
| Not Found | Confirmar manualmente; posiblemente el contenido está en un anexo no cargado |
| N/A | Verificar la condición de aplicabilidad — si la condición sí aplica al proyecto, hay error de identificación |
| Error | Reintentar; si persiste, reportar |

### 7. Señales de alerta — cuándo desconfiar del veredicto

- ⚠️ Razonamiento **sin enumerar TESTS** → el modelo no aplicó la rúbrica mecánicamente.
- ⚠️ Subjetividad **Alta** + evidencia escueta o genérica → caso candidato a revisión humana.
- ⚠️ Veredictos contradictorios entre criterios de la misma subsección → indagar inconsistencia.
- ⚠️ **N/A** en criterios marcados como *Aplicabilidad: Siempre* → bug, reportar.
- ⚠️ Evidencia que cita texto que **no aparece literalmente** en el PRODOC → posible alucinación.

### 8. Orden sugerido para revisar resultados

1. Filtrar por **Status = Error** → resolver primero (suelen ser pocos).
2. Filtrar por **Subjetividad = Alta** → revisar todos manualmente (12 criterios).
3. Filtrar por **Respuesta = Not Found** → verificar si hay contenido en anexos no cargados.
4. Filtrar por **Respuesta = Partial** → leer TESTS para identificar qué elementos faltan; es la columna más informativa para retroalimentar al equipo de proyecto.
5. Muestrear **Respuesta = Yes** (~10%) → auditoría de calidad del modelo.
            """
        )

    # Detailed view
    st.markdown("### Resultados detallados")
    show_cols = [
        "ID",
        "Subsección",
        "Criterio",
        "Respuesta",
        "Razonamiento",
        "Evidencia",
        "Tipo",
        "Subjetividad",
        "Transversales",
        "Status",
    ]
    available_cols = [c for c in show_cols if c in df_res.columns]
    st.dataframe(df_res[available_cols], use_container_width=True, height=500)

    # Download xlsx
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_res[available_cols].to_excel(writer, index=False, sheet_name="Resultados v3")
    st.download_button(
        "📥 Descargar resultados v3 (.xlsx)",
        buf.getvalue(),
        file_name=f"valoracion_v3_{os.path.splitext(st.session_state.get('v3_doc_name','sin_nombre'))[0]}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
