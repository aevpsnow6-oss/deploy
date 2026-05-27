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

        # --- En una frase + analogía ---
        st.markdown(
            """
### En una frase

La herramienta lee tu PRODOC y, para cada uno de los **76 criterios de calidad** de la
OIT, decide si **se cumple, se cumple parcialmente, no se cumple, o no se encontró
información** — explicándote por qué.

### Cómo funciona, en términos sencillos

Imagina a un revisor experto leyendo tu PRODOC con una lista de verificación en mano.
Para cada criterio:

1. Tiene una **lista clara** de lo que debe encontrar en el documento.
2. Va al PRODOC y **verifica cada punto** uno por uno.
3. Aplica una **regla simple** para decidir el veredicto.
4. Te entrega el veredicto **junto con el resultado de cada chequeo**, para que tú puedas
   verificarlo.

Eso es exactamente lo que hace esta herramienta. La diferencia con la pestaña principal
(Tab 1) es que aquí las reglas son **explícitas y mecánicas** — no dependen de la
interpretación subjetiva del modelo.

---
            """
        )

        # --- 1. Veredictos ---
        st.markdown(
            """
### 1. ¿Qué significa cada veredicto?

La columna **Respuesta** te dice el resultado de la evaluación. Hay 6 valores posibles:

| Veredicto | Significado | Ejemplo concreto |
|---|---|---|
| 🟢 **Yes** | El criterio se cumple completamente. | El PRODOC cita el Convenio núm. 190 por número y lo integra a la estrategia y los indicadores. |
| 🟡 **Partial** | Se cumple en parte. Faltan elementos específicos. | Cita el Convenio núm. 190 pero solo en antecedentes, no lo integra a la estrategia. |
| 🔴 **No** | El criterio no se cumple. | No menciona ningún convenio por número, solo «las normas de la OIT» genéricamente. |
| ⚫ **Not Found** | No se encontró información para evaluar (distinto de **No**: aquí ni siquiera había contenido relacionado). | La sección sobre normas internacionales del trabajo no existe en el documento. |
| ⚪ **N/A** | El criterio no aplica al proyecto. | Criterio 4.4.3 (plantilla DCOMM para proyectos > 5 millones USD): tu proyecto es de $800,000 → no aplica. |
| ⚠️ **Error** | Falla técnica en la evaluación. | Revisar columna Razonamiento. Reintentar. |

💡 **Distinción importante**: **No** ≠ **Not Found**. *No* significa que el documento sí
trata el tema pero falla el criterio; *Not Found* significa que el documento ni siquiera
toca el tema.

---
            """
        )

        # --- 2. Razonamiento ---
        st.markdown(
            """
### 2. ¿Cómo está estructurado el "Razonamiento"?

El razonamiento te muestra **exactamente por qué el modelo llegó a ese veredicto**.
Sigue siempre el mismo formato:
            """
        )
        st.code(
            "T1: verdadero — Cita explícita de P&B en sección 1.2\n"
            "T2: falso — No se identifica DWCP del país\n"
            "T3: verdadero — CPO ABC-101 mencionado en anexo\n"
            "T4: verdadero — Aparece el verbo «contribuye a»\n"
            "DECISIÓN: la regla pide T1, T2, T3 y T4 los cuatro verdaderos.\n"
            "          T2 falló → no se cumple la regla de Sí.\n"
            "          Sí se cumple la regla de Parcial.\n"
            "VEREDICTO: Partial",
            language="text",
        )
        st.markdown(
            """
**Cómo leerlo:**
- Cada **T** es un chequeo independiente. Resultado: *verdadero* (se cumple en el
  documento) o *falso* (no se cumple).
- La línea **DECISIÓN** explica qué regla se aplicó y por qué.
- El **VEREDICTO** final es la conclusión.

💡 **Por qué importa**: este desglose te permite ver *exactamente qué chequeo falló*. Si
el veredicto es Parcial y T2 falló, ya sabes qué hay que arreglar en el proyecto (en este
caso, añadir referencia explícita al DWCP del país). Es información accionable, no solo
un veredicto.
            """
        )
        st.warning(
            "**Si el razonamiento NO sigue este formato** (no enumera T1, T2…), es señal de "
            "alerta: el modelo no aplicó la rúbrica mecánicamente. Verifica manualmente antes "
            "de confiar en el veredicto."
        )
        st.markdown("---")

        # --- 3. Evidencia ---
        st.markdown(
            """
### 3. ¿Cómo está estructurada la "Evidencia"?

Son citas textuales del PRODOC que respaldan el veredicto. Hay dos tipos válidos:
            """
        )
        st.success(
            "✅ **Cita afirmativa** — el modelo copia un pasaje del PRODOC que demuestra que el "
            "criterio se cumple. Ejemplo:\n\n"
            "> *«El proyecto se alinea con el resultado P&B 2.3 del bienio 2024-25 y contribuye "
            "al CPO ABC-101 del DWCP de Honduras.»*"
        )
        st.info(
            "📝 **Ausencia documentada** — el modelo declara que cierto contenido NO aparece en "
            "el documento. Es evidencia legítima para sustentar *No* o *Not Found*. Ejemplo:\n\n"
            "> *«No se encontró una sección de análisis de riesgos en el documento.»*"
        )
        st.error(
            "⚠️ **Alerta de alucinación**: si la evidencia cita texto que tú **no encuentras** "
            "en tu PRODOC al buscarlo literalmente, puede ser una alucinación del modelo. "
            "Reportar."
        )
        st.markdown("---")

        # --- 4. Subjetividad ---
        st.markdown(
            """
### 4. ¿Qué es la columna "Subjetividad"?

Algunos criterios son **fáciles de verificar mecánicamente** (¿aparece un código ODS
como 8.5.2? Sí o no). Otros requieren **juicio cualitativo** (¿la teoría del cambio es
"convincente"?). La columna **Subjetividad** te dice cuánto juicio cualitativo queda
después de aplicar la rúbrica:

| Nivel | # criterios | Qué tan reproducible es | Qué hacer |
|---|---|---|---|
| 🟢 **Baja** | 27 | Muy reproducible — dos corridas del mismo PRODOC darían el mismo resultado. | Confiar. Auditar ~10% al azar. |
| 🟡 **Media** | 37 | Reproducible en general; algunos casos en el borde pueden variar. | Confiar; verificar manualmente los *Partial* y *Not Found*. |
| 🟠 **Alta** | 12 | Juicio cualitativo irreducible — puede variar entre corridas. | Tratar como hipótesis. Leer razonamiento y evidencia manualmente. |

💡 **En la práctica**: si tienes poco tiempo, enfoca tu revisión humana en los **12
criterios de Subjetividad Alta**. Para esos, el modelo es tu asistente, no tu juez final.

---
            """
        )

        # --- 5. DEDICADO vs MARCO ---
        st.markdown(
            """
### 5. El filtro DEDICADO vs MARCO — el concepto más importante

Este filtro aplica a los criterios que evalúan **temas transversales**: género,
discapacidad, pueblos indígenas, normas laborales, medio ambiente, explotación y abuso
sexuales (EAS), etc.

**El problema que resuelve**: muchas propuestas mencionan estos temas, pero solo de
manera decorativa, sin asignarles recursos o acciones específicas. Una propuesta puede
decir *"beneficiará a mujeres, jóvenes, personas con discapacidad y comunidades indígenas"*
y aun así no tener una sola actividad dedicada a alguno de esos grupos.

El filtro distingue dos formas en que aparece un tema en el documento:
            """
        )
        st.error(
            "❌ **MARCO** — solo aparece en listas de grupos o lenguaje inclusivo genérico. "
            "**NO cuenta** para cumplimiento.\n\n"
            "*Ejemplo*: «El proyecto beneficiará a poblaciones vulnerables, incluyendo mujeres, "
            "personas con discapacidad y comunidades indígenas, entre otros grupos.»\n\n"
            "Aquí «personas con discapacidad» aparece, pero no hay nada dedicado a ese grupo: "
            "ni sub-objetivo, ni indicador, ni actividad, ni partida, ni meta."
        )
        st.success(
            "✅ **DEDICADO** — hay algo específicamente para el sujeto. **Sí cuenta**.\n\n"
            "*Ejemplo*: «Indicador 3.2: número de personas con discapacidad capacitadas en el "
            "oficio (meta: 200). Partida presupuestaria 4.1.5: USD 35,000 para accesibilidad "
            "de materiales y lenguaje de señas.»\n\n"
            "Aquí hay un indicador específico, una meta cuantificable y una partida "
            "presupuestaria — tres elementos DEDICADOS."
        )
        st.info(
            "💡 **Consecuencia práctica**: una propuesta puede mencionar género 50 veces y aún "
            "así sacar **No** si todas las menciones son MARCO. Lo que cuenta es la "
            "dedicación operativa, no la frecuencia del lenguaje."
        )

        st.markdown(
            """
#### ¿Cómo se diferencia este filtro respecto a la pestaña Tab 1 actual?

Existe en ambas pestañas, pero **se aplica diferente**. Resumen breve:

| Aspecto | Tab 1 (actual) | Tab 7 (v3, esta pestaña) |
|---|---|---|
| **¿A qué criterios se aplica?** | A todos | Solo a los criterios cuya rúbrica lo invoca |
| **¿Cuántos elementos son obligatorios?** | Siempre 5: sub-objetivo, indicador, actividad, presupuesto, meta | Varía por criterio. Algunos elementos son opcionales según la naturaleza del proyecto |
| **¿Cómo se traduce a veredicto?** | Escala fija: 0→No, 1-2→Parcial, 3-4→Parcial-o-Sí, 5→Sí | Cada criterio tiene su regla propia (p.ej. ≥2 de 3 obligatorios = Sí) |
| **¿Cómo se ve en el razonamiento?** | Prosa con etiquetas [DEDICATED]/[FRAMING] | Lista de chequeos T1/T2/T3 con verdadero/falso explícito |

**Por qué importa**: el mismo PRODOC puede recibir veredictos diferentes entre Tab 1 y
esta pestaña en criterios transversales. La razón habitual: la rúbrica de v3 (validada
con la revisión de su equipo) admite *Sí* con menos elementos DEDICADOS cuando la
naturaleza del proyecto lo justifica.
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
