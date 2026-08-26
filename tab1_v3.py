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

import os
from typing import Any

import pandas as pd
import streamlit as st
import tab1_v3_core as v3_core
from tab1_v3_core import (
    MAX_WORKERS,
    RUBRICA_V3_PATH,
    STABILITY_REPEATS,
    STABILITY_THRESHOLD_PCT,
    _extract_section,
    _extract_subsection,
)


@st.cache_data
def load_rubrica_v3() -> pd.DataFrame:
    """Load the v3 rubric xlsx and return one row per criterion."""
    return v3_core.load_rubrica_v3()


def extract_docx_text(uploaded_file) -> str:
    """Extract full text from an uploaded .docx file."""
    return v3_core.extract_docx_text_from_bytes(uploaded_file.read())


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
- **Estabilidad estándar**: cada criterio se evalúa 10 veces y se reporta un resultado modal
  con porcentaje de estabilidad.
- **Anclas verificables**: patrones de texto concretos a buscar (códigos CPO, convenios, regex de
  indicadores ODS, etc.).
- **Filtro DEDICADO vs MARCO selectivo**: se aplica solo donde la rúbrica del criterio lo invoca,
  no globalmente a los 76.
- **Razonamiento auditable**: el modelo enumera el resultado de cada TEST antes del resultado final.
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
    if st.button("▶️ Evaluar con v3 (10 corridas por criterio)", key="v3_run", type="primary"):
        if df_filtered.empty:
            st.warning("No hay criterios para evaluar con los filtros actuales.")
            return

        progress = st.progress(0.0)
        status = st.empty()
        total_calls = len(df_filtered) * STABILITY_REPEATS

        def update_progress(done: int, total: int) -> None:
            progress.progress(done / total)
            status.text(
                f"Corridas completadas {done}/{total} "
                f"({STABILITY_REPEATS} por criterio)"
            )

        results = v3_core.evaluate_criteria(
            client,
            df_filtered,
            text,
            max_workers=MAX_WORKERS,
            progress_callback=update_progress,
        )
        st.session_state["v3_results"] = results
        progress.empty()
        status.empty()
        st.success(
            f"✅ Evaluación v3 completa: {len(results)} criterios, "
            f"{total_calls} corridas."
        )

    # ---- Results ----
    if "v3_results" not in st.session_state:
        return

    results = st.session_state["v3_results"]
    df_public = v3_core.results_to_public_dataframe(results)
    df_res = v3_core.results_to_dataframe(results)

    # Summary metrics
    st.markdown("### Resumen de valoración")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", len(df_res))
    c2.metric("Yes", int((df_res["Respuesta"] == "Yes").sum()))
    c3.metric("Partial", int((df_res["Respuesta"] == "Partial").sum()))
    c4.metric("No", int((df_res["Respuesta"] == "No").sum()))
    c5.metric(
        "Not Found / N/A / Error",
        int(df_res["Respuesta"].isin(["Not Found", "N/A", "Error"]).sum()),
    )

    if "Estabilidad (%)" in df_res.columns:
        avg_stability = (
            float(df_res["Estabilidad (%)"].mean()) if not df_res.empty else 0.0
        )
        stable_count = int((df_res["Estabilidad (%)"] >= STABILITY_THRESHOLD_PCT).sum())
        unstable_count = int((df_res["Estabilidad (%)"] < STABILITY_THRESHOLD_PCT).sum())
        s1, s2, s3 = st.columns(3)
        s1.metric("Estabilidad media", f"{avg_stability:.1f}%")
        s2.metric(f"Estables (≥{STABILITY_THRESHOLD_PCT:.0f}%)", stable_count)
        s3.metric(f"Inestables (<{STABILITY_THRESHOLD_PCT:.0f}%)", unstable_count)

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
3. Aplica una **regla simple** para decidir el resultado de valoración.
4. Te entrega el resultado **junto con el resultado de cada chequeo**, para que tú puedas
   verificarlo.

Eso es exactamente lo que hace esta herramienta. La diferencia con la pestaña principal
(Tab 1) es que aquí las reglas son **explícitas y mecánicas** — no dependen de la
interpretación subjetiva del modelo.

---
            """
        )

        # --- 1. Resultados ---
        st.markdown(
            """
### 1. ¿Qué significa cada resultado?

La columna **Resultado de valoración** te dice el resultado de la revisión asistida.
No es una determinación oficial de la OIT. Hay 6 valores posibles:

| Resultado | Significado | Ejemplo concreto |
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

        # --- 2. Estabilidad ---
        st.markdown(
            f"""
### 2. ¿Qué significa la columna "Estabilidad (%)"?

Cada criterio se evalúa **{STABILITY_REPEATS} veces de forma independiente**. La tabla muestra
un solo resultado: el resultado que apareció más veces. La columna **Estabilidad (%)** indica
qué porcentaje de las {STABILITY_REPEATS} corridas coincidió con ese resultado modal.

| Ejemplo | Lectura |
|---|---|
| **Yes**, Estabilidad 100% | Las {STABILITY_REPEATS} corridas dieron Yes. Resultado muy estable. |
| **Yes**, Estabilidad 80% | 8 de {STABILITY_REPEATS} corridas dieron Yes. Cumple el umbral de estabilidad. |
| **Yes**, Estabilidad 60%, Resultado Alternativo: Partial (3/4 restantes) | El resultado modal es Yes, pero es inestable; cuando cambia, tiende a Partial. |

Se considera **estable** un resultado con **≥{STABILITY_THRESHOLD_PCT:.0f}%** de estabilidad.
Los resultados por debajo de ese umbral deben revisarse manualmente aunque la etiqueta modal
parezca favorable.

---
            """
        )

        # --- 3. Tipos de criterio ---
        st.markdown(
            """
### 3. ¿Qué significa cada tipo de criterio?

La columna **Tipo** te dice qué clase de juicio hace la rúbrica. Hay 6 tipos base y
algunos criterios combinan varios (p.ej. "Lista transversal SMART"). Saber el tipo
te ayuda a entender por qué el razonamiento luce como luce.

| Tipo base | Qué evalúa | Cómo se ve la regla | Ejemplo |
|---|---|---|---|
| **Binario** | Presencia o ausencia clara, sin matices. | Un solo TEST: sí/no. | "¿El presupuesto de evaluación está en partida separada?" |
| **Lista de verificación** | Múltiples elementos atómicos (A, B, C…). | Regla por cantidad: cumple X de N elementos. | El análisis del problema requiere descripción + fuentes + cuantificación + delimitación. |
| **Calidad narrativa** | Coherencia, claridad o convicción de un argumento. | Tests más interpretativos. Subjetividad usualmente Media o Alta. | "¿La teoría del cambio es plausible para no especialistas?" |
| **Condicional** | Solo aplica si una condición previa se cumple. | Primero verifica condición; si no se cumple → N/A. | Plantilla DCOMM exigida solo si presupuesto > 5M USD. |
| **Transversal** | Tema transversal (género, discapacidad, NIT, EAS, indígenas, ambiente). | Aplica el filtro DEDICADO vs MARCO (sección 7). | Inclusión de personas con discapacidad. |
| **Calibración** | Compara contra un benchmark externo (etiqueta CPO, marcador de género, % de presupuesto). | Pregunta: "¿el nivel del documento coincide con el benchmark?" | El nivel de ambición sobre discapacidad está a la par con la etiqueta del CPO. |

#### Tipos compuestos

Algunos criterios combinan dos o más de los tipos base. La etiqueta lo refleja:

- **Lista transversal** = lista de verificación + filtro DEDICADO vs MARCO
- **Lista condicional** = lista + se evalúa solo si se cumple una condición previa
- **Lista condicional transversal** = los tres combinados
- **Condicional binario** = binario + condicional (común en criterios con umbral USD)
- **Calibración transversal** = calibración + filtro DEDICADO vs MARCO
- **Binario calidad** = binario donde el "sí/no" depende de una valoración cualitativa
- **Binario presencia** = binario sobre presencia simple de un elemento (sección, partida)
- **Lista de verificación SMART** = lista que verifica los 5 atributos SMART (Específico, Medible, Alcanzable, Relevante, Temporal)

💡 **Por qué importa**: el tipo te indica el nivel de rigor mecánico del resultado.
*Binario* y *Lista* son muy reproducibles. *Calidad narrativa* y los compuestos con
"transversal" tienen más espacio interpretativo — por eso suelen aparecer con
Subjetividad Media o Alta.

---
            """
        )

        # --- 4. Razonamiento ---
        st.markdown(
            """
### 4. ¿Cómo está estructurado el "Razonamiento"?

El razonamiento técnico te muestra **exactamente por qué el modelo llegó a ese resultado**.
En la descarga aparece en la hoja **Resultado Diagnostico** y sigue este formato:
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
            "RESULTADO: Partial",
            language="text",
        )
        st.markdown(
            """
**Cómo leerlo:**
- Cada **T** es un chequeo independiente. Resultado: *verdadero* (se cumple en el
  documento) o *falso* (no se cumple).
- La línea **DECISIÓN** explica qué regla se aplicó y por qué.
- El **RESULTADO** final es la conclusión técnica.

💡 **Por qué importa**: este desglose te permite ver *exactamente qué chequeo falló*. Si
el resultado es Parcial y T2 falló, ya sabes qué hay que arreglar en el proyecto (en este
caso, añadir referencia explícita al DWCP del país). Es información accionable, no solo
una etiqueta final.
            """
        )
        st.warning(
            "**Si el razonamiento NO sigue este formato** (no enumera T1, T2…), es señal de "
            "alerta: el modelo no aplicó la rúbrica mecánicamente. Verifica manualmente antes "
            "de confiar en el resultado."
        )
        st.markdown("---")

        # --- 5. Evidencia ---
        st.markdown(
            """
### 5. ¿Cómo está estructurada la "Evidencia"?

Son citas textuales del PRODOC que respaldan el resultado. Hay dos tipos válidos:
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

        # --- 6. Subjetividad ---
        st.markdown(
            """
### 6. ¿Qué es la columna "Subjetividad"?

Algunos criterios son **fáciles de verificar mecánicamente** (¿aparece un código ODS
como 8.5.2? Sí o no). Otros requieren **juicio cualitativo** (¿la teoría del cambio es
"convincente"?). La columna **Subjetividad** te dice cuánto juicio cualitativo queda
después de aplicar la rúbrica:

| Nivel | # criterios | Qué tan reproducible es | Qué hacer |
|---|---|---|---|
| 🟢 **Baja** | 27 | Muy reproducible — sus 10 corridas suelen devolver el mismo resultado. | Confiar. Auditar ~10% al azar. |
| 🟡 **Media** | 37 | Reproducible en general; algunos casos en el borde pueden variar. | Confiar; verificar manualmente los *Partial* y *Not Found*. |
| 🟠 **Alta** | 12 | Juicio cualitativo irreducible — puede variar entre corridas. | Tratar como hipótesis. Leer razonamiento y evidencia manualmente. |

💡 **En la práctica**: si tienes poco tiempo, enfoca tu revisión humana en los **12
criterios de Subjetividad Alta**. Para esos, el modelo es tu asistente, no tu juez final.

---
            """
        )

        # --- 7. DEDICADO vs MARCO ---
        st.markdown(
            """
### 7. El filtro DEDICADO vs MARCO — el concepto más importante

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
            "❌ **MARCO** — la mención del sujeto NO cuenta como cumplimiento si encaja en "
            "cualquiera de estos 5 patrones (definición idéntica a Tab 1):\n\n"
            "1. Mención en el **objetivo general / declaración de impacto** que enumera varios "
            "grupos.\n"
            "2. **Listas de partes interesadas, consulta o participantes** de investigación.\n"
            "3. **Enumeraciones de alcance** («…entre otros», «…incluyendo X, Y, Z»).\n"
            "4. **Lenguaje boilerplate** de inclusión.\n"
            "5. Cualquier pasaje donde el sujeto aparece en una **lista de ≥3 grupos sin "
            "seguimiento dedicado**.\n\n"
            "*Ejemplo*: «El proyecto beneficiará a poblaciones vulnerables, incluyendo mujeres, "
            "personas con discapacidad y comunidades indígenas, entre otros grupos.» → MARCO "
            "(encaja en los patrones 1 y 3)."
        )
        st.success(
            "✅ **DEDICADO** — la mención SÍ cuenta si es cualquiera de estos 5 elementos "
            "(definición idéntica a Tab 1):\n\n"
            "A. **Sub-objetivo, resultado o producto** cuyo título/propósito nombra al sujeto.\n"
            "B. **Indicador desagregado** por el sujeto o que lo mide específicamente.\n"
            "C. **Actividad** cuyo propósito principal aborda al sujeto.\n"
            "D. **Partida presupuestaria** o asignación de recursos para el sujeto.\n"
            "E. **Meta cuantificable** relativa al sujeto.\n\n"
            "*Ejemplo*: «Indicador 3.2: número de personas con discapacidad capacitadas en el "
            "oficio (meta: 200). Partida presupuestaria 4.1.5: USD 35,000 para accesibilidad de "
            "materiales y lenguaje de señas.» → DEDICADO (cumple B, D y E)."
        )
        st.info(
            "💡 **Consecuencia práctica**: una propuesta puede mencionar género 50 veces y aún "
            "así sacar **No** si todas las menciones son MARCO. Lo que cuenta es la "
            "dedicación operativa, no la frecuencia del lenguaje."
        )

        st.markdown(
            """
#### ¿Cómo se diferencia este filtro respecto a la pestaña Tab 1 actual?

La **definición es idéntica** (los 5 patrones de MARCO y los 5 elementos de DEDICADO
de arriba son los mismos en ambas pestañas). Lo que cambia es **cómo se aplica**:

| Aspecto | Tab 1 (actual) | Tab 7 (v3, esta pestaña) |
|---|---|---|
| **Definición de DEDICADO / MARCO** | Los 5 patrones / 5 elementos descritos arriba | **Idénticos** a Tab 1 |
| **¿A qué criterios se aplica?** | A todos | Solo a los criterios cuya rúbrica lo invoca (los marcados con Transversales ≠ Ninguno) |
| **¿Cuántos elementos son obligatorios?** | Siempre 5: A/B/C/D/E con peso igual | Varía por criterio. Algunos elementos son opcionales según la naturaleza del proyecto |
| **¿Cómo se traduce a resultado?** | Escala fija: 0→No, 1-2→Parcial, 3-4→Parcial-o-Sí, 5→Sí | Cada criterio tiene su regla propia (p.ej. ≥2 de 3 obligatorios = Sí) |
| **¿Cómo se ve en el razonamiento?** | Prosa con etiquetas [DEDICATED]/[FRAMING] | Lista de chequeos T1/T2/T3 con verdadero/falso explícito |

**Por qué importa**: el mismo PRODOC puede recibir resultados diferentes entre Tab 1 y
esta pestaña en criterios transversales. La razón habitual: la rúbrica de v3 (validada
con la revisión de su equipo) admite *Sí* con menos elementos DEDICADOS cuando la
naturaleza del proyecto lo justifica.
            """
        )

    # Results view: public first, audit detail second.
    st.markdown("### Lectura amigable")
    st.caption(
        "📌 La columna **Pregunta orientadora** es el enunciado general de la "
        "subsección del cuestionario OIT (1.1, 1.2, …). Aparece como contexto y "
        "**no es evaluada**. Solo la columna **Criterio** dispara los TESTS de la rúbrica. "
        "La columna **Estabilidad (%)** resume las 10 corridas independientes por criterio. "
        "Los criterios largos se mantienen completos en la tabla y en la descarga."
    )
    st.dataframe(df_public, use_container_width=True, height=500)

    with st.expander("Auditoría técnica: TESTS, decisión y evidencia completa", expanded=False):
        st.dataframe(df_res, use_container_width=True, height=500)

    # Download xlsx
    xlsx_bytes = v3_core.results_to_xlsx_bytes(results)
    st.download_button(
        "📥 Descargar resultados v3 (.xlsx)",
        xlsx_bytes,
        file_name=f"valoracion_v3_{os.path.splitext(st.session_state.get('v3_doc_name','sin_nombre'))[0]}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
