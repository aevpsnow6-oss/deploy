from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


BASE = Path(__file__).resolve().parent
UPDATED = "30 de abril de 2026"
APP = "oli_v6_deploy.py"


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False, color=None):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(str(text))
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_after = Pt(3)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        set_cell_text(hdr[i], h, bold=True, color=(255, 255, 255))
        shade_cell(hdr[i], "002F6C")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value)
    if widths:
        for row in table.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = Inches(width)
    doc.add_paragraph()
    return table


def add_bullets(doc, items, level=0):
    style = "List Bullet" if level == 0 else "List Bullet 2"
    for item in items:
        doc.add_paragraph(item, style=style)


def add_numbered(doc, items):
    for idx, item in enumerate(items, start=1):
        para = doc.add_paragraph()
        para.paragraph_format.left_indent = Inches(0.25)
        para.paragraph_format.first_line_indent = Inches(-0.25)
        para.add_run(f"{idx}. {item}")


def h(doc, text, level=1):
    doc.add_heading(text, level=level)


def p(doc, text="", style=None):
    para = doc.add_paragraph(style=style)
    if text:
        para.add_run(text)
    return para


def codeblock(doc, lines):
    if isinstance(lines, str):
        lines = lines.splitlines()
    for line in lines:
        para = doc.add_paragraph()
        run = para.add_run(line)
        run.font.name = "Courier New"
        run.font.size = Pt(9)
        para.paragraph_format.space_after = Pt(0)


def set_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)

    for name, size, color in [
        ("Title", 22, "002F6C"),
        ("Heading 1", 16, "002F6C"),
        ("Heading 2", 13, "0072CE"),
        ("Heading 3", 11.5, "333333"),
    ]:
        style = styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)


def cover(doc, title, subtitle):
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_p.add_run(title)
    run.bold = True
    run.font.size = Pt(20)
    run.font.color.rgb = RGBColor(0, 47, 108)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Caja de Herramientas para el Mejor Desempeño de los Proyectos")
    r.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = RGBColor(0, 47, 108)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(f"{subtitle}\nFuente técnica: {APP}\nActualizado: {UPDATED}")
    doc.add_paragraph()


def save_doc(filename, title, subtitle, build):
    doc = Document()
    set_styles(doc)
    cover(doc, title, subtitle)
    build(doc)
    doc.save(BASE / filename)


def doc1(doc):
    h(doc, "1. Alcance y fuente de verdad", 1)
    p(doc, "Esta documentación técnica describe exclusivamente la aplicación principal oli_v6_deploy.py. Los archivos oli_v6_deploy_core.py y oli_v6_deploy_recommendations.py son variantes separadas y no se usan como base de esta actualización documental.")
    p(doc, "La aplicación principal es una app Streamlit monolítica, con funciones auxiliares, lógica de extracción documental, evaluación por IA, clasificación de recomendaciones, visualización y exportación en un único archivo Python.")

    h(doc, "2. Arquitectura de alto nivel", 1)
    add_table(doc, ["Capa", "Implementación en el código", "Responsabilidad"], [
        ["Interfaz", "Streamlit con st.tabs, st.session_state, st.cache_data y CSS institucional ILO", "Carga de documentos, selección de criterios, visualizaciones y descargas."],
        ["Procesamiento documental", "python-docx, docx2python, BeautifulSoup y extracción jerárquica de encabezados", "Convierte DOCX/TXT en texto trazable por sección, párrafo, tabla y metadatos."],
        ["IA generativa", "OpenAI SDK mediante OpenAI(api_key=os.getenv('OPENAI_API_KEY'))", "Evaluaciones, síntesis, chat documental, crítica de respuestas y resúmenes ejecutivos."],
        ["Búsqueda vectorial", "text-embedding-3-large, numpy, torch y FAISS IndexFlatIP", "Carga un índice de recomendaciones heredado y usa FAISS para RAG en documentos grandes; Tabs 5 y 6 clasifican con similitud coseno en numpy."],
        ["Datos", "Archivos Excel, CSV pipe-separated y tensores .pt en el directorio de la app o rutas definidas por st.secrets", "No hay base de datos SQL; los archivos son el almacén operativo."],
        ["Exportación", "xlsxwriter, BytesIO y zipfile", "Genera XLSX y ZIP con resultados, evidencias, estructura extraída y reportes."],
    ], widths=[1.3, 2.4, 3.1])

    h(doc, "3. Componentes funcionales activos", 1)
    add_table(doc, ["Pestaña", "Nombre visible", "Función principal"], [
        ["1", "Valoración Preliminar de Calidad de Proyectos", "Evalúa documentos de diseño contra Appraisal Checklist_2025 es-419.xlsx, aplica análisis A-E y regla A/B para temas transversales, genera crítica y síntesis por pregunta, subsección y sección."],
        ["2", "Diagnóstico de Atributos Específicos", "Evalúa secciones seleccionadas con rúbricas de Rubricas_6ago2025.xlsx para participación durante evaluación, género y transición justa moderna."],
        ["3", "Diagnóstico de Sostenibilidad del Proyecto", "Evalúa sostenibilidad con Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx y visualiza puntajes por dimensión y criterio."],
        ["4", "Pregúntale a tus Documentos", "Chat con uno o varios DOCX/TXT, usando contexto completo para documentos pequeños y RAG con FAISS para documentos grandes."],
        ["5", "Clasificación de Recomendaciones", "Clasifica recomendaciones en español contra Frame_Recommendations_English.xlsx, con filtros, treemaps, evolución temporal, análisis profundo y resumen."],
        ["6", "Recommendation Classification", "Versión en inglés de la clasificación de recomendaciones con la misma lógica de Tab 5."],
    ], widths=[0.7, 2.2, 4.1])

    h(doc, "4. Dependencias", 1)
    p(doc, "requirements.txt fija o declara las siguientes dependencias principales:")
    codeblock(doc, [
        "streamlit==1.27.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "faiss-cpu==1.7.4",
        "openai>=1.0.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "torch==2.0.1",
        "plotly==5.15.0",
        "xlsxwriter==3.1.2",
        "python-dotenv==1.0.0",
        "openpyxl",
        "rich==13.9.0",
        "python-docx==0.8.11",
        "docx2python",
        "tiktoken>=0.5.0",
    ])

    h(doc, "5. Modelos de IA y límites de contexto", 1)
    add_bullets(doc, [
        "Modelo principal de evaluación, crítica, síntesis y chat: gpt-5-mini.",
        "Embeddings: text-embedding-3-large.",
        "Análisis profundo de recomendaciones: selector entre gpt-4o-mini y gpt-4o; la función de análisis de pares usa gpt-4o-mini por defecto.",
        "Límite documental operativo: truncate_to_token_limit corta a aproximadamente 110,000 tokens para dejar espacio al prompt y respuesta.",
        "En el chat documental, textos de hasta 100,000 caracteres usan contexto completo; textos mayores generan fragmentos de 2,000 caracteres con solape de 300 y RAG con FAISS.",
    ])

    h(doc, "5.1 Regla de temas transversales en Tab 1", 2)
    p(doc, "Tab 1 mantiene el marco A-E general para preguntas con sujeto específico, pero aplica una regla reducida cuando la pregunta corresponde a un tema transversal configurado.")
    add_table(doc, ["Tema transversal reconocido", "Criterios de detección en código"], [
        ["Género", "Aliases como género, igualdad de género, enfoque de género, mujeres y niñas."],
        ["No discriminación", "Aliases como no discriminación, antidiscriminación, discriminación, igualdad de oportunidades y trato igualitario."],
        ["Discapacidad", "Aliases como discapacidad, personas con discapacidad, PCD, accesibilidad e inclusión de personas con discapacidad."],
        ["Diálogo social y tripartismo", "Aliases como diálogo social, tripartismo, tripartita, sindicatos, organizaciones de trabajadores y organizaciones de empleadores."],
        ["Sostenibilidad medioambiental", "Aliases como sostenibilidad ambiental, medio ambiente, ambiental, cambio climático, acción climática y economía verde."],
    ], widths=[2.1, 4.7])
    add_table(doc, ["Criterio transversal", "Definición operativa"], [
        ["A", "Presencia operacional del tema en objetivo, producto o actividad."],
        ["B", "Presupuesto, recursos o línea presupuestaria para alguna actividad correspondiente."],
    ], widths=[1.3, 5.5])
    p(doc, "La calificación automática para estos temas es: Yes cuando A y B están presentes; Partial cuando solo A o solo B está presente; No cuando no están presentes ni A ni B.")

    h(doc, "6. Inicialización y caché", 1)
    add_bullets(doc, [
        "load_data() carga df_complete_all_full.xlsx, df_split_actions.xlsx y prepara columnas normalizadas.",
        "load_extended_data() agrega analyzed_recommendations_plans_v5.csv cuando está disponible y guarda analyzed_df en st.session_state.",
        "load_embeddings() carga emb_Recomm_rec_cl_4.pt y Recommendation_RAG_Metadata.pt y construye un índice FAISS en memoria; en el código actual ese índice pertenece al flujo de búsqueda de recomendaciones que quedó comentado, no a las pestañas activas de clasificación.",
        "Las funciones de carga usan st.cache_data; limpiar caché de Streamlit fuerza una recarga de archivos.",
        "El bloque principal inicializa datos y embeddings dos veces en el archivo; el comportamiento final es equivalente, pero una futura limpieza puede consolidarlo.",
    ])

    h(doc, "7. Despliegue", 1)
    p(doc, "Ejecución local estándar:")
    codeblock(doc, "streamlit run oli_v6_deploy.py")
    p(doc, "En producción, si STREAMLIT_ENV=production, el código toma rutas de archivos desde st.secrets. La clave de OpenAI se lee desde la variable de entorno OPENAI_API_KEY.")
    add_table(doc, ["Secreto o variable", "Uso en el código"], [
        ["OPENAI_API_KEY", "Autenticación del cliente OpenAI."],
        ["df_path", "Ruta de df_complete_all_full.xlsx en producción."],
        ["df_raw_path", "Ruta de df_split_actions.xlsx en producción."],
        ["embeddings_path", "Ruta de emb_Recomm_rec_cl_4.pt en producción."],
        ["structured_embeddings_path", "Ruta de Recommendation_RAG_Metadata.pt en producción."],
        ["analyzed_recommendations_path", "Ruta opcional de analyzed_recommendations_plans_v5.csv en producción."],
        ["lessons_embeddings_path", "Ruta de emb_LL_ll_cl_4.pt para lecciones aprendidas."],
        ["structured_lessons_path", "Ruta de lessons_metadata.pt para lecciones aprendidas."],
    ], widths=[2.1, 4.7])

    h(doc, "8. Riesgos técnicos operativos", 1)
    add_bullets(doc, [
        "La app no implementa autenticación, autorización ni auditoría propia; depende del control del entorno de hosting.",
        "Los documentos cargados y sus extractos se envían a OpenAI para análisis; deben tratarse como datos potencialmente sensibles.",
        "El almacén de datos es file-based; reemplazar un archivo de entrada cambia el comportamiento de la app sin migraciones de esquema.",
        "Las rutas locales son relativas al directorio desde el cual se ejecuta Streamlit.",
        "Los archivos de salida XLSX/ZIP pueden contener citas textuales del documento original y deben manejarse con la misma confidencialidad.",
    ])


def doc2(doc):
    h(doc, "1. Alcance del código fuente", 1)
    p(doc, "El código documentado es oli_v6_deploy.py. Las variantes oli_v6_deploy_core.py y oli_v6_deploy_recommendations.py existen en el proyecto, pero esta documentación no infiere comportamiento desde ellas.")
    p(doc, "Regla de mantenimiento: cuando se cambie la app principal, revisar si el mismo cambio debe propagarse a las variantes separadas antes de entregar una versión operativa al equipo ILO.")

    h(doc, "2. Estructura relevante del directorio", 1)
    add_table(doc, ["Archivo", "Rol operativo"], [
        ["oli_v6_deploy.py", "Aplicación Streamlit principal y fuente de verdad para esta documentación."],
        ["requirements.txt", "Dependencias Python usadas por la app."],
        ["df_complete_all_full.xlsx", "Dataset principal de recomendaciones enriquecidas."],
        ["df_split_actions.xlsx", "Dataset crudo/desagregado usado para recomendaciones y años."],
        ["analyzed_recommendations_plans_v5.csv", "Análisis adicional de recomendaciones y planes; separador pipe."],
        ["emb_Recomm_rec_cl_4.pt", "Embeddings de recomendaciones para FAISS."],
        ["Recommendation_RAG_Metadata.pt", "Metadatos alineados al embedding de recomendaciones."],
        ["Appraisal Checklist_2025 es-419.xlsx", "Preguntas de valoración preliminar, hoja rubric."],
        ["Rubricas_6ago2025.xlsx", "Rúbricas de atributos específicos usadas por Tab 2."],
        ["Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx", "Rúbrica activa de sostenibilidad para Tab 3."],
        ["Recommendations_World.xlsx", "Recomendaciones mundiales para clasificación en Tabs 5 y 6."],
        ["Frame_Recommendations_English.xlsx", "Marco de dimensiones/subdimensiones para clasificación."],
    ], widths=[2.7, 4.1])

    h(doc, "3. Instalación local", 1)
    add_numbered(doc, [
        "Usar Python compatible con las dependencias fijadas del proyecto.",
        "Crear y activar un entorno virtual.",
        "Instalar dependencias con pip install -r requirements.txt.",
        "Configurar OPENAI_API_KEY como variable de entorno.",
        "Ejecutar streamlit run oli_v6_deploy.py desde el directorio del proyecto.",
    ])
    codeblock(doc, [
        "python -m venv .venv",
        "source .venv/bin/activate",
        "pip install -r requirements.txt",
        "streamlit run oli_v6_deploy.py",
    ])
    p(doc, "La clave OPENAI_API_KEY debe configurarse en el entorno o gestor de secretos elegido por ILO; no se debe registrar ni compartir su valor real en la documentación.")

    h(doc, "4. Mapa de funciones principales", 1)
    add_table(doc, ["Bloque", "Funciones o clases", "Descripción"], [
        ["Excel/exportación", "to_excel", "Convierte DataFrames en XLSX usando xlsxwriter y desactiva strings_to_urls."],
        ["Tokens", "truncate_to_token_limit", "Recorta texto a límites de tokens con tiktoken."],
        ["Embeddings y RAG", "get_embedding_with_retry, find_similar_recommendations, find_recommendations_by_term_matching, load_embeddings", "Genera embeddings y mantiene funciones de búsqueda FAISS/matching textual; el buscador de recomendaciones asociado está comentado en la interfaz actual."],
        ["Análisis de recomendaciones", "AnalysisCache, analyze_recommendation_plan_pair, run_row_analysis, generate_executive_summary", "Evalúa coherencia de planes, calidad, factibilidad, impacto, innovación y genera resumen ejecutivo."],
        ["Carga de datos", "prepare_additional_data, load_data, load_extended_data", "Normaliza columnas, años, categorías y fusiona análisis adicional."],
        ["Extracción documental", "extract_docx_structure_enhanced, validate_extraction, extract_document_content", "Extrae jerarquía, tablas, métricas y texto desde DOCX."],
        ["Valoración preliminar", "load_appraisal_questions, analyze_question_with_llm_tab1, detect_transversal_matter, _critic_impl, _apply_critic_gate_and_render, _apply_transversal_gate_and_render, synthesize_subsection_analysis, synthesize_section_analysis, create_results_download_with_sections", "Pregunta por pregunta, aplica crítica A-E o regla transversal A/B y genera XLSX/ZIP multinivel."],
        ["Sostenibilidad y atributos", "evaluate_criterion_with_llm, synthesize_evaluations", "Evalúa criterios con puntuación 1-5, análisis y evidencia."],
        ["Clasificación", "EmbeddingsCache, classify_recommendations, verify_match_with_llm", "Clasifica recomendaciones contra marco, con caché persistente y verificación opcional por LLM."],
    ], widths=[1.6, 2.5, 2.7])

    h(doc, "5. Convenciones de estado", 1)
    add_bullets(doc, [
        "Cada pestaña usa claves propias de st.session_state, por ejemplo tab1_results_df, document_extracted_tab2, tab3_results, classified_world_df y deep_analysis_df.",
        "Los resultados se mantienen durante la sesión y pueden limpiarse con botones de limpiar resultados.",
        "La carga de nuevos archivos se detecta con hash(uploaded_file.getvalue()) y reinicia el estado de extracción cuando corresponde.",
        "Los outputs descargables se generan en memoria con BytesIO; no se guardan automáticamente en disco.",
    ])

    h(doc, "5.1 Lógica especial de temas transversales", 2)
    p(doc, "TRANSVERSAL_MATTERS es un diccionario en código que reconoce género, no discriminación, discapacidad, diálogo social y tripartismo, y sostenibilidad medioambiental mediante aliases normalizados sin acentos. Cuando detect_transversal_matter encuentra uno de esos temas en la pregunta, _critic_impl usa CRITIC_SCHEMA_TRANSVERSAL y _apply_transversal_gate_and_render en vez del esquema A-E general.")
    p(doc, "La salida de razonamiento para esos casos inicia con una línea de auditoría de criterios transversales: A corresponde a objetivo/producto/actividad y B a presupuesto. La regla aplicada es Yes=A+B, Partial=A o B, No=sin A ni B.")

    h(doc, "6. Manejo de errores", 1)
    add_bullets(doc, [
        "Falta de archivos críticos: la app muestra st.error o st.warning y detiene el flujo de esa pestaña cuando el recurso es obligatorio.",
        "Falta de OPENAI_API_KEY: la app muestra warning global y las funciones que requieren OpenAI devuelven error o advertencia.",
        "Errores de análisis por IA: se capturan y se devuelven en columnas Error o mensajes de chat.",
        "Respuestas JSON inválidas: varios flujos intentan limpiar fences de Markdown; si falla el parseo, se devuelve score 0 o score por defecto con error explicativo.",
        "Rate limits: algunas evaluaciones reintentan con backoff; en otros casos se reduce concurrencia por diseño.",
    ])

    h(doc, "7. Pruebas y verificación recomendadas", 1)
    add_bullets(doc, [
        "Ejecutar test_two_part_parsing.py al cambiar parse_two_part_question o la lógica de preguntas en dos partes.",
        "Probar manualmente una pregunta por cada tema transversal para confirmar detección por alias y aplicación de la regla A/B.",
        "Probar manualmente un DOCX con Heading 1/2 y tablas en Tabs 1, 2 y 3.",
        "Probar chat con TXT pequeño y DOCX grande para validar los dos caminos: contexto completo y RAG.",
        "Probar Tab 5 con una muestra limitada y LLM Verification apagado y encendido.",
        "Verificar que cada descarga XLSX/ZIP abre correctamente y conserva columnas de evidencia.",
    ])

    h(doc, "8. Deuda técnica visible", 1)
    add_bullets(doc, [
        "Hay bloques comentados extensos de versiones anteriores; no son funcionales pero dificultan mantenimiento.",
        "La inicialización de datos y embeddings aparece dos veces en el bloque principal.",
        "Algunos mensajes de error de Tab 3 todavía mencionan PRODOC_rubric.xlsx aunque la rúbrica activa es Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx.",
        "Existe un bloque heredado comentado de Tab 2 que también carga gender_rubric; no es el flujo activo. La implementación activa sí carga la hoja rubric_gender_ y la expone como Integración del Enfoque de Género.",
        "La lista de aliases de TRANSVERSAL_MATTERS está definida en código; si ILO cambia terminología oficial o agrega temas, debe actualizarse ese diccionario y probarse Tab 1.",
        "El archivo lessons_metadata.pt es esperado por load_lessons_embeddings, pero no aparece en el inventario local observado; si esa funcionalidad se activa, debe suministrarse.",
        "La app no separa claramente capa de interfaz, dominio y servicios externos; cambios grandes deben hacerse con pruebas manuales por pestaña.",
    ])


def doc3(doc):
    h(doc, "1. Modelo de datos", 1)
    p(doc, "La aplicación no utiliza una base de datos relacional. El modelo de datos es file-based: Excel, CSV delimitado por pipe, tensores PyTorch y cachés pickle locales.")
    p(doc, "Las relaciones son lógicas y se realizan en memoria con pandas. No hay migraciones, constraints SQL ni control transaccional.")

    h(doc, "2. Inventario de archivos de datos", 1)
    add_table(doc, ["Archivo", "Cargado por", "Uso"], [
        ["df_complete_all_full.xlsx", "load_data", "Base principal. Crea index_df desde ID_Recomendacion, normaliza nombres de columnas y dimension/subdim."],
        ["df_split_actions.xlsx", "load_data", "Base raw con Recommendation date; aporta year e index_df para completar faltantes."],
        ["analyzed_recommendations_plans_v5.csv", "load_extended_data", "CSV con sep='|'. Agrega análisis adicional, tags, scores y métricas para visualizaciones."],
        ["emb_Recomm_rec_cl_4.pt", "load_embeddings", "Matriz de embeddings de recomendaciones."],
        ["Recommendation_RAG_Metadata.pt", "load_embeddings", "Metadatos alineados a emb_Recomm_rec_cl_4.pt."],
        ["emb_LL_ll_cl_4.pt", "load_lessons_embeddings", "Embeddings para lecciones aprendidas."],
        ["lessons_metadata.pt", "load_lessons_embeddings", "Metadatos esperados para lecciones aprendidas; debe existir si se usa ese flujo."],
        ["Appraisal Checklist_2025 es-419.xlsx", "load_appraisal_questions", "Hoja rubric; requiere Pregunta_Realizada y usa Tema para renumerar preguntas."],
        ["TRANSVERSAL_MATTERS", "Tab 1", "Diccionario en código, no archivo externo. Reconoce género, no discriminación, discapacidad, diálogo social y tripartismo, y sostenibilidad medioambiental para aplicar regla A/B."],
        ["Rubricas_6ago2025.xlsx", "Tab 2", "Hojas rubric_engagement, rubric_performance, rubric_parteval, rubric_gender_, rubric_TJ_Traditional y rubric_TJ_TJ; la UI activa evalúa parteval, género y TJ moderno."],
        ["Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx", "Tab 3", "Hoja rubric; fuente activa de criterios de sostenibilidad."],
        ["Recommendations_World.xlsx", "Tabs 5 y 6", "Dataset mundial para clasificación; puede reemplazarse por upload XLSX con Recommendation description."],
        ["Frame_Recommendations_English.xlsx", "Tabs 5 y 6", "Marco de referencia; debe contener texto_merged, dimension y subdim."],
        ["embeddings_cache.pkl", "EmbeddingsCache", "Caché local persistente de embeddings generados en clasificación."],
        ["analysis_cache.pkl", "AnalysisCache", "Caché local persistente de análisis profundo de recomendaciones."],
    ], widths=[2.5, 1.7, 2.8])

    h(doc, "3. Relaciones lógicas", 1)
    add_bullets(doc, [
        "df_complete_all_full.xlsx y df_split_actions.xlsx se alinean por index_df.",
        "analyzed_recommendations_plans_v5.csv se fusiona con df por index_df cuando hay columnas nuevas.",
        "emb_Recomm_rec_cl_4.pt debe mantener el mismo orden que Recommendation_RAG_Metadata.pt para el índice de recomendaciones cargado en memoria.",
        "Recommendations_World.xlsx se clasifica contra cada fila de Frame_Recommendations_English.xlsx usando texto_merged como definición semántica; este flujo usa similitud coseno con numpy, no FAISS.",
        "Los datasets de recomendaciones pueden tener filas duplicadas por múltiples atributos; Tabs 5 y 6 muestran conteos de registros y recomendaciones únicas.",
    ])

    h(doc, "4. Transformaciones principales", 1)
    add_table(doc, ["Transformación", "Detalle"], [
        ["Normalización de columnas", "Los espacios y puntos se reemplazan por guion bajo en load_data."],
        ["Años", "Recommendation_date y Recommendation date se convierten con pandas; años previos a 2018 en analyzed_df se fijan a 2018."],
        ["Categorías", "prepare_additional_data estandariza campos categóricos, listas/tags y clasificaciones cuando existen."],
        ["FAISS", "El índice de recomendaciones se construye al inicio y Tab 4 crea un índice temporal para RAG cuando el texto cargado supera 100,000 caracteres."],
        ["Temas transversales", "En Tab 1, las preguntas detectadas por TRANSVERSAL_MATTERS se califican con A=objetivo/producto/actividad y B=presupuesto; Yes requiere A+B, Partial requiere A o B, No requiere ausencia de ambos."],
        ["Clasificación", "Cada recomendación única se embebe una vez, se normaliza y se compara por similitud coseno contra el marco usando numpy."],
        ["Exportación", "Los resultados se escriben a XLSX con columnas ordenadas y strings_to_urls desactivado."],
    ], widths=[2.0, 4.8])

    h(doc, "5. Campos mínimos por flujo", 1)
    add_table(doc, ["Flujo", "Campos mínimos requeridos"], [
        ["Valoración preliminar", "Appraisal Checklist: Pregunta_Realizada; se recomienda Tema. Las preguntas de temas transversales se reconocen por aliases en código."],
        ["Atributos específicos", "Rubricas_6ago2025: Indicador, Dimensión y niveles de desempeño por hoja activa."],
        ["Sostenibilidad", "Rúbrica activa: columnas de dimensión, criterio, indicador y niveles según hoja rubric."],
        ["Clasificación", "Recommendations_World: Recommendation description. Frame: texto_merged, dimension, subdim."],
        ["Análisis profundo", "Recommendation description y Action plan; Comments mejora el análisis si existe."],
        ["Resumen ejecutivo", "Recommendation description; usa Management response, Comments y Action plan si existen."],
    ], widths=[2.0, 4.8])

    h(doc, "6. Respaldo y actualización de datos", 1)
    add_bullets(doc, [
        "Antes de reemplazar un Excel, CSV o tensor .pt, conservar una copia fechada fuera del directorio runtime.",
        "Después de reemplazar un archivo de datos, reiniciar la app o limpiar caché de Streamlit.",
        "Si se regeneran embeddings, regenerar también el archivo de metadatos alineado; no mezclar tensores y metadatos de corridas distintas.",
        "Si cambia el esquema de Recommendations_World.xlsx o Frame_Recommendations_English.xlsx, validar Tabs 5 y 6 con una muestra pequeña antes del procesamiento completo.",
        "No editar manualmente embeddings_cache.pkl o analysis_cache.pkl; si se sospecha corrupción, borrar el archivo con la app detenida para que se regenere.",
    ])


def doc4(doc):
    h(doc, "1. Configuración obligatoria", 1)
    add_table(doc, ["Elemento", "Valor esperado"], [
        ["Comando local", "streamlit run oli_v6_deploy.py"],
        ["Variable de IA", "OPENAI_API_KEY disponible como variable de entorno."],
        ["Modo producción", "STREAMLIT_ENV=production cuando las rutas se toman de st.secrets."],
        ["Directorio de ejecución", "Directorio que contiene los Excel, CSV y .pt requeridos, salvo que se usen rutas de producción."],
        ["Dependencias", "Instaladas desde requirements.txt."],
    ], widths=[2.0, 4.8])

    h(doc, "2. Secretos y rutas", 1)
    p(doc, "El código lee OPENAI_API_KEY con os.getenv. En Streamlit Cloud u otro hosting, la clave debe exponerse como variable de entorno efectiva para el proceso Python.")
    add_table(doc, ["Nombre", "Cuándo se usa", "Contenido esperado"], [
        ["df_path", "Producción", "Ruta accesible a df_complete_all_full.xlsx."],
        ["df_raw_path", "Producción", "Ruta accesible a df_split_actions.xlsx."],
        ["embeddings_path", "Producción", "Ruta accesible a emb_Recomm_rec_cl_4.pt."],
        ["structured_embeddings_path", "Producción", "Ruta accesible a Recommendation_RAG_Metadata.pt."],
        ["analyzed_recommendations_path", "Producción opcional", "Ruta accesible a analyzed_recommendations_plans_v5.csv."],
        ["lessons_embeddings_path", "Producción para lecciones", "Ruta accesible a emb_LL_ll_cl_4.pt."],
        ["structured_lessons_path", "Producción para lecciones", "Ruta accesible a lessons_metadata.pt."],
    ], widths=[2.0, 1.8, 3.0])

    h(doc, "3. Control de acceso", 1)
    p(doc, "oli_v6_deploy.py no implementa usuarios, roles, contraseñas ni permisos por funcionalidad. El control de acceso debe aplicarse fuera de la app, en Streamlit Cloud, servidor institucional, proxy, red privada o mecanismo equivalente.")
    add_bullets(doc, [
        "Solo usuarios autorizados deben poder abrir la aplicación.",
        "El repositorio y los archivos de datos deben tener permisos separados: lectura para operación, escritura solo para mantenimiento.",
        "El acceso a OPENAI_API_KEY debe limitarse al entorno de ejecución y administradores técnicos.",
        "Las descargas generadas por la app son responsabilidad del usuario que las produce.",
    ])

    h(doc, "4. Seguridad de datos", 1)
    add_bullets(doc, [
        "Los documentos subidos se procesan en memoria y mediante archivos temporales; los temporales se eliminan al finalizar la extracción cuando el flujo llega a limpieza.",
        "El contenido de documentos, preguntas y evidencias puede enviarse a OpenAI para generar evaluaciones y respuestas.",
        "st.session_state conserva texto extraído y resultados durante la sesión activa.",
        "Los archivos descargables XLSX/ZIP pueden contener citas textuales, razonamientos y evidencia del documento fuente.",
        "La aplicación no cifra salidas locales ni implementa retención de datos; estas políticas deben definirse en el entorno institucional.",
    ])

    h(doc, "5. Parámetros de operación segura", 1)
    add_table(doc, ["Parámetro", "Valor en código", "Impacto"], [
        ["MAX_WORKERS global para valoración preliminar", "48", "Alta concurrencia para preguntas; puede exigir límites de API suficientes."],
        ["MAX_WORKERS Tab 2", "3", "Reduce riesgo de rate limit en evaluación por rúbricas."],
        ["MAX_WORKERS Tab 3", "8", "Procesa criterios de sostenibilidad en paralelo."],
        ["Chat RAG", ">100,000 caracteres", "Evita enviar documentos grandes completos en cada consulta."],
        ["Token limit", "110,000 tokens", "Recorta documentos muy extensos antes de llamar al modelo."],
        ["Clasificación batch", "20 textos por lote lógico", "Controla generación de embeddings de recomendaciones únicas."],
    ], widths=[2.2, 1.6, 3.0])

    h(doc, "6. Lista de verificación antes de producción", 1)
    add_bullets(doc, [
        "Confirmar que OPENAI_API_KEY existe en el proceso runtime y que la cuenta tiene presupuesto suficiente.",
        "Confirmar que todos los archivos de datos obligatorios abren con pandas/torch en el entorno de producción.",
        "Configurar acceso privado a la app en el hosting.",
        "Ejecutar una prueba por pestaña con documentos no sensibles.",
        "Verificar que las descargas se generan, se abren y no exponen más información de la necesaria.",
        "Documentar internamente quién puede reemplazar archivos de datos y quién puede desplegar código.",
    ])

    h(doc, "7. Respuesta a incidentes", 1)
    add_numbered(doc, [
        "Deshabilitar el acceso externo a la app desde el hosting o proxy.",
        "Revocar y rotar OPENAI_API_KEY si se sospecha exposición.",
        "Conservar copias de archivos descargados o logs disponibles solo si la política institucional lo permite.",
        "Restaurar archivos de datos desde respaldo verificado.",
        "Reactivar la app después de una prueba completa de Tabs 1 a 6.",
    ])


def doc5(doc):
    h(doc, "1. Descripción funcional", 1)
    p(doc, "La aplicación ayuda a revisar documentos de proyecto y evaluación, consultar documentos cargados, clasificar recomendaciones y analizar respuestas institucionales. La interfaz activa tiene seis pestañas.")

    h(doc, "2. Requisitos para usuarios", 1)
    add_bullets(doc, [
        "Usar navegador moderno con conexión estable.",
        "Contar con autorización institucional para cargar documentos en la aplicación.",
        "Subir DOCX con estilos de encabezado de Word cuando se use valoración, atributos o sostenibilidad.",
        "Revisar que los resultados de IA sean evidencia de apoyo y no sustituyan revisión técnica humana.",
    ])

    h(doc, "3. Navegación principal", 1)
    add_table(doc, ["Pestaña", "Cuándo usarla", "Salida principal"], [
        ["Valoración Preliminar de Calidad de Proyectos", "Revisión integral de documento de diseño de proyecto.", "ZIP con XLSX de preguntas, análisis por subsección/sección, plantilla y resumen TXT."],
        ["Diagnóstico de Atributos Específicos", "Evaluación focalizada de participación en evaluación, género o transición justa.", "ZIP con XLSX por rúbrica seleccionada."],
        ["Diagnóstico de Sostenibilidad del Proyecto", "Revisión de sostenibilidad de PRODOC o documento afín.", "ZIP con XLSX y gráficos de puntajes."],
        ["Pregúntale a tus Documentos", "Consulta conversacional sobre uno o varios DOCX/TXT.", "Respuesta en pantalla con memoria durante la sesión."],
        ["Clasificación de Recomendaciones", "Clasificar recomendaciones en español y analizar subconjuntos.", "Treemaps, evolución temporal, análisis profundo XLSX y resumen ejecutivo XLSX."],
        ["Recommendation Classification", "Mismo flujo de clasificación en inglés.", "English outputs and downloads."],
    ], widths=[2.2, 2.5, 2.1])

    h(doc, "4. Uso de Tab 1: Valoración Preliminar", 1)
    add_numbered(doc, [
        "Descargar o revisar la rúbrica Appraisal Checklist_2025 es-419.xlsx desde la pestaña si es necesario.",
        "Subir un archivo DOCX con encabezados de Word.",
        "Presionar Extraer Documento y revisar la estructura extraída.",
        "Seleccionar secciones y filtros de preguntas si aplica.",
        "Presionar Analizar documento.",
        "Revisar métricas, tabla de resultados, razonamiento, evidencia y evaluación crítica.",
        "Descargar appraisal_checklist_results.zip.",
    ])
    p(doc, "Interpretación: las respuestas son Yes, No, Partial o Not Found. Para preguntas con sujeto específico, la app aplica el marco A-E: sub-objetivo/output, indicador, actividad, presupuesto y meta cuantificable. El total A-E puede ajustar automáticamente el veredicto.")
    p(doc, "Excepción metodológica: cuando la pregunta corresponde a un tema transversal configurado (género, no discriminación, discapacidad, diálogo social y tripartismo, o sostenibilidad medioambiental), la app aplica la regla reducida A/B: A significa presencia del tema en objetivo, producto o actividad; B significa presupuesto o recursos asociados. La calificación es Yes si A y B están presentes, Partial si solo A o solo B está presente, y No si no aparece ninguno.")

    h(doc, "5. Uso de Tab 2: Diagnóstico de Atributos Específicos", 1)
    add_numbered(doc, [
        "Subir DOCX y extraer estructura.",
        "Seleccionar secciones del documento que se evaluarán.",
        "Elegir una o más rúbricas activas: participación durante evaluación, integración de género o transición justa moderna.",
        "Seleccionar criterios dentro de cada rúbrica.",
        "Presionar Procesar y Evaluar.",
        "Revisar columnas Criterio, Dimensión, Score, Análisis, Evidencia, Error y Rúbrica.",
        "Descargar resultados_rubricas.zip.",
    ])

    h(doc, "6. Uso de Tab 3: Diagnóstico de Sostenibilidad", 1)
    add_numbered(doc, [
        "Descargar la rúbrica de sostenibilidad si se necesita revisar criterios.",
        "Subir DOCX, extraer estructura y seleccionar secciones.",
        "Seleccionar dimensiones y criterios.",
        "Presionar Procesar y Evaluar.",
        "Revisar resultados por Dimensión, Criterio, Indicador, Score, Análisis y Evidencia.",
        "Usar gráficos de promedio por dimensión y puntaje por criterio para priorizar mejoras.",
        "Descargar resultados_evaluacion_prodoc.zip.",
    ])

    h(doc, "7. Uso de Tab 4: Pregúntale a tus Documentos", 1)
    add_numbered(doc, [
        "Subir uno o más archivos DOCX o TXT.",
        "Confirmar que aparecen como documentos activos.",
        "Escribir preguntas específicas; pedir citas breves y metadatos cuando se necesite trazabilidad.",
        "Para documentos pequeños, la app usa contexto completo; para grandes, usa RAG automáticamente.",
        "Mantener la sesión abierta si se necesita conservar la memoria del chat.",
    ])
    p(doc, "La respuesta debe basarse en los archivos cargados. Si la información no aparece explícitamente, el usuario debe tratar cualquier inferencia como apoyo preliminar y verificarla en el documento.")

    h(doc, "8. Uso de Tabs 5 y 6: Clasificación de Recomendaciones", 1)
    add_numbered(doc, [
        "Elegir archivos predeterminados o subir un XLSX con la columna Recommendation description.",
        "Aplicar filtros previos por ubicación, tiempo, temática, unidad técnica, fuente, progreso u otros campos disponibles.",
        "Revisar conteo de registros y recomendaciones únicas antes de iniciar.",
        "Decidir si activar LLM Verification: mayor precisión, más lento y más costoso.",
        "Presionar Iniciar Clasificación.",
        "Explorar treemaps por dimensión/subdimensión y evolución temporal por categorías.",
        "Usar Herramientas Avanzadas de IA para análisis profundo o resumen ejecutivo sobre el subconjunto filtrado.",
        "Descargar analisis_profundo.xlsx, resumen_ejecutivo.xlsx, deep_analysis.xlsx o executive_summary.xlsx según el idioma de la pestaña.",
    ])

    h(doc, "9. Buenas prácticas operativas", 1)
    add_bullets(doc, [
        "Empezar con secciones o muestras pequeñas cuando se evalúa un documento nuevo.",
        "Verificar evidencias textuales antes de usar resultados en informes formales.",
        "Guardar descargas con nombre de proyecto, fecha y pestaña usada.",
        "No subir documentos que no estén autorizados para procesamiento por servicios externos de IA.",
        "Limpiar resultados o recargar la app al cambiar de caso de análisis.",
    ])

    h(doc, "10. Problemas frecuentes", 1)
    add_table(doc, ["Síntoma", "Causa probable", "Acción"], [
        ["No detecta secciones", "El DOCX no usa estilos Heading de Word.", "Corregir encabezados en Word y volver a extraer."],
        ["OPENAI API key not found", "OPENAI_API_KEY no está disponible.", "Configurar variable de entorno y reiniciar app."],
        ["La clasificación no inicia", "Falta Recommendations_World.xlsx o Frame_Recommendations_English.xlsx.", "Restaurar archivos o subir XLSX válido."],
        ["Resultados con Error", "Respuesta JSON inválida, rate limit o fallo de API.", "Reintentar con menos criterios/secciones o revisar cuota de API."],
        ["Descarga no contiene lo esperado", "Resultados anteriores persistieron en sesión.", "Usar Limpiar resultados y repetir el flujo."],
    ], widths=[2.0, 2.3, 2.5])


def doc6(doc):
    h(doc, "1. Transferencia operativa", 1)
    p(doc, "La operación diaria debe quedar en manos del equipo ILO. El desarrollador original queda como soporte de último recurso para dudas técnicas que no puedan resolverse con esta documentación, revisión del código y pruebas locales.")
    add_table(doc, ["Rol", "Responsabilidad"], [
        ["Equipo ILO propietario", "Administrar acceso, datos, ejecución, resultados y decisiones metodológicas."],
        ["Administrador técnico ILO", "Gestionar despliegue, variables de entorno, archivos de datos y respaldos."],
        ["Usuarios analistas", "Ejecutar flujos, revisar evidencia, descargar resultados y aplicar juicio técnico."],
        ["Desarrollador original", "Soporte excepcional para arquitectura, debugging complejo o cambios que excedan mantenimiento ordinario."],
    ], widths=[2.1, 4.7])

    h(doc, "2. Contacto de último recurso", 1)
    add_bullets(doc, [
        "Canal heredado de la documentación previa: ageidv@gmail.com.",
        "Uso recomendado: solo después de reproducir el problema localmente, revisar logs/mensajes de la app y aislar la pestaña o función afectada.",
        "Incluir al contactar: fecha, pestaña, archivo de entrada no sensible o muestra, pasos para reproducir, mensaje de error y captura de pantalla si la política interna lo permite.",
    ])

    h(doc, "3. Rutina de mantenimiento", 1)
    add_table(doc, ["Frecuencia", "Actividad"], [
        ["Semanal durante operación activa", "Confirmar que la app abre, OPENAI_API_KEY funciona y un documento de prueba procesa en Tab 4."],
        ["Mensual", "Probar una muestra en Tabs 1, 2, 3, 5 y 6; verificar que descargas abren."],
        ["Antes de cada actualización de datos", "Respaldar Excel, CSV, .pt y cachés relevantes."],
        ["Después de cada actualización de código", "Ejecutar pruebas manuales por pestaña y test_two_part_parsing.py si cambió lógica de preguntas."],
        ["Cada 6-12 meses", "Revisar modelos OpenAI, costos, límites, dependencias y compatibilidad de Streamlit."],
    ], widths=[2.2, 4.6])

    h(doc, "4. Recomendaciones para futuras actualizaciones", 1)
    add_bullets(doc, [
        "Separar gradualmente el monolito en módulos: carga de datos, extracción documental, clientes IA, evaluación y UI.",
        "Eliminar bloques comentados obsoletos después de confirmar que no son necesarios.",
        "Consolidar la doble inicialización de load_extended_data y load_embeddings.",
        "Agregar pruebas unitarias para parse_two_part_question, _apply_critic_gate_and_render, clasificación y exportación XLSX.",
        "Agregar pruebas unitarias para detect_transversal_matter y _apply_transversal_gate_and_render si la regla A/B se vuelve crítica para reportes institucionales.",
        "Crear un set pequeño de documentos de prueba no sensibles para validación reproducible.",
        "Documentar cambios de esquema de cada Excel junto con fecha y responsable ILO.",
        "Si se mantienen las variantes core y recommendations, sincronizar cambios desde oli_v6_deploy.py y probar cada variante por separado.",
    ])

    h(doc, "5. Recomendaciones de IA y costos", 1)
    add_bullets(doc, [
        "Mantener gpt-5-mini como modelo principal mientras sus resultados sean consistentes con validación humana.",
        "Usar LLM Verification en clasificación solo para subconjuntos donde la precisión adicional justifique costo y tiempo.",
        "Usar filtros previos y límites de filas antes de análisis profundo de recomendaciones.",
        "Conservar cachés embeddings_cache.pkl y analysis_cache.pkl cuando se procesan lotes recurrentes; eliminarlos solo si se sospecha corrupción o cambio sustantivo de datos.",
        "Revisar si el índice FAISS de recomendaciones cargado por load_embeddings sigue siendo necesario en la app principal, porque la interfaz activa de búsqueda de recomendaciones está comentada y las pestañas de clasificación no lo usan.",
        "Registrar internamente estimaciones de costo por flujo con generate_cost_report.py si se institucionaliza uso frecuente.",
    ])

    h(doc, "6. Criterios de aceptación de cambios", 1)
    add_bullets(doc, [
        "La app inicia sin errores con streamlit run oli_v6_deploy.py.",
        "Tab 1 procesa un DOCX de prueba, genera tabla y descarga ZIP.",
        "Tab 1 reconoce al menos un tema transversal y muestra la línea de auditoría de criterios transversales A/B en el razonamiento.",
        "Tab 2 evalúa al menos un criterio seleccionado y descarga ZIP.",
        "Tab 3 genera resultados y gráficos para una muestra.",
        "Tab 4 responde sobre un TXT pequeño y un DOCX grande.",
        "Tabs 5 y 6 clasifican una muestra limitada y muestran treemaps.",
        "No aparecen campos por completar, credenciales reales ni rutas personales innecesarias en documentación de entrega.",
    ])

    h(doc, "7. Paquete mínimo de handoff", 1)
    add_bullets(doc, [
        "Código principal: oli_v6_deploy.py.",
        "Dependencias: requirements.txt.",
        "Documentación actualizada: los seis archivos DOCX en documentation.",
        "Datos y rúbricas activos listados en la documentación de base de datos.",
        "Instrucciones internas de despliegue del hosting elegido por ILO.",
        "Registro ILO de responsables de acceso, datos y despliegue.",
    ])


DOCS = [
    ("1_Documentacion_Tecnica.docx", "DOCUMENTACIÓN TÉCNICA DEL SISTEMA", "Arquitectura, componentes y despliegue", doc1),
    ("2_Documentacion_Codigo_Fuente.docx", "DOCUMENTACIÓN DEL CÓDIGO FUENTE", "Guía de mantenimiento para la app principal", doc2),
    ("3_Documentacion_Base_Datos.docx", "DOCUMENTACIÓN DE BASE DE DATOS", "Inventario file-based y relaciones lógicas", doc3),
    ("4_Documentacion_Configuracion_Seguridad.docx", "DOCUMENTACIÓN DE CONFIGURACIÓN Y SEGURIDAD", "Variables, secretos, acceso y protección de datos", doc4),
    ("5_Documentacion_Funcional_Operativa.docx", "DOCUMENTACIÓN FUNCIONAL Y OPERATIVA", "Manual operativo para usuarios y administradores", doc5),
    ("6_Otros_Contacto_Recomendaciones.docx", "INFORMACIÓN ADICIONAL DE CIERRE", "Contacto, mantenimiento y recomendaciones futuras", doc6),
]


def main():
    for filename, title, subtitle, builder in DOCS:
        save_doc(filename, title, subtitle, builder)
        print(f"updated {filename}")


if __name__ == "__main__":
    main()
