from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


BASE = Path(__file__).resolve().parent
EN_DIR = BASE / "en"
UPDATED = "11 de mayo de 2026"
UPDATED_EN = "May 11, 2026"
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


def keep_row_together(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    keep_row_together(table.rows[0])
    repeat_table_header(table.rows[0])
    for i, h in enumerate(headers):
        set_cell_text(hdr[i], h, bold=True, color=(255, 255, 255))
        shade_cell(hdr[i], "002F6C")
    for row in rows:
        table_row = table.add_row()
        keep_row_together(table_row)
        cells = table_row.cells
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


def cover(doc, title, subtitle, language="es"):
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
    r = sub.add_run(
        "Project Performance Toolkit"
        if language == "en"
        else "Caja de Herramientas para el Mejor Desempeño de los Proyectos"
    )
    r.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = RGBColor(0, 47, 108)

    source_label = "Technical source" if language == "en" else "Fuente técnica"
    updated_label = "Updated" if language == "en" else "Actualizado"
    updated_value = UPDATED_EN if language == "en" else UPDATED

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(f"{subtitle}\n{source_label}: {APP}\n{updated_label}: {updated_value}")
    doc.add_paragraph()


def save_doc(filename, title, subtitle, build, output_dir=BASE, language="es"):
    doc = Document()
    set_styles(doc)
    cover(doc, title, subtitle, language=language)
    build(doc)
    doc.save(output_dir / filename)


def doc1(doc):
    h(doc, "1. Alcance y fuente de verdad", 1)
    p(doc, "Esta documentación técnica describe exclusivamente la aplicación principal oli_v6_deploy.py. Los archivos oli_v6_deploy_core.py y oli_v6_deploy_recommendations.py son variantes separadas y no se usan como base de esta actualización documental.")
    p(doc, "La aplicación principal es una app Streamlit monolítica, con funciones auxiliares, lógica de extracción documental, evaluación por IA, clasificación de recomendaciones, visualización y exportación en un único archivo Python.")

    h(doc, "2. Arquitectura de alto nivel", 1)
    add_table(doc, ["Capa", "Implementación en el código", "Responsabilidad"], [
        ["Interfaz", "Streamlit con st.tabs, st.session_state, st.cache_data y CSS institucional ILO", "Carga de documentos, selección de criterios, visualizaciones y descargas."],
        ["Procesamiento documental", "python-docx, docx2python, BeautifulSoup y extracción jerárquica de encabezados", "Convierte DOCX/TXT en texto trazable por sección, párrafo, tabla y metadatos."],
        ["IA generativa", "OpenAI SDK mediante OpenAI(api_key=os.getenv('OPENAI_API_KEY'))", "Evaluaciones, síntesis, chat documental, crítica de respuestas y resúmenes ejecutivos."],
        ["Búsqueda vectorial", "text-embedding-3-large, numpy, torch y FAISS IndexFlatIP", "Carga un índice FAISS de recomendaciones al iniciar la app y usa FAISS para RAG en documentos grandes de Tab 4; Tabs 5 y 6 clasifican con similitud coseno en numpy, no con FAISS."],
        ["Datos", "Archivos Excel, CSV pipe-separated y tensores .pt en el directorio de la app o rutas definidas por st.secrets", "No hay base de datos SQL; los archivos son el almacén operativo."],
        ["Exportación", "xlsxwriter, BytesIO y zipfile", "Genera XLSX y ZIP con resultados, evidencias, estructura extraída y reportes."],
    ], widths=[1.3, 2.4, 3.1])

    h(doc, "3. Componentes funcionales activos", 1)
    add_table(doc, ["Pestaña", "Nombre visible", "Función principal"], [
        ["1", "Valoración Preliminar de Calidad de Proyectos", "Evalúa documentos de diseño contra Appraisal Checklist_2025 es-419.xlsx, aplica análisis A-E y regla A/B para temas transversales, genera crítica y síntesis por pregunta, subsección y sección."],
        ["2", "Diagnóstico de Atributos Específicos", "Evalúa secciones seleccionadas con rúbricas de Rubricas_6ago2025.xlsx para participación durante evaluación, género y transición justa moderna."],
        ["3", "Diagnóstico de Sostenibilidad del Proyecto", "Evalúa sostenibilidad con Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx y visualiza puntajes por dimensión y criterio."],
        ["4", "Pregúntale a tus Documentos", "Chat con uno o varios DOCX/TXT, usando contexto completo para documentos pequeños y RAG con FAISS para documentos grandes."],
        ["5", "Clasificación de Recomendaciones", "Clasifica recomendaciones desde una interfaz en español contra Frame_Recommendations_English.xlsx; el campo requerido del archivo de recomendaciones es Recommendation description."],
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
        "Análisis profundo de recomendaciones: la UI muestra un selector entre gpt-4o-mini y gpt-4o, pero el código actual no pasa esa selección a run_row_analysis; analyze_recommendation_plan_pair usa gpt-4o-mini por defecto.",
        "Límite documental operativo: truncate_to_token_limit corta a aproximadamente 110,000 tokens para dejar espacio al prompt y respuesta.",
        "En el chat documental, textos totales de hasta 100,000 caracteres usan contexto completo, luego se recortan a 110,000 tokens; textos mayores generan fragmentos de 2,000 caracteres con solape de 300 y RAG con FAISS.",
    ])

    h(doc, "5.1 Regla de preguntas en dos partes en Tab 1", 2)
    p(doc, "Tab 1 detecta preguntas compuestas mediante parse_two_part_question, con patrones para una declaración seguida de pregunta, separadores explícitos como 'específicamente', dos signos de pregunta, y casos unidos sin espacio. Cuando se detectan dos partes, la Parte 2 es el foco evaluativo y la Parte 1 es solo marco contextual.")
    add_bullets(doc, [
        "El razonamiento del primer análisis debe iniciar con 'Se identificaron 2 partes en esta pregunta.'.",
        "La evidencia se etiqueta como [DEDICATED] cuando aporta elementos específicos al sujeto evaluado y como [FRAMING] cuando es solo contexto, lista de grupos o lenguaje general.",
        "El crítico de Tab 1 vuelve a evaluar con el documento completo y el código aplica una compuerta mecánica sobre el veredicto.",
        "Si el modelo no mantiene el foco en Parte 2, la fila puede quedar con Status = Partial aunque el análisis haya terminado.",
    ])

    h(doc, "5.2 Regla de temas transversales en Tab 1", 2)
    p(doc, "Tab 1 mantiene el marco A-E general para preguntas con sujeto específico, pero lo omite por completo cuando la pregunta corresponde a un tema transversal configurado.")
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
    p(doc, "La calificación automática para estos temas es: Yes cuando A y B están presentes; Partial cuando solo A o solo B está presente; No cuando no están presentes ni A ni B. Indicadores y metas no cuentan para la regla transversal. El crítico puede sobrescribir mecánicamente un veredicto inconsistente con A/B.")

    h(doc, "5.3 Comportamiento preciso de Tab 4", 2)
    add_bullets(doc, [
        "DOCX se extrae con python-docx leyendo párrafos; TXT se lee como texto plano UTF-8 con errors='ignore'. Este flujo no usa el extractor jerárquico enriquecido, por lo que tablas, metadatos de encabezado, páginas y citas por sección no son confiables en Tab 4.",
        "Para documentos grandes, los embeddings de fragmentos se guardan en st.session_state; en cada pregunta se reconstruye un FAISS IndexFlatIP, se normalizan los vectores y se recuperan hasta 15 fragmentos vectoriales.",
        "El contexto RAG agrega además hasta 5 fragmentos por coincidencia exacta de palabras clave, deduplicados contra los fragmentos vectoriales.",
        "La llamada al modelo incluye solo los últimos 5 mensajes del chat.",
        "La detección de cambio de archivos usa la lista de nombres, no hash de contenido; si se sube otro archivo con el mismo nombre puede quedar estado anterior hasta reiniciar sesión o cambiar nombres.",
        "La ruta de documentos pequeños exige responder solo con el documento; la ruta RAG permite inferencia con pistas contextuales, por lo que las conclusiones deben verificarse contra el archivo fuente.",
    ])

    h(doc, "6. Inicialización y caché", 1)
    add_bullets(doc, [
        "load_data() carga df_complete_all_full.xlsx, df_split_actions.xlsx y prepara columnas normalizadas.",
        "load_extended_data() agrega analyzed_recommendations_plans_v5.csv cuando está disponible y guarda analyzed_df en st.session_state.",
        "load_embeddings() carga emb_Recomm_rec_cl_4.pt y Recommendation_RAG_Metadata.pt y construye un índice FAISS en memoria. Aunque la interfaz visible de búsqueda de recomendaciones está comentada, esta carga sigue siendo una dependencia activa porque se ejecuta durante el arranque.",
        "Las funciones de carga usan st.cache_data; limpiar caché de Streamlit fuerza una recarga de archivos.",
        "El bloque principal inicializa datos y embeddings dos veces en el archivo; el comportamiento final es equivalente, pero una futura limpieza puede consolidarlo.",
        "Tabs 5 y 6 usan embeddings_cache.pkl para cachear embeddings de clasificación y analysis_cache.pkl para análisis profundo de recomendaciones.",
        "Tab 4 guarda documentos, fragmentos y embeddings solo en st.session_state durante la sesión.",
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
        ["lessons_embeddings_path", "Ruta de emb_LL_ll_cl_4.pt para una función de lecciones que no está conectada a una pestaña activa."],
        ["structured_lessons_path", "Ruta de lessons_metadata.pt para una función de lecciones que no está conectada a una pestaña activa."],
    ], widths=[2.1, 4.7])

    h(doc, "8. Riesgos técnicos operativos", 1)
    add_bullets(doc, [
        "La app no implementa autenticación, autorización, auditoría, aislamiento por tenant, escaneo de archivos cargados ni redacción automática de secretos; depende del control del entorno de hosting.",
        "Los documentos cargados y sus extractos se envían a OpenAI para análisis; deben tratarse como datos potencialmente sensibles.",
        "El almacén de datos es file-based; reemplazar un archivo de entrada cambia el comportamiento de la app sin migraciones de esquema.",
        "Las rutas locales son relativas al directorio desde el cual se ejecuta Streamlit.",
        "Los archivos de salida XLSX/ZIP pueden contener citas textuales del documento original y deben manejarse con la misma confidencialidad.",
        "Los cachés pickle locales no están cifrados ni verificados por integridad.",
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
        ["emb_Recomm_rec_cl_4.pt", "Embeddings de recomendaciones cargados en arranque para el índice FAISS."],
        ["Recommendation_RAG_Metadata.pt", "Metadatos alineados al embedding de recomendaciones cargado en arranque."],
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
        ["Embeddings y RAG", "get_embedding_with_retry, find_similar_recommendations, find_recommendations_by_term_matching, load_embeddings", "Genera embeddings y carga un índice FAISS en arranque; la búsqueda visible asociada a recomendaciones está comentada, pero la dependencia sigue activa."],
        ["Chat documental", "Tab 4, truncate_to_token_limit, client.embeddings.create, faiss.IndexFlatIP", "Usa contexto completo para textos pequeños y RAG con fragmentos cacheados en sesión para textos grandes."],
        ["Análisis de recomendaciones", "AnalysisCache, analyze_recommendation_plan_pair, run_row_analysis, generate_executive_summary", "Evalúa coherencia de planes, calidad, factibilidad, impacto, innovación y genera resumen ejecutivo; el modelo efectivo del análisis profundo es gpt-4o-mini aunque la UI muestre selector."],
        ["Carga de datos", "prepare_additional_data, load_data, load_extended_data", "Normaliza columnas, años, categorías y fusiona análisis adicional."],
        ["Extracción documental", "extract_docx_structure_enhanced, validate_extraction, extract_document_content", "Extrae jerarquía, tablas, métricas y texto desde DOCX."],
        ["Valoración preliminar", "load_appraisal_questions, analyze_question_with_llm_tab1, detect_transversal_matter, _critic_impl, _apply_critic_gate_and_render, _apply_transversal_gate_and_render, synthesize_subsection_analysis, synthesize_section_analysis, create_results_download_with_sections", "Pregunta por pregunta, aplica crítica A-E o regla transversal A/B y genera XLSX/ZIP multinivel."],
        ["Sostenibilidad y atributos", "evaluate_criterion_with_llm, synthesize_evaluations", "Evalúa criterios con puntuación 1-5, análisis y evidencia."],
        ["Clasificación", "EmbeddingsCache, classify_recommendations, classify_recommendations_en, verify_match_with_llm", "Clasifica recomendaciones contra marco por similitud coseno en numpy, con caché persistente y verificación opcional por LLM."],
    ], widths=[1.6, 2.5, 2.7])

    h(doc, "5. Convenciones de estado", 1)
    add_bullets(doc, [
        "Cada pestaña usa claves propias de st.session_state, por ejemplo tab1_results_df, document_extracted_tab2, tab3_results, doc_chat_docs, doc_chat_embeddings, classified_world_df y deep_analysis_df.",
        "Los resultados se mantienen durante la sesión y pueden limpiarse con botones de limpiar resultados.",
        "La carga de nuevos archivos se detecta con hash(uploaded_file.getvalue()) y reinicia el estado de extracción cuando corresponde.",
        "Excepción: Tab 4 detecta cambios de documentos por lista de nombres, no por hash de contenido.",
        "Los outputs descargables se generan en memoria con BytesIO; no se guardan automáticamente en disco.",
    ])

    h(doc, "5.1 Lógica especial de preguntas en dos partes", 2)
    p(doc, "parse_two_part_question separa preguntas compuestas y privilegia la Parte 2. analyze_question_with_llm_tab1 instruye al modelo a iniciar el razonamiento con 'Se identificaron 2 partes en esta pregunta.' y a usar la Parte 1 solo como contexto. _critic_impl repite la evaluación con el documento completo y _apply_critic_gate_and_render ajusta el veredicto con base en el conteo de elementos dedicados.")
    p(doc, "El sistema distingue evidencia [DEDICATED] de [FRAMING]. La evidencia de framing, como listas amplias de grupos o lenguaje general de inclusión, no puede justificar Partial o Yes por sí sola.")

    h(doc, "5.2 Lógica especial de temas transversales", 2)
    p(doc, "TRANSVERSAL_MATTERS es un diccionario en código que reconoce género, no discriminación, discapacidad, diálogo social y tripartismo, y sostenibilidad medioambiental mediante aliases normalizados sin acentos. Cuando detect_transversal_matter encuentra uno de esos temas en la pregunta, _critic_impl usa CRITIC_SCHEMA_TRANSVERSAL y _apply_transversal_gate_and_render en vez del esquema A-E general.")
    p(doc, "La salida de razonamiento para esos casos inicia con una línea de auditoría de criterios transversales: A corresponde a objetivo/producto/actividad y B a presupuesto. La regla aplicada es Yes=A+B, Partial=A o B, No=sin A ni B. Indicadores y metas no cuentan para la regla transversal.")

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
        "Probar chat con TXT pequeño y DOCX grande para validar los dos caminos: contexto completo y RAG; confirmar que cambiar contenido con el mismo nombre de archivo requiere reiniciar sesión o renombrar el archivo.",
        "Probar Tab 5 con una muestra limitada y LLM Verification apagado y encendido.",
        "Verificar que cada descarga XLSX/ZIP abre correctamente y conserva columnas de evidencia.",
    ])

    h(doc, "8. Deuda técnica visible", 1)
    add_bullets(doc, [
        "Hay bloques comentados extensos de versiones anteriores; no son funcionales pero dificultan mantenimiento.",
        "La inicialización de datos y embeddings aparece dos veces en el bloque principal.",
        "Tab 4 promete trazabilidad por sección/página, pero la extracción de DOCX para chat solo lee párrafos con python-docx; no garantiza tablas, páginas ni metadatos jerárquicos.",
        "Tabs 5 y 6 tienen un selector de modelo para análisis profundo que no controla el modelo efectivo; run_row_analysis usa gpt-4o-mini.",
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
        ["emb_Recomm_rec_cl_4.pt", "load_embeddings", "Matriz de embeddings de recomendaciones; dependencia activa de arranque porque load_embeddings se ejecuta aunque la búsqueda visible esté comentada."],
        ["Recommendation_RAG_Metadata.pt", "load_embeddings", "Metadatos alineados a emb_Recomm_rec_cl_4.pt; dependencia activa de arranque."],
        ["emb_LL_ll_cl_4.pt", "load_lessons_embeddings", "Embeddings para lecciones aprendidas; helper no conectado a una pestaña activa."],
        ["lessons_metadata.pt", "load_lessons_embeddings", "Metadatos esperados para lecciones aprendidas; helper no conectado a una pestaña activa."],
        ["Appraisal Checklist_2025 es-419.xlsx", "load_appraisal_questions", "Hoja rubric; requiere Pregunta_Realizada y usa Tema para renumerar preguntas."],
        ["TRANSVERSAL_MATTERS", "Tab 1", "Diccionario en código, no archivo externo. Reconoce género, no discriminación, discapacidad, diálogo social y tripartismo, y sostenibilidad medioambiental para aplicar regla A/B."],
        ["Rubricas_6ago2025.xlsx", "Tab 2", "La UI activa evalúa rubric_parteval, rubric_gender_ y rubric_TJ_TJ. Otras hojas existen o se cargan en código heredado, pero no son opciones activas."],
        ["Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx", "Tab 3", "Hoja rubric; fuente activa de criterios de sostenibilidad."],
        ["Recommendations_World.xlsx", "Tabs 5 y 6", "Dataset mundial para clasificación; puede reemplazarse por upload XLSX con Recommendation description."],
        ["Frame_Recommendations_English.xlsx", "Tabs 5 y 6", "Marco de referencia; debe contener texto_merged, dimension y subdim."],
        ["embeddings_cache.pkl", "EmbeddingsCache", "Caché local persistente de embeddings generados en clasificación de Tabs 5 y 6; no está cifrado ni verificado."],
        ["analysis_cache.pkl", "AnalysisCache", "Caché local persistente de análisis profundo de recomendaciones; no está cifrado ni verificado."],
    ], widths=[2.5, 1.7, 2.8])

    h(doc, "3. Relaciones lógicas", 1)
    add_bullets(doc, [
        "df_complete_all_full.xlsx y df_split_actions.xlsx se alinean por index_df.",
        "analyzed_recommendations_plans_v5.csv se fusiona con df por index_df cuando hay columnas nuevas.",
        "emb_Recomm_rec_cl_4.pt debe mantener el mismo orden que Recommendation_RAG_Metadata.pt para el índice de recomendaciones cargado en memoria.",
        "Recommendations_World.xlsx se clasifica contra cada fila de Frame_Recommendations_English.xlsx usando texto_merged como definición semántica; este flujo usa similitud coseno con numpy, no FAISS.",
        "Tabs 5 y 6 deduplican embeddings por texto de recomendación, pero el resultado vuelve al nivel de fila para conservar atributos repetidos.",
        "Los datasets de recomendaciones pueden tener filas duplicadas por múltiples atributos; Tabs 5 y 6 muestran conteos de registros y recomendaciones únicas.",
    ])

    h(doc, "4. Transformaciones principales", 1)
    add_table(doc, ["Transformación", "Detalle"], [
        ["Normalización de columnas", "Los espacios y puntos se reemplazan por guion bajo en load_data."],
        ["Años", "Recommendation_date y Recommendation date se convierten con pandas; años previos a 2018 en analyzed_df se fijan a 2018."],
        ["Categorías", "prepare_additional_data estandariza campos categóricos, listas/tags y clasificaciones cuando existen."],
        ["FAISS", "El índice de recomendaciones se construye al inicio. Tab 4 reconstruye un IndexFlatIP por pregunta para RAG de documentos grandes, con fragmentos ya embebidos y guardados en sesión."],
        ["Temas transversales", "En Tab 1, las preguntas detectadas por TRANSVERSAL_MATTERS omiten A-E y se califican con A=objetivo/producto/actividad y B=presupuesto; Yes requiere A+B, Partial requiere A o B, No requiere ausencia de ambos."],
        ["Clasificación", "Cada recomendación única se embebe una vez, se normaliza y se compara por similitud coseno contra el marco usando numpy. La asignación base usa top 3; la verificación opcional por LLM reranquea top 10; dimensiones/subdimensiones secundarias se conservan si similitud >= 0.60."],
        ["Exportación", "Los resultados se escriben a XLSX con columnas ordenadas y strings_to_urls desactivado."],
    ], widths=[2.0, 4.8])

    h(doc, "5. Campos mínimos por flujo", 1)
    add_table(doc, ["Flujo", "Campos mínimos requeridos"], [
        ["Valoración preliminar", "Appraisal Checklist: Pregunta_Realizada; se recomienda Tema. Las preguntas de temas transversales se reconocen por aliases en código."],
        ["Atributos específicos", "Rubricas_6ago2025: hojas activas rubric_parteval, rubric_gender_ y rubric_TJ_TJ, con Indicador, Dimensión y niveles de desempeño."],
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
        "No asumir que emb_LL_ll_cl_4.pt o lessons_metadata.pt están en uso operativo si la función de lecciones no se reconecta a una pestaña activa.",
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
        ["lessons_embeddings_path", "Producción para helper no activo", "Ruta accesible a emb_LL_ll_cl_4.pt si se reconecta la función de lecciones."],
        ["structured_lessons_path", "Producción para helper no activo", "Ruta accesible a lessons_metadata.pt si se reconecta la función de lecciones."],
    ], widths=[2.0, 1.8, 3.0])

    h(doc, "3. Control de acceso", 1)
    p(doc, "oli_v6_deploy.py no implementa usuarios, roles, contraseñas, permisos por funcionalidad, auditoría, aislamiento por tenant, escaneo de archivos cargados ni redacción de secretos. El control debe aplicarse fuera de la app, en Streamlit Cloud, servidor institucional, proxy, red privada o mecanismo equivalente.")
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
        "Los cachés pickle locales, incluidas embeddings_cache.pkl y analysis_cache.pkl, no están cifrados ni verificados por integridad.",
        "La aplicación no cifra salidas locales ni implementa retención de datos; estas políticas deben definirse en el entorno institucional.",
    ])

    h(doc, "4.1 Capas de caché activas", 2)
    add_table(doc, ["Capa", "Ubicación", "Riesgo operativo"], [
        ["st.cache_data", "Memoria/proceso Streamlit", "Puede mantener archivos, rúbricas o datos anteriores hasta limpiar caché o reiniciar."],
        ["Tab 4", "st.session_state", "Guarda documentos, fragmentos y embeddings por sesión; cambios con el mismo nombre de archivo pueden no detectarse."],
        ["Tabs 5 y 6", "embeddings_cache.pkl", "Persistente local, sin cifrado ni control de integridad."],
        ["Análisis profundo", "analysis_cache.pkl", "Persistente local, sin cifrado ni control de integridad."],
        ["SimpleHierarchicalStore", "~/document_store/embedding_cache.pkl", "Helper heredado para evaluación jerárquica; no es el mecanismo principal de Tab 4."],
    ], widths=[1.7, 2.3, 2.8])

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
        ["Clasificación de Recomendaciones", "Clasificar recomendaciones desde interfaz en español usando la columna Recommendation description.", "Treemaps, evolución temporal, análisis profundo XLSX y resumen ejecutivo XLSX."],
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
    p(doc, "Interpretación: las respuestas son Yes, No, Partial o Not Found. Para preguntas con sujeto específico no transversal, la app aplica el marco A-E: sub-objetivo/output, indicador, actividad, presupuesto y meta cuantificable. El total A-E puede ajustar automáticamente el veredicto.")
    p(doc, "Preguntas en dos partes: la Parte 2 es la pregunta evaluada y la Parte 1 solo enmarca. El razonamiento puede mostrar 'Se identificaron 2 partes en esta pregunta.' y la evidencia puede venir marcada como [DEDICATED] o [FRAMING]. Solo la evidencia dedicada puede sostener un Partial o Yes.")
    p(doc, "Excepción metodológica: cuando la pregunta corresponde a un tema transversal configurado (género, no discriminación, discapacidad, diálogo social y tripartismo, o sostenibilidad medioambiental), la app no usa A-E. Aplica la regla reducida A/B: A significa presencia del tema en objetivo, producto o actividad; B significa presupuesto o recursos asociados. Indicadores y metas no cuentan. La calificación es Yes si A y B están presentes, Partial si solo A o solo B está presente, y No si no aparece ninguno.")
    p(doc, "Distinguir género en Tab 1 y Tab 2: en Tab 1 género es un tema transversal con regla A/B; en Tab 2 'Integración del Enfoque de Género' es una rúbrica Excel de Rubricas_6ago2025.xlsx, hoja rubric_gender_, evaluada con puntaje 1-5 por criterios seleccionados.")
    p(doc, "La descarga principal de Tab 1 es appraisal_checklist_results.zip e incluye appraisal_checklist_results.xlsx, appraisal_checklist_rubric_template.xlsx cuando el archivo fuente está disponible, y appraisal_checklist_summary.txt. La estructura extraída se descarga aparte como estructura_documento_tab1_<archivo>.xlsx.")

    h(doc, "5. Uso de Tab 2: Diagnóstico de Atributos Específicos", 1)
    add_numbered(doc, [
        "Subir DOCX y extraer estructura.",
        "Seleccionar secciones del documento que se evaluarán.",
        "Elegir una o más rúbricas activas: participación durante evaluación/metodología, integración del enfoque de género o transición justa enfoque moderno.",
        "Seleccionar criterios dentro de cada rúbrica.",
        "Presionar Procesar y Evaluar.",
        "Revisar columnas Criterio, Dimensión, Score, Análisis, Evidencia, Error y Rúbrica.",
        "Descargar resultados_rubricas.zip.",
    ])
    p(doc, "La estructura extraída se puede descargar como estructura_documento_tab2_<archivo>.xlsx antes de evaluar.")

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
    p(doc, "La estructura extraída se puede descargar como estructura_documento_tab3_<archivo>.xlsx antes de evaluar.")

    h(doc, "7. Uso de Tab 4: Pregúntale a tus Documentos", 1)
    add_numbered(doc, [
        "Subir uno o más archivos DOCX o TXT.",
        "Confirmar que aparecen como documentos activos.",
        "Escribir preguntas específicas; pedir citas breves cuando se necesite trazabilidad, pero verificar manualmente sección/página porque Tab 4 no usa el extractor jerárquico enriquecido.",
        "Para documentos pequeños (hasta 100,000 caracteres totales), la app usa contexto completo recortado a 110,000 tokens; para grandes, usa RAG automáticamente.",
        "Mantener la sesión abierta si se necesita conservar la memoria del chat.",
    ])
    p(doc, "En RAG, la app usa fragmentos de 2,000 caracteres con solape de 300, recupera hasta 15 fragmentos por FAISS y hasta 5 por coincidencia de palabras clave, y envía solo los últimos 5 mensajes del chat. Si se reemplaza un archivo por otro con el mismo nombre, reiniciar sesión o renombrarlo para evitar estado anterior.")
    p(doc, "La respuesta debe basarse en los archivos cargados. La ruta RAG permite inferencias desde pistas contextuales; el usuario debe tratarlas como apoyo preliminar y verificarlas en el documento.")

    h(doc, "8. Uso de Tabs 5 y 6: Clasificación de Recomendaciones", 1)
    add_numbered(doc, [
        "Elegir archivos predeterminados o subir un XLSX con la columna Recommendation description.",
        "Aplicar filtros previos por ubicación, tiempo, temática, unidad técnica, fuente, progreso u otros campos disponibles.",
        "Revisar conteo de registros y recomendaciones únicas antes de iniciar.",
        "Decidir si activar LLM Verification: mayor precisión, más lento y más costoso.",
        "Presionar Iniciar Clasificación.",
        "Explorar treemaps por dimensión/subdimensión y evolución temporal por categorías. La asignación base usa top 3 por similitud coseno; LLM Verification reranquea top 10; dimensiones secundarias se conservan si similitud >= 0.60.",
        "Usar Herramientas Avanzadas de IA para análisis profundo o resumen ejecutivo sobre el subconjunto filtrado. Nota: el selector de modelo de análisis profundo no controla el modelo efectivo en el código actual; se usa gpt-4o-mini.",
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
        ["Tab 4 responde sobre archivo anterior", "Se subió otro archivo con el mismo nombre y el estado se detecta por nombre.", "Renombrar el archivo o reiniciar la sesión de Streamlit."],
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
        "Revisar si load_embeddings debe seguir ejecutándose en arranque si la búsqueda visible de recomendaciones permanece comentada.",
        "Corregir Tab 4 si ILO necesita citas por sección/página: debe usar extractor jerárquico, no solo python-docx paragraphs.",
        "Conectar el selector de modelo de análisis profundo a run_row_analysis o retirarlo de la UI.",
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
        "Revisar si el índice FAISS de recomendaciones cargado por load_embeddings sigue siendo necesario en la app principal, porque la interfaz activa de búsqueda de recomendaciones está comentada y las pestañas de clasificación no lo usan aunque la carga se ejecuta en arranque.",
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
        "Tab 4 se prueba reemplazando un archivo por otro con el mismo nombre para confirmar el procedimiento operativo de reinicio o renombrado.",
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


def doc_en1(doc):
    h(doc, "1. Scope and source of truth", 1)
    p(doc, "This technical documentation describes only the main Streamlit application, oli_v6_deploy.py. The files oli_v6_deploy_core.py and oli_v6_deploy_recommendations.py are split variants and are not used as factual sources for this documentation.")
    p(doc, "The main app is a monolithic Streamlit file that includes document extraction, LLM evaluation, recommendation classification, visualization, caching, and export logic.")

    h(doc, "2. High-level architecture", 1)
    add_table(doc, ["Layer", "Implementation", "Responsibility"], [
        ["Interface", "Streamlit with st.tabs, st.session_state, st.cache_data, and ILO styling", "Document uploads, rubric selection, charts, result tables, and downloads."],
        ["Document processing", "python-docx, docx2python, BeautifulSoup, and enhanced heading extraction", "Turns DOCX/TXT content into text and structured extraction tables for Tabs 1-3."],
        ["Generative AI", "OpenAI SDK with OpenAI(api_key=os.getenv('OPENAI_API_KEY'))", "Evaluations, critique passes, synthesis, document chat, deep analysis, and executive summaries."],
        ["Vector search", "text-embedding-3-large, numpy, torch, and FAISS IndexFlatIP", "Loads a recommendation FAISS index at startup and uses FAISS for large-document RAG in Tab 4. Tabs 5 and 6 use numpy cosine similarity, not FAISS."],
        ["Data", "Excel files, pipe-separated CSV, PyTorch tensors, and local pickle caches", "File-based operational store; there is no SQL database."],
        ["Exports", "xlsxwriter, BytesIO, and zipfile", "Creates XLSX and ZIP outputs with answers, evidence, extracted structure, and reports."],
    ], widths=[1.3, 2.4, 3.1])

    h(doc, "3. Active functional components", 1)
    add_table(doc, ["Tab", "Visible name", "Main function"], [
        ["1", "Valoración Preliminar de Calidad de Proyectos", "Evaluates project design documents against Appraisal Checklist_2025 es-419.xlsx, including A-E critique, two-part question handling, and transversal A/B scoring."],
        ["2", "Diagnóstico de Atributos Específicos", "Evaluates selected document sections using active rubrics for evaluation participation/methodology, gender integration, and modern just transition."],
        ["3", "Diagnóstico de Sostenibilidad del Proyecto", "Evaluates sustainability with Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx and charts scores by dimension and criterion."],
        ["4", "Pregúntale a tus Documentos", "Chat over one or more DOCX/TXT files, using full context for small uploads and FAISS RAG for large uploads."],
        ["5", "Clasificación de Recomendaciones", "Spanish-language UI for classifying records with a required Recommendation description column against Frame_Recommendations_English.xlsx."],
        ["6", "Recommendation Classification", "English-language recommendation classification workflow using the same underlying logic as Tab 5."],
    ], widths=[0.7, 2.2, 4.1])

    h(doc, "4. Dependencies", 1)
    p(doc, "requirements.txt declares the operating dependency set, including Streamlit, pandas, numpy, faiss-cpu, OpenAI, torch, plotly, xlsxwriter, python-docx, docx2python, openpyxl, and tiktoken.")

    h(doc, "5. AI models and context limits", 1)
    add_bullets(doc, [
        "Main evaluation, critique, synthesis, and chat model: gpt-5-mini.",
        "Embedding model: text-embedding-3-large.",
        "Deep recommendation analysis shows a UI selector for gpt-4o-mini and gpt-4o, but the current code does not pass that selection into run_row_analysis; analyze_recommendation_plan_pair effectively uses gpt-4o-mini.",
        "truncate_to_token_limit keeps document context to about 110,000 tokens.",
        "Tab 4 uses full context for total uploaded text up to 100,000 characters; larger uploads are split into 2,000-character chunks with 300-character overlap and processed with FAISS RAG.",
    ])

    h(doc, "5.1 Tab 1 two-part questions", 2)
    p(doc, "parse_two_part_question detects compound questions through regex patterns such as statement-then-question, explicit separators, two question marks, and run-together clauses. When two parts are detected, Part 2 drives the answer and Part 1 is framing only.")
    add_bullets(doc, [
        "The first-pass reasoning is instructed to begin with 'Se identificaron 2 partes en esta pregunta.'.",
        "Evidence may be marked [DEDICATED] or [FRAMING]. Framing evidence alone cannot justify Partial or Yes.",
        "The critic re-checks the answer against the full document and the code layer mechanically applies the A-E gate.",
        "If the model fails to focus on Part 2, the row can receive Status = Partial even when the call completes.",
    ])

    h(doc, "5.2 Tab 1 transversal matters", 2)
    p(doc, "TRANSVERSAL_MATTERS recognizes five configured themes: gender, non-discrimination, disability, social dialogue and tripartism, and environmental sustainability. These questions bypass the A-E framework entirely.")
    add_table(doc, ["Transversal criterion", "Operational definition"], [
        ["A", "Operational presence of the theme in an objective, product/output, or activity."],
        ["B", "Budget, resources, or a budget line for a corresponding activity."],
    ], widths=[1.5, 5.3])
    p(doc, "The verdict is mechanical: Yes = A and B; Partial = A or B; No = neither A nor B. Indicators and targets do not count for transversal scoring. Gender in Tab 1 is this A/B transversal rule; gender in Tab 2 is a separate 1-5 Excel rubric from Rubricas_6ago2025.xlsx, sheet rubric_gender_.")

    h(doc, "5.3 Tab 4 document chat", 2)
    add_bullets(doc, [
        "DOCX chat extraction reads paragraph text with python-docx; it does not use the enhanced hierarchical extractor. Tables, page numbers, section metadata, and precise section/page citations are not reliable in Tab 4.",
        "For large uploads, chunk embeddings are stored in st.session_state. Each question rebuilds a FAISS IndexFlatIP, normalizes vectors, retrieves up to 15 vector chunks, then adds up to 5 keyword chunks.",
        "Only the last 5 chat messages are sent to the model.",
        "File changes are detected by filename list, not content hash. Replacing a file with another file of the same name can leave stale session state until the app session is reset or the file is renamed.",
        "The small-document path is stricter about using only the document. The RAG path allows inference from contextual clues, so any inferred answer must be manually verified.",
    ])

    h(doc, "6. Startup and cache", 1)
    add_bullets(doc, [
        "load_data() reads df_complete_all_full.xlsx and df_split_actions.xlsx.",
        "load_extended_data() optionally merges analyzed_recommendations_plans_v5.csv and stores analyzed_df in st.session_state.",
        "load_embeddings() reads emb_Recomm_rec_cl_4.pt and Recommendation_RAG_Metadata.pt and builds a FAISS index during startup. This remains an active runtime dependency even though the visible recommendation-search UI is commented.",
        "Tabs 5 and 6 use embeddings_cache.pkl for classification embeddings and analysis_cache.pkl for deep recommendation-plan analysis.",
        "Tab 4 stores uploaded document text, chunks, and chunk embeddings in st.session_state only.",
        "The main file initializes data and embeddings twice; the behavior is equivalent but should be consolidated in maintenance.",
    ])

    h(doc, "7. Deployment", 1)
    p(doc, "Run locally with:")
    codeblock(doc, "streamlit run oli_v6_deploy.py")
    p(doc, "OPENAI_API_KEY is read from the process environment. Production file paths are read from st.secrets only when STREAMLIT_ENV=production.")
    add_table(doc, ["Secret or variable", "Use"], [
        ["OPENAI_API_KEY", "OpenAI client authentication."],
        ["df_path", "Production path to df_complete_all_full.xlsx."],
        ["df_raw_path", "Production path to df_split_actions.xlsx."],
        ["embeddings_path", "Production path to emb_Recomm_rec_cl_4.pt."],
        ["structured_embeddings_path", "Production path to Recommendation_RAG_Metadata.pt."],
        ["analyzed_recommendations_path", "Optional production path to analyzed_recommendations_plans_v5.csv."],
        ["lessons_embeddings_path", "Path for an inactive lessons helper if that feature is reconnected."],
        ["structured_lessons_path", "Path for an inactive lessons helper if that feature is reconnected."],
    ], widths=[2.2, 4.6])

    h(doc, "8. Operational risks", 1)
    add_bullets(doc, [
        "The app has no built-in authentication, authorization, audit log, tenant isolation, upload scanning, or secret redaction.",
        "Uploaded documents, extracted text, evidence, questions, and generated outputs may be sent to OpenAI.",
        "The data store is file-based; replacing an input file changes app behavior without migrations or schema controls.",
        "Generated XLSX/ZIP files can contain verbatim source-document evidence and must be handled as sensitive outputs.",
        "Local pickle caches are not encrypted and are not integrity-checked.",
    ])


def doc_en2(doc):
    h(doc, "1. Source-code scope", 1)
    p(doc, "The documented source file is oli_v6_deploy.py. The split variants exist in the repository but are not used as factual sources for this documentation.")
    p(doc, "Maintenance rule: when the main app changes, separately decide whether the change must be propagated to the core and recommendations variants before delivery.")

    h(doc, "2. Relevant project files", 1)
    add_table(doc, ["File", "Operational role"], [
        ["oli_v6_deploy.py", "Main Streamlit app and source of truth for these documents."],
        ["requirements.txt", "Python dependencies."],
        ["df_complete_all_full.xlsx", "Main enriched recommendation dataset."],
        ["df_split_actions.xlsx", "Raw/split recommendation dataset used for years and missing records."],
        ["analyzed_recommendations_plans_v5.csv", "Pipe-separated recommendation/action-plan analysis add-on."],
        ["emb_Recomm_rec_cl_4.pt", "Recommendation embeddings loaded at startup for FAISS."],
        ["Recommendation_RAG_Metadata.pt", "Metadata aligned to recommendation embeddings."],
        ["Appraisal Checklist_2025 es-419.xlsx", "Tab 1 appraisal questions, sheet rubric."],
        ["Rubricas_6ago2025.xlsx", "Tab 2 active sheets: rubric_parteval, rubric_gender_, and rubric_TJ_TJ."],
        ["Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx", "Active Tab 3 sustainability rubric."],
        ["Recommendations_World.xlsx", "Default recommendation records for Tabs 5 and 6."],
        ["Frame_Recommendations_English.xlsx", "Reference frame with texto_merged, dimension, and subdim."],
    ], widths=[2.8, 4.0])

    h(doc, "3. Local installation", 1)
    add_numbered(doc, [
        "Use a Python version compatible with the pinned dependencies.",
        "Create and activate a virtual environment.",
        "Install dependencies with pip install -r requirements.txt.",
        "Set OPENAI_API_KEY in the runtime environment.",
        "Run streamlit run oli_v6_deploy.py from the project directory.",
    ])
    codeblock(doc, [
        "python -m venv .venv",
        "source .venv/bin/activate",
        "pip install -r requirements.txt",
        "streamlit run oli_v6_deploy.py",
    ])

    h(doc, "4. Main function map", 1)
    add_table(doc, ["Block", "Functions or classes", "Description"], [
        ["Excel/export", "to_excel, create_results_download_with_sections", "Writes XLSX/ZIP outputs in memory and disables automatic URL conversion."],
        ["Tokens", "truncate_to_token_limit", "Cuts prompt context to token limits with tiktoken."],
        ["Embeddings and RAG", "get_embedding_with_retry, load_embeddings, faiss.IndexFlatIP", "Loads a startup FAISS index and supports large-document RAG in Tab 4."],
        ["Document extraction", "extract_docx_structure_enhanced, validate_extraction, extract_document_content", "Extracts hierarchy, tables, and text for Tabs 1-3. Tab 4 chat does not use the enhanced extractor."],
        ["Tab 1 appraisal", "load_appraisal_questions, parse_two_part_question, analyze_question_with_llm_tab1, _critic_impl", "Evaluates each question, handles two-part focus, A-E gates, and transversal A/B gates."],
        ["Tab 2/3 rubric evaluation", "evaluate_criterion_with_llm, synthesize_evaluations", "Scores selected criteria 1-5 with evidence."],
        ["Recommendation classification", "EmbeddingsCache, classify_recommendations, classify_recommendations_en, verify_match_with_llm", "Classifies records against the frame with numpy cosine similarity and optional LLM verification."],
        ["Deep recommendation analysis", "AnalysisCache, analyze_recommendation_plan_pair, run_row_analysis", "Uses local pickle cache; effective model is gpt-4o-mini in current code."],
    ], widths=[1.6, 2.6, 2.6])

    h(doc, "5. State conventions", 1)
    add_bullets(doc, [
        "Tabs use separate st.session_state keys such as tab1_results_df, document_extracted_tab2, tab3_results, doc_chat_docs, doc_chat_embeddings, classified_world_df, and deep_analysis_df.",
        "Tabs 1-3 detect uploaded-file changes with file-content hashes and reset extraction state.",
        "Tab 4 detects document changes only by filename list.",
        "Downloadable outputs are generated in memory and are not automatically saved to disk.",
    ])

    h(doc, "6. Error handling and verification", 1)
    add_bullets(doc, [
        "Missing critical files produce Streamlit errors or warnings and stop the affected flow.",
        "Missing OPENAI_API_KEY produces a global warning; OpenAI-dependent functions then fail or return errors.",
        "Invalid JSON responses are cleaned where possible; if parsing fails, flows return default error records.",
        "Run test_two_part_parsing.py when changing two-part parsing or Tab 1 question logic.",
        "Manually test one question per configured transversal theme after changing aliases or scoring.",
        "Test Tab 4 with both a small TXT and a large DOCX; also test replacing a file with the same filename.",
        "Test Tabs 5 and 6 with a limited sample, with LLM Verification off and on.",
    ])

    h(doc, "7. Visible technical debt", 1)
    add_bullets(doc, [
        "Large commented legacy blocks remain in the main file.",
        "Startup data and embeddings are initialized twice.",
        "load_embeddings remains active at startup although the visible recommendation-search UI is commented.",
        "Tab 4 asks users for citations and metadata, but its DOCX chat extraction only reads paragraphs.",
        "Tabs 5 and 6 expose a deep-analysis model selector that does not control the effective model.",
        "Some Tab 3 error messages still mention PRODOC_rubric.xlsx even though the active rubric is Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx.",
        "The app lacks a clean separation between UI, domain logic, extraction, and external-service clients.",
    ])


def doc_en3(doc):
    h(doc, "1. Data model", 1)
    p(doc, "The application does not use a relational database. Its data model is file-based: Excel workbooks, pipe-separated CSV, PyTorch tensors, local pickle caches, and in-memory Streamlit session state.")

    h(doc, "2. Data-file inventory", 1)
    add_table(doc, ["File or object", "Loaded by", "Use"], [
        ["df_complete_all_full.xlsx", "load_data", "Main recommendation base; creates index_df from ID_Recomendacion and normalizes dimension/subdim fields."],
        ["df_split_actions.xlsx", "load_data", "Raw base with Recommendation date; contributes year and missing index_df records."],
        ["analyzed_recommendations_plans_v5.csv", "load_extended_data", "Pipe-separated add-on with tags, scores, and analysis metrics."],
        ["emb_Recomm_rec_cl_4.pt", "load_embeddings", "Recommendation embedding matrix; active startup dependency."],
        ["Recommendation_RAG_Metadata.pt", "load_embeddings", "Metadata aligned to recommendation embeddings; active startup dependency."],
        ["emb_LL_ll_cl_4.pt", "load_lessons_embeddings", "Lessons embeddings for an inactive helper, not an active UI tab."],
        ["lessons_metadata.pt", "load_lessons_embeddings", "Lessons metadata for an inactive helper, not an active UI tab."],
        ["Appraisal Checklist_2025 es-419.xlsx", "load_appraisal_questions", "Tab 1 question source; requires Pregunta_Realizada and uses Tema for renumbering."],
        ["TRANSVERSAL_MATTERS", "Tab 1", "In-code dictionary for gender, non-discrimination, disability, social dialogue/tripartism, and environmental sustainability aliases."],
        ["Rubricas_6ago2025.xlsx", "Tab 2", "Active sheets are rubric_parteval, rubric_gender_, and rubric_TJ_TJ."],
        ["Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx", "Tab 3", "Active sustainability rubric, sheet rubric."],
        ["Recommendations_World.xlsx", "Tabs 5 and 6", "Default classification dataset; can be replaced with an uploaded XLSX containing Recommendation description."],
        ["Frame_Recommendations_English.xlsx", "Tabs 5 and 6", "Reference frame; requires texto_merged, dimension, and subdim."],
        ["embeddings_cache.pkl", "EmbeddingsCache", "Persistent local cache for classification embeddings; not encrypted or integrity-checked."],
        ["analysis_cache.pkl", "AnalysisCache", "Persistent local cache for deep analysis; not encrypted or integrity-checked."],
    ], widths=[2.6, 1.7, 2.7])

    h(doc, "3. Logical relationships", 1)
    add_bullets(doc, [
        "df_complete_all_full.xlsx and df_split_actions.xlsx align by index_df.",
        "analyzed_recommendations_plans_v5.csv is merged into the main dataframe by index_df when new columns exist.",
        "emb_Recomm_rec_cl_4.pt must stay aligned with Recommendation_RAG_Metadata.pt.",
        "Recommendations_World.xlsx records are compared against Frame_Recommendations_English.xlsx rows using texto_merged as semantic reference text.",
        "Recommendation datasets may contain repeated rows because a recommendation can have multiple attributes; Tabs 5 and 6 show both row counts and unique recommendation counts.",
    ])

    h(doc, "4. Main transformations", 1)
    add_table(doc, ["Transformation", "Details"], [
        ["Column normalization", "load_data replaces spaces and dots with underscores."],
        ["Years", "Recommendation_date and Recommendation date are parsed with pandas; years before 2018 in analyzed_df are forced to 2018."],
        ["FAISS", "A recommendation FAISS index is built at startup. Tab 4 rebuilds an IndexFlatIP per large-document query using session-cached chunk embeddings."],
        ["Transversal scoring", "Configured themes bypass A-E and use A=objective/product/activity and B=budget/resources. Yes=A+B, Partial=A or B, No=neither."],
        ["Classification", "Embeddings are normalized and compared with numpy cosine similarity. Base assignment uses top 3; optional LLM verification reranks top 10; secondary dimensions/subdimensions require similarity >=0.60."],
        ["Exports", "Results are written to XLSX and ZIP files with stable column ordering."],
    ], widths=[2.0, 4.8])

    h(doc, "5. Minimum fields by workflow", 1)
    add_table(doc, ["Workflow", "Minimum required fields"], [
        ["Tab 1 appraisal", "Appraisal Checklist: Pregunta_Realizada; Tema recommended. Transversal questions are recognized by in-code aliases."],
        ["Tab 2 attributes", "Rubricas_6ago2025 active sheets with Indicador, Dimensión, and performance-level columns."],
        ["Tab 3 sustainability", "Active rubric fields for dimension, criterion, indicator, and levels from sheet rubric."],
        ["Tabs 5 and 6 classification", "Recommendation file: Recommendation description. Frame: texto_merged, dimension, subdim."],
        ["Deep analysis", "Recommendation description and Action plan; Comments improves analysis when present."],
        ["Executive summary", "Recommendation description; Management response, Comments, and Action plan are used when available."],
    ], widths=[2.0, 4.8])

    h(doc, "6. Backups and data refresh", 1)
    add_bullets(doc, [
        "Keep a dated copy of each Excel, CSV, tensor, and cache before replacing it.",
        "After replacing data files, restart the app or clear Streamlit cache.",
        "If embeddings are regenerated, regenerate aligned metadata in the same run.",
        "Validate Tabs 5 and 6 with a small sample when either Recommendations_World.xlsx or Frame_Recommendations_English.xlsx changes schema.",
        "Do not manually edit embeddings_cache.pkl or analysis_cache.pkl; delete them with the app stopped if corruption is suspected.",
    ])


def doc_en4(doc):
    h(doc, "1. Required configuration", 1)
    add_table(doc, ["Item", "Expected value"], [
        ["Local command", "streamlit run oli_v6_deploy.py"],
        ["AI variable", "OPENAI_API_KEY available in the process environment."],
        ["Production mode", "STREAMLIT_ENV=production when file paths should come from st.secrets."],
        ["Working directory", "Directory containing required Excel, CSV, and .pt files unless production secrets provide paths."],
        ["Dependencies", "Installed from requirements.txt."],
    ], widths=[2.0, 4.8])

    h(doc, "2. Secrets and paths", 1)
    p(doc, "OPENAI_API_KEY is read through os.getenv. Data paths are read from st.secrets only when STREAMLIT_ENV=production.")
    add_table(doc, ["Name", "When used", "Expected content"], [
        ["df_path", "Production", "Path to df_complete_all_full.xlsx."],
        ["df_raw_path", "Production", "Path to df_split_actions.xlsx."],
        ["embeddings_path", "Production", "Path to emb_Recomm_rec_cl_4.pt."],
        ["structured_embeddings_path", "Production", "Path to Recommendation_RAG_Metadata.pt."],
        ["analyzed_recommendations_path", "Optional production", "Path to analyzed_recommendations_plans_v5.csv."],
        ["lessons_embeddings_path", "Inactive helper", "Path to emb_LL_ll_cl_4.pt if lessons are reconnected."],
        ["structured_lessons_path", "Inactive helper", "Path to lessons_metadata.pt if lessons are reconnected."],
    ], widths=[2.0, 1.8, 3.0])

    h(doc, "3. Access control", 1)
    p(doc, "The app does not implement users, roles, passwords, per-feature permissions, audit logs, tenant isolation, upload scanning, or secret redaction. Those controls must be provided by the hosting environment, institutional network, proxy, or equivalent access layer.")
    add_bullets(doc, [
        "Only authorized ILO users should be able to open the app.",
        "Repository write access and operational data write access should be restricted to technical maintainers.",
        "OPENAI_API_KEY must be visible only to the runtime and authorized administrators.",
        "Users are responsible for protecting any downloads they generate.",
    ])

    h(doc, "4. Data security", 1)
    add_bullets(doc, [
        "Uploaded documents are processed in memory and, in extraction flows, through temporary files that are removed when cleanup completes.",
        "Document text, questions, evidence, and prompts can be sent to OpenAI.",
        "st.session_state holds extracted text and results during the active session.",
        "Generated XLSX/ZIP outputs can contain verbatim source text, reasoning, and evidence.",
        "Local pickle caches are not encrypted and are not integrity-checked.",
        "The app does not implement output encryption or retention policy; ILO must define those policies in the runtime environment.",
    ])

    h(doc, "4.1 Active cache layers", 2)
    add_table(doc, ["Layer", "Location", "Operational risk"], [
        ["st.cache_data", "Streamlit process memory/cache", "May keep older data/rubric loads until cache clear or restart."],
        ["Tab 4", "st.session_state", "Stores documents, chunks, and embeddings for the session; same-name file replacement can leave stale state."],
        ["Tabs 5 and 6", "embeddings_cache.pkl", "Persistent local cache, unencrypted and unchecked."],
        ["Deep analysis", "analysis_cache.pkl", "Persistent local cache, unencrypted and unchecked."],
        ["SimpleHierarchicalStore", "~/document_store/embedding_cache.pkl", "Legacy helper cache; not the main Tab 4 mechanism."],
    ], widths=[1.7, 2.3, 2.8])

    h(doc, "5. Safe operation parameters", 1)
    add_table(doc, ["Parameter", "Code value", "Impact"], [
        ["MAX_WORKERS global for Tab 1", "48", "High concurrency; requires sufficient API limits."],
        ["MAX_WORKERS Tab 2", "3", "Reduces rate-limit risk for rubric evaluation."],
        ["MAX_WORKERS Tab 3", "8", "Parallel sustainability criteria processing."],
        ["Tab 4 RAG threshold", ">100,000 characters", "Avoids sending very large full documents on each question."],
        ["Token limit", "110,000 tokens", "Truncates long context before model calls."],
        ["Classification batch concept", "20 texts", "Controls embedding generation for unique recommendations."],
    ], widths=[2.2, 1.6, 3.0])

    h(doc, "6. Production checklist", 1)
    add_bullets(doc, [
        "Confirm OPENAI_API_KEY exists in the runtime process and account budget is available.",
        "Confirm all required files open with pandas or torch in the production environment.",
        "Configure private access to the app.",
        "Run one non-sensitive test per tab before release.",
        "Verify downloads open and do not expose more information than expected.",
        "Document who may replace data files and who may deploy code.",
    ])

    h(doc, "7. Incident response", 1)
    add_numbered(doc, [
        "Disable external access to the app at the hosting or proxy layer.",
        "Revoke and rotate OPENAI_API_KEY if exposure is suspected.",
        "Retain downloads or logs only if internal policy allows it.",
        "Restore data files from verified backups.",
        "Reactivate the app after testing Tabs 1 through 6.",
    ])


def doc_en5(doc):
    h(doc, "1. Functional description", 1)
    p(doc, "The app helps users review project and evaluation documents, chat with uploaded documents, classify recommendations, and analyze institutional action plans. The active interface has six top-level tabs.")

    h(doc, "2. User requirements", 1)
    add_bullets(doc, [
        "Use a modern browser and stable connection.",
        "Have institutional authorization to upload documents to the app.",
        "Use DOCX files with real Word heading styles for Tabs 1-3.",
        "Treat AI outputs as evidence-supported assistance, not a replacement for technical review.",
    ])

    h(doc, "3. Main navigation", 1)
    add_table(doc, ["Tab", "Use it for", "Main output"], [
        ["Valoración Preliminar de Calidad de Proyectos", "Integrated project design review.", "ZIP with question-level XLSX, subsection/section analyses, rubric template, and TXT summary."],
        ["Diagnóstico de Atributos Específicos", "Focused evaluation of participation methodology, gender integration, or modern just transition.", "ZIP with XLSX files by selected rubric."],
        ["Diagnóstico de Sostenibilidad del Proyecto", "Sustainability review of a PRODOC or related project document.", "ZIP with XLSX files and score charts."],
        ["Pregúntale a tus Documentos", "Conversational review of one or more DOCX/TXT files.", "On-screen chat with session memory."],
        ["Clasificación de Recomendaciones", "Spanish UI for recommendation classification using Recommendation description.", "Treemaps, trends, deep-analysis XLSX, and executive-summary XLSX."],
        ["Recommendation Classification", "English UI for the same classification workflow.", "English outputs and downloads."],
    ], widths=[2.2, 2.5, 2.1])

    h(doc, "4. Tab 1: Preliminary Project Quality Appraisal", 1)
    add_numbered(doc, [
        "Download or review Appraisal Checklist_2025 es-419.xlsx if needed.",
        "Upload a DOCX with Word headings.",
        "Extract the document and review the extracted structure.",
        "Select sections and optional question filters.",
        "Run document analysis.",
        "Review metrics, result rows, reasoning, evidence, and critical evaluation.",
        "Download appraisal_checklist_results.zip.",
    ])
    p(doc, "For non-transversal specific-subject questions, the app uses the A-E framework: sub-objective/output, indicator, activity, budget, and quantifiable target. The code can mechanically adjust verdicts based on the A-E total.")
    p(doc, "For two-part questions, Part 2 is evaluated and Part 1 is framing only. Reasoning may state 'Se identificaron 2 partes en esta pregunta.' Evidence tagged [FRAMING] cannot by itself support Partial or Yes.")
    p(doc, "For configured transversal themes, A-E is not used. A means operational presence in objective/product/activity; B means budget/resources. Yes requires A and B; Partial requires A or B; No means neither. Indicators and targets do not count.")
    p(doc, "Tab 1 gender is a transversal A/B topic. Tab 2 gender is a separate rubric, 'Integración del Enfoque de Género', from Rubricas_6ago2025.xlsx sheet rubric_gender_, scored 1-5.")
    p(doc, "The main ZIP contains appraisal_checklist_results.xlsx, appraisal_checklist_rubric_template.xlsx when the source workbook is available, and appraisal_checklist_summary.txt. Extracted structure can also be downloaded as estructura_documento_tab1_<filename>.xlsx.")

    h(doc, "5. Tab 2: Specific Attributes Diagnosis", 1)
    add_numbered(doc, [
        "Upload a DOCX and extract structure.",
        "Select document sections for evaluation.",
        "Select one or more active rubrics: evaluation participation/methodology, gender integration, or modern just transition.",
        "Select criteria within each rubric.",
        "Run processing and evaluation.",
        "Review Criterio, Dimensión, Score, Análisis, Evidencia, Error, and Rúbrica columns.",
        "Download resultados_rubricas.zip.",
    ])
    p(doc, "The extracted structure can be downloaded separately as estructura_documento_tab2_<filename>.xlsx.")

    h(doc, "6. Tab 3: Project Sustainability Diagnosis", 1)
    add_numbered(doc, [
        "Download the sustainability rubric if review is needed.",
        "Upload DOCX, extract structure, and select sections.",
        "Select dimensions and criteria.",
        "Run processing and evaluation.",
        "Review Dimensión, Criterio, Indicador, Score, Análisis, and Evidencia.",
        "Use charts to prioritize weak dimensions and criteria.",
        "Download resultados_evaluacion_prodoc.zip.",
    ])
    p(doc, "The extracted structure can be downloaded separately as estructura_documento_tab3_<filename>.xlsx.")

    h(doc, "7. Tab 4: Ask Your Documents", 1)
    add_numbered(doc, [
        "Upload one or more DOCX or TXT files.",
        "Confirm active filenames and preview text.",
        "Ask specific questions. Request short quotes where useful, but verify section/page references manually.",
        "For up to 100,000 total characters, the app uses full context truncated to 110,000 tokens.",
        "For larger uploads, the app uses 2,000-character chunks with 300-character overlap, retrieves up to 15 FAISS chunks plus up to 5 keyword chunks, and sends the last 5 chat messages.",
    ])
    p(doc, "If replacing a document with another file of the same name, restart the session or rename the file. The chat detects changes by filename list, not content hash.")
    p(doc, "RAG answers can include inference from contextual clues; verify them in the source document before using them in formal work.")

    h(doc, "8. Tabs 5 and 6: Recommendation Classification", 1)
    add_numbered(doc, [
        "Use the default file or upload an XLSX with Recommendation description.",
        "Apply filters by location, time, theme, technical unit, source, progress, or available fields.",
        "Review row count and unique recommendation count before starting.",
        "Decide whether to enable LLM Verification; it can improve matching but is slower and more expensive.",
        "Start classification.",
        "Review dimension/subdimension treemaps and time trends.",
        "Use advanced AI tools for deep analysis or executive summary on the filtered subset.",
        "Download analisis_profundo.xlsx, resumen_ejecutivo.xlsx, deep_analysis.xlsx, or executive_summary.xlsx depending on tab language.",
    ])
    p(doc, "Base classification uses normalized embeddings and numpy cosine similarity, with top 3 assignment. Optional LLM verification reranks top 10 candidates. Secondary dimensions/subdimensions are kept only when similarity is at least 0.60. The output remains row-level even though embeddings are deduplicated by recommendation text.")
    p(doc, "Current limitation: the deep-analysis model selector does not control the effective model; the analysis function uses gpt-4o-mini.")

    h(doc, "9. Common problems", 1)
    add_table(doc, ["Symptom", "Likely cause", "Action"], [
        ["Sections are not detected", "DOCX lacks Word Heading styles.", "Fix headings in Word and extract again."],
        ["OPENAI API key not found", "OPENAI_API_KEY is not available to the process.", "Configure the environment variable and restart."],
        ["Classification does not start", "Missing Recommendations_World.xlsx or Frame_Recommendations_English.xlsx, or missing required columns.", "Restore files or upload a valid XLSX."],
        ["Rows show Error", "Invalid JSON, rate limit, or API failure.", "Retry with fewer criteria/sections or check API quota."],
        ["Tab 4 answers from an old file", "Different content was uploaded with the same filename.", "Rename the file or restart the Streamlit session."],
    ], widths=[2.0, 2.3, 2.5])


def doc_en6(doc):
    h(doc, "1. Operational handoff", 1)
    p(doc, "Daily operation should sit with the ILO team. The original developer should remain a last-resort support contact for issues that cannot be resolved through this documentation, code review, and local reproduction.")
    add_table(doc, ["Role", "Responsibility"], [
        ["ILO owner team", "Own access, data, execution, outputs, and methodological decisions."],
        ["ILO technical administrator", "Manage deployment, environment variables, data files, and backups."],
        ["Analyst users", "Run workflows, review evidence, download results, and apply technical judgment."],
        ["Original developer", "Exceptional support for architecture, complex debugging, or changes outside ordinary maintenance."],
    ], widths=[2.1, 4.7])

    h(doc, "2. Last-resort contact", 1)
    add_bullets(doc, [
        "Legacy contact from prior documentation: ageidv@gmail.com.",
        "Recommended use: only after reproducing the issue locally, reviewing app messages/logs, and isolating the affected tab or function.",
        "Include date, tab, non-sensitive input sample if possible, reproduction steps, error message, and screenshot if internal policy permits.",
    ])

    h(doc, "3. Maintenance routine", 1)
    add_table(doc, ["Frequency", "Activity"], [
        ["Weekly during active operation", "Confirm the app opens, OPENAI_API_KEY works, and a test document processes in Tab 4."],
        ["Monthly", "Test a sample in Tabs 1, 2, 3, 5, and 6; confirm downloads open."],
        ["Before each data update", "Back up Excel, CSV, .pt, and relevant cache files."],
        ["After each code update", "Run manual tests by tab and test_two_part_parsing.py if question logic changed."],
        ["Every 6-12 months", "Review OpenAI models, cost, limits, dependencies, and Streamlit compatibility."],
    ], widths=[2.2, 4.6])

    h(doc, "4. Future update recommendations", 1)
    add_bullets(doc, [
        "Gradually split the monolith into data loading, document extraction, AI clients, evaluation logic, and UI modules.",
        "Remove obsolete commented blocks after confirming they are not needed.",
        "Consolidate the duplicate load_extended_data and load_embeddings startup calls.",
        "Decide whether load_embeddings should still run at startup while visible recommendation search remains commented.",
        "Fix Tab 4 if section/page citations are required by using the enhanced hierarchical extractor.",
        "Wire the deep-analysis model selector into run_row_analysis or remove the selector.",
        "Add tests for parse_two_part_question, _apply_critic_gate_and_render, detect_transversal_matter, _apply_transversal_gate_and_render, classification, and XLSX exports.",
        "Maintain a small non-sensitive regression document set for reproducible validation.",
        "Record schema changes to each Excel file with date and ILO owner.",
        "If the split variants remain in use, propagate relevant changes from oli_v6_deploy.py and test each variant separately.",
    ])

    h(doc, "5. AI and cost recommendations", 1)
    add_bullets(doc, [
        "Keep gpt-5-mini as the main evaluation model while human validation remains consistent.",
        "Use LLM Verification in classification only when added precision justifies cost and latency.",
        "Filter and limit rows before deep recommendation analysis.",
        "Keep embeddings_cache.pkl and analysis_cache.pkl for recurring batches; delete them only if corrupt or after substantive data changes.",
        "Review the startup FAISS recommendation index because classification tabs do not use it, although it still loads.",
        "Track internal cost estimates with generate_cost_report.py if frequent institutional use continues.",
    ])

    h(doc, "6. Change acceptance criteria", 1)
    add_bullets(doc, [
        "The app starts with streamlit run oli_v6_deploy.py.",
        "Tab 1 processes a test DOCX, produces results, recognizes at least one transversal theme, and shows the A/B audit line.",
        "Tab 2 evaluates at least one selected criterion and downloads a ZIP.",
        "Tab 3 generates results and charts for a sample.",
        "Tab 4 answers on both a small TXT and a large DOCX, and the same-filename replacement procedure is understood.",
        "Tabs 5 and 6 classify a limited sample and show treemaps.",
        "Documentation contains no fill-in fields, real credentials, or unnecessary personal paths.",
    ])

    h(doc, "7. Minimum handoff package", 1)
    add_bullets(doc, [
        "Main code: oli_v6_deploy.py.",
        "Dependencies: requirements.txt.",
        "Updated Spanish documentation: six DOCX files in documentation.",
        "Updated English documentation: six DOCX files in documentation/en.",
        "Active data files and rubrics listed in the data model documentation.",
        "ILO internal deployment instructions for the chosen hosting environment.",
        "ILO register of access, data, and deployment owners.",
    ])


DOCS = [
    ("1_Documentacion_Tecnica.docx", "DOCUMENTACIÓN TÉCNICA DEL SISTEMA", "Arquitectura, componentes y despliegue", doc1),
    ("2_Documentacion_Codigo_Fuente.docx", "DOCUMENTACIÓN DEL CÓDIGO FUENTE", "Guía de mantenimiento para la app principal", doc2),
    ("3_Documentacion_Base_Datos.docx", "DOCUMENTACIÓN DE BASE DE DATOS", "Inventario file-based y relaciones lógicas", doc3),
    ("4_Documentacion_Configuracion_Seguridad.docx", "DOCUMENTACIÓN DE CONFIGURACIÓN Y SEGURIDAD", "Variables, secretos, acceso y protección de datos", doc4),
    ("5_Documentacion_Funcional_Operativa.docx", "DOCUMENTACIÓN FUNCIONAL Y OPERATIVA", "Manual operativo para usuarios y administradores", doc5),
    ("6_Otros_Contacto_Recomendaciones.docx", "INFORMACIÓN ADICIONAL DE CIERRE", "Contacto, mantenimiento y recomendaciones futuras", doc6),
]

DOCS_EN = [
    ("1_Technical_Documentation.docx", "SYSTEM TECHNICAL DOCUMENTATION", "Architecture, components, and deployment", doc_en1),
    ("2_Source_Code_Documentation.docx", "SOURCE CODE DOCUMENTATION", "Maintenance guide for the main app", doc_en2),
    ("3_Data_Model_Documentation.docx", "DATA MODEL DOCUMENTATION", "File-based inventory and logical relationships", doc_en3),
    ("4_Configuration_Security_Documentation.docx", "CONFIGURATION AND SECURITY DOCUMENTATION", "Variables, secrets, access, and data protection", doc_en4),
    ("5_Functional_Operational_Documentation.docx", "FUNCTIONAL AND OPERATIONAL DOCUMENTATION", "User and administrator operating manual", doc_en5),
    ("6_Additional_Contact_Recommendations.docx", "ADDITIONAL CLOSEOUT INFORMATION", "Contact, maintenance, and future recommendations", doc_en6),
]


def main():
    EN_DIR.mkdir(exist_ok=True)
    for filename, title, subtitle, builder in DOCS:
        save_doc(filename, title, subtitle, builder)
        print(f"updated {filename}")
    for filename, title, subtitle, builder in DOCS_EN:
        save_doc(filename, title, subtitle, builder, output_dir=EN_DIR, language="en")
        print(f"updated en/{filename}")


if __name__ == "__main__":
    main()
