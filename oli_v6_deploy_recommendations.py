"""ILO recommendations classification app.

Tabs:
    1. Clasificación de Recomendaciones (Spanish)
    2. Recommendation Classification (English)
"""

import concurrent.futures
import pickle
import streamlit as st
import pandas as pd
import json
import tempfile
import docx
import numpy as np
import faiss
import os
import re
from matplotlib import pyplot as plt
import seaborn as sns
from io import BytesIO
import torch
import time
import plotly.express as px
import plotly.graph_objects as go
from docx2python import docx2python
from io import BytesIO
import streamlit as st
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import zipfile


# Use environment variables for API keys
# For local development - use .env file or set environment variables
# For Streamlit Cloud - set these in the app settings
openai_api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI with new SDK (v1.0.0+)
from openai import OpenAI
client = OpenAI(api_key=openai_api_key)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Import tiktoken for accurate token counting
try:
    import tiktoken
    # Use cl100k_base encoding (standard for GPT-4 and GPT-5 models)
    encoding = tiktoken.get_encoding("cl100k_base")
except ImportError:
    encoding = None
    # Warning will be shown when function is first called if needed

# Helper function to truncate text to a token limit
def truncate_to_token_limit(text, max_tokens=110000, encoding_obj=None):
    """
    Truncate text to a maximum number of tokens.
    
    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens (default 110K for GPT-5-mini's 128K context window)
        encoding_obj: tiktoken encoding object (if None, falls back to character estimation)
    
    Returns:
        Truncated text that fits within the token limit
    """
    if not text:
        return text
    
    if encoding_obj is None:
        # Fallback: use character-based estimation (4 chars per token for Spanish)
        # 110K tokens * 4 = 440K characters
        max_chars = max_tokens * 4
        return text[:max_chars]
    
    # Count tokens
    tokens = encoding_obj.encode(text)
    
    # If within limit, return full text
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate to max_tokens
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding_obj.decode(truncated_tokens)
    
    return truncated_text

# Function to get embeddings from OpenAI
def get_embedding_with_retry(text, model='text-embedding-3-large', max_retries=3, delay=1):
    if not openai_api_key:
        st.error("No se encontró la clave de API de OpenAI. Por favor, configura la variable de entorno OPENAI_API_KEY.")
        return None
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=text, model=model)
            return np.array(response.data[0].embedding)
        except Exception as e:
            st.warning(f"Intento {attempt + 1} fallido: {str(e)}")
            time.sleep(delay)
    return None

# Function to generate executive summary using LLM
def generate_executive_summary(recommendations_text, max_output_tokens=3000):
    """
    Generates an executive summary from a list of recommendations using OpenAI.
    """
    if not recommendations_text:
        return "No hay recomendaciones para resumir."

    # Truncate to avoid context limit (approx 120k chars is safe for gpt-4o-mini's 128k tokens, keeping room for output)
    # Using existing helper
    truncated_input = truncate_to_token_limit(recommendations_text, max_tokens=100000, encoding_obj=encoding)

    system_prompt = """Eres un asistente experto en análisis de evaluaciones de proyectos de desarrollo. 
    Tu tarea es generar un Resumen Ejecutivo exhaustivo, informativo y detallado basado en las recomendaciones, respuestas de gestión y planes de acción proporcionados.
    
    El resumen debe:
    1. Identificar los temas principales y patrones recurrentes.
    2. Resaltar las acciones críticas sugeridas.
    3. Incorporar la perspectiva de la respuesta de gestión si está disponible.
    4. Estar estructurado con títulos claros y puntos clave (bullet points).
    5. Ser profesional y directo.
    6. Estar en Español.
    """
    
    user_prompt = f"Aquí están los datos de las recomendaciones (incluyendo descripción, respuesta de gestión, comentarios, etc.):\n\n{truncated_input}\n\nPor favor, genera el Resumen Ejecutivo ahora."

    if not openai_api_key:
        return "Error: No se encontró la clave API de OpenAI."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Cost-effective model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=max_output_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generando el resumen: {str(e)}"

def verify_match_with_llm(rec_text, candidates, api_key):
    """
    Uses LLM to verify which of the candidates is the best match for the recommendation.
    candidates: list of dicts {'dimension':..., 'subdim':..., 'definition':...}
    Returns: index of the best match in candidates list, or -1 if none fit well.
    """
    if not api_key: 
        return 0 # Fallback to top 1 if no key
        
    candidates_str = ""
    for i, c in enumerate(candidates):
        # We assume 'texto_merged' acts as the definition/description
        candidates_str += f"Option {i+1}:\nCategory: {c['dimension']} - {c['subdim']}\nDefinition/Context: {c['texto_merged']}\n\n"
        
    system_prompt = """You are an expert in classifying development project recommendations.
    You will be given a 'Recommendation' and a list of 'Options' (categories with definitions).
    Your task is to select the Option that BEST fits the Recommendation based on the definition.
    
    CRITICAL INSTRUCTIONS:
    1. Read the Recommendation and the Definitions carefully.
    2. **STRICT SEMANTIC MATCH**: The recommendation must describe the *same action* and *same object* as the definition.
    3. **IGNORE SUPERFICIAL KEYWORDS**: Do not select an option just because it shares a word (e.g., "Pensions") if the *context* is different (e.g., "Political backing" vs "Financial funding").
    4. **REQUIREMENT CHECK**: If a definition requires a specific mechanism (e.g., "Financing", "Legislation", "Training"), the recommendation MUST explicitly mention it.
    5. Return ONLY the number of the best option (1, 2, 3...).
    6. If none of them fit reasonably well, return 0.
    """
    
    user_prompt = f"Recommendation: \"{rec_text}\"\n\n{candidates_str}\n\nWhich option is the best fit? (Return just the number, e.g. 1)"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # Deterministic
            max_tokens=6
        )
        content = response.choices[0].message.content.strip()
        # Parse number
        import re
        match = re.search(r'\d+', content)
        if match:
            val = int(match.group())
            if 1 <= val <= len(candidates):
                return val - 1 # 0-indexed
            else:
                return -1 # None or 0
        return 0 # Fallback to Top 1 if parse fails
    except:
        return 0 # Fallback to Top 1 on error

# --- Analysis Cache (for Deep Analysis) ---
class AnalysisCache:
    def __init__(self, cache_file="analysis_cache.pkl"):
        self.cache_file = cache_file
        self.cache = {}
        self.load()

    def load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            except:
                self.cache = {}

    def save(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        self.cache[key] = value

    def generate_key(self, rec, plan):
        # Create a hash based on rec + plan content
        content = f"{str(rec).strip()}|{str(plan).strip()}"
        return content 

# --- Deep Analysis Logic ---
INNOVATION_RUBRIC_PROMPT = """\
RÚBRICA DE INNOVACIÓN (5 dimensiones, cada una 1–5). Evalúe internamente las 5 dimensiones, luego derive rec_innovation_score como el promedio redondeado mapeado a la escala categórica (1=Very low, 2=Low, 3=Medium, 4=High, 5=Very High).

D1. NOVEDAD RELEVANTE — qué tan diferente es la recomendación respecto de prácticas habituales. NO premiar lenguaje tecnológico ni aspiracional; NO confundir tamaño con innovación.
  1=Muy baja: reitera práctica normal (reuniones de seguimiento, actualizar base de beneficiarios, informes trimestrales).
  2=Baja: ajuste incremental a práctica conocida (mejorar formato de reportes, ampliar capacitaciones existentes).
  3=Media: adapta práctica conocida a nuevo contexto/población (incorporar teoría de cambio donde no existía, adaptar capacitación a comunidades rurales).
  4=Alta: mecanismo poco común en ese contexto (red interinstitucional formal, fondo competitivo, laboratorio de innovación).
  5=Muy alta: solución no rutinaria con lógica nueva (sistema nacional interoperable, arquitectura institucional nueva, ecosistema digital colaborativo).

D2. VALOR PARA CONSTITUYENTES — mejora acceso, calidad, eficiencia, inclusión o resultados para usuarios/beneficiarios/mandantes. NO premiar mejoras internas sin impacto en usuarios.
  1=Muy bajo: sin problema ni beneficiario claro (fortalecer gestión, optimizar procesos administrativos).
  2=Bajo: mejora marginal/indirecta sin impacto en resultados.
  3=Medio: responde a un problema con impacto limitado o parcial.
  4=Alto: mejora claramente calidad/cobertura/pertinencia (servicios adaptados a poblaciones vulnerables, reducir barreras de acceso).
  5=Muy alto: transforma acceso/beneficio (acceso universal antes inexistente, eliminar barreras estructurales).

D3. APRENDIZAJE, PRUEBA Y ESCALAMIENTO — mecanismos para experimentar, medir, aprender y replicar.
  1=Muy bajo: actividad única sin aprendizaje (un taller, un informe final).
  2=Bajo: aprendizaje implícito sin estructura.
  3=Medio: potencial de réplica pero sin cómo aprender/mejorar.
  4=Alto: piloto/seguimiento/mejora progresiva estructurados.
  5=Muy alto: ciclo completo piloto→medición→retroalimentación→escalamiento.

D4. CAMBIO SISTÉMICO O INSTITUCIONAL — modifica estructuras, reglas, coordinación o gobernanza (no solo actividades). NO confundir cobertura con cambio sistémico.
  1=Muy bajo: actividad aislada (capacitaciones, materiales, talleres).
  2=Bajo: mejora operativa local sin afectar estructura.
  3=Medio: cambio dentro de una unidad/programa.
  4=Alto: cambio organizacional o de coordinación entre actores (red interinstitucional con roles, nuevo modelo de gobernanza en implementación).
  5=Muy alto: transformación sistémica (sistema nacional integrado, rediseño de gobernanza de política pública, arquitectura multisectorial nueva).

D5. FACTIBILIDAD ESTRATÉGICA — viable en el contexto institucional, político, técnico y operativo actual. Penalizar aspiracional/vago sin mecanismo.
  1=Muy baja: irreal (sistema nacional sin actores ni recursos, transformación sin estrategia).
  2=Baja: requiere condiciones altamente improbables o no descritas.
  3=Media: implementable con ajustes, recursos o coordinación adicional.
  4=Alta: coherente con capacidades, actores y contexto (escalar intervención ya probada).
  5=Muy alta: condiciones de implementación claras (piloto exitoso con actores comprometidos, modelo validado en contextos similares).
"""


def analyze_recommendation_plan_pair(recommendation, action_plan, comments, client, model="gpt-4o-mini"):
    """
    Send recommendation, action plan, and comments to OpenAI API for analysis
    """
    if pd.isna(recommendation) or pd.isna(action_plan):
        return None # Skip empty
    
    # Handle NaN comments
    if pd.isna(comments):
        comments = "No additional comments provided."
    
    # Define ILO development project relevant tags
    ilo_tags_list = [
        "capacity_building", "training", "employment_creation", "gender_equality", 
        "decent_work", "labor_rights", "social_dialogue", "social_protection",
        "occupational_safety", "child_labor", "forced_labor", "labor_migration",
        "working_conditions", "skills_development", "social_inclusion", "monitoring_evaluation",
        "institutional_strengthening", "policy_development", "knowledge_management",
        "sustainability", "stakeholder_engagement", "data_collection", "technical_assistance",
        "project_design", "implementation_methodology", "resource_allocation", 
        "coordination", "partnership_building", "innovation", "digital_transformation"
    ]
    
    rec_tags_list = ['governance', 'participation', 'gender_issues', 'just_transition', 'institutional_strenghtening', 
                'public_policy_incidence', 'financial_sustainability']
    
    prompt = f"""
    Analyze the following recommendation, its corresponding action plan, and additional comments:
    
    RECOMMENDATION:
    {recommendation}
    
    ACTION PLAN:
    {action_plan}
    
    ADDITIONAL COMMENTS:
    {comments}
    
    Please extract and return ONLY a JSON object with the following fields:
    1. extracted_actions_from_rec: List all specific actions requested in the recommendation
    2. actions_proposed_in_plan: List all specific actions mentioned in the action plan
    3. difficulties_mentioned: Any difficulties or challenges mentioned in implementing the recommendation (look carefully in the action plan AND comments for these)
    4. reasons_for_rejection: If the recommendation wasn't fully accepted, reasons given (pay special attention to justifications in the comments)
    5. rejection_difficulty_classification: Classify the reasons (Financial, Technical, Political, Low priority, Unjustified, Third party dependency, Time constraints, Cultural/behavioral, Local operational constraints, Mandate limitations, Other). At most 3.
    6. coherence_score: Score from 0-10 how well the action plan addresses the recommendation
    7. coherence_rationale: Detailed explanation (6-8 sentences).
    8. plan_quality_score: Score from 0-10 (specificity, feasibility).
    9. plan_quality_rationale: Detailed explanation (6-8 sentences).
    10. attention_level_score: Score from 0-10 (priority given).
    11. attention_level_rationale: Detailed explanation (6-8 sentences).
    12. overall_score: Score from 0-10.
    13. overall_score_rationale: Extensive analysis (6-8 sentences).
    14. tags: Select 2-5 most relevant tags from: {", ".join(ilo_tags_list)}
    
    Now, analyze the recommendation on its own:

    {INNOVATION_RUBRIC_PROMPT}

    15. rec_innovation_score: Apply the 5-dimension innovation rubric above. Score each dimension 1–5 internally, then return the overall level as one of (Very low, Low, Medium, High, Very High), derived from the rounded average of the five dimension scores (1→Very low, 2→Low, 3→Medium, 4→High, 5→Very High).
    16. rec_innovation_rationale: 3-4 sentences. Explicitly cite the per-dimension scores in the form "D1=x, D2=x, D3=x, D4=x, D5=x" and briefly justify the weakest/strongest dimensions. Do not reward tech/aspirational language or size alone.
    17. rec_precision_and_clarity: (Very low, Low, Medium, High, Very High).
    18. rec_precision_and_clarity_rationale: Explanation (3-4 sentences).
    19. rec_additional_tags: Select 2-5 relevant tags from: {", ".join(rec_tags_list)}
    20. rec_operational_feasibility: (Very low, Low, Medium, High, Very High).
    21. rec_operational_feasibility_rationale: Explanation (3-4 sentences).
    22. rec_timeline: (short, medium, long).
    23. rec_timeline_rationale: Explanation (3-4 sentences).
    24. rec_expected_impact: (Very low, Low, Medium, High, Very High).
    25. rec_expected_impact_rationale: Explanation (3-4 sentences).
    26. rec_intervention_approach: (processes, results, policy).
    27. rec_intervention_approach_rationale: Explanation (3-4 sentences).
    
    Respond ONLY with the JSON object.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You analyze recommendations and action plans, returning structured JSON results."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        return {
            "error": str(e),
            "coherence_score": 0,
            "overall_score": 0
        }

def run_row_analysis(args):
    """
    Worker for parallel analysis
    """
    idx, rec, plan, comments, api_key, cache_obj = args
    
    # Check cache first
    key = cache_obj.generate_key(rec, plan)
    cached = cache_obj.get(key)
    if cached:
        return idx, cached, True # True = from cache
        
    # If not in cache, call API
    client = OpenAI(api_key=api_key)
    result = analyze_recommendation_plan_pair(rec, plan, comments, client)
    
    return idx, result, False # False = new call

# --- Begin: SimpleHierarchicalStore and RAG logic from megaparse_example.py ---
import pickle
from typing import List, Dict, Any
import numpy as np
import json
import os
import openai


# ============= MAIN APP CODE =============

# Set page config
st.set_page_config(layout="wide")

# ============= ILO GLOBAL STYLES =============
# ILO Color Palette: Red (#C8102E), Blue (#002F6C), White (#FFFFFF)
# Secondary: Light Blue (#0072CE), Light Gray (#F5F5F5)
st.markdown("""
<style>
/* ===== ILO Color Variables ===== */
:root {
    --ilo-red: #C8102E;
    --ilo-blue: #002F6C;
    --ilo-light-blue: #0072CE;
    --ilo-white: #FFFFFF;
    --ilo-light-gray: #F5F5F5;
    --ilo-dark-gray: #333333;
}

/* ===== Global Typography ===== */
h1, h2, h3 {
    color: var(--ilo-blue) !important;
    font-weight: 600 !important;
}

h4, h5, h6 {
    color: var(--ilo-dark-gray) !important;
}

/* ===== Primary Buttons ===== */
.stButton > button {
    background-color: var(--ilo-blue) !important;
    color: var(--ilo-white) !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.25rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: var(--ilo-light-blue) !important;
    box-shadow: 0 4px 12px rgba(0, 47, 108, 0.25) !important;
}

/* ===== Download Buttons ===== */
.stDownloadButton > button {
    background-color: var(--ilo-red) !important;
    color: var(--ilo-white) !important;
    border: none !important;
    border-radius: 6px !important;
}

.stDownloadButton > button:hover {
    background-color: #A00D24 !important;
    box-shadow: 0 4px 12px rgba(200, 16, 46, 0.3) !important;
}

/* ===== Tabs Styling ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: var(--ilo-light-gray);
    padding: 0.5rem;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--ilo-blue) !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    background-color: var(--ilo-blue) !important;
    color: var(--ilo-white) !important;
}

/* ===== Expanders ===== */
.streamlit-expanderHeader {
    background-color: var(--ilo-light-gray) !important;
    border-radius: 8px !important;
    color: var(--ilo-blue) !important;
    font-weight: 500 !important;
}

/* ===== Info/Warning/Success Boxes ===== */
.stAlert > div[data-baseweb="notification"] {
    border-radius: 8px !important;
}

div[data-testid="stNotificationContentInfo"] {
    background-color: rgba(0, 114, 206, 0.1) !important;
    border-left: 4px solid var(--ilo-light-blue) !important;
}

div[data-testid="stNotificationContentSuccess"] {
    background-color: rgba(40, 167, 69, 0.1) !important;
    border-left: 4px solid #28a745 !important;
}

div[data-testid="stNotificationContentWarning"] {
    background-color: rgba(200, 16, 46, 0.08) !important;
    border-left: 4px solid var(--ilo-red) !important;
}

/* ===== Checkboxes ===== */
.stCheckbox > label > div[data-testid="stMarkdownContainer"] {
    color: var(--ilo-dark-gray) !important;
}

/* ===== Sliders ===== */
.stSlider > div > div > div > div {
    background-color: var(--ilo-blue) !important;
}

/* ===== Select boxes ===== */
.stSelectbox > div > div {
    border-color: var(--ilo-blue) !important;
}

/* ===== DataFrames ===== */
.stDataFrame {
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
}

/* ===== Progress bars ===== */
.stProgress > div > div > div > div {
    background-color: var(--ilo-blue) !important;
}

/* ===== Metric cards ===== */
[data-testid="stMetricValue"] {
    color: var(--ilo-blue) !important;
}

/* ===== Reference boxes (custom) ===== */
.reference-box {
    border-left: 4px solid var(--ilo-blue) !important;
    background: linear-gradient(135deg, #f0f4f8, #e8eef5) !important;
    padding: 0.85rem 1rem;
    border-radius: 8px;
    font-size: 0.95rem;
    color: var(--ilo-dark-gray);
    margin-bottom: 0.75rem;
    box-shadow: 0 4px 14px rgba(0, 47, 108, 0.08);
}

/* ===== Horizontal rule ===== */
hr {
    border-top: 2px solid var(--ilo-blue) !important;
}

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background-color: var(--ilo-light-gray) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--ilo-blue) !important;
}

/* ===== File uploader ===== */
.stFileUploader > div > div {
    border: 2px dashed var(--ilo-blue) !important;
    border-radius: 8px !important;
}

/* ===== Radio buttons ===== */
.stRadio > div {
    gap: 0.5rem;
}

/* ===== Multiselect ===== */
.stMultiSelect > div > div {
    border-color: var(--ilo-blue) !important;
}

/* ===== Spinner ===== */
.stSpinner > div {
    border-top-color: var(--ilo-blue) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <h2 style='text-align:center; color:#002F6C; margin-top:0;'>
        Caja de Herramientas para el Mejor Desempeño de los Proyectos
        <br>
        <span style='font-size:0.8em; font-weight:500;'>Toolkit for Better Project Performance</span>
    </h2>
    <h3 style='text-align:center; color:#002F6C; margin-top:0;'>
        Usando Evidencia de las Evaluaciones
        <br>
        <span style='font-size:0.85em; font-weight:500;'>Using Evidence from Evaluations</span>
    </h3>
    <hr style='border-top: 2px solid #002F6C;'>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs([
    "Clasificación de Recomendaciones",
    "Recommendation Classification",
])

with tab1:
    st.header("Clasificación de Recomendaciones")
    st.info("""
    **🌍 Clasificación Inteligente de Recomendaciones**

    Esta herramienta permite clasificar automáticamente las recomendaciones contenidas en el archivo `Recommendations_World.xlsx` 
    asignándolas a las subdimensiones definidas en el marco de trabajo `Frame_Recommendations_English.xlsx`.

    **Optimizaciones:**
    1.  **Filtrado Previo:** Filtra por región/país antes de procesar para ahorrar costos y tiempo.
    2.  **Caché Persistente:** Los embeddings se guardan localmente para no regenerarlos en futuras ejecuciones.
    3.  **Deduplicación:** Textos idénticos se procesan una sola vez.

    **Funcionamiento:**
    1.  Carga los datos de recomendaciones y el marco de referencia.
    2.  Calcula la similitud semántica (usando OpenAI Embeddings) entre cada recomendación y las definiciones de subdimensiones.
    3.  Asigna la subdimensión más relevante.
    4.  Visualiza la distribución mediante Treemaps interactivos.
    """)

    # --- Persistent Cache Class ---
    class EmbeddingsCache:
        def __init__(self, cache_file="embeddings_cache.pkl"):
            self.cache_file = cache_file
            self.cache = {}
            self.load_cache()

        def load_cache(self):
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, 'rb') as f:
                        self.cache = pickle.load(f)
                except Exception:
                    self.cache = {}

        def save_cache(self):
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception:
                pass

        def get(self, text):
            return self.cache.get(text)

        def set(self, text, embedding):
            self.cache[text] = embedding

        def save(self):
            self.save_cache()

    # Initialize cache in session state if not exists
    if 'embeddings_cache_obj' not in st.session_state:
        st.session_state['embeddings_cache_obj'] = EmbeddingsCache()

    # --- Load Data ---
    @st.cache_data
    def load_world_data():
        try:
            # Check if files exist
            if not os.path.exists('Recommendations_World.xlsx') or not os.path.exists('Frame_Recommendations_English.xlsx'):
                return None, None
            
            rec_df = pd.read_excel('Recommendations_World.xlsx')
            frame_df = pd.read_excel('Frame_Recommendations_English.xlsx')
            return rec_df, frame_df
        except Exception as e:
            st.error(f"Error cargando los archivos de datos: {e}")
            return None, None

    
    # --- Data Source Selection ---
    st.markdown("### Fuente de Datos")
    data_source = st.radio("Selecciona origen de las recomendaciones:", 
                          ["Archivos Predeterminados (World)", "Cargar Archivo Propio (.xlsx)"], 
                          horizontal=True,
                          key="data_source_radio")
    
    rec_df_world = None
    frame_df_world = None
    
    # helper to just get frame (reusing cached function for now)
    _, frame_default = load_world_data()
    frame_df_world = frame_default

    if data_source == "Archivos Predeterminados (World)":
        rec_default, _ = load_world_data()
        rec_df_world = rec_default
        
        if rec_df_world is None:
             st.warning("⚠️ No se encontró el archivo 'Recommendations_World.xlsx' por defecto.")

    else:
        uploaded_file = st.file_uploader("Sube tu archivo de recomendaciones (Excel):", type=["xlsx"], key="custom_rec_upload")
        if uploaded_file:
            try:
                rec_df_world = pd.read_excel(uploaded_file)
                # Validation
                if 'Recommendation description' not in rec_df_world.columns:
                    st.error("❌ El archivo debe contener una columna llamada 'Recommendation description'.")
                    rec_df_world = None
                else:
                    st.success(f"Archivo cargado correctamente: {len(rec_df_world)} filas.")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
        else:
            st.info("Esperando archivo...")

    if rec_df_world is None or frame_df_world is None:
        if frame_df_world is None:
             st.warning("⚠️ No se encontró el archivo 'Frame_Recommendations_English.xlsx'. es necesario para la clasificación.")
    else:
        
        # --- DATA PREP: Extract Year ---
        if 'Recommendation date' in rec_df_world.columns:
            # Try parsing with existing format or coerse
            rec_df_world['Recommendation date'] = pd.to_datetime(rec_df_world['Recommendation date'], errors='coerce')
            rec_df_world['Year'] = rec_df_world['Recommendation date'].dt.year
            # Fill NaNs with a placeholder if needed, or leave as NaN
            # rec_df_world['Year'] = rec_df_world['Year'].fillna(0).astype(int)
        
        # --- Filters (PRE-PROCESSING) ---
        st.markdown("### 1. Filtros de Selección")
        st.caption("Selecciona los filtros para definir el subconjunto de datos a clasificar. Esto optimiza el tiempo y costo de procesamiento.")
        
        # Initialize filter selection dictionaries
        filters = {}
        
        # --- Group 1: Ubicación y Tiempo ---
        with st.expander("📍 Ubicación y Tiempo", expanded=True):
            col_grp1_1, col_grp1_2 = st.columns(2)
            
            with col_grp1_1:
                # Region Filter
                regions = sorted([str(x) for x in rec_df_world['Region(s)'].unique() if not pd.isna(x)]) if 'Region(s)' in rec_df_world.columns else []
                filters['Region(s)'] = st.multiselect("Región(es):", options=regions, default=[], key="tab6_region_filter")
                
                # Admin Unit
                if 'Recommendation administrative unit' in rec_df_world.columns:
                    admin_units = sorted([str(x) for x in rec_df_world['Recommendation administrative unit'].unique() if not pd.isna(x)])
                    filters['Recommendation administrative unit'] = st.multiselect("Unidad Administrativa:", options=admin_units, default=[], key="tab6_admin_unit")
            
            with col_grp1_2:
                # Country Filter (Cascading)
                available_countries_df = rec_df_world.copy()
                if filters['Region(s)']:
                     if 'Region(s)' in available_countries_df.columns:
                        available_countries_df = available_countries_df[available_countries_df['Region(s)'].isin(filters['Region(s)'])]
                
                countries = sorted([str(x) for x in available_countries_df['Country(ies)'].unique() if not pd.isna(x)]) if 'Country(ies)' in available_countries_df.columns else []
                filters['Country(ies)'] = st.multiselect("País(es):", options=countries, default=[], key="tab6_country_filter")
                
                # Year Filter
                if 'Year' in rec_df_world.columns:
                    years = sorted([int(x) for x in rec_df_world['Year'].unique() if not pd.isna(x)])
                    filters['Year'] = st.multiselect("Año:", options=years, default=[], key="tab6_year_filter")

        # --- Group 2: Temática y Técnica ---
        with st.expander("📚 Temática y Técnica", expanded=False):
            col_grp2_1, col_grp2_2 = st.columns(2)
            
            with col_grp2_1:
                if 'Evaluation theme(s)' in rec_df_world.columns:
                    themes = sorted([str(x) for x in rec_df_world['Evaluation theme(s)'].unique() if not pd.isna(x)])
                    filters['Evaluation theme(s)'] = st.multiselect("Temática de Evaluación:", options=themes, default=[], key="tab6_eval_theme")
                
                if 'Recommendation theme' in rec_df_world.columns:
                    rec_themes = sorted([str(x) for x in rec_df_world['Recommendation theme'].unique() if not pd.isna(x)])
                    filters['Recommendation theme'] = st.multiselect("Temática de Recomendación:", options=rec_themes, default=[], key="tab6_rec_theme")

                if 'Technical unit(s)' in rec_df_world.columns:
                    tech_units = sorted([str(x) for x in rec_df_world['Technical unit(s)'].unique() if not pd.isna(x)])
                    filters['Technical unit(s)'] = st.multiselect("Unidad Técnica:", options=tech_units, default=[], key="tab6_tech_unit")

            with col_grp2_2:
                if 'Funding source(s)' in rec_df_world.columns:
                    fundings = sorted([str(x) for x in rec_df_world['Funding source(s)'].unique() if not pd.isna(x)])
                    filters['Funding source(s)'] = st.multiselect("Fuente de Financiamiento:", options=fundings, default=[], key="tab6_funding")
                
                if 'Evaluation nature' in rec_df_world.columns:
                    natures = sorted([str(x) for x in rec_df_world['Evaluation nature'].unique() if not pd.isna(x)])
                    filters['Evaluation nature'] = st.multiselect("Naturaleza de Evaluación:", options=natures, default=[], key="tab6_eval_nature")
                
                if 'Evaluation type' in rec_df_world.columns:
                    types = sorted([str(x) for x in rec_df_world['Evaluation type'].unique() if not pd.isna(x)])
                    filters['Evaluation type'] = st.multiselect("Tipo de Evaluación:", options=types, default=[], key="tab6_eval_type")

        # --- Group 3: Gestión y Respuesta ---
        with st.expander("⚙️ Gestión y Respuesta", expanded=False):
             col_grp3_1, col_grp3_2 = st.columns(2)
             
             with col_grp3_1:
                 if 'Evaluation timing' in rec_df_world.columns:
                    timings = sorted([str(x) for x in rec_df_world['Evaluation timing'].unique() if not pd.isna(x)])
                    filters['Evaluation timing'] = st.multiselect("Momento de Evaluación:", options=timings, default=[], key="tab6_eval_timing")

                 if 'Progress' in rec_df_world.columns:
                    progresses = sorted([str(x) for x in rec_df_world['Progress'].unique() if not pd.isna(x)])
                    filters['Progress'] = st.multiselect("Progreso:", options=progresses, default=[], key="tab6_progress")

             with col_grp3_2:
                 if 'Management response' in rec_df_world.columns:
                     responses = sorted([str(x) for x in rec_df_world['Management response'].unique() if not pd.isna(x)])
                     filters['Management response'] = st.multiselect("Respuesta de Administración:", options=responses, default=[], key="tab6_mgmt_response")
                 
                 # Optional: Evaluation document type if needed
                 if 'Evaluation document type' in rec_df_world.columns:
                     doc_types = sorted([str(x) for x in rec_df_world['Evaluation document type'].unique() if not pd.isna(x)])
                     filters['Evaluation document type'] = st.multiselect("Tipo de Documento:", options=doc_types, default=[], key="tab6_doc_type")


        # Apply ALL filters
        start_count = len(rec_df_world)
        filtered_rec_df = rec_df_world.copy()
        
        for col_name, selected_values in filters.items():
            if selected_values and col_name in filtered_rec_df.columns:
                # Handle Year specifically if it's float/int comparison issues, but isin handles standard types well
                filtered_rec_df = filtered_rec_df[filtered_rec_df[col_name].isin(selected_values)]

            
        end_count = len(filtered_rec_df)

        # Show row count AND unique-recommendation count: a region with many rows
        # may contain far fewer unique recommendations because the source file repeats
        # rows for multi-attribute records (themes, sources, etc.). Embedding cost is
        # driven by uniques, so analysts need both numbers before clicking process.
        id_col_pre = 'Recommendation ID' if 'Recommendation ID' in filtered_rec_df.columns else 'Recommendation description'
        unique_count_pre = filtered_rec_df[id_col_pre].nunique()
        total_unique_pre = rec_df_world[id_col_pre].nunique()
        st.markdown(
            f"**Registros seleccionados:** {end_count} de {start_count}  |  "
            f"**Recomendaciones únicas:** {unique_count_pre} de {total_unique_pre}"
        )
        st.caption("⚠️ El número de registros puede ser mayor que el número de recomendaciones únicas porque algunas recomendaciones tienen múltiples atributos (ej. múltiples temas), generando filas duplicadas en el archivo original. El costo de clasificación se basa en las recomendaciones únicas.")

        # --- Classification Logic ---
        
        # Persistent state for classified data
        if 'classified_world_df' not in st.session_state:
            st.session_state['classified_world_df'] = None

        def classify_recommendations(target_df, reference_df, cache_obj, use_llm_verification=False):
            """
            Classifies recommendations in target_df based on similarity to texts in reference_df.
            Uses persistent cache and deduplication.
            """
            
            progress_bar = st.progress(0, text="Iniciando proceso...")

            # 1. Embed Reference Frame
            ref_embeddings = []
            ref_metadata = []
            
            # Check if we have 'texto_merged'
            if 'texto_merged' not in reference_df.columns:
                st.error("El archivo Frame debe tener la columna 'texto_merged'.")
                return None

            # Generate embeddings for Frame (with cache)
            for idx, row in reference_df.iterrows():
                text = str(row['texto_merged'])
                
                # Try cache first
                emb = cache_obj.get(text)
                if emb is None:
                    emb = get_embedding_with_retry(text) 
                    if emb is not None:
                        cache_obj.set(text, emb)
                
                if emb is not None:
                    ref_embeddings.append(emb)
                    ref_metadata.append({
                        'dimension': row.get('dimension', 'Unknown'),
                        'subdim': row.get('subdim', 'Unknown'),
                        'texto_merged': text
                    })
                
                progress_val = 0.1 * (idx + 1) / len(reference_df)
                progress_bar.progress(progress_val, text=f"Procesando marco de referencia {idx+1}/{len(reference_df)}")
            
            cache_obj.save() # Save frame embeddings
            
            if not ref_embeddings:
                st.error("No se pudieron generar embeddings para el marco de referencia.")
                return None

            ref_embeddings = np.array(ref_embeddings)
            
            # Normalize frame embeddings
            ref_norms = np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
            ref_embeddings_norm = ref_embeddings / (ref_norms + 1e-10)
            
            # 2. Embed Unique Recommendations (Deduplication)
            start_time = time.time()
            total_recs = len(target_df)
            
            # Identify unique texts to embed
            unique_texts = target_df['Recommendation description'].dropna().unique()
            unique_embeddings = {}
            
            new_embeddings_count = 0
            
            # Embed unique texts
            BATCH_SIZE = 20
            for i in range(0, len(unique_texts), BATCH_SIZE):
                batch_texts = unique_texts[i:i+BATCH_SIZE]
                for text in batch_texts:
                    text_str = str(text)
                    # Try cache
                    emb = cache_obj.get(text_str)
                    if emb is None:
                        emb = get_embedding_with_retry(text_str)
                        if emb is not None:
                            cache_obj.set(text_str, emb)
                            new_embeddings_count += 1
                    
                    if emb is not None:
                        unique_embeddings[text_str] = emb
                
                progress_pct = 0.1 + (0.4 * (i + len(batch_texts)) / len(unique_texts))
                progress_bar.progress(progress_pct, text=f"Generando embeddings únicos: {min(i+len(batch_texts), len(unique_texts))}/{len(unique_texts)}")
                
                # Save cache periodically
                if new_embeddings_count > 50:
                    cache_obj.save()
                    new_embeddings_count = 0

            cache_obj.save() # Final save

        # 3. Classify Rows (Match) - Top 3
            classified_rows = []
            
            progress_bar.progress(0.5, text="Asignando clasificaciones (Top 3) " + ("y verificando con LLM..." if use_llm_verification else "..."))
            
            for i, (idx, row) in enumerate(target_df.iterrows()):
                rec_text = str(row.get('Recommendation description', ''))
                
                # Default values
                vals = {
                    'assigned_dimension': 'Unclassified',
                    'assigned_subdim': 'Unclassified',
                    'matched_frame_text': '',
                    'similarity_score': 0.0,
                    'Otras dimensiones': '',
                    'Otras subdimensiones': ''
                }

                if rec_text in unique_embeddings:
                    rec_emb = unique_embeddings[rec_text]
                    
                    # Cosine similarity
                    rec_emb_norm = rec_emb / (np.linalg.norm(rec_emb) + 1e-10)
                    similarities = np.dot(ref_embeddings_norm, rec_emb_norm)
                    
                    # Get Top Matches for context
                    # Even if verifying, we start with top candidates from embeddings
                    
                    if use_llm_verification:
                        top_n_candidates = 10
                    else:
                        top_n_candidates = 3
                        
                    top_k_indices = np.argsort(similarities)[-top_n_candidates:][::-1]
                    
                    best_idx = top_k_indices[0] # Default to embedding top 1
                    
                    # LLM Verification (Reranking)
                    if use_llm_verification:
                        # Construct candidates list (Top 10)
                        candidates = []
                        for k in top_k_indices:
                             candidates.append(ref_metadata[k])
                        
                        # Call LLM
                        verified_idx = verify_match_with_llm(rec_text, candidates, openai_api_key)
                        
                        if verified_idx >= 0 and verified_idx < len(top_k_indices):
                            best_idx = top_k_indices[verified_idx]
                        # Else fallback to embedding best_idx (0)
                    
                    # Assign Best Match (Verified or Embedding)
                    best_match = ref_metadata[best_idx]
                    
                    vals['assigned_dimension'] = best_match['dimension']
                    vals['assigned_subdim'] = best_match['subdim']
                    vals['matched_frame_text'] = best_match['texto_merged']
                    vals['similarity_score'] = similarities[best_idx]
                    
                    # Process Top 2 & 3 (Alternatives)
                    # Note: If LLM picked #2, then #1 and #3 become alternatives.
                    # For simplicity, we stick to the Embedding Ranking for "Others" 
                    # OR we could just list the other 2 from the Top 3 set.
                    # Let's simple: List the *other* indices from top_k_indices that are NOT best_idx
                    
                    secondary_dims = []
                    secondary_subdims = []
                    
                    similarity_threshold = 0.60
                    
                    for k_idx in top_k_indices:
                        if k_idx == best_idx: continue # Skip the chosen one
                        
                        if similarities[k_idx] >= similarity_threshold:
                            match_k = ref_metadata[k_idx]
                            secondary_dims.append(match_k['dimension'])
                            secondary_subdims.append(match_k['subdim'])
                        
                    vals['Otras dimensiones'] = "; ".join(secondary_dims)
                    vals['Otras subdimensiones'] = "; ".join(secondary_subdims)
                
                # Merge original row data with new classification data
                # We want to keep ALL original columns
                row_dict = row.to_dict()
                row_dict.update(vals)
                classified_rows.append(row_dict)
                
                if i % 10 == 0: # Update faster if slow LLM
                     progress_bar.progress(0.5 + (0.5 * (i + 1) / total_recs), text=f"Clasificando registros {i+1}/{total_recs}")

            progress_bar.empty()
            return pd.DataFrame(classified_rows)

        # Button to run classification
        st.markdown("### 2. Ejecución")
        
        # LLM Verification Toggle
        use_llm_chk = st.checkbox("✅ Usar Verificación con LLM (Alta Precisión, Más Lento)", value=False, help="Si se activa, el sistema usará IA para leer las definiciones y corregir la clasificación (ej. distinguir 'Pensiones' financiera de 'Pensiones' política).")

        if start_count == 0:
            st.warning("El archivo de recomendaciones está vacío.")
        elif end_count == 0:
             st.warning("No hay registros que coincidan con los filtros seleccionados. Ajusta los filtros.")
        else:
            if st.button("🚀 Iniciar Clasificación", key="start_classification_tab6"):
                with st.spinner("Ejecutando clasificación optimizada..."):
                    classified_df = classify_recommendations(filtered_rec_df, frame_df_world, st.session_state['embeddings_cache_obj'], use_llm_verification=use_llm_chk)
                    if classified_df is not None and not classified_df.empty:
                        st.session_state['classified_world_df'] = classified_df
                        st.success(f"¡Clasificación completada! {len(classified_df)} recomendaciones procesadas.")
                    else:
                        st.error("Hubo un problema durante la clasificación.")

        # --- Visualization ---
        df_viz = st.session_state['classified_world_df']
        
        if df_viz is not None:
            st.markdown("---")
            st.subheader("📊 Visualización de Resultados")
            
            
            # --- Visual Filters (Post-Classification) ---
            st.markdown("### 🔎 Filtros Visuales")
            st.warning("Estos filtros afectan solo a los gráficos y tablas a continuación, sin re-ejecutar la clasificación.")
            
            # Prepare filter options (based on current results to be safe)
            # Create a localized copy for filtering
            df_filtered_viz = df_viz.copy()

            # 1. Year Slider (Range)
            if 'Year' in df_viz.columns:
                 # Handle NaNs or zeros logic 
                 valid_years = sorted([int(x) for x in df_viz['Year'].unique() if pd.notna(x) and x > 1900])
                 if valid_years:
                     min_year, max_year = min(valid_years), max(valid_years)
                     year_range = st.slider(
                         "Intervalo de Años:",
                         min_value=min_year,
                         max_value=max_year,
                         value=(min_year, max_year),
                         key="viz_year_slider"
                     )
                     df_filtered_viz = df_filtered_viz[
                         (df_filtered_viz['Year'] >= year_range[0]) & 
                         (df_filtered_viz['Year'] <= year_range[1])
                     ]

            # 2. Metadata Filters (multiselects) - Grouped
            with st.expander("Más Filtros Visuales (Metadatos)", expanded=False):
                col_vf1, col_vf2, col_vf3 = st.columns(3)
                
                with col_vf1:
                    # Admin Unit
                    if 'Recommendation administrative unit' in df_filtered_viz.columns:
                         opts = sorted([str(x) for x in df_viz['Recommendation administrative unit'].unique() if pd.notna(x)])
                         sel_admin = st.multiselect("Unidad Admin:", options=opts, key="viz_admin_unit")
                         if sel_admin:
                             df_filtered_viz = df_filtered_viz[df_filtered_viz['Recommendation administrative unit'].isin(sel_admin)]

                with col_vf2:
                    # Country
                    if 'Country(ies)' in df_filtered_viz.columns:
                         opts = sorted([str(x) for x in df_viz['Country(ies)'].unique() if pd.notna(x)])
                         sel_country = st.multiselect("País:", options=opts, key="viz_country")
                         if sel_country:
                             df_filtered_viz = df_filtered_viz[df_filtered_viz['Country(ies)'].isin(sel_country)]

                with col_vf3:
                    # Dimension (Pre-filter for everything?? Or just let the treemap handle it? User asked for filters)
                    # Let's add Dimension here too affects everything
                    opts = sorted([str(x) for x in df_viz['assigned_dimension'].unique() if pd.notna(x)])
                    sel_dim_global = st.multiselect("Dimensión:", options=opts, key="viz_dim_global")
                    if sel_dim_global:
                         df_filtered_viz = df_filtered_viz[df_filtered_viz['assigned_dimension'].isin(sel_dim_global)]


            # Count Unique Recommendations
            # Use 'Recommendation ID' if available, otherwise 'Recommendation description'
            id_col = 'Recommendation ID' if 'Recommendation ID' in df_filtered_viz.columns else 'Recommendation description'
            unique_count = df_filtered_viz[id_col].nunique()
            
            st.markdown(f"**Registros visualizados:** {len(df_filtered_viz)} | **Recomendaciones Únicas:** {unique_count}")
            st.warning("⚠️ Nota: El número de registros puede ser mayor que el número de recomendaciones únicas debido a que algunas recomendaciones tienen múltiples atributos (ej. múltiples temas), generando filas duplicadas en el archivo original.")

            st.markdown("---") 
            
            # Helper to get deduplicated DF for strict counts
            # We use this for Treemaps and Single-Value Evolution
            df_viz_dedup = df_filtered_viz.drop_duplicates(subset=[id_col]).copy() 

            
            # --- Chart Logic (Updated to use df_filtered_viz) ---
            
            # --- Treemap 1: Dimension ---
            available_dims = sorted(df_filtered_viz['assigned_dimension'].unique())
            
            # We use a selectbox for explicit control (MOVED TO TOP)
            manual_dim = st.selectbox(
                "Filtrar Subdimensión por Dimensión (Treemaps):", 
                options=["Todos"] + available_dims, 
                index=0,
                key="manual_dim_filter_top"
            )
            
            selected_dim_label = None
            if manual_dim != "Todos":
                selected_dim_label = manual_dim
            
            # --- Treemap 1: Dimension ---
            st.markdown("#### Por Dimensión")
            st.caption("Selecciona una dimensión en el gráfico para ver detalles (opcional).")
            
            # USE DEDUPLICATED DATA FOR TREEMAPS
            dim_counts = df_viz_dedup['assigned_dimension'].value_counts().reset_index()
            dim_counts.columns = ['dimension', 'count']
            
            fig_dim = px.treemap(
                dim_counts,
                path=['dimension'],
                values='count',
                title='Recomendaciones Únicas por Dimensión',
                color='dimension'
            )
            fig_dim.update_traces(textinfo="label+value", textfont_size=20)
            
            selection_dim = st.plotly_chart(fig_dim, on_select="rerun", key="treemap_dim_select", use_container_width=True)
            
            if isinstance(selection_dim, dict) and "selection" in selection_dim and "points" in selection_dim["selection"]:
                 points = selection_dim["selection"]["points"]
                 if points:
                     pt = points[0]
                     clicked_label = pt.get('label') or pt.get('x') or pt.get('id')
                     if clicked_label and clicked_label in available_dims and manual_dim == "Todos":
                         selected_dim_label = clicked_label
                         st.info(f"Filtro aplicado por selección en gráfico: {selected_dim_label}")

            # --- Treemap 2: Subdim ---
            st.markdown("#### Por Subdimensión")
            
            # USE DEDUPLICATED DATA FOR TREEMAPS
            df_sub = df_viz_dedup.copy()
            title_suffix = ""
            
            if selected_dim_label:
                if selected_dim_label in df_sub['assigned_dimension'].unique():
                    df_sub = df_sub[df_sub['assigned_dimension'] == selected_dim_label]
                    title_suffix = f" ({selected_dim_label})"
            
            if not df_sub.empty:
                subdim_counts = df_sub['assigned_subdim'].value_counts().reset_index()
                subdim_counts.columns = ['subdim', 'count']
                
                fig_sub = px.treemap(
                    subdim_counts,
                    path=['subdim'],
                    values='count',
                    title=f'Recomendaciones Únicas por Subdimensión{title_suffix}',
                    color='subdim'
                )
                fig_sub.update_traces(textinfo="label+value", textfont_size=20)
                st.plotly_chart(fig_sub, use_container_width=True)
            else:
                st.info("No hay datos para mostrar.")

            st.markdown("---")
            
            # --- Evolution Plots (New) ---
            st.subheader("📈 Evolución Temporal")
            
            evo_toggle = st.radio("Tipo de Gráfico:", ["Absoluto", "Porcentaje (100%)"], horizontal=True, key="evo_chart_type")
            is_percent = (evo_toggle == "Porcentaje (100%)")
            bar_norm_val = 'percent' if is_percent else None
            
            # Helper to plot evolution
            def plot_evolution(df_in, cat_col, title_prefix):
                if 'Year' not in df_in.columns or df_in.empty:
                    st.info("No hay datos de Año para mostrar evolución.")
                    return
                    
                # Filter out bad years (0, nan)
                df_clean = df_in[df_in['Year'] > 1900].copy()
                if df_clean.empty:
                     st.info("No hay datos válidos de año.")
                     return

                # Check cardinality
                unique_vals = df_clean[cat_col].nunique()
                if unique_vals > 20: 
                     st.warning(f"Hay muchos valores únicos ({unique_vals}) en {cat_col}. Se muestran los Top 20.")
                     top_20 = df_clean[cat_col].value_counts().nlargest(20).index
                     df_clean = df_clean[df_clean[cat_col].isin(top_20)]

                # Group
                df_grouped = df_clean.groupby(['Year', cat_col]).size().reset_index(name='count')
                
                # Determine title suffix
                t_suffix = " (%)" if is_percent else ""
                
                # Create figure
                # We use 'relative' barmode. If is_percent is true, we update layout with barnorm='percent'
                fig = px.bar(
                    df_grouped, 
                    x="Year", 
                    y="count", 
                    color=cat_col, 
                    title=f"Evolución de {title_prefix}{t_suffix}",
                    barmode='relative', 
                )
                
                if is_percent:
                     fig.update_layout(barnorm='percent')
                
                st.plotly_chart(fig, use_container_width=True)
            
            
            # Define the tabs - Expanded List
            tab_labels = [
                "Por País", "Por Unidad Admin", "Por Dimensión", "Por Subdimensión",
                "Resp. Gestión", "Temática Eval.", "Temática Rec.", "Unidad Técnica",
                "Fuente Fondo", "Naturaleza", "Tipo Eval.", "Momento", "Progreso", "Tipo Doc."
            ]
            
            tabs = st.tabs(tab_labels)
            
            # 0. Por País
            with tabs[0]:
                 if 'Country(ies)' in df_viz_dedup.columns:
                     plot_evolution(df_viz_dedup, 'Country(ies)', "País")
                 else:
                     st.info("Columna 'Country(ies)' no disponible.")

            # 1. Por Unidad Admin
            with tabs[1]:
                 if 'Recommendation administrative unit' in df_viz_dedup.columns:
                     plot_evolution(df_viz_dedup, 'Recommendation administrative unit', "Unidad Administrativa")
                 else:
                     st.info("Columna 'Recommendation administrative unit' no disponible.")

            # 2. Por Dimensión
            with tabs[2]:
                 plot_evolution(df_viz_dedup, 'assigned_dimension', "Dimensión")

            # 3. Por Subdimensión
            with tabs[3]:
                 plot_evolution(df_viz_dedup, 'assigned_subdim', "Subdimensión")
                 
            # 4. Resp. Gestión (Management response) - Single value usually
            with tabs[4]:
                if 'Management response' in df_viz_dedup.columns:
                    plot_evolution(df_viz_dedup, 'Management response', "Respuesta de Gestión")
                else:
                    st.info("Columna 'Management response' no disponible.")
            
            # 5. Temática Eval. (Evaluation theme(s)) - MULTI VALUE (Keep duplicates to show frequency)
            with tabs[5]:
                if 'Evaluation theme(s)' in df_filtered_viz.columns:
                    plot_evolution(df_filtered_viz, 'Evaluation theme(s)', "Temática de Evaluación")
                else:
                    st.info("Columna 'Evaluation theme(s)' no disponible.")

            # 6. Temática Rec. (Recommendation theme) - MULTI VALUE (Keep duplicates)
            with tabs[6]:
                if 'Recommendation theme' in df_filtered_viz.columns:
                    plot_evolution(df_filtered_viz, 'Recommendation theme', "Temática de Recomendación")
                else:
                    st.info("Columna 'Recommendation theme' no disponible.")

            # 7. Unidad Técnica (Technical unit(s)) - MULTI VALUE ? (Likely yes if it has (s))
            with tabs[7]:
                if 'Technical unit(s)' in df_filtered_viz.columns:
                    plot_evolution(df_filtered_viz, 'Technical unit(s)', "Unidad Técnica")
                else:
                    st.info("Columna 'Technical unit(s)' no disponible.")

            # 8. Fuente Fondo (Funding source(s)) - MULTI VALUE
            with tabs[8]:
                if 'Funding source(s)' in df_filtered_viz.columns:
                    plot_evolution(df_filtered_viz, 'Funding source(s)', "Fuente de Financiamiento")
                else:
                    st.info("Columna 'Funding source(s)' no disponible.")

            # 9. Naturaleza (Evaluation nature) - Single
            with tabs[9]:
                if 'Evaluation nature' in df_viz_dedup.columns:
                    plot_evolution(df_viz_dedup, 'Evaluation nature', "Naturaleza de Evaluación")
                else:
                    st.info("Columna 'Evaluation nature' no disponible.")
            
            # 10. Tipo Eval. (Evaluation type) - Single
            with tabs[10]:
                if 'Evaluation type' in df_viz_dedup.columns:
                    plot_evolution(df_viz_dedup, 'Evaluation type', "Tipo de Evaluación")
                else:
                    st.info("Columna 'Evaluation type' no disponible.")

            # 11. Momento (Evaluation timing) - Single
            with tabs[11]:
                if 'Evaluation timing' in df_viz_dedup.columns:
                    plot_evolution(df_viz_dedup, 'Evaluation timing', "Momento de Evaluación")
                else:
                    st.info("Columna 'Evaluation timing' no disponible.")

            # 12. Progreso (Progress) - Single
            with tabs[12]:
                if 'Progress' in df_viz_dedup.columns:
                    plot_evolution(df_viz_dedup, 'Progress', "Progreso")
                else:
                    st.info("Columna 'Progress' no disponible.")

            # 13. Tipo Doc. (Evaluation document type) - Single
            with tabs[13]:
                if 'Evaluation document type' in df_viz_dedup.columns:
                    plot_evolution(df_viz_dedup, 'Evaluation document type', "Tipo de Documento")
                else:
                    st.info("Columna 'Evaluation document type' no disponible.")

            # --- SECCION 3: HERRAMIENTAS AVANZADAS DE IA (NUEVA) ---
            st.markdown("---")
            st.subheader("🤖 3. Herramientas Avanzadas de IA")
            st.info("Utiliza herramientas de Inteligencia Artificial para analizar en profundidad o resumir las recomendaciones filtradas.")

            # --- Shared Filters for AI Tools ---
            st.markdown("##### 1. Definir Subconjunto para Análisis")
            
            df_ai_base = df_filtered_viz.copy()
            ai_filters = {}
            
            with st.expander("🔎 Filtros Globales para Análisis AI", expanded=True):
                 c_ai1, c_ai2 = st.columns(2)
                 
                 with c_ai1:
                      if 'Region(s)' in df_viz.columns:
                          opts = sorted([str(x) for x in df_viz['Region(s)'].unique() if pd.notna(x)])
                          ai_filters['Region(s)'] = st.multiselect("Región:", opts, key="ai_region")
                      
                      if 'Country(ies)' in df_viz.columns:
                           opts = sorted([str(x) for x in df_viz['Country(ies)'].unique() if pd.notna(x)])
                           ai_filters['Country(ies)'] = st.multiselect("País:", opts, key="ai_country")
                      
                      if 'Evaluation theme(s)' in df_viz.columns:
                           opts = sorted([str(x) for x in df_viz['Evaluation theme(s)'].unique() if pd.notna(x)])
                           ai_filters['Evaluation theme(s)'] = st.multiselect("Temática Eval:", opts, key="ai_eval_theme")

                 with c_ai2:
                      if 'Year' in df_viz.columns:
                          opts = sorted([int(x) for x in df_viz['Year'].unique() if pd.notna(x)])
                          ai_filters['Year'] = st.multiselect("Año:", opts, key="ai_year")
                      
                      if 'Management response' in df_viz.columns:
                          opts = sorted([str(x) for x in df_viz['Management response'].unique() if pd.notna(x)])
                          ai_filters['Management response'] = st.multiselect("Resp. Gestión:", opts, key="ai_mgmt")
                      
                      if 'assigned_dimension' in df_viz.columns:
                          opts = sorted([str(x) for x in df_viz['assigned_dimension'].unique() if pd.notna(x)])
                          ai_filters['assigned_dimension'] = st.multiselect("Dimensión:", opts, key="ai_dim")

            # Apply Filters
            for col, vals in ai_filters.items():
                if vals and col in df_ai_base.columns:
                    df_ai_base = df_ai_base[df_ai_base[col].isin(vals)]
            
            st.write(f"**Total Registros Filtrados:** {len(df_ai_base)}")
            
            st.markdown("---")
            
            # --- Dual Action Buttons ---
            col_deep, col_summ = st.columns(2)
            
            # --- LEFT: Deep Analysis ---
            with col_deep:
                st.markdown("#### 🧠 Análisis Profundo")
                st.caption("Evalúa coherencia, calidad e innovación de planes de acción.")
                
                # Prepare Deep Data
                df_deep_ready = df_ai_base.copy()
                if 'Action plan' in df_deep_ready.columns:
                    df_deep_ready = df_deep_ready[df_deep_ready['Action plan'].notna() & (df_deep_ready['Action plan'].astype(str).str.strip() != "")]
                valid_deep_count = len(df_deep_ready)
                
                st.metric("Válidos (con Plan)", valid_deep_count)
                
                with st.expander("Configuración"):
                    deep_model = st.selectbox("Modelo:", ["gpt-4o-mini", "gpt-4o"], index=0, key="deep_model_sel")
                    limit_rows_deep = st.number_input("Límite (0=Todos):", min_value=0, value=0, step=10, key="deep_limit")
                
                if st.button("🚀 Iniciar Análisis", key="btn_run_deep"):
                     if df_deep_ready.empty:
                         st.warning("No hay datos válidos.")
                     else:
                         id_col_deep = 'Recommendation ID' if 'Recommendation ID' in df_deep_ready.columns else 'Recommendation description'
                         df_deep_input = df_deep_ready.drop_duplicates(subset=[id_col_deep]).copy()
                         if limit_rows_deep > 0:
                             df_deep_input = df_deep_input.head(limit_rows_deep)
                        
                         st.info(f"Procesando all {len(df_deep_input)} recomendaciones..." if limit_rows_deep == 0 else f"Procesando {len(df_deep_input)} recomendaciones...")
                         
                         analysis_cache = AnalysisCache()
                         args_list = []
                         for idx, row in df_deep_input.iterrows():
                             rec = str(row.get('Recommendation description', ''))
                             plan = str(row.get('Action plan', ''))
                             comments = str(row.get('Comments', '')) if 'Comments' in row else ""
                             args_list.append((idx, rec, plan, comments, openai_api_key, analysis_cache))
                         
                         results_map = {}
                         pbar_deep = st.progress(0, text="Analizando...")
                         completed_count = 0
                         
                         with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                             futures = [executor.submit(run_row_analysis, arg) for arg in args_list]
                             for future in concurrent.futures.as_completed(futures):
                                 idx, result, _ = future.result()
                                 if result:
                                    results_map[idx] = result
                                 completed_count += 1
                                 # Update progress every time, but with higher concurrency it will feel 'batched'
                                 pbar_deep.progress(completed_count / len(args_list), text=f"Analizando... {completed_count}/{len(args_list)}")
                         
                         pbar_deep.empty()
                         analysis_cache.save()
                         
                         analysis_list = []
                         for idx, res in results_map.items():
                             res['original_index'] = idx
                             analysis_list.append(res)
                         
                         if analysis_list:
                             df_res = pd.DataFrame(analysis_list)
                             df_res.set_index('original_index', inplace=True)
                             df_final_deep = df_deep_input.join(df_res)
                             
                             list_cols = ['extracted_actions_from_rec', 'actions_proposed_in_plan', 'rejection_difficulty_classification', 'tags', 'rec_additional_tags']
                             for col in list_cols:
                                 if col in df_final_deep.columns:
                                     df_final_deep[col] = df_final_deep[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
                             
                             st.session_state['deep_analysis_df'] = df_final_deep
                             st.success("¡Análisis completado!")

            # --- RIGHT: Summary Generation ---
            with col_summ:
                st.markdown("#### ✨ Resumen Ejecutivo")
                st.caption("Genera una síntesis narrativa de los hallazgos.")
                
                st.metric("Total a Resumir", len(df_ai_base))
                
                with st.expander("Configuración"):
                    max_tokens_val = st.slider("Longitud (Tokens):", 500, 4000, 3000, 100, key="summ_len")
                
                if st.button("📝 Generar Resumen Global", key="btn_run_summ"):
                    if df_ai_base.empty:
                        st.warning("No hay datos.")
                    else:
                        with st.spinner("Generando resumen..."):
                            id_col_summ = 'Recommendation ID' if 'Recommendation ID' in df_ai_base.columns else 'Recommendation description'
                            df_summ_unique = df_ai_base.drop_duplicates(subset=[id_col_summ])
                            
                            text_list = []
                            for idx, row in df_summ_unique.iterrows():
                                 desc = str(row.get('Recommendation description', ''))
                                 mgmt = str(row.get('Management response', '')) if 'Management response' in row else "N/A"
                                 comments = str(row.get('Comments', '')) if 'Comments' in row else "N/A"
                                 action = str(row.get('Action plan', '')) if 'Action plan' in row else "N/A"
                                 text_list.append(f"--- Rec ---\nDesc: {desc}\nResp: {mgmt}\nCom: {comments}\nPlan: {action}\n")
                            
                            full_text = "\n".join(text_list)
                            summary_text = generate_executive_summary(full_text, max_output_tokens=max_tokens_val)
                            st.session_state['summary_result'] = summary_text
                            st.success("¡Resumen generado!")

            # --- Display Results Areas (Full Width) ---
            
            # 1. Deep Analysis Results
            if 'deep_analysis_df' in st.session_state:
                st.markdown("---")
                st.subheader("📊 Resultados: Análisis Profundo")
                df_final_deep = st.session_state['deep_analysis_df']
                
                c_m1, c_m2 = st.columns(2)
                c_m1.metric("Coherencia Prom.", f"{df_final_deep['coherence_score'].mean():.2f}")
                c_m2.metric("Calidad Plan Prom.", f"{df_final_deep['plan_quality_score'].mean():.2f}")
                
                st.dataframe(df_final_deep[['Recommendation description', 'coherence_score', 'plan_quality_score', 'rec_innovation_score', 'rejection_difficulty_classification', 'tags']], use_container_width=True)
                
                # Export Deep
                out_deep = BytesIO()
                # Drop columns AE:AT (indices 30-45) and AW:BG (indices 48-58) before export
                _cols_to_drop_deep = []
                if len(df_final_deep.columns) > 30:
                    _cols_to_drop_deep += list(df_final_deep.columns[30:min(46, len(df_final_deep.columns))])
                if len(df_final_deep.columns) > 48:
                    _cols_to_drop_deep += list(df_final_deep.columns[48:min(59, len(df_final_deep.columns))])
                df_final_deep_export = df_final_deep.drop(columns=_cols_to_drop_deep, errors='ignore')
                with pd.ExcelWriter(out_deep, engine='xlsxwriter') as writer:
                    df_final_deep_export.to_excel(writer, index=False, sheet_name='Analisis_Profundo')
                
                st.download_button("📥 Descargar Reporte Análisis (.xlsx)", out_deep.getvalue(), "analisis_profundo.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # Visualizations Deep
                st.markdown("##### 📈 Distribución de Métricas Clave")
                
                col_v1, col_v2 = st.columns(2)
                
                if 'coherence_score' in df_final_deep.columns:
                    with col_v1:
                        fig_d1 = px.histogram(df_final_deep, x="coherence_score", nbins=10, title="Distribución de Coherencia", color_discrete_sequence=['#636EFA'])
                        st.plotly_chart(fig_d1, use_container_width=True)
                
                if 'plan_quality_score' in df_final_deep.columns:
                     with col_v2:
                        fig_d2 = px.histogram(df_final_deep, x="plan_quality_score", nbins=10, title="Distribución de Calidad del Plan", color_discrete_sequence=['#EF553B'])
                        st.plotly_chart(fig_d2, use_container_width=True)
                
                col_v3, col_v4 = st.columns(2)
                
                if 'attention_level_score' in df_final_deep.columns:
                     with col_v3:
                        fig_d3 = px.histogram(df_final_deep, x="attention_level_score", nbins=10, title="Nivel de Atención", color_discrete_sequence=['#00CC96'])
                        st.plotly_chart(fig_d3, use_container_width=True)

                if 'rec_innovation_score' in df_final_deep.columns:
                     with col_v4:
                         # Categorical
                         fig_pie = px.pie(df_final_deep, names='rec_innovation_score', title="Nivel de Innovación", hole=0.3)
                         st.plotly_chart(fig_pie, use_container_width=True)
                
                st.markdown("##### 🚦 Factibilidad e Impacto")
                col_v5, col_v6 = st.columns(2)
                
                if 'rec_operational_feasibility' in df_final_deep.columns:
                    with col_v5:
                        fig_feas = px.histogram(df_final_deep, x='rec_operational_feasibility', title="Factibilidad Operativa", color='rec_operational_feasibility')
                        st.plotly_chart(fig_feas, use_container_width=True)
                
                if 'rec_expected_impact' in df_final_deep.columns:
                    with col_v6:
                         fig_imp = px.histogram(df_final_deep, x='rec_expected_impact', title="Impacto Esperado", color='rec_expected_impact')
                         st.plotly_chart(fig_imp, use_container_width=True)

            # 2. Summary Results
            if 'summary_result' in st.session_state:
                st.markdown("---")
                st.subheader("📄 Resultados: Resumen Ejecutivo (Results: Executive Summary)")
                st.markdown(st.session_state['summary_result'])
                
                # Export Summary
                out_summ = BytesIO()
                with pd.ExcelWriter(out_summ, engine='xlsxwriter') as writer:
                    # Save filtering data
                    df_ai_base.to_excel(writer, index=False, sheet_name='Datos Base')
                    pd.DataFrame({'Resumen': [st.session_state['summary_result']]}).to_excel(writer, index=False, sheet_name='Resumen')
                
                st.download_button("📥 Descargar Resumen + Datos (.xlsx)", out_summ.getvalue(), "resumen_ejecutivo.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab2:
    st.header("Recommendation Classification")
    st.info("""
    **🌍 Intelligent Recommendation Classification**

    This tool automatically classifies the recommendations contained in the `Recommendations_World.xlsx` file,
    assigning them to the subdimensions defined in the `Frame_Recommendations_English.xlsx` framework.

    **Optimizations:**
    1.  **Pre-filtering:** Filter by region/country before processing to save costs and time.
    2.  **Persistent Cache:** Embeddings are saved locally to avoid regenerating them in future runs.
    3.  **Deduplication:** Identical texts are processed only once.

    **How it works:**
    1.  Loads recommendation data and the reference framework.
    2.  Calculates semantic similarity (using OpenAI Embeddings) between each recommendation and subdimension definitions.
    3.  Assigns the most relevant subdimension.
    4.  Visualizes the distribution using interactive Treemaps.
    """)

    # Reuse EmbeddingsCache and load_world_data defined in tab5
    if 'embeddings_cache_obj' not in st.session_state:
        st.session_state['embeddings_cache_obj'] = EmbeddingsCache()

    # --- Data Source Selection ---
    st.markdown("### Data Source")
    data_source_en = st.radio("Select recommendations source:",
                          ["Default Files (World)", "Upload Custom File (.xlsx)"],
                          horizontal=True,
                          key="data_source_radio_en")

    rec_df_world_en = None
    frame_df_world_en = None

    _, frame_default_en = load_world_data()
    frame_df_world_en = frame_default_en

    if data_source_en == "Default Files (World)":
        rec_default_en, _ = load_world_data()
        rec_df_world_en = rec_default_en

        if rec_df_world_en is None:
             st.warning("⚠️ Default file 'Recommendations_World.xlsx' not found.")

    else:
        uploaded_file_en = st.file_uploader("Upload your recommendations file (Excel):", type=["xlsx"], key="custom_rec_upload_en")
        if uploaded_file_en:
            try:
                rec_df_world_en = pd.read_excel(uploaded_file_en)
                if 'Recommendation description' not in rec_df_world_en.columns:
                    st.error("❌ The file must contain a column named 'Recommendation description'.")
                    rec_df_world_en = None
                else:
                    st.success(f"File loaded successfully: {len(rec_df_world_en)} rows.")
            except Exception as e:
                st.error(f"Error reading the file: {e}")
        else:
            st.info("Waiting for file...")

    if rec_df_world_en is None or frame_df_world_en is None:
        if frame_df_world_en is None:
             st.warning("⚠️ File 'Frame_Recommendations_English.xlsx' not found. It is required for classification.")
    else:

        # --- DATA PREP: Extract Year ---
        if 'Recommendation date' in rec_df_world_en.columns:
            rec_df_world_en['Recommendation date'] = pd.to_datetime(rec_df_world_en['Recommendation date'], errors='coerce')
            rec_df_world_en['Year'] = rec_df_world_en['Recommendation date'].dt.year

        # --- Filters (PRE-PROCESSING) ---
        st.markdown("### 1. Selection Filters")
        st.caption("Select filters to define the data subset to classify. This optimizes processing time and cost.")

        filters_en = {}

        # --- Group 1: Location and Time ---
        with st.expander("📍 Location and Time", expanded=True):
            col_grp1_1_en, col_grp1_2_en = st.columns(2)

            with col_grp1_1_en:
                regions_en = sorted([str(x) for x in rec_df_world_en['Region(s)'].unique() if not pd.isna(x)]) if 'Region(s)' in rec_df_world_en.columns else []
                filters_en['Region(s)'] = st.multiselect("Region(s):", options=regions_en, default=[], key="tab6en_region_filter")

                if 'Recommendation administrative unit' in rec_df_world_en.columns:
                    admin_units_en = sorted([str(x) for x in rec_df_world_en['Recommendation administrative unit'].unique() if not pd.isna(x)])
                    filters_en['Recommendation administrative unit'] = st.multiselect("Administrative Unit:", options=admin_units_en, default=[], key="tab6en_admin_unit")

            with col_grp1_2_en:
                available_countries_df_en = rec_df_world_en.copy()
                if filters_en['Region(s)']:
                     if 'Region(s)' in available_countries_df_en.columns:
                        available_countries_df_en = available_countries_df_en[available_countries_df_en['Region(s)'].isin(filters_en['Region(s)'])]

                countries_en = sorted([str(x) for x in available_countries_df_en['Country(ies)'].unique() if not pd.isna(x)]) if 'Country(ies)' in available_countries_df_en.columns else []
                filters_en['Country(ies)'] = st.multiselect("Country(ies):", options=countries_en, default=[], key="tab6en_country_filter")

                if 'Year' in rec_df_world_en.columns:
                    years_en = sorted([int(x) for x in rec_df_world_en['Year'].unique() if not pd.isna(x)])
                    filters_en['Year'] = st.multiselect("Year:", options=years_en, default=[], key="tab6en_year_filter")

        # --- Group 2: Theme and Technical ---
        with st.expander("📚 Theme and Technical", expanded=False):
            col_grp2_1_en, col_grp2_2_en = st.columns(2)

            with col_grp2_1_en:
                if 'Evaluation theme(s)' in rec_df_world_en.columns:
                    themes_en = sorted([str(x) for x in rec_df_world_en['Evaluation theme(s)'].unique() if not pd.isna(x)])
                    filters_en['Evaluation theme(s)'] = st.multiselect("Evaluation Theme:", options=themes_en, default=[], key="tab6en_eval_theme")

                if 'Recommendation theme' in rec_df_world_en.columns:
                    rec_themes_en = sorted([str(x) for x in rec_df_world_en['Recommendation theme'].unique() if not pd.isna(x)])
                    filters_en['Recommendation theme'] = st.multiselect("Recommendation Theme:", options=rec_themes_en, default=[], key="tab6en_rec_theme")

                if 'Technical unit(s)' in rec_df_world_en.columns:
                    tech_units_en = sorted([str(x) for x in rec_df_world_en['Technical unit(s)'].unique() if not pd.isna(x)])
                    filters_en['Technical unit(s)'] = st.multiselect("Technical Unit:", options=tech_units_en, default=[], key="tab6en_tech_unit")

            with col_grp2_2_en:
                if 'Funding source(s)' in rec_df_world_en.columns:
                    fundings_en = sorted([str(x) for x in rec_df_world_en['Funding source(s)'].unique() if not pd.isna(x)])
                    filters_en['Funding source(s)'] = st.multiselect("Funding Source:", options=fundings_en, default=[], key="tab6en_funding")

                if 'Evaluation nature' in rec_df_world_en.columns:
                    natures_en = sorted([str(x) for x in rec_df_world_en['Evaluation nature'].unique() if not pd.isna(x)])
                    filters_en['Evaluation nature'] = st.multiselect("Evaluation Nature:", options=natures_en, default=[], key="tab6en_eval_nature")

                if 'Evaluation type' in rec_df_world_en.columns:
                    types_en = sorted([str(x) for x in rec_df_world_en['Evaluation type'].unique() if not pd.isna(x)])
                    filters_en['Evaluation type'] = st.multiselect("Evaluation Type:", options=types_en, default=[], key="tab6en_eval_type")

        # --- Group 3: Management and Response ---
        with st.expander("⚙️ Management and Response", expanded=False):
             col_grp3_1_en, col_grp3_2_en = st.columns(2)

             with col_grp3_1_en:
                 if 'Evaluation timing' in rec_df_world_en.columns:
                    timings_en = sorted([str(x) for x in rec_df_world_en['Evaluation timing'].unique() if not pd.isna(x)])
                    filters_en['Evaluation timing'] = st.multiselect("Evaluation Timing:", options=timings_en, default=[], key="tab6en_eval_timing")

                 if 'Progress' in rec_df_world_en.columns:
                    progresses_en = sorted([str(x) for x in rec_df_world_en['Progress'].unique() if not pd.isna(x)])
                    filters_en['Progress'] = st.multiselect("Progress:", options=progresses_en, default=[], key="tab6en_progress")

             with col_grp3_2_en:
                 if 'Management response' in rec_df_world_en.columns:
                     responses_en = sorted([str(x) for x in rec_df_world_en['Management response'].unique() if not pd.isna(x)])
                     filters_en['Management response'] = st.multiselect("Management Response:", options=responses_en, default=[], key="tab6en_mgmt_response")

                 if 'Evaluation document type' in rec_df_world_en.columns:
                     doc_types_en = sorted([str(x) for x in rec_df_world_en['Evaluation document type'].unique() if not pd.isna(x)])
                     filters_en['Evaluation document type'] = st.multiselect("Document Type:", options=doc_types_en, default=[], key="tab6en_doc_type")

        # Apply ALL filters
        start_count_en = len(rec_df_world_en)
        filtered_rec_df_en = rec_df_world_en.copy()

        for col_name_en, selected_values_en in filters_en.items():
            if selected_values_en and col_name_en in filtered_rec_df_en.columns:
                filtered_rec_df_en = filtered_rec_df_en[filtered_rec_df_en[col_name_en].isin(selected_values_en)]

        end_count_en = len(filtered_rec_df_en)

        # Show row count AND unique-recommendation count: source rows repeat for
        # multi-attribute records (themes, sources, etc.). Embedding cost is driven
        # by uniques, so analysts need both numbers before clicking process.
        id_col_pre_en = 'Recommendation ID' if 'Recommendation ID' in filtered_rec_df_en.columns else 'Recommendation description'
        unique_count_pre_en = filtered_rec_df_en[id_col_pre_en].nunique()
        total_unique_pre_en = rec_df_world_en[id_col_pre_en].nunique()
        st.markdown(
            f"**Selected records:** {end_count_en} of {start_count_en}  |  "
            f"**Unique recommendations:** {unique_count_pre_en} of {total_unique_pre_en}"
        )
        st.caption("⚠️ The number of records may be higher than the number of unique recommendations because some recommendations have multiple attributes (e.g. multiple themes), generating duplicate rows in the original file. Classification cost is driven by unique recommendations.")

        # --- Classification Logic ---

        if 'classified_world_df_en' not in st.session_state:
            st.session_state['classified_world_df_en'] = None

        def classify_recommendations_en(target_df, reference_df, cache_obj, use_llm_verification=False):
            """
            Classifies recommendations in target_df based on similarity to texts in reference_df.
            Uses persistent cache and deduplication.
            """

            progress_bar = st.progress(0, text="Starting process...")

            # 1. Embed Reference Frame
            ref_embeddings = []
            ref_metadata = []

            if 'texto_merged' not in reference_df.columns:
                st.error("The Frame file must have the column 'texto_merged'.")
                return None

            for idx, row in reference_df.iterrows():
                text = str(row['texto_merged'])

                emb = cache_obj.get(text)
                if emb is None:
                    emb = get_embedding_with_retry(text)
                    if emb is not None:
                        cache_obj.set(text, emb)

                if emb is not None:
                    ref_embeddings.append(emb)
                    ref_metadata.append({
                        'dimension': row.get('dimension', 'Unknown'),
                        'subdim': row.get('subdim', 'Unknown'),
                        'texto_merged': text
                    })

                progress_val = 0.1 * (idx + 1) / len(reference_df)
                progress_bar.progress(progress_val, text=f"Processing reference framework {idx+1}/{len(reference_df)}")

            cache_obj.save()

            if not ref_embeddings:
                st.error("Could not generate embeddings for the reference framework.")
                return None

            ref_embeddings = np.array(ref_embeddings)

            ref_norms = np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
            ref_embeddings_norm = ref_embeddings / (ref_norms + 1e-10)

            # 2. Embed Unique Recommendations (Deduplication)
            start_time = time.time()
            total_recs = len(target_df)

            unique_texts = target_df['Recommendation description'].dropna().unique()
            unique_embeddings = {}

            new_embeddings_count = 0

            BATCH_SIZE = 20
            for i in range(0, len(unique_texts), BATCH_SIZE):
                batch_texts = unique_texts[i:i+BATCH_SIZE]
                for text in batch_texts:
                    text_str = str(text)
                    emb = cache_obj.get(text_str)
                    if emb is None:
                        emb = get_embedding_with_retry(text_str)
                        if emb is not None:
                            cache_obj.set(text_str, emb)
                            new_embeddings_count += 1

                    if emb is not None:
                        unique_embeddings[text_str] = emb

                progress_pct = 0.1 + (0.4 * (i + len(batch_texts)) / len(unique_texts))
                progress_bar.progress(progress_pct, text=f"Generating unique embeddings: {min(i+len(batch_texts), len(unique_texts))}/{len(unique_texts)}")

                if new_embeddings_count > 50:
                    cache_obj.save()
                    new_embeddings_count = 0

            cache_obj.save()

        # 3. Classify Rows (Match) - Top 3
            classified_rows = []

            progress_bar.progress(0.5, text="Assigning classifications (Top 3) " + ("and verifying with LLM..." if use_llm_verification else "..."))

            for i, (idx, row) in enumerate(target_df.iterrows()):
                rec_text = str(row.get('Recommendation description', ''))

                vals = {
                    'assigned_dimension': 'Unclassified',
                    'assigned_subdim': 'Unclassified',
                    'matched_frame_text': '',
                    'similarity_score': 0.0,
                    'Otras dimensiones': '',
                    'Otras subdimensiones': ''
                }

                if rec_text in unique_embeddings:
                    rec_emb = unique_embeddings[rec_text]

                    rec_emb_norm = rec_emb / (np.linalg.norm(rec_emb) + 1e-10)
                    similarities = np.dot(ref_embeddings_norm, rec_emb_norm)

                    if use_llm_verification:
                        top_n_candidates = 10
                    else:
                        top_n_candidates = 3

                    top_k_indices = np.argsort(similarities)[-top_n_candidates:][::-1]

                    best_idx = top_k_indices[0]

                    if use_llm_verification:
                        candidates = []
                        for k in top_k_indices:
                             candidates.append(ref_metadata[k])

                        verified_idx = verify_match_with_llm(rec_text, candidates, openai_api_key)

                        if verified_idx >= 0 and verified_idx < len(top_k_indices):
                            best_idx = top_k_indices[verified_idx]

                    best_match = ref_metadata[best_idx]

                    vals['assigned_dimension'] = best_match['dimension']
                    vals['assigned_subdim'] = best_match['subdim']
                    vals['matched_frame_text'] = best_match['texto_merged']
                    vals['similarity_score'] = similarities[best_idx]

                    secondary_dims = []
                    secondary_subdims = []

                    similarity_threshold = 0.60

                    for k_idx in top_k_indices:
                        if k_idx == best_idx: continue

                        if similarities[k_idx] >= similarity_threshold:
                            match_k = ref_metadata[k_idx]
                            secondary_dims.append(match_k['dimension'])
                            secondary_subdims.append(match_k['subdim'])

                    vals['Otras dimensiones'] = "; ".join(secondary_dims)
                    vals['Otras subdimensiones'] = "; ".join(secondary_subdims)

                row_dict = row.to_dict()
                row_dict.update(vals)
                classified_rows.append(row_dict)

                if i % 10 == 0:
                     progress_bar.progress(0.5 + (0.5 * (i + 1) / total_recs), text=f"Classifying records {i+1}/{total_recs}")

            progress_bar.empty()
            return pd.DataFrame(classified_rows)

        # Button to run classification
        st.markdown("### 2. Execution")

        use_llm_chk_en = st.checkbox("✅ Use LLM Verification (High Precision, Slower)", value=False, help="If enabled, the system will use AI to read definitions and correct classification (e.g. distinguish financial 'Pensions' from policy 'Pensions').", key="tab6en_llm_chk")

        if start_count_en == 0:
            st.warning("The recommendations file is empty.")
        elif end_count_en == 0:
             st.warning("No records match the selected filters. Adjust the filters.")
        else:
            if st.button("🚀 Start Classification", key="start_classification_tab6en"):
                with st.spinner("Running optimized classification..."):
                    classified_df_en = classify_recommendations_en(filtered_rec_df_en, frame_df_world_en, st.session_state['embeddings_cache_obj'], use_llm_verification=use_llm_chk_en)
                    if classified_df_en is not None and not classified_df_en.empty:
                        st.session_state['classified_world_df_en'] = classified_df_en
                        st.success(f"Classification complete! {len(classified_df_en)} recommendations processed.")
                    else:
                        st.error("There was a problem during classification.")

        # --- Visualization ---
        df_viz_en = st.session_state['classified_world_df_en']

        if df_viz_en is not None:
            st.markdown("---")
            st.subheader("📊 Results Visualization")

            # --- Visual Filters (Post-Classification) ---
            st.markdown("### 🔎 Visual Filters")
            st.warning("These filters only affect the charts and tables below, without re-running the classification.")

            df_filtered_viz_en = df_viz_en.copy()

            # 1. Year Slider (Range)
            if 'Year' in df_viz_en.columns:
                 valid_years_en = sorted([int(x) for x in df_viz_en['Year'].unique() if pd.notna(x) and x > 1900])
                 if valid_years_en:
                     min_year_en, max_year_en = min(valid_years_en), max(valid_years_en)
                     year_range_en = st.slider(
                         "Year Range:",
                         min_value=min_year_en,
                         max_value=max_year_en,
                         value=(min_year_en, max_year_en),
                         key="viz_year_slider_en"
                     )
                     df_filtered_viz_en = df_filtered_viz_en[
                         (df_filtered_viz_en['Year'] >= year_range_en[0]) &
                         (df_filtered_viz_en['Year'] <= year_range_en[1])
                     ]

            # 2. Metadata Filters (multiselects) - Grouped
            with st.expander("More Visual Filters (Metadata)", expanded=False):
                col_vf1_en, col_vf2_en, col_vf3_en = st.columns(3)

                with col_vf1_en:
                    if 'Recommendation administrative unit' in df_filtered_viz_en.columns:
                         opts = sorted([str(x) for x in df_viz_en['Recommendation administrative unit'].unique() if pd.notna(x)])
                         sel_admin_en = st.multiselect("Admin Unit:", options=opts, key="viz_admin_unit_en")
                         if sel_admin_en:
                             df_filtered_viz_en = df_filtered_viz_en[df_filtered_viz_en['Recommendation administrative unit'].isin(sel_admin_en)]

                with col_vf2_en:
                    if 'Country(ies)' in df_filtered_viz_en.columns:
                         opts = sorted([str(x) for x in df_viz_en['Country(ies)'].unique() if pd.notna(x)])
                         sel_country_en = st.multiselect("Country:", options=opts, key="viz_country_en")
                         if sel_country_en:
                             df_filtered_viz_en = df_filtered_viz_en[df_filtered_viz_en['Country(ies)'].isin(sel_country_en)]

                with col_vf3_en:
                    opts = sorted([str(x) for x in df_viz_en['assigned_dimension'].unique() if pd.notna(x)])
                    sel_dim_global_en = st.multiselect("Dimension:", options=opts, key="viz_dim_global_en")
                    if sel_dim_global_en:
                         df_filtered_viz_en = df_filtered_viz_en[df_filtered_viz_en['assigned_dimension'].isin(sel_dim_global_en)]

            # Count Unique Recommendations
            id_col_en = 'Recommendation ID' if 'Recommendation ID' in df_filtered_viz_en.columns else 'Recommendation description'
            unique_count_en = df_filtered_viz_en[id_col_en].nunique()

            st.markdown(f"**Displayed records:** {len(df_filtered_viz_en)} | **Unique Recommendations:** {unique_count_en}")
            st.warning("⚠️ Note: The number of records may be higher than the number of unique recommendations because some recommendations have multiple attributes (e.g. multiple themes), generating duplicate rows in the original file.")

            st.markdown("---")

            df_viz_dedup_en = df_filtered_viz_en.drop_duplicates(subset=[id_col_en]).copy()

            # --- Treemap 1: Dimension ---
            available_dims_en = sorted(df_filtered_viz_en['assigned_dimension'].unique())

            manual_dim_en = st.selectbox(
                "Filter Subdimension by Dimension (Treemaps):",
                options=["All"] + available_dims_en,
                index=0,
                key="manual_dim_filter_top_en"
            )

            selected_dim_label_en = None
            if manual_dim_en != "All":
                selected_dim_label_en = manual_dim_en

            # --- Treemap 1: Dimension ---
            st.markdown("#### By Dimension")
            st.caption("Select a dimension in the chart to see details (optional).")

            dim_counts_en = df_viz_dedup_en['assigned_dimension'].value_counts().reset_index()
            dim_counts_en.columns = ['dimension', 'count']

            fig_dim_en = px.treemap(
                dim_counts_en,
                path=['dimension'],
                values='count',
                title='Unique Recommendations by Dimension',
                color='dimension'
            )
            fig_dim_en.update_traces(textinfo="label+value", textfont_size=20)

            selection_dim_en = st.plotly_chart(fig_dim_en, on_select="rerun", key="treemap_dim_select_en", use_container_width=True)

            if isinstance(selection_dim_en, dict) and "selection" in selection_dim_en and "points" in selection_dim_en["selection"]:
                 points_en = selection_dim_en["selection"]["points"]
                 if points_en:
                     pt_en = points_en[0]
                     clicked_label_en = pt_en.get('label') or pt_en.get('x') or pt_en.get('id')
                     if clicked_label_en and clicked_label_en in available_dims_en and manual_dim_en == "All":
                         selected_dim_label_en = clicked_label_en
                         st.info(f"Filter applied by chart selection: {selected_dim_label_en}")

            # --- Treemap 2: Subdim ---
            st.markdown("#### By Subdimension")

            df_sub_en = df_viz_dedup_en.copy()
            title_suffix_en = ""

            if selected_dim_label_en:
                if selected_dim_label_en in df_sub_en['assigned_dimension'].unique():
                    df_sub_en = df_sub_en[df_sub_en['assigned_dimension'] == selected_dim_label_en]
                    title_suffix_en = f" ({selected_dim_label_en})"

            if not df_sub_en.empty:
                subdim_counts_en = df_sub_en['assigned_subdim'].value_counts().reset_index()
                subdim_counts_en.columns = ['subdim', 'count']

                fig_sub_en = px.treemap(
                    subdim_counts_en,
                    path=['subdim'],
                    values='count',
                    title=f'Unique Recommendations by Subdimension{title_suffix_en}',
                    color='subdim'
                )
                fig_sub_en.update_traces(textinfo="label+value", textfont_size=20)
                st.plotly_chart(fig_sub_en, use_container_width=True)
            else:
                st.info("No data to display.")

            st.markdown("---")

            # --- Evolution Plots ---
            st.subheader("📈 Temporal Evolution")

            evo_toggle_en = st.radio("Chart Type:", ["Absolute", "Percentage (100%)"], horizontal=True, key="evo_chart_type_en")
            is_percent_en = (evo_toggle_en == "Percentage (100%)")

            def plot_evolution_en(df_in, cat_col, title_prefix):
                if 'Year' not in df_in.columns or df_in.empty:
                    st.info("No Year data to show evolution.")
                    return

                df_clean = df_in[df_in['Year'] > 1900].copy()
                if df_clean.empty:
                     st.info("No valid year data.")
                     return

                unique_vals = df_clean[cat_col].nunique()
                if unique_vals > 20:
                     st.warning(f"Too many unique values ({unique_vals}) in {cat_col}. Showing Top 20.")
                     top_20 = df_clean[cat_col].value_counts().nlargest(20).index
                     df_clean = df_clean[df_clean[cat_col].isin(top_20)]

                df_grouped = df_clean.groupby(['Year', cat_col]).size().reset_index(name='count')

                t_suffix = " (%)" if is_percent_en else ""

                fig = px.bar(
                    df_grouped,
                    x="Year",
                    y="count",
                    color=cat_col,
                    title=f"Evolution of {title_prefix}{t_suffix}",
                    barmode='relative',
                )

                if is_percent_en:
                     fig.update_layout(barnorm='percent')

                st.plotly_chart(fig, use_container_width=True)

            # Define the tabs - Expanded List
            tab_labels_en = [
                "By Country", "By Admin Unit", "By Dimension", "By Subdimension",
                "Mgmt. Response", "Eval. Theme", "Rec. Theme", "Technical Unit",
                "Funding Source", "Nature", "Eval. Type", "Timing", "Progress", "Doc. Type"
            ]

            tabs_en = st.tabs(tab_labels_en)

            # 0. By Country
            with tabs_en[0]:
                 if 'Country(ies)' in df_viz_dedup_en.columns:
                     plot_evolution_en(df_viz_dedup_en, 'Country(ies)', "Country")
                 else:
                     st.info("Column 'Country(ies)' not available.")

            # 1. By Admin Unit
            with tabs_en[1]:
                 if 'Recommendation administrative unit' in df_viz_dedup_en.columns:
                     plot_evolution_en(df_viz_dedup_en, 'Recommendation administrative unit', "Administrative Unit")
                 else:
                     st.info("Column 'Recommendation administrative unit' not available.")

            # 2. By Dimension
            with tabs_en[2]:
                 plot_evolution_en(df_viz_dedup_en, 'assigned_dimension', "Dimension")

            # 3. By Subdimension
            with tabs_en[3]:
                 plot_evolution_en(df_viz_dedup_en, 'assigned_subdim', "Subdimension")

            # 4. Management Response
            with tabs_en[4]:
                if 'Management response' in df_viz_dedup_en.columns:
                    plot_evolution_en(df_viz_dedup_en, 'Management response', "Management Response")
                else:
                    st.info("Column 'Management response' not available.")

            # 5. Evaluation Theme - MULTI VALUE (Keep duplicates to show frequency)
            with tabs_en[5]:
                if 'Evaluation theme(s)' in df_filtered_viz_en.columns:
                    plot_evolution_en(df_filtered_viz_en, 'Evaluation theme(s)', "Evaluation Theme")
                else:
                    st.info("Column 'Evaluation theme(s)' not available.")

            # 6. Recommendation Theme - MULTI VALUE (Keep duplicates)
            with tabs_en[6]:
                if 'Recommendation theme' in df_filtered_viz_en.columns:
                    plot_evolution_en(df_filtered_viz_en, 'Recommendation theme', "Recommendation Theme")
                else:
                    st.info("Column 'Recommendation theme' not available.")

            # 7. Technical Unit - MULTI VALUE
            with tabs_en[7]:
                if 'Technical unit(s)' in df_filtered_viz_en.columns:
                    plot_evolution_en(df_filtered_viz_en, 'Technical unit(s)', "Technical Unit")
                else:
                    st.info("Column 'Technical unit(s)' not available.")

            # 8. Funding Source - MULTI VALUE
            with tabs_en[8]:
                if 'Funding source(s)' in df_filtered_viz_en.columns:
                    plot_evolution_en(df_filtered_viz_en, 'Funding source(s)', "Funding Source")
                else:
                    st.info("Column 'Funding source(s)' not available.")

            # 9. Evaluation Nature
            with tabs_en[9]:
                if 'Evaluation nature' in df_viz_dedup_en.columns:
                    plot_evolution_en(df_viz_dedup_en, 'Evaluation nature', "Evaluation Nature")
                else:
                    st.info("Column 'Evaluation nature' not available.")

            # 10. Evaluation Type
            with tabs_en[10]:
                if 'Evaluation type' in df_viz_dedup_en.columns:
                    plot_evolution_en(df_viz_dedup_en, 'Evaluation type', "Evaluation Type")
                else:
                    st.info("Column 'Evaluation type' not available.")

            # 11. Evaluation Timing
            with tabs_en[11]:
                if 'Evaluation timing' in df_viz_dedup_en.columns:
                    plot_evolution_en(df_viz_dedup_en, 'Evaluation timing', "Evaluation Timing")
                else:
                    st.info("Column 'Evaluation timing' not available.")

            # 12. Progress
            with tabs_en[12]:
                if 'Progress' in df_viz_dedup_en.columns:
                    plot_evolution_en(df_viz_dedup_en, 'Progress', "Progress")
                else:
                    st.info("Column 'Progress' not available.")

            # 13. Document Type
            with tabs_en[13]:
                if 'Evaluation document type' in df_viz_dedup_en.columns:
                    plot_evolution_en(df_viz_dedup_en, 'Evaluation document type', "Document Type")
                else:
                    st.info("Column 'Evaluation document type' not available.")

            # --- SECTION 3: ADVANCED AI TOOLS ---
            st.markdown("---")
            st.subheader("🤖 3. Advanced AI Tools")
            st.info("Use Artificial Intelligence tools to deeply analyze or summarize the filtered recommendations.")

            # --- Shared Filters for AI Tools ---
            st.markdown("##### 1. Define Subset for Analysis")

            df_ai_base_en = df_filtered_viz_en.copy()
            ai_filters_en = {}

            with st.expander("🔎 Global Filters for AI Analysis", expanded=True):
                 c_ai1_en, c_ai2_en = st.columns(2)

                 with c_ai1_en:
                      if 'Region(s)' in df_viz_en.columns:
                          opts = sorted([str(x) for x in df_viz_en['Region(s)'].unique() if pd.notna(x)])
                          ai_filters_en['Region(s)'] = st.multiselect("Region:", opts, key="ai_region_en")

                      if 'Country(ies)' in df_viz_en.columns:
                           opts = sorted([str(x) for x in df_viz_en['Country(ies)'].unique() if pd.notna(x)])
                           ai_filters_en['Country(ies)'] = st.multiselect("Country:", opts, key="ai_country_en")

                      if 'Evaluation theme(s)' in df_viz_en.columns:
                           opts = sorted([str(x) for x in df_viz_en['Evaluation theme(s)'].unique() if pd.notna(x)])
                           ai_filters_en['Evaluation theme(s)'] = st.multiselect("Eval. Theme:", opts, key="ai_eval_theme_en")

                 with c_ai2_en:
                      if 'Year' in df_viz_en.columns:
                          opts = sorted([int(x) for x in df_viz_en['Year'].unique() if pd.notna(x)])
                          ai_filters_en['Year'] = st.multiselect("Year:", opts, key="ai_year_en")

                      if 'Management response' in df_viz_en.columns:
                          opts = sorted([str(x) for x in df_viz_en['Management response'].unique() if pd.notna(x)])
                          ai_filters_en['Management response'] = st.multiselect("Mgmt. Response:", opts, key="ai_mgmt_en")

                      if 'assigned_dimension' in df_viz_en.columns:
                          opts = sorted([str(x) for x in df_viz_en['assigned_dimension'].unique() if pd.notna(x)])
                          ai_filters_en['assigned_dimension'] = st.multiselect("Dimension:", opts, key="ai_dim_en")

            # Apply Filters
            for col_ai, vals_ai in ai_filters_en.items():
                if vals_ai and col_ai in df_ai_base_en.columns:
                    df_ai_base_en = df_ai_base_en[df_ai_base_en[col_ai].isin(vals_ai)]

            st.write(f"**Total Filtered Records:** {len(df_ai_base_en)}")

            st.markdown("---")

            # --- Dual Action Buttons ---
            col_deep_en, col_summ_en = st.columns(2)

            # --- LEFT: Deep Analysis ---
            with col_deep_en:
                st.markdown("#### 🧠 Deep Analysis")
                st.caption("Evaluates coherence, quality and innovation of action plans.")

                df_deep_ready_en = df_ai_base_en.copy()
                if 'Action plan' in df_deep_ready_en.columns:
                    df_deep_ready_en = df_deep_ready_en[df_deep_ready_en['Action plan'].notna() & (df_deep_ready_en['Action plan'].astype(str).str.strip() != "")]
                valid_deep_count_en = len(df_deep_ready_en)

                st.metric("Valid (with Plan)", valid_deep_count_en)

                with st.expander("Configuration"):
                    deep_model_en = st.selectbox("Model:", ["gpt-4o-mini", "gpt-4o"], index=0, key="deep_model_sel_en")
                    limit_rows_deep_en = st.number_input("Limit (0=All):", min_value=0, value=0, step=10, key="deep_limit_en")

                if st.button("🚀 Start Analysis", key="btn_run_deep_en"):
                     if df_deep_ready_en.empty:
                         st.warning("No valid data.")
                     else:
                         id_col_deep_en = 'Recommendation ID' if 'Recommendation ID' in df_deep_ready_en.columns else 'Recommendation description'
                         df_deep_input_en = df_deep_ready_en.drop_duplicates(subset=[id_col_deep_en]).copy()
                         if limit_rows_deep_en > 0:
                             df_deep_input_en = df_deep_input_en.head(limit_rows_deep_en)

                         st.info(f"Processing all {len(df_deep_input_en)} recommendations..." if limit_rows_deep_en == 0 else f"Processing {len(df_deep_input_en)} recommendations...")

                         analysis_cache_en = AnalysisCache()
                         args_list_en = []
                         for idx, row in df_deep_input_en.iterrows():
                             rec = str(row.get('Recommendation description', ''))
                             plan = str(row.get('Action plan', ''))
                             comments = str(row.get('Comments', '')) if 'Comments' in row else ""
                             args_list_en.append((idx, rec, plan, comments, openai_api_key, analysis_cache_en))

                         results_map_en = {}
                         pbar_deep_en = st.progress(0, text="Analyzing...")
                         completed_count_en = 0

                         with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                             futures_en = [executor.submit(run_row_analysis, arg) for arg in args_list_en]
                             for future in concurrent.futures.as_completed(futures_en):
                                 idx, result, _ = future.result()
                                 if result:
                                    results_map_en[idx] = result
                                 completed_count_en += 1
                                 pbar_deep_en.progress(completed_count_en / len(args_list_en), text=f"Analyzing... {completed_count_en}/{len(args_list_en)}")

                         pbar_deep_en.empty()
                         analysis_cache_en.save()

                         analysis_list_en = []
                         for idx, res in results_map_en.items():
                             res['original_index'] = idx
                             analysis_list_en.append(res)

                         if analysis_list_en:
                             df_res_en = pd.DataFrame(analysis_list_en)
                             df_res_en.set_index('original_index', inplace=True)
                             df_final_deep_en = df_deep_input_en.join(df_res_en)

                             list_cols_en = ['extracted_actions_from_rec', 'actions_proposed_in_plan', 'rejection_difficulty_classification', 'tags', 'rec_additional_tags']
                             for col in list_cols_en:
                                 if col in df_final_deep_en.columns:
                                     df_final_deep_en[col] = df_final_deep_en[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

                             st.session_state['deep_analysis_df_en'] = df_final_deep_en
                             st.success("Analysis complete!")

            # --- RIGHT: Summary Generation ---
            with col_summ_en:
                st.markdown("#### ✨ Executive Summary")
                st.caption("Generates a narrative synthesis of findings.")

                st.metric("Total to Summarize", len(df_ai_base_en))

                with st.expander("Configuration"):
                    max_tokens_val_en = st.slider("Length (Tokens):", 500, 4000, 3000, 100, key="summ_len_en")

                if st.button("📝 Generate Global Summary", key="btn_run_summ_en"):
                    if df_ai_base_en.empty:
                        st.warning("No data.")
                    else:
                        with st.spinner("Generating summary..."):
                            id_col_summ_en = 'Recommendation ID' if 'Recommendation ID' in df_ai_base_en.columns else 'Recommendation description'
                            df_summ_unique_en = df_ai_base_en.drop_duplicates(subset=[id_col_summ_en])

                            text_list_en = []
                            for idx, row in df_summ_unique_en.iterrows():
                                 desc = str(row.get('Recommendation description', ''))
                                 mgmt = str(row.get('Management response', '')) if 'Management response' in row else "N/A"
                                 comments = str(row.get('Comments', '')) if 'Comments' in row else "N/A"
                                 action = str(row.get('Action plan', '')) if 'Action plan' in row else "N/A"
                                 text_list_en.append(f"--- Rec ---\nDesc: {desc}\nResp: {mgmt}\nCom: {comments}\nPlan: {action}\n")

                            full_text_en = "\n".join(text_list_en)
                            summary_text_en = generate_executive_summary(full_text_en, max_output_tokens=max_tokens_val_en)
                            st.session_state['summary_result_en'] = summary_text_en
                            st.success("Summary generated!")

            # --- Display Results Areas (Full Width) ---

            # 1. Deep Analysis Results
            if 'deep_analysis_df_en' in st.session_state:
                st.markdown("---")
                st.subheader("📊 Results: Deep Analysis")
                df_final_deep_en = st.session_state['deep_analysis_df_en']

                c_m1_en, c_m2_en = st.columns(2)
                c_m1_en.metric("Avg. Coherence", f"{df_final_deep_en['coherence_score'].mean():.2f}")
                c_m2_en.metric("Avg. Plan Quality", f"{df_final_deep_en['plan_quality_score'].mean():.2f}")

                st.dataframe(df_final_deep_en[['Recommendation description', 'coherence_score', 'plan_quality_score', 'rec_innovation_score', 'rejection_difficulty_classification', 'tags']], use_container_width=True)

                # Export Deep
                out_deep_en = BytesIO()
                # Drop columns AE:AT (indices 30-45) and AW:BG (indices 48-58) before export
                _cols_to_drop_deep_en = []
                if len(df_final_deep_en.columns) > 30:
                    _cols_to_drop_deep_en += list(df_final_deep_en.columns[30:min(46, len(df_final_deep_en.columns))])
                if len(df_final_deep_en.columns) > 48:
                    _cols_to_drop_deep_en += list(df_final_deep_en.columns[48:min(59, len(df_final_deep_en.columns))])
                df_final_deep_en_export = df_final_deep_en.drop(columns=_cols_to_drop_deep_en, errors='ignore')
                with pd.ExcelWriter(out_deep_en, engine='xlsxwriter') as writer:
                    df_final_deep_en_export.to_excel(writer, index=False, sheet_name='Deep_Analysis')

                st.download_button("📥 Download Analysis Report (.xlsx)", out_deep_en.getvalue(), "deep_analysis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # Visualizations Deep
                st.markdown("##### 📈 Key Metrics Distribution")

                col_v1_en, col_v2_en = st.columns(2)

                if 'coherence_score' in df_final_deep_en.columns:
                    with col_v1_en:
                        fig_d1_en = px.histogram(df_final_deep_en, x="coherence_score", nbins=10, title="Coherence Distribution", color_discrete_sequence=['#636EFA'])
                        st.plotly_chart(fig_d1_en, use_container_width=True)

                if 'plan_quality_score' in df_final_deep_en.columns:
                     with col_v2_en:
                        fig_d2_en = px.histogram(df_final_deep_en, x="plan_quality_score", nbins=10, title="Plan Quality Distribution", color_discrete_sequence=['#EF553B'])
                        st.plotly_chart(fig_d2_en, use_container_width=True)

                col_v3_en, col_v4_en = st.columns(2)

                if 'attention_level_score' in df_final_deep_en.columns:
                     with col_v3_en:
                        fig_d3_en = px.histogram(df_final_deep_en, x="attention_level_score", nbins=10, title="Attention Level", color_discrete_sequence=['#00CC96'])
                        st.plotly_chart(fig_d3_en, use_container_width=True)

                if 'rec_innovation_score' in df_final_deep_en.columns:
                     with col_v4_en:
                         fig_pie_en = px.pie(df_final_deep_en, names='rec_innovation_score', title="Innovation Level", hole=0.3)
                         st.plotly_chart(fig_pie_en, use_container_width=True)

                st.markdown("##### 🚦 Feasibility and Impact")
                col_v5_en, col_v6_en = st.columns(2)

                if 'rec_operational_feasibility' in df_final_deep_en.columns:
                    with col_v5_en:
                        fig_feas_en = px.histogram(df_final_deep_en, x='rec_operational_feasibility', title="Operational Feasibility", color='rec_operational_feasibility')
                        st.plotly_chart(fig_feas_en, use_container_width=True)

                if 'rec_expected_impact' in df_final_deep_en.columns:
                    with col_v6_en:
                         fig_imp_en = px.histogram(df_final_deep_en, x='rec_expected_impact', title="Expected Impact", color='rec_expected_impact')
                         st.plotly_chart(fig_imp_en, use_container_width=True)

            # 2. Summary Results
            if 'summary_result_en' in st.session_state:
                st.markdown("---")
                st.subheader("📄 Results: Executive Summary")
                st.markdown(st.session_state['summary_result_en'])

                # Export Summary
                out_summ_en = BytesIO()
                with pd.ExcelWriter(out_summ_en, engine='xlsxwriter') as writer:
                    df_ai_base_en.to_excel(writer, index=False, sheet_name='Base Data')
                    pd.DataFrame({'Summary': [st.session_state['summary_result_en']]}).to_excel(writer, index=False, sheet_name='Summary')

                st.download_button("📥 Download Summary + Data (.xlsx)", out_summ_en.getvalue(), "executive_summary.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
