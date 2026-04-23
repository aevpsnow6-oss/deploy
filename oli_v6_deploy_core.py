"""ILO project quality assessment app (core tabs).

Tabs:
    1. Valoración Preliminar de Calidad de Proyectos
    2. Diagnóstico de Atributos Específicos
    3. Diagnóstico de Sostenibilidad del Proyecto
    4. Pregúntale a tus Documentos
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


def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_urls': False}}) as writer:
        df.to_excel(writer, index=False, sheet_name='Datos Filtrados')
    processed_data = output.getvalue()
    return processed_data

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

# JSON Schemas for structured rubric responses (Tabs 1-4).
# Using strict structured outputs eliminates JSON-parse fallbacks AND gives a
# verifiable 'parte_enfocada' signal that the model committed to Part 2.
RUBRIC_SCHEMA_TWO_PART = {
    "name": "rubric_response_two_part",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "Respuesta": {"type": "string", "enum": ["Yes", "No", "Partial", "Not Found"]},
            "Razonamiento": {"type": "string"},
            "Evidencia": {"type": "string"},
            "parte_enfocada": {"type": "string", "enum": ["Parte 2", "Parte 1", "Ambas"]},
        },
        "required": ["Respuesta", "Razonamiento", "Evidencia", "parte_enfocada"],
    },
}

RUBRIC_SCHEMA_SINGLE = {
    "name": "rubric_response_single",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "Respuesta": {"type": "string", "enum": ["Yes", "No", "Partial", "Not Found"]},
            "Razonamiento": {"type": "string"},
            "Evidencia": {"type": "string"},
        },
        "required": ["Respuesta", "Razonamiento", "Evidencia"],
    },
}

# Structured schema for the critic stage. Forcing A–E enumeration as separate enum
# fields (not free text) prevents the model from skipping the count or rebranding
# Part-1 content as "dedicated". Each A–E has a mandatory evidence snippet so the
# code layer can later sanity-check that "presente" is backed by a verbatim quote.
CRITIC_SCHEMA_TWO_PART = {
    "name": "critic_response_two_part",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subject_part2": {"type": "string"},
            "named_subject_specific": {"type": "string"},
            "general_clause_ignored": {"type": "string"},
            "dedicated_A_sub_objective": {"type": "string", "enum": ["presente", "ausente"]},
            "dedicated_A_evidence": {"type": "string"},
            "dedicated_B_indicator": {"type": "string", "enum": ["presente", "ausente"]},
            "dedicated_B_evidence": {"type": "string"},
            "dedicated_C_activity": {"type": "string", "enum": ["presente", "ausente"]},
            "dedicated_C_evidence": {"type": "string"},
            "dedicated_D_budget": {"type": "string", "enum": ["presente", "ausente"]},
            "dedicated_D_evidence": {"type": "string"},
            "dedicated_E_target": {"type": "string", "enum": ["presente", "ausente"]},
            "dedicated_E_evidence": {"type": "string"},
            "dedicated_total": {"type": "integer"},
            "verdict": {"type": "string", "enum": ["Yes", "No", "Partial", "Not Found", "Keep"]},
            "justification": {"type": "string"},
            "recommendations": {"type": "string"},
        },
        "required": [
            "subject_part2",
            "named_subject_specific", "general_clause_ignored",
            "dedicated_A_sub_objective", "dedicated_A_evidence",
            "dedicated_B_indicator", "dedicated_B_evidence",
            "dedicated_C_activity", "dedicated_C_evidence",
            "dedicated_D_budget", "dedicated_D_evidence",
            "dedicated_E_target", "dedicated_E_evidence",
            "dedicated_total", "verdict", "justification", "recommendations",
        ],
    },
}

CRITIC_SCHEMA_SINGLE = {
    "name": "critic_response_single",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subject": {"type": "string"},
            "named_subject_specific": {"type": "string"},
            "general_clause_ignored": {"type": "string"},
            "dedicated_A_sub_objective": {"type": "string", "enum": ["presente", "ausente"]},
            "dedicated_A_evidence": {"type": "string"},
            "dedicated_B_indicator": {"type": "string", "enum": ["presente", "ausente"]},
            "dedicated_B_evidence": {"type": "string"},
            "dedicated_C_activity": {"type": "string", "enum": ["presente", "ausente"]},
            "dedicated_C_evidence": {"type": "string"},
            "dedicated_D_budget": {"type": "string", "enum": ["presente", "ausente"]},
            "dedicated_D_evidence": {"type": "string"},
            "dedicated_E_target": {"type": "string", "enum": ["presente", "ausente"]},
            "dedicated_E_evidence": {"type": "string"},
            "dedicated_total": {"type": "integer"},
            "verdict": {"type": "string", "enum": ["Yes", "No", "Partial", "Not Found", "Keep"]},
            "justification": {"type": "string"},
            "recommendations": {"type": "string"},
        },
        "required": [
            "subject",
            "named_subject_specific", "general_clause_ignored",
            "dedicated_A_sub_objective", "dedicated_A_evidence",
            "dedicated_B_indicator", "dedicated_B_evidence",
            "dedicated_C_activity", "dedicated_C_evidence",
            "dedicated_D_budget", "dedicated_D_evidence",
            "dedicated_E_target", "dedicated_E_evidence",
            "dedicated_total", "verdict", "justification", "recommendations",
        ],
    },
}


def _is_framing_evidence(evidence: str) -> tuple:
    """Heuristic detector for FRAMING-style evidence.

    Returns (is_framing, reason). Conservative by design — we only flag the clearest
    FRAMING patterns so legitimate DEDICATED quotes are not downgraded.
    """
    if not evidence or not evidence.strip():
        return True, "evidence empty"
    ev_lower = evidence.lower()

    # Enumeration markers — these almost always indicate the subject is in a list.
    framing_markers = [
        "among others", "among which", "including,", "including ",
        "such as", "inter alia", "and others", "etc.", "etcetera",
        "entre otros", "entre otras", "incluyendo", "incluidos",
        "incluidas", "como por ejemplo",
    ]
    for m in framing_markers:
        if m in ev_lower:
            return True, f"contains enumeration marker '{m.strip()}'"

    # 3+ commas → likely a list of groups.
    if ev_lower.count(",") >= 3:
        return True, f"{ev_lower.count(',')} comma-separated items (list pattern)"

    # Very short stubs (< 12 chars) are unlikely to describe a dedicated element.
    if len(evidence.strip()) < 12:
        return True, "evidence too short to describe a dedicated element"

    return False, ""


def _apply_critic_gate_and_render(result: dict, is_two_part: bool) -> str:
    """Enforce the A–E → verdict mapping at the code layer and render the display text.

    Two layers of protection:
      1. Evidence validation: each 'presente' is downgraded to 'ausente' if its
         evidence quote looks like a FRAMING list (enumeration markers or 3+ commas).
      2. Verdict override: if TOTAL=0 after validation and the model proposed
         Partial/Yes, the verdict is forced to No, with an audit note.
    """
    letters = ['A', 'B', 'C', 'D', 'E']
    fields = {
        'A': ('dedicated_A_sub_objective', 'dedicated_A_evidence', 'sub-objetivo/output'),
        'B': ('dedicated_B_indicator',     'dedicated_B_evidence', 'indicador'),
        'C': ('dedicated_C_activity',      'dedicated_C_evidence', 'actividad dedicada'),
        'D': ('dedicated_D_budget',        'dedicated_D_evidence', 'línea presupuestaria'),
        'E': ('dedicated_E_target',        'dedicated_E_evidence', 'meta cuantificable'),
    }
    states = {ltr: (result.get(fields[ltr][0], 'ausente') or 'ausente') for ltr in letters}

    # Layer 1: evidence validation. Downgrade FRAMING-backed 'presente' to 'ausente'.
    downgrades = []
    for ltr in letters:
        if states[ltr] == 'presente':
            evidence = result.get(fields[ltr][1], '') or ''
            is_framing, reason = _is_framing_evidence(evidence)
            if is_framing:
                states[ltr] = 'ausente'
                downgrades.append(f"{ltr} ({reason})")

    actual_total = sum(1 for ltr in letters if states[ltr] == 'presente')

    model_verdict = (result.get('verdict') or '').strip() or 'No'
    subject = (
        result.get('named_subject_specific')
        or result.get('subject_part2')
        or result.get('subject')
        or 'el sujeto específico'
    ).strip()
    ignored = (result.get('general_clause_ignored') or '').strip()

    # Layer 2: verdict override at TOTAL=0.
    override_note = ''
    if actual_total == 0 and model_verdict not in ('No', 'Not Found'):
        override_note = f" [Ajuste automático: el modelo propuso '{model_verdict}' pero TOTAL=0 obliga a 'No'.]"
        model_verdict = 'No'
    elif actual_total >= 3 and model_verdict in ('No', 'Not Found'):
        override_note = f" [Nota: TOTAL={actual_total} con modelo='{model_verdict}'; revisar manualmente.]"

    if downgrades:
        override_note += f" [Downgrade por evidencia FRAMING: {'; '.join(downgrades)}.]"
    if ignored:
        override_note += f" [Cláusula general ignorada: '{ignored}'.]"

    conteo_line = (
        f"Elementos dedicados Parte 2 [{subject}]: "
        f"A={states['A']}, B={states['B']}, C={states['C']}, "
        f"D={states['D']}, E={states['E']}. TOTAL={actual_total}."
    ) if is_two_part else (
        f"Elementos dedicados [{subject}]: "
        f"A={states['A']}, B={states['B']}, C={states['C']}, "
        f"D={states['D']}, E={states['E']}. TOTAL={actual_total}."
    )

    justification = (result.get('justification') or '').strip()
    recommendations = (result.get('recommendations') or '').strip()

    body_parts = [conteo_line + override_note]
    if justification:
        body_parts.append(justification)
    if recommendations and model_verdict in ('No', 'Partial', 'Not Found'):
        if not recommendations.lower().startswith('para mejorar la calificación'):
            recommendations = f"**Para mejorar la calificación** debiese incluirse {recommendations}"
        body_parts.append(recommendations)

    return f"VEREDICTO: {model_verdict}\n\n" + "\n\n".join(body_parts)


def load_appraisal_questions():
    """Load and cache appraisal questions from Excel file"""
    try:
        df = pd.read_excel('./Appraisal Checklist_2025 es-419.xlsx', sheet_name='rubric')
        if 'Pregunta_Realizada' not in df.columns:
            return None, "La columna 'Pregunta_Realizada' no se encontró en el archivo de Excel."
        
        # Replace first 3 characters of Pregunta_Realizada with Tema numbering
        if 'Tema' in df.columns:
            df['Pregunta_Realizada'] = df.apply(
                lambda row: str(row['Tema']) + ' ' + str(row['Pregunta_Realizada'])[3:].strip() 
                if pd.notna(row['Tema']) and pd.notna(row['Pregunta_Realizada']) and len(str(row['Pregunta_Realizada'])) > 3
                else row['Pregunta_Realizada'], 
                axis=1
            )
        
        return df, None
    except FileNotFoundError:
        return None, "No se encontró el archivo Appraisal Checklist_2025 es-419.xlsx. Asegúrate de que exista en el directorio de la aplicación."
    except Exception as e:
        return None, f"Error al cargar el archivo de preguntas: {str(e)}"

def extract_document_content(uploaded_file):
    """Extract and process content from uploaded DOCX file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Extract content using docx2python
        doc_result = docx2python(tmp_file_path)
        
        # Get file stats
        file_size = os.path.getsize(tmp_file_path)
        
        # Extract structured content (you'll need to implement extract_docx_structure)
        # For now, using simple text extraction
        full_text = doc_result.text
        word_count = len(full_text.split())
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return {
            'text': full_text,
            'file_size': file_size,
            'word_count': word_count,
            'success': True
        }
    
    except Exception as e:
        return {
            'text': '',
            'file_size': 0,
            'word_count': 0,
            'success': False,
            'error': str(e)
        }


def parse_critic_verdict(critic_text):
    """Extract the VEREDICTO tag from the first line of a critic response.

    Returns (verdict, body) where:
      - verdict ∈ {"Yes", "No", "Partial", "Not Found", "Keep", None}
      - body is the critic text with the verdict line stripped (original text if no verdict parsed)
    """
    if not isinstance(critic_text, str) or not critic_text.strip():
        return None, critic_text or ""
    m = re.match(
        r'\s*VEREDICTO\s*:\s*(Not\s*Found|Partial|Yes|No|Keep)\s*[\.\n:]?',
        critic_text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None, critic_text.strip()
    raw = m.group(1).strip()
    canonical = {
        'yes': 'Yes',
        'no': 'No',
        'partial': 'Partial',
        'not found': 'Not Found',
        'notfound': 'Not Found',
        'keep': 'Keep',
    }
    verdict = canonical.get(re.sub(r'\s+', ' ', raw.lower()), None)
    body = critic_text[m.end():].lstrip()
    return verdict, body


def parse_two_part_question(question):
    """
    Detect and parse two-part rubric questions where:
    - Part 1 (broader): Sets the general context
    - Part 2 (specific): Asks the critical detail that needs primary focus

    Returns dict with 'is_two_part', 'part1', 'part2', 'full_question'.
    """
    import re

    if not isinstance(question, str):
        return {'is_two_part': False, 'part1': None, 'part2': None, 'full_question': question}

    # New: statement-then-question form. "X clause. ¿Specific clause?" — total ? count = 1
    # but the period+¿ boundary makes it clearly two-clause.
    m = re.search(r'^([^¿]+[\.:])\s*¿\s*(.+\?)\s*$', question.strip(), re.UNICODE)
    if m:
        p1 = m.group(1).strip().rstrip('.:').strip()
        p2 = '¿' + m.group(2).strip()
        if len(p1) > 8 and len(p2) > 8:
            return {'is_two_part': True, 'part1': p1, 'part2': p2, 'full_question': question}

    # New: single-? conjunctive split. "X y se integra Y" — one question mark but two
    # clauses joined by " y (se|la|el|los|las) ". Fire only when the question mentions a
    # specific subject marker (disability, gender, indigenous, etc.) to avoid over-splitting.
    specific_markers = [
        'discapacidad', 'género', 'genero', 'mujeres', 'indígena', 'indigena',
        'juventud', 'jóvenes', 'jovenes', 'afro', 'migrante', 'rural',
        'infancia', 'niñas', 'niños', 'lgbt', 'personas con discapacidad',
    ]
    if question.count('?') < 2:
        lower = question.lower()
        if any(mk in lower for mk in specific_markers):
            split_m = None
            for m2 in re.finditer(r'\s+y\s+(se|la|el|los|las|con|de|una?)\s+', question, re.IGNORECASE | re.UNICODE):
                split_m = m2
            if split_m:
                p1 = question[:split_m.start()].strip()
                p2 = question[split_m.end():].strip()
                if len(p1) > 8 and len(p2) > 8 and any(mk in p2.lower() for mk in specific_markers):
                    if not p1.endswith('?'):
                        p1 = p1.rstrip('.,;:') + '?'
                    if not p2.endswith('?'):
                        p2 = p2.rstrip('.,;:') + '?'
                    if not p2.startswith('¿'):
                        p2 = '¿' + p2[0].lower() + p2[1:] if p2 else p2
                    return {'is_two_part': True, 'part1': p1, 'part2': p2, 'full_question': question}
        return {'is_two_part': False, 'part1': None, 'part2': None, 'full_question': question}

    # Pattern 1: Explicit keyword separator ("específicamente", "además", ...).
    keywords = ['específicamente', 'en particular', 'en concreto', 'puntualmente', 'además']
    for keyword in keywords:
        pattern = rf'(.+?\?)\s*{keyword}[,:]?\s*¿\s*(.+?\?)'
        match = re.search(pattern, question, re.IGNORECASE | re.UNICODE)
        if match:
            return {
                'is_two_part': True,
                'part1': match.group(1).strip(),
                'part2': '¿' + match.group(2).strip(),
                'full_question': question,
            }

    # Pattern 2a: Two clauses joined with explicit ¿ separator (Spanish convention).
    # Unicode-safe: matches lowercase, accented capitals (Á/Í/Ó/Ú/Ñ), digits, etc.
    match = re.search(r'(.+?\?)\s*¿\s*(.+\?)', question, re.UNICODE)
    if match:
        return {
            'is_two_part': True,
            'part1': match.group(1).strip(),
            'part2': '¿' + match.group(2).strip(),
            'full_question': question,
        }

    # Pattern 2b: Run-together — first clause ends in '?', second starts immediately
    # with a non-whitespace character (rubric formatting quirk, e.g. "...gráfico?Se identifican...").
    match = re.search(r'(.+?\?)(\S.+\?)', question, re.UNICODE)
    if match:
        part2 = match.group(2).strip()
        if not part2.startswith('¿'):
            part2 = '¿' + part2
        return {
            'is_two_part': True,
            'part1': match.group(1).strip(),
            'part2': part2,
            'full_question': question,
        }

    return {'is_two_part': False, 'part1': None, 'part2': None, 'full_question': question}

def analyze_question_with_llm_tab1(question, document_text):
    """
    Analyze a single question against the document using LLM with full context (up to 110K tokens).
    TAB1-SPECIFIC VERSION: Explicitly states when 2+ parts are detected in the question.
    """
    try:
        # Truncate to ~110K tokens to maximize context while leaving room for prompts and response
        combined_text = truncate_to_token_limit(document_text or "", max_tokens=110000, encoding_obj=encoding)

        if not combined_text.strip():
            return {
                'Pregunta': question,
                'Respuesta': 'Not Found',
                'Razonamiento': 'No se encontró contenido en el documento para analizar.',
                'Evidencia': '',
                'Status': 'Success'
            }

        # Parse question to detect two-part structure
        parsed = parse_two_part_question(question)

        # Customize system prompt based on question structure
        if parsed['is_two_part']:
            system_content = """You are an expert document analyst specializing in ILO project quality assessment.

**CRITICAL INSTRUCTION FOR TWO-PART QUESTIONS:**

This question has TWO PARTS with different priorities:
- **Part 2 (Specific Focus - PRIMARY)**: The critical question that requires detailed analysis. Drives the final answer.
- **Part 1 (Broader Context)**: Provides the general framework.

**YOUR ANALYSIS APPROACH:**
1. **Answer Part 2 IN DEPTH FIRST.** This drives the final Respuesta.
2. **Then relate Part 2's findings back to Part 1** (broader context).
3. The final Respuesta is determined PRIMARILY by Part 2; Part 1 provides framing only.

**Response fields (returned via structured output):**
- "Respuesta": Yes / No / Partial / Not Found — based PRIMARILY on Part 2.
- "Razonamiento": MUST begin with "Se identificaron 2 partes en esta pregunta." Then answer Part 2 concisely, then briefly connect to Part 1. Max 120 words. Be terse — no filler, no restating the question.
- "Evidencia": Quotes supporting Part 2 FIRST, then supporting evidence for Part 1. Max 180 words. Trim quotes to the essential span; avoid long blocks.
- "parte_enfocada": Set to "Parte 2" when your analysis was driven by Part 2 (the expected default). Use "Ambas" only if both parts truly required equal weight. Use "Parte 1" only if Part 2 was unanswerable from the document.

**SCOPE LOCK (CRITICAL):**
La pregunta nombra un sujeto o alcance específico (p.ej. "personas con discapacidad", "mujeres rurales", "trabajo infantil en minería", una región, un sector o un período concreto). Tu análisis debe limitarse ESTRICTAMENTE a ese sujeto exacto.
1. Identifica el sujeto exacto de la pregunta antes de responder.
2. Evidencia sobre sujetos relacionados pero distintos (otras poblaciones vulnerables, otros grupos, otros sectores, otras regiones, otros períodos) NO cuenta como respuesta. Ejemplo: si la pregunta es sobre discapacidad, evidencia sobre mujeres, pueblos indígenas o juventud NO la satisface.
3. Si el sujeto nombrado está ausente del documento pero sujetos relacionados están ampliamente cubiertos, Respuesta debe ser "Not Found" o "No" — NO "Partial" ni "Yes". Sub-inclusión es preferible a sobre-inclusión.
4. En tu Razonamiento, declara explícitamente cuál sujeto exacto se analizó y confirma que la evidencia trata sobre ese sujeto (no sobre uno relacionado).

**EVIDENCE-ROLE FILTER & DECISION GATE FOR PARTE 2 (APPLY FIRST — DRIVES THE Respuesta):**

Before choosing a Respuesta, classify every candidate evidence passage about the Parte-2 subject (e.g., personas con discapacidad, género, pueblos indígenas, mujeres rurales) as either FRAMING or DEDICATED.

FRAMING mention (does NOT count toward Parte 2, even if the subject is named):
  - Overall objective / impact statement naming multiple groups
  - Stakeholder, consultation, or research participant lists
  - Monitoring scope enumerations ("...among others", "...including X, Y, Z")
  - Boilerplate inclusion language
  - Any passage where the subject appears in a list of 3+ groups without dedicated follow-up

DEDICATED element (counts toward Parte 2):
  A. A sub-objective, outcome, or output whose title/purpose names the subject
  B. An indicator disaggregated by or specifically targeting the subject
  C. An activity whose primary purpose addresses the subject
  D. A budget line or resource allocation for the subject
  E. A quantifiable target for the subject

Count of DEDICATED elements (A–E) — NOT raw mention count — drives Respuesta:
  - 0 dedicated elements (only FRAMING mentions)      → "No" or "Not Found"
  - 1–2 dedicated elements                            → "Partial"
  - 3–4 dedicated elements                            → "Partial" (or "Yes" only if substantive and at-par with any claim/label in Parte 1)
  - 5 dedicated elements with substantive evidence    → "Yes"

If every citable quote is a FRAMING mention, Respuesta MUST be "No" or "Not Found" — regardless of how many times the subject is named. Passing mentions in stakeholder lists, consultation rosters, or "among others" phrases DO NOT qualify for "Partial".

In "Evidencia", prefix each cited quote with [DEDICATED] or [FRAMING]. You may NOT justify "Partial" or "Yes" using only [FRAMING] quotes.

Siempre responder en español. Enfoque: 95% en Parte 2, 5% en su relación con Parte 1. La Parte 1 solo existe para enmarcar brevemente; NO analices la Parte 1 de forma independiente."""

            # Part 2 listed FIRST to exploit primacy bias toward the priority clause.
            # Full original question deliberately omitted — including it re-anchors the model on Part 1.
            user_content = f"""PREGUNTA CON DOS PARTES — RESPONDE PRIMERO LA PARTE 2.

**PARTE 2 (Enfoque Específico - PRIORITARIO):**
{parsed['part2']}

**PARTE 1 (Contexto General — solo para enmarcar):**
{parsed['part1']}

**Texto del Documento:**
{combined_text}

RECUERDA:
1. COMENZAR tu Razonamiento con "Se identificaron 2 partes en esta pregunta."
2. Enfoca tu análisis PRINCIPALMENTE en la Parte 2 (pregunta específica).
3. Luego explica cómo se relaciona con la Parte 1 (contexto general).
4. Marca "parte_enfocada" como "Parte 2" salvo que el documento no permita responderla."""

            response_format = {"type": "json_schema", "json_schema": RUBRIC_SCHEMA_TWO_PART}

        else:
            # Original single-question prompt
            system_content = """You are an expert document analyst. Analyze the document against the given question and provide a structured JSON response with exactly this format and always respond in Spanish:
            {
                "Respuesta": "Yes/No/Partial/Not Found",
                "Razonamiento": "Concise explanation (max 120 words, terse)",
                "Evidencia": "Trimmed text excerpts supporting the answer (max 180 words)"
            }

**SCOPE LOCK (CRITICAL):**
The question names a specific subject/scope (e.g., a specific population like people with disabilities; a specific sector, region, or period). Your analysis must be STRICTLY limited to that exact named subject.
1. Identify the exact subject of the question before answering.
2. Evidence about related-but-different subjects (other vulnerable populations, other groups, other sectors/regions/periods) does NOT count. Example: if the question is about disability, evidence about women, indigenous peoples, or youth does NOT satisfy it.
3. If the named subject is absent but related subjects are extensively covered, Respuesta must be "Not Found" or "No" — NOT "Partial" and NOT "Yes". Under-inclusion is preferred over over-inclusion.
4. In your Razonamiento, explicitly state which exact subject was analyzed and confirm the evidence concerns that subject (not a related one).

**EVIDENCE-ROLE FILTER & DECISION GATE (APPLY FIRST — DRIVES Respuesta):**

Classify every candidate evidence passage about the named specific subject as either FRAMING or DEDICATED.

FRAMING mention (does NOT count, even if the subject is named):
  - Overall objective / impact statement naming multiple groups
  - Stakeholder, consultation, or research participant lists
  - Monitoring scope enumerations ("...among others", "...including X, Y, Z")
  - Boilerplate inclusion language
  - Any passage where the subject appears in a list of 3+ groups without dedicated follow-up

DEDICATED element (counts):
  A. A sub-objective, outcome, or output whose title/purpose names the subject
  B. An indicator disaggregated by or targeting the subject
  C. An activity whose primary purpose addresses the subject
  D. A budget line or resource allocation for the subject
  E. A quantifiable target for the subject

Count of DEDICATED elements (A–E) — NOT raw mention count — drives Respuesta:
  - 0 (only FRAMING mentions)                        → "No" or "Not Found"
  - 1–2                                              → "Partial"
  - 3–4                                              → "Partial" (or "Yes" only if substantive)
  - 5 with substantive evidence                      → "Yes"

If every citable quote is a FRAMING mention, Respuesta MUST be "No" or "Not Found". In "Evidencia", prefix each quote with [DEDICATED] or [FRAMING]. You may NOT justify "Partial" or "Yes" using only [FRAMING] quotes."""

            user_content = f"Question: {question}\n\nDocument Text: {combined_text}"
            response_format = {"type": "json_schema", "json_schema": RUBRIC_SCHEMA_SINGLE}

        # Bump reasoning effort for two-part: the 70/30 weighting needs more deliberation.
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=3000,
            reasoning_effort="medium" if parsed['is_two_part'] else "minimal",
            response_format=response_format,
        )

        content = resp.choices[0].message.content
        if not content or not content.strip():
            return {
                'Pregunta': question,
                'Respuesta': 'Error',
                'Razonamiento': 'No se recibió respuesta del modelo.',
                'Evidencia': '',
                'Status': 'Error'
            }

        # Structured outputs guarantees valid JSON conforming to the schema.
        try:
            result = json.loads(content.strip())
        except json.JSONDecodeError:
            return {
                'Pregunta': question,
                'Respuesta': 'Error',
                'Razonamiento': 'Error al procesar la respuesta del modelo.',
                'Evidencia': '',
                'Status': 'Error'
            }

        # Flag rows where the model failed to commit to Part 2 focus despite a two-part question.
        # 'Partial' is the analyst's signal to spot-check the row.
        status = 'Success'
        if parsed['is_two_part'] and result.get('parte_enfocada') != 'Parte 2':
            status = 'Partial'

        return {
            'Pregunta': question,
            'Respuesta': result.get('Respuesta', 'Not Found'),
            'Razonamiento': result.get('Razonamiento', ''),
            'Evidencia': result.get('Evidencia', ''),
            'Status': status,
        }

    except Exception as e:
        return {
            'Pregunta': question,
            'Respuesta': 'Error',
            'Razonamiento': f'Analysis failed: {str(e)}',
            'Evidencia': '',
            'Status': 'Error'
        }

def analyze_question_with_llm(question, document_text):
    """Analyze a single question against the document using LLM with full context (up to 110K tokens)."""
    try:
        # Truncate to ~110K tokens to maximize context while leaving room for prompts and response
        combined_text = truncate_to_token_limit(document_text or "", max_tokens=110000, encoding_obj=encoding)

        if not combined_text.strip():
            return {
                'Pregunta': question,
                'Respuesta': 'Not Found',
                'Razonamiento': 'No se encontró contenido en el documento para analizar.',
                'Evidencia': '',
                'Status': 'Success'
            }

        # Parse question to detect two-part structure
        parsed = parse_two_part_question(question)

        # Customize system prompt based on question structure
        if parsed['is_two_part']:
            system_content = """You are an expert document analyst specializing in ILO project quality assessment.

**CRITICAL INSTRUCTION FOR TWO-PART QUESTIONS:**

This question has TWO PARTS with different priorities:
- **Part 2 (Specific Focus - PRIMARY)**: The critical question that requires detailed analysis. Drives the final answer.
- **Part 1 (Broader Context)**: Provides the general framework.

**YOUR ANALYSIS APPROACH:**
1. **Answer Part 2 IN DEPTH FIRST.** This drives the final Respuesta.
2. **Then relate Part 2's findings back to Part 1** (broader context).
3. The final Respuesta is determined PRIMARILY by Part 2; Part 1 provides framing only.

**Response fields (returned via structured output):**
- "Respuesta": Yes / No / Partial / Not Found — based PRIMARILY on Part 2.
- "Razonamiento": Begin by answering Part 2 concisely, then briefly connect to Part 1. Max 120 words. Be terse — no filler, no restating the question.
- "Evidencia": Quotes supporting Part 2 FIRST, then supporting evidence for Part 1. Max 180 words. Trim quotes to the essential span; avoid long blocks.
- "parte_enfocada": Set to "Parte 2" when your analysis was driven by Part 2 (the expected default). Use "Ambas" only if both parts truly required equal weight. Use "Parte 1" only if Part 2 was unanswerable from the document.

**SCOPE LOCK (CRITICAL):**
The question names a specific subject or scope (e.g., "people with disabilities", "rural women", "child labor in mining", a specific region/sector/period). Your analysis must be STRICTLY limited to that exact named subject.
1. Identify the exact subject of the question before answering.
2. Evidence about related-but-different subjects (other vulnerable populations, other groups, other sectors, other regions, other periods) does NOT count as answering the question. Example: if the question is about disability, evidence about women, indigenous peoples, or youth does NOT satisfy it.
3. If the named subject is absent from the document but related subjects are extensively covered, Respuesta must be "Not Found" or "No" — NOT "Partial" and NOT "Yes". Under-inclusion is preferred over over-inclusion.
4. In your Razonamiento, explicitly state which exact subject was analyzed and confirm the evidence concerns that subject (not a related one).

**EVIDENCE-ROLE FILTER & DECISION GATE FOR PART 2 (APPLY FIRST — DRIVES THE Respuesta):**

Before choosing a Respuesta, classify every candidate evidence passage about the Part-2 subject (e.g., personas con discapacidad, género, pueblos indígenas, rural women) as either FRAMING or DEDICATED.

FRAMING mention (does NOT count toward Part 2, even if the subject is named):
  - Overall objective / impact statement naming multiple groups
  - Stakeholder, consultation, or research participant lists
  - Monitoring scope enumerations ("...among others", "...including X, Y, Z")
  - Boilerplate inclusion language
  - Any passage where the subject appears in a list of 3+ groups without dedicated follow-up

DEDICATED element (counts toward Part 2):
  A. A sub-objective, outcome, or output whose title/purpose names the subject
  B. An indicator disaggregated by or specifically targeting the subject
  C. An activity whose primary purpose addresses the subject
  D. A budget line or resource allocation for the subject
  E. A quantifiable target for the subject

Count of DEDICATED elements (A–E) — NOT raw mention count — drives Respuesta:
  - 0 dedicated elements (only FRAMING mentions)      → "No" or "Not Found"
  - 1–2 dedicated elements                            → "Partial"
  - 3–4 dedicated elements                            → "Partial" (or "Yes" only if substantive and at-par with any claim/label in Part 1)
  - 5 dedicated elements with substantive evidence    → "Yes"

If every citable quote is a FRAMING mention, Respuesta MUST be "No" or "Not Found" — regardless of how many times the subject is named. Passing mentions in stakeholder lists, consultation rosters, or "among others" phrases DO NOT qualify for "Partial".

In "Evidencia", prefix each cited quote with [DEDICATED] or [FRAMING]. You may NOT justify "Partial" or "Yes" using only [FRAMING] quotes.

Always respond in Spanish. Focus 95% on Part 2, 5% on how it relates to Part 1. Part 1 exists only for brief framing; do NOT analyze Part 1 independently."""

            # Part 2 listed FIRST to exploit primacy bias toward the priority clause.
            # Full original question deliberately omitted — including it re-anchors the model on Part 1.
            user_content = f"""PREGUNTA CON DOS PARTES — RESPONDE PRIMERO LA PARTE 2.

**PARTE 2 (Enfoque Específico - PRIORITARIO):**
{parsed['part2']}

**PARTE 1 (Contexto General — solo para enmarcar):**
{parsed['part1']}

**Texto del Documento:**
{combined_text}

RECUERDA: Enfoca tu análisis PRINCIPALMENTE en la Parte 2 (pregunta específica), luego explica cómo se relaciona con la Parte 1 (contexto general). Marca "parte_enfocada" como "Parte 2" salvo que el documento no permita responderla."""

            response_format = {"type": "json_schema", "json_schema": RUBRIC_SCHEMA_TWO_PART}

        else:
            # Original single-question prompt
            system_content = """You are an expert document analyst. Analyze the document against the given question and provide a structured JSON response with exactly this format and always respond in Spanish:
            {
                "Respuesta": "Yes/No/Partial/Not Found",
                "Razonamiento": "Concise explanation (max 120 words, terse)",
                "Evidencia": "Trimmed text excerpts supporting the answer (max 180 words)"
            }

**SCOPE LOCK (CRITICAL):**
The question names a specific subject/scope (e.g., a specific population like people with disabilities; a specific sector, region, or period). Your analysis must be STRICTLY limited to that exact named subject.
1. Identify the exact subject of the question before answering.
2. Evidence about related-but-different subjects (other vulnerable populations, other groups, other sectors/regions/periods) does NOT count. Example: if the question is about disability, evidence about women, indigenous peoples, or youth does NOT satisfy it.
3. If the named subject is absent but related subjects are extensively covered, Respuesta must be "Not Found" or "No" — NOT "Partial" and NOT "Yes". Under-inclusion is preferred over over-inclusion.
4. In your Razonamiento, explicitly state which exact subject was analyzed and confirm the evidence concerns that subject (not a related one).

**EVIDENCE-ROLE FILTER & DECISION GATE (APPLY FIRST — DRIVES Respuesta):**

Classify every candidate evidence passage about the named specific subject as either FRAMING or DEDICATED.

FRAMING mention (does NOT count, even if the subject is named):
  - Overall objective / impact statement naming multiple groups
  - Stakeholder, consultation, or research participant lists
  - Monitoring scope enumerations ("...among others", "...including X, Y, Z")
  - Boilerplate inclusion language
  - Any passage where the subject appears in a list of 3+ groups without dedicated follow-up

DEDICATED element (counts):
  A. A sub-objective, outcome, or output whose title/purpose names the subject
  B. An indicator disaggregated by or targeting the subject
  C. An activity whose primary purpose addresses the subject
  D. A budget line or resource allocation for the subject
  E. A quantifiable target for the subject

Count of DEDICATED elements (A–E) — NOT raw mention count — drives Respuesta:
  - 0 (only FRAMING mentions)                        → "No" or "Not Found"
  - 1–2                                              → "Partial"
  - 3–4                                              → "Partial" (or "Yes" only if substantive)
  - 5 with substantive evidence                      → "Yes"

If every citable quote is a FRAMING mention, Respuesta MUST be "No" or "Not Found". In "Evidencia", prefix each quote with [DEDICATED] or [FRAMING]. You may NOT justify "Partial" or "Yes" using only [FRAMING] quotes."""

            user_content = f"Question: {question}\n\nDocument Text: {combined_text}"
            response_format = {"type": "json_schema", "json_schema": RUBRIC_SCHEMA_SINGLE}

        # Bump reasoning effort for two-part: the 70/30 weighting needs more deliberation.
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=3000,
            reasoning_effort="medium" if parsed['is_two_part'] else "minimal",
            response_format=response_format,
        )

        content = resp.choices[0].message.content
        if not content or not content.strip():
            return {
                'Pregunta': question,
                'Respuesta': 'Error',
                'Razonamiento': 'No se recibió respuesta del modelo.',
                'Evidencia': '',
                'Status': 'Error'
            }

        # Structured outputs guarantees valid JSON conforming to the schema.
        try:
            result = json.loads(content.strip())
        except json.JSONDecodeError:
            return {
                'Pregunta': question,
                'Respuesta': 'Error',
                'Razonamiento': 'Error al procesar la respuesta del modelo.',
                'Evidencia': '',
                'Status': 'Error'
            }

        # Flag rows where the model failed to commit to Part 2 focus despite a two-part question.
        status = 'Success'
        if parsed['is_two_part'] and result.get('parte_enfocada') != 'Parte 2':
            status = 'Partial'

        return {
            'Pregunta': question,
            'Respuesta': result.get('Respuesta', 'Not Found'),
            'Razonamiento': result.get('Razonamiento', ''),
            'Evidencia': result.get('Evidencia', ''),
            'Status': status,
        }

    except Exception as e:
        return {
            'Pregunta': question,
            'Respuesta': 'Error',
            'Razonamiento': f'Analysis failed: {str(e)}',
            'Evidencia': '',
            'Status': 'Error'
        }

def _critic_impl(question, answer, reasoning, evidence, document_text=""):
    """Shared critic implementation used by both public wrappers below.

    Uses structured output (JSON schema) to force the model to commit to each A–E
    element explicitly. The code layer then renders the display text and overrides
    the verdict when it is inconsistent with the A–E count (e.g., model said Partial
    but all five elements are ausente → verdict forced to No).
    """
    try:
        doc_context = truncate_to_token_limit(document_text or "", max_tokens=110000, encoding_obj=encoding)
        parsed = parse_two_part_question(question)

        if parsed['is_two_part']:
            system_content = _CRITIC_TWO_PART_SYSTEM
            user_content = f"""PREGUNTA CON DOS PARTES — EVALÚA LA PARTE 2 EXCLUSIVAMENTE.

**PARTE 2 (Sujeto Específico - el único que debe evaluarse):**
{parsed['part2']}

**PARTE 1 (Contexto — solo referencia, NO evaluar, NO contar hacia A–E):**
{parsed['part1']}

**Respuesta del Documento:** {answer}
**Razonamiento del Documento:** {reasoning}
**Evidencia del Documento:** {evidence}

**Contexto Completo del Documento:**
{doc_context}

Completa los campos estructurados. Para CADA elemento A–E, decide presente/ausente basándote en si hay contenido del documento DEDICADO al sujeto de la Parte 2. El Objetivo General, outputs generales, listas de grupos consultados, y cualquier contenido de la Parte 1 NO cuentan."""
            schema = CRITIC_SCHEMA_TWO_PART
        else:
            system_content = _CRITIC_SINGLE_SYSTEM
            user_content = f"""Pregunta: {question}

**Respuesta del Documento:** {answer}
**Razonamiento del Documento:** {reasoning}
**Evidencia del Documento:** {evidence}

**Contexto Completo del Documento:**
{doc_context}

Completa los campos estructurados. Para CADA elemento A–E, decide presente/ausente basándote en si hay contenido dedicado al sujeto específico de la pregunta."""
            schema = CRITIC_SCHEMA_SINGLE

        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=2000,
            reasoning_effort="minimal",
            response_format={"type": "json_schema", "json_schema": schema},
        )

        content = resp.choices[0].message.content
        if not content or not content.strip():
            return "VEREDICTO: Keep\n\nNo se generó evaluación crítica."

        try:
            result = json.loads(content.strip())
        except json.JSONDecodeError:
            return "VEREDICTO: Keep\n\nError al procesar la evaluación crítica estructurada."

        return _apply_critic_gate_and_render(result, is_two_part=parsed['is_two_part'])

    except Exception as e:
        return f"VEREDICTO: Keep\n\nError en evaluación crítica: {str(e)}"


_CRITIC_TWO_PART_SYSTEM = """You are an expert in project quality appraisal (ILO standards).

You assess whether a document adequately addresses the SPECIFIC subject of the question. Your output is a structured JSON (enforced by schema). The verdict is derived mechanically from the A–E count.

**SUBJECT EXTRACTION (MANDATORY — DO THIS FIRST):**

Before filling any A–E field, populate:
- named_subject_specific: the MOST SPECIFIC subject the question asks about. If the question names a target population or theme (personas con discapacidad, género, pueblos indígenas, mujeres rurales, juventud, afrodescendientes, migrantes), that is the subject — always. Never use the broader framing clause.
- general_clause_ignored: if the question has a broader framing clause (e.g., "¿se identifica el objetivo general del programa/proyecto?"), put its text here. This clause is IGNORED in the evaluation. If the question has no broader clause, set this to "".

**EXAMPLES (study these carefully):**

Q: "¿Se identifica claramente el objetivo general del programa/proyecto y se integra la inclusión de personas con discapacidad a la par de la etiqueta CPO?"
→ named_subject_specific: "personas con discapacidad / inclusión de la discapacidad a la par de la etiqueta CPO"
→ general_clause_ignored: "¿Se identifica claramente el objetivo general del programa/proyecto?"

Q: "Se identifica el objetivo general. ¿Se aborda la inclusión de personas con discapacidad?"
→ named_subject_specific: "personas con discapacidad"
→ general_clause_ignored: "Se identifica el objetivo general."

Q: "¿El programa incluye actividades dirigidas a personas con discapacidad?"
→ named_subject_specific: "personas con discapacidad"
→ general_clause_ignored: ""

NEVER set named_subject_specific to "objetivo general", "marco lógico", or any other structural/framing concept when the question also names a population or theme. The population/theme ALWAYS wins.

**WHAT IS A "DEDICATED" ELEMENT FOR PART 2:**
For the EXACT named Part-2 subject (e.g., "personas con discapacidad", "género", "pueblos indígenas"), decide presente/ausente for each:
  A. Sub-objective / outcome / output whose title or primary purpose EXPLICITLY names the subject
  B. Indicator disaggregated by, or specifically targeting, the subject
  C. Activity whose PRIMARY purpose is to address the subject (not incidental mention)
  D. Budget line or resource allocation specifically for the subject
  E. Quantifiable target for the subject (e.g., "N personas con discapacidad beneficiadas")

**WHAT DOES NOT COUNT — always mark ausente:**
- Overall Objective / Specific Objective / Impact statements naming multiple groups (e.g., "inclusive economic development", "vulnerable populations including X, Y, Z")
- Stakeholder lists, consultation rosters, research participant lists where the subject appears among others
- Phrases like "...among others", "...including X, Y, Z", "...such as"
- Any passage where the subject appears in a list of 3+ groups without dedicated follow-up
- Part-1 (broader context) content — it is invisible to the A–E test
- General project activities that merely mention the subject once

**FOR EACH A–E ELEMENT:**
- If presente: dedicated_X_evidence MUST contain the verbatim quote (max 50 words) from the document. The quote must name the Part-2 subject specifically. If the quote is a list of 3+ groups, the element is NOT presente — mark ausente. If you cannot produce a qualifying verbatim quote, mark ausente.
- If ausente: dedicated_X_evidence should be "No aparece en el documento" or a brief gap note.

**verdict FIELD (your proposal — the code layer will override if inconsistent with A–E):**
  TOTAL=0 → "No" (or "Not Found" if subject is completely unmentioned)
  TOTAL=1-2 → "Partial"
  TOTAL=3-4 → "Partial" (or "Yes" only if evidence is substantive and at-par with any Part-1 claim)
  TOTAL=5 → "Yes"

**justification FIELD (Spanish, 2-4 sentences):**
Lead with the verdict: "El No se asignó debido a…" / "Se asignó Partial porque existen N elementos dedicados…" / "Se asignó Yes porque…".
PROHIBITED openings and phrasings:
- "el documento define claramente el Objetivo General…"
- "presenta múltiples outputs/actividades sobre [tema general]…"
- "si bien… sin embargo…" (any contrastive structure praising Part 1)
- Any mention of Overall Objective, Specific Objective, or broader-topic outputs as strengths
- Hedging softeners: "en cierta medida", "aunque de forma general", "podría considerarse adecuado"
Focus ENTIRELY on Part-2 gaps. If TOTAL=0, say so plainly; do not soften.

**recommendations FIELD (Spanish):**
Specific items needed to reach Yes. Example: "un sub-objetivo específico sobre personas con discapacidad, indicadores desagregados, línea presupuestaria dedicada, actividades con accesibilidad, meta cuantificable". Empty if verdict is Yes or Keep.

Focus 99% on Part 2. Part 1 is framing — do NOT evaluate it, do NOT count it toward A–E, do NOT cite it as a strength."""

_CRITIC_SINGLE_SYSTEM = """You are an expert in project quality appraisal (ILO standards).

You assess whether a document adequately addresses the SPECIFIC subject named in the question. Your output is a structured JSON (enforced by schema). The verdict is derived mechanically from the A–E count.

**SUBJECT EXTRACTION (MANDATORY — DO THIS FIRST):**

Before filling any A–E field, populate:
- named_subject_specific: the MOST SPECIFIC subject the question asks about. If the question names a target population or theme (personas con discapacidad, género, pueblos indígenas, mujeres rurales, juventud, afrodescendientes, migrantes), that is the subject — always. Never use the broader framing clause.
- general_clause_ignored: if the question has a broader framing clause (e.g., "¿se identifica el objetivo general del programa/proyecto?"), put its text here. This clause is IGNORED in the evaluation. If the question has no broader clause, set this to "".

**EXAMPLES (study these carefully):**

Q: "¿Se identifica claramente el objetivo general del programa/proyecto y se integra la inclusión de personas con discapacidad a la par de la etiqueta CPO?"
→ named_subject_specific: "personas con discapacidad / inclusión de la discapacidad a la par de la etiqueta CPO"
→ general_clause_ignored: "¿Se identifica claramente el objetivo general del programa/proyecto?"

Q: "Se identifica el objetivo general. ¿Se aborda la inclusión de personas con discapacidad?"
→ named_subject_specific: "personas con discapacidad"
→ general_clause_ignored: "Se identifica el objetivo general."

Q: "¿El programa incluye actividades dirigidas a personas con discapacidad?"
→ named_subject_specific: "personas con discapacidad"
→ general_clause_ignored: ""

NEVER set named_subject_specific to "objetivo general", "marco lógico", or any other structural/framing concept when the question also names a population or theme. The population/theme ALWAYS wins.

**WHAT IS A "DEDICATED" ELEMENT:**
For the EXACT named subject of the question, decide presente/ausente for each:
  A. Sub-objective / outcome / output whose title or primary purpose EXPLICITLY names the subject
  B. Indicator disaggregated by, or specifically targeting, the subject
  C. Activity whose PRIMARY purpose is to address the subject
  D. Budget line or resource allocation specifically for the subject
  E. Quantifiable target for the subject

**WHAT DOES NOT COUNT — always ausente:**
- Overall objectives or impact statements naming multiple groups
- Stakeholder, consultation, or research-participant lists
- Phrases like "...among others", "...including X, Y, Z"
- Any list of 3+ groups without dedicated follow-up

**FOR EACH A–E:**
- If presente: dedicated_X_evidence MUST be a verbatim quote (max 50 words) that names the subject specifically. If you cannot, mark ausente.
- If ausente: brief gap note or "No aparece en el documento".

**verdict GUIDANCE (your proposal — code may override):**
  TOTAL=0 → "No" (or "Not Found")
  TOTAL=1-2 → "Partial"
  TOTAL=3-4 → "Partial" (or "Yes" if substantive)
  TOTAL=5 → "Yes"

**justification FIELD:** Spanish, 2-4 sentences. Lead with verdict. PROHIBITED: opening with praise of broader-topic content, "sin embargo" contrastive structure, "si bien…" hedging. Focus on the named subject's gaps.

**recommendations FIELD:** Spanish, specific items needed to reach Yes. Empty for Yes/Keep."""


def analyze_question_with_critical_opinion_tab1(question, answer, reasoning, evidence, document_text=""):
    """TAB1-SPECIFIC wrapper. Delegates to the shared structured-output critic.

    The structured schema (CRITIC_SCHEMA_TWO_PART / CRITIC_SCHEMA_SINGLE) forces the
    model to commit to each A–E element; _apply_critic_gate_and_render then overrides
    the verdict at the code layer when the count and verdict disagree.
    """
    return _critic_impl(question, answer, reasoning, evidence, document_text)


def analyze_question_with_critical_opinion(question, answer, reasoning, evidence, document_text=""):
    """General wrapper. Delegates to the shared structured-output critic."""
    return _critic_impl(question, answer, reasoning, evidence, document_text)


def extract_section_number(question_text):
    """Extract section number from question text (e.g., '1.1 ¿Pregunta?' -> 1)"""
    import re
    match = re.match(r'(\d+)\.', str(question_text).strip())
    return int(match.group(1)) if match else None

def extract_subsection_number(question_text):
    """Extract full section.subsection from question text (e.g., '1.1 ¿Pregunta?' -> '1.1', '2.3 ¿Pregunta?' -> '2.3')"""
    import re
    match = re.match(r'(\d+\.\d+)', str(question_text).strip())
    return match.group(1) if match else None

def parse_subsection_for_sorting(subsection_str):
    """Convert subsection string to tuple for proper sorting (e.g., '1.1' -> (1, 1), '2.10' -> (2, 10))"""
    try:
        parts = str(subsection_str).split('.')
        return (int(parts[0]), int(parts[1]))
    except:
        return (999, 999)

def parse_question_sort_key(question_text):
    """Extract the full numeric prefix from question text as a tuple for fine-grained sorting.

    Examples: '1.1 ¿...?' -> (1, 1); '1.1.2 ¿...?' -> (1, 1, 2); '2.10 ¿...?' -> (2, 10).
    Unlike parse_subsection_for_sorting which caps at two levels, this preserves every
    numeric segment so nested numbering (1.1, 1.1.1, 1.1.2, 1.2) sorts naturally and
    does not collapse to a single subsection bucket.
    """
    import re
    match = re.match(r'^\s*(\d+(?:\.\d+)*)', str(question_text).strip())
    if not match:
        return (999, 999, 999)
    try:
        return tuple(int(p) for p in match.group(1).split('.'))
    except ValueError:
        return (999, 999, 999)

def synthesize_subsection_analysis(subsection_id, subsection_questions_df):
    """Synthesize subsection-level analysis from individual question answers within that subsection"""
    try:
        # Build context from individual question Q&A pairs
        qa_context = "\n\n".join([
            f"Pregunta {row['Pregunta']}:\nRespuesta: {row['Respuesta']}\nRazonamiento: {row['Razonamiento']}"
            for _, row in subsection_questions_df.iterrows()
        ])
        
        # Single efficient LLM call per subsection
        prompt = f"""Based on the following individual question answers from subsection {subsection_id} of a document evaluation, 
synthesize a comprehensive subsection-level analysis.

**IMPORTANT — STRUCTURED DATA IN EACH QUESTION:**
Each question's Razonamiento may begin with an enumeration line in the form:
  "Elementos dedicados [Parte 2] [<sujeto>]: A=<presente|ausente>, B=..., C=..., D=..., E=.... TOTAL=<N>."
where A=sub-objetivo/output, B=indicador, C=actividad dedicada, D=línea presupuestaria, E=meta cuantificable.
These enumerations are the authoritative evidence base — you MUST aggregate them, not restate them.

Subsection {subsection_id} - Individual Q&A:
{qa_context}

Provide a concise subsection-level analysis (1-2 paragraphs) that:
1. Identifies the specific subject(s) evaluated across the questions (e.g., personas con discapacidad, género).
2. Aggregates the A–E pattern: for each element (A through E), state how many questions found it presente vs ausente. Name the systematic gap (e.g., "en 4 de 5 preguntas la línea presupuestaria y la meta cuantificable están ausentes").
3. Synthesizes 2–3 concrete, evidence-backed strengths or gaps.
4. Provides a clear overall assessment (Yes / Partial / No distribution across the subsection).

If different questions in the subsection address different subjects, aggregate A–E separately per subject.

Format as JSON with exactly this structure:
{{"subsection_analysis": "Spanish text, 1-2 paragraphs, ending with the aggregated A-E summary"}}

Return ONLY the JSON, no other text."""

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert document analyst. Synthesize individual question findings into clear subsection-level insights. Always respond in Spanish."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=1500,
            reasoning_effort="minimal"
        )
        
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return result.get('subsection_analysis', 'Error en síntesis')
    
    except Exception as e:
        return f"Error generando análisis de subsección: {str(e)}"

def synthesize_section_analysis(section_num, subsection_analyses_dict):
    """Synthesize section-level analysis from subsection analyses"""
    try:
        # Build context from subsection analyses
        subsection_context = "\n\n".join([
            f"Subsección {subsec_id}:\n{analysis}"
            for subsec_id, analysis in sorted(subsection_analyses_dict.items(), key=lambda x: parse_subsection_for_sorting(x[0]))
        ])
        
        # Single efficient LLM call per section
        prompt = f"""Based on the following subsection analyses from section {section_num} of a document evaluation, 
synthesize a comprehensive section-level analysis.

**IMPORTANT — STRUCTURED DATA IN EACH SUBSECTION:**
Each subsection analysis may include an aggregated A–E summary (A=sub-objetivo, B=indicador, C=actividad, D=presupuesto, E=meta cuantificable, per specific subject such as personas con discapacidad, género, pueblos indígenas). Roll these up at the section level: identify which elements are systematically missing across the entire section, and for which subjects.

Section {section_num} - Subsection Analyses:
{subsection_context}

Provide a detailed section-level analysis (2-3 paragraphs) that:
1. Rolls up the A–E patterns across subsections: which dedicated elements are consistently absent for which specific subjects?
2. Identifies overarching structural gaps (e.g., "budget allocation and quantifiable targets are absent across inclusion-focused questions in this section").
3. Provides strategic recommendations prioritized by which missing elements would have the largest effect if added.

Format as JSON with exactly this structure:
{{"section_analysis": "Spanish text, 2-3 paragraphs, explicitly naming which A-E elements are the most common gaps"}}

Return ONLY the JSON, no other text."""

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert document analyst. Synthesize subsection findings into clear, actionable section-level insights. Always respond in Spanish."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=2000,
            reasoning_effort="minimal"
        )
        
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return result.get('section_analysis', 'Error en síntesis')
    
    except Exception as e:
        return f"Error generando análisis de sección: {str(e)}"

def synthesize_critical_evaluation_subsection(subsection_id, critical_opinions_df):
    """Synthesize critical evaluations at subsection level from individual question critical opinions"""
    try:
        # Build context from individual critical opinions
        critical_context = "\n\n".join([
            f"Pregunta {row['Pregunta']}:\nEvaluación Crítica: {row['Evaluación Crítica']}"
            for _, row in critical_opinions_df.iterrows()
        ])
        
        # Single efficient LLM call per subsection
        prompt = f"""Based on the following individual critical evaluations from subsection {subsection_id}, 
synthesize a comprehensive subsection-level critical assessment.

**IMPORTANT — STRUCTURED DATA IN EACH CRITICAL EVALUATION:**
Each critical evaluation may begin with an enumeration line in the form:
  "Elementos dedicados [Parte 2] [<sujeto>]: A=..., B=..., C=..., D=..., E=.... TOTAL=<N>."
and may include audit notes like "[Ajuste automático: ... obliga a 'No']" or "[Cláusula general ignorada: ...]" or "[Downgrade por evidencia FRAMING: ...]". Read these as authoritative — they record where the automated grading intervened.

Subsection {subsection_id} - Individual Critical Evaluations:
{critical_context}

Provide a concise subsection-level critical assessment (1-2 paragraphs) that:
1. Aggregates the A–E pattern across the subsection and identifies the most common absent elements per specific subject.
2. Counts and reports how many questions received an automatic verdict override (e.g., "3 de 5 preguntas tuvieron veredicto ajustado a 'No' por TOTAL=0") — this is a strong signal of systematic weakness.
3. Flags FRAMING-evidence downgrades and cláusula-general ignoradas that reveal where the document relies on boilerplate instead of dedicated attention.
4. Recommends the 2–3 highest-priority interventions for the subsection (tied to the missing A–E elements).

Format as JSON with exactly this structure:
{{"critical_evaluation": "Spanish text, 1-2 paragraphs, including explicit counts of overrides and the dominant missing element"}}

Return ONLY the JSON, no other text."""

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert critic of project quality. Synthesize individual critical opinions into clear subsection-level critical assessment. Be direct about inadequacies. Always respond in Spanish."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=1500,
            reasoning_effort="minimal"
        )
        
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return result.get('critical_evaluation', 'Error en síntesis crítica')
    
    except Exception as e:
        return f"Error generando evaluación crítica de subsección: {str(e)}"

def synthesize_critical_evaluation_section(section_num, critical_subsection_dict):
    """Synthesize critical evaluations at section level from subsection critical assessments"""
    try:
        # Build context from subsection critical evaluations
        critical_context = "\n\n".join([
            f"Subsección {subsec_id}:\n{evaluation}"
            for subsec_id, evaluation in sorted(critical_subsection_dict.items(), key=lambda x: parse_subsection_for_sorting(x[0]))
        ])
        
        # Single efficient LLM call per section
        prompt = f"""Based on the following subsection critical evaluations from section {section_num}, 
synthesize a comprehensive section-level critical assessment.

**IMPORTANT — STRUCTURED DATA IN EACH SUBSECTION EVALUATION:**
Each subsection critical evaluation may reference A–E aggregated patterns (A=sub-objetivo, B=indicador, C=actividad, D=presupuesto, E=meta cuantificable) and may report automatic verdict overrides ("TOTAL=0 obliga a 'No'"), FRAMING downgrades, or cláusula-general ignoradas. Treat these as hard signals, not narrative flourishes.

Section {section_num} - Subsection Critical Evaluations:
{critical_context}

Provide a detailed section-level critical assessment (2-3 paragraphs) that:
1. Rolls up the A–E absences across the entire section: which elements are most systematically missing, and for which specific subjects?
2. Counts and reports the total automatic verdict overrides across the section (a high count signals a systemic issue, not isolated gaps).
3. Identifies cross-subsection patterns of reliance on FRAMING mentions or general-clause substitution.
4. Provides a prioritized list of strategic actions — ordered by which missing elements would close the most gaps if added.

Format as JSON with exactly this structure:
{{"critical_evaluation": "Spanish text, 2-3 paragraphs, grounded in the aggregated A-E counts and override counts"}}

Return ONLY the JSON, no other text."""

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert critic of project quality. Synthesize subsection critical findings into clear, actionable section-level critical assessment. Be direct and strategic. Always respond in Spanish."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=2000,
            reasoning_effort="minimal"
        )
        
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return result.get('critical_evaluation', 'Error en síntesis crítica')
    
    except Exception as e:
        return f"Error generando evaluación crítica de sección: {str(e)}"

def create_results_download_with_sections(results_df, subsection_analyses, subsection_critical_evaluations, section_analyses, section_critical_evaluations, filename_base="appraisal_checklist"):
    """Create ZIP file with results including three separate sheets for questions, subsections, and sections"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_urls': False}}) as writer:
            workbook = writer.book
            
            # Format definitions
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#002F6C',
                'font_color': 'white',
                'border': 1,
                'valign': 'vcenter',
                'text_wrap': True
            })
            section_header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#0072CE',
                'font_color': 'white',
                'border': 1,
                'valign': 'vcenter',
                'text_wrap': True,
                'font_size': 12
            })
            subsection_header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4A90E2',
                'font_color': 'white',
                'border': 1,
                'valign': 'vcenter',
                'text_wrap': True,
                'font_size': 11
            })
            merged_format = workbook.add_format({
                'border': 1,
                'valign': 'top',
                'text_wrap': True,
                'bg_color': '#F5F5F5'
            })
            subsection_merged_format = workbook.add_format({
                'border': 1,
                'valign': 'top',
                'text_wrap': True,
                'bg_color': '#E8F4F8',
                'italic': True
            })
            normal_format = workbook.add_format({
                'border': 1,
                'valign': 'top',
                'text_wrap': True
            })
            
            # Extract and ensure sorting columns exist
            if '_section' not in results_df.columns:
                results_df['_section'] = results_df['Pregunta'].apply(extract_section_number)
            if '_subsection' not in results_df.columns:
                results_df['_subsection'] = results_df['Pregunta'].apply(extract_subsection_number)
            if '_sort_key' not in results_df.columns:
                results_df['_sort_key'] = results_df['_subsection'].apply(parse_subsection_for_sorting)
            
            # Always re-derive _sort_key from the question text so persisted session-state
            # DataFrames (which may have been built with the old two-level sort key) are
            # re-sorted by the full numeric prefix.
            results_df['_sort_key'] = results_df['Pregunta'].apply(parse_question_sort_key)

            # Primary sort by full numeric prefix; _orig_idx breaks ties if available.
            sort_cols = ['_sort_key']
            if '_orig_idx' in results_df.columns:
                sort_cols.append('_orig_idx')
            results_df_sorted = results_df.sort_values(by=sort_cols).reset_index(drop=True)
            
            # ===== SHEET 1: Questions (Preguntas) =====
            sheet_questions = workbook.add_worksheet('1. Preguntas')
            questions_data = results_df_sorted[['_subsection', 'Pregunta', 'Respuesta', 'Razonamiento', 'Evidencia', 'Status']].copy()
            questions_data.columns = ['Subsección', 'Pregunta', 'Respuesta', 'Razonamiento', 'Evidencia', 'Status']
            questions_data.to_excel(writer, index=False, sheet_name='1. Preguntas', startrow=0)

            # Format headers
            for col_num, value in enumerate(questions_data.columns.values):
                writer.sheets['1. Preguntas'].write(0, col_num, value, header_format)

            # Set column widths
            writer.sheets['1. Preguntas'].set_column('A:A', 12)  # Subsección
            writer.sheets['1. Preguntas'].set_column('B:B', 35)  # Pregunta
            writer.sheets['1. Preguntas'].set_column('C:C', 12)  # Respuesta
            writer.sheets['1. Preguntas'].set_column('D:D', 55)  # Razonamiento (includes critical evaluation)
            writer.sheets['1. Preguntas'].set_column('E:E', 40)  # Evidencia
            writer.sheets['1. Preguntas'].set_column('F:F', 10)  # Status
            
            # ===== SHEET 2: Subsection Analysis (Análisis por Subsección) =====
            sheet_subsections = workbook.add_worksheet('2. Análisis Subsecciones')
            
            # Get unique subsections in sorted order
            subsections_sorted = sorted(results_df_sorted['_subsection'].dropna().unique(), key=parse_subsection_for_sorting)
            
            # Write headers
            subsec_headers = ['Subsección', 'Sección', 'Análisis de Subsección']
            for col_num, header in enumerate(subsec_headers):
                sheet_subsections.write(0, col_num, header, header_format)
            
            current_row = 1
            for subsection_id in subsections_sorted:
                section_num = extract_section_number(subsection_id)
                analysis_text = subsection_analyses.get(subsection_id, "No se generó análisis")
                critical_text = subsection_critical_evaluations.get(subsection_id, "No se generó evaluación crítica")
                
                sheet_subsections.write(current_row, 0, subsection_id, normal_format)
                sheet_subsections.write(current_row, 1, f"Sección {section_num}", normal_format)
                sheet_subsections.write(current_row, 2, analysis_text, subsection_merged_format)
                sheet_subsections.write(current_row, 3, critical_text, subsection_merged_format)
                current_row += 1
            
            # Set column widths
            sheet_subsections.set_column('A:A', 12)  # Subsección
            sheet_subsections.set_column('B:B', 15)  # Sección
            sheet_subsections.set_column('C:C', 80)  # Análisis
            sheet_subsections.set_column('D:D', 80)  # Evaluación Crítica
            
            # ===== SHEET 3: Section Analysis (Análisis por Sección) =====
            sheet_sections = workbook.add_worksheet('3. Análisis Secciones')
            
            # Get unique sections in sorted order
            sections_sorted = sorted(results_df_sorted['_section'].dropna().unique())
            
            # Write headers
            section_headers = ['Sección', 'Análisis de Sección', 'Evaluación Crítica']
            for col_num, header in enumerate(section_headers):
                sheet_sections.write(0, col_num, header, header_format)
            
            current_row = 1
            for section_num in sections_sorted:
                section_num = int(section_num)
                analysis_text = section_analyses.get(section_num, "No se generó análisis")
                critical_text = section_critical_evaluations.get(section_num, "No se generó evaluación crítica")
                
                sheet_sections.write(current_row, 0, f"Sección {section_num}", section_header_format)
                sheet_sections.write(current_row, 1, analysis_text, merged_format)
                sheet_sections.write(current_row, 2, critical_text, merged_format)
                current_row += 1
            
            # Set column widths
            sheet_sections.set_column('A:A', 15)  # Sección
            sheet_sections.set_column('B:B', 90)  # Análisis
            sheet_sections.set_column('C:C', 90)  # Evaluación Crítica
        
        excel_buffer.seek(0)
        zipf.writestr(f"{filename_base}_results.xlsx", excel_buffer.getvalue())

        # Add original template if available
        try:
            with open('./Appraisal Checklist_2025 es-419.xlsx', 'rb') as f:
                template_data = f.read()
                zipf.writestr(f"{filename_base}_rubric_template.xlsx", template_data)
        except FileNotFoundError:
            pass
        
        # Add summary report
        summary = f"""
Resumen del Análisis de la Lista de la Valoración Preliminar de la Calidad
===========================================================

Archivo Excel generado con 3 hojas:
  1. Preguntas: Análisis individual de cada pregunta (ordenado por subsección)
  2. Análisis Subsecciones: Síntesis de preguntas agrupadas por subsección
  3. Análisis Secciones: Síntesis ejecutiva por sección

Total de preguntas analizadas: {len(results_df)}
Análisis exitosos: {len(results_df[results_df['Status'] == 'Success'])}
Análisis fallidos: {len(results_df[results_df['Status'] == 'Error'])}

Subsecciones analizadas: {len(subsection_analyses)}
Secciones analizadas: {len(section_analyses)}

Distribución de respuestas:
{results_df['Respuesta'].value_counts().to_string()}
        """
        zipf.writestr(f"{filename_base}_summary.txt", summary)
    
    zip_buffer.seek(0)
    return zip_buffer

with tab1:
    st.header("📋 Valoración Preliminar de Calidad de Proyectos (Preliminary Project Quality Appraisal)")
    
    # Load questions
    df_appraisal, error_msg = load_appraisal_questions()
    
    if error_msg:
        st.error(error_msg)
        st.stop()
    
    # Download button for the rubric file (directly on page, no expander)
    # Download button for the rubric file (directly on page, no expander)
    try:
        with open('./Appraisal Checklist_2025 es-419.xlsx', 'rb') as f:
            st.download_button(
                label="📥 Descargar archivo rúbrica de Valoración preliminar",
                data=f,
                file_name="Appraisal Checklist_2025 es-419.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_appraisal_rubric"
            )
    except FileNotFoundError:
        st.warning("Archivo de rúbrica no disponible para descarga.")
    
    # Instrucciones
    st.markdown("""
    ### Instrucciones
    
    1. **Subir documento**: Selecciona un Documento de Diseño de Proyecto en archivo de formato DOCX para el análisis de valoración preliminar de calidad.
    2. **Procesar**: Haz clic en 'Analizar documento' para iniciar la evaluación
    3. **Revisar resultados**: Examina los resultados del análisis en la tabla interactiva
    4. **Descargar**: Obtén todos los resultados y las pruebas en un archivo ZIP
    
    ---
    """)
    
  
    # Initialize session state for Tab 1 results persistence
    if 'tab1_results_df' not in st.session_state:
        st.session_state['tab1_results_df'] = None
    if 'tab1_doc_stats' not in st.session_state:
        st.session_state['tab1_doc_stats'] = None

    # Initialize session state for Tab 1
    if 'document_extracted_tab1' not in st.session_state:
        st.session_state['document_extracted_tab1'] = False
    if 'selected_sections_tab1' not in st.session_state:
        st.session_state['selected_sections_tab1'] = []
    
    # Document upload
    st.subheader("📄 Carga de documento")
    
    # Warning box about document requirements
    st.warning("""
    **⚠️ Requisitos importantes para la carga de documentos:**
    
    **📝 Formato del documento:**
    - Solo se aceptan archivos en formato **.docx** (Word 2007 o posterior)
    - El documento debe estar **correctamente formateado** usando los estilos de encabezado de Word (Heading 1, Heading 2, etc.)
    - **CRÍTICO:** Las secciones del documento deben estar identificadas con **encabezados usando estilos estándar de Word**. Sin encabezados apropiados, el texto no se extraerá correctamente y las secciones no se identificarán.
    
    **📊 Límites de contexto:**
    - El sistema procesa hasta **110,000 tokens** (~440,000 caracteres, aproximadamente **150-200 páginas**) por documento
    - Documentos que excedan este límite serán truncados automáticamente
    - Se recomienda dividir documentos muy extensos (más de ~180 páginas) en secciones más pequeñas si es necesario
    
    **✅ Mejores prácticas:**
    - Usa estilos de Word (Título 1, Título 2, etc.) para identificar secciones principales
    - Evita usar texto en negrita o mayúsculas como sustituto de encabezados
    - Asegúrate de que el documento esté guardado correctamente antes de subirlo
    """)
    
    uploaded_file = st.file_uploader(
        "Sube un DOCX para la evaluación:",
        type=["docx"],
        key="appraisal_file_uploader",
        help="Selecciona un documento de Word (.docx) para el análisis de valoración preliminar de calidad"
    )
    
    # Document Extraction Section
    st.markdown("---")
    st.markdown("### 📥 Extracción de Documento")
    
    if uploaded_file is not None:
        file_hash = hash(uploaded_file.getvalue())
        file_changed = st.session_state.get('last_file_hash_tab1') != file_hash
        
        if file_changed:
            st.session_state['document_extracted_tab1'] = False
            st.session_state['last_file_hash_tab1'] = None
        
        if st.button("🔍 Extraer Documento", key="extract_document_tab1", type="primary"):
            if uploaded_file is None:
                st.error("Por favor suba un archivo DOCX primero.")
                st.stop()
            
            with st.spinner("Extrayendo documento..."):
                try:
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
                    tmp_file.write(uploaded_file.read())
                    tmp_file.close()
                    
                    progress_bar = st.progress(0, text="Leyendo y extrayendo contenido del DOCX...")
                    doc_result = docx2python(tmp_file.name)
                    
                    # Use enhanced extraction
                    df, tables_data, extraction_stats = extract_docx_structure_enhanced(tmp_file.name)
                    progress_bar.progress(0.2, text="Documento cargado. Procesando estructura...")
                    
                    # Extract sections
                    header_1_values = df['header_1'].dropna().unique()
                    llm_summary_rows = []
                    
                    for idx, header in enumerate(header_1_values):
                        section_df = df[df['header_1'] == header].copy()
                        full_text = '\n'.join(section_df['content'].astype(str).tolist()).strip()
                        section_words = len(full_text.split())
                        section_paras = len(section_df[section_df['source_type'] == 'paragraph'])
                        section_tables = len(section_df[section_df['source_type'] == 'table'])
                        
                        llm_summary_rows.append({
                            'header_1': header,
                            'llm_paragraph': full_text if full_text else "",
                            'n_words': section_words,
                            'n_paragraphs': section_paras,
                            'n_tables': section_tables
                        })

                    progress_bar.progress(0.5, text="Secciones extraídas.")
                    
                    # Create exploded dataframe
                    llm_summary_df = pd.DataFrame(llm_summary_rows)
                    exploded_df = llm_summary_df.assign(
                        llm_paragraph=llm_summary_df['llm_paragraph'].str.split('\n')
                    ).explode('llm_paragraph')
                    exploded_df = exploded_df.reset_index(drop=True)
                    exploded_df = exploded_df[exploded_df['llm_paragraph'].str.strip() != '']
                    
                    # Get full text
                    full_document_text = "\n\n".join(exploded_df['llm_paragraph'].tolist())
                    
                    # Store in session state
                    file_size = os.path.getsize(tmp_file.name)
                    n_words = exploded_df['llm_paragraph'].str.split().str.len().sum()
                    n_paragraphs = len(exploded_df)
                    
                    st.session_state['full_document_text_tab1'] = full_document_text
                    st.session_state['appraisal_document_stats'] = {
                        'file_size': file_size,
                        'word_count': n_words,
                        'n_words': n_words
                    }
                    st.session_state['exploded_df_tab1'] = exploded_df
                    st.session_state['extraction_df_tab1'] = df
                    st.session_state['tables_data_tab1'] = tables_data
                    st.session_state['extraction_stats_tab1'] = extraction_stats
                    st.session_state['sections_df_tab1'] = llm_summary_df
                    st.session_state['selected_sections_tab1'] = list(header_1_values)  # Select all by default
                    st.session_state['last_file_hash_tab1'] = file_hash
                    st.session_state['document_extracted_tab1'] = True
                    
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                    
                    progress_bar.progress(1.0, text="Extracción completa.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error procesando el documento: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    st.stop()
        
        # Show extraction results if document is extracted
        if st.session_state.get('document_extracted_tab1', False) and not file_changed:
            st.success("✅ Documento extraído con éxito")
            
            # Download button for extracted document structure
            extraction_df = st.session_state.get('extraction_df_tab1', pd.DataFrame())
            if not extraction_df.empty:
                excel_data = to_excel(extraction_df)
                # Get filename from extraction_df or use default
                filename_base = extraction_df['filename'].iloc[0] if 'filename' in extraction_df.columns and not extraction_df['filename'].empty else "documento"
                filename_base = filename_base.replace('.docx', '').replace('.doc', '')
                st.download_button(
                    label="📥 Descargar estructura extraída del documento (Excel)",
                    data=excel_data,
                    file_name=f"estructura_documento_tab1_{filename_base}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_extraction_tab1"
                )
                st.caption("El archivo incluye todas las columnas de encabezados (header_1 a header_6), contenido, tipo de fuente, y metadatos de extracción.")
            
            # Display header_1 sections and their content
            extraction_df = st.session_state.get('extraction_df_tab1', pd.DataFrame())
            if not extraction_df.empty:
                with st.expander("📋 Ver estructura extraída del documento (encabezados nivel 1 y contenido)", expanded=False):
                    st.markdown("**Estructura del documento extraído (solo encabezados nivel 1):**")
                    
                    # Get unique header_1 values
                    header_1_sections = extraction_df[extraction_df['header_1'].notna() & (extraction_df['header_1'] != '')]['header_1'].unique()
                    
                    for h1 in header_1_sections:
                        st.markdown(f"### {h1}")
                        
                        # Get all content for this header_1 section
                        section_df = extraction_df[extraction_df['header_1'] == h1]
                        section_content = section_df[section_df['source_type'] == 'paragraph']['content'].tolist()
                        
                        # Display content
                        if section_content:
                            full_text = '\n\n'.join([str(c) for c in section_content if pd.notna(c) and str(c).strip()])
                            if full_text.strip():
                                st.text(full_text)
                                st.caption(f"Total: {len(full_text):,} caracteres")
                        else:
                            st.info("Esta sección no tiene contenido de párrafos extraído.")
                        
                        st.markdown("---")
            
            sections_df = st.session_state.get('sections_df_tab1', pd.DataFrame())
            
            if not sections_df.empty:
                header_1_values = sections_df['header_1'].tolist()
                
                # Section selector
                st.markdown("### 🔍 Selección de Secciones para Evaluación")
                st.info("Selecciona las secciones que deseas incluir en la evaluación. Por defecto, todas las secciones están seleccionadas.")
                
                # Guidance about section extraction
                if len(header_1_values) == 0:
                    st.error("⚠️ **No se detectaron secciones en el documento.** Esto puede deberse a que el documento no usa estilos de encabezado de Word (Heading 1, Heading 2, etc.). Por favor, verifica que tu documento tenga encabezados formateados correctamente.")
                elif len(header_1_values) < 3:
                    st.warning("⚠️ **Se detectaron pocas secciones** en el documento. Si esperabas más secciones, verifica que el documento use estilos de encabezado de Word (Heading 1, Heading 2, etc.) para identificar las secciones principales.")
                
                # Initialize selected sections if not exists
                if 'selected_sections_tab1' not in st.session_state:
                    st.session_state['selected_sections_tab1'] = list(header_1_values)
                
                # Section selection interface
                selected_sections = st.session_state.get('selected_sections_tab1', list(header_1_values)).copy()
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    if st.button("✅ Seleccionar Todas", key="select_all_sections_tab1"):
                        st.session_state['selected_sections_tab1'] = list(header_1_values)
                        st.rerun()
                    
                    if st.button("❌ Deseleccionar Todas", key="deselect_all_sections_tab1"):
                        st.session_state['selected_sections_tab1'] = []
                        st.rerun()
                
                with col1:
                    st.markdown("**Secciones disponibles:**")
                    for section in header_1_values:
                        section_info = sections_df[sections_df['header_1'] == section].iloc[0]
                        is_selected = section in selected_sections
                        
                        checkbox_label = f"**{section}** ({section_info['n_words']:,} palabras, {section_info['n_paragraphs']} párrafos)"
                        
                        checkbox_key = f"section_checkbox_{section}_tab1"
                        new_selection = st.checkbox(checkbox_label, value=is_selected, key=checkbox_key)
                        
                        if new_selection and section not in selected_sections:
                            selected_sections.append(section)
                        elif not new_selection and section in selected_sections:
                            selected_sections.remove(section)
                        
                        # Add expandable preview of extracted content
                        with st.expander(f"👁️ Ver contenido: {section}", expanded=False):
                            section_content = section_info['llm_paragraph']
                            if section_content and section_content.strip():
                                st.text_area(
                                    "Contenido extraído:",
                                    value=section_content,
                                    height=200,
                                    key=f"content_preview_{section}_tab1",
                                    label_visibility="collapsed"
                                )
                                st.caption(f"Total: {len(section_content):,} caracteres")
                            else:
                                st.info("Esta sección no tiene contenido extraído.")
                            
                            # Show table-extracted text
                            tables_data = st.session_state.get('tables_data_tab1', [])
                            section_tables = [t for t in tables_data if t.get('section') == section]
                            
                            if section_tables:
                                st.markdown("---")
                                st.markdown("#### 📊 Texto extraído desde tablas")
                                for table_info in section_tables:
                                    table_num = table_info.get('table_number', 'N/A')
                                    table_data = table_info.get('data', [])
                                    if table_data:
                                        # Format table as text
                                        table_text = '\n'.join([' | '.join(str(cell) for cell in row) for row in table_data])
                                        st.text_area(
                                            f"Tabla {table_num}:",
                                            value=table_text,
                                            height=150,
                                            key=f"table_preview_{section}_table{table_num}_tab1",
                                            label_visibility="collapsed"
                                        )
                                        st.caption(f"Tabla {table_num}: {len(table_data)} filas, {len(table_data[0]) if table_data else 0} columnas")
                            else:
                                st.markdown("---")
                                st.markdown("#### 📊 Texto extraído desde tablas")
                                st.info("No se encontraron tablas en esta sección.")
                
                # Update session state
                st.session_state['selected_sections_tab1'] = selected_sections
                
                # Show selection summary
                if selected_sections:
                    selected_df = sections_df[sections_df['header_1'].isin(selected_sections)]
                    total_selected_words = selected_df['n_words'].sum()
                    total_selected_paras = selected_df['n_paragraphs'].sum()
                    
                    # Estimate tokens
                    if encoding:
                        selected_text = "\n\n".join(selected_df['llm_paragraph'].tolist())
                        estimated_tokens = len(encoding.encode(selected_text))
                    else:
                        estimated_tokens = total_selected_words * 1.2
                    
                    st.success(f"✅ {len(selected_sections)} secciones seleccionadas | "
                              f"{total_selected_words:,} palabras | "
                              f"~{estimated_tokens:,} tokens estimados")
    
    # Filter section for questions (optional, for filtering which questions to analyze)
    st.markdown("---")
    st.subheader("🔍 Filtros de Preguntas (opcional)")
    st.info("Selecciona secciones y subsecciones de preguntas para un análisis enfocado. Deja en blanco para analizar todas las preguntas.")
    
    # Get unique sections and subsections from questions
    df_with_sections = df_appraisal.copy()
    df_with_sections['_section'] = df_with_sections['Pregunta_Realizada'].apply(extract_section_number)
    df_with_sections['_subsection'] = df_with_sections['Pregunta_Realizada'].apply(extract_subsection_number)
    
    # Section Names Mapping
    SECTION_NAMES = {
        1: "Pertinencia",
        2: "Apropiación y sostenibilidad",
        3: "Gestión orientada a los resultados",
        4: "Transparencia y rendición de cuentas",
        5: "Presentación de la propuesta"
    }

    all_sections = sorted(df_with_sections['_section'].dropna().unique())
    
    # Create two columns for filters
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        # Format section options with names
        section_options = [f"{int(s)}. {SECTION_NAMES.get(int(s), 'Sección ' + str(int(s)))}" for s in all_sections]
        
        selected_sections_filter = st.multiselect(
            "Selecciona Secciones:",
            options=section_options,
            key="filter_sections_tab1",
            help="Selecciona una o más secciones para analizar"
        )
    
    # Extract section numbers from filter selections
    filtered_section_nums = []
    if selected_sections_filter:
        for s in selected_sections_filter:
            try:
                # Extract number from "1. Name" format
                filtered_section_nums.append(int(s.split('.')[0]))
            except:
                pass

    # Filter available subsections based on selected sections
    if filtered_section_nums:
        # Show only subsections belonging to selected sections
        available_subsections_df = df_with_sections[df_with_sections['_section'].isin(filtered_section_nums)]
    else:
        # Show all subsections if no section is selected
        available_subsections_df = df_with_sections
        
    all_subsections = sorted(available_subsections_df['_subsection'].dropna().unique(), 
                            key=lambda x: parse_subsection_for_sorting(x) if pd.notna(x) else (999, 999))

    with col_filter2:
        selected_subsections_filter = st.multiselect(
            "Selecciona Subsecciones:",
            options=[f"Subsección {s}" for s in all_subsections],
            key="filter_subsections_tab1",
            help="Selecciona una o más subsecciones para analizar"
        )
    
    # Extract subsection IDs
    filtered_subsection_ids = [s.replace("Subsección ", "") for s in selected_subsections_filter] if selected_subsections_filter else None
    
    # Processing button
    st.markdown("---")
    st.markdown("### ⚙️ Procesamiento y Evaluación")
    
    # Warning about AI results verification
    st.warning("""
    **⚠️ Importante - Verificación de Resultados:**
    
    Los resultados generados por esta herramienta utilizan inteligencia artificial y deben ser **verificados y corroborados** antes de su uso.
    
    - La IA puede cometer errores, interpretaciones incorrectas o pasar por alto información relevante
    - Los análisis y puntuaciones son **sugerencias** basadas en el contenido del documento, no son definitivos
    - Se recomienda revisar manualmente las evidencias citadas y validar las conclusiones
    - Los resultados deben ser contrastados con conocimiento experto y documentación adicional cuando sea necesario
    
    Esta herramienta es un **asistente de análisis** que facilita la revisión, pero la responsabilidad final de la evaluación recae en el usuario.
    """)
    
    if st.button('🔍 Analizar documento', key="appraisal_process_button", type="primary"):
        # Check prerequisites
        if not st.session_state.get('document_extracted_tab1', False):
            st.error("❌ Por favor extrae el documento primero usando el botón 'Extraer Documento'.")
            st.stop()
        
        if uploaded_file is None:
            st.warning("⚠️ Por favor suba un archivo DOCX primero.")
            st.stop()
        
        # Get selected sections or use full document
        selected_sections = st.session_state.get('selected_sections_tab1', [])
        sections_df = st.session_state.get('sections_df_tab1', pd.DataFrame())
        
        if selected_sections and not sections_df.empty:
            # Filter to selected sections only
            selected_df = sections_df[sections_df['header_1'].isin(selected_sections)]
            document_text = "\n\n".join(selected_df['llm_paragraph'].tolist())
            st.info(f"📌 Evaluando {len(selected_sections)} secciones seleccionadas")
        else:
            # Fallback to full document
            document_text = st.session_state.get('full_document_text_tab1', '')
            if not selected_sections:
                st.warning("⚠️ No hay secciones seleccionadas. Usando documento completo.")
        
        if not document_text:
            st.error("No se pudo recuperar el texto del documento.")
            st.stop()
        
        # Get questions for analysis
        questions = df_appraisal['Pregunta_Realizada'].dropna().unique().tolist()
        
        # Apply filters if selected
        if filtered_section_nums or filtered_subsection_ids:
            questions_df_temp = df_appraisal[df_appraisal['Pregunta_Realizada'].isin(questions)].copy()
            questions_df_temp['_section'] = questions_df_temp['Pregunta_Realizada'].apply(extract_section_number)
            questions_df_temp['_subsection'] = questions_df_temp['Pregunta_Realizada'].apply(extract_subsection_number)
            
            # Filter by selected sections and/or subsections
            if filtered_section_nums and filtered_subsection_ids:
                # Both filters selected - apply AND logic
                filtered_questions = questions_df_temp[
                    (questions_df_temp['_section'].isin(filtered_section_nums)) |
                    (questions_df_temp['_subsection'].isin(filtered_subsection_ids))
                ]
            elif filtered_section_nums:
                # Only sections selected
                filtered_questions = questions_df_temp[questions_df_temp['_section'].isin(filtered_section_nums)]
            else:
                # Only subsections selected
                filtered_questions = questions_df_temp[questions_df_temp['_subsection'].isin(filtered_subsection_ids)]
            
            questions = filtered_questions['Pregunta_Realizada'].dropna().unique().tolist()
            
            # Show filter info
            st.info(f"📌 Análisis filtrado: {len(questions)} preguntas seleccionadas de {len(df_appraisal)}")
        
        if not questions:
            st.error("❌ No se encontraron preguntas para el análisis con los filtros aplicados.")
            st.stop()
        
        # Analyze questions
        st.markdown("### 🔍 Progreso del análisis")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        critical_opinions = {}
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all questions for document-based analysis first
            # TAB1: Using tab1-specific function that explicitly states when 2 parts are detected
            # Tag each future with its original index so within-subsection ordering survives
            # the parallel as_completed dispatch.
            future_to_question_doc = {
                executor.submit(analyze_question_with_llm_tab1, q, document_text): (idx, q)
                for idx, q in enumerate(questions)
            }

            # Process document-based analyses
            completed = 0
            for future in as_completed(future_to_question_doc):
                idx, question = future_to_question_doc[future]
                result = future.result()
                result['_orig_idx'] = idx
                results.append(result)
                completed += 1

                # Update progress
                progress = completed / len(questions)
                progress_bar.progress(progress * 0.5)  # First half of progress bar
                status_text.text(f"Análisis documentales: {completed}/{len(questions)} preguntas")

            # Create results DataFrame to get answers for critical evaluation
            results_df_temp = pd.DataFrame(results)

            # Now submit critical evaluations with answer, reasoning, evidence AND document context
            # TAB1: Using tab1-specific function that checks for proper two-part acknowledgment
            future_to_question_critical = {
                executor.submit(
                    analyze_question_with_critical_opinion_tab1,
                    row['Pregunta'],
                    row['Respuesta'],
                    row['Razonamiento'],
                    row['Evidencia'],
                    document_text  # Pass selected sections text for context
                ): row['Pregunta']
                for _, row in results_df_temp.iterrows()
            }
            
            # Process critical evaluations
            completed = 0
            for future in as_completed(future_to_question_critical):
                question = future_to_question_critical[future]
                critical_opinion = future.result()
                critical_opinions[question] = critical_opinion
                completed += 1
                
                # Update progress
                progress = completed / len(questions)
                progress_bar.progress(0.5 + progress * 0.5)  # Second half of progress bar
                status_text.text(f"Evaluaciones críticas: {completed}/{len(questions)} preguntas")

        # Apply critic verdict. The critic is authoritative and its body becomes the single
        # coherent Razonamiento — whether it overrides the answer or confirms it. This gives the
        # reader one reasoning column instead of an original reasoning plus a separate critique.
        # Respuesta Original preserves the pre-critic grade for audit.
        override_count = 0
        unparsed_count = 0
        for result in results:
            q = result.get('Pregunta')
            critic_raw = critical_opinions.get(q, "") or ""
            verdict, critic_body = parse_critic_verdict(critic_raw)
            original_answer = result.get('Respuesta', '')
            result['Respuesta Original'] = original_answer

            if verdict is None:
                unparsed_count += 1
            elif verdict != 'Keep' and verdict != original_answer:
                result['Respuesta'] = verdict
                override_count += 1

            if critic_body:
                result['Razonamiento'] = critic_body

            critical_opinions[q] = critic_body if critic_body else critic_raw

        if override_count:
            st.info(f"🔄 {override_count} respuesta(s) ajustada(s) por la evaluación crítica.")
        if unparsed_count:
            st.warning(
                f"⚠️ {unparsed_count} evaluación(es) crítica(s) sin veredicto parseable; "
                "se mantuvo la respuesta original."
            )

        # Create results DataFrame - sort by subsection for proper ordering
        results_df = pd.DataFrame(results)

        # Add critical opinions to dataframe
        results_df['Evaluación Crítica'] = results_df['Pregunta'].map(critical_opinions).fillna("No disponible")
        results_df['_section_num'] = results_df['Pregunta'].apply(extract_section_number)
        results_df['_subsection'] = results_df['Pregunta'].apply(extract_subsection_number)
        # Sort key uses the FULL numeric prefix (e.g., "1.1.2" -> (1,1,2)) so nested
        # numbering sorts naturally instead of collapsing to the two-level subsection.
        results_df['_sort_key'] = results_df['Pregunta'].apply(parse_question_sort_key)

        # Primary sort by full numeric prefix; _orig_idx breaks ties for questions that
        # share the same numeric prefix, keeping rubric order within that group.
        results_df = results_df.sort_values(by=['_sort_key', '_orig_idx']).reset_index(drop=True)
        
        # THREE-LEVEL SYNTHESIS: Question -> Subsection -> Section
        st.markdown("### 📈 Síntesis Multinivel")
        
        # Level 1: Subsection synthesis (grouping individual questions)
        st.info("Generando análisis a nivel de subsección...")
        subsections = sorted(results_df['_subsection'].dropna().unique(), key=parse_subsection_for_sorting)
        subsection_analyses = {}
        
        subsection_progress = st.progress(0)
        for idx, subsection_id in enumerate(subsections):
            subsection_df = results_df[results_df['_subsection'] == subsection_id].copy()
            
            # Generate subsection analysis from individual questions
            subsection_analysis = synthesize_subsection_analysis(
                subsection_id,
                subsection_df[['Pregunta', 'Respuesta', 'Razonamiento']]
            )
            subsection_analyses[subsection_id] = subsection_analysis
            
            subsection_progress.progress((idx + 1) / len(subsections))
        
        # Level 2: Critical evaluation synthesis at subsection level
        st.info("Generando evaluaciones críticas a nivel de subsección...")
        subsection_critical_evaluations = {}
        
        critical_subsection_progress = st.progress(0)
        for idx, subsection_id in enumerate(subsections):
            subsection_df = results_df[results_df['_subsection'] == subsection_id].copy()
            
            # Generate critical evaluation synthesis from individual critical opinions
            critical_evaluation = synthesize_critical_evaluation_subsection(
                subsection_id,
                subsection_df[['Pregunta', 'Evaluación Crítica']]
            )
            subsection_critical_evaluations[subsection_id] = critical_evaluation
            
            critical_subsection_progress.progress((idx + 1) / len(subsections))
        
        # Level 3: Section synthesis (grouping subsections)
        st.info("Generando análisis a nivel de sección...")
        sections = sorted(results_df['_section_num'].dropna().unique())
        section_analyses = {}
        
        section_progress = st.progress(0)
        for idx, section_num in enumerate(sections):
            section_num = int(section_num)
            # Get all subsections for this section
            section_subsections = {
                subsec_id: subsection_analyses[subsec_id]
                for subsec_id in subsections
                if subsec_id.startswith(f"{section_num}.")
            }
            
            # Generate section analysis from subsection analyses
            section_analysis = synthesize_section_analysis(
                section_num,
                section_subsections
            )
            section_analyses[section_num] = section_analysis
            
            section_progress.progress((idx + 1) / len(sections))
        
        # Level 4: Critical evaluation synthesis at section level
        st.info("Generando evaluaciones críticas a nivel de sección...")
        section_critical_evaluations = {}
        
        critical_section_progress = st.progress(0)
        for idx, section_num in enumerate(sections):
            section_num = int(section_num)
            # Get all critical subsection evaluations for this section
            section_critical_subsections = {
                subsec_id: subsection_critical_evaluations[subsec_id]
                for subsec_id in subsections
                if subsec_id.startswith(f"{section_num}.")
            }
            
            # Generate critical evaluation from critical subsection evaluations
            critical_evaluation = synthesize_critical_evaluation_section(
                section_num,
                section_critical_subsections
            )
            section_critical_evaluations[section_num] = critical_evaluation
            
            critical_section_progress.progress((idx + 1) / len(sections))
        
        # Store all results in session state
        st.session_state['tab1_results_df'] = results_df
        st.session_state['tab1_subsection_analyses'] = subsection_analyses
        st.session_state['tab1_subsection_critical_evaluations'] = subsection_critical_evaluations
        st.session_state['tab1_section_analyses'] = section_analyses
        st.session_state['tab1_section_critical_evaluations'] = section_critical_evaluations

        # Get document stats from session state
        doc_stats = st.session_state.get('appraisal_document_stats', {})
        st.session_state['tab1_doc_stats'] = {
            'file_size': doc_stats.get('file_size', 0),
            'word_count': doc_stats.get('word_count', 0)
        }
        
        # Display results
        st.markdown("### 📊 Resultados del análisis")
        # Interpretation guide — documents the A-E framework and verdict mapping.
        with st.expander("📖 ¿Cómo interpretar los resultados?", expanded=False):
            st.markdown("""
**Marco de evaluación A–E (para preguntas con sujeto específico)**

Cuando la pregunta nombra un sujeto o tema específico (ej. *personas con discapacidad*, *género*, *pueblos indígenas*, *mujeres rurales*), el sistema evalúa cinco elementos dedicados a ese sujeto:

- **A. Sub-objetivo / output** cuyo título o propósito nombra explícitamente al sujeto
- **B. Indicador** desagregado por o específico para el sujeto
- **C. Actividad** cuyo propósito principal es abordar al sujeto (no mención incidental)
- **D. Línea presupuestaria** o asignación de recursos específica para el sujeto
- **E. Meta cuantificable** para el sujeto (ej. "N personas con discapacidad beneficiadas")

Cada pregunta muestra una línea de enumeración al inicio de la Razonamiento:

> `Elementos dedicados [Parte 2] [<sujeto>]: A=<presente|ausente>, B=..., C=..., D=..., E=... . TOTAL=<N>.`

El **TOTAL** (de 0 a 5) determina la respuesta automáticamente:

| TOTAL | Respuesta | Interpretación |
|-------|-----------|----------------|
| 0 | **No** (o Not Found) | El sujeto no recibe atención dedicada; las menciones son solo contextuales |
| 1–2 | **Partial** | Inclusión parcial; existen algunos elementos dedicados pero faltan otros esenciales |
| 3–4 | **Partial** (o Yes si la evidencia es sustantiva y al-par con Parte 1) | Inclusión mayormente integrada |
| 5 | **Yes** | Inclusión completa y verificable |

**Qué cuenta como DEDICATED vs FRAMING**

- **DEDICATED** (cuenta para A–E): una frase que nombra al sujeto específicamente Y describe un elemento concreto (un output, indicador, actividad, línea de presupuesto, meta). Ejemplo: *"Output 2.3: 100 personas con discapacidad capacitadas en derechos laborales para 2027."*
- **FRAMING** (NO cuenta): menciones al sujeto dentro de listas de grupos, declaraciones generales o lenguaje de inclusión genérico. Ejemplos: *"incluyendo personas con discapacidad, mujeres, pueblos indígenas, entre otros"*, o *"consultas con organizaciones de personas con discapacidad"* dentro de una lista de actores.

**Notas de auditoría que puede ver en la Razonamiento**

- `[Ajuste automático: el modelo propuso \'…\' pero TOTAL=0 obliga a \'No\']` — el sistema detectó que el modelo hedged y corrigió el veredicto automáticamente.
- `[Downgrade por evidencia FRAMING: C (contains \'among others\')]` — el modelo marcó un elemento como presente pero su evidencia era en realidad una lista de grupos; fue reclasificado a ausente.
- `[Cláusula general ignorada: \'…\']` — el modelo identificó una cláusula de contexto general en la pregunta y confirmó que NO la evaluó (solo el foco específico).

**Para preguntas estructurales** (¿está claro el objetivo general? ¿está completo el marco lógico?), el marco A–E se aplica con flexibilidad: A representa la presencia del elemento estructural, y B–E reflejan indicadores, actividades, presupuesto y metas asociados. Un TOTAL=1 (solo A presente) puede indicar que el elemento estructural existe pero no está operacionalizado.
""")


        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de preguntas", len(results_df))
        with col2:
            success_count = len(results_df[results_df['Status'] == 'Success'])
            st.metric("Exitosas", success_count)
        with col3:
            error_count = len(results_df[results_df['Status'] == 'Error'])
            st.metric("Errores", error_count)
        with col4:
            if success_count > 0:
                yes_count = len(results_df[results_df['Respuesta'] == 'Yes'])
                st.metric("Respuestas 'Sí'", yes_count)
        
        # Results table
        st.markdown("#### 📋 Resultados detallados")
        
        # Filter options
        col1, col2 = st.columns([1, 1])
        with col1:
            response_filter = st.selectbox(
                "Filtrar por respuesta:",
                ['Todas'] + results_df['Respuesta'].unique().tolist(),
                key="response_filter"
            )
        
        # Apply filter
        filtered_df = results_df.copy()
        if response_filter != 'Todas':
            filtered_df = filtered_df[filtered_df['Respuesta'] == response_filter]
        
        # Display filtered results
        st.dataframe(
            filtered_df[['Pregunta', 'Respuesta', 'Razonamiento', 'Evidencia']], 
            use_container_width=True,
            height=400
        )
        
        # Download section
        st.markdown("### 📥 Descargar resultados")
        
        if len(results_df) > 0:
            # Get analyses from session state
            subsection_analyses = st.session_state.get('tab1_subsection_analyses', {})
            subsection_critical_evaluations = st.session_state.get('tab1_subsection_critical_evaluations', {})
            section_analyses = st.session_state.get('tab1_section_analyses', {})
            section_critical_evaluations = st.session_state.get('tab1_section_critical_evaluations', {})
            zip_buffer = create_results_download_with_sections(results_df, subsection_analyses, subsection_critical_evaluations, section_analyses, section_critical_evaluations)
            
            st.download_button(
                label="📦 Descargar resultados en ZIP",
                data=zip_buffer,
                file_name="appraisal_checklist_results.zip",
                mime="application/zip",
                key="appraisal_download_button"
            )
            
            st.success("✅ ¡Análisis completo! Usa el botón de descarga para obtener todos los resultados.")
        else:
            st.warning("⚠️ No hay resultados para descargar.")
    
    else:
        # Check if there are persisted results in session state
        if st.session_state.get('tab1_results_df') is not None:
            results_df = st.session_state['tab1_results_df']
            doc_stats = st.session_state.get('tab1_doc_stats', {})
            
            # Display persisted results
            st.markdown("### 📊 Resultados del análisis (guardados)")
            # Interpretation guide — documents the A-E framework and verdict mapping.
            with st.expander("📖 ¿Cómo interpretar los resultados?", expanded=False):
                st.markdown("""
**Marco de evaluación A–E (para preguntas con sujeto específico)**

Cuando la pregunta nombra un sujeto o tema específico (ej. *personas con discapacidad*, *género*, *pueblos indígenas*, *mujeres rurales*), el sistema evalúa cinco elementos dedicados a ese sujeto:

- **A. Sub-objetivo / output** cuyo título o propósito nombra explícitamente al sujeto
- **B. Indicador** desagregado por o específico para el sujeto
- **C. Actividad** cuyo propósito principal es abordar al sujeto (no mención incidental)
- **D. Línea presupuestaria** o asignación de recursos específica para el sujeto
- **E. Meta cuantificable** para el sujeto (ej. "N personas con discapacidad beneficiadas")

Cada pregunta muestra una línea de enumeración al inicio de la Razonamiento:

> `Elementos dedicados [Parte 2] [<sujeto>]: A=<presente|ausente>, B=..., C=..., D=..., E=... . TOTAL=<N>.`

El **TOTAL** (de 0 a 5) determina la respuesta automáticamente:

| TOTAL | Respuesta | Interpretación |
|-------|-----------|----------------|
| 0 | **No** (o Not Found) | El sujeto no recibe atención dedicada; las menciones son solo contextuales |
| 1–2 | **Partial** | Inclusión parcial; existen algunos elementos dedicados pero faltan otros esenciales |
| 3–4 | **Partial** (o Yes si la evidencia es sustantiva y al-par con Parte 1) | Inclusión mayormente integrada |
| 5 | **Yes** | Inclusión completa y verificable |

**Qué cuenta como DEDICATED vs FRAMING**

- **DEDICATED** (cuenta para A–E): una frase que nombra al sujeto específicamente Y describe un elemento concreto (un output, indicador, actividad, línea de presupuesto, meta). Ejemplo: *"Output 2.3: 100 personas con discapacidad capacitadas en derechos laborales para 2027."*
- **FRAMING** (NO cuenta): menciones al sujeto dentro de listas de grupos, declaraciones generales o lenguaje de inclusión genérico. Ejemplos: *"incluyendo personas con discapacidad, mujeres, pueblos indígenas, entre otros"*, o *"consultas con organizaciones de personas con discapacidad"* dentro de una lista de actores.

**Notas de auditoría que puede ver en la Razonamiento**

- `[Ajuste automático: el modelo propuso \'…\' pero TOTAL=0 obliga a \'No\']` — el sistema detectó que el modelo hedged y corrigió el veredicto automáticamente.
- `[Downgrade por evidencia FRAMING: C (contains \'among others\')]` — el modelo marcó un elemento como presente pero su evidencia era en realidad una lista de grupos; fue reclasificado a ausente.
- `[Cláusula general ignorada: \'…\']` — el modelo identificó una cláusula de contexto general en la pregunta y confirmó que NO la evaluó (solo el foco específico).

**Para preguntas estructurales** (¿está claro el objetivo general? ¿está completo el marco lógico?), el marco A–E se aplica con flexibilidad: A representa la presencia del elemento estructural, y B–E reflejan indicadores, actividades, presupuesto y metas asociados. Un TOTAL=1 (solo A presente) puede indicar que el elemento estructural existe pero no está operacionalizado.
""")


            
            if doc_stats:
                st.info(f"""
                **Resumen del documento analizado:**
                - Tamaño del archivo: {doc_stats.get('file_size', 0)/1024:.2f} KB
                - Número de palabras: {doc_stats.get('word_count', 0):,}
                """)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de preguntas", len(results_df))
            with col2:
                success_count = len(results_df[results_df['Status'] == 'Success'])
                st.metric("Exitosas", success_count)
            with col3:
                error_count = len(results_df[results_df['Status'] == 'Error'])
                st.metric("Errores", error_count)
            with col4:
                if success_count > 0:
                    yes_count = len(results_df[results_df['Respuesta'] == 'Yes'])
                    st.metric("Respuestas 'Sí'", yes_count)
            
            # Results table
            st.markdown("#### 📋 Resultados detallados")
            
            # Filter options
            col1, col2 = st.columns([1, 1])
            with col1:
                response_filter = st.selectbox(
                    "Filtrar por respuesta:",
                    ['Todas'] + results_df['Respuesta'].unique().tolist(),
                    key="response_filter_persisted"
                )
            
            # Apply filter
            filtered_df = results_df.copy()
            if response_filter != 'Todas':
                filtered_df = filtered_df[filtered_df['Respuesta'] == response_filter]
            
            # Display filtered results
            st.dataframe(
                filtered_df[['Pregunta', 'Respuesta', 'Razonamiento', 'Evidencia']], 
                use_container_width=True,
                height=400
            )
            
            # Download section
            st.markdown("### 📥 Descargar resultados")
            
            if len(results_df) > 0:
                # Get analyses from session state
                subsection_analyses = st.session_state.get('tab1_subsection_analyses', {})
                subsection_critical_evaluations = st.session_state.get('tab1_subsection_critical_evaluations', {})
                section_analyses = st.session_state.get('tab1_section_analyses', {})
                section_critical_evaluations = st.session_state.get('tab1_section_critical_evaluations', {})
                zip_buffer = create_results_download_with_sections(results_df, subsection_analyses, subsection_critical_evaluations, section_analyses, section_critical_evaluations)
                
                st.download_button(
                    label="📦 Descargar resultados en ZIP",
                    data=zip_buffer,
                    file_name="appraisal_checklist_results.zip",
                    mime="application/zip",
                    key="appraisal_download_button_persisted"
                )
            
            # Clear results button
            if st.button("🗑️ Limpiar resultados", key="clear_tab1_results"):
                st.session_state['tab1_results_df'] = None
                st.session_state['tab1_doc_stats'] = None
                st.rerun()
        else:
            if uploaded_file:
                st.info("👆 Haz clic en 'Analizar documento' para iniciar la evaluación de la valoración preliminar de calidad.")
            else:
                st.info("📁 Por favor sube un archivo DOCX para comenzar.")
