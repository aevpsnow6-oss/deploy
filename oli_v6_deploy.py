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

# --- Utility function for Excel export ---
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Filtered Data')
    processed_data = output.getvalue()
    return processed_data

# Use environment variables for API keys
# For local development - use .env file or set environment variables
# For Streamlit Cloud - set these in the app settings
openai_api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI - use only the older API version
import openai
openai.api_key = openai_api_key

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Function to get embeddings from OpenAI
def get_embedding_with_retry(text, model='text-embedding-3-large', max_retries=3, delay=1):
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(input=text, model=model)
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(delay)
    return None

# Function to find similar recommendations using embeddings

def find_similar_recommendations(query_embedding, index, doc_embeddings, structured_embeddings, score_threshold=0.5, top_n=20):
    # Normalize query embedding for cosine similarity
    query_embedding = np.array(query_embedding).reshape(1, -1)
    # Search the index
    try:
        distances, indices = index.search(query_embedding, index.ntotal)

        # Filter results based on the score threshold
        filtered_recommendations = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(structured_embeddings) and dist >= score_threshold:
                metadata = structured_embeddings[idx]
                recommendation = {
                    "recommendation": metadata["text"],
                    "similarity": float(dist),  # Convert to float for JSON serialization
                    "country": metadata["country"],
                    "year": metadata["year"],
                    "eval_id": metadata["eval_id"]
                }
                filtered_recommendations.append(recommendation)
            if len(filtered_recommendations) >= top_n:
                break
        return filtered_recommendations
    except Exception as e:
        st.error(f"Error in similarity search: {str(e)}")
        return []

# Function to find recommendations by term matching

def find_recommendations_by_term_matching(query, doc_texts, structured_embeddings, top_n=10):
    try:
        matched_recommendations = []
        query_lower = query.lower()
        for idx, text in enumerate(doc_texts):
            if isinstance(text, str) and query_lower in text.lower():
                if idx < len(structured_embeddings):
                    metadata = structured_embeddings[idx]
                    matched_recommendations.append({
                        "recommendation": text,
                        "country": metadata["country"],
                        "year": metadata["year"],
                        "eval_id": metadata["eval_id"]
                    })
        matched_recommendations = sorted(matched_recommendations, key=lambda x: len(str(x["recommendation"])))
        return matched_recommendations[:top_n]
    except Exception as e:
        st.error(f"Error in term matching: {str(e)}")
        return []

# ============= DOCX PARSING FUNCTIONS =============

# --- Begin: SimpleHierarchicalStore and RAG logic from megaparse_example.py ---
import pickle
from typing import List, Dict, Any
import numpy as np
import json
import os
import openai

class SimpleHierarchicalStore:
    def __init__(self, use_cache=True, cache_dir=None):
        self.documents = {}
        self.sections = {}
        self.paragraphs = []
        self.use_cache = use_cache
        self.cache_dir = cache_dir or os.getcwd()
        self.embedding_cache = {}
        self.query_cache = {}
        self.storage_dir = cache_dir or os.path.join(os.path.expanduser("~"), "document_store")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.cache_file = os.path.join(self.storage_dir, "embedding_cache.pkl")
        if use_cache:
            self._load_cache()
    def _load_cache(self):
        if not self.use_cache:
            return
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
            except Exception:
                self.embedding_cache = {}
        else:
            self.embedding_cache = {}
    def _hash_text(self, text):
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    def _save_cache(self):
        if not self.use_cache:
            return
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception:
            pass
    def get_embedding(self, text: str):
        if not text or text.isspace():
            return [0.0] * 1536
        if self.use_cache:
            text_hash = hash(text)
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
        try:
            response = openai.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            if self.use_cache:
                self.embedding_cache[hash(text)] = embedding
                if len(self.embedding_cache) % 100 == 0:
                    self._save_cache()
            return embedding
        except Exception:
            return [0.0] * 1536
    def add_documents(self, df, content_column='content', section_column='header_1', batch_size=20):
        doc_id = df['document_id'].iloc[0] if 'document_id' in df.columns else 'doc1'
        self.documents[doc_id] = {'embedding': self.get_embedding(' '.join(df[content_column].astype(str).tolist()))}
        for _, row in df.iterrows():
            section_id = row.get(section_column, '')
            if pd.isna(section_id):
                section_id = '_default_section'
            section_text = str(row.get(content_column, ''))
            if not section_text.strip():
                continue
            section_embedding = self.get_embedding(section_text)
            self.sections[(doc_id, section_id)] = {
                'text': section_text,
                'embedding': section_embedding
            }
            self.paragraphs.append({
                'text': section_text,
                'embedding': section_embedding,
                'document_id': doc_id,
                'section_id': section_id,
                'position': row.get('paragraph_number', 0)
            })
    def cosine_similarity(self, embedding1, embedding2):
        if not embedding1 or not embedding2:
            return 0
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    def score_rubric_directly(self, rubric_elements: Dict, top_n_paragraphs: int = 10) -> Dict:
        print(f"[score_rubric_directly] Evaluating criterion: {rubric_elements}")
        results = {}
        for criterion, descriptions in rubric_elements.items():
            print(f"[score_rubric_directly] Evaluating criterion: {criterion}")
            criterion_embedding = self.get_embedding(criterion)
            paragraph_scores = []
            for p in self.paragraphs:
                similarity = self.cosine_similarity(criterion_embedding, p['embedding'])
                paragraph_scores.append((p, similarity))
            paragraph_scores.sort(key=lambda x: x[1], reverse=True)
            top_paragraphs = paragraph_scores[:top_n_paragraphs]
            context_text = '\n\n---\n\n'.join([p[0]['text'] for p in top_paragraphs])
            try:
                analysis = self.analyze_criterion(criterion, context_text, descriptions)
                print(f"[score_rubric_directly] Analysis result for '{criterion}': {analysis}")
                results[criterion] = {
                    'analysis': analysis,
                    'context': context_text,
                    'score': analysis.get('score', 0),
                    'confidence': analysis.get('confidence', 0),
                    'top_paragraphs': [{'text': p[0]['text'], 'similarity': p[1]} for p in top_paragraphs[:3]]
                }
            except Exception as e:
                print(f"[score_rubric_directly] Exception for '{criterion}': {e}")
                results[criterion] = {
                    'analysis': {'error': str(e)},
                    'context': context_text,
                    'score': 0,
                    'confidence': 0
                }
        print(f"[score_rubric_directly] Final results: {results}")
        return results
    def analyze_criterion(self, criterion: str, context: str, descriptions: list) -> dict:
        print(f"[analyze_criterion] Called with criterion: {criterion}")
        print(f"[analyze_criterion] Context: {context[:200]} ...")
        print(f"[analyze_criterion] Descriptions: {descriptions}")
        prompt = f"""
        You are evaluating a document against a specific criterion. 
        Criterion: {criterion}
        Descriptions of scoring levels:
        {json.dumps(descriptions, indent=2)}
        Document content to evaluate:
        {context}
        Please analyze how well the document meets this criterion. Provide:
        1. A detailed analysis (2-3 paragraphs)
        2. A score from 1-5 (where 1 is lowest and 5 is highest)
        3. Key evidence from the document that supports your score
        4. Any recommendations for improvement
        5. A confidence level (0-1) indicating how confident you are in this assessment
        Format your response as a JSON object with the following keys:
        {"analysis": "your detailed analysis here", "score": numeric_score_between_1_and_5, "evidence": "key evidence from the document", "recommendations": "your recommendations for improvement", "confidence": confidence_level_between_0_and_1}
        Return only the JSON object, nothing else.
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are an expert document evaluator that provides detailed analysis and scoring based on specific criteria."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            print(f"[analyze_criterion] Raw response: {raw}")
            parsed = json.loads(raw)
            print(f"[analyze_criterion] Parsed response: {parsed}")
            return parsed
        except Exception as e:
            print(f"[analyze_criterion] Exception: {e}")
            return {'score': 0, 'analysis': f'Error: {str(e)}'}
# --- End: SimpleHierarchicalStore and RAG logic ---

# --- HIERARCHICAL RAG RUBRIC EVALUATION ---
def add_rubric_evaluation_section(exploded_df, toc, toc_hierarchy):
    """
    Add a new section for rubric-based evaluation of the document using hierarchical RAG pipeline.
    Allows users to select rubric type, choose criteria, evaluate selected sections, view results, and download as CSV/Excel.
    """
    import streamlit as st
    import pandas as pd
    from io import BytesIO
    from collections import defaultdict

    st.markdown("### Evaluación por Rúbrica (Hierarchical RAG)")

    # Initialize session state keys with unique prefixes to avoid conflicts
    session_keys = {
        'selected_sections': 'rubric_eval_selected_sections',
        'sections_confirmed': 'rubric_eval_sections_confirmed',
        'selected_criteria': 'rubric_eval_selected_criteria'
    }

    # --- Section Selection ---
    if session_keys['selected_sections'] not in st.session_state:
        st.session_state[session_keys['selected_sections']] = []
    if session_keys['sections_confirmed'] not in st.session_state:
        st.session_state[session_keys['sections_confirmed']] = False

    st.markdown("#### 1. Selección de Secciones para Evaluación")
    main_sections = []
    for level, headings in sorted(toc_hierarchy.items()):
        if headings and not main_sections:
            main_sections = headings
    valid_selected_sections = [s for s in st.session_state[session_keys['selected_sections']] if s in main_sections]
    if not valid_selected_sections and main_sections:
        valid_selected_sections = [main_sections[0]]
    selected_sections = st.multiselect(
        "Seleccione las secciones del documento que desea evaluar:",
        options=main_sections,
        default=valid_selected_sections,
        key="rubric_section_multiselect_eval"
    )
    if st.button("Confirmar Secciones para Evaluación"):
        if not selected_sections:
            st.warning("Por favor seleccione al menos una sección para evaluar.")
        else:
            st.session_state[session_keys['selected_sections']] = selected_sections
            st.session_state[session_keys['sections_confirmed']] = True
            st.success(f"Secciones confirmadas para evaluación: {', '.join(selected_sections)}")
    if st.session_state[session_keys['selected_sections']]:
        st.info(f"Secciones actualmente seleccionadas para evaluación: {', '.join(st.session_state[session_keys['selected_sections']])}")

    # --- Rubric Selection & Evaluation ---
    if st.session_state[session_keys['sections_confirmed']]:
        st.markdown("#### 2. Selección de Rúbrica y Criterios")
        rubric_type = st.selectbox(
            "Seleccione tipo de rúbrica para evaluación:",
            ["Participación (Engagement)", "Desempeño (Performance)"],
            index=0,
            key='rubric_type_tab2'
        )
        if rubric_type == "Participación (Engagement)":
            rubric_df = load_engagement_rubric()
            criteria_col = "Criterion"
            short_col = "crit_short"
            group_col = "Criterio"
        else:
            rubric_df = load_performance_rubric()
            criteria_col = "subdim"
            short_col = None
            group_col = "dimension"
        st.markdown("##### Estructura de la Rúbrica")
        rubric_groups = rubric_df.groupby(group_col)
        cols = st.columns([1, 3])
        with cols[0]:
            st.markdown("**Categorías**")
            categories = list(rubric_groups.groups.keys())
            selected_category = st.radio(
                "Seleccione una categoría:",
                categories,
                label_visibility="collapsed"
            )
        with cols[1]:
            st.markdown("**Criterios**")
            if selected_category:
                category_criteria = rubric_df[rubric_df[group_col] == selected_category]
                if session_keys['selected_criteria'] not in st.session_state:
                    st.session_state[session_keys['selected_criteria']] = {}
                all_criteria_in_category = st.checkbox(f"Seleccionar todos los criterios en '{selected_category}'")
                selected_criteria_ids = []
                for _, criterion_row in category_criteria.iterrows():
                    criterion_id = criterion_row[criteria_col]
                    criterion_name = criterion_row[short_col] if short_col and short_col in criterion_row else criterion_id
                    if criterion_id not in st.session_state[session_keys['selected_criteria']]:
                        st.session_state[session_keys['selected_criteria']][criterion_id] = False
                    if all_criteria_in_category:
                        st.session_state[session_keys['selected_criteria']][criterion_id] = True
                    is_selected = st.checkbox(
                        criterion_name,
                        value=st.session_state[session_keys['selected_criteria']][criterion_id],
                        key=f"criterion_{criterion_id}"
                    )
                    st.session_state[session_keys['selected_criteria']][criterion_id] = is_selected
                    if is_selected:
                        selected_criteria_ids.append(criterion_id)
        with st.expander("Opciones avanzadas de selección de criterios"):
            select_all_criteria = st.checkbox("Seleccionar TODOS los criterios de todas las categorías")
            if select_all_criteria:
                for _, row in rubric_df.iterrows():
                    criterion_id = row[criteria_col]
                    st.session_state[session_keys['selected_criteria']][criterion_id] = True
                st.success("Todos los criterios han sido seleccionados.")
            if st.button("Limpiar todas las selecciones"):
                for criterion_id in st.session_state[session_keys['selected_criteria']]:
                    st.session_state[session_keys['selected_criteria']][criterion_id] = False
                st.success("Todas las selecciones han sido limpiadas.")
        if st.button("Ver Detalles de Criterios Seleccionados"):
            selected_any = any(st.session_state[session_keys['selected_criteria']].values())
            if not selected_any:
                st.warning("Por favor seleccione al menos un criterio para ver sus detalles.")
            else:
                st.markdown("##### Detalles de Criterios Seleccionados")
                selected_criteria_df = rubric_df[rubric_df[criteria_col].isin(
                    [cid for cid, selected in st.session_state[session_keys['selected_criteria']].items() if selected]
                )]
                for _, criterion_row in selected_criteria_df.iterrows():
                    criterion_id = criterion_row[criteria_col]
                    criterion_name = criterion_row[short_col] if short_col and short_col in criterion_row else criterion_id
                    with st.expander(f"{criterion_name}", expanded=True):
                        levels_df = rubric_to_levels_df(criterion_row, criteria_col)
                        st.table(levels_df)
        if st.button("Iniciar Evaluación de Criterios Seleccionados (RAG)"):
            selected_any = any(st.session_state[session_keys['selected_criteria']].values())
            if not selected_any:
                st.warning("Por favor seleccione al menos un criterio para evaluar.")
            else:
                st.markdown("#### 3. Evaluación de Criterios (RAG)")
                selected_criteria_ids = [cid for cid, selected in st.session_state[session_keys['selected_criteria']].items() if selected]
                filtered_df = exploded_df[exploded_df['header_1'].isin(st.session_state[session_keys['selected_sections']])].copy()
                if filtered_df.empty:
                    st.warning("No se encontraron párrafos en las secciones seleccionadas.")
                    return
                rubric_dict = {}
                for cid in selected_criteria_ids:
                    crit_row = rubric_df[rubric_df[criteria_col] == cid].iloc[0]
                    levels = rubric_to_levels_df(crit_row, criteria_col)
                    rubric_dict[cid] = levels['Description'].tolist()
                st.info(f"Evaluando {len(selected_criteria_ids)} criterios sobre {len(filtered_df)} párrafos.")

                # Progress bar for rubric evaluation (per section and per criterion)
                section_list = list(filtered_df['header_1'].unique())
                total_steps = len(section_list) * len(selected_criteria_ids)
                progress_bar = st.progress(0, text="Progreso de evaluación por rúbrica")
                progress_count = 0
                results = {}
                store = SimpleHierarchicalStore(use_cache=True)
                filtered_df['document_id'] = 'doc1'  # Single doc context
                store.add_documents(filtered_df)

                # Evaluate per section and per criterion
                for section in section_list:
                    section_df = filtered_df[filtered_df['header_1'] == section]
                    for cid in selected_criteria_ids:
                        rubric_dict_single = {cid: rubric_dict[cid]}
                        # Use top_n_paragraphs=10 or as appropriate
                        result = store.score_rubric_directly(rubric_dict_single, top_n_paragraphs=10)
                        # result is a dict keyed by cid
                        if cid not in results:
                            results[cid] = {'score': 0, 'context': '', 'analysis': {}}
                        # Merge/accumulate results per criterion
                        if cid in result:
                            # If multiple sections, you may want to aggregate or just take the last/first
                            # Here, we take the last section's result for simplicity
                            results[cid] = result[cid]
                        progress_count += 1
                        progress_bar.progress(progress_count / total_steps, text=f"Evaluando sección '{section}' y criterio '{cid}'")
                progress_bar.empty()

                eval_rows = []
                for cid, result in results.items():
                    criterion_name = cid
                    if short_col:
                        crit_row = rubric_df[rubric_df[criteria_col] == cid].iloc[0]
                        criterion_name = crit_row[short_col] if short_col in crit_row else cid
                    score = result.get('score', 0)

                    if create_filtered:
                        filtered_sections = {}
                        progress_bar = st.progress(0, text="Progreso de filtrado de secciones")
                        total_sections = len(selected_sections)
                        for idx, section in enumerate(selected_sections):
                            if section in sections_content:
                                filtered_sections[section] = sections_content[section]
                            progress_bar.progress((idx + 1) / total_sections, text=f"Filtrando sección '{section}' ({idx + 1}/{total_sections})")
                        progress_bar.empty()
                        st.session_state.filtered_sections = filtered_sections
                        # Convert to a dataframe for Excel export
                        filtered_data = []
                        for section, paragraphs in filtered_sections.items():
                            # Get the section level from TOC
                            section_level = 0
                            for heading, level in toc:
                                if heading == section:
                                    section_level = level
                                    break
                            # Process paragraphs based on content type
                            in_table = False
                            table_content = []
                            table_rows = []
                            header_row = None
                            for text in paragraphs:
                                if text == '[TABLE_START]':
                                    in_table = True
                                    table_content = []
                                    table_rows = []
                                    header_row = None
                                elif text == '[TABLE_END]':
                                    in_table = False
                                    # Process collected table content - store the processed table
                                    if table_rows:
                                        # Create a JSON representation of the table
                                        table_data = {
                                            'header': header_row if header_row else [],
                                            'rows': table_rows
                                        }
                                        filtered_data.append({
                                            'section': section,
                                            'level': section_level,
                                            'content_type': 'table',
                                            'text': json.dumps(table_data)
                                        })
                                elif text.startswith('[TABLE_HEADER]'):
                                    # Process header row
                                    cells = text[14:].split('|')
                                    header_row = cells
                                    table_content.append(text)
                                elif text.startswith('[TABLE_ROW]'):
                                    # Process data row
                                    cells = text[11:].split('|')
                                    table_rows.append(cells)
                                    table_content.append(text)
                                else:
                                    # Regular paragraph
                                    filtered_data.append({
                                        'section': section,
                                        'level': section_level,
                                        'content_type': 'paragraph',
                                        'text': text
                                    })
                                filtered_df = pd.DataFrame(filtered_data)
                                
                                if not filtered_df.empty:
                                    st.success(f"Salida filtrada creada con {len(filtered_df)} elementos de {len(filtered_sections)} secciones.")
                                    
                                    # Show a preview
                                    with st.expander("Vista Previa del Contenido Filtrado"):
                                        st.dataframe(filtered_df[['section', 'level', 'content_type', 'text']])
                                    
                                    # Download button for the filtered document
                                    excel_data = BytesIO()
                                    filtered_df.to_excel(excel_data, index=False)
                                    excel_data.seek(0)
                                    
                                    st.download_button(
                                        label="Descargar Documento Filtrado",
                                        data=excel_data,
                                        file_name="filtered_document.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                else:
                                    st.warning("La salida filtrada está vacía. Por favor seleccione al menos una sección con contenido.")
                            else:
                                st.warning("Por favor seleccione al menos una sección para crear una salida filtrada.")
                        else:
                            st.warning("No se encontraron secciones con encabezados en el documento.")
                    else:
                        st.warning("No se encontró contenido en el documento.")
                
                # Tab 3: Rubric Evaluation
                with doc_tabs[2]:
                    try:
                        # Call the rubric evaluation function with the document sections
                        if 'add_rubric_evaluation_section' in globals():
                            add_rubric_evaluation_section(sections_content, toc, toc_hierarchy)
                        else:
                            st.info("La función de evaluación por rúbrica no está disponible. Por favor actualice el código con la implementación de esta función.")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())

# ============= VISUALIZATION FUNCTIONS =============

# Function to prepare additional data for new visualizations
def prepare_additional_data(df):
    """
    Prepare additional columns needed for the new visualizations.
    This should be called after the original data loading.
    """
    # Make sure we have datetime and year
    if 'Recommendation_date' in df.columns:
        df['year'] = pd.to_datetime(df['Recommendation_date']).dt.year
        df['year'] = df['year'].fillna(2023).astype(int)
    
    # Standardize categorical fields (if they exist in the dataframe)
    categorical_mappings = {
        'rec_innovation_score': {
            'very high': 'Very High', 'high': 'High', 'High': 'High',
            'medium': 'Medium', 'Medium': 'Medium',
            'low': 'Low', 'very low': 'Very Low'
        },
        'rec_precision_and_clarity': {
            'high': 'High', 'High': 'High',
            'medium': 'Medium', 'Medium': 'Medium',
            'low': 'Low', 'Low': 'Low'
        },
        'rec_expected_impact': {
            'high': 'High', 'High': 'High',
            'medium': 'Medium', 'Medium': 'Medium',
            'low': 'Low', 'Low': 'Low'
        },
        'rec_intervention_approach': {
            'policy': 'Policy', 'Policy': 'Policy',
            'process': 'Process', 'Process': 'Process'
        },
        'rec_operational_feasibility': {
            'high': 'High', 'High': 'High',
            'medium': 'Medium', 'Medium': 'Medium',
            'low': 'Low', 'Low': 'Low'
        },
        'rec_timeline': {
            'short': 'Short', 'Short': 'Short',
            'medium': 'Medium', 'Medium': 'Medium',
            'long': 'Long', 'Long': 'Long'
        }
    }
    
    # Apply standardization to columns that exist in the dataframe
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    
    # Process tags if they exist
    if 'tags' in df.columns:
        # Convert tags to list format if needed
        df['tags'] = df['tags'].apply(
            lambda x: [x] if isinstance(x, str) and not pd.isna(x) else 
                     x if isinstance(x, list) else []
        )
    
    # Process rejection difficulty classification if it exists
    if 'rejection_difficulty_classification' in df.columns:
        # Clean brackets and prepare for analysis
        df['clean_classification'] = df['rejection_difficulty_classification'].apply(
            lambda x: x.replace('[', '').replace(']', '').strip() if isinstance(x, str) else ""
        )
        
        # Split tags
        df['clean_tags'] = df['clean_classification'].apply(
            lambda x: [tag.strip().strip("'").strip('"') for tag in x.split(',') if tag.strip()] 
            if isinstance(x, str) and x else []
        )
    
    return df

# Function to plot score evolution over time - simpler fix
def plot_score_evolution(filtered_df):
    """
    Creates a line plot showing the evolution of scores over time.
    Fixed to handle potential string concatenation issues.
    """
    # Only include columns that end with '_score' and exclude any problematic columns
    score_columns = [col for col in filtered_df.columns if col.endswith('_score') and col != 'clean_tags']
    
    if not score_columns or filtered_df.empty:
        st.warning("No score data available for the selected filters.")
        return
    
    # Create a copy of the filtered dataframe to avoid modifying the original
    df_scores = filtered_df.copy()
    
    # Clean up each score column to handle possible string concatenation issues
    for col in score_columns:
        # Check if the column contains long concatenated strings (like 'MediumMediumMedium...')
        mask = df_scores[col].astype(str).str.len() > 15
        if mask.any():
            # Set problematic values to NaN
            df_scores.loc[mask, col] = np.nan
        
        # Try to convert to numeric, coercing errors to NaN
        df_scores[col] = pd.to_numeric(df_scores[col], errors='coerce')
    
    # Calculate yearly averages for each score type
    yearly_scores = df_scores.groupby('year')[score_columns].mean().reset_index()
    
    # Only proceed if we have data
    if yearly_scores.empty or yearly_scores[score_columns].isna().all().all():
        st.warning("No usable score data available for the selected years.")
        return
    
    # Create a Plotly line chart
    fig = go.Figure()
    
    # Add a line for each score
    for column in score_columns:
        # Skip columns with all NaN values
        if yearly_scores[column].isna().all():
            continue
            
        # Create a more readable label
        label = column.replace('_score', '').replace('_', ' ').title()
        
        fig.add_trace(go.Scatter(
            x=yearly_scores['year'], 
            y=yearly_scores[column],
            mode='lines+markers+text',
            name=label,
            text=yearly_scores[column].round(2),
            textposition="top center",
            line=dict(width=3)
        ))
    
    # Update layout
    fig.update_layout(
        title='Evolución de Puntuaciones Promedio por Año',
        xaxis_title='Año',
        yaxis_title='Puntuación Promedio (0-10)',
        yaxis=dict(range=[0, 10]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified",
        height=500
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig

# Function to create a composition plot (stacked bar chart)
def create_composition_plot(filtered_df, var_name, title):
    """
    Creates a stacked bar chart showing the composition of a variable over time.
    Harmonizes 'process' and 'processes' if var_name is 'dimension'.
    """
    if var_name not in filtered_df.columns or filtered_df.empty:
        st.warning(f"No data available for {var_name} with the selected filters.")
        return

    # Group by year and variable, then count
    var_by_year = filtered_df.groupby(['year', var_name]).size().unstack(fill_value=0)
    # Calculate percentages
    var_by_year_pct = var_by_year.div(var_by_year.sum(axis=1), axis=0) * 100
    # Only proceed if we have data
    if var_by_year_pct.empty:
        st.warning(f"No data available for {var_name} with the selected years.")
        return
    
    # Create a Plotly stacked bar chart
    fig = go.Figure()
    
    # Calculate cumulative percentages for text positioning
    cumulative = pd.DataFrame(0, index=var_by_year_pct.index, columns=['cum'])
    
    # Add a bar for each category
    for category in var_by_year_pct.columns:
        fig.add_trace(go.Bar(
            x=var_by_year_pct.index,
            y=var_by_year_pct[category],
            name=category,
            text=var_by_year_pct[category].round(1).astype(str) + '%',
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=20),  # Increased label font size
            hoverinfo='name+y'
        ))
        
        # Update cumulative for next bar
        cumulative['cum'] += var_by_year_pct[category]
    
    # Update layout
    # Remove undefined or empty title
    # Fix undefined or empty title: if title is None, '', or 'undefined' (any case/whitespace), do not show a title
    if title is None or str(title).strip() == '' or str(title).strip().lower() == 'undefined':
        layout_title = ''
    else:
        layout_title = title
    fig.update_layout(
        title=layout_title,
        xaxis_title='Año',
        yaxis_title='Porcentaje (%)',
        barmode='stack',
        uniformtext=dict(mode='hide', minsize=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        height=500
    )
    
    # Add sample count annotations on top of each bar
    total_counts = filtered_df.groupby('year').size()
    for year in var_by_year_pct.index:
        if year in total_counts.index:
            fig.add_annotation(
                x=year,
                y=105,
                text=f"n={total_counts[year]}",
                showarrow=False,
                font=dict(size=14, color="black")
            )
    
    return fig

# Function to create tag composition plot
def create_tag_composition_plot(filtered_df, top_n=8):
    """
    Creates a stacked bar chart showing the composition of tags over time.
    """
    if 'tags' not in filtered_df.columns or filtered_df.empty:
        st.warning("No tag data available for the selected filters.")
        return
    
    # Explode the dataframe by tags
    exploded_df = filtered_df.explode('tags')
    
    # Remove rows with empty tags
    exploded_df = exploded_df.dropna(subset=['tags']).reset_index(drop=True)
    
    if exploded_df.empty:
        st.warning("No tag data available after filtering.")
        return
    
    # Count occurrences of each tag
    tag_counts = exploded_df['tags'].value_counts()
    
    # Get the top N tags
    top_tags = tag_counts.head(top_n).index.tolist()
    
    # Replace non-top tags with 'Other tags'
    exploded_df['tag_category'] = exploded_df['tags'].apply(
        lambda x: x if x in top_tags else 'Other tags'
    )
    
    # Count yearly occurrences for each tag category
    yearly_tag_counts = exploded_df.groupby(['year', 'tag_category']).size().unstack(fill_value=0)
    
    # Calculate the percentage of each tag category per year
    yearly_tag_percentages = yearly_tag_counts.div(yearly_tag_counts.sum(axis=1), axis=0) * 100
    
    # Only proceed if we have data
    if yearly_tag_percentages.empty:
        st.warning("No tag data available for the selected years.")
        return
    
    # Create a Plotly stacked bar chart
    fig = go.Figure()
    
    # Add a bar for each tag category
    for category in yearly_tag_percentages.columns:
        fig.add_trace(go.Bar(
            x=yearly_tag_percentages.index,
            y=yearly_tag_percentages[category],
            name=category,
            text=yearly_tag_percentages[category].round(1).astype(str) + '%',
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=14),
            hoverinfo='name+y'
        ))
    
    # Update layout
    fig.update_layout(
        title='Evolución de la Composición de Etiquetas por Año',
        xaxis_title='Año',
        yaxis_title='Porcentaje (%)',
        barmode='stack',
        uniformtext=dict(mode='hide', minsize=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5
        ),
        height=500
    )
    
    # Add sample count annotations on top of each bar
    total_counts = exploded_df.groupby('year').size()
    for year in yearly_tag_percentages.index:
        if year in total_counts.index:
            fig.add_annotation(
                x=year,
                y=105,
                text=f"n={total_counts[year]}",
                showarrow=False,
                font=dict(size=14, color="black")
            )
    
    return fig

# Function to create rejection difficulty classification plot
def create_difficulty_classification_plot(filtered_df, top_n=8):
    """
    Creates a stacked bar chart showing the composition of rejection difficulty classifications over time.
    """
    if 'clean_tags' not in filtered_df.columns or filtered_df.empty:
        st.warning("No classification data available for the selected filters.")
        return
    
    # Explode the dataframe by tags
    exploded_df = filtered_df.explode('clean_tags')
    
    # Remove rows with empty tags
    exploded_df = exploded_df[
        exploded_df['clean_tags'].apply(lambda x: isinstance(x, str) and len(x) > 0)
    ].reset_index(drop=True)
    
    if exploded_df.empty:
        st.warning("No classification data available after filtering.")
        return
    
    # Count occurrences of each tag
    tag_counts = exploded_df['clean_tags'].value_counts()
    
    # Get the top N tags
    top_tags = tag_counts.head(top_n).index.tolist()
    
    # Replace non-top tags with 'Other tags'
    exploded_df['tag_category'] = exploded_df['clean_tags'].apply(
        lambda x: x if x in top_tags else 'Other tags'
    )
    
    # Count yearly occurrences for each tag category
    yearly_tag_counts = exploded_df.groupby(['year', 'tag_category']).size().unstack(fill_value=0)
    
    # Calculate the percentage of each tag category per year
    yearly_tag_percentages = yearly_tag_counts.div(yearly_tag_counts.sum(axis=1), axis=0) * 100
    
    # Only proceed if we have data
    if yearly_tag_percentages.empty:
        st.warning("No classification data available for the selected years.")
        return
    
    # Create a Plotly stacked bar chart
    fig = go.Figure()
    
    # Add a bar for each classification category
    for category in yearly_tag_percentages.columns:
        fig.add_trace(go.Bar(
            x=yearly_tag_percentages.index,
            y=yearly_tag_percentages[category],
            name=category,
            text=yearly_tag_percentages[category].round(1).astype(str) + '%',
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=14),
            hoverinfo='name+y'
        ))
    
    # Update layout
    fig.update_layout(
        title='Evolución de la Composición de Clasificaciones de Dificultad de Rechazo por Año',
        xaxis_title='Año',
        yaxis_title='Porcentaje (%)',
        barmode='stack',
        uniformtext=dict(mode='hide', minsize=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5
        ),
        height=500
    )
    
    # Add sample count annotations on top of each bar
    total_counts = exploded_df.groupby('year').size()
    for year in yearly_tag_percentages.index:
        if year in total_counts.index:
            fig.add_annotation(
                x=year,
                y=105,
                text=f"n={total_counts[year]}",
                showarrow=False,
                font=dict(size=14, color="black")
            )
    
    return fig

# # Function to add advanced visualization section to the Streamlit app
# def add_advanced_visualization_section(filtered_df):
#     """
#     Adds an advanced visualization section to the Streamlit app.
#     """
#     st.markdown("#### Visualizaciones Avanzadas")

#     # --- Score Evolution ---
#     st.markdown("<h4 style='margin-top: 2em;'>Evolución de Puntuaciones Promedio por Año</h4>", unsafe_allow_html=True)
#     score_fig = plot_score_evolution(filtered_df)
#     if score_fig:
#         # Remove inline title from plot
#         st.plotly_chart(score_fig, use_container_width=True)

#     # --- Variable Composition ---
#     st.markdown("<h4 style='margin-top: 2em;'>Composición por Variable</h4>", unsafe_allow_html=True)
#     available_vars = [col for col in filtered_df.columns if col.startswith('rec_')]
#     if available_vars:
#         var_mapping = {
#             'rec_innovation_score': 'Nivel de Innovación',
#             'rec_precision_and_clarity': 'Precisión y Claridad',
#             'rec_expected_impact': 'Impacto Esperado',
#             'rec_intervention_approach': 'Enfoque de Intervención',
#             'rec_operational_feasibility': 'Factibilidad Operativa',
#             'rec_timeline': 'Plazo de Implementación'
#         }
#         var_options = {var_mapping.get(var, var): var for var in available_vars if var in var_mapping}
#         if var_options:
#             selected_var_label = st.selectbox(
#                 "Seleccione una variable para visualizar:", 
#                 options=list(var_options.keys())
#             )
#             selected_var = var_options[selected_var_label]
#             var_titles = {
#                 'rec_innovation_score': 'Composición de Niveles de Innovación por Año',
#                 'rec_precision_and_clarity': 'Composición de Niveles de Precisión y Claridad por Año',
#                 'rec_expected_impact': 'Composición de Niveles de Impacto Esperado por Año',
#                 'rec_intervention_approach': 'Composición de Enfoques de Intervención por Año',
#                 'rec_operational_feasibility': 'Composición de Niveles de Factibilidad Operativa por Año',
#                 'rec_timeline': 'Composición de Plazos de Implementación por Año'
#             }
#             composition_fig = create_composition_plot(
#                 filtered_df, 
#                 selected_var, 
#                 var_titles.get(selected_var, f'Composición de {selected_var_label} por Año')
#             )
#             if composition_fig:
#                 st.plotly_chart(composition_fig, use_container_width=True)
#         else:
#             st.warning("No se encontraron variables de composición en los datos filtrados.")
#     else:
#         st.warning("No se encontraron variables de composición en los datos filtrados.")

#     # --- Difficulty Classification ---
#     st.markdown("<h4 style='margin-top: 2em;'>Clasificación de Dificultad de Rechazo</h4>", unsafe_allow_html=True)
#     if 'clean_tags' in filtered_df.columns:
#         top_n = st.slider("Número de clasificaciones principales a mostrar:", min_value=3, max_value=15, value=8, key='diff_class_slider')
#         diff_fig = create_difficulty_classification_plot(filtered_df, top_n)
#         if diff_fig:
#             st.plotly_chart(diff_fig, use_container_width=True)
#     else:
#         st.warning("No se encontraron datos de clasificación de dificultad en los datos filtrados.")

# Fixed version of the function with tab-specific keys
def add_advanced_visualization_section(filtered_df, tab_id="tab1"):
    """
    Adds an advanced visualization section to the Streamlit app.
    
    Parameters:
    -----------
    filtered_df : pandas DataFrame
        The filtered dataframe to visualize
    tab_id : str
        The tab identifier to make widget keys unique (default: "tab1")
    """
    st.markdown("#### Calidad de  Respuesta Institucional a Recomendaciones")

    # --- Score Evolution ---
    st.markdown("<h4 style='margin-top: 2em;'>Evolución de Puntuaciones Promedio por Año</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        score_fig = plot_score_evolution(filtered_df)
        if score_fig:
            # Remove inline title from plot
            st.plotly_chart(score_fig, use_container_width=True)
    with col2:
        # Make the legend box visually match the plot height
        st.markdown("""
            <style>
            textarea[data-baseweb="textarea"] {
                min-height: 500px !important;
                height: 500px !important;
                max-height: 700px;
            }
            </style>
        """, unsafe_allow_html=True)
        legend_definitions = st.text_area(
            "Leyenda:",
            value="""
            • Coherencia: Nivel de alineación entre la recomendación y el plan de acción.
            • Calidad del plan: Nivel de especificidad y claridad del plan de acción.
            • Nivel de atención: Nivel de consideración hacia las prioridades de la recomendación, en el plan de acción.
            • Puntuación agregada: Promedio de las puntuaciones de coherencia, calidad del plan y nivel de atención.
            """,
            key=f"legend_definitions_{tab_id}",
            height=500
        )


    # --- Variable Composition ---
    st.markdown("<h4 style='margin-top: 2em;'>Composición de Recomendaciones por Atributo</h4>", unsafe_allow_html=True)
    available_vars = [col for col in filtered_df.columns if col.startswith('rec_')]
    if available_vars:
        var_mapping = {
            'rec_innovation_score': 'Nivel de Innovación',
            'rec_precision_and_clarity': 'Precisión y Claridad',
            'rec_expected_impact': 'Impacto Esperado',
            'rec_intervention_approach': 'Enfoque de Intervención',
            'rec_operational_feasibility': 'Factibilidad Operativa',
            'rec_timeline': 'Plazo de Implementación'
        }
        var_options = {var_mapping.get(var, var): var for var in available_vars if var in var_mapping}
        if var_options:
            # Use tab_id parameter to create unique key for this selectbox
            selected_var_label = st.selectbox(
                "Seleccione una variable para visualizar:", 
                options=list(var_options.keys()),
                key=f"variable_{tab_id}"  # This ensures unique keys across tabs
            )
            selected_var = var_options[selected_var_label]
            var_titles = {
                'rec_innovation_score': 'Composición de Niveles de Innovación por Año',
                'rec_precision_and_clarity': 'Composición de Niveles de Precisión y Claridad por Año',
                'rec_expected_impact': 'Composición de Niveles de Impacto Esperado por Año',
                'rec_intervention_approach': 'Composición de Enfoques de Intervención por Año',
                'rec_operational_feasibility': 'Composición de Niveles de Factibilidad Operativa por Año',
                'rec_timeline': 'Composición de Plazos de Implementación por Año'
            }
            composition_fig = create_composition_plot(
                filtered_df, 
                selected_var, 
                var_titles.get(selected_var, f'Composición de {selected_var_label} por Año')
            )
            if composition_fig:
                st.plotly_chart(composition_fig, use_container_width=True)
        else:
            st.warning("No se encontraron variables de composición en los datos filtrados.")
    else:
        st.warning("No se encontraron variables de composición en los datos filtrados.")

    # --- Difficulty Classification ---
    st.markdown("<h4 style='margin-top: 2em;'>Principales Barreras de Implementación</h4>", unsafe_allow_html=True)
    if 'clean_tags' in filtered_df.columns:
        # Use tab_id parameter to create unique key for this slider
        top_n = st.slider(
            "Número de clasificaciones principales a mostrar:", 
            min_value=3, max_value=15, value=8, 
            key=f"diff_class_slider_{tab_id}"  # This ensures unique keys across tabs
        )
        diff_fig = create_difficulty_classification_plot(filtered_df, top_n)
        if diff_fig:
            st.plotly_chart(diff_fig, use_container_width=True)
    else:
        st.warning("No se encontraron datos de barreras en los datos filtrados.")
# ============= DATA LOADING FUNCTIONS =============

# Load data - use relative paths for deployment
# Define paths as relative to the current directory or using st.secrets for Streamlit Cloud
@st.cache_data
def load_data():
    # Replace with a check for environment, use st.secrets for paths in production
    if os.getenv("STREAMLIT_ENV") == "production":
        # Use st.secrets for file paths in production
        df_path = st.secrets["df_path"]
        df_raw_path = st.secrets["df_raw_path"]
        embeddings_path = st.secrets["embeddings_path"]
    else:
        # Use relative paths for local development
        df_path = "./df_complete_all_full.xlsx"
        df_raw_path = "./df_split_actions.xlsx"
        embeddings_path = "./emb_Recomm_rec_cl_4.pt"
    
    df = pd.read_excel(df_path)
    df['index_df'] = df['ID_Recomendacion']
    # Replace spaces and dots with underscore in column names
    df.columns = df.columns.str.replace(' ', '_').str.replace('.', '_')
    df.rename(columns={'Dimension': 'dimension', 
                       'Subdimension': 'subdim'}, inplace=True)

    # Raw data
    df_raw = pd.read_excel(df_raw_path)
    df_raw['year'] = pd.to_datetime(df_raw['Recommendation date'], format='%Y-%m-%d').dt.year
    df_raw.columns = df_raw.columns.str.replace(' ', '_').str.replace('.', '_')
    missing_index_df = df_raw[~df_raw['index_df'].isin(df['index_df'])]
    
    # Reset index of both DataFrames before concatenation
    df = df.reset_index(drop=True)
    
    # Concatenate with ignore_index=True to avoid index conflicts
    df = pd.concat([df, missing_index_df], axis=0)
    
    # Processes
    df['year'] = pd.to_datetime(df['Recommendation_date'], format='%Y-%m-%d').dt.year
    df['year'] = df['year'].fillna(2023).astype(int)
    df['year'] = df['year'].astype(int)
    df['dimension'] = df['dimension'].fillna('Sin Clasificar')
    df['subdim'] = df['subdim'].fillna('Sin Clasificar')
    df['Management_response'] = df['Management_response'].fillna('Sin respuesta')
    df['Management_response'] = df['Management_response'].replace('Partially completed', 'Partially Completed')
    
    return df, df_raw

@st.cache_data
def load_extended_data():
    """
    Load and prepare additional data for enhanced visualizations.
    This extends the existing load_data function with new data processing.
    """
    # First load the original data
    df, df_raw = load_data()
    
    # Load additional analysis data if available
    try:
        if os.getenv("STREAMLIT_ENV") == "production":
            analyzed_path = st.secrets.get("analyzed_recommendations_path", None)
        else:
            # Use a relative or absolute path based on your setup
            analyzed_path = "./analyzed_recommendations_plans_v5.csv"
        
        if analyzed_path and os.path.exists(analyzed_path):
            # Load the analyzed recommendations with pipe separator
            analyzed_df = pd.read_csv(analyzed_path, sep='|')
            
            # Convert dates and ensure year column
            analyzed_df['Recommendation date'] = pd.to_datetime(analyzed_df['Recommendation date'])
            analyzed_df['year'] = analyzed_df['Recommendation date'].dt.year
            
            # Change years prior to 2018 to 2018 to match the original analysis
            analyzed_df.loc[analyzed_df['year'] < 2018, 'year'] = 2018
            
            # Prepare additional columns needed for the visualizations
            analyzed_df = prepare_additional_data(analyzed_df)
            
            # Merge with the original dataframe if needed (based on a common key)
            if 'index_df' in df.columns and 'index_df' in analyzed_df.columns:
                # Select only new columns from analyzed_df to avoid duplicates
                new_cols = [col for col in analyzed_df.columns if col not in df.columns]
                if new_cols:  # Only proceed if there are new columns to add
                    # Include the key column
                    merge_cols = ['index_df'] + new_cols
                    
                    # Merge the new data
                    df = pd.merge(df, analyzed_df[merge_cols], on='index_df', how='left')
            
            # Store the analyzed dataframe in session state for potential use elsewhere
            st.session_state['analyzed_df'] = analyzed_df
        else:
            st.warning("Additional analysis data file not found. Some visualizations may not be available.")
            st.session_state['analyzed_df'] = None
            
    except Exception as e:
        st.warning(f"Note: Additional analysis data could not be loaded. Some visualizations may not be available. Error: {str(e)}")
        st.session_state['analyzed_df'] = None
    
    # Process the main dataframe with the additional preparation
    df = prepare_additional_data(df)
    
    return df, df_raw

# Load embeddings
@st.cache_data
def load_embeddings():
    if os.getenv("STREAMLIT_ENV") == "production":
        embeddings_path = st.secrets["embeddings_path"]
        structured_embeddings_path = st.secrets["structured_embeddings_path"]
    else:
        embeddings_path = "./emb_Recomm_rec_cl_4.pt"
        structured_embeddings_path = "./Recommendation_RAG_Metadata.pt"
    
    doc_embeddings = torch.load(embeddings_path)
    doc_embeddings = np.array(doc_embeddings)
    
    structured_embeddings = torch.load(structured_embeddings_path)
    
    # Create a FAISS index
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(doc_embeddings)
    
    return doc_embeddings, structured_embeddings, index

import concurrent.futures

def process_text_analysis(combined_text, map_template, combine_template_prefix, user_template_part):
    """
    Process text analysis in chunks and combine results (parallelized).
    
    Parameters:
    -----------
    combined_text : str
        The combined text to analyze
    map_template : str
        Template for the initial summarization of chunks
    combine_template_prefix : str
        Prefix for the template used to combine summaries
    user_template_part : str
        User-defined part of the template for final analysis
        
    Returns:
    --------
    str
        Analyzed and summarized text
    """
    if not combined_text:
        return None
    
    text_chunks = split_text(combined_text)
    chunk_summaries = []
    MAX_WORKERS = min(8, len(text_chunks)) if text_chunks else 1

    def summarize_chunk(chunk):
        return summarize_text(chunk, map_template)

    # Parallelize chunk summarization
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(summarize_chunk, chunk) for chunk in text_chunks]
        for future in concurrent.futures.as_completed(futures):
            summary = future.result()
            if summary:
                chunk_summaries.append(summary)

    # Preserve original chunk order (since as_completed doesn't)
    chunk_summaries_ordered = []
    if chunk_summaries and len(chunk_summaries) == len(text_chunks):
        # If all succeeded, sort by chunk order
        chunk_map = {futures[i]: i for i in range(len(futures))}
        # But as_completed gives no order, so instead:
        # Re-run in order
        for i in range(len(text_chunks)):
            result = futures[i].result()
            if result:
                chunk_summaries_ordered.append(result)
    else:
        chunk_summaries_ordered = chunk_summaries

    if chunk_summaries_ordered:
        combined_summaries = " ".join(chunk_summaries_ordered)
        final_template = combine_template_prefix + user_template_part
        return summarize_text(combined_summaries, final_template)
    
    return None


def split_text(text, max_length=1500):
    """
    Split text into chunks of specified maximum length.
    
    Parameters:
    -----------
    text : str
        The text to split
    max_length : int, optional
        Maximum length of each chunk (default: 1500)
        
    Returns:
    --------
    list
        List of text chunks
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def build_combined_text(df, selections):
    """
    Build combined text from selected text sources.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the text columns
    selections : dict
        Dictionary indicating which text sources to include
        
    Returns:
    --------
    str
        Combined text from all selected sources
    """
    texts = []
    if selections['recommendations']:
        texts.append(" ".join(df['Recommendation_description'].astype(str).dropna().unique()))
    if selections['lessons']:
        texts.append(" ".join(df['Lessons_learned_description'].astype(str).dropna().unique()))
    if selections['practices']:
        texts.append(" ".join(df['Good_practices_description'].astype(str).dropna().unique()))
    if selections['plans']:
        texts.append(" ".join(df['Action_plan'].astype(str).dropna().unique()))
    return " ".join(texts)

def summarize_text(text, prompt_template):
    """
    Summarize text using OpenAI API.
    
    Parameters:
    -----------
    text : str
        The text to summarize
    prompt_template : str
        Template for the prompt
        
    Returns:
    --------
    str
        Summarized text
    """
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
        
    prompt = prompt_template.format(text=text)
    try:
        # Use only the older OpenAI SDK (<1.0.0)
        response = openai.ChatCompletion.create(
            model="gpt-4.1-mini",  # or whatever model you're using
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes and analyzes texts."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None

# ============= MAIN APP CODE =============

# Set page config
st.set_page_config(layout="wide")
st.markdown("""
    <h2 style='text-align:center; color:#3498db; margin-top:0;'>Caja de Herramientas para la Mejora de Proyectos</h3>
    <hr style='border-top: 2px solid #3498db;'>
""", unsafe_allow_html=True)

# Initialize data and embeddings - wrap in try/except for better error handling
try:
    # Use the extended data loading function that includes the new visualizations data
    df, df_raw = load_extended_data()
    doc_embeddings, structured_embeddings, index = load_embeddings()
    doc_texts = df_raw['Recommendation_description'].tolist()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()


# Check for API key before running the app
if not openai_api_key:
    st.warning("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in Streamlit Cloud.")
    st.info("For local development, you can use a .env file or set the environment variable.")
    # Continue with limited functionality or show instructions on setup

# Initialize data and embeddings - wrap in try/except for better error handling
try:
    # Use the extended data loading function that includes the new visualizations data
    df, df_raw = load_extended_data()
    doc_embeddings, structured_embeddings, index = load_embeddings()
    doc_texts = df_raw['Recommendation_description'].tolist()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Function to get embeddings for lessons learned
def get_lessons_embedding_with_retry(text, model='text-embedding-3-large', max_retries=3, delay=1):
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(input=text, model=model)
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(delay)
    return None

# Function to find similar lessons learned using embeddings
def find_similar_lessons(query_embedding, lessons_index, lessons_embeddings, structured_lessons, score_threshold=0.5, top_n=20):
    # Normalize query embedding for cosine similarity
    query_embedding = np.array(query_embedding).reshape(1, -1)
    # Search the index
    try:
        distances, indices = lessons_index.search(query_embedding, lessons_index.ntotal)

        # Filter results based on the score threshold
        filtered_lessons = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(structured_lessons) and dist >= score_threshold:
                metadata = structured_lessons[idx]
                lesson = {
                    "lesson": metadata["text"],
                    "similarity": float(dist),  # Convert to float for JSON serialization
                    "country": metadata["country"],
                    "year": metadata["year"],
                    "eval_id": metadata["eval_id"]
                }
                filtered_lessons.append(lesson)
            if len(filtered_lessons) >= top_n:
                break
        return filtered_lessons
    except Exception as e:
        st.error(f"Error in similarity search: {str(e)}")
        return []

# Function to find lessons by term matching
def find_lessons_by_term_matching(query, lessons_texts, structured_lessons, top_n=10):
    try:
        matched_lessons = []
        query_lower = query.lower()
        for idx, text in enumerate(lessons_texts):
            if isinstance(text, str) and query_lower in text.lower():
                if idx < len(structured_lessons):
                    metadata = structured_lessons[idx]
                    matched_lessons.append({
                        "lesson": text,
                        "country": metadata["country"],
                        "year": metadata["year"],
                        "eval_id": metadata["eval_id"]
                    })
        matched_lessons = sorted(matched_lessons, key=lambda x: len(str(x["lesson"])))
        return matched_lessons[:top_n]
    except Exception as e:
        st.error(f"Error in term matching: {str(e)}")
        return []

# Function to load and prepare lessons embeddings
@st.cache_data
def load_lessons_embeddings():
    try:
        if os.getenv("STREAMLIT_ENV") == "production":
            lessons_embeddings_path = st.secrets["lessons_embeddings_path"]
            structured_lessons_path = st.secrets["structured_lessons_path"]
        else:
            lessons_embeddings_path = "./emb_LL_ll_cl_4.pt"
            structured_lessons_path = "./lessons_metadata.pt"
        
        # Load embeddings and metadata
        lessons_embeddings = torch.load(lessons_embeddings_path)
        lessons_embeddings = np.array(lessons_embeddings)
        
        structured_lessons = torch.load(structured_lessons_path)
        
        # Create a FAISS index
        dimension = lessons_embeddings.shape[1]
        lessons_index = faiss.IndexFlatIP(dimension)
        lessons_index.add(lessons_embeddings)
        
        return lessons_embeddings, structured_lessons, lessons_index
    except Exception as e:
        st.error(f"Error loading lessons embeddings: {str(e)}")
        # Return placeholder data to avoid errors
        return np.array([]), [], None
    
    
# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([ "Valoración Preliminar de Calidad de Proyectos",
                                         "Acercamiento a Valoración de Calidad de Proyectos",
                                         "Diagnóstico de Sostenibilidad del Proyecto",
                                         "Pregúntale a tus Documentos",
                                         "Estadísticas sobre Recomendaciones de Evaluaciones y sus Planes de Acción"])

#-----------------------#-----------------------#
#-----------------------#-----------------------#
# Tab 2: Revisión por criterios con trazabilidad
with tab2:
    st.header("Acercamiento a Valoración de Calidad de Proyectos")

    # Descriptive text box
    st.info("""
    **📋 Descripción de la herramienta:**

    Sube un Word (.docx) para evaluarlo con criterios y niveles de desempeño (rúbricas) alineados a la OIT. La herramienta extrae secciones clave, aplica la matriz de criterios y asigna puntajes 1–5 con análisis narrativo y evidencia trazable (citas + metadatos). Puedes exportar a Excel (Criterio, Dimensión, Score, Análisis, Evidencia, Error, Rúbrica).

    Si hay vacíos o inconsistencias, se señalan en "Error" para su ajuste. Este diagnóstico en formato EXCEL sirve para revisar propuestas antes de enviarlas a donantes, verificar aspectos puntuales de informes de evaluación o de ejecución, comprobar coherencia con P&B, DWCP y marcos UNSDCF, elaborar notas técnicas con sustento y respaldar la rendición de cuentas ante mandantes y donantes.
    """)

    # Read rubrics from Excel files as in megaparse_example.py
    import pandas as pd
    engagement_rubric = {}
    performance_rubric = {}
    parteval_rubric = {}
    gender_rubric = {}
    tj_traditional_rubric = {}
    tj_just_transition_rubric = {}

    try:
        df_rubric_engagement = pd.read_excel('./Rubricas_6ago2025.xlsx', sheet_name='rubric_engagement')
        df_rubric_engagement.drop(columns=['Unnamed: 0', 'Criterio'], inplace=True, errors='ignore')
        for idx, row in df_rubric_engagement.iterrows():
            indicador = row['Indicador']
            dimension = row.get('Dimensión', 'No especificada')  # Get dimension, default if not present
            valores = row.drop(['Indicador', 'Dimensión'], errors='ignore').values.tolist()
            engagement_rubric[indicador] = {'valores': valores, 'dimension': dimension}

        df_rubric_performance = pd.read_excel('./Rubricas_6ago2025.xlsx', sheet_name='rubric_performance')
        df_rubric_performance.drop(columns=['dimension'], inplace=True, errors='ignore')
        for idx, row in df_rubric_performance.iterrows():
            criterio = row['subdim']
            dimension = row.get('Dimensión', 'No especificada')  # Get dimension, default if not present
            valores = row.drop(['subdim', 'Dimensión'], errors='ignore').values.tolist()
            performance_rubric[criterio] = {'valores': valores, 'dimension': dimension}

        df_rubric_parteval = pd.read_excel('./Rubricas_6ago2025.xlsx', sheet_name='rubric_parteval')
        df_rubric_parteval.drop(columns=['Criterio'], inplace=True, errors='ignore')
        for idx, row in df_rubric_parteval.iterrows():
            indicador = row['Indicador']
            dimension = row.get('Dimensión', 'No especificada')  # Get dimension, default if not present
            valores = row.drop(['Indicador', 'Dimensión'], errors='ignore').values.tolist()
            parteval_rubric[indicador] = {'valores': valores, 'dimension': dimension}

        df_rubric_gender = pd.read_excel('./Rubricas_6ago2025.xlsx', sheet_name='rubric_gender_')
        df_rubric_gender.drop(columns=['Criterio'], inplace=True, errors='ignore')
        for idx, row in df_rubric_gender.iterrows():
            indicador = row['Indicador']
            dimension = row.get('Dimensión', 'No especificada')  # Get dimension, default if not present
            valores = row.drop(['Indicador', 'Dimensión'], errors='ignore').values.tolist()
            gender_rubric[indicador] = {'valores': valores, 'dimension': dimension}

        # Load TJ Traditional rubric from Rubricas_6ago2025.xlsx
        try:
            df_rubric_tj_traditional = pd.read_excel('./Rubricas_6ago2025.xlsx', sheet_name='rubric_TJ_Traditional')
            df_rubric_tj_traditional.drop(columns=['Criterio'], inplace=True, errors='ignore')
            for idx, row in df_rubric_tj_traditional.iterrows():
                indicador = row['Indicador']
                if pd.notna(indicador) and str(indicador).strip():
                    dimension = row.get('Dimensión', 'No especificada')  # Get dimension, default if not present
                    valores = row.drop(['Indicador', 'Dimensión'], errors='ignore').values.tolist()
                    tj_traditional_rubric[indicador] = {'valores': valores, 'dimension': dimension}
        except Exception as e:
            st.error(f"❌ Error cargando TJ Tradicional: {e}")
            st.write("Sheets disponibles:", list(pd.ExcelFile('./Rubricas_6ago2025.xlsx').sheet_names))

        # Load TJ Just Transition rubric from Rubricas_6ago2025.xlsx
        try:
            df_rubric_tj_just_transition = pd.read_excel('./Rubricas_6ago2025.xlsx', sheet_name='rubric_TJ_TJ')
            df_rubric_tj_just_transition.drop(columns=['Criterio'], inplace=True, errors='ignore')
            for idx, row in df_rubric_tj_just_transition.iterrows():
                indicador = row['Indicador']
                if pd.notna(indicador) and str(indicador).strip():
                    dimension = row.get('Dimensión', 'No especificada')  # Get dimension, default if not present
                    valores = row.drop(['Indicador', 'Dimensión'], errors='ignore').values.tolist()
                    tj_just_transition_rubric[indicador] = {'valores': valores, 'dimension': dimension}
        except Exception as e:
            st.error(f"❌ Error cargando TJ Transición Justa: {e}")
            st.write("Sheets disponibles:", list(pd.ExcelFile('./Rubricas_6ago2025.xlsx').sheet_names))
    except Exception as e:
        st.error(f"Error leyendo las rúbricas: {e}")

    # Debug: Show Excel sheet names and rubric status
    try:
        available_sheets = pd.ExcelFile('./Rubricas_6ago2025.xlsx').sheet_names
        st.warning(f"**Sheets disponibles en Rubricas_6ago2025.xlsx:** {', '.join(available_sheets)}")
    except Exception as e:
        st.error(f"Error leyendo Excel: {e}")

    st.error(f"""
    **ESTADO DE RÚBRICAS:**
    - Engagement: {len(engagement_rubric)} criterios
    - Performance: {len(performance_rubric)} criterios
    - Parteval: {len(parteval_rubric)} criterios
    - Gender: {len(gender_rubric)} criterios
    - TJ Traditional: {len(tj_traditional_rubric)} criterios  ⚠️
    - TJ Just Transition: {len(tj_just_transition_rubric)} criterios  ⚠️
    """)

    # Function to extract document structure
    def extract_docx_structure(docx_path):
        from docx import Document
        doc = Document(docx_path)
        filename = os.path.basename(docx_path)
        rows = []
        current_headers = {i: '' for i in range(1, 7)}
        para_counter = 0

        def get_header_level(style_name):
            for i in range(1, 7):
                if style_name.lower().startswith(f'heading {i}'.lower()):
                    return i
            return None

        def header_dict():
            return {f'header_{i}': current_headers[i] for i in range(1, 7)}

        for para in doc.paragraphs:
            para_counter += 1
            level = get_header_level(para.style.name)
            if level and 1 <= level <= 6:
                current_headers[level] = para.text.strip()
                for l in range(level+1, 7):
                    current_headers[l] = ''
                rows.append({
                    'filename': filename,
                    **header_dict(),
                    'content': '',
                    'source_type': 'heading',
                    'paragraph_number': para_counter,
                    'page_number': None
                })
            elif para.text.strip():
                rows.append({
                    'filename': filename,
                    **header_dict(),
                    'content': para.text.strip(),
                    'source_type': 'paragraph',
                    'paragraph_number': para_counter,
                    'page_number': None
                })
        return pd.DataFrame(rows)

    # Function to split text into chunks respecting the token limit
    def split_text_into_chunks(text, max_tokens=7000):
        import re
        # Split by paragraphs first
        paragraphs = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0

        # Rough estimate: 1 token ≈ 4 characters in Spanish
        tokens_per_char = 0.25

        for para in paragraphs:
            # Estimate tokens in this paragraph
            para_tokens = len(para) * tokens_per_char

            # If adding this paragraph would exceed the max, start a new chunk
            if current_length + para_tokens > max_tokens and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_tokens
            else:
                current_chunk.append(para)
                current_length += para_tokens

        # Add the last chunk if there's content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    # Function to directly evaluate content against a criterion using LLM
    def evaluate_criterion_with_llm(document_text, criterion, descriptions):
        """Evaluate document against a criterion directly with LLM"""

        # Split document into manageable chunks if needed
        chunks = split_text_into_chunks(document_text)

        # If text fits in one chunk, evaluate directly
        if len(chunks) == 1:
            return evaluate_single_chunk(chunks[0], criterion, descriptions)

        # For multiple chunks, evaluate each and then synthesize
        chunk_results = []
        for i, chunk in enumerate(chunks):
            st.info(f"Evaluando criterio '{criterion}' - Fragmento {i+1}/{len(chunks)}")
            result = evaluate_single_chunk(chunk, criterion, descriptions)
            chunk_results.append(result)

        # Synthesize results from all chunks
        return synthesize_evaluations(chunk_results, criterion, descriptions)

    # Function to evaluate a single text chunk
    def evaluate_single_chunk(text_chunk, criterion, descriptions):
        """Evaluate a single text chunk against a criterion with expanded analysis and evidence"""
        import json

        # Build prompt
        prompt = f"""
        Estás evaluando un documento contra un criterio específico.
        
        Criterio: {criterion}
        
        Descripciones de los niveles de puntuación:
        {json.dumps(descriptions, indent=2)}
        
        Contenido del documento a evaluar:
        {text_chunk}
        
        Analiza qué tan bien el documento cumple con este criterio. Proporciona:
        
        1. Un análisis DETALLADO (2-3 párrafos) que explique a fondo el razonamiento detrás de tu evaluación. Proporciona un razonamiento profundo que abarque los aspectos del criterio.
        
        2. Una puntuación de 1-5 (donde 1 es la más baja y 5 es la más alta).
        
        3. EVIDENCIA del documento que respalde tu puntuación. Incluye entre 5-8 citas textuales del documento, indicando cómo cada fragmento contribuye a tu evaluación.
        
        Formatea tu respuesta como un objeto JSON con las siguientes claves:
        {{"analysis": "tu análisis detallado aquí", "score": puntuación_numérica_entre_1_y_5, "evidence": "citas textuales del documento (5-8 párrafos)"}}
        
        Devuelve solo el objeto JSON, nada más.
        """

        # Call LLM using OpenAI v0.28 syntax
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "Eres un experto evaluador de documentos que proporciona análisis detallados basados en criterios específicos. Tu evidencia cita fragmentos del texto original."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=7000
            )
            raw = response["choices"][0]["message"]["content"].strip()
            parsed = json.loads(raw)
            return parsed
        except Exception as e:
            return {'score': 0, 'analysis': f'Error: {str(e)}', 'evidence': ''}

    # Function to synthesize evaluations
    def synthesize_evaluations(chunk_results, criterion, descriptions):
        """Synthesize evaluations from multiple document chunks with expanded analysis and evidence"""
        import json

        # Extract and format the individual evaluations for the synthesis
        individual_evals = []
        all_evidence = []

        for i, result in enumerate(chunk_results):
            individual_evals.append(f"Evaluación del fragmento {i+1}:\n" +
                                    f"Puntuación: {result.get('score', 0)}\n" +
                                    f"Análisis: {result.get('analysis', '')}")

            # Collect all evidence
            evidence = result.get('evidence', '')
            if evidence:
                all_evidence.append(f"Evidencia del fragmento {i+1}:\n{evidence}")

        # Define separator outside the f-string to avoid backslash issues
        separator = "\n\n"

        # Create a synthesis prompt
        synthesis_prompt = f"""
        Has evaluado un documento dividido en múltiples fragmentos contra el criterio: {criterion}
        
        Aquí están las evaluaciones individuales de cada fragmento:
        
        {separator.join(individual_evals)}
        
        Basándote en estas evaluaciones individuales, proporciona:
        
        1. Un análisis DETALLADO (2-3 párrafos) que integre los hallazgos clave de todos los fragmentos. Este análisis debe ser comprensivo y abarcar los aspectos relevantes encontrados en el documento.
        
        2. Una puntuación general de 1-5 (puedes promediar las puntuaciones o ajustar según sea necesario)
        
        3. Las evidencias más importantes del documento. Selecciona las 8-10 citas textuales más relevantes de los fragmentos individuales.
        
        Formatea tu respuesta como un objeto JSON con las siguientes claves:
        {{"analysis": "tu análisis global detallado aquí", "score": puntuación_general_entre_1_y_5, "evidence": "las citas textuales más relevantes del documento (8-10 párrafos)"}}
        
        Devuelve solo el objeto JSON, nada más.
        """

        # Call LLM for synthesis using OpenAI v0.28 syntax
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "Eres un experto evaluador de documentos que sintetiza análisis de múltiples fragmentos de texto para producir evaluaciones detalladas con evidencia textual."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=7000
            )
            raw = response["choices"][0]["message"]["content"].strip()
            parsed = json.loads(raw)
            return parsed
        except Exception as e:
            # If synthesis fails, combine results manually in a more limited way
            avg_score = sum(r.get('score', 0) for r in chunk_results) / len(chunk_results)
            # Take only the first paragraph of each analysis to avoid token limits
            analysis_parts = []
            for r in chunk_results:
                analysis = r.get('analysis', '')
                first_para = analysis.split('\n\n')[0] if '\n\n' in analysis else analysis
                analysis_parts.append(first_para)

            # Take only the first few evidence items
            evidence_parts = []
            evidence_count = 0
            for evidence in all_evidence:
                parts = evidence.split('\n\n')
                # Add up to 2 evidence parts per chunk
                for part in parts[:2]:
                    if evidence_count < 8:  # Limit to 8 total evidence parts
                        evidence_parts.append(part)
                        evidence_count += 1

            return {
                'score': avg_score,
                'analysis': separator.join(analysis_parts),
                'evidence': separator.join(evidence_parts)
            }

    # Document upload interface
    uploaded_file = st.file_uploader("Suba un archivo DOCX para evaluación:", type=["docx"])

    # Move instructions/info to the top of the tab
    st.info("""
    **Instrucciones:**
    1. Suba un archivo DOCX y presione el botón 'Procesar y Evaluar'.
    2. Revise los resultados de cada rúbrica en la tabla interactiva.
    3. Visualice las puntuaciones promedio por dimensión y subdimensión en los gráficos de barras.
    4. Descargue todos los resultados y evidencias en un archivo ZIP.
    """)
    # Unified process, evaluate, and download button
    st.markdown("#### Procesamiento y Evaluación de Documento")
    st.markdown('---')
    if st.button('Procesar y Evaluar'):
        # Only process if file is uploaded and not already processed for this file
        if uploaded_file is not None:
            file_hash = hash(uploaded_file.getvalue())
            if st.session_state.get('last_file_hash') != file_hash:
                with st.spinner("Procesando documento..."):
                    try:
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
                        tmp_file.write(uploaded_file.read())
                        tmp_file.close()
                        progress_bar = st.progress(0, text="Leyendo y extrayendo contenido del DOCX...")
                        doc_result = docx2python(tmp_file.name)
                        df = extract_docx_structure(tmp_file.name)
                        progress_bar.progress(0.2, text="Documento cargado. Procesando estructura...")
                        header_1_values = df['header_1'].dropna().unique()
                        llm_summary_rows = []
                        llm_progress = st.progress(0, text="Procesando secciones con LLM...")
                        total_sections = len(header_1_values)
                        for idx, header in enumerate(header_1_values):
                            section_df = df[df['header_1'] == header].copy()
                            full_text = '\n'.join(section_df['content'].astype(str).tolist()).strip()
                            if not full_text:
                                llm_output = ""
                            else:
                                llm_progress.progress((idx+1)/total_sections, text=f"Procesando sección: {header}")
                                try:
                                    response = openai.ChatCompletion.create(
                                        model="gpt-4.1-mini",
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant that rewrites extracted document content into well-structured, formal paragraphs. Do not rewrite the original content, just reconstruct it in proper, coherent paragraphs, without rephrasing or paraphrasing or rewording."},
                                            {"role": "user", "content": full_text}
                                        ],
                                        max_tokens=1024,
                                        temperature=0.01,
                                    )
                                    llm_output = response["choices"][0]["message"]["content"].strip()
                                except Exception as e:
                                    llm_output = f"[LLM ERROR: {e}]"
                            llm_summary_rows.append({'header_1': header, 'llm_paragraph': llm_output})
                        llm_progress.progress(1.0, text="LLM parsing completado.")
                        llm_summary_df = pd.DataFrame(llm_summary_rows)
                        llm_summary_df['n_words'] = llm_summary_df['llm_paragraph'].str.split().str.len()
                        exploded_df = llm_summary_df.assign(
                            llm_paragraph=llm_summary_df['llm_paragraph'].str.split('\n')
                        ).explode('llm_paragraph')
                        exploded_df = exploded_df.reset_index(drop=True)
                        exploded_df = exploded_df[exploded_df['llm_paragraph'].str.strip() != '']
                        full_document_text = "\n\n".join(exploded_df['llm_paragraph'].tolist())
                        file_size = os.path.getsize(tmp_file.name)
                        n_words = exploded_df['llm_paragraph'].str.split().str.len().sum()
                        n_paragraphs = len(exploded_df)
                        st.session_state['full_document_text'] = full_document_text
                        st.session_state['document_stats'] = {
                            'file_size': file_size,
                            'n_words': n_words,
                            'n_paragraphs': n_paragraphs
                        }
                        st.session_state['exploded_df'] = exploded_df
                        st.session_state['last_file_hash'] = file_hash
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass
                        progress_bar.progress(0.8, text="Documento procesado. Listo para evaluación.")
                        st.info(f"**Resumen del documento:**\n\n" + 
                                f"- Tamaño del archivo: {file_size/1024:.2f} KB\n" + 
                                f"- Número de palabras: {n_words}\n" + 
                                f"- Número de párrafos: {n_paragraphs}")
                        st.markdown("#### Estructura extraída del documento:")
                        st.dataframe(exploded_df, use_container_width=True)
                        progress_bar.progress(1.0, text="Procesamiento completo.")
                    except Exception as e:
                        st.error(f"Error procesando el documento: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                        st.stop()
        # Now, always run rubric evaluation if document is processed
        document_text = st.session_state.get('full_document_text', '')
        if not document_text:
            st.error("No se pudo recuperar el texto del documento. Por favor, vuelva a cargar el archivo.")
            st.stop()
        rubrics = [
            ("Participación (Engagement)", engagement_rubric),
            ("Desempeño (Performance)", performance_rubric),
            ("Participación Evaluada (Parteval)", parteval_rubric),
            ("Género (Gender)", gender_rubric),
            ("TJ (Tradicional)", tj_traditional_rubric),
            ("TJ (Transición Justa)", tj_just_transition_rubric)
        ]
        rubric_results = []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        MAX_WORKERS = 48
        def eval_one_criterion(args):
            crit, descriptions, dimension, rubric_name = args
            try:
                result = evaluate_criterion_with_llm(document_text, crit, descriptions)
                return {
                    'Criterio': crit,
                    'Dimensión': dimension,
                    'Score': result.get('score', 0),
                    'Análisis': result.get('analysis', ''),
                    'Evidencia': result.get('evidence', ''),
                    'Error': result.get('error', '') if 'error' in result else '',
                    'Rúbrica': rubric_name
                }
            except Exception as e:
                return {
                    'Criterio': crit,
                    'Dimensión': dimension,
                    'Score': 0,
                    'Análisis': '',
                    'Evidencia': '',
                    'Error': str(e),
                    'Rúbrica': rubric_name
                }
        for rubric_name, rubric_dict in rubrics:
            # Skip empty rubrics
            if not rubric_dict:
                st.warning(f"Saltando rúbrica {rubric_name}: sin criterios cargados")
                continue

            rubric_analysis_data = []
            n_criteria = len(rubric_dict)
            progress = st.progress(0, text=f"Iniciando evaluación por rúbrica: {rubric_name}...")
            with st.spinner(f'Evaluando documento por rúbrica: {rubric_name}...'):
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {
                        executor.submit(eval_one_criterion, (
                            crit,
                            rubric_data['valores'] if isinstance(rubric_data, dict) else rubric_data,
                            rubric_data.get('dimension', 'No especificada') if isinstance(rubric_data, dict) else 'No especificada',
                            rubric_name
                        )): (crit, idx)
                        for idx, (crit, rubric_data) in enumerate(rubric_dict.items())
                    }
                    completed = 0
                    for future in as_completed(futures):
                        result = future.result()
                        rubric_analysis_data.append(result)
                        completed += 1
                        crit, idx = futures[future]
                        progress.progress(completed / n_criteria, text=f"Evaluando criterio: {crit}")
            rubric_results.append((rubric_name, pd.DataFrame(rubric_analysis_data)))
        # Show and allow download of both results only after evaluation
        if rubric_results:
            for rubric_name, rubric_analysis_df in rubric_results:
                st.markdown(f'#### Resultados de la evaluación por rúbrica: {rubric_name}')
                if not rubric_analysis_df.empty:
                    # Ensure 'Evidencia' column is present
                    if 'Evidencia' not in rubric_analysis_df.columns:
                        rubric_analysis_df['Evidencia'] = ''

                    # Reorder columns to show in logical order: Criterio, Dimensión, Score, Análisis, Evidencia, Error, Rúbrica
                    cols = rubric_analysis_df.columns.tolist()
                    desired_order = ['Criterio', 'Dimensión', 'Score', 'Análisis', 'Evidencia', 'Error', 'Rúbrica']
                    new_order = [col for col in desired_order if col in cols]
                    remaining_cols = [col for col in cols if col not in desired_order]
                    final_order = new_order + remaining_cols
                    rubric_analysis_df = rubric_analysis_df[final_order]
                    # Normalize 'Evidencia' column to always be a string
                    if 'Evidencia' in rubric_analysis_df.columns:
                        rubric_analysis_df['Evidencia'] = rubric_analysis_df['Evidencia'].apply(
                            lambda x: "\n".join(x) if isinstance(x, list) else (str(x) if x is not None else "")
                        )
                    st.dataframe(rubric_analysis_df, use_container_width=True)
                else:
                    st.warning(f"No se generaron resultados para la rúbrica: {rubric_name}")
            # Provide a zip download for both results
            import io, zipfile
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                for rubric_name, rubric_analysis_df in rubric_results:
                    # Normalize 'Evidencia' column to always be a string before exporting
                    if 'Evidencia' in rubric_analysis_df.columns:
                        rubric_analysis_df['Evidencia'] = rubric_analysis_df['Evidencia'].apply(
                            lambda x: "\n".join(x) if isinstance(x, list) else (str(x) if x is not None else "")
                        )
                    csv = rubric_analysis_df.to_csv(index=False)
                    arcname = f"evaluacion_rubrica_{rubric_name.replace(' ', '_').lower()}.csv"
                    zipf.writestr(arcname, csv)
            zip_buffer.seek(0)
            st.download_button(
                label="Descargar ambos resultados como ZIP",
                data=zip_buffer,
                file_name="resultados_rubricas.zip",
                mime="application/zip"
            )
        else:
            st.warning("No se generaron resultados para ninguna rúbrica.")
    else:
        st.info("Por favor suba un archivo DOCX para comenzar y pulse el botón para procesar y evaluar.")

#===================######################=====================
# ================== TAB 5: DOCUMENT CHAT =====================
with tab4:
    st.header("Pregúntale a tus Documentos")

    # Presentation box
    st.info("""

    Arrastra uno o más DOCX/TXT (max 200MB) y conversa directamente con su contenido. Este chat responde solo con la información de los archivos cargados—no recurre a fuentes externas—para ayudarte a revisar propuestas antes de enviarlas, aclarar pasajes de informes, preparar notas técnicas y comprobar coherencias entre objetivos, actividades y resultados.

    Al formular tus preguntas, indica el nivel de detalle que necesitas y pide que las respuestas incluyan citas breves entre comillas y metadatos (título, sección/página, año) para mantener la trazabilidad. Si un dato no existe en los archivos, se marcará "ND" sin inventar información.

    **Úsalo para:**
    - Aprender de experiencias de evaluación
    - Profundizar en el conocimiento de secciones de los informes
    - Preparar notas técnicas a partir de esta evidencia
    - Verificar la coherencia entre objetivos, actividades y resultados
    - Generar tablas copiables a Excel
    - Comparar varios documentos (hasta de 200MB) en una misma conversación

    *Nota: El chat mantiene memoria de la conversación durante la sesión activa.*
    """)

    # Session state for chat and document
    if 'doc_chat_history' not in st.session_state:
        st.session_state['doc_chat_history'] = []
    if 'doc_chat_docs' not in st.session_state:
        st.session_state['doc_chat_docs'] = []

    uploaded_files = st.file_uploader("Sube uno o más archivos DOCX o TXT para chatear:", type=["docx", "txt"], accept_multiple_files=True)
    if 'doc_chat_docs' not in st.session_state:
        st.session_state['doc_chat_docs'] = []
    if uploaded_files:
        # Only reset docs and chat history if files changed
        uploaded_filenames = sorted([f.name for f in uploaded_files])
        existing_filenames = sorted([doc['filename'] for doc in st.session_state['doc_chat_docs']]) if st.session_state['doc_chat_docs'] else []
        if uploaded_filenames != existing_filenames:
            st.session_state['doc_chat_docs'] = []
            for uploaded_file in uploaded_files:
                try:
                    if uploaded_file.name.endswith(".docx"):
                        from docx import Document
                        doc = Document(uploaded_file)
                        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                    else:
                        full_text = uploaded_file.read().decode("utf-8", errors="ignore")
                    st.session_state['doc_chat_docs'].append({
                        "filename": uploaded_file.name,
                        "text": full_text
                    })
                except Exception as e:
                    st.error(f"Error al procesar el documento '{uploaded_file.name}': {str(e)}")
            st.session_state['doc_chat_history'] = []  # Only reset chat when new files uploaded
            st.success(f"{len(st.session_state['doc_chat_docs'])} documento(s) cargado(s) y listo(s) para chatear.")

    # Show filenames and previews
    if st.session_state['doc_chat_docs']:
        st.info(f"Documentos activos: {', '.join([doc['filename'] for doc in st.session_state['doc_chat_docs']])}")
        for doc in st.session_state['doc_chat_docs']:
            with st.expander(f"Vista previa: {doc['filename']} (primeros 500 caracteres)"):
                st.write(doc['text'][:500] + ("..." if len(doc['text']) > 500 else ""))

    # Chat interface
    if st.session_state['doc_chat_docs']:
        with st.form("doc_chat_form", clear_on_submit=True):
            user_input = st.text_area("Escribe tu preguntas (para las respuestas se considera el texto completo de los documentos cargados):", key="doc_chat_input")
            submitted = st.form_submit_button("Enviar pregunta")
            if submitted and user_input.strip():
                # Add user message to history
                if 'doc_chat_history' not in st.session_state:
                    st.session_state['doc_chat_history'] = []
                st.session_state['doc_chat_history'].append({"role": "user", "content": user_input.strip()})
                # Chunking: combine all docs, split into 7000-character chunks for safety
                import openai
                import numpy as np
                import faiss
                # 1. Split all docs into larger, overlapping chunks (~1500 chars, 300 overlap)
                def chunk_text(text, chunk_size=2000, overlap=300):
                    chunks = []
                    start = 0
                    while start < len(text):
                        end = min(start + chunk_size, len(text))
                        chunks.append(text[start:end])
                        start += chunk_size - overlap
                    return chunks
                all_chunks = []
                for doc in st.session_state['doc_chat_docs']:
                    all_chunks.extend(chunk_text(doc['text']))
                if not all_chunks:
                    st.session_state['doc_chat_history'].append({"role": "assistant", "content": "[No se encontraron fragmentos en los documentos.]"})
                else:
                    question = user_input.strip()
                    # 2. Detect summary/broad queries
                    summary_keywords = [
                        "resumen", "conclusiones", "hallazgos", "principales", "summary", "conclusion"
                    ]
                    is_summary_query = any(kw in question.lower() for kw in summary_keywords)
                    emb_model = "text-embedding-3-large"
                    if is_summary_query:
                        # Fallback: use as much of the full document(s) as fits
                        context = "\n---\n".join(doc['text'][:12000] for doc in st.session_state['doc_chat_docs'])
                        messages = [
                            {"role": "system", "content": "Eres un asistente experto en análisis documental. Responde usando solo la información del documento proporcionado."},
                            {"role": "system", "content": f"Texto del documento:\n{context}"}
                        ]
                        for msg in st.session_state['doc_chat_history'][-5:]:
                            messages.append(msg)
                        try:
                            response = openai.ChatCompletion.create(
                                model="gpt-4.1-mini",
                                messages=messages,
                                max_tokens=2048,
                                temperature=0.3
                            )
                            answer = response['choices'][0]['message']['content'].strip()
                            st.session_state['doc_chat_history'].append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.session_state['doc_chat_history'].append({"role": "assistant", "content": f"[Error al obtener respuesta: {str(e)}]"})
                    else:
                        # 3. Get embeddings for all chunks (batch)
                        try:
                            chunk_embs_resp = openai.Embedding.create(input=all_chunks, model=emb_model)
                            chunk_embs = [item["embedding"] for item in chunk_embs_resp["data"]]
                        except Exception as e:
                            st.session_state['doc_chat_history'].append({"role": "assistant", "content": f"[Error al obtener embeddings de los fragmentos: {str(e)}]"})
                            chunk_embs = []
                        # 4. Embed user question
                        try:
                            question_emb = openai.Embedding.create(input=question, model=emb_model)["data"][0]["embedding"]
                        except Exception as e:
                            st.session_state['doc_chat_history'].append({"role": "assistant", "content": f"[Error al obtener embedding de la pregunta: {str(e)}]"})
                            question_emb = None
                        # 5. Use faiss to retrieve top N relevant chunks
                        if chunk_embs and question_emb:
                            import faiss
                            import numpy as np
                            import re
                            dim = len(chunk_embs[0])
                            xb = np.array(chunk_embs).astype('float32')
                            index = faiss.IndexFlatIP(dim)
                            # Normalize for cosine similarity
                            faiss.normalize_L2(xb)
                            xq = np.array([question_emb]).astype('float32')
                            faiss.normalize_L2(xq)
                            top_n = 50
                            D, I = index.search(xq, top_n)
                            selected_chunks = [all_chunks[i] for i in I[0] if i < len(all_chunks)]
                            # Also retrieve all chunks with exact matches for key question terms
                            stopwords = set([
                                'el','la','los','las','de','del','y','en','a','un','una','que','por','con','para','es','al','se','su','sus','o','u','como','más','menos','le','lo','su','the','and','of','in','to','for','is','on','at','by','an','or','as','be','are','was','were','from','it','this','that','with','but','not','can','may','do','does'
                            ])
                            qwords = [w for w in re.findall(r'\w+', question.lower()) if w not in stopwords and len(w) > 2]
                            keyword_chunks = []
                            for chunk in all_chunks:
                                chunk_lc = chunk.lower()
                                if any(qw in chunk_lc for qw in qwords):
                                    keyword_chunks.append(chunk)
                            # Merge, deduplicate (embedding + keyword)
                            seen = set()
                            merged_chunks = []
                            for chunk in selected_chunks + keyword_chunks:
                                if chunk not in seen:
                                    merged_chunks.append(chunk)
                                    seen.add(chunk)
                            context = '\n---\n'.join(merged_chunks)
                            # 6. Send to LLM
                            messages = [
                                {"role": "system", "content": "Eres un asistente experto en análisis documental. Responde usando solo la información del documento proporcionado. Si la información no es explícita, infiere la respuesta usando pistas contextuales y tu capacidad de síntesis."},
                                {"role": "system", "content": f"Fragmentos relevantes del documento:\n{context}"}
                            ]
                            for msg in st.session_state['doc_chat_history'][-5:]:
                                messages.append(msg)
                            try:
                                response = openai.ChatCompletion.create(
                                    model="gpt-4.1-mini",
                                    messages=messages,
                                    max_tokens=2048,
                                    temperature=0.3
                                )
                                answer = response['choices'][0]['message']['content'].strip()
                                st.session_state['doc_chat_history'].append({"role": "assistant", "content": answer})
                            except Exception as e:
                                st.session_state['doc_chat_history'].append({"role": "assistant", "content": f"[Error al obtener respuesta: {str(e)}]"})

        # Display chat history in a persistent, scrollable container
        st.markdown(
            '''
            <div style="height:600px; overflow-y:auto; border:1px solid #333; border-radius:8px; padding:1em; background:#18191a; margin-bottom:1em;">
            '''
            +
            "".join(
                f"<div style='margin-bottom:1em; color:#2980b9;'><b>Tú:</b> {msg['content']}</div>" if msg['role']=='user'
                else f"<div style='margin-bottom:1em; color:#27ae60;'><b>Asistente:</b> {msg['content']}</div>"
                for msg in st.session_state['doc_chat_history']
            )
            +
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Sube uno o más documentos válidos para comenzar el chat.")

# ================== TAB 6: EVALUACIÓN DE PRODOCS =====================
with tab3:
    st.header("Diagnóstico de Sostenibilidad del Proyecto")
    
    # Read rubric from Excel file
    import pandas as pd
    prodoc_rubric = {}
    
    try:
        # Load rubric from PRODOC_rubric.xlsx
        df_rubric_prodoc = pd.read_excel('./PRODOC_rubric.xlsx', sheet_name='rubric')
        
        # Check if 'Indicador' column exists
        if 'Indicador' not in df_rubric_prodoc.columns:
            st.error("La columna 'Indicador' no existe en el archivo Excel.")
            # Try to use the first column as 'Indicador' if it exists
            if len(df_rubric_prodoc.columns) > 0:
                indicador_col = df_rubric_prodoc.columns[0]
                st.warning(f"Usando la columna '{indicador_col}' como columna de indicadores.")
                df_rubric_prodoc.rename(columns={indicador_col: 'Indicador'}, inplace=True)
            else:
                prodoc_rubric = {}
                st.error("No se pudo encontrar una columna para los criterios.")
        
        # Process each row to extract criteria and values
        for idx, row in df_rubric_prodoc.iterrows():
            # Get the indicator value
            indicador = row['Indicador']
            
            # Skip empty indicators
            if pd.isna(indicador) or str(indicador).strip() == '':
                continue
            
            # Convert to string if it's not already
            indicador = str(indicador).strip()
            
            # Get level columns (Nivel 0, Nivel 1, etc.)
            level_cols = [col for col in df_rubric_prodoc.columns if col.startswith('Nivel')]
            
            # Extract values from level columns
            valores = []
            for col in level_cols:
                val = row[col]
                if not pd.isna(val) and str(val).strip() != '':
                    valores.append(str(val).strip())
            
            # Store in our rubric dictionary
            prodoc_rubric[indicador] = valores
        
        # Success message with count of loaded criteria
        st.success(f"Rúbrica cargada correctamente desde PRODOC_rubric.xlsx: {len(prodoc_rubric)} criterios cargados.")
    except FileNotFoundError:
        st.error("No se encontró el archivo PRODOC_rubric.xlsx. Por favor, asegúrese de que existe en el directorio de la aplicación.")
    except Exception as e:
        st.error(f"Error al cargar la rúbrica desde PRODOC_rubric.xlsx: {str(e)}")
    
    # Display the loaded rubric
    with st.expander("Ver rúbrica cargada"):
        st.subheader("Criterios de Evaluación PRODOC")
        for criterion, values in prodoc_rubric.items():
            st.markdown(f"**{criterion}**: {values}")
    
    # Instrucciones generales
    st.info("""
    **Instrucciones:**
    1. Suba los archivos DOCX en las secciones correspondientes.
    2. Presione el botón de evaluación en cada sección para analizar el documento.
    3. Revise los resultados de cada rúbrica en las tablas interactivas.
    4. Visualice las puntuaciones promedio por dimensión en los gráficos de barras.
    5. Descargue todos los resultados y evidencias en archivos ZIP.
    """)

    # Sección 1: Documento PRODOC
    st.markdown("### 📄 Sección 1: Documento PRODOC")
    st.info("Suba el documento PRODOC para evaluación de sostenibilidad del proyecto.")
    uploaded_file_prodoc = st.file_uploader("Suba un archivo DOCX del PRODOC:", type=["docx"], key="prodoc_file_uploader")

    if st.button('🔍 Procesar y Evaluar PRODOC', key="prodoc_process_button"):
        # Only process if file is uploaded
        if uploaded_file_prodoc is not None:
            uploaded_file = uploaded_file_prodoc
            document_type = "PRODOC"
            st.markdown(f"#### Procesando documento {document_type}...")

            file_hash = hash(uploaded_file.getvalue())
            if st.session_state.get('prodoc_last_file_hash') != file_hash:
                with st.spinner("Procesando documento..."):
                    try:
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
                        tmp_file.write(uploaded_file.read())
                        tmp_file.close()
                        progress_bar = st.progress(0, text="Leyendo y extrayendo contenido del DOCX...")
                        doc_result = docx2python(tmp_file.name)
                        df = extract_docx_structure(tmp_file.name)
                        progress_bar.progress(0.2, text="Documento cargado. Procesando estructura...")
                        header_1_values = df['header_1'].dropna().unique()
                        llm_summary_rows = []
                        llm_progress = st.progress(0, text="Procesando secciones con LLM...")
                        total_sections = len(header_1_values)
                        for idx, header in enumerate(header_1_values):
                            section_df = df[df['header_1'] == header].copy()
                            full_text = '\n'.join(section_df['content'].astype(str).tolist()).strip()
                            if not full_text:
                                llm_output = ""
                            else:
                                llm_progress.progress((idx+1)/total_sections, text=f"Procesando sección: {header}")
                                try:
                                    response = openai.ChatCompletion.create(
                                        model="gpt-4.1-mini",
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant that rewrites extracted document content into well-structured, formal paragraphs. Do not rewrite the original content, just reconstruct it in proper, coherent paragraphs, without rephrasing or paraphrasing or rewording."},
                                            {"role": "user", "content": full_text}
                                        ],
                                        max_tokens=1024,
                                        temperature=0.01,
                                    )
                                    llm_output = response["choices"][0]["message"]["content"].strip()
                                except Exception as e:
                                    llm_output = f"[LLM ERROR: {e}]"
                            llm_summary_rows.append({'header_1': header, 'llm_paragraph': llm_output})
                        llm_progress.progress(1.0, text="LLM parsing completado.")
                        llm_summary_df = pd.DataFrame(llm_summary_rows)
                        llm_summary_df['n_words'] = llm_summary_df['llm_paragraph'].str.split().str.len()
                        exploded_df = llm_summary_df.assign(
                            llm_paragraph=llm_summary_df['llm_paragraph'].str.split('\n')
                        ).explode('llm_paragraph')
                        exploded_df = exploded_df.reset_index(drop=True)
                        exploded_df = exploded_df[exploded_df['llm_paragraph'].str.strip() != '']
                        full_document_text = "\n\n".join(exploded_df['llm_paragraph'].tolist())
                        file_size = os.path.getsize(tmp_file.name)
                        n_words = exploded_df['llm_paragraph'].str.split().str.len().sum()
                        n_paragraphs = len(exploded_df)
                        st.session_state['prodoc_full_document_text'] = full_document_text
                        st.session_state['prodoc_document_stats'] = {
                            'file_size': file_size,
                            'n_words': n_words,
                            'n_paragraphs': n_paragraphs
                        }
                        st.session_state['prodoc_exploded_df'] = exploded_df
                        st.session_state['prodoc_last_file_hash'] = file_hash
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass
                        progress_bar.progress(0.8, text="Documento procesado. Listo para evaluación.")
                        st.info(f"**Resumen del documento:**\n\n" + 
                                f"- Tamaño del archivo: {file_size/1024:.2f} KB\n" + 
                                f"- Número de palabras: {n_words}\n" + 
                                f"- Número de párrafos: {n_paragraphs}")
                        st.markdown("#### Estructura extraída del documento:")
                        st.dataframe(exploded_df, use_container_width=True)
                        progress_bar.progress(1.0, text="Procesamiento completo.")
                    except Exception as e:
                        st.error(f"Error procesando el documento: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                        st.stop()
            
            # Now, always run rubric evaluation if document is processed
            document_text = st.session_state.get('prodoc_full_document_text', '')
            if not document_text:
                st.error("No se pudo recuperar el texto del documento. Por favor, vuelva a cargar el archivo.")
                st.stop()
                
            # Define the rubric to evaluate, using the same structure as tab4
            rubrics = [
                ("Evaluación PRODOC", prodoc_rubric)
            ]
            
            rubric_results = []
            from concurrent.futures import ThreadPoolExecutor, as_completed
            MAX_WORKERS = 48
            
            def eval_one_criterion(args):
                crit, descriptions, dimension, rubric_name = args
                try:
                    result = evaluate_criterion_with_llm(document_text, crit, descriptions)
                    return {
                        'Criterio': crit,
                        'Dimensión': dimension,
                        'Score': result.get('score', 0),
                        'Análisis': result.get('analysis', ''),
                        'Evidencia': result.get('evidence', ''),
                        'Error': result.get('error', '') if 'error' in result else '',
                        'Rúbrica': rubric_name
                    }
                except Exception as e:
                    return {
                        'Criterio': crit,
                        'Dimensión': dimension,
                        'Score': 0,
                        'Análisis': '',
                        'Evidencia': '',
                        'Error': str(e),
                        'Rúbrica': rubric_name
                    }
            
            for rubric_name, rubric_dict in rubrics:
                rubric_analysis_data = []
                n_criteria = len(rubric_dict)
                progress = st.progress(0, text=f"Iniciando evaluación por rúbrica: {rubric_name}...")
                with st.spinner(f'Evaluando documento por rúbrica: {rubric_name}...'):
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = {
                            executor.submit(eval_one_criterion, (
                                crit,
                                rubric_data['valores'] if isinstance(rubric_data, dict) else rubric_data,
                                rubric_data.get('dimension', 'No especificada') if isinstance(rubric_data, dict) else 'No especificada',
                                rubric_name
                            )): (crit, idx)
                            for idx, (crit, rubric_data) in enumerate(rubric_dict.items())
                        }
                        completed = 0
                        for future in as_completed(futures):
                            result = future.result()
                            rubric_analysis_data.append(result)
                            completed += 1
                            crit, idx = futures[future]
                            progress.progress(completed / n_criteria, text=f"Evaluando criterio: {crit}")
                rubric_results.append((rubric_name, pd.DataFrame(rubric_analysis_data)))
            
            # Show and allow download of results only after evaluation
            if rubric_results:
                for rubric_name, rubric_analysis_df in rubric_results:
                    st.markdown(f'#### Resultados de la evaluación por rúbrica: {rubric_name}')
                    if not rubric_analysis_df.empty:
                        # Ensure 'Evidencia' column is present
                        if 'Evidencia' not in rubric_analysis_df.columns:
                            rubric_analysis_df['Evidencia'] = ''

                        # Reorder columns to show in logical order: Criterio, Dimensión, Score, Análisis, Evidencia, Error, Rúbrica
                        cols = rubric_analysis_df.columns.tolist()
                        desired_order = ['Criterio', 'Dimensión', 'Score', 'Análisis', 'Evidencia', 'Error', 'Rúbrica']
                        new_order = [col for col in desired_order if col in cols]
                        remaining_cols = [col for col in cols if col not in desired_order]
                        final_order = new_order + remaining_cols
                        rubric_analysis_df = rubric_analysis_df[final_order]
                        # Normalize 'Evidencia' column to always be a string
                        if 'Evidencia' in rubric_analysis_df.columns:
                            rubric_analysis_df['Evidencia'] = rubric_analysis_df['Evidencia'].apply(
                                lambda x: "\n".join(x) if isinstance(x, list) else (str(x) if x is not None else "")
                            )
                        st.dataframe(rubric_analysis_df, use_container_width=True)
                    else:
                        st.warning(f"No se generaron resultados para la rúbrica: {rubric_name}")
                
                # Create visualizations for the results
                st.markdown("### Visualización de Resultados")
                
                # Prepare data for visualization
                all_scores = []
                for rubric_name, df in rubric_results:
                    for _, row in df.iterrows():
                        all_scores.append({
                            'Criterio': row['Criterio'],
                            'Puntuación': row['Score']
                        })
                
                scores_df = pd.DataFrame(all_scores)
                
                # Calculate the overall average score
                overall_avg = scores_df['Puntuación'].mean()
                
                # Sort the dataframe by score in descending order
                scores_df = scores_df.sort_values(by='Puntuación', ascending=False)
                
                # Create a horizontal bar chart instead of vertical for better label readability
                fig = go.Figure()
                
                # Create short identifiers for criteria (e.g., "Criterio 1", "Criterio 2", etc.)
                scores_df['Criterio_ID'] = [f"Criterio {i+1}" for i in range(len(scores_df))]
                
                # Create custom hover text with full criteria description
                scores_df['Hover_Text'] = scores_df.apply(
                    lambda row: f"<b>{row['Criterio_ID']}</b><br>{row['Criterio']}<br>Puntuación: {row['Puntuación']:.2f}", 
                    axis=1
                )
                
                # Add the bars - horizontal orientation with hover text
                fig.add_trace(go.Bar(
                    y=scores_df['Criterio_ID'],  # Using short identifiers
                    x=scores_df['Puntuación'],
                    text=scores_df['Puntuación'].round(2),
                    textposition='auto',
                    marker_color='#3498db',
                    orientation='h',  # Horizontal bars
                    name='Puntuación',
                    hovertext=scores_df['Hover_Text'],
                    hoverinfo='text'
                ))
                
                # Add the average line - vertical for horizontal chart
                fig.add_trace(go.Scatter(
                    y=scores_df['Criterio_ID'],
                    x=[overall_avg] * len(scores_df),
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name=f'Promedio General: {overall_avg:.2f}'
                ))
                
                # Update layout for better readability
                fig.update_layout(
                    title='Puntuación por Criterio (Ordenado de Mayor a Menor)',
                    xaxis_title='Puntuación',
                    yaxis_title='',  # No need for y-axis title
                    xaxis=dict(range=[0, 5.5]),  # Now x-axis has the scores
                    height=max(400, len(scores_df) * 35),  # Reduced height since we're using shorter labels
                    width=800,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=20, r=20, t=80, b=50),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                
                # Configure y-axis for cleaner look
                fig.update_yaxes(
                    automargin=True  # Automatically adjust margins to fit labels
                )
                
                # Add grid lines for better score reference
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Provide a zip download for all results
                import io, zipfile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zipf:
                    for rubric_name, rubric_analysis_df in rubric_results:
                        # Normalize 'Evidencia' column to always be a string before exporting
                        if 'Evidencia' in rubric_analysis_df.columns:
                            rubric_analysis_df['Evidencia'] = rubric_analysis_df['Evidencia'].apply(
                                lambda x: "\n".join(x) if isinstance(x, list) else (str(x) if x is not None else "")
                            )
                        csv = rubric_analysis_df.to_csv(index=False)
                        arcname = f"evaluacion_prodoc_{rubric_name.replace(' ', '_').lower()}.csv"
                        zipf.writestr(arcname, csv)
                zip_buffer.seek(0)
                st.download_button(
                    label="Descargar resultados como ZIP",
                    data=zip_buffer,
                    file_name="resultados_evaluacion_prodoc.zip",
                    mime="application/zip",
                    key="prodoc_download_button"
                )
            else:
                st.warning("No se generaron resultados para ninguna rúbrica.")
        else:
            st.info("Por favor suba un archivo DOCX para comenzar y pulse el botón para procesar y evaluar.")
    else:
        st.info("Por favor suba un archivo DOCX del PRODOC para comenzar.")

    # Sección 2: Documento de Evaluación
    st.markdown("### 📋 Sección 2: Documento de Evaluación")
    st.info("Suba el documento de evaluación para análisis comparativo con la misma rúbrica de sostenibilidad.")
    uploaded_file_evaluation = st.file_uploader("Suba un archivo DOCX de evaluación:", type=["docx"], key="evaluation_file_uploader")

    if st.button('🔍 Procesar y Evaluar Documento de Evaluación', key="evaluation_process_button"):
        # Only process if file is uploaded
        if uploaded_file_evaluation is not None:
            uploaded_file = uploaded_file_evaluation
            document_type = "Evaluación"
            st.markdown(f"#### Procesando documento {document_type}...")

            file_hash = hash(uploaded_file.getvalue())
            if st.session_state.get('evaluation_last_file_hash') != file_hash:
                with st.spinner("Procesando documento de evaluación..."):
                    try:
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
                        tmp_file.write(uploaded_file.read())
                        tmp_file.close()
                        progress_bar = st.progress(0, text="Leyendo y extrayendo contenido del DOCX...")
                        doc_result = docx2python(tmp_file.name)
                        df = extract_docx_structure(tmp_file.name)

                        # Use the same rubric processing logic as PRODOC
                        all_sections = extract_all_sections(doc_result)

                        # Process with the same rubric (prodoc_rubric)
                        results_evaluation = process_with_rubrics(all_sections, prodoc_rubric, document_type, progress_bar)

                        # Store in session state with different key
                        st.session_state['evaluation_results'] = results_evaluation
                        st.session_state['evaluation_sections'] = all_sections
                        st.session_state['evaluation_last_file_hash'] = file_hash
                        st.session_state['evaluation_document_processed'] = True

                        os.unlink(tmp_file.name)
                        progress_bar.progress(1.0, text="¡Análisis completado!")

                        st.success("✅ Documento de evaluación procesado exitosamente")

                    except Exception as e:
                        st.error(f"Error al procesar el documento de evaluación: {str(e)}")
                        st.session_state['evaluation_document_processed'] = False
                        import traceback
                        st.error(f"Detalles del error: {traceback.format_exc()}")
            else:
                st.info("✅ Este archivo ya ha sido procesado. Mostrando resultados existentes.")
                st.session_state['evaluation_document_processed'] = True
        else:
            st.info("Por favor suba un archivo DOCX de evaluación para comenzar.")
    else:
        st.info("Por favor suba un archivo DOCX de evaluación para comenzar.")

    # Display evaluation results if processed
    if st.session_state.get('evaluation_document_processed', False) and 'evaluation_results' in st.session_state:
        st.markdown("#### 📊 Resultados de Evaluación - Documento de Evaluación")
        results_evaluation = st.session_state['evaluation_results']

        # Same display logic as PRODOC
        if results_evaluation:
            display_results(results_evaluation, document_type="Evaluación", tab_key="evaluation")
        else:
            st.warning("No se generaron resultados para ninguna rúbrica del documento de evaluación.")

# ================== TAB 7: APPRAISAL CHECKLIST (IMPROVED) =====================
# Configuration
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
MAX_WORKERS = 48  # Reduced from 48 to avoid rate limits
OPENAI_MODEL = "gpt-4.1-mini"  # Correct model name

@st.cache_data
def load_appraisal_questions():
    """Load and cache appraisal questions from Excel file"""
    try:
        df = pd.read_excel('./APPRAISAL_rubric.xlsx', sheet_name='rubric')
        if 'Pregunta_Realizada' not in df.columns:
            return None, "Column 'Pregunta_Realizada' not found in Excel file."
        return df, None
    except FileNotFoundError:
        return None, "APPRAISAL_rubric.xlsx not found. Please ensure it exists in the app directory."
    except Exception as e:
        return None, f"Error loading questions file: {str(e)}"

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

def analyze_question_with_llm(question, document_text):
    """Analyze a single question against the document using LLM"""
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert document analyst. Analyze the document against the given question and provide a structured JSON response with exactly this format:
                    {
                        "Respuesta": "Yes/No/Partial/Not Found",
                        "Razonamiento": "Brief explanation of your analysis (max 200 words)",
                        "Evidencia": "Specific text excerpts that support your answer (max 300 words)"
                    }"""
                },
                {
                    "role": "user", 
                    "content": f"Question: {question}\n\nDocument Text: {document_text[:4000]}..."  # Limit context
                }
            ],
            max_tokens=800,
            temperature=0.1,
        )
        
        content = response["choices"][0]["message"]["content"].strip()
        result = json.loads(content)
        
        return {
            'Pregunta': question,
            'Respuesta': result.get('Respuesta', 'Error'),
            'Razonamiento': result.get('Razonamiento', ''),
            'Evidencia': result.get('Evidencia', ''),
            'Status': 'Success'
        }
    
    except Exception as e:
        return {
            'Pregunta': question,
            'Respuesta': 'Error',
            'Razonamiento': f'Analysis failed: {str(e)}',
            'Evidencia': '',
            'Status': 'Error'
        }

def create_results_download(results_df, filename_base="appraisal_checklist"):
    """Create ZIP file with results for download"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        # Add CSV file
        csv_content = results_df.to_csv(index=False)
        zipf.writestr(f"{filename_base}_results.csv", csv_content)
        
        # Add summary report
        summary = f"""
Appraisal Checklist Analysis Summary
===================================

Total Questions Analyzed: {len(results_df)}
Successful Analyses: {len(results_df[results_df['Status'] == 'Success'])}
Failed Analyses: {len(results_df[results_df['Status'] == 'Error'])}

Response Distribution:
{results_df['Respuesta'].value_counts().to_string()}
        """
        zipf.writestr(f"{filename_base}_summary.txt", summary)
    
    zip_buffer.seek(0)
    return zip_buffer

# Main tab interface
with tab1:
    st.header("📋 Valoración Preliminar de Calidad de Proyectos")
    
    # Load questions
    df_appraisal, error_msg = load_appraisal_questions()
    
    if error_msg:
        st.error(error_msg)
        st.stop()
    
    # Display loaded questions
    with st.expander("📝 View Loaded Questions"):
        if df_appraisal is not None:
            st.subheader("Appraisal Checklist Questions")
            for idx, row in df_appraisal.iterrows():
                st.markdown(f"**{idx + 1}.** {row['Pregunta_Realizada']}")
        else:
            st.warning("No se han cargado preguntas")
    
    # Instrucciones
    st.markdown("""
    ### Instrucciones
    
    1. **Subir documento**: Selecciona un archivo DOCX para el análisis de tasación
    2. **Procesar**: Haz clic en 'Analizar documento' para iniciar la evaluación
    3. **Revisar resultados**: Examina los resultados del análisis en la tabla interactiva
    4. **Descargar**: Obtén todos los resultados y las pruebas en un archivo ZIP
    
    ---
    """)
    
    # Sección de carga y procesamiento de archivo
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload DOCX for evaluation:",
            type=["docx"],
            key="appraisal_file_uploader",
            help="Select a Word document (.docx) for appraisal analysis"
        )
    
    with col2:
        st.subheader("📊 Document Stats")
        if 'appraisal_document_stats' in st.session_state:
            stats = st.session_state['appraisal_document_stats']
            
            # Safe access to stats with fallbacks
            file_size = stats.get('file_size', 0)
            word_count = stats.get('word_count', stats.get('n_words', 0))  # Try both keys
            
            if file_size > 0:
                st.metric("File Size", f"{file_size/1024:.1f} KB")
            if word_count > 0:
                st.metric("Words", f"{word_count:,}")
        else:
            st.info("Upload a document to see stats")
    
    # Processing button
    if st.button('🔍 Analyze Document', key="appraisal_process_button", type="primary"):
        if uploaded_file is None:
            st.warning("⚠️ Please upload a DOCX file first.")
            st.stop()
        
        # Extract document content
        with st.spinner("📖 Extracting document content..."):
            doc_result = extract_document_content(uploaded_file)
        
        if not doc_result['success']:
            st.error(f"❌ Error processing document: {doc_result['error']}")
            st.stop()
        
        # Store document stats (using consistent key names)
        st.session_state['appraisal_document_stats'] = {
            'file_size': doc_result['file_size'],
            'word_count': doc_result['word_count'],
            'n_words': doc_result['word_count']  # Keep both for compatibility
        }
        
        # Display document info
        st.success("✅ Document processed successfully!")
        st.info(f"""
        **Document Summary:**
        - File Size: {doc_result['file_size']/1024:.2f} KB
        - Word Count: {doc_result['word_count']:,}
        """)
        
        # Get questions for analysis
        questions = df_appraisal['Pregunta_Realizada'].dropna().unique().tolist()
        
        if not questions:
            st.error("❌ No questions found for analysis.")
            st.stop()
        
        # Analyze questions
        st.markdown("### 🔍 Analysis Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all questions for analysis
            future_to_question = {
                executor.submit(analyze_question_with_llm, q, doc_result['text']): q 
                for q in questions
            }
            
            # Process completed analyses
            completed = 0
            for future in as_completed(future_to_question):
                result = future.result()
                results.append(result)
                completed += 1
                
                # Update progress
                progress = completed / len(questions)
                progress_bar.progress(progress)
                status_text.text(f"Analyzed {completed}/{len(questions)} questions")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Display results
        st.markdown("### 📊 Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Questions", len(results_df))
        with col2:
            success_count = len(results_df[results_df['Status'] == 'Success'])
            st.metric("Successful", success_count)
        with col3:
            error_count = len(results_df[results_df['Status'] == 'Error'])
            st.metric("Errors", error_count)
        with col4:
            if success_count > 0:
                yes_count = len(results_df[results_df['Respuesta'] == 'Yes'])
                st.metric("'Yes' Responses", yes_count)
        
        # Results table
        st.markdown("#### 📋 Detailed Results")
        
        # Filter options
        col1, col2 = st.columns([1, 1])
        with col1:
            response_filter = st.selectbox(
                "Filter by Response:",
                ['All'] + results_df['Respuesta'].unique().tolist(),
                key="response_filter"
            )
        
        # Apply filter
        filtered_df = results_df.copy()
        if response_filter != 'All':
            filtered_df = filtered_df[filtered_df['Respuesta'] == response_filter]
        
        # Display filtered results
        st.dataframe(
            filtered_df[['Pregunta', 'Respuesta', 'Razonamiento', 'Evidencia']], 
            use_container_width=True,
            height=400
        )
        
        # Download section
        st.markdown("### 📥 Download Results")
        
        if len(results_df) > 0:
            zip_buffer = create_results_download(results_df)
            
            st.download_button(
                label="📦 Download Results as ZIP",
                data=zip_buffer,
                file_name="appraisal_checklist_results.zip",
                mime="application/zip",
                key="appraisal_download_button"
            )
            
            st.success("✅ Analysis complete! Use the download button above to get all results.")
        else:
            st.warning("⚠️ No results to download.")
    
    else:
        if uploaded_file:
            st.info("👆 Click 'Analyze Document' to start the appraisal evaluation.")
        else:
            st.info("📁 Please upload a DOCX file to begin.")

#--------------------------#-------------------------------#
#--------------------------#-------------------------------#
# Tab 8: Tablero de seguimiento de recomendaciones y planes de acción
with tab5:
    st.header("📊 Estadísticas sobre Recomendaciones de Evaluaciones y sus Planes de Acción")

    # Comprehensive presentation box
    st.info("""

    En esta ventana se presentan gráficos y estadísticas derivados de las respuestas institucionales a las recomendaciones (incluidos sus planes de acción) ya procesadas por la herramienta. El panel permite seguir el estado de respuesta (completadas, parcialmente completadas, acción no planificada, acción no tomada, rechazadas, sin respuesta), analizar la calidad de la respuesta (coherencia con el plan, calidad del plan, nivel de atención), observar la evolución en el tiempo y la composición por país, año y dimensión (gobernanza, participación, género, transición justa, capacidades, sostenibilidad financiera, incidencia).

    También muestra atributos de las recomendaciones (innovación, precisión/claridad, viabilidad, horizonte temporal, impacto esperado) e identifica barreras de implementación.

    **🎯 Orientado al nivel directivo y unidades de programación para:**
    - Conducir conversaciones de seguimiento: qué ocurrió después de decidir implementar (o no) cada recomendación
    - Identificar por qué quedaron pendientes y qué ajustes corresponden
    - Facilitar acuerdos operativos (responsables y plazos)
    - Realizar reprogramaciones cuando haya cuellos de botella
    - Escalar barreras críticas
    - Preparar notas técnicas/minutas con evidencia gráfica para rendición de cuentas ante mandantes

    **💡 En síntesis:** El tablero transforma la evidencia en prioridades de acción verificables y ciclos de aprendizaje para cerrar brechas y sostener avances.

    **🔍 Análisis por similitud:** Use la opción "Por similitud" para estimar la correspondencia entre el núcleo de la recomendación y la respuesta/plan. Una similitud baja puede señalar un desajuste de alcance, actores o resultados esperados; tómelo como insumo para discutir ajustes y seguimiento con los equipos.

    *Sugerencia: comience con un umbral de 0.70 y ajústelo según el contexto.*
    """)

    # Chat section for querying similar recommendations
    st.header("Búsqueda de Recomendaciones")
    st.markdown("### Búsqueda")

    # Input for user query
    user_query = st.text_input("Búsqueda en recomendaciones:", value="¿Qué aspectos deben mejorarse sobre coordinación con partes interesadas?", key='user_query_tab8')

    # Search method selection
    search_method = st.radio("Método de búsqueda:", ["Por Similitud", "Por Coincidencia de Términos"], key='search_method_tab8')

    # Slider for similarity score threshold (only relevant for similarity search)
    if search_method == "Por Similitud":
        score_threshold = st.slider("Umbral de similitud:", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key='score_threshold_tab8')

    # Function to display results
    def display_results(results):
        if results:
            st.markdown("#### Recomendaciones similares")
            for i, result in enumerate(results):
                st.markdown(f"**Recomendación {i+1}:**")
                st.markdown(f"**Texto:** {result['recommendation']}")
                if "similarity" in result:
                    st.markdown(f"**Puntuación de similitud:** {result['similarity']:.2f}")
                st.markdown(f"**País:** {result['country']}")
                st.markdown(f"**Año:** {result['year']}")
                st.markdown(f"**Número de evaluación:** {result['eval_id']}")
                st.markdown("---")
        else:
            st.write("No se encontraron recomendaciones para la búsqueda.")

    # Button to search for recommendations
    if st.button("Buscar Recomendaciones", key='search_button_tab8'):
        if user_query:
            with st.spinner('Buscando recomendaciones...'):
                if search_method == "Por Similitud":
                    query_embedding = get_embedding_with_retry(user_query)
                    if query_embedding is not None:
                        results = find_similar_recommendations(query_embedding, index, doc_embeddings, structured_embeddings, score_threshold)
                        display_results(results)
                    else:
                        st.error("No se pudo generar el embedding para la consulta.")
                else:
                    results = find_recommendations_by_term_matching(user_query, doc_texts, structured_embeddings)
                    display_results(results)

    # Use the main recommendations dataset
    filtered_df = df.copy()

    # --- FILTER ROW WITHIN TAB ---
    st.markdown("### Filtros")
    filter_container = st.container()

    with filter_container:
        # Create 3 columns for filters
        col1, col2, col3 = st.columns(3)

        with col1:
            # Administrative unit filter
            office_options = ['All'] + sorted([str(x) for x in df['Recommendation_administrative_unit'].unique() if not pd.isna(x)])
            selected_offices_viz = st.multiselect('Unidad Administrativa:',
                                             options=office_options,
                                             default='All',
                                             key='unidad_administrativa_viz')

            # Country filter
            country_options = ['All'] + sorted([str(x) for x in df['Country(ies)'].unique() if not pd.isna(x)])
            selected_countries_viz = st.multiselect('País:',
                                              options=country_options,
                                              default='All',
                                              key='pais_viz')

        with col2:
            # Year filter with slider
            min_year = int(df['year'].min())
            max_year = max(int(df['year'].max()), 2025)
            selected_year_range_viz = st.slider('Rango de Años:',
                                          min_value=min_year,
                                          max_value=max_year,
                                          value=(min_year, max_year),
                                          key='rango_anos_viz')

            # Evaluation theme filter
            evaltheme_options = ['All'] + sorted([str(x) for x in df['Theme_cl'].unique() if not pd.isna(x)])
            selected_evaltheme_viz = st.multiselect('Tema (Evaluación):',
                                              options=evaltheme_options,
                                              default='All',
                                              key='tema_eval_viz')

        with col3:
            # Recommendation theme filter
            rectheme_options = ['All'] + sorted([str(x) for x in df['Recommendation_theme'].unique() if not pd.isna(x)])
            selected_rectheme_viz = st.multiselect('Tema (Recomendación):',
                                             options=rectheme_options,
                                             default='All',
                                             key='tema_recomendacion_viz')

            # Management response filter
            mgtres_options = ['All'] + sorted([str(x) for x in df['Management_response'].unique() if not pd.isna(x)])
            selected_mgtres_viz = st.multiselect('Respuesta de gerencia:',
                                           options=mgtres_options,
                                           default='All',
                                           key='respuesta_gerencia_viz')

    # Apply filters
    if 'All' not in selected_offices_viz and selected_offices_viz:
        filtered_df = filtered_df[filtered_df['Recommendation_administrative_unit'].isin(selected_offices_viz)]

    if 'All' not in selected_countries_viz and selected_countries_viz:
        filtered_df = filtered_df[filtered_df['Country(ies)'].isin(selected_countries_viz)]

    filtered_df = filtered_df[
        (filtered_df['year'] >= selected_year_range_viz[0]) &
        (filtered_df['year'] <= selected_year_range_viz[1])
    ]

    if 'All' not in selected_evaltheme_viz and selected_evaltheme_viz:
        filtered_df = filtered_df[filtered_df['Theme_cl'].isin(selected_evaltheme_viz)]

    if 'All' not in selected_rectheme_viz and selected_rectheme_viz:
        filtered_df = filtered_df[filtered_df['Recommendation_theme'].isin(selected_rectheme_viz)]

    if 'All' not in selected_mgtres_viz and selected_mgtres_viz:
        filtered_df = filtered_df[filtered_df['Management_response'].isin(selected_mgtres_viz)]

    # Remove duplicates for plotting
    filtered_df_unique = filtered_df.drop_duplicates(subset=['index_df'])

    # --- KPIs FOR VISUALIZATION TAB ---
    if not filtered_df_unique.empty:
        # Main KPIs
        total_kpi_labels = [
            "Total Recomendaciones",
            "Países",
            "Evaluaciones",
            "Años Cubiertos"
        ]
        total_kpi_values = [
            filtered_df_unique.shape[0],
            filtered_df_unique['Country(ies)'].nunique(),
            filtered_df_unique['Evaluation_number'].nunique() if 'Evaluation_number' in filtered_df_unique.columns else filtered_df_unique['Evaluation number'].nunique() if 'Evaluation number' in filtered_df_unique.columns else 0,
            filtered_df_unique['year'].nunique()
        ]

        total_cols = st.columns(4)
        total_kpi_html = [
            f"""
            <div style='text-align:center;'>
                <span style='font-size:1.2em; font-weight:700;'>{label}</span><br>
                <span style='font-size:2.3em; font-weight:700; color:#3498db;'>{value}</span>
            </div>
            """
            for label, value in zip(total_kpi_labels, total_kpi_values)
        ]

        for col, html in zip(total_cols, total_kpi_html):
            col.markdown(html, unsafe_allow_html=True)

        st.markdown("<hr style='border-top: 1px solid #e1e4e8;'>", unsafe_allow_html=True)

        # KPIs for management response statuses
        mgmt_labels = [
            ("Completadas", filtered_df_unique[filtered_df_unique['Management_response'] == 'Completed'].shape[0], '#27ae60'),
            ("Parcialmente Completadas", filtered_df_unique[filtered_df_unique['Management_response'] == 'Partially Completed'].shape[0], '#f7b731'),
            ("Acción no tomada aún", filtered_df_unique[filtered_df_unique['Management_response'] == 'Action not yet taken'].shape[0], '#fd9644'),
            ("Rechazadas", filtered_df_unique[filtered_df_unique['Management_response'] == 'Rejected'].shape[0], '#8854d0'),
            ("Acción no planificada", filtered_df_unique[filtered_df_unique['Management_response'] == 'No Action Planned'].shape[0], '#3867d6'),
            ("Sin respuesta", filtered_df_unique[filtered_df_unique['Management_response'] == 'Sin respuesta'].shape[0], '#eb3b5a'),
        ]

        st.markdown("<span style='font-size:1.6em; font-weight:700;'>Respuesta de Gerencia</span>", unsafe_allow_html=True)
        kpi_cols = st.columns(3)
        for i, (label, value, color) in enumerate(mgmt_labels):
            kpi_cols[i % 3].markdown(
                f"""
                <div style='text-align:center;'>
                    <span style='font-size:1.2em; font-weight:700;'>{label}</span><br>
                    <span style='font-size:2.3em; font-weight:700; color:{color};'>{value}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Display plots if data is available
        if not filtered_df.empty:
            country_counts = filtered_df_unique['Country(ies)'].value_counts()

            # Add CSS for dashboard styling
            st.markdown('<style>.dashboard-subtitle {font-size: 1.3rem; font-weight: 600; margin-bottom: 0.2em; margin-top: 1.2em; color: #3498db;}</style>', unsafe_allow_html=True)

            # Create two columns for charts
            row1_col1, row1_col2 = st.columns(2)

            with row1_col1:
                st.markdown('<div class="dashboard-subtitle">Número de Recomendaciones por País</div>', unsafe_allow_html=True)
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(
                    y=country_counts.index.tolist(),
                    x=country_counts.values.tolist(),
                    orientation='h',
                    text=country_counts.values.tolist(),
                    textposition='auto',
                    marker_color='#3498db',
                    hovertemplate='%{y}: %{x} recomendaciones'
                ))

                # Fixed height for alignment with year plot
                fixed_height = 500
                fig1.update_layout(
                    xaxis_title='Número de Recomendaciones',
                    yaxis_title='País',
                    margin=dict(t=10, l=10, r=10, b=40),
                    font=dict(size=22),
                    height=fixed_height,
                    plot_bgcolor='white',
                    showlegend=False
                )
                fig1.update_xaxes(showgrid=True, gridcolor='LightGray')
                fig1.update_yaxes(showgrid=False)
                st.plotly_chart(fig1, use_container_width=True)

            with row1_col2:
                st.markdown('<div class="dashboard-subtitle">Número de Recomendaciones por Año</div>', unsafe_allow_html=True)
                year_counts = filtered_df_unique['year'].value_counts().sort_index()
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=year_counts.index.astype(str).tolist(),
                    y=year_counts.values.tolist(),
                    text=year_counts.values.tolist(),
                    textposition='auto',
                    marker_color='#3498db',
                    hovertemplate='Año %{x}: %{y} recomendaciones',
                    textfont=dict(size=22)
                ))
                fig2.update_layout(
                    xaxis_title='Año',
                    yaxis_title='Número de Recomendaciones',
                    margin=dict(t=10, l=10, r=10, b=40),
                    font=dict(size=22),
                    height=500,
                    plot_bgcolor='white',
                    showlegend=False
                )
                fig2.update_xaxes(showgrid=True, gridcolor='LightGray', tickangle=45, title_font=dict(size=22), tickfont=dict(size=20))
                fig2.update_yaxes(showgrid=True, gridcolor='LightGray', title_font=dict(size=22), tickfont=dict(size=20))
                st.plotly_chart(fig2, use_container_width=True)

            # Dimension treemap
            st.markdown('<div class="dashboard-subtitle">Composición de Recomendaciones por Dimensión</div>', unsafe_allow_html=True)

            # Clean and prepare dimension data
            import numpy as np
            filtered_df['dimension'] = filtered_df['dimension'].astype(str).str.strip().str.lower().replace({
                'processes': 'process', 'process': 'process', 'nan': np.nan, 'none': np.nan, '': np.nan
            })
            filtered_df['dimension'] = filtered_df['dimension'].replace({'process': 'Process'})
            filtered_df = filtered_df[filtered_df['dimension'].notna()]

            filtered_df['rec_intervention_approach'] = filtered_df['rec_intervention_approach'].astype(str).str.strip().str.lower().replace({
                'processes': 'process', 'process': 'process', 'nan': np.nan, 'none': np.nan, '': np.nan
            })
            filtered_df['rec_intervention_approach'] = filtered_df['rec_intervention_approach'].replace({'process': 'Process'})
            filtered_df = filtered_df[filtered_df['rec_intervention_approach'].notna()]

            # Count recommendations by dimension
            dimension_counts = filtered_df.groupby('dimension').agg({
                'index_df': 'nunique'
            }).reset_index()

            # Calculate percentages and format text
            dimension_counts['percentage'] = dimension_counts['index_df'] / dimension_counts['index_df'].sum() * 100
            dimension_counts['text'] = dimension_counts.apply(
                lambda row: f"{row['dimension']}<br>Recomendaciones: {row['index_df']}<br>Porcentaje: {row['percentage']:.2f}%",
                axis=1
            )
            dimension_counts['font_size'] = dimension_counts['index_df'] / dimension_counts['index_df'].max() * 30 + 10  # Scale font size

            # Remove 'Sin Clasificar' and capitalize dimension labels
            dimension_counts = dimension_counts[dimension_counts['dimension'].str.lower() != 'sin clasificar']
            dimension_counts['dimension'] = dimension_counts['dimension'].astype(str).str.title()

            # Create treemap
            fig3 = px.treemap(
                dimension_counts,
                path=['dimension'],
                values='index_df',
                title='Composición de Recomendaciones por Dimensión',
                hover_data={'text': True, 'index_df': False, 'percentage': False},
                custom_data=['text']
            )
            fig3.update_traces(
                textinfo='label+value',
                hovertemplate='%{customdata[0]}',
                textfont_size=32
            )
            fig3.update_layout(
                margin=dict(t=50, l=25, r=25, b=25),
                width=900,
                height=500,
                title_font_size=32,
                font=dict(size=28),
                legend_font_size=28
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Subdimension treemap
            # Harmonize process/processes before plotting subdimensions
            filtered_df['dimension'] = filtered_df['dimension'].replace({'processes': 'Process', 'process': 'Process', 'Process': 'Process'})

            # Remove 'Sin Clasificar' from both dimension and subdimension
            filtered_df = filtered_df[filtered_df['dimension'].str.lower() != 'sin clasificar']
            filtered_df = filtered_df[filtered_df['subdim'].str.lower() != 'sin clasificar']

            # Capitalize dimension and subdimension labels
            filtered_df['dimension'] = filtered_df['dimension'].astype(str).str.title()
            filtered_df['subdim'] = filtered_df['subdim'].astype(str).str.title()

            # Count by subdimension
            subdimension_counts = filtered_df.groupby(['dimension', 'subdim']).agg({
                'index_df': 'nunique'
            }).reset_index()

            # Calculate percentages and format text
            subdimension_counts['percentage'] = subdimension_counts['index_df'] / subdimension_counts['index_df'].sum() * 100
            subdimension_counts['text'] = subdimension_counts.apply(
                lambda row: f"{row['subdim']}<br>Recomendaciones: {row['index_df']}<br>Porcentaje: {row['percentage']:.2f}%",
                axis=1
            )
            subdimension_counts['font_size'] = subdimension_counts['index_df'] / subdimension_counts['index_df'].max() * 30 + 10

            # Create treemap
            fig4 = px.treemap(
                subdimension_counts,
                path=['dimension', 'subdim'],
                values='index_df',
                title='Composición de Recomendaciones por Subdimensión',
                hover_data={'text': True, 'index_df': False, 'percentage': False},
                custom_data=['text']
            )
            fig4.update_traces(
                textinfo='label+value',
                hovertemplate='%{customdata[0]}',
                textfont_size=32
            )
            fig4.update_layout(
                margin=dict(t=50, l=25, r=25, b=25),
                width=900,
                height=500,
                title_font_size=32,
                font=dict(size=28),
                legend_font_size=28
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Add the advanced visualization section
            add_advanced_visualization_section(filtered_df, tab_id="tab8")
        else:
            st.warning("No hay datos disponibles para los filtros seleccionados.")
    else:
        st.warning("No hay datos disponibles para los filtros seleccionados.")
