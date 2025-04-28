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
# Initialize OpenAI - use only the older API version
import openai
openai.api_key = openai_api_key

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
        results = {}
        for criterion, descriptions in rubric_elements.items():
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
                results[criterion] = {
                    'analysis': analysis,
                    'context': context_text,
                    'score': analysis.get('score', 0),
                    'confidence': analysis.get('confidence', 0),
                    'top_paragraphs': [{'text': p[0]['text'], 'similarity': p[1]} for p in top_paragraphs[:3]]
                }
            except Exception as e:
                results[criterion] = {
                    'analysis': {'error': str(e)},
                    'context': context_text,
                    'score': 0,
                    'confidence': 0
                }
        return results
    def analyze_criterion(self, criterion: str, context: str, descriptions: list) -> dict:
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
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert document evaluator that provides detailed analysis and scoring based on specific criteria."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except Exception as e:
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

    # --- Section Selection ---
    if 'selected_sections_for_eval' not in st.session_state:
        st.session_state.selected_sections_for_eval = []
    if 'evaluation_sections_confirmed' not in st.session_state:
        st.session_state.evaluation_sections_confirmed = False

    st.markdown("#### 1. Selección de Secciones para Evaluación")
    main_sections = []
    for level, headings in sorted(toc_hierarchy.items()):
        if headings and not main_sections:
            main_sections = headings
    valid_selected_sections = [s for s in st.session_state.selected_sections_for_eval if s in main_sections]
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
            st.session_state.selected_sections_for_eval = selected_sections
            st.session_state.evaluation_sections_confirmed = True
            st.success(f"Secciones confirmadas para evaluación: {', '.join(selected_sections)}")
    if st.session_state.selected_sections_for_eval:
        st.info(f"Secciones actualmente seleccionadas para evaluación: {', '.join(st.session_state.selected_sections_for_eval)}")

    # --- Rubric Selection & Evaluation ---
    if st.session_state.evaluation_sections_confirmed:
        st.markdown("#### 2. Selección de Rúbrica y Criterios")
        rubric_type = st.selectbox(
            "Seleccione tipo de rúbrica para evaluación:",
            ["Participación (Engagement)", "Desempeño (Performance)"],
            index=0
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
                if 'selected_criteria' not in st.session_state:
                    st.session_state.selected_criteria = {}
                all_criteria_in_category = st.checkbox(f"Seleccionar todos los criterios en '{selected_category}'")
                selected_criteria_ids = []
                for _, criterion_row in category_criteria.iterrows():
                    criterion_id = criterion_row[criteria_col]
                    criterion_name = criterion_row[short_col] if short_col and short_col in criterion_row else criterion_id
                    if criterion_id not in st.session_state.selected_criteria:
                        st.session_state.selected_criteria[criterion_id] = False
                    if all_criteria_in_category:
                        st.session_state.selected_criteria[criterion_id] = True
                    is_selected = st.checkbox(
                        criterion_name,
                        value=st.session_state.selected_criteria[criterion_id],
                        key=f"criterion_{criterion_id}"
                    )
                    st.session_state.selected_criteria[criterion_id] = is_selected
                    if is_selected:
                        selected_criteria_ids.append(criterion_id)
        with st.expander("Opciones avanzadas de selección de criterios"):
            select_all_criteria = st.checkbox("Seleccionar TODOS los criterios de todas las categorías")
            if select_all_criteria:
                for _, row in rubric_df.iterrows():
                    criterion_id = row[criteria_col]
                    st.session_state.selected_criteria[criterion_id] = True
                st.success("Todos los criterios han sido seleccionados.")
            if st.button("Limpiar todas las selecciones"):
                for criterion_id in st.session_state.selected_criteria:
                    st.session_state.selected_criteria[criterion_id] = False
                st.success("Todas las selecciones han sido limpiadas.")
        if st.button("Ver Detalles de Criterios Seleccionados"):
            selected_any = any(st.session_state.selected_criteria.values())
            if not selected_any:
                st.warning("Por favor seleccione al menos un criterio para ver sus detalles.")
            else:
                st.markdown("##### Detalles de Criterios Seleccionados")
                selected_criteria_df = rubric_df[rubric_df[criteria_col].isin(
                    [cid for cid, selected in st.session_state.selected_criteria.items() if selected]
                )]
                for _, criterion_row in selected_criteria_df.iterrows():
                    criterion_id = criterion_row[criteria_col]
                    criterion_name = criterion_row[short_col] if short_col and short_col in criterion_row else criterion_id
                    with st.expander(f"{criterion_name}", expanded=True):
                        levels_df = rubric_to_levels_df(criterion_row, criteria_col)
                        st.table(levels_df)
        if st.button("Iniciar Evaluación de Criterios Seleccionados (RAG)"):
            selected_any = any(st.session_state.selected_criteria.values())
            if not selected_any:
                st.warning("Por favor seleccione al menos un criterio para evaluar.")
            else:
                st.markdown("#### 3. Evaluación de Criterios (RAG)")
                selected_criteria_ids = [cid for cid, selected in st.session_state.selected_criteria.items() if selected]
                filtered_df = exploded_df[exploded_df['header_1'].isin(st.session_state.selected_sections_for_eval)].copy()
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
                        result = store.score_rubric_directly(rubric_dict_single, df=section_df, top_n_paragraphs=10)
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
{{ ... }}
                                if create_filtered:
                                    filtered_sections = {}
                                    progress_bar = st.progress(0, text="Progreso de filtrado de secciones")
                                    filtered_sections = {}
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
{{ ... }}
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
                    # Call the rubric evaluation function with the document sections
                    if 'add_rubric_evaluation_section' in globals():
                        add_rubric_evaluation_section(sections_content, toc, toc_hierarchy)
                    else:
                        st.info("La función de evaluación por rúbrica no está disponible. Por favor actualice el código con la implementación de esta función.")
                    
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                
    else:
        st.info("Por favor suba un archivo DOCX para comenzar.")
        
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
            textfont=dict(size=14),
            hoverinfo='name+y'
        ))
        
        # Update cumulative for next bar
        cumulative['cum'] += var_by_year_pct[category]
    
    # Update layout
    fig.update_layout(
        title=title,
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

# Function to add advanced visualization section to the Streamlit app
def add_advanced_visualization_section(filtered_df):
    """
    Adds an advanced visualization section to the Streamlit app.
    """
    st.markdown("#### Visualizaciones Avanzadas")
    
    # Create tabs for different visualization types
    viz_tabs = st.tabs([
        "Evolución de Puntuaciones", 
        "Composición por Variable", 
        # "Composición de Etiquetas",
        "Clasificación de Dificultad"
    ])
    
    # Tab 1: Score Evolution
    with viz_tabs[0]:
        st.markdown("##### Evolución de Puntuaciones Promedio por Año")
        score_fig = plot_score_evolution(filtered_df)
        if score_fig:
            st.plotly_chart(score_fig, use_container_width=True)
    
    # Tab 2: Variable Composition
    with viz_tabs[1]:
        st.markdown("##### Composición por Variable")
        
        # List of variables that can be visualized
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
            
            # Filter to only variables that exist in the dataframe
            var_options = {var_mapping.get(var, var): var for var in available_vars if var in var_mapping}
            
            if var_options:
                selected_var_label = st.selectbox(
                    "Seleccione una variable para visualizar:", 
                    options=list(var_options.keys())
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
    
    # # Tab 3: Tag Composition
    # with viz_tabs[2]:
    #     st.markdown("##### Evolución de la Composición de Etiquetas")
        
    #     if 'tags' in filtered_df.columns:
    #         top_n = st.slider("Número de etiquetas principales a mostrar:", min_value=3, max_value=15, value=8)
    #         tag_fig = create_tag_composition_plot(filtered_df, top_n)
    #         if tag_fig:
    #             st.plotly_chart(tag_fig, use_container_width=True)
    #     else:
    #         st.warning("No se encontraron datos de etiquetas en los datos filtrados.")
    
    # Tab 4: Difficulty Classification
    with viz_tabs[2]:
        st.markdown("##### Clasificación de Dificultad de Rechazo")
        
        if 'clean_tags' in filtered_df.columns:
            top_n = st.slider("Número de clasificaciones principales a mostrar:", min_value=3, max_value=15, value=8, key='diff_class_slider')
            diff_fig = create_difficulty_classification_plot(filtered_df, top_n)
            if diff_fig:
                st.plotly_chart(diff_fig, use_container_width=True)
        else:
            st.warning("No se encontraron datos de clasificación de dificultad en los datos filtrados.")

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

def process_text_analysis(combined_text, map_template, combine_template_prefix, user_template_part):
    """
    Process text analysis in chunks and combine results.
    
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
    
    for chunk in text_chunks:
        summary = summarize_text(chunk, map_template)
        if summary:
            chunk_summaries.append(summary)
            time.sleep(1)  # Rate limiting
    
    if chunk_summaries:
        combined_summaries = " ".join(chunk_summaries)
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
            model="gpt-4o-mini",  # or whatever model you're using
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
st.markdown("<h3 style='text-align: center;'>Análisis Automatizado de Recomendaciones, BBPP, LLAA e Informes de Evaluación</h3>", unsafe_allow_html=True)

# Check for API key before running the app
if not openai_api_key:
    st.warning("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in Streamlit Cloud.")
    st.info("For local development, you can use a .env file or set the environment variable.")
    # Continue with limited functionality or show instructions on setup

# Initialize session state if not already set
if 'similar_df' not in st.session_state:
    st.session_state['similar_df'] = pd.DataFrame()

# Initialize data and embeddings - wrap in try/except for better error handling
try:
    # Use the extended data loading function that includes the new visualizations data
    df, df_raw = load_extended_data()
    doc_embeddings, structured_embeddings, index = load_embeddings()
    doc_texts = df_raw['Recommendation_description'].tolist()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Tabs
# Modify the tabs in oli_v5_deploy.py
tab1, tab2, tab3 = st.tabs(["Análisis de Textos y Recomendaciones Similares", 
                           "Búsqueda de Recomendaciones",
                           "Subir y Procesar Documentos"])

# Tab 1: Filters, Text Analysis and Similar Recommendations
with tab1:
    st.header("Análisis de Textos y Recomendaciones Similares")

    # Sidebar for filters
    st.sidebar.title('Criterios de Búsqueda')

    with st.sidebar.expander("Oficina Regional", expanded=False):
        office_options = ['All'] + list(df['Recommendation_administrative_unit'].unique())
        selected_offices = st.multiselect('Oficina Regional', options=office_options, default='All')
    
    with st.sidebar.expander("País", expanded=False):
        country_options = ['All'] + list(df['Country(ies)'].unique())
        selected_countries = st.multiselect('País', options=country_options, default='All')

    with st.sidebar.expander("Año", expanded=False):
        year_options = ['All'] + list(df['year'].unique())
        selected_years = st.multiselect('Año', options=year_options, default='All')
        
    # Dimension filter
    with st.sidebar.expander("Dimensión", expanded=False):
        dimension_options = ['All'] + list(df['dimension'].unique())
        selected_dimensions = st.multiselect('Dimensión', options=dimension_options, default='All')

    # Subdimension filter
    if 'All' in selected_dimensions or not selected_dimensions:
        subdimension_options = ['All'] + list(df['subdim'].unique())
    else:
        subdimension_options = ['All'] + list(df[df['dimension'].isin(selected_dimensions)]['subdim'].unique())

    with st.sidebar.expander("Subdimensión", expanded=False):
        selected_subdimensions = st.multiselect('Subdimensión', options=subdimension_options, default='All')

    # Evaluation theme filter
    if 'All' in selected_dimensions or 'All' in selected_subdimensions or not selected_dimensions or not selected_subdimensions:
        evaltheme_options = ['All'] + list(df['Theme_cl'].unique())
    else:
        evaltheme_options = ['All'] + list(df[(df['dimension'].isin(selected_dimensions)) & (df['subdim'].isin(selected_subdimensions))]['Theme_cl'].unique())

    with st.sidebar.expander("Tema (Evaluación)", expanded=False):
        selected_evaltheme = st.multiselect('Tema (Evaluación)', options=evaltheme_options, default='All')

    # Recommendation theme filter
    if 'All' in selected_dimensions or 'All' in selected_subdimensions or 'All' in selected_evaltheme or not selected_dimensions or not selected_subdimensions or not selected_evaltheme:
        rectheme_options = ['All'] + list(df['Recommendation_theme'].unique())
    else:
        rectheme_options = ['All'] + list(df[(df['dimension'].isin(selected_dimensions)) & 
                                            (df['subdim'].isin(selected_subdimensions)) & 
                                            (df['Theme_cl'].isin(selected_evaltheme))]['Recommendation_theme'].unique())

    with st.sidebar.expander("Tema (Recomendación)", expanded=False):
        selected_rectheme = st.multiselect('Tema (Recomendación)', options=rectheme_options, default='All')

    # Management response filter
    if 'All' in selected_dimensions or 'All' in selected_subdimensions or 'All' in selected_evaltheme or 'All' in selected_rectheme or not selected_dimensions or not selected_subdimensions or not selected_evaltheme or not selected_rectheme:
        mgtres_options = ['All'] + list(df['Management_response'].unique())
    else:
        mgtres_options = ['All'] + list(df[(df['dimension'].isin(selected_dimensions)) & 
                                        (df['subdim'].isin(selected_subdimensions)) & 
                                        (df['Theme_cl'].isin(selected_evaltheme)) & 
                                        (df['Recommendation_theme'].isin(selected_rectheme))]['Management_response'].unique())

    with st.sidebar.expander("Respuesta de gerencia", expanded=False):
        selected_mgtres = st.multiselect('Respuesta de gerencia', options=mgtres_options, default='All')

    # Text source selection before analysis button
    with st.sidebar.expander("Fuentes de Texto", expanded=False):
        analyze_recommendations = st.checkbox('Recomendaciones', value=True)
        analyze_lessons = st.checkbox('Lecciones Aprendidas', value=False) 
        analyze_practices = st.checkbox('Buenas Prácticas', value=False)
        analyze_plans = st.checkbox('Planes de Acción', value=False)
        select_all = st.checkbox('Seleccionar Todas las Fuentes')
        if select_all:
            analyze_recommendations = analyze_lessons = analyze_practices = analyze_plans = True

    # Filter dataframe based on user selection
    filtered_df = df.copy()

    if 'All' not in selected_offices:
        filtered_df = filtered_df[filtered_df['Recommendation_administrative_unit'].isin(selected_offices)]
    if 'All' not in selected_countries:
        filtered_df = filtered_df[filtered_df['Country(ies)'].isin(selected_countries)]
    if 'All' not in selected_years:
        filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
    if 'All' not in selected_dimensions:
        filtered_df = filtered_df[filtered_df['dimension'].isin(selected_dimensions)]
    if 'All' not in selected_subdimensions:
        filtered_df = filtered_df[filtered_df['subdim'].isin(selected_subdimensions)]
    if 'All' not in selected_evaltheme:
        filtered_df = filtered_df[filtered_df['Theme_cl'].isin(selected_evaltheme)]
    if 'All' not in selected_rectheme:
        filtered_df = filtered_df[filtered_df['Recommendation_theme'].isin(selected_rectheme)]
    if 'All' not in selected_mgtres:
        filtered_df = filtered_df[filtered_df['Management_response'].isin(selected_mgtres)]

    # Extract unique texts
    unique_texts = filtered_df['Recommendation_description'].unique()
    unique_texts_str = [str(text) for text in unique_texts]  # Convert each element to string

    # Create summary table
    filtered_df_unique = filtered_df.drop_duplicates(subset=['index_df'])
    summary_data = {
        'Métrica': [
            'Número de Recomendaciones',
            'Países',
            'Años',
            'Número de Evaluaciones',
            'Completadas',
            'Parcialmente Completadas',
            'Acción no tomada aún',
            'Rechazadas',
            'Acción no planificada',
            'Sin respuesta'
        ],
        'Conteo': [
            len(unique_texts),
            filtered_df['Country(ies)'].nunique(),
            filtered_df['year'].nunique(),
            filtered_df['Evaluation_number'].nunique(),
            filtered_df_unique[filtered_df_unique['Management_response'] == 'Completed'].shape[0],
            filtered_df_unique[filtered_df_unique['Management_response'] == 'Partially Completed'].shape[0],
            filtered_df_unique[filtered_df_unique['Management_response'] == 'Action not yet taken'].shape[0],
            filtered_df_unique[filtered_df_unique['Management_response'] == 'Rejected'].shape[0],
            filtered_df_unique[filtered_df_unique['Management_response'] == 'No Action Planned'].shape[0],
            filtered_df_unique[filtered_df_unique['Management_response'] == 'Sin respuesta'].shape[0]
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    # Display summary table with better formatting
    st.markdown("#### Información General")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Recurrencia")
        st.table(summary_df.head(4).set_index('Métrica').style.hide(axis="index"))

    with col2:
        st.markdown("##### Respuesta de Gerencia")
        st.table(summary_df.tail(6).set_index('Métrica').style.hide(axis="index"))

    # Display plots if data is available
    if not filtered_df.empty:
        country_counts = filtered_df_unique['Country(ies)'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(18, 17))
        country_counts.plot(kind='barh', ax=ax1)
        ax1.set_xlabel('Número de Recomendaciones', fontsize=18)
        ax1.set_ylabel('País', fontsize=20)
        ax1.set_title('Número de Recomendaciones por País', fontsize=18)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        for i in ax1.patches:
            ax1.text(i.get_width() + 0.3, i.get_y() + 0.5, 
                    str(round((i.get_width()), 2)), fontsize=14, color='dimgrey')

        # Treemap: Recommendations by Dimension
        dimension_counts = filtered_df.groupby('dimension').agg({
        'index_df': 'nunique'
        }).reset_index()
        dimension_counts['percentage'] = dimension_counts['index_df'] / dimension_counts['index_df'].sum() * 100
        dimension_counts['text'] = dimension_counts.apply(lambda row: f"{row['dimension']}<br>Recomendaciones: {row['index_df']}<br>Porcentaje: {row['percentage']:.2f}%", axis=1)
        dimension_counts['font_size'] = dimension_counts['index_df'] / dimension_counts['index_df'].max() * 30 + 10  # Scale font size

        fig3 = px.treemap(dimension_counts, path=['dimension'], values='index_df',
                        title='Composición de Recomendaciones por Dimensión',
                        hover_data={'text': True, 'index_df': False, 'percentage': False},
                        custom_data=['text'])

        fig3.update_traces(textinfo='label+value', hovertemplate='%{customdata[0]}')
        fig3.update_layout(margin=dict(t=50, l=25, r=25, b=25), width=800, height=400)

        # Plot for recommendations by year
        year_counts = filtered_df_unique['year'].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(18, 14))
        year_counts.plot(kind='bar', ax=ax2)
        ax2.set_title('Número de Recomendaciones por Año', fontsize = 20)
        ax2.set_xlabel('Año', fontsize = 20)
        ax2.set_ylabel('Número de Recomendaciones')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        # Treemap: Recommendations by Subdimension
        subdimension_counts = filtered_df.groupby(['dimension', 'subdim']).agg({
        'index_df': 'nunique'
        }).reset_index()
        subdimension_counts['percentage'] = subdimension_counts['index_df'] / subdimension_counts['index_df'].sum() * 100
        subdimension_counts['text'] = subdimension_counts.apply(lambda row: f"{row['subdim']}<br>Recomendaciones: {row['index_df']}<br>Porcentaje: {row['percentage']:.2f}%", axis=1)
        subdimension_counts['font_size'] = subdimension_counts['index_df'] / subdimension_counts['index_df'].max() * 30 + 10  # Scale font size

        fig4 = px.treemap(subdimension_counts, path=['dimension', 'subdim'], values='index_df',
                        title='Composición de Recomendaciones por Subdimensión',
                        hover_data={'text': True, 'index_df': False, 'percentage': False},
                        custom_data=['text'])

        fig4.update_traces(textinfo='label+value', hovertemplate='%{customdata[0]}')
        fig4.update_layout(margin=dict(t=50, l=25, r=25, b=25), width=800, height=400)

        # Display plots side by side
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig1)

        with col2:
            st.pyplot(fig2)

        st.plotly_chart(fig3)
        st.plotly_chart(fig4)
        
        # Add the advanced visualization section
        with st.expander("Visualizaciones Avanzadas", expanded=False):
            add_advanced_visualization_section(filtered_df)
    else:
        st.write("No data available for the selected filters.")

    # Add a text area for the user to input the custom combine template part
    user_template_part = st.sidebar.text_area("Instrucción de análisis", 
                                          value="""Produce un breve resumen en español del conjunto completo. Después, incluye una lista con viñetas que resuma las acciones recomendadas y los actores específicos a quienes están dirigidas, así como otra lista con viñetas para los temas principales y recurrentes. Este formato debe aclarar qué se propone y a quién está dirigida cada recomendación. Adicionalmente, genera una lista con viñetas de los puntos más importantes a considerar cuando se planee abordar estas recomendaciones en el futuro. Por favor, refiérete al texto como un conjunto de recomendaciones, no como un documento o texto.""")

    combine_template_prefix = "The following is a set of summaries:\n{text}\n"

    # Define the map prompt for initial summarization of chunks
    map_template = """Summarize the following text: {text}
    Helpful Answer:"""

    # Analysis button logic
    if st.button('Analizar Textos'):
        selections = {
            'recommendations': analyze_recommendations,
            'lessons': analyze_lessons,
            'practices': analyze_practices,
            'plans': analyze_plans
        }
        
        combined_text = build_combined_text(filtered_df, selections)
        
        if combined_text:
            with st.spinner('Analizando textos...'):
                result = process_text_analysis(combined_text, map_template, combine_template_prefix, user_template_part)
                if result:
                    st.markdown(f"<div style='text-align: justify;'>{result}</div>", unsafe_allow_html=True)
                else:
                    st.error("No se pudo generar el análisis.")
        else:
            st.warning("Por favor seleccione al menos una fuente de texto para analizar.")

    # Button to download the filtered dataframe as Excel file
    if st.button('Descargar Datos Filtrados'):
        filtered_data = to_excel(filtered_df)
        st.download_button(label='📥 Descargar Excel', data=filtered_data, file_name='filtered_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Tab 2: Search
with tab2:
    st.header("Búsqueda de Recomendaciones")
    
    # Chat section for querying similar recommendations
    st.markdown("### Búsqueda")

    # Input for user query
    user_query = st.text_input("Pregunte sobre las recomendaciones:", value="¿Qué aspectos deben mejorarse sobre coordinación con partes interesadas?")

    # Search method selection
    search_method = st.radio("Método de búsqueda:", ["Por Similitud", "Por Coincidencia de Términos"])

    # Slider for similarity score threshold (only relevant for similarity search)
    if search_method == "Por Similitud":
        score_threshold = st.slider("Umbral de similitud:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

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
    if st.button("Buscar Recomendaciones"):
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
        else:
            st.warning("Introduzca una consulta para buscar recomendaciones.")
            
#==============================================================================================
# Tab 3: Document Upload and Parsing
with tab3:
    add_document_upload_tab()
