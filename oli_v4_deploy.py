import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from matplotlib import pyplot as plt
import seaborn as sns
from io import BytesIO
import torch
import time
import plotly.express as px
import plotly.graph_objects as go

# Use environment variables for API keys
# For local development - use .env file or set environment variables
# For Streamlit Cloud - set these in the app settings
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI - use only the older API version
import openai
openai.api_key = openai_api_key

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

# Initialize data and embeddings - wrap in try/except for better error handling
try:
    df, df_raw = load_data()
    doc_embeddings, structured_embeddings, index = load_embeddings()
    doc_texts = df_raw['Recommendation_description'].tolist()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Function to get embeddings from OpenAI
def get_embedding_with_retry(text, model='text-embedding-3-large', max_retries=3, delay=1):
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
        
    for attempt in range(max_retries):
        try:
            # Use older OpenAI SDK
            response = openai.Embedding.create(input=text, model=model)
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(delay)
    return None

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
        
        # Sort recommendations by length or relevance if needed
        matched_recommendations = sorted(matched_recommendations, key=lambda x: len(str(x["recommendation"])))
        
        return matched_recommendations[:top_n]
    except Exception as e:
        st.error(f"Error in term matching: {str(e)}")
        return []

def summarize_text(text, prompt_template):
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
        
    prompt = prompt_template.format(text=text)
    try:
        # Use only the older OpenAI SDK (<1.0.0)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes and analyzes texts."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None

def split_text(text, max_length=1500):
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

def process_text_analysis(combined_text, map_template, combine_template_prefix, user_template_part):
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

def build_combined_text(df, selections):
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

# Function to convert dataframe to Excel for download
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Filtered Data')
    processed_data = output.getvalue()
    return processed_data

# Set page config
st.markdown("<h3 style='text-align: center;'>Oli: Análisis Automatizado de Recomendaciones</h3>", unsafe_allow_html=True)

# Check for API key before running the app
if not openai_api_key:
    st.warning("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in Streamlit Cloud.")
    st.info("For local development, you can use a .env file or set the environment variable.")
    # Continue with limited functionality or show instructions on setup

# Initialize session state if not already set
if 'similar_df' not in st.session_state:
    st.session_state['similar_df'] = pd.DataFrame()

# Tabs
tab1, tab2 = st.tabs(["Análisis de Textos y Recomendaciones Similares", "Búsqueda de Recomendaciones"])

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
