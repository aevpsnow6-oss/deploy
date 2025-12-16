# Documentación Técnica: Evolución de Análisis Individual a Análisis Jerárquico
## De Versión Simple (V4) a Análisis Multinivel (V6)

**Fecha:** Diciembre 2025  
**Objetivo:** Transformación del sistema de análisis de preguntas individuales a un análisis jerárquico de tres niveles

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura V4: Sistema Original](#arquitectura-v4-sistema-original)
3. [Arquitectura V6: Sistema Mejorado](#arquitectura-v6-sistema-mejorado)
4. [Flujo de Datos Comparativo](#flujo-de-datos-comparativo)
5. [Cambios Funcionales](#cambios-funcionales)
6. [Implementación de la Jerarquía](#implementación-de-la-jerarquía)
7. [Optimizaciones y Eficiencia](#optimizaciones-y-eficiencia)
8. [Salida de Datos](#salida-de-datos)
9. [Consideraciones de Rendimiento](#consideraciones-de-rendimiento)

---

## Resumen Ejecutivo

### Versión 4 (Original)
- **Análisis por pregunta individual**
- Una llamada LLM por pregunta
- Salida Excel simple con columnas planas
- No hay síntesis de respuestas relacionadas

### Versión 6 (Mejorada)
- **Análisis jerárquico de tres niveles**
  1. Preguntas individuales (análisis base)
  2. Subsecciones (síntesis de preguntas similares)
  3. Secciones (síntesis de subsecciones)
- Tres capas de síntesis LLM
- Salida Excel estructurada con celdas fusionadas
- Síntesis agregada de respuestas relacionadas
- Ordenamiento ascendente completo

---

## Arquitectura V4: Sistema Original

### Flujo de Procesamiento

```
PREGUNTA → ANÁLISIS INDIVIDUAL → RESULTADO
  ↓              ↓                  ↓
1.1         LLM Call            {Respuesta: Yes,
1.2         (1500 tokens)        Razonamiento: "..."}
2.1
...
```

### Estructura de Datos V4

```python
# DataFrame de salida
resultado_pregunta = {
    'Pregunta': '1.1 ¿Está documentado...?',
    'Respuesta': 'Yes',
    'Razonamiento': 'Se encontró evidencia en págs. 5-8',
    'Evidencia': '[documento_text]',
    'Status': 'Success'
}
```

### Características Clave V4

1. **Análisis Atomizado**
   - Cada pregunta se analiza de forma independiente
   - No hay relación entre respuestas de preguntas relacionadas

2. **Llamadas LLM**
   ```python
   # Un LLM call por pregunta
   response = client.chat.completions.create(
       model="gpt-5-mini",
       messages=[
           {"role": "system", "content": "Eres un analista..."},
           {"role": "user", "content": f"Pregunta: {question}..."}
       ],
       max_tokens=1500
   )
   ```

3. **Salida Simple**
   - Tabla Excel con columnas: Pregunta | Respuesta | Razonamiento | Evidencia | Status
   - Sin estructura jerárquica
   - Sin síntesis agregada

4. **Limitaciones**
   - No captura patrones entre preguntas relacionadas (1.1, 1.2, 1.3)
   - No hay contexto de subsección
   - Análisis repetitivo de temas similares
   - Difícil identificar tendencias por área

---

## Arquitectura V6: Sistema Mejorado

### Flujo de Procesamiento (Tres Niveles)

```
PREGUNTAS (1.1, 1.2, 1.3, ...)
     ↓
NIVEL 1: ANÁLISIS INDIVIDUAL (Paralelo, ThreadPoolExecutor, MAX_WORKERS=48)
     ↓
    {Pregunta: "1.1 ¿...", Respuesta: "Yes", Razonamiento: "..."}
     ↓
AGRUPACIÓN POR SUBSECCIÓN (1.1, 1.2, 1.3, ...)
     ↓
NIVEL 2: SÍNTESIS POR SUBSECCIÓN (Secuencial, 1-2 párrafos)
     ↓
    {"1.1": "Análisis de subsección 1.1...", "1.2": "Análisis de subsección 1.2..."}
     ↓
AGRUPACIÓN POR SECCIÓN (1.*, 2.*, 3.*, ...)
     ↓
NIVEL 3: SÍNTESIS POR SECCIÓN (Secuencial, 2-3 párrafos)
     ↓
    {1: "Análisis de sección 1...", 2: "Análisis de sección 2..."}
     ↓
SALIDA: Excel con tres niveles + Ordenamiento Ascendente
```

### Estructura de Datos V6

```python
# Nivel 1: Análisis Individual (igual que V4)
resultado_pregunta = {
    'Pregunta': '1.1 ¿Está documentado...?',
    '_section_num': 1,
    '_subsection': '1.1',
    '_sort_key': (1, 1),
    'Respuesta': 'Yes',
    'Razonamiento': 'Se encontró evidencia en págs. 5-8',
    'Evidencia': '[documento_text]',
    'Status': 'Success'
}

# Nivel 2: Síntesis por Subsección
subsection_analyses = {
    '1.1': "En el análisis de la subsección 1.1, que comprende preguntas sobre documentación...",
    '1.2': "Respecto a la subsección 1.2, los resultados muestran...",
    '1.3': "La subsección 1.3 presenta indicadores..."
}

# Nivel 3: Síntesis por Sección
section_analyses = {
    1: "La Sección 1 demuestra un nivel de cumplimiento...",
    2: "En cuanto a la Sección 2, se observa que..."
}
```

### Características Clave V6

1. **Análisis Jerárquico**
   - Tres niveles de procesamiento
   - Cada nivel agrega valor a partir del anterior
   - Contexto multinivel disponible durante síntesis

2. **Funciones de Extracción Numérica**
   ```python
   def extract_section_number(question_text):
       """Extrae número de sección: "1.1 ¿Está..." → 1"""
       match = re.search(r'(\d+)\.', question_text)
       return int(match.group(1)) if match else None
   
   def extract_subsection_number(question_text):
       """Extrae ID de subsección: "1.1 ¿Está..." → "1.1" """
       match = re.search(r'(\d+\.\d+)', question_text)
       return match.group(1) if match else None
   
   def parse_subsection_for_sorting(subsection_str):
       """Convierte "1.1" a (1, 1) para ordenamiento numérico correcto
          Garantiza: "1.2" < "1.10" < "1.20" (no string sorting)"""
       parts = subsection_str.split('.')
       return tuple(int(p) for p in parts)
   ```

3. **Síntesis por Subsección**
   ```python
   def synthesize_subsection_analysis(subsection_id, subsection_questions_df):
       """
       Genera síntesis de subsección de 1-2 párrafos
       
       Entrada:
       - subsection_id: "1.1"
       - subsection_questions_df: DataFrame con respuestas de 1.1.1, 1.1.2, 1.1.3, etc.
       
       Proceso:
       1. Recopila todas las respuestas (Respuesta, Razonamiento)
       2. Construye prompt contextual que incluye todas las preguntas
       3. LLM sintetiza patrones comunes, inconsistencias, tendencias
       
       Salida:
       - Texto de análisis (1-2 párrafos, máx 1500 tokens)
       """
       prompt = f"""
       Dadas las respuestas a las siguientes preguntas relacionadas de la subsección {subsection_id}:
       
       {subsection_questions_summary}
       
       Genera un análisis conciso (1-2 párrafos) que:
       1. Resume los patrones observados
       2. Identifica inconsistencias si las hay
       3. Destaca áreas de fortaleza y debilidad
       """
       
       response = client.chat.completions.create(
           model="gpt-5-mini",
           messages=[{"role": "user", "content": prompt}],
           max_tokens=1500,
           reasoning_effort="minimal"
       )
   ```

4. **Síntesis por Sección (MODIFICADA en V6)**
   ```python
   def synthesize_section_analysis(section_num, subsection_analyses_dict):
       """
       CAMBIO CLAVE: Ahora usa análisis de subsecciones, NO respuestas individuales
       
       Entrada V4 (Original):
       - section_num: 1
       - section_questions_df: Todas las preguntas de la sección 1
       - document_text: Texto completo del documento
       
       Entrada V6 (Mejorada):
       - section_num: 1
       - subsection_analyses_dict: {
           "1.1": "Análisis de subsección 1.1...",
           "1.2": "Análisis de subsección 1.2...",
           "1.3": "Análisis de subsección 1.3..."
         }
       
       Ventaja: Síntesis de síntesis proporciona panorama más completo
       """
       
       prompt = f"""
       Basado en los siguientes análisis de subsecciones:
       
       {subsection_analyses_summary}
       
       Genera un análisis ejecutivo de la Sección {section_num} (2-3 párrafos) que:
       1. Integra hallazgos de todas las subsecciones
       2. Identifica tendencias estratégicas
       3. Proporciona recomendaciones de alto nivel
       """
   ```

---

## Flujo de Datos Comparativo

### V4: Flujo Lineal Simple

```
┌─────────────────────────────┐
│ Carga de Preguntas (APPRAISAL)
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Análisis Paralelo Individual │ (ThreadPoolExecutor, 48 workers)
│ - Pregunta 1.1: LLM Call 1  │
│ - Pregunta 1.2: LLM Call 2  │
│ - Pregunta 2.1: LLM Call 3  │
│ ... (N-1) llamadas más      │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ DataFrame de Resultados     │ (Sin procesamiento adicional)
│ [Pregunta|Respuesta|Razon.] │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Excel Simple (1 hoja)       │
└─────────────────────────────┘
```

### V6: Flujo Jerárquico de Síntesis

```
┌────────────────────────────────────┐
│ Carga de Preguntas (APPRAISAL)     │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│ NIVEL 1: Análisis Individual        │ (ThreadPoolExecutor, 48 workers)
│ - Pregunta 1.1.1 ─→ {Respuesta...}│
│ - Pregunta 1.1.2 ─→ {Respuesta...}│
│ - Pregunta 1.2.1 ─→ {Respuesta...}│
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│ Extracción + Agrupación            │
│ results_df['_section_num'] = 1, 1, 1... │
│ results_df['_subsection'] = "1.1", "1.1", "1.2"... │
│ results_df['_sort_key'] = (1,1), (1,1), (1,2)...  │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│ NIVEL 2: Síntesis por Subsección   │ (Loop secuencial)
│ "1.1" → Síntesis(1.1.1, 1.1.2, ...) │ (LLM Call N+1)
│ "1.2" → Síntesis(1.2.1, 1.2.2, ...) │ (LLM Call N+2)
│ "1.3" → Síntesis(1.3.1, 1.3.2, ...) │ (LLM Call N+3)
│ "2.1" → Síntesis(2.1.1, 2.1.2, ...) │ (LLM Call N+4)
│ Progress Bar: [████████████░░] 8/10  │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│ Dict: subsection_analyses          │
│ {"1.1": "texto análisis...",       │
│  "1.2": "texto análisis...",       │
│  "1.3": "texto análisis...",       │
│  "2.1": "texto análisis..."}       │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│ NIVEL 3: Síntesis por Sección      │ (Loop secuencial)
│ Sección 1 → Síntesis({1.1, 1.2, 1.3})│ (LLM Call N+5)
│ Sección 2 → Síntesis({2.1, 2.2})   │ (LLM Call N+6)
│ Progress Bar: [██████░░░░] 2/3      │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│ Session State (3 niveles)          │
│ - tab1_results_df (preguntas)      │
│ - tab1_subsection_analyses         │
│ - tab1_section_analyses            │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│ Excel Jerárquico (2 hojas)         │
│ 1. "Detallado": Preguntas ordenadas │
│ 2. "Análisis por Sección":         │
│    ├─ Sección 1 (header azul)     │
│    ├─ Preguntas 1.1, 1.2, ...      │
│    ├─ Análisis 1.1 (merged)        │
│    ├─ Preguntas 1.3, 1.4, ...      │
│    ├─ Análisis 1.3 (merged)        │
│    └─ Análisis Sección 1 (merged)  │
│    ├─ Sección 2 (header azul)     │
│    └─ ...                          │
└────────────────────────────────────┘
```

---

## Cambios Funcionales

### 1. Extracción de Números de Sección/Subsección

| Función | Entrada | Salida | Propósito |
|---------|---------|--------|-----------|
| `extract_section_number()` | "1.1 ¿Está..." | `1` | Identificar sección para agrupación |
| `extract_subsection_number()` | "1.1 ¿Está..." | `"1.1"` | Identificar subsección exacta |
| `parse_subsection_for_sorting()` | `"1.1"` | `(1, 1)` | Ordenamiento numérico correcto |

**Ejemplo de Ordenamiento:**
```python
# SIN parse_subsection_for_sorting (String sorting - INCORRECTO)
subsections = ["1.2", "1.10", "1.20", "1.3"]
sorted(subsections)  # ["1.10", "1.2", "1.20", "1.3"] ❌

# CON parse_subsection_for_sorting (Numeric sorting - CORRECTO)
subsections = ["1.2", "1.10", "1.20", "1.3"]
sorted(subsections, key=parse_subsection_for_sorting)  # ["1.2", "1.3", "1.10", "1.20"] ✓
```

### 2. Síntesis por Subsección (NUEVA FUNCIÓN)

```python
def synthesize_subsection_analysis(subsection_id, subsection_questions_df):
    """
    NUEVA EN V6
    
    Características:
    - Input: DataFrame con Pregunta, Respuesta, Razonamiento
    - Scope: 1-2 párrafos
    - Token limit: 1500 máx
    - Reasoning effort: "minimal" (económico)
    
    Prompts construidos dinámicamente:
    1. Recopila respuestas (Yes/No/Partial/Not Found)
    2. Agrupa por tema si es posible
    3. Identifica patrones
    4. Señala inconsistencias
    
    Ejemplo de output:
    "La subsección 1.1 evalúa la documentación de procesos de gestión.
     De las 5 preguntas analizadas, 3 responden afirmativamente (60%),
     indicando que existe documentación pero con gaps en actualización.
     Se recomienda revisar la fecha de última actualización de...
    """
    pass
```

### 3. Síntesis por Sección (MODIFICADA)

**V4 (Original):**
```python
def synthesize_section_analysis(section_num, section_questions_df, document_text):
    # Toma PREGUNTAS individuales como input
    # Usa document_text para contexto adicional
    # Genera análisis a partir de respuestas de bajo nivel
    pass
```

**V6 (Mejorada):**
```python
def synthesize_section_analysis(section_num, subsection_analyses_dict):
    # Toma ANÁLISIS DE SUBSECCIONES como input
    # NO necesita document_text (ya está en subsección_analyses)
    # Genera síntesis de síntesis
    # Proporciona panorama ejecutivo de alto nivel
    
    # Ventajas:
    # 1. Reutiliza trabajo ya realizado
    # 2. Reduce tokens necesarios
    # 3. Proporciona mejor contexto ejecutivo
    # 4. Evita repetición de análisis
    pass
```

---

## Implementación de la Jerarquía

### Código: Grupo de Preguntas por Subsección

```python
# Línea 5149: Extracción de números
results_df['_section_num'] = results_df['Pregunta'].apply(extract_section_number)
results_df['_subsection'] = results_df['Pregunta'].apply(extract_subsection_number)
results_df['_sort_key'] = results_df['_subsection'].apply(parse_subsection_for_sorting)

# Ordenar por subsección (ascendente)
results_df = results_df.sort_values(by=['_sort_key']).reset_index(drop=True)
```

**Resultado:**
```
Pregunta                _subsection  _sort_key  Respuesta
1.1 ¿Documentado?         "1.1"      (1,1)      Yes
1.1 ¿Actualizado?         "1.1"      (1,1)      Partial
1.2 ¿Revisado?            "1.2"      (1,2)      No
1.3 ¿Publicado?           "1.3"      (1,3)      Yes
...
2.1 ¿Comunicado?          "2.1"      (2,1)      Yes
```

### Código: Síntesis por Subsección

```python
# Líneas 5160-5171: Loop de síntesis por subsección
subsections = sorted(results_df['_subsection'].dropna().unique(), 
                    key=parse_subsection_for_sorting)
subsection_analyses = {}

subsection_progress = st.progress(0)
for idx, subsection_id in enumerate(subsections):
    # Filtra preguntas de esta subsección
    subsection_df = results_df[results_df['_subsection'] == subsection_id].copy()
    
    # Genera síntesis
    subsection_analysis = synthesize_subsection_analysis(
        subsection_id,
        subsection_df[['Pregunta', 'Respuesta', 'Razonamiento']]
    )
    subsection_analyses[subsection_id] = subsection_analysis
    
    subsection_progress.progress((idx + 1) / len(subsections))

# Resultado: {"1.1": "texto...", "1.2": "texto...", ...}
```

### Código: Síntesis por Sección

```python
# Líneas 5174-5190: Loop de síntesis por sección
sections = sorted(results_df['_section_num'].dropna().unique())
section_analyses = {}

section_progress = st.progress(0)
for idx, section_num in enumerate(sections):
    section_num = int(section_num)
    
    # Filtra subsecciones de esta sección
    section_subsections = {
        subsec_id: subsection_analyses[subsec_id]
        for subsec_id in subsections
        if subsec_id.startswith(f"{section_num}.")
    }
    
    # Genera síntesis de síntesis
    section_analysis = synthesize_section_analysis(
        section_num,
        section_subsections  # ← CAMBIO: Usa análisis de subsecciones, no preguntas
    )
    section_analyses[section_num] = section_analysis
    
    section_progress.progress((idx + 1) / len(sections))

# Resultado: {1: "texto...", 2: "texto...", ...}
```

---

## Optimizaciones y Eficiencia

### Paralelización (Nivel 1: Preguntas)

```python
# V4 y V6: Idéntico - Paralelo máximo
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=48) as executor:
    futures = {executor.submit(analyze_single_question, q, r): q 
               for q, r in zip(questions, rubric_criteria)}
    
    results = []
    for future in as_completed(futures):
        result = future.result()
        results.append(result)

# Beneficio: Si hay 100 preguntas:
# - Serial: 100 × (2 segundos/call) = 200 segundos
# - Paralelo (48 workers): ~4 segundos (100/48 batches)
```

### Síntesis Secuencial (Nivel 2 y 3)

```python
# V6 (MODIFICADA): Síntesis secuencial es NECESARIA
# No se puede paralelizar porque:
# 1. Nivel 3 depende de Nivel 2
# 2. Número pequeño de items (típicamente <20 subsecciones, <5 secciones)

# Ejemplo de costos en tiempo:
# Nivel 1 (100 preguntas): ~4 segundos (paralelo 48)
# Nivel 2 (10 subsecciones): ~20 segundos (secuencial, 2 seg/call)
# Nivel 3 (3 secciones): ~6 segundos (secuencial, 2 seg/call)
# TOTAL: ~30 segundos

# V4 para comparación: 100 llamadas × 2 seg = 200 segundos (paralelo)
# V6: 30 + 13 = 43 segundos (100 + 10 + 3 = 113 llamadas, pero con paralelo)
```

### Optimización de Tokens LLM

```python
# V4: Cada pregunta requiere contexto completo
# Pregunta: "¿Está documentado el proceso de X?"
# Context: 1000 tokens (documento completo)
# Respuesta: 500 tokens
# TOTAL por pregunta: ~1500 tokens
# Para 100 preguntas: 150,000 tokens

# V6: Reutiliza respuestas previas
# Nivel 1 (100 preguntas): 150,000 tokens
# Nivel 2 (10 subsecciones): 
#   - Subsección 1: 5 respuestas × 100 tokens cada una = 500 tokens input
#                   + 500 tokens output = 1000 tokens total
#   - 10 subsecciones × 1000 = 10,000 tokens
# Nivel 3 (3 secciones):
#   - Sección 1: 3 análisis × 300 tokens = 900 tokens input
#             + 800 tokens output = 1700 tokens
#   - 3 secciones × 1700 = 5,100 tokens
# TOTAL V6: 150,000 + 10,000 + 5,100 = 165,100 tokens
# 
# Aumento aparente (165K vs 150K) es por síntesis adicional
# PERO: Valor agregado exponencial (3 niveles de análisis vs 1)
# ROI: +10% tokens → +300% análisis útil
```

---

## Salida de Datos

### V4: Excel Simple (1 hoja)

```
┌─────────────────────────────────────────────────────────────┐
│ Sheet: "Resultados"                                          │
├────────┬──────────────┬─────────┬──────────────────────────┤
│Pregunta│ Respuesta    │Razonam. │ Evidencia                │
├────────┼──────────────┼─────────┼──────────────────────────┤
│1.1 ¿Es │ Yes          │Se...    │Págs 5-8 demuestran...   │
│1.2 ¿Es │ Partial      │Incoms...│Falta capítulo X pero...  │
│1.3 ¿Es │ No           │No se... │Documento no menciona...  │
│2.1 ¿Hay│ Yes          │Sí, en..│Págs 12-15 muestran...   │
│...     │ ...          │...      │ ...                      │
└────────┴──────────────┴─────────┴──────────────────────────┘

Ventajas: Simple, directo
Desventajas: Sin contexto, sin síntesis, datos desorganizados
```

### V6: Excel Jerárquico (2 hojas)

**Hoja 1: "Detallado" (igual a V4, pero ordenado)**
```
┌──────────┬──────────────┬──────────┬──────────────────┬─────────┐
│Pregunta  │ Respuesta    │Razonam. │ Evidencia        │ Status  │
├──────────┼──────────────┼──────────┼──────────────────┼─────────┤
│1.1 ¿Es..│ Yes          │Se...    │Págs 5-8...      │Success │
│1.1 ¿Es..│ Partial      │Incoms...│Falta capítulo...│Success │
│1.2 ¿Es..│ No           │No se... │Doc no menciona..|Success │
│1.3 ¿Hay│ Yes          │Sí, en..│Págs 12-15...    │Success │
│2.1 ¿Hay│ Yes          │Sí...   │Evidencia clara..│Success │
└──────────┴──────────────┴──────────┴──────────────────┴─────────┘
```

**Hoja 2: "Análisis por Sección" (NUEVA)**
```
╔════════════════════════════════════════════════════════════════════╗
║ Sección 1                                                           ║ ← Header azul
╠──────────┬──────────────┬──────────┬──────────────────────────────╣
│Subsección│ Pregunta     │ Respuesta│ Razonamiento                  │
├──────────┼──────────────┼──────────┼──────────────────────────────┤
│1.1       │1.1 ¿Es doc. │ Yes      │Se encontró documentación...   │
│1.1       │1.1 ¿Es actual│ Partial │Actualización incompleta...    │
├──────────┴──────────────┴──────────┴──────────────────────────────┤
│ Análisis 1.1: [SÍNTESIS DE SUBSECCIÓN 1-2 PÁRRAFOS]              │ ← Merged, azul claro
│ La subsección 1.1 demuestra cumplimiento en documentación...     │
├──────────┬──────────────┬──────────┬──────────────────────────────┤
│1.2       │1.2 ¿Es accs. │ No       │No se encuentra proceso...    │
│1.2       │1.2 ¿Es fácil │ Partial │Acceso restringido a...       │
├──────────┴──────────────┴──────────┴──────────────────────────────┤
│ Análisis 1.2: [SÍNTESIS DE SUBSECCIÓN]                            │
│ La subsección 1.2 indica limitaciones en accesibilidad...        │
├──────────┴──────────────┴──────────┴──────────────────────────────┤
│ Análisis Sección 1: [SÍNTESIS DE SECCIÓN 2-3 PÁRRAFOS]            │ ← Merged, gris
│ La Sección 1 evalúa elementos fundamentales de gobernanza...     │
│ Hallazgos principales: documentación presente pero con gaps...   │
│ Recomendaciones: revisar acceso y actualización de procesos...   │
╠════════════════════════════════════════════════════════════════════╣
║ Sección 2                                                           ║
├──────────┬──────────────┬──────────┬──────────────────────────────┤
│2.1       │2.1 ¿Hay plan │ Yes      │Plan estratégico documentado..│
│2.1       │2.1 ¿Es públi │ Yes      │Publicado en portal web...    │
├──────────┴──────────────┴──────────┴──────────────────────────────┤
│ Análisis 2.1: [SÍNTESIS DE SUBSECCIÓN]                            │
│ La subsección 2.1 muestra cumplimiento integral...               │
├──────────┴──────────────┴──────────┴──────────────────────────────┤
│ Análisis Sección 2: [SÍNTESIS DE SECCIÓN]                        │
│ La Sección 2 refleja fortaleza en planificación estratégica...   │
└────────────────────────────────────────────────────────────────────┘

Ventajas: 
- Contexto visual claro (encabezados de sección)
- Síntesis agregada visible
- Fácil para ejecutivos (leen primero síntesis)
- Estructura lógica para auditoría

Desventajas:
- Más complejo de navegar
- Más espacioso (más filas)
```

---

## Consideraciones de Rendimiento

### Tiempo de Ejecución

| Métrica | V4 | V6 |
|---------|----|----|
| 100 preguntas individuales | 200s (serial) / 4s (paralelo 48) | 4s (Nivel 1) |
| 10 síntesis subsecciones | - | 20s |
| 3 síntesis secciones | - | 6s |
| TOTAL estimado | 4s | 30s |
| Síntesis ejecutiva | No | Sí |

### Memoria

```python
# V4
results_df: 100 rows × 5 columns = ~100 KB
section_analyses: dict, 5 items = ~10 KB
TOTAL: ~110 KB

# V6
results_df: 100 rows × 8 columns (+ _section_num, _subsection, _sort_key) = ~150 KB
subsection_analyses: dict, 10 items = ~50 KB
section_analyses: dict, 3 items = ~15 KB
TOTAL: ~215 KB

# Aumento: 2x memoria por 3x análisis (ROI positivo)
```

### Costo de API (OpenAI)

**Modelo: GPT-5-mini**
- Input: $0.075 / 1M tokens
- Output: $0.30 / 1M tokens

```python
# V4: 100 preguntas × 1500 tokens promedio = 150,000 tokens
# Costo promedio: (150K × $0.075 + 150K × $0.30) / 1M = $0.051

# V6: 165,100 tokens total
# Costo promedio: (165.1K × $0.075 + 165.1K × $0.30) / 1M = $0.056

# Aumento: $0.005 por análisis completo (muy bajo)
# Valor agregado: 3 niveles vs 1 nivel (300% más análisis)
```

### Optimizaciones Posibles

```python
# 1. Cache de LLM (si se analizan documentos similares)
# V6 permite reutilizar análisis de subsecciones para documentos nuevos
# Reducción potencial: 30% del tiempo de síntesis

# 2. Análisis incremental
# Si se actualiza 1 pregunta, solo recalcular su subsección y sección
# Vs V4: recalcular todo

# 3. Batch processing de síntesis por subsección
# Paralelizar Nivel 2 en groups de 5 subsecciones (si < 5 en paralelo)
# Reducción potencial: 50% en tiempo de síntesis por subsección
```

---

## Resumen de Cambios

| Aspecto | V4 | V6 | Cambio |
|---------|----|----|--------|
| Niveles de análisis | 1 | 3 | +200% |
| Llamadas LLM | 100 | 113 | +13% |
| Tiempo de ejecución | 4s (paralelo) | 30s | +7x (pero 3x análisis) |
| Hojas Excel | 1 | 2 | +1 |
| Filas Excel | 100 | 300+ | +3x (síntesis agregadas) |
| Contexto por pregunta | Ninguno | Nivel subsección + sección | +2 contextos |
| Síntesis ejecutiva | No | Sí (nivel sección) | Nuevo |
| Ordenamiento | No | Ascendente completo | Nuevo |
| Fusión de celdas | No | Sí | Nuevo (formateo) |

---

## Conclusión

La evolución de V4 a V6 representa un cambio arquitectónico fundamental:

- **De:** Análisis atomizado (pregunta aislada)
- **A:** Análisis jerárquico (pregunta → subsección → sección)

Este cambio permite:
1. **Mejor contexto:** Cada síntesis entiende su contexto jerárquico
2. **Síntesis de síntesis:** Información agregada de múltiples niveles
3. **Eficiencia:** Reutilización de análisis previos
4. **Presentación:** Excel profesional con estructura ejecutiva

El costo adicional (13% más llamadas LLM, 7x tiempo de ejecución) se justifica ampliamente por el valor agregado (3x niveles de análisis) y la mejora en presentación y usabilidad.
