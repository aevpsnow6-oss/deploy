# Documentación Técnica: Evolución del Motor de Análisis
## De Versión Simple (V4) a Pipeline de 6 Pasadas (V6.1)

**Fecha:** Marzo 2026
**Objetivo:** Transformación del sistema de análisis de preguntas individuales a un pipeline jerárquico de 6 pasadas con evaluación crítica en cada nivel

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura V4: Sistema Original](#arquitectura-v4-sistema-original)
3. [Arquitectura V6: Pipeline de 6 Pasadas](#arquitectura-v6-pipeline-de-6-pasadas)
4. [Flujo de Datos Comparativo](#flujo-de-datos-comparativo)
5. [Detección de Preguntas de Dos Partes](#detección-de-preguntas-de-dos-partes)
6. [Gestión de Contexto con tiktoken](#gestión-de-contexto-con-tiktoken)
7. [Cambios Funcionales](#cambios-funcionales)
8. [Salida de Datos](#salida-de-datos)
9. [Consideraciones de Rendimiento](#consideraciones-de-rendimiento)

---

## Resumen Ejecutivo

### Versión 4 (Original)
- **Análisis por pregunta individual**
- Una llamada LLM por pregunta
- Salida Excel simple con columnas planas
- No hay síntesis ni evaluación crítica

### Versión 6.1 (Actual)
- **Pipeline de 6 pasadas** con evaluación crítica en paralelo y síntesis jerárquica
  1. Pasada 1 — Análisis individual (paralelo, ThreadPoolExecutor MAX_WORKERS=48)
  2. Pasada 2 — Evaluación crítica individual (paralelo, ThreadPoolExecutor MAX_WORKERS=48)
  3. Pasada 3 — Síntesis por subsección (secuencial)
  4. Pasada 4 — Evaluación crítica de subsección (secuencial)
  5. Pasada 5 — Síntesis por sección (secuencial)
  6. Pasada 6 — Evaluación crítica de sección (secuencial)
- Gestión precisa de contexto mediante `tiktoken` (límite de 110,000 tokens)
- Detección automática de preguntas de dos partes (`parse_two_part_question`)
- Salida: ZIP con 3 hojas Excel + plantilla de rúbrica original
- Tiempo estimado total: 3–10 minutos dependiendo del tamaño del documento

---

## Arquitectura V4: Sistema Original

### Flujo de Procesamiento

```
PREGUNTA → ANÁLISIS INDIVIDUAL → RESULTADO
  ↓              ↓                  ↓
1.1         LLM Call            {Respuesta: Yes,
1.2         (1 pasada)           Razonamiento: "..."}
2.1
...
```

### Estructura de Datos V4

```python
resultado_pregunta = {
    'Pregunta': '1.1 ¿Está documentado...?',
    'Respuesta': 'Yes',
    'Razonamiento': 'Se encontró evidencia en págs. 5-8',
    'Evidencia': '[texto_documento]',
    'Status': 'Success'
}
```

### Características Clave V4

1. **Análisis Atomizado** — cada pregunta se analiza de forma independiente sin relación entre preguntas.

2. **Una pasada LLM** — `analyze_question_with_llm()` genera un JSON con `Respuesta`, `Razonamiento` y `Evidencia`.

3. **Limitación de contexto basada en caracteres** — el texto del documento se truncaba usando estimación de caracteres (aprox. 4 chars/token), sin control preciso.

4. **Salida Simple** — Tabla Excel con columnas: `Pregunta | Respuesta | Razonamiento | Evidencia | Status`, sin síntesis ni jerarquía.

5. **Limitaciones**
   - No captura patrones entre preguntas relacionadas
   - No hay evaluación de la calidad del análisis
   - Difícil identificar tendencias por área o sección

---

## Arquitectura V6: Pipeline de 6 Pasadas

### Diagrama del Pipeline Completo

```
PREGUNTAS (1.1, 1.2, 1.3, ...)
     ↓
═══════════════════════════════════════════════════════════
PASADA 1: ANÁLISIS INDIVIDUAL (Paralelo — ThreadPoolExecutor, MAX_WORKERS=48)
  analyze_question_with_llm_tab1(question, document_text)
  → {Respuesta, Razonamiento, Evidencia} por cada pregunta
═══════════════════════════════════════════════════════════
     ↓
═══════════════════════════════════════════════════════════
PASADA 2: EVALUACIÓN CRÍTICA INDIVIDUAL (Paralelo — ThreadPoolExecutor, MAX_WORKERS=48)
  analyze_question_with_critical_opinion_tab1(question, answer, reasoning, evidence, doc_text)
  → Evaluación crítica (texto libre, máx 150 palabras) por cada pregunta
═══════════════════════════════════════════════════════════
     ↓
 Agrupación por subsección (_subsection) y sección (_section_num)
 Ordenamiento ascendente via parse_subsection_for_sorting()
     ↓
═══════════════════════════════════════════════════════════
PASADA 3: SÍNTESIS POR SUBSECCIÓN (Secuencial)
  synthesize_subsection_analysis(subsection_id, subsection_questions_df)
  → Análisis sintetizado 1-2 párrafos por subsección
═══════════════════════════════════════════════════════════
     ↓
═══════════════════════════════════════════════════════════
PASADA 4: EVALUACIÓN CRÍTICA DE SUBSECCIÓN (Secuencial)
  synthesize_critical_evaluation_subsection(subsection_id, critical_opinions_df)
  → Evaluación crítica sintetizada 1-2 párrafos por subsección
═══════════════════════════════════════════════════════════
     ↓
═══════════════════════════════════════════════════════════
PASADA 5: SÍNTESIS POR SECCIÓN (Secuencial)
  synthesize_section_analysis(section_num, subsection_analyses_dict)
  → Análisis sintetizado 2-3 párrafos por sección
═══════════════════════════════════════════════════════════
     ↓
═══════════════════════════════════════════════════════════
PASADA 6: EVALUACIÓN CRÍTICA DE SECCIÓN (Secuencial)
  synthesize_critical_evaluation_section(section_num, critical_subsection_dict)
  → Evaluación crítica sintetizada 2-3 párrafos por sección
═══════════════════════════════════════════════════════════
     ↓
SALIDA: ZIP con 3 hojas Excel + rúbrica original
  create_results_download_with_sections()
```

### Configuración Central

```python
MAX_WORKERS = 48          # Pasadas 1 y 2 (paralelas)
OPENAI_MODEL = "gpt-5-mini"  # Todas las pasadas LLM
TOKEN_LIMIT = 110_000     # Ventana de contexto por llamada (tiktoken cl100k_base)
```

### Estructura de Datos V6

```python
# Nivel pregunta (Pasadas 1 y 2)
resultado_pregunta = {
    'Pregunta': '1.1 ¿Está documentado...?',
    '_section_num': 1,
    '_subsection': '1.1',
    '_sort_key': (1, 1),
    'Respuesta': 'Yes',
    'Razonamiento': 'Se identificaron 2 partes en esta pregunta. ...',
    'Evidencia': '[citas directas del documento]',
    'Evaluación Crítica': 'El análisis identifica correctamente las dos partes...',
    'Status': 'Success'
}

# Nivel subsección (Pasadas 3 y 4)
subsection_analyses = {
    '1.1': "En el análisis de la subsección 1.1, que comprende...",
    '1.2': "Respecto a la subsección 1.2, los resultados muestran..."
}
critical_subsection_analyses = {
    '1.1': "La subsección 1.1 presenta brechas en la documentación de...",
    '1.2': "La evaluación crítica revela que subsección 1.2..."
}

# Nivel sección (Pasadas 5 y 6)
section_analyses = {
    1: "La Sección 1 demuestra un nivel de cumplimiento...",
    2: "En cuanto a la Sección 2, se observa que..."
}
critical_section_analyses = {
    1: "Críticamente, la Sección 1 presenta riesgos sistémicos en...",
    2: "La evaluación estratégica de la Sección 2 indica..."
}
```

---

## Flujo de Datos Comparativo

### V4: Flujo Lineal

```
┌─────────────────────────────┐
│ Carga de Preguntas (rubrica) │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Análisis Paralelo (1 pasada)│ (ThreadPoolExecutor, 48 workers)
│ - Pregunta 1.1 → LLM Call  │
│ - Pregunta 1.2 → LLM Call  │
│ - Pregunta 2.1 → LLM Call  │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Excel Simple (1 hoja)       │
└─────────────────────────────┘
```

### V6.1: Flujo de 6 Pasadas

```
┌────────────────────────────────────────┐
│ Carga de Preguntas + Documento         │
│ truncate_to_token_limit(doc, 110000)   │  ← tiktoken cl100k_base
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│ PASADA 1: Análisis Individual          │  (Paralelo, 48 workers)
│ analyze_question_with_llm_tab1()       │
│ - Detecta preguntas de dos partes      │
│ - Genera Respuesta/Razonamiento/Evidencia│
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│ PASADA 2: Evaluación Crítica           │  (Paralelo, 48 workers)
│ analyze_question_with_critical_tab1()  │
│ - Verifica tratamiento de 2 partes     │
│ - Genera columna "Evaluación Crítica"  │
└──────────────┬─────────────────────────┘
               ↓
 Extracción + Agrupación + Ordenamiento
 results_df['_section_num'], ['_subsection'], ['_sort_key']
               ↓
┌────────────────────────────────────────┐
│ PASADA 3: Síntesis Subsección          │  (Secuencial)
│ synthesize_subsection_analysis()       │
│ - max_completion_tokens=1500           │
│ - reasoning_effort="minimal"           │
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│ PASADA 4: Crítica Subsección           │  (Secuencial)
│ synthesize_critical_evaluation_subsec()│
│ - max_completion_tokens=1500           │
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│ PASADA 5: Síntesis Sección             │  (Secuencial)
│ synthesize_section_analysis()          │
│ - max_completion_tokens=2000           │
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│ PASADA 6: Crítica Sección              │  (Secuencial)
│ synthesize_critical_evaluation_sec()   │
│ - max_completion_tokens=2000           │
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│ create_results_download_with_sections()│
│ Salida ZIP con 3 hojas Excel           │
│ + rúbrica original PRODOC_rubric.xlsx  │
└────────────────────────────────────────┘
```

---

## Detección de Preguntas de Dos Partes

**Función:** `parse_two_part_question()` — Líneas 6052–6105

Muchas preguntas de la rúbrica contienen dos sub-preguntas implícitas. La función aplica dos patrones de detección en orden:

**Patrón 1 — Palabras clave separadoras:**
```
"¿Se ha documentado el proceso? Específicamente, ¿se actualiza trimestralmente?"
                                 ↑
                           separador detectado
```
Palabras clave: `específicamente`, `en particular`, `en concreto`, `puntualmente`, `además`

**Patrón 2 — Doble signo de interrogación:**
```
"¿Se ha documentado el proceso?¿Se audita anualmente?"
                               ↑
                         segundo ¿ detectado
```

**Resultado cuando se detecta una pregunta de dos partes:**

```python
{
    'is_two_part': True,
    'part1': '¿Se ha documentado el proceso?',   # Contexto general
    'part2': '¿Se actualiza trimestralmente?',    # Foco específico (70% del peso)
    'full_question': '...'
}
```

El prompt del LLM en las Pasadas 1 y 2 cambia según este resultado:
- **Pregunta simple:** prompt estándar con JSON `{Respuesta, Razonamiento, Evidencia}`
- **Pregunta de dos partes:** prompt especializado que exige comenzar el razonamiento con `"Se identificaron 2 partes en esta pregunta."` y ponderar la Parte 2 al 70%

En la Pasada 2 (evaluación crítica), se verifica explícitamente que el razonamiento reconozca las dos partes; no hacerlo se considera un fallo crítico.

---

## Gestión de Contexto con tiktoken

**Función:** `truncate_to_token_limit()` — Líneas 54–86

```python
try:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
except ImportError:
    encoding = None  # fallback: estimación por caracteres

def truncate_to_token_limit(text, max_tokens=110000, encoding_obj=None):
    if encoding_obj is None:
        # Fallback: 4 chars/token para español → 110K × 4 = 440K chars
        return text[:max_tokens * 4]
    tokens = encoding_obj.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding_obj.decode(tokens[:max_tokens])
```

El límite de 110,000 tokens reserva ~18,000 tokens de margen para el prompt del sistema y la respuesta dentro de la ventana de 128K del modelo GPT-5-mini.

**Cambio respecto a V4:**

| Aspecto | V4 | V6.1 |
|---------|-----|------|
| Método | Truncado por caracteres | Truncado por tokens (tiktoken) |
| Límite | ~400K chars (estimación) | 110,000 tokens (preciso) |
| Encoding | N/A | cl100k_base (GPT-4/GPT-5 estándar) |
| Fallback | No | Sí (estimación por caracteres) |

---

## Cambios Funcionales

### Funciones de Extracción y Ordenamiento (sin cambio respecto a V4 funcional)

| Función | Entrada | Salida | Propósito |
|---------|---------|--------|-----------|
| `extract_section_number()` | `"1.1 ¿Está..."` | `1` | Identificar sección |
| `extract_subsection_number()` | `"1.1 ¿Está..."` | `"1.1"` | Identificar subsección |
| `parse_subsection_for_sorting()` | `"1.1"` | `(1, 1)` | Ordenamiento numérico correcto |

### Funciones de Análisis (versiones Tab1 especializadas, nuevas en V6)

| Función | V4 | V6.1 |
|---------|-----|------|
| `analyze_question_with_llm_tab1()` | No existía | Pasada 1; detecta dos partes |
| `analyze_question_with_critical_opinion_tab1()` | No existía | Pasada 2; verifica dos partes |
| `synthesize_subsection_analysis()` | Existía | Sin cambio |
| `synthesize_section_analysis()` | Existía | Sin cambio |
| `synthesize_critical_evaluation_subsection()` | No existía | Pasada 4 |
| `synthesize_critical_evaluation_section()` | No existía | Pasada 6 |

### Paralelización

Las Pasadas 1 y 2 se ejecutan en paralelo con dos `ThreadPoolExecutor` independientes:

```python
# Pasada 1: Análisis individual
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures_analysis = {
        executor.submit(analyze_question_with_llm_tab1, q, doc_text): q
        for q in questions
    }

# Pasada 2: Evaluación crítica (inicia después de completar Pasada 1)
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures_critical = {
        executor.submit(analyze_question_with_critical_opinion_tab1,
                        q, answer, reasoning, evidence, doc_text): q
        for q, answer, reasoning, evidence in results_pasada1
    }
```

Las Pasadas 3–6 son secuenciales por diseño (cada pasada depende de la anterior).

---

## Salida de Datos

### V4: Excel Simple (1 hoja)

```
Sheet: "Resultados"
┌──────────┬──────────┬──────────────┬──────────────┬─────────┐
│ Pregunta │ Respuesta│ Razonamiento │ Evidencia    │ Status  │
├──────────┼──────────┼──────────────┼──────────────┼─────────┤
│ 1.1 ¿Es..│ Yes      │ Se encontró..│ Págs 5-8...  │ Success │
│ 1.2 ¿Es..│ Partial  │ Incompleto...│ Falta cap... │ Success │
│ 2.1 ¿Hay │ Yes      │ Sí, en...    │ Págs 12-15.. │ Success │
└──────────┴──────────┴──────────────┴──────────────┴─────────┘
```

### V6.1: ZIP con 3 Hojas Excel + Rúbrica

Función: `create_results_download_with_sections()` — Líneas 6846–6995

El archivo ZIP descargable contiene:

**Hoja "1. Preguntas"** — Análisis individual con evaluación crítica

```
┌──────────┬──────────┬──────────────┬──────────────┬───────────────────┬─────────┐
│ Pregunta │ Respuesta│ Razonamiento │ Evidencia    │ Evaluación Crítica│ Status  │
├──────────┼──────────┼──────────────┼──────────────┼───────────────────┼─────────┤
│ 1.1 ¿Es..│ Yes      │ Se identific.│ Págs 5-8...  │ El análisis ide...│ Success │
│ 1.2 ¿Es..│ Partial  │ Se identific.│ Falta cap... │ La evaluación cr..│ Success │
│ 2.1 ¿Hay │ Yes      │ Plan documen.│ Págs 12-15.. │ Correcto pero omi.│ Success │
└──────────┴──────────┴──────────────┴──────────────┴───────────────────┴─────────┘
```
**Nueva columna `Evaluación Crítica`** contiene el output de la Pasada 2.

**Hoja "2. Análisis Subsecciones"** — Síntesis y crítica por subsección

```
┌─────────────┬────────────────────────────────┬──────────────────────────────────┐
│ Subsección  │ Análisis de Subsección         │ Evaluación Crítica Subsección     │
├─────────────┼────────────────────────────────┼──────────────────────────────────┤
│ 1.1         │ La subsección 1.1 evalúa...    │ Críticamente, la subsección 1.1..│
│ 1.2         │ Respecto a 1.2, los resultados │ La evaluación detecta brechas en..│
│ 2.1         │ La subsección 2.1 muestra...   │ El análisis de 2.1 es sólido...  │
└─────────────┴────────────────────────────────┴──────────────────────────────────┘
```

**Hoja "3. Análisis Secciones"** — Síntesis y crítica por sección

```
┌─────────┬────────────────────────────────┬──────────────────────────────────────┐
│ Sección │ Análisis de Sección            │ Evaluación Crítica Sección            │
├─────────┼────────────────────────────────┼──────────────────────────────────────┤
│ 1       │ La Sección 1 demuestra...      │ Estratégicamente, la Sección 1...    │
│ 2       │ En cuanto a la Sección 2...    │ La evaluación de Sección 2 revela... │
└─────────┴────────────────────────────────┴──────────────────────────────────────┘
```

**Archivo adicional:** `PRODOC_rubric.xlsx` (rúbrica original, incluida en el ZIP para referencia).

---

## Consideraciones de Rendimiento

### Tiempo de Ejecución Estimado

| Pasada | Operación | Modo | Tiempo típico |
|--------|-----------|------|---------------|
| 1 | Análisis individual (100 preguntas) | Paralelo 48 workers | 4–8 s |
| 2 | Evaluación crítica (100 preguntas) | Paralelo 48 workers | 4–8 s |
| 3 | Síntesis subsecciones (10 subsec.) | Secuencial | 20–30 s |
| 4 | Crítica subsecciones (10 subsec.) | Secuencial | 20–30 s |
| 5 | Síntesis secciones (3 secciones) | Secuencial | 6–12 s |
| 6 | Crítica secciones (3 secciones) | Secuencial | 6–12 s |
| **Total** | | | **~60–100 s (doc pequeño) / 3–10 min (doc grande)** |

### Llamadas LLM Estimadas

```
N = número de preguntas
S = número de subsecciones
K = número de secciones

Total de llamadas = N (Pasada 1) + N (Pasada 2) + S (Pasada 3) + S (Pasada 4) + K (Pasada 5) + K (Pasada 6)

Ejemplo con 100 preguntas, 10 subsecciones, 3 secciones:
= 100 + 100 + 10 + 10 + 3 + 3 = 226 llamadas

Comparación V4: 100 llamadas
Incremento V6.1: +126% llamadas → +500% análisis generado
```

### Costo de API (referencial, GPT-5-mini)

| Versión | Llamadas | Tokens aprox. | Costo estimado |
|---------|----------|---------------|----------------|
| V4 | ~100 | ~150K | ~$0.05 |
| V6.1 | ~226 | ~350K | ~$0.12 |

El incremento de costo (~$0.07 adicional por análisis completo) se justifica por la profundidad del análisis generado.

---

## Resumen de Cambios V4 → V6.1

| Aspecto | V4 | V6.1 | Cambio |
|---------|----|------|--------|
| Pasadas LLM por documento | 1 | 6 | +500% |
| Evaluación crítica | No | Sí (Pasadas 2, 4, 6) | Nuevo |
| Detección preguntas 2 partes | No | Sí (`parse_two_part_question`) | Nuevo |
| Gestión de tokens | Caracteres (estimación) | tiktoken cl100k_base (exacto) | Mejorado |
| Llamadas LLM (100 preguntas) | ~100 | ~226 | +126% |
| Hojas de salida | 1 | 3 + rúbrica en ZIP | +2 hojas |
| Columna Evaluación Crítica | No | Sí | Nuevo |
| Tiempo total estimado | ~4 s | 3–10 min | Mayor pero más completo |
| Tabs de clasificación de recomendaciones | No | Tab 5 + Tab 6 | Nuevo módulo |

---

## Conclusión

La evolución de V4 a V6.1 transforma el sistema de un analizador de preguntas individuales a un motor de evaluación jerárquica con perspectiva crítica integrada. Las seis pasadas del pipeline permiten que cada nivel de síntesis se construya sobre el trabajo previo, produciendo un análisis ejecutivo de alto valor sin duplicar el procesamiento del documento original.

El costo adicional en tiempo (~3–10 min vs ~4 s) y en llamadas a la API (~226 vs ~100) es proporcional al valor agregado: el usuario recibe no solo respuestas individuales sino síntesis por subsección, por sección, y evaluaciones críticas en cada nivel, todo empaquetado en un ZIP con tres hojas temáticas.
