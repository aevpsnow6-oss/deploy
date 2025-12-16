# Excel Output Structure Changes - V6

## Summary
Modified the Excel output to use **3 separate sheets** instead of 2, with complete sorting enforcement across all levels.

## Previous Structure (2 sheets)
1. **"Detallado"** - All questions in flat table
2. **"Análisis por Sección"** - Hierarchical view with merged cells

## New Structure (3 sheets)

### Sheet 1: "1. Preguntas" (Questions)
- **Purpose**: Individual question analysis
- **Columns**: Subsección | Pregunta | Respuesta | Razonamiento | Evidencia | Status
- **Sorting**: ✅ Enforced ascending order by subsection (1.1, 1.2, 1.3... 1.10, 1.20)
- **Format**: Clean table with subsection identifier in first column

### Sheet 2: "2. Análisis Subsecciones" (Subsection Analysis)
- **Purpose**: Synthesized analysis at subsection level
- **Columns**: Subsección | Sección | Análisis de Subsección
- **Sorting**: ✅ Enforced ascending order by subsection
- **Content**: One row per subsection with 1-2 paragraph synthesis
- **Format**: Light blue background for analysis column

### Sheet 3: "3. Análisis Secciones" (Section Analysis)
- **Purpose**: Executive-level synthesis at section level
- **Columns**: Sección | Análisis de Sección
- **Sorting**: ✅ Enforced ascending order by section number
- **Content**: One row per section with 2-3 paragraph synthesis
- **Format**: Gray background for analysis column, blue header for section

## Key Improvements

### 1. Sorting Enforcement
```python
# Ensure sorting columns exist
if '_sort_key' not in results_df.columns:
    results_df['_sort_key'] = results_df['_subsection'].apply(parse_subsection_for_sorting)

# Sort entire dataframe once
results_df_sorted = results_df.sort_values(by='_sort_key').reset_index(drop=True)

# Use sorted dataframe for all sheets
```

### 2. Cleaner Separation
- **Questions**: Raw data, easy to filter/pivot
- **Subsections**: Mid-level synthesis, shows patterns within subsections
- **Sections**: High-level synthesis, executive summary

### 3. Better Usability
- Sheet names prefixed with numbers (1., 2., 3.) for clear hierarchy
- Each sheet is self-contained and sortable
- No merged cells in questions sheet (easier to work with in Excel)
- Proper column widths for readability

## Column Widths

### Sheet 1 (Questions)
- Subsección: 12 chars
- Pregunta: 35 chars
- Respuesta: 12 chars
- Razonamiento: 50 chars
- Evidencia: 40 chars
- Status: 10 chars

### Sheet 2 (Subsections)
- Subsección: 12 chars
- Sección: 15 chars
- Análisis: 80 chars

### Sheet 3 (Sections)
- Sección: 15 chars
- Análisis: 90 chars

## Updated Summary Report
The ZIP file summary now reflects the 3-sheet structure:

```
Archivo Excel generado con 3 hojas:
  1. Preguntas: Análisis individual de cada pregunta (ordenado por subsección)
  2. Análisis Subsecciones: Síntesis de preguntas agrupadas por subsección
  3. Análisis Secciones: Síntesis ejecutiva por sección
```

## Benefits

1. **Clear Hierarchy**: Three distinct levels visible in sheet organization
2. **Proper Sorting**: All data sorted numerically (not alphabetically)
3. **Easy Navigation**: Numbered sheet names guide users through analysis levels
4. **Data Integrity**: Questions sheet is clean tabular data without merged cells
5. **Executive Friendly**: Section analysis sheet provides quick high-level view
6. **Complete Coverage**: All three levels of analysis clearly presented

## Technical Implementation

### Sorting Function
Uses `parse_subsection_for_sorting()` to convert subsection strings to numeric tuples:
- "1.1" → (1, 1)
- "1.10" → (1, 10)
- "2.3" → (2, 3)

This ensures correct numeric ordering: 1.1, 1.2, 1.3, ... 1.9, 1.10, 1.20

### Data Flow
1. Extract section/subsection/sort_key from all questions
2. Sort entire dataframe by sort_key
3. Generate Sheet 1 from sorted data
4. Iterate subsections in sorted order for Sheet 2
5. Iterate sections in sorted order for Sheet 3
