# Tab3 Counting-Based Rubric Implementation

## Overview
Enhanced Tab3's evaluation function to handle **counting-based rubrics** where scores are determined by explicit counts of:
1. **Types of stakeholders** (Government, Employers, Workers)
2. **Forms/ways of participation** for each stakeholder type

## Problem Statement
The sustainability rubric includes indicators like:
> "Los mandantes participan activamente en el diseño e implementación del Proyecto"

The scoring levels require specific counts:
- **Level 1**: 1 stakeholder in 1 form
- **Level 2**: 1 stakeholder in 2+ forms OR 2+ stakeholders in 1 form
- **Level 3**: 2 stakeholders in 2 forms each
- **Level 4**: 2+ stakeholders in 2+ forms each
- **Level 5**: 3 stakeholders in 2+ forms each

## Implementation

### Function: `evaluate_criterion_with_llm()` (line 4888)

**Auto-Detection:**
The function automatically detects stakeholder counting rubrics by looking for keywords:
- `mandante`, `participan`, `stakeholder`, `actores`, `gobierno`, `empleador`, `trabajador`

**When Detected:**
Uses an enhanced counting-focused prompt that instructs the LLM to:

1. **Identify and count stakeholders by category:**
   - Government (gobierno, autoridades, ministerios, funcionarios públicos)
   - Employers (empleadores, empresas, organizaciones patronales, cámaras)
   - Workers (trabajadores, sindicatos, organizaciones de trabajadores)

2. **Identify and count participation forms for EACH stakeholder:**
   - Design, meetings, discussions, comments, information provision, co-implementation, commitments, etc.

3. **Structure the analysis with explicit counts:**
   ```
   CONTEO DE MANDANTES:
   - Gobierno: 2 mandantes identificados
     * Formas de participación: diseño, reuniones (2 formas)
   - Empleadores: 1 mandante identificado
     * Formas de participación: diseño, comentarios, co-implementación (3 formas)
   - Trabajadores: 1 mandante identificado
     * Formas de participación: reuniones, discusiones (2 formas)

   TOTAL: 3 tipos de mandantes participando en múltiples formas

   JUSTIFICACIÓN DEL PUNTAJE:
   Se identifican 3 tipos de mandantes (gobierno, empleadores, trabajadores)
   participando en 2 o más formas cada uno. Esto corresponde al Nivel 5 de
   la rúbrica que requiere "3 mandantes en 2+ formas cada uno".
   ```

4. **Justify the score based on the counts**

## Example Output

**Indicator:**
> "1.1 Los mandantes participan activamente en el diseño e implementación del Proyecto"

**Scoring Levels:**
- Nivel 1: Ningún mandante participa / 1 mandante en 1 forma
- Nivel 2: 1 mandante en 2+ formas
- Nivel 3: 2 mandantes en 2 formas cada uno
- Nivel 4: 2+ mandantes en 2+ formas cada uno
- Nivel 5: 3 mandantes en 2+ formas cada uno

**Analysis Output:**
```json
{
  "analysis": "CONTEO DE MANDANTES:\n\n- Gobierno: Se identifican 2 entidades gubernamentales (Ministerio de Trabajo y Autoridades locales) participando activamente.\n  * Formas de participación: diseño del proyecto, reuniones de coordinación, provisión de información (3 formas)\n\n- Empleadores: Se identifica 1 organización patronal (Cámara de Comercio)\n  * Formas de participación: diseño, reuniones, compromisos de implementación (3 formas)\n\n- Trabajadores: Se identifican 2 sindicatos participando\n  * Formas de participación: diseño, reuniones, discusiones de implementación (3 formas)\n\nTOTAL: 3 tipos de mandantes participando en 3 formas diferentes cada uno\n\nJUSTIFICACIÓN DEL PUNTAJE:\nSe identifican los 3 tipos de mandantes (gobierno, empleadores, trabajadores) participando activamente en múltiples formas (diseño, reuniones, implementación). Cada tipo participa en al menos 3 formas diferentes. Esto cumple con el criterio del Nivel 5 que requiere '3 mandantes en 2+ formas cada uno'. Por lo tanto, se asigna la puntuación máxima de 5.",

  "score": 5,

  "evidence": [
    "El Ministerio de Trabajo participó activamente en el diseño del proyecto",
    "Se realizaron reuniones de coordinación con las autoridades locales",
    "La Cámara de Comercio firmó compromisos para la implementación",
    "Los sindicatos participaron en las discusiones sobre diseño",
    "Se documentaron reuniones tripartitas entre gobierno, empleadores y trabajadores",
    "Cada mandante asumió compromisos específicos de co-implementación"
  ]
}
```

## Key Features

1. **Automatic Detection**: No manual configuration needed—automatically detects counting rubrics
2. **Structured Counts**: Presents counts in a clear, verifiable format
3. **Category-based**: Groups stakeholders by the three ILO constituent categories
4. **Transparent Scoring**: Score justification explicitly references the counts
5. **Evidence-based**: Provides specific quotes supporting each stakeholder and participation form
6. **Backwards Compatible**: Non-counting rubrics use the original evaluation logic

## Benefits

1. **Objective Evaluation**: Scores based on verifiable counts, not subjective assessment
2. **Transparent Logic**: Users can verify the counting and scoring
3. **Audit Trail**: Clear evidence of which stakeholders and participation forms were identified
4. **Consistency**: Same counting methodology applied across all evaluations
5. **ILO-Aligned**: Respects the tripartite structure (government, employers, workers)

## When It Activates

The counting logic activates when the criterion/indicator contains keywords related to:
- Stakeholder participation (mandante, stakeholder, actores)
- Actions (participan, participación)
- ILO constituents (gobierno, empleador, trabajador)

For all other rubric indicators, the original evaluation logic is used.

## Testing

To test the counting functionality:
1. Upload a project document to Tab3
2. Select a stakeholder participation indicator (e.g., "1.1 Los mandantes participan...")
3. Run the analysis
4. Verify the output includes:
   - Explicit counts by stakeholder category
   - Forms of participation listed for each
   - Score justification based on the counts
