# Two-Part Question Implementation

## Overview
Enhanced the LLM analysis functions in `oli_v6_deploy.py` to properly handle two-part rubric questions where:
- **Part 1 (Broader Context)**: Sets the general framework
- **Part 2 (Specific Focus - PRIMARY)**: The critical question requiring detailed analysis

## Implementation Strategy
- **Tab1 (Appraisal Rubric)**: Uses specialized functions that **explicitly state** when 2 parts are detected
- **Other Tabs**: Use standard functions with intelligent two-part handling (no explicit statement)

## Changes Made

### 1. New Function: `parse_two_part_question()` (line 5446)
Automatically detects and parses two-part questions using two patterns:

**Pattern 1 (Priority)**: Questions with keyword separators
- Keywords: "específicamente", "en particular", "en concreto", "puntualmente", "además"
- Example: `¿Se definen indicadores? Específicamente, ¿se incluyen metas cuantificables?`
- Result:
  - Part 1: `¿Se definen indicadores?`
  - Part 2: `¿se incluyen metas cuantificables?`

**Pattern 2**: Questions run together without separators
- Example: `¿La teoría del cambio está clara?Se identifican los supuestos?`
- Result:
  - Part 1: `¿La teoría del cambio está clara?`
  - Part 2: `¿Se identifican los supuestos?`

### 2. TAB1-SPECIFIC: `analyze_question_with_llm_tab1()` (line 5501)
**Used exclusively in Tab1 (Appraisal Rubric)** - explicitly states when 2 parts are detected:

**For Two-Part Questions:**
- **MANDATORY OPENING**: "Se identificaron 2 partes en esta pregunta."
- Instructs LLM to **prioritize Part 2** (70% focus)
- Clearly separates the two parts in the prompt
- Required structure: "Se identificaron 2 partes en esta pregunta. [Part 2 detailed analysis]. Esto [afecta/se relaciona con] [Part 1] porque [explanation]"

**For Single Questions:**
- Uses standard prompt (no special handling)

### 3. STANDARD: `analyze_question_with_llm()` (line 5635)
**Used in Tab2, Tab3, Tab4** - intelligent handling without explicit statement:

**For Two-Part Questions:**
- Instructs LLM to **prioritize Part 2** (70% focus)
- Clearly separates the two parts in the prompt
- Requires analysis to start with Part 2, then connect to Part 1
- Example structure: "Respecto a [Part 2]: [detailed analysis]... Esto afecta [Part 1] porque [explanation]"

**For Single Questions:**
- Uses original prompt (unchanged behavior)

### 4. TAB1-SPECIFIC: `analyze_question_with_critical_opinion_tab1()` (line 5763)
**Used exclusively in Tab1** - checks for proper two-part acknowledgment:

**Critical Evaluation Includes:**
1. **FIRST**: Does the reasoning BEGIN with "Se identificaron 2 partes" acknowledgment?
   - If NO: This is flagged as a CRITICAL FAILURE
2. Does the answer adequately address Part 2 (specific question)?
3. Does it address Part 1 (broader context)?
4. Is there coherence between Part 2 and Part 1?

### 5. STANDARD: `analyze_question_with_critical_opinion()` (line 5895)
Critical evaluation now assesses both parts with proper priority:

**Evaluation Criteria:**
1. **PRIMARY**: Does the answer adequately address Part 2 (specific question)?
2. **SECONDARY**: Does it address Part 1 (broader context)?
3. **INTEGRATION**: Is there coherence between Part 2 findings and Part 1 context?

**Extra Critical When:**
- Part 2 is answered superficially
- Evidence focuses on Part 1 but neglects Part 2
- Reasoning doesn't connect Part 2 to Part 1

## Example Question
```
6.1 ¿La teoría del cambio está claramente expresada de manera plausible
para el no especialista, en la narrativa y/o como gráfico?Se identifican
los supuestos clave, incluidos los resultados de intervenciones pasadas,
en curso o planificadas?
```

**Parsed Result:**
- **Part 1 (Broader)**: "¿La teoría del cambio está claramente expresada...?"
- **Part 2 (Specific - PRIMARY)**: "¿Se identifican los supuestos clave...?"

### In TAB1 (Appraisal Rubric):

**Razonamiento Field Will Start With:**
```
Se identificaron 2 partes en esta pregunta. [Detailed analysis of whether key assumptions
are identified - Part 2]. Esto afecta la teoría del cambio [Part 1] porque [explanation
of how Part 2 findings relate to Part 1].
```

**Evaluación Crítica Will Check:**
- ✓ Does the reasoning begin with "Se identificaron 2 partes"?
- ✓ Is Part 2 (assumptions) analyzed thoroughly?
- ✓ Is Part 1 (theory of change) addressed in context?
- ✓ Is there clear connection between the two parts?

### In Other Tabs (Tab2, Tab3, Tab4):

**LLM Will (without explicit statement):**
1. First analyze whether key assumptions are identified (Part 2)
2. Then explain how this affects the clarity of the theory of change (Part 1)
3. Provide evidence primarily for Part 2, then supporting evidence for Part 1

## Testing
Run `test_two_part_parsing.py` to verify the parsing logic works correctly.

## Where Changes Are Applied

### Tab1 (Valoración Preliminar - APPRAISAL Rubric)
✅ **Uses tab1-specific functions** (`analyze_question_with_llm_tab1`, `analyze_question_with_critical_opinion_tab1`)
- Line 6785: Calls `analyze_question_with_llm_tab1`
- Line 6809: Calls `analyze_question_with_critical_opinion_tab1`
- **Behavior**: Explicitly states "Se identificaron 2 partes en esta pregunta" when detected
- **Critical Eval**: Checks if the explicit acknowledgment is present

### Other Tabs (Tab2, Tab3, Tab4)
✅ **Uses standard functions** (`analyze_question_with_llm`, `analyze_question_with_critical_opinion`)
- **Behavior**: Intelligently handles two-part questions without explicit statement
- **Critical Eval**: Assesses quality of Part 2 focus and Part 1 connection

## Benefits

### For Tab1 (Appraisal Rubric):
1. **Explicit recognition**: Users clearly see when a question has multiple parts
2. **Transparent analysis**: The "Se identificaron 2 partes" statement provides clarity
3. **Quality control**: Critical evaluation verifies proper structure recognition
4. **More focused analysis**: LLM concentrates on the specific question (Part 2) that really matters
5. **Better evidence**: Evidence is weighted 70% toward the critical aspect (Part 2)
6. **Contextual understanding**: Analysis still considers how the specific relates to the broader

### For All Tabs:
1. **Automatic detection**: No manual marking needed - works automatically on all rubric questions
2. **Backwards compatible**: Single-part questions work exactly as before
3. **No rubric changes**: Excel files remain unchanged

## No Changes Required
The implementation is automatic - no changes needed to:
- Rubric Excel files
- Question formats
- User interface

Simply run the analysis as usual:
- **Tab1**: Two-part questions will include the explicit statement
- **Other tabs**: Two-part questions will be intelligently prioritized without explicit statement
