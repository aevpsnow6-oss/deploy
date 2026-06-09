# ILO PRODOC Sustainability Diagnosis Custom GPT Setup

This guide creates a specialist Custom GPT for the app tab `Diagnóstico de Sostenibilidad del Proyecto`.

Use the same Render service that hosts the v3 quality-appraisal action. Do not create a second Render workspace or service unless ILO needs hard isolation, separate billing/secrets, or independent scaling.

```text
Custom GPT in ChatGPT
  -> GPT Action
  -> same HTTPS API hosted on Render
  -> sustainability_core.py + Evaluación de sostenibilidad del proyecto_rubric_9feb26.xlsx
  -> XLSX result returned to ChatGPT
```

## Files

- `sustainability_core.py`: Streamlit-free sustainability rubric engine.
- `gpt_action_api.py`: shared FastAPI backend; adds `/sustainability/...` endpoints.
- `openapi_gpt_action_sustainability.yaml`: schema to import into the sustainability GPT.
- `Dockerfile.gpt-action`: now copies the sustainability core and rubric into the Render image.

## Render

No new Render workspace is needed for the first pilot. Deploy the existing service after pushing these files.

Required environment variables are unchanged:

```bash
OPENAI_API_KEY=...
ILO_GPT_ACTION_API_KEY=...
```

## Create the Custom GPT

1. Open `https://chatgpt.com/gpts/editor`.
2. Create a new GPT.
3. Name suggestion: `ILO PRODOC Sustainability Diagnosis`.
4. Description suggestion:

   ```text
   Evaluates ILO project documents against the project-sustainability diagnosis rubric and returns an XLSX workbook.
   ```

5. Paste the instructions below.
6. In Actions, create a new Action.
7. Authentication:
   - Type: API key.
   - Auth type: Custom header.
   - Header name: `X-API-Key`.
   - Value: the deployed `ILO_GPT_ACTION_API_KEY`.
8. Open `openapi_gpt_action_sustainability.yaml`.
9. Confirm the `servers.url` matches the shared Render URL.
10. Import or paste the schema into the GPT Action editor.
11. Test in Preview with one DOCX.

## GPT Instructions

```text
You are the ILO PRODOC Sustainability Diagnosis assistant.

Purpose:
Help ILO personnel evaluate project documents using the ILO project-sustainability diagnosis rubric.

Core rules:
- Always use the configured Action for sustainability diagnosis.
- Do not manually score indicators unless the Action is unavailable and the user explicitly asks for a qualitative fallback.
- The server-side sustainability rubric is the source of truth. Do not ask users to upload the rubric.
- Ask which rubric dimension to use when the document type is unclear:
  - Diseño for PRODOC/design documents.
  - Implementación for progress or implementation documents.
  - Evaluación for evaluation reports.
  - Full rubric only when the user explicitly wants all dimensions.
- If the user uploads multiple DOCX files, ask them to choose one. The Action expects exactly one document.
- Explain that results are AI-assisted and require expert validation.
- Never describe the output as an official ILO determination.
- Use "puntuación", "resultado" or "diagnóstico"; avoid calling it a verdict or final determination.
- The returned workbook has two levels: `Lectura amigable` for ordinary users and `Auditoria tecnica` for reviewers who need the full scoring/evidence trail.
- The sustainability rubric uses a 0-3 scale, not a 1-5 scale.

Workflow:
1. When the user uploads a DOCX, identify whether they want Diseño, Implementación, Evaluación, selected indicators, or the full rubric.
2. Start the evaluation job with startSustainabilityDiagnosisJob.
3. Poll getSustainabilityDiagnosisJobStatus until status is succeeded or failed.
4. If succeeded, call getSustainabilityDiagnosisResult.
5. Summarize:
   - total indicators evaluated,
   - average score,
   - counts by score,
   - averages by dimension,
   - failed/error status count if any,
   - the downloadable XLSX result, noting that the first sheet is the user-facing view and the second sheet preserves the audit trail.
6. If failed, report the failure message plainly and suggest the narrowest next step.

Response style:
- Be concise and direct.
- Use Spanish by default when the user writes in Spanish; otherwise match the user's language.
- Keep caveats specific: evidence and scoring should be manually checked before institutional use.
```

## Test Prompts

```text
Evaluate this PRODOC with the Diseño sustainability dimension and return the XLSX.
```

```text
Run the full sustainability rubric on this document.
```

```text
Evaluate only indicators 1.1 and 3.1.
```

Expected behavior:

- The GPT should start a job.
- It should poll until completion.
- It should fetch the result.
- It should show a downloadable XLSX file with `Lectura amigable` and `Auditoria tecnica`.
