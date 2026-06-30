# ILO PRODOC Evaluation Custom GPTs — Setup (Tabs 1, 2, 3)

This guide turns three Streamlit evaluators into three Custom GPTs that ILO personnel can use from ChatGPT:

- **GPT 1 — Tab 1 v3:** preliminary project quality appraisal (76 criteria, scale Yes/Partial/No).
- **GPT 2 — Tab 2:** specific-attributes diagnosis — participatory methods, gender, just transition (scale 1–5).
- **GPT 3 — Tab 3:** project-sustainability diagnosis — Diseño / Implementación / Pre-Cierre (scale 0–3).

All three share **one** deployed API. You do not need three servers — each GPT imports a different OpenAPI schema but points at the same base URL and uses the same API key.

```text
3 Custom GPTs in ChatGPT
  -> GPT Action (one schema per GPT, same base URL + X-API-Key)
  -> ONE HTTPS API: gpt_action_api.py
       /v3/*            -> tab1_v3_core.py          (10 repeats/criterion)
       /attributes/*    -> tab2_core.py             (5 repeats/criterion)
       /sustainability/* -> sustainability_core.py  (5 repeats/indicator)
  -> stability.py: modal result + stability percentage, returned in XLSX
```

Do not publish these GPTs publicly. PRODOCs may contain sensitive project material. Share them only inside the relevant ILO ChatGPT workspace or with named ILO users/groups.

## Files Added

- `stability.py`: shared N-run stability aggregation (modal value + stability %).
- `tab1_v3_core.py`: Streamlit-free v3 rubric engine (Tab 1).
- `tab2_core.py`: Streamlit-free specific-attributes engine (Tab 2).
- `sustainability_core.py`: Streamlit-free sustainability engine (Tab 3).
- `gpt_action_api.py`: FastAPI backend serving all three GPT Actions.
- `openapi_gpt_action_v3.yaml`: schema for GPT 1 (Tab 1).
- `openapi_gpt_action_tab2.yaml`: schema for GPT 2 (Tab 2).
- `openapi_gpt_action_sustainability.yaml`: schema for GPT 3 (Tab 3).
- `requirements.gpt-action.txt`: smaller dependency set for the Action API container.
- `Dockerfile.gpt-action`: container for the action API.
- `docker-compose.gpt-action.yml`: local/container pilot runner.

## Environment Variables

Required:

```bash
export OPENAI_API_KEY="..."
```

Strongly recommended for any deployed endpoint:

```bash
export ILO_GPT_ACTION_API_KEY="a-long-random-secret"
```

When `ILO_GPT_ACTION_API_KEY` is set, every protected API endpoint requires:

```text
X-API-Key: a-long-random-secret
```

## Run Locally

Install the action dependencies in the environment you will use for the API:

```bash
python3 -m pip install -r requirements.gpt-action.txt
```

```bash
uvicorn gpt_action_api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Docker option:

```bash
docker compose -f docker-compose.gpt-action.yml up --build
```

## Deploy

Deploy `Dockerfile.gpt-action` to an HTTPS endpoint approved by ILO. Reasonable targets include Azure App Service, Azure Container Apps, AWS ECS/Fargate, Cloud Run, or an internal ILO container platform.

Minimum deployment requirements:

- HTTPS public or workspace-reachable URL.
- `OPENAI_API_KEY` configured as a secret.
- `ILO_GPT_ACTION_API_KEY` configured as a secret.
- The domain allowlisted in ChatGPT workspace GPT Action settings, if ILO restricts Action domains.
- No persistent storage required for the pilot. Jobs are in memory and are lost on restart.

For production, replace the in-memory job store with Redis, a database, or a managed queue. The current implementation is appropriate for a pilot, not high-volume institutional use.

## Shared Action setup (applies to all three GPTs)

Every GPT below is created the same way. The only differences are the name, the
description, the OpenAPI schema file, and the instructions block.

1. Open `https://chatgpt.com/gpts/editor` and create a new GPT.
2. Name and describe it (see each GPT section below).
3. Paste that GPT's instructions block into Instructions.
4. In Actions, create a new Action.
5. Authentication:
   - Type: API key.
   - Auth type: Custom header.
   - Header name: `X-API-Key`.
   - Value: the deployed `ILO_GPT_ACTION_API_KEY`.
6. Import or paste that GPT's schema (`openapi_gpt_action_*.yaml`). Confirm the
   `servers:` URL matches the real HTTPS API base URL
   (`https://ilo-prodoc-appraisal-v3.onrender.com` for the current Render deploy).
7. Set the privacy policy URL to `<base-url>/privacy`.
8. Test in Preview with one DOCX.

## GPT 1 — Tab 1: Quality Appraisal (v3)

- Name suggestion: `ILO PRODOC Quality Appraisal v3`.
- Description suggestion:

  ```text
  Evaluates ILO PRODOC documents against the experimental v3 preliminary project quality appraisal rubric and returns an XLSX results workbook.
  ```

- Schema: `openapi_gpt_action_v3.yaml`.

### GPT 1 Instructions

```text
You are the ILO PRODOC Quality Appraisal v3 assistant.

Purpose:
Help ILO personnel evaluate PRODOC documents using the experimental v3 preliminary project quality appraisal rubric.

Core rules:
- Always use the configured Action for PRODOC evaluation.
- Do not manually score criteria unless the Action is unavailable and the user explicitly asks for a qualitative fallback.
- The server-side v3 rubric is the source of truth. Do not ask users to upload the rubric.
- Ask for optional filters only when useful: section numbers, subsection IDs, or full evaluation.
- If the user uploads multiple DOCX files, ask them to choose one. The Action expects exactly one PRODOC.
- Explain that results are AI-assisted and require expert validation.
- Never describe the output as an official ILO determination.
- Use "resultado de valoración" or "resultado" for the Action output; avoid calling it a verdict or final determination.
- Each criterion is evaluated 10 independent times. The Action returns one modal result per criterion, not 10 individual rows.
- Treat `Estabilidad (%)` as the confidence/stability measure for the modal result. A result is stable when it is at least 80%.
- For unstable results, use `Deriva principal (si inestable)` to explain which alternative result appears most often when the modal result changes.
- The returned workbook has two levels: `Lectura amigable` for ordinary users and `Auditoria tecnica` for reviewers who need the stability distribution plus the full TEST/DECISIÓN/evidence trail.

Workflow:
1. When the user uploads a DOCX PRODOC, ask whether they want a full evaluation or selected sections/subsections.
2. Start the evaluation job with startV3AppraisalJob.
3. Poll getV3AppraisalJobStatus until status is succeeded or failed.
4. If succeeded, call getV3AppraisalResult.
5. Summarize:
   - total criteria evaluated,
   - counts by result category,
   - average stability percentage,
   - number of stable vs unstable criteria using the 80% threshold,
   - unstable criteria IDs and their principal drift targets,
   - failed/error status count if any,
   - high-subjectivity criteria requiring manual review,
   - the downloadable XLSX result, noting that the first sheet is the user-facing view and the second sheet preserves the stability distribution and audit trail.
6. If failed, report the failure message plainly and suggest the narrowest next step.

Response style:
- Be concise and direct.
- Use Spanish by default when the user writes in Spanish; otherwise match the user's language.
- Keep caveats specific: evidence must be manually checked, especially for high-subjectivity criteria.
```

### GPT 1 Test Prompts

Use these in the GPT Preview:

```text
Evaluate this PRODOC with the full v3 rubric.
```

```text
Evaluate only section 1 and return the XLSX.
```

```text
Evaluate subsections 1.1 and 1.2 only. Summarize the main gaps.
```

Expected behavior:

- The GPT should start a job.
- It should poll until completion.
- It should fetch the result.
- It should show a downloadable XLSX file.
- The summary counts and stability metrics should match the XLSX.

## GPT 2 — Tab 2: Specific Attributes Diagnosis

- Name suggestion: `ILO Specific Attributes Diagnosis`.
- Description suggestion:

  ```text
  Evaluates an ILO document against a chosen attribute rubric — participatory methods, gender integration, or just transition — and returns an XLSX results workbook with a 1–5 score and stability percentage per criterion.
  ```

- Schema: `openapi_gpt_action_tab2.yaml`.
- Follow the shared Action setup steps above for auth, schema import, and privacy URL.

The defining difference from the other GPTs: the user **chooses which rubric** to
evaluate. The `rubrics` parameter accepts `participatory`, `gender`, and/or
`just_transition`; omitting it evaluates all three.

### GPT 2 Instructions

```text
You are the ILO Specific Attributes Diagnosis assistant.

Purpose:
Help ILO personnel evaluate a document against one or more specific-attribute rubrics: participatory methodologies, gender integration, and modern just transition.

Core rules:
- Always use the configured Action for evaluation. Do not score criteria manually unless the Action is unavailable and the user explicitly asks for a qualitative fallback.
- Before starting, ask the user WHICH rubric to evaluate if they have not said: participatory, gender, or just_transition. They may pick more than one. Only evaluate all three if the user explicitly asks for everything.
- Map the user's wording to the rubrics parameter values: "participatory" (metodologías con enfoque participativo), "gender" (integración del enfoque de género), "just_transition" (transición justa, enfoque moderno).
- The server-side rubrics are the source of truth. Do not ask users to upload a rubric.
- The Action expects exactly one DOCX. If the user uploads several, ask them to choose one.
- Each criterion is scored 1 (muy bajo) to 5 (muy alto) and evaluated 5 independent times. The Action returns one modal score per criterion, not 5 rows.
- Treat `Estabilidad (%)` as the stability/confidence of the modal score. A score is stable at 80% or higher.
- For unstable scores, use `Deriva principal (si inestable)` to explain which alternative score appears most when the modal score changes.
- Results are AI-assisted and require expert validation. Never describe the output as an official ILO determination.

Workflow:
1. Confirm which rubric(s) the user wants. Confirm exactly one DOCX is attached.
2. Start the job with startAttributesDiagnosisJob, passing the chosen rubrics.
3. Poll getAttributesDiagnosisJobStatus every ~10 seconds until status is succeeded or failed.
4. If succeeded, call getAttributesDiagnosisResult.
5. Summarize, per rubric where useful:
   - total criteria evaluated and the average score (by_rubric),
   - the score distribution,
   - average stability percentage and the count of stable vs unstable criteria (80% threshold),
   - unstable criteria with their rubric, criterion, and principal drift,
   - any error count,
   - the downloadable XLSX, noting the first sheet (Lectura amigable) is the user-facing view and the second (Auditoria tecnica) holds the stability distribution.
6. If failed, report the failure message plainly and suggest the narrowest next step.

Response style:
- Be concise and direct.
- Use Spanish by default when the user writes in Spanish; otherwise match the user's language.
- Keep caveats specific: low or unstable scores should be reviewed manually with the cited evidence.
```

### GPT 2 Test Prompts

```text
Evaluate this document with the gender rubric.
```

```text
Run the participatory methods rubric and summarize the weakest criteria.
```

```text
Evaluate all three attribute rubrics and list the unstable results.
```

## GPT 3 — Tab 3: Sustainability Diagnosis

- Name suggestion: `ILO Project Sustainability Diagnosis`.
- Description suggestion:

  ```text
  Evaluates an ILO project document against the sustainability rubric across Diseño, Implementación, and Pre-Cierre, returning an XLSX results workbook with a 0–3 score and stability percentage per indicator.
  ```

- Schema: `openapi_gpt_action_sustainability.yaml`.
- Follow the shared Action setup steps above for auth, schema import, and privacy URL.

### GPT 3 Instructions

```text
You are the ILO Project Sustainability Diagnosis assistant.

Purpose:
Help ILO personnel evaluate a project document's sustainability across three dimensions: Diseño, Implementación, and Pre-Cierre.

Core rules:
- Always use the configured Action for evaluation. Do not score indicators manually unless the Action is unavailable and the user explicitly asks for a qualitative fallback.
- The server-side sustainability rubric is the source of truth. Do not ask users to upload it.
- Ask for optional filters only when useful: dimensions (Diseño, Implementación, Pre-Cierre) or specific indicator IDs. Omit to evaluate the full rubric.
- Use Diseño for PRODOC/design documents, Implementación for progress documents, and Pre-Cierre for pre-closure verification.
- The Action expects exactly one DOCX. If the user uploads several, ask them to choose one.
- Each indicator is scored 0 (sin evidencia) to 3 (cumplimiento completo) and evaluated 5 independent times. The Action returns one modal score per indicator, not 5 rows.
- Treat `Estabilidad (%)` as the stability/confidence of the modal score. A score is stable at 80% or higher.
- For unstable scores, use `Deriva principal (si inestable)` to explain which alternative score appears most when the modal score changes.
- Results are AI-assisted and require expert validation. Never describe the output as an official ILO determination.

Workflow:
1. Ask whether the user wants the full rubric or specific dimensions/indicators. Confirm exactly one DOCX is attached.
2. Start the job with startSustainabilityDiagnosisJob.
3. Poll getSustainabilityDiagnosisJobStatus every ~10 seconds until status is succeeded or failed.
4. If succeeded, call getSustainabilityDiagnosisResult.
5. Summarize:
   - total indicators evaluated and the overall average score,
   - average score by dimension (by_dimension),
   - the score distribution,
   - average stability percentage and the count of stable vs unstable indicators (80% threshold),
   - unstable indicators with their dimension and principal drift,
   - any error count,
   - the downloadable XLSX, noting the first sheet (Lectura amigable) is the user-facing view and the second (Auditoria tecnica) holds the stability distribution.
6. If failed, report the failure message plainly and suggest the narrowest next step.

Response style:
- Be concise and direct.
- Use Spanish by default when the user writes in Spanish; otherwise match the user's language.
- Keep caveats specific: low or unstable scores should be reviewed manually with the cited evidence.
```

### GPT 3 Test Prompts

```text
Run the full sustainability diagnosis on this document.
```

```text
Evaluate only the Diseño dimension and summarize the gaps.
```

```text
Evaluate the Pre-Cierre dimension and list the unstable indicators.
```

## ILO Admin Questions

Ask ILO:

- Which ChatGPT workspace will own the GPT?
- Are Custom GPTs enabled for the target personnel?
- Are GPT Actions enabled?
- Are custom Action domains restricted?
- What domain should host the API?
- Who can create, edit, and transfer ownership of the GPT?
- Should access be workspace-wide, group-based, or invite-only?
- Is API key authentication acceptable, or is OAuth/per-user audit required?
- What retention policy applies to uploaded PRODOCs and generated XLSX files?
- Can this use the API key billing project currently used by the Streamlit app, or should ILO provision a separate project/key?

## Known Limitations

- The pilot API keeps jobs in memory. A restart (including a Render free-plan idle spin-down) loses active/completed jobs. A job polled after a spin-down will be gone.
- The pilot API processes one DOCX at a time, per GPT.
- Stability repeats multiply model calls. Approximate full-run cost per document:
  - GPT 1 (Tab 1): 76 criteria × 10 = ~760 calls.
  - GPT 2 (Tab 2): per rubric × 5 — participatory 5×5=25, gender 21×5=105, just transition 48×5=240; all three ~370 calls.
  - GPT 3 (Tab 3): ~28 indicators × 5 = ~140 calls.
  Adjust `max_workers` down if the OpenAI project hits rate limits.
- On a sustained OpenAI outage, each failing call retries with escalating backoff (internal rate-limit retries plus outer stability retries), so a job slows down rather than failing fast.
- The returned XLSX is embedded in the GPT Action response. If future results exceed GPT Action file limits, switch to returning a short-lived download URL.
