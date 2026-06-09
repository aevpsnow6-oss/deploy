# ILO PRODOC Quality Appraisal v3 Custom GPT Setup

This guide turns the experimental Tab 1 v3 evaluator into a Custom GPT that ILO personnel can use from ChatGPT.

The correct deployment pattern is:

```text
Custom GPT in ChatGPT
  -> GPT Action
  -> HTTPS API hosted by ILO or an approved provider
  -> tab1_v3_core.py + Rubrica_Tab1_Detallada_Full_v3.xlsx
  -> XLSX result returned to ChatGPT
```

Do not publish this GPT publicly. PRODOCs may contain sensitive project material. Share it only inside the relevant ILO ChatGPT workspace or with named ILO users/groups.

## Files Added

- `tab1_v3_core.py`: Streamlit-free v3 rubric engine.
- `gpt_action_api.py`: FastAPI backend for GPT Actions.
- `openapi_gpt_action_v3.yaml`: schema to import into the Custom GPT editor.
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

## Create the Custom GPT

1. Open `https://chatgpt.com/gpts/editor`.
2. Create a new GPT.
3. Name suggestion: `ILO PRODOC Quality Appraisal v3`.
4. Description suggestion:

   ```text
   Evaluates ILO PRODOC documents against the experimental v3 preliminary project quality appraisal rubric and returns an XLSX results workbook.
   ```

5. Paste the instructions below.
6. In Actions, create a new Action.
7. Authentication:
   - Type: API key.
   - Auth type: Custom header.
   - Header name: `X-API-Key`.
   - Value: the deployed `ILO_GPT_ACTION_API_KEY`.
8. Open `openapi_gpt_action_v3.yaml`.
9. Replace:

   ```yaml
   https://YOUR_ILO_APPROVED_DOMAIN
   ```

   with the real HTTPS API base URL.

10. Import or paste the schema into the GPT Action editor.
11. Test in Preview with one DOCX PRODOC.

## GPT Instructions

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
- The returned workbook has two levels: `Lectura amigable` for ordinary users and `Auditoria tecnica` for reviewers who need the full TEST/DECISIÓN/evidence trail.

Workflow:
1. When the user uploads a DOCX PRODOC, ask whether they want a full evaluation or selected sections/subsections.
2. Start the evaluation job with startV3AppraisalJob.
3. Poll getV3AppraisalJobStatus until status is succeeded or failed.
4. If succeeded, call getV3AppraisalResult.
5. Summarize:
   - total criteria evaluated,
   - counts by result category,
   - failed/error status count if any,
   - high-subjectivity criteria requiring manual review,
   - the downloadable XLSX result, noting that the first sheet is the user-facing view and the second sheet preserves the audit trail.
6. If failed, report the failure message plainly and suggest the narrowest next step.

Response style:
- Be concise and direct.
- Use Spanish by default when the user writes in Spanish; otherwise match the user's language.
- Keep caveats specific: evidence must be manually checked, especially for high-subjectivity criteria.
```

## Test Prompts

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
- The summary counts should match the XLSX.

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

- The pilot API keeps jobs in memory. A restart loses active/completed jobs.
- The pilot API processes one DOCX at a time.
- Full evaluations can trigger up to 76 parallel model calls. Adjust `max_workers` down if the OpenAI project hits rate limits.
- The returned XLSX is embedded in the GPT Action response. If future results exceed GPT Action file limits, switch to returning a short-lived download URL.
