"""FastAPI backend for ILO PRODOC Custom GPT actions.

The API follows the GPT Actions file-passing convention: ChatGPT sends uploaded
files in a JSON field named `openaiFileIdRefs`, each with a short-lived
download_link. Results are returned with `openaiFileResponse` so ChatGPT exposes
the generated XLSX to the user.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import secrets
import threading
import time
import uuid
import urllib.request
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field

import sustainability_core
import tab1_v3_core as v3_core
import tab2_core

APP_TITLE = "ILO PRODOC Evaluation Action API"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL", "https://ilo-prodoc-appraisal-v3.onrender.com"
).rstrip("/")

app = FastAPI(
    title=APP_TITLE,
    version="0.1.0",
    description=(
        "Runs ILO PRODOC evaluation rubrics and returns XLSX result files for "
        "Custom GPT Actions."
    ),
)

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


class OpenAIFileRef(BaseModel):
    name: str = Field(..., description="Original file name shown in ChatGPT.")
    id: str | None = Field(None, description="Stable ChatGPT file id.")
    mime_type: str | None = Field(None, description="MIME type inferred by ChatGPT.")
    download_link: str = Field(..., description="Short-lived URL for downloading the file.")


class EvaluationJobRequest(BaseModel):
    openaiFileIdRefs: list[OpenAIFileRef] = Field(
        ...,
        description=(
            "Exactly one DOCX PRODOC uploaded by the user. ChatGPT populates "
            "this from the conversation file attachment."
        ),
        min_length=1,
    )
    sections: list[int] | None = Field(
        None,
        description="Optional rubric section numbers to evaluate, e.g. [1, 2].",
    )
    subsections: list[str] | None = Field(
        None,
        description="Optional rubric subsection IDs to evaluate, e.g. ['1.1', '2.3'].",
    )
    max_workers: int = Field(
        v3_core.MAX_WORKERS,
        ge=1,
        le=v3_core.MAX_WORKERS,
        description="Parallel OpenAI calls for criterion evaluation.",
    )


class SustainabilityJobRequest(BaseModel):
    openaiFileIdRefs: list[OpenAIFileRef] = Field(
        ...,
        description=(
            "Exactly one DOCX project document uploaded by the user. ChatGPT "
            "populates this from the conversation file attachment."
        ),
        min_length=1,
    )
    dimensions: list[str] | None = Field(
        None,
        description=(
            "Optional sustainability rubric dimensions to evaluate. Supported "
            "values are Diseño, Implementación, and Pre-Cierre."
        ),
    )
    indicators: list[str] | None = Field(
        None,
        description="Optional indicator IDs to evaluate, e.g. ['1.1', '3.1'].",
    )
    max_workers: int = Field(
        sustainability_core.MAX_WORKERS,
        ge=1,
        le=sustainability_core.MAX_WORKERS,
        description="Parallel OpenAI calls for sustainability indicator evaluation.",
    )


class AttributesJobRequest(BaseModel):
    openaiFileIdRefs: list[OpenAIFileRef] = Field(
        ...,
        description=(
            "Exactly one DOCX document uploaded by the user. ChatGPT populates "
            "this from the conversation file attachment."
        ),
        min_length=1,
    )
    rubrics: list[str] | None = Field(
        None,
        description=(
            "Which rubric(s) to evaluate. Supported values are 'participatory', "
            "'gender', and 'just_transition'. Omit to evaluate all three."
        ),
    )
    max_workers: int = Field(
        tab2_core.MAX_WORKERS,
        ge=1,
        le=tab2_core.MAX_WORKERS,
        description="Parallel OpenAI calls for repeated criterion evaluation.",
    )


class JobCreated(BaseModel):
    job_id: str
    status: str
    message: str
    # El POST prepara el documento de forma síncrona para que el GPT pueda
    # confirmar la recepción antes de terminar el turno inicial.
    start_line: str | None = None
    document_name: str | None = None
    document_word_count: int | None = None
    criteria_count: int | None = None
    total: int | None = None
    estimated_seconds: int | None = None


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Protect deployed endpoints when ILO_GPT_ACTION_API_KEY is configured."""
    expected = os.getenv("ILO_GPT_ACTION_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


def _now() -> float:
    return time.time()


def _set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = _now()


def _get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return dict(job)


# Cada mensaje de estado hace una única consulta. La espera corta permite que esa
# consulta capture avances recientes sin mantener el turno del usuario bloqueado.
POLL_MAX_WAIT_SECONDS = 25.0
POLL_DEFAULT_WAIT_SECONDS = 3.0
_TERMINAL_STATUSES = {"succeeded", "failed"}


async def _await_job(job_id: str, wait_seconds: float) -> dict[str, Any]:
    """Return as soon as the job reaches a terminal state, or when time runs out."""
    job = _get_job(job_id)
    budget = min(max(wait_seconds, 0.0), POLL_MAX_WAIT_SECONDS)
    deadline = time.monotonic() + budget
    while job.get("status") not in _TERMINAL_STATUSES and time.monotonic() < deadline:
        await asyncio.sleep(1.0)
        job = _get_job(job_id)
    return job


def _eta_text(seconds: int | None) -> str:
    if not seconds:
        return ""
    if seconds < 60:
        return " Queda menos de un minuto."
    return f" Quedan unos {max(1, round(seconds / 60))} minutos."


def _progress_line(job: dict[str, Any], completed: int, total: int,
                   percent: float | None, eta_seconds: int | None) -> str:
    """A sentence the GPT can relay verbatim, so it never has to invent one."""
    status = job.get("status")
    if status == "failed":
        return f"La evaluación falló: {job.get('error') or job.get('message') or 'error desconocido'}"
    if status == "succeeded":
        return "Evaluación completada. Preparando el archivo de resultados."

    # El encabezado del documento sólo en la primera línea informativa: repetirlo
    # en cada sondeo convierte el avance en ruido.
    doc = job.get("document_name")
    palabras = job.get("document_word_count")
    encabezado = ""
    if completed == 0 and doc:
        encabezado = (f"Documento recibido: {doc} ({palabras:,} palabras). ".replace(",", ".")
                      if palabras else f"Documento recibido: {doc}. ")

    if total and completed < total:
        pct = f" ({round(percent)}%)" if percent is not None else ""
        return (f"{encabezado}Evaluando: {completed} de {total} consultas"
                f"{pct}.{_eta_text(eta_seconds)}")
    if total:
        return f"{encabezado}Consolidando resultados ({total} de {total} consultas)."
    return f"{encabezado}{job.get('message') or 'Preparando la evaluación…'}"


def _job_status_payload(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    """Status plus the numbers the GPT needs to narrate progress in one line."""
    completed = int(job.get("completed") or 0)
    total = int(job.get("total") or 0)
    percent = round(100.0 * completed / total, 1) if total else None

    eta_seconds = None
    started = job.get("created_at")
    if started and completed and total and completed < total:
        elapsed = max(_now() - float(started), 0.001)
        rate = completed / elapsed
        if rate > 0:
            eta_seconds = int(round((total - completed) / rate))

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "message": job.get("message"),
        "progress_line": _progress_line(job, completed, total, percent, eta_seconds),
        "completed": completed,
        "total": total,
        "percent_complete": percent,
        "eta_seconds": eta_seconds,
        "document_name": job.get("document_name"),
        "document_word_count": job.get("document_word_count"),
        "error": job.get("error"),
    }


def _is_docx(ref: OpenAIFileRef) -> bool:
    name = ref.name.lower()
    mime = (ref.mime_type or "").lower()
    return name.endswith(".docx") or mime.endswith(
        "vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def _download_docx_file(file_refs: list[OpenAIFileRef]) -> tuple[str, bytes]:
    docx_refs = [ref for ref in file_refs if _is_docx(ref)]
    if len(docx_refs) != 1:
        raise ValueError(
            f"Expected exactly one DOCX PRODOC file; received {len(docx_refs)} DOCX files."
        )

    ref = docx_refs[0]
    request = urllib.request.Request(ref.download_link, headers={"User-Agent": "ilo-gpt-action/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - URL comes from ChatGPT file ref
        content = response.read()
    if not content:
        raise ValueError("Downloaded DOCX file is empty.")
    return ref.name, content


def _safe_filename_stem(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return stem or "prodoc"


# Ritmo observado en producción: ~48 llamadas en paralelo, ~1.6 s por llamada.
SECONDS_PER_CALL = 0.30


def _estimate_seconds(total_calls: int) -> int:
    return max(15, int(round(total_calls * SECONDS_PER_CALL)))


def _prepare_v3_job(request: EvaluationJobRequest) -> dict[str, Any]:
    """Download, extract and filter before the POST returns.

    Doing this synchronously costs a few seconds on the POST but lets the GPT
    tell the user what it received and how long it will take, instead of going
    silent while a background thread works.
    """
    filename, docx_bytes = _download_docx_file(request.openaiFileIdRefs)
    document_text = v3_core.extract_docx_text_from_bytes(docx_bytes)
    word_count = len(document_text.split())
    rubric = v3_core.load_rubrica_v3()
    criteria = v3_core.filter_rubric(rubric, request.sections, request.subsections)
    if criteria.empty:
        raise ValueError("No v3 rubric criteria matched the selected filters.")
    total_calls = len(criteria) * v3_core.STABILITY_REPEATS
    return {
        "filename": filename,
        "document_text": document_text,
        "word_count": word_count,
        "criteria": criteria,
        "total_calls": total_calls,
        "estimated_seconds": _estimate_seconds(total_calls),
    }


def _run_evaluation_job(
    job_id: str, request: EvaluationJobRequest, prepared: dict[str, Any]
) -> None:
    try:
        filename = prepared["filename"]
        document_text = prepared["document_text"]
        word_count = prepared["word_count"]
        criteria = prepared["criteria"]

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured on the API server.")
        client = OpenAI(api_key=openai_api_key)

        def progress(done: int, total: int) -> None:
            _set_job(
                job_id,
                completed=done,
                total=total,
                message=f"Completed {done}/{total} repeated criterion calls.",
            )

        total_calls = prepared["total_calls"]
        _set_job(
            job_id,
            status="running",
            message=(
                "Evaluating criteria with OpenAI "
                f"({v3_core.STABILITY_REPEATS} independent runs per criterion)."
            ),
            completed=0,
            total=total_calls,
            document_name=filename,
            document_word_count=word_count,
        )
        results = v3_core.evaluate_criteria(
            client,
            criteria,
            document_text,
            max_workers=request.max_workers,
            progress_callback=progress,
        )
        summary = v3_core.summarize_results(results)
        xlsx_bytes = v3_core.results_to_xlsx_bytes(results, criteria)
        xlsx_name = f"valoracion_v3_{_safe_filename_stem(filename)}.xlsx"

        _set_job(
            job_id,
            status="succeeded",
            message="Evaluation complete.",
            completed=total_calls,
            total=total_calls,
            summary=summary,
            result_filename=xlsx_name,
            result_xlsx_bytes=xlsx_bytes,
            download_token=secrets.token_urlsafe(32),
        )
    except Exception as exc:  # noqa: BLE001 - report job failures to the GPT
        _set_job(job_id, status="failed", message=str(exc), error=str(exc))


def _run_sustainability_job(job_id: str, request: SustainabilityJobRequest) -> None:
    try:
        _set_job(job_id, status="running", message="Downloading DOCX from ChatGPT.")
        filename, docx_bytes = _download_docx_file(request.openaiFileIdRefs)

        _set_job(job_id, message="Extracting DOCX text.")
        document_text = sustainability_core.extract_docx_text_from_bytes(docx_bytes)
        word_count = len(document_text.split())

        _set_job(job_id, message="Loading and filtering sustainability rubric.")
        rubric = sustainability_core.load_sustainability_rubric()
        indicators = sustainability_core.filter_rubric(
            rubric,
            dimensions=request.dimensions,
            indicators=request.indicators,
        )
        if indicators.empty:
            raise ValueError("No sustainability rubric indicators matched the selected filters.")

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured on the API server.")
        client = OpenAI(api_key=openai_api_key)

        def progress(done: int, total: int) -> None:
            _set_job(
                job_id,
                completed=done,
                total=total,
                message=f"Completed {done}/{total} repeated indicator calls.",
            )

        total_calls = len(indicators) * sustainability_core.STABILITY_REPEATS
        _set_job(
            job_id,
            message=(
                "Evaluating sustainability indicators with OpenAI "
                f"({sustainability_core.STABILITY_REPEATS} independent runs per indicator)."
            ),
            completed=0,
            total=total_calls,
            document_name=filename,
            document_word_count=word_count,
        )
        results = sustainability_core.evaluate_indicators(
            client,
            indicators,
            document_text,
            max_workers=request.max_workers,
            progress_callback=progress,
        )
        summary = sustainability_core.summarize_results(results)
        xlsx_bytes = sustainability_core.results_to_xlsx_bytes(results)
        xlsx_name = f"diagnostico_sostenibilidad_{_safe_filename_stem(filename)}.xlsx"

        _set_job(
            job_id,
            status="succeeded",
            message="Sustainability diagnosis complete.",
            completed=total_calls,
            total=total_calls,
            summary=summary,
            result_filename=xlsx_name,
            result_xlsx_b64=base64.b64encode(xlsx_bytes).decode("ascii"),
            result_preview=sustainability_core.results_to_public_dataframe(results)
            .head(10)
            .to_dict("records"),
        )
    except Exception as exc:  # noqa: BLE001 - report job failures to the GPT
        _set_job(job_id, status="failed", message=str(exc), error=str(exc))


def _run_attributes_job(job_id: str, request: AttributesJobRequest) -> None:
    try:
        _set_job(job_id, status="running", message="Downloading DOCX from ChatGPT.")
        filename, docx_bytes = _download_docx_file(request.openaiFileIdRefs)

        _set_job(job_id, message="Extracting DOCX text.")
        document_text = tab2_core.extract_docx_text_from_bytes(docx_bytes)
        word_count = len(document_text.split())

        _set_job(job_id, message="Selecting attribute rubric(s).")
        rubric_keys = tab2_core.resolve_rubric_keys(request.rubrics)
        criteria_count = tab2_core.count_criteria(rubric_keys)
        if criteria_count == 0:
            raise ValueError("The selected rubric(s) contain no criteria to evaluate.")

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured on the API server.")
        client = OpenAI(api_key=openai_api_key)

        def progress(done: int, total: int) -> None:
            _set_job(
                job_id,
                completed=done,
                total=total,
                message=f"Completed {done}/{total} repeated criterion calls.",
            )

        total_calls = criteria_count * tab2_core.STABILITY_REPEATS
        _set_job(
            job_id,
            message=(
                "Evaluating attribute criteria with OpenAI "
                f"({tab2_core.STABILITY_REPEATS} independent runs per criterion)."
            ),
            completed=0,
            total=total_calls,
            document_name=filename,
            document_word_count=word_count,
        )
        results = tab2_core.evaluate_rubrics(
            client,
            rubric_keys,
            document_text,
            max_workers=request.max_workers,
            progress_callback=progress,
        )
        summary = tab2_core.summarize_results(results)
        xlsx_bytes = tab2_core.results_to_xlsx_bytes(results)
        xlsx_name = f"diagnostico_atributos_{_safe_filename_stem(filename)}.xlsx"

        _set_job(
            job_id,
            status="succeeded",
            message="Attribute diagnosis complete.",
            completed=total_calls,
            total=total_calls,
            summary=summary,
            result_filename=xlsx_name,
            result_xlsx_b64=base64.b64encode(xlsx_bytes).decode("ascii"),
            result_preview=tab2_core.results_to_public_dataframe(results).head(10).to_dict("records"),
        )
    except Exception as exc:  # noqa: BLE001 - report job failures to the GPT
        _set_job(job_id, status="failed", message=str(exc), error=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


PRIVACY_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Privacy Policy - ILO PRODOC Evaluation Actions</title>
<style>
body{font-family:system-ui,sans-serif;max-width:48rem;margin:2rem auto;padding:0 1rem;line-height:1.6;color:#1a1a1a}
h1{font-size:1.6rem}h2{font-size:1.15rem;margin-top:2rem}small{color:#555}
</style>
</head>
<body>
<h1>Privacy Policy</h1>
<p><small>ILO PRODOC evaluation actions (pilot). Last updated: 2026-06-09.</small></p>

<p>This service supports pilot Custom GPTs that evaluate ILO project documents
against server-side quality-appraisal and sustainability rubrics. This policy
describes how data submitted to the service is handled.</p>

<h2>What we process</h2>
<p>When you submit a document, ChatGPT sends the service a short-lived download
link to the uploaded DOCX file. The service downloads the file, extracts its
text, and evaluates it against the relevant server-side rubric. We do not
request or process any information beyond the document you choose to submit.</p>

<h2>Third-party processing</h2>
<p>Document text is sent to the OpenAI API to perform the evaluation. OpenAI
processes this content under its API data-usage terms; API inputs are not used to
train OpenAI models by default. The service is hosted on Render.</p>

<h2>Retention</h2>
<p>This pilot does not use a database. Evaluation jobs and their results are held
only in server memory while a job runs and are discarded when the service
restarts. We do not persist uploaded documents or generated result files to disk.</p>

<h2>Access control</h2>
<p>Evaluation endpoints require an API key shared only with the configured Custom
GPTs. The service is intended for authorized ILO personnel, not public use.</p>

<h2>Caveats</h2>
<p>Results are AI-assisted and require expert validation. They do not constitute an
official ILO determination.</p>

<h2>Contact</h2>
<p>Direct questions about this pilot to the ILO team that operates this GPT.</p>
</body>
</html>"""


@app.get("/privacy", response_class=HTMLResponse)
def privacy() -> str:
    return PRIVACY_HTML


@app.post("/v3/jobs", response_model=JobCreated, dependencies=[Depends(require_api_key)])
def start_v3_appraisal_job(request: EvaluationJobRequest) -> JobCreated:
    try:
        prepared = _prepare_v3_job(request)
    except Exception as exc:  # el GPT recibe el fallo de inmediato, no como job fallido
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    minutos = max(1, round(prepared["estimated_seconds"] / 60))
    start_line = (
        f"Documento recibido: {prepared['filename']} "
        f"({prepared['word_count']:,} palabras).".replace(",", ".")
        + f" Evaluando {len(prepared['criteria'])} criterios "
        f"({prepared['total_calls']} consultas al modelo). "
        f"Tiempo estimado: {minutos} minuto{'s' if minutos != 1 else ''}."
    )
    _set_job(
        job_id,
        status="running",
        message="Evaluation started.",
        created_at=_now(),
        completed=0,
        total=prepared["total_calls"],
        document_name=prepared["filename"],
        document_word_count=prepared["word_count"],
    )
    thread = threading.Thread(
        target=_run_evaluation_job, args=(job_id, request, prepared), daemon=True
    )
    thread.start()
    return JobCreated(
        job_id=job_id,
        status="running",
        message=(
            "Evaluation started. Relay start_line verbatim, ask the user to write "
            "'estado' to check progress, and end this turn. Do not poll now."
        ),
        start_line=start_line,
        document_name=prepared["filename"],
        document_word_count=prepared["word_count"],
        criteria_count=len(prepared["criteria"]),
        total=prepared["total_calls"],
        estimated_seconds=prepared["estimated_seconds"],
    )


@app.get("/v3/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_v3_appraisal_job(job_id: str, wait_seconds: float = POLL_DEFAULT_WAIT_SECONDS) -> dict[str, Any]:
    job = await _await_job(job_id, wait_seconds)
    return _job_status_payload(job_id, job)


@app.get("/v3/jobs/{job_id}/result", dependencies=[Depends(require_api_key)])
def get_v3_appraisal_result(job_id: str) -> dict[str, Any]:
    job = _get_job(job_id)
    if job.get("status") != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.get('status')}; result is available only after success.",
        )

    token = job.get("download_token")
    if not token or not job.get("result_xlsx_bytes"):
        raise HTTPException(status_code=500, detail="Completed result file is unavailable.")

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "summary": job.get("summary"),
        "openaiFileResponse": [
            f"{PUBLIC_BASE_URL}/v3/jobs/{job_id}/file?token={token}"
        ],
    }


@app.get("/v3/jobs/{job_id}/file", include_in_schema=False)
def download_v3_appraisal_result(job_id: str, token: str) -> Response:
    """Serve the completed workbook to OpenAI's file fetcher via an opaque URL."""
    job = _get_job(job_id)
    expected = job.get("download_token")
    if not expected or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=404, detail="Result file not found.")

    content = job.get("result_xlsx_bytes")
    if not isinstance(content, bytes):
        raise HTTPException(status_code=404, detail="Result file not found.")
    filename = job.get("result_filename", "valoracion_v3_resultados.xlsx")
    return Response(
        content=content,
        media_type=XLSX_MIME,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/sustainability/jobs", response_model=JobCreated, dependencies=[Depends(require_api_key)])
def start_sustainability_diagnosis_job(request: SustainabilityJobRequest) -> JobCreated:
    job_id = str(uuid.uuid4())
    _set_job(
        job_id,
        status="queued",
        message="Job queued.",
        created_at=_now(),
        completed=0,
        total=0,
    )
    thread = threading.Thread(target=_run_sustainability_job, args=(job_id, request), daemon=True)
    thread.start()
    return JobCreated(
        job_id=job_id,
        status="queued",
        message=(
            "Sustainability diagnosis job started. Poll "
            "/sustainability/jobs/{job_id} until status is succeeded or failed."
        ),
    )


@app.get("/sustainability/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_sustainability_diagnosis_job(job_id: str, wait_seconds: float = POLL_DEFAULT_WAIT_SECONDS) -> dict[str, Any]:
    job = await _await_job(job_id, wait_seconds)
    return _job_status_payload(job_id, job)


@app.get("/sustainability/jobs/{job_id}/result", dependencies=[Depends(require_api_key)])
def get_sustainability_diagnosis_result(job_id: str) -> dict[str, Any]:
    job = _get_job(job_id)
    if job.get("status") != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.get('status')}; result is available only after success.",
        )

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "summary": job.get("summary"),
        "result_preview": job.get("result_preview", []),
        "openaiFileResponse": [
            {
                "name": job.get("result_filename", "diagnostico_sostenibilidad_resultados.xlsx"),
                "mime_type": XLSX_MIME,
                "content": job["result_xlsx_b64"],
            }
        ],
    }


@app.post("/attributes/jobs", response_model=JobCreated, dependencies=[Depends(require_api_key)])
def start_attributes_diagnosis_job(request: AttributesJobRequest) -> JobCreated:
    job_id = str(uuid.uuid4())
    _set_job(
        job_id,
        status="queued",
        message="Job queued.",
        created_at=_now(),
        completed=0,
        total=0,
    )
    thread = threading.Thread(target=_run_attributes_job, args=(job_id, request), daemon=True)
    thread.start()
    return JobCreated(
        job_id=job_id,
        status="queued",
        message=(
            "Attribute diagnosis job started. Poll "
            "/attributes/jobs/{job_id} until status is succeeded or failed."
        ),
    )


@app.get("/attributes/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_attributes_diagnosis_job(job_id: str, wait_seconds: float = POLL_DEFAULT_WAIT_SECONDS) -> dict[str, Any]:
    job = await _await_job(job_id, wait_seconds)
    return _job_status_payload(job_id, job)


@app.get("/attributes/jobs/{job_id}/result", dependencies=[Depends(require_api_key)])
def get_attributes_diagnosis_result(job_id: str) -> dict[str, Any]:
    job = _get_job(job_id)
    if job.get("status") != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.get('status')}; result is available only after success.",
        )

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "summary": job.get("summary"),
        "result_preview": job.get("result_preview", []),
        "openaiFileResponse": [
            {
                "name": job.get("result_filename", "diagnostico_atributos_resultados.xlsx"),
                "mime_type": XLSX_MIME,
                "content": job["result_xlsx_b64"],
            }
        ],
    }
