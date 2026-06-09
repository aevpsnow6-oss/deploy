# %%
"""Interactive stability audit for the Tab 1 v3 PRODOC quality rubric.

Run this file cell-by-cell in VS Code, Cursor, PyCharm, or another editor that
understands `# %%` cells. The audit repeats the same rubric evaluation several
times over one or more DOCX PRODOCs and measures stability at two levels:

1. Final answer stability per criterion (`Respuesta`).
2. Logical-condition stability per criterion (`T1`, `T2`, ... parsed from
   `Razonamiento` and compared against the tests expected by the rubric).

The script writes live CSV checkpoints while it runs, then a final XLSX workbook.
"""

# %%
# 1. Imports and repo setup

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

try:
    SCRIPT_PATH = Path(__file__).resolve()
    REPO_ROOT = SCRIPT_PATH.parents[1]
except NameError:
    REPO_ROOT = Path.cwd()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)

import tab1_v3_core as v3_core  # noqa: E402

# %%
# 2. Configuration

# Put one or more .docx files in this folder before running the audit.
PRODOCS_DIR = REPO_ROOT / "stability_inputs" / "prodocs"

# Outputs are ignored by git. Each run creates a timestamped subfolder.
OUTPUT_ROOT = REPO_ROOT / "stability_outputs" / "v3"

# Cost/runtime controls. Full v3 is 76 criteria per document per repeat.
N_REPEATS = 3
MAX_WORKERS = 4
CHECKPOINT_EVERY_RESULTS = 10

# Optional filters for smaller pilots. Leave as None for full rubric.
SECTIONS: list[int] | None = None
SUBSECTIONS: list[str] | None = None
CRITERION_IDS: list[str] | None = None

# Safety switch: keep False until the setup/preview cells look right.
RUN_AUDIT = False

# %%
# 3. Helpers: file discovery, hashes, and rubric filtering

RUBRIC_LOGIC_COLUMNS = [
    "Rúbrica — Sí",
    "Rúbrica — Parcial",
    "Rúbrica — No",
    "Rúbrica — No aplica",
]

TEST_ID_RE = re.compile(r"\b(T\d+)\b", re.IGNORECASE)
TEST_DEF_RE = re.compile(
    r"\b(T\d+)\s*:\s*(.*?)(?=\n\s*T\d+\s*:|\n\s*DECISI[ÓO]N\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
TEST_LINE_RE = re.compile(r"^\s*(T\d+)\s*[:=-]\s*(.+?)\s*$", re.IGNORECASE)
RESULT_LINE_RE = re.compile(
    r"^\s*RESULTADO\s*[:=-]\s*(Yes|No|Partial|Not Found|N/A)\b",
    re.IGNORECASE,
)
VALID_ANSWERS = {"Yes", "No", "Partial", "Not Found", "N/A", "Error"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact_text(value: Any, max_chars: int = 500) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def test_sort_key(test_id: str) -> int:
    match = re.search(r"\d+", str(test_id))
    return int(match.group(0)) if match else 9999


def list_prodocs(prodocs_dir: Path) -> list[Path]:
    prodocs_dir.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in prodocs_dir.glob("*.docx") if not path.name.startswith("~$"))


def load_selected_rubric() -> pd.DataFrame:
    rubric = v3_core.load_rubrica_v3()
    criteria = v3_core.filter_rubric(rubric, sections=SECTIONS, subsections=SUBSECTIONS)
    if CRITERION_IDS:
        wanted = {str(item) for item in CRITERION_IDS}
        criteria = criteria[criteria["ID"].astype(str).isin(wanted)].reset_index(drop=True)
    return criteria


def extract_expected_tests(row: pd.Series) -> dict[str, str]:
    """Return expected T-tests for one rubric row, keyed by T id."""
    definitions: dict[str, str] = {}
    all_text_chunks = []
    for col in RUBRIC_LOGIC_COLUMNS:
        value = row.get(col, "")
        text = "" if value is None or pd.isna(value) else str(value)
        all_text_chunks.append(text)
        for match in TEST_DEF_RE.finditer(text):
            test_id = match.group(1).upper()
            definitions.setdefault(test_id, compact_text(match.group(2), max_chars=700))

    if definitions:
        return dict(sorted(definitions.items(), key=lambda item: test_sort_key(item[0])))

    all_ids = {match.group(1).upper() for match in TEST_ID_RE.finditer("\n".join(all_text_chunks))}
    return {test_id: "" for test_id in sorted(all_ids, key=test_sort_key)}


def normalize_condition_value(raw_value: str) -> str:
    value = raw_value.strip().lower()
    value = value.replace("á", "a").replace("í", "i").replace("é", "e")
    if re.match(r"^(verdadero|true|si|cumple)\b", value):
        return "TRUE"
    if re.match(r"^(falso|false|no|no cumple)\b", value):
        return "FALSE"
    return "UNKNOWN"


def parse_reasoning(reasoning: Any) -> tuple[dict[str, dict[str, str]], str]:
    """Parse T-condition values and the RESULTADO line from model reasoning."""
    text = "" if reasoning is None else str(reasoning)
    conditions: dict[str, dict[str, str]] = {}
    reasoning_result = ""

    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue

        test_match = TEST_LINE_RE.match(clean)
        if test_match:
            test_id = test_match.group(1).upper()
            raw_value = test_match.group(2)
            conditions[test_id] = {
                "condition_value": normalize_condition_value(raw_value),
                "condition_line": clean,
            }
            continue

        result_match = RESULT_LINE_RE.match(clean)
        if result_match:
            candidate = result_match.group(1)
            reasoning_result = "N/A" if candidate.upper() == "N/A" else candidate.title()
            if reasoning_result == "Not Found":
                reasoning_result = "Not Found"

    return conditions, reasoning_result


# %%
# 4. Preview setup before spending API calls

prodoc_files = list_prodocs(PRODOCS_DIR)
criteria_df = load_selected_rubric()
expected_tests_by_id = {
    str(row["ID"]): extract_expected_tests(row)
    for _, row in criteria_df.iterrows()
}

print(f"Repo root: {REPO_ROOT}")
print(f"PRODOC folder: {PRODOCS_DIR}")
print(f"PRODOC files found: {len(prodoc_files)}")
for path in prodoc_files:
    print(f" - {path.name}")
print(f"Criteria selected: {len(criteria_df)}")
print(f"Repeats: {N_REPEATS}")
print(f"Planned model calls: {len(prodoc_files) * len(criteria_df) * N_REPEATS}")
print(f"Run enabled: {RUN_AUDIT}")

sample_counts = pd.Series({crit_id: len(tests) for crit_id, tests in expected_tests_by_id.items()})
print("Expected logical tests per criterion:")
print(sample_counts.describe().to_string())

# %%
# 5. Core audit runner

def result_to_rows(
    *,
    run_id: str,
    repeat: int,
    doc_path: Path,
    doc_sha256: str,
    row: pd.Series,
    result: dict[str, Any],
    elapsed_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    criterion_id = str(row["ID"])
    reasoning = result.get("Razonamiento", "")
    parsed_conditions, reasoning_result = parse_reasoning(reasoning)
    expected_tests = extract_expected_tests(row)

    answer = str(result.get("Respuesta", "Error"))
    if answer not in VALID_ANSWERS:
        answer = "Error"

    raw_row = {
        "run_id": run_id,
        "repeat": repeat,
        "document_name": doc_path.name,
        "document_sha256": doc_sha256,
        "criterion_id": criterion_id,
        "subsection": result.get("Subsección", ""),
        "criterion": result.get("Criterio", row.get("Criterio a evaluar", "")),
        "subjectivity": result.get("Subjetividad", row.get("Subjetividad residual (v3)", "")),
        "transversals": result.get("Transversales", row.get("Aspectos transversales", "")),
        "answer": answer,
        "reasoning_result": reasoning_result,
        "answer_matches_reasoning_result": (
            reasoning_result == answer if reasoning_result else pd.NA
        ),
        "status": result.get("Status", ""),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "reasoning": reasoning,
        "evidence": result.get("Evidencia", ""),
    }

    all_test_ids = set(expected_tests) | set(parsed_conditions)
    condition_rows = []
    for test_id in sorted(all_test_ids, key=test_sort_key):
        parsed = parsed_conditions.get(test_id)
        expected = test_id in expected_tests
        condition_rows.append(
            {
                "run_id": run_id,
                "repeat": repeat,
                "document_name": doc_path.name,
                "document_sha256": doc_sha256,
                "criterion_id": criterion_id,
                "condition_id": test_id,
                "expected_condition": expected,
                "expected_condition_text": expected_tests.get(test_id, ""),
                "condition_value": parsed["condition_value"] if parsed else "MISSING",
                "condition_line": parsed["condition_line"] if parsed else "",
                "answer": answer,
                "status": result.get("Status", ""),
            }
        )

    return raw_row, condition_rows


def write_live_checkpoints(
    output_dir: Path,
    raw_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(raw_rows).to_csv(output_dir / "raw_results_live.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(output_dir / "condition_results_live.csv", index=False)


def evaluate_one_row(
    client: OpenAI,
    document_text: str,
    row: pd.Series,
) -> tuple[dict[str, Any], float]:
    started = time.time()
    result = v3_core.evaluate_criterion(client, row, document_text)
    return result, time.time() - started


def run_stability_audit() -> dict[str, pd.DataFrame | Path]:
    if not prodoc_files:
        raise FileNotFoundError(f"No .docx files found in {PRODOCS_DIR}")
    if criteria_df.empty:
        raise ValueError("No rubric criteria selected.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    run_id = datetime.now().strftime("v3_stability_%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    client = OpenAI(api_key=api_key)
    raw_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    rubric_hash = sha256_file(REPO_ROOT / v3_core.RUBRICA_V3_PATH)
    prompt_hash = sha256_text(v3_core.V3_SYSTEM_PROMPT + v3_core.V3_USER_PROMPT_TEMPLATE)
    total_calls = len(prodoc_files) * len(criteria_df) * N_REPEATS
    completed_calls = 0

    print(f"Writing outputs to: {output_dir}")
    print(f"Total model calls planned: {total_calls}")

    for doc_path in prodoc_files:
        doc_sha256 = sha256_file(doc_path)
        document_text = v3_core.extract_docx_text_from_path(str(doc_path))
        word_count = len(document_text.split())

        manifest_rows.append(
            {
                "run_id": run_id,
                "document_name": doc_path.name,
                "document_sha256": doc_sha256,
                "document_words": word_count,
                "rubric_path": v3_core.RUBRICA_V3_PATH,
                "rubric_sha256": rubric_hash,
                "model": v3_core.MODEL,
                "prompt_sha256": prompt_hash,
                "n_repeats": N_REPEATS,
                "max_workers": MAX_WORKERS,
                "sections": json.dumps(SECTIONS, ensure_ascii=False),
                "subsections": json.dumps(SUBSECTIONS, ensure_ascii=False),
                "criterion_ids": json.dumps(CRITERION_IDS, ensure_ascii=False),
            }
        )

        print(f"\nDocument: {doc_path.name} ({word_count:,} words)")

        for repeat in range(1, N_REPEATS + 1):
            print(f"  Repeat {repeat}/{N_REPEATS} started")
            repeat_started = time.time()

            with ThreadPoolExecutor(max_workers=max(1, int(MAX_WORKERS))) as executor:
                futures = {
                    executor.submit(evaluate_one_row, client, document_text, row): row
                    for _, row in criteria_df.iterrows()
                }

                for future in as_completed(futures):
                    row = futures[future]
                    result, elapsed_seconds = future.result()
                    raw_row, parsed_condition_rows = result_to_rows(
                        run_id=run_id,
                        repeat=repeat,
                        doc_path=doc_path,
                        doc_sha256=doc_sha256,
                        row=row,
                        result=result,
                        elapsed_seconds=elapsed_seconds,
                    )
                    raw_rows.append(raw_row)
                    condition_rows.extend(parsed_condition_rows)
                    completed_calls += 1

                    print(
                        f"    {completed_calls}/{total_calls} "
                        f"{doc_path.name} repeat={repeat} "
                        f"criterion={raw_row['criterion_id']} "
                        f"answer={raw_row['answer']} "
                        f"conditions={len(parsed_condition_rows)}"
                    )

                    if completed_calls % CHECKPOINT_EVERY_RESULTS == 0:
                        write_live_checkpoints(output_dir, raw_rows, condition_rows)

            write_live_checkpoints(output_dir, raw_rows, condition_rows)
            print(f"  Repeat {repeat}/{N_REPEATS} finished in {time.time() - repeat_started:.1f}s")

    raw_df = pd.DataFrame(raw_rows)
    condition_df = pd.DataFrame(condition_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    answer_stability_df = summarize_answer_stability(raw_df)
    condition_stability_df = summarize_condition_stability(condition_df)
    document_summary_df = summarize_by_document(answer_stability_df, condition_stability_df, raw_df)

    write_final_outputs(
        output_dir=output_dir,
        manifest_df=manifest_df,
        raw_df=raw_df,
        condition_df=condition_df,
        answer_stability_df=answer_stability_df,
        condition_stability_df=condition_stability_df,
        document_summary_df=document_summary_df,
    )

    print(f"\nFinal workbook: {output_dir / 'stability_audit_v3.xlsx'}")
    return {
        "output_dir": output_dir,
        "manifest": manifest_df,
        "raw": raw_df,
        "conditions": condition_df,
        "answer_stability": answer_stability_df,
        "condition_stability": condition_stability_df,
        "document_summary": document_summary_df,
    }


# %%
# 6. Stability summaries

def value_counts_json(values: pd.Series) -> str:
    counts = values.fillna("MISSING").astype(str).value_counts(dropna=False).to_dict()
    return json.dumps(counts, ensure_ascii=False, sort_keys=True)


def modal_value(values: pd.Series) -> str:
    counts = values.fillna("MISSING").astype(str).value_counts(dropna=False)
    return str(counts.index[0]) if not counts.empty else ""


def agreement_rate(values: pd.Series) -> float:
    counts = values.fillna("MISSING").astype(str).value_counts(dropna=False)
    if counts.empty:
        return 0.0
    return round(float(counts.iloc[0]) / float(counts.sum()), 4)


def summarize_answer_stability(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["document_name", "document_sha256", "criterion_id"]
    for keys, group in raw_df.groupby(group_cols, dropna=False):
        document_name, document_sha256, criterion_id = keys
        answers = group["answer"].fillna("Error").astype(str)
        statuses = group["status"].fillna("").astype(str)
        rows.append(
            {
                "document_name": document_name,
                "document_sha256": document_sha256,
                "criterion_id": criterion_id,
                "criterion": group["criterion"].iloc[0],
                "subjectivity": group["subjectivity"].iloc[0],
                "transversals": group["transversals"].iloc[0],
                "n_runs": len(group),
                "modal_answer": modal_value(answers),
                "answer_agreement_rate": agreement_rate(answers),
                "answer_counts": value_counts_json(answers),
                "distinct_answers": int(answers.nunique(dropna=False)),
                "status_counts": value_counts_json(statuses),
                "reasoning_result_mismatch_count": int(
                    (group["answer_matches_reasoning_result"] == False).sum()  # noqa: E712
                ),
                "needs_review": bool(
                    answers.nunique(dropna=False) > 1
                    or "Error" in set(answers)
                    or (group["answer_matches_reasoning_result"] == False).any()  # noqa: E712
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["document_name", "answer_agreement_rate", "criterion_id"],
        ascending=[True, True, True],
    )


def summarize_condition_stability(condition_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["document_name", "document_sha256", "criterion_id", "condition_id"]
    for keys, group in condition_df.groupby(group_cols, dropna=False):
        document_name, document_sha256, criterion_id, condition_id = keys
        values = group["condition_value"].fillna("MISSING").astype(str)
        rows.append(
            {
                "document_name": document_name,
                "document_sha256": document_sha256,
                "criterion_id": criterion_id,
                "condition_id": condition_id,
                "expected_condition": bool(group["expected_condition"].max()),
                "expected_condition_text": group["expected_condition_text"].dropna().iloc[0]
                if group["expected_condition_text"].notna().any()
                else "",
                "n_runs": len(group),
                "modal_condition_value": modal_value(values),
                "condition_agreement_rate": agreement_rate(values),
                "condition_counts": value_counts_json(values),
                "distinct_condition_values": int(values.nunique(dropna=False)),
                "missing_count": int((values == "MISSING").sum()),
                "unknown_count": int((values == "UNKNOWN").sum()),
                "needs_review": bool(
                    values.nunique(dropna=False) > 1
                    or (values == "MISSING").any()
                    or (values == "UNKNOWN").any()
                    or not bool(group["expected_condition"].max())
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["document_name", "condition_agreement_rate", "criterion_id", "condition_id"],
        ascending=[True, True, True, True],
    )


def summarize_by_document(
    answer_stability_df: pd.DataFrame,
    condition_stability_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for document_name, answer_group in answer_stability_df.groupby("document_name", dropna=False):
        condition_group = condition_stability_df[
            condition_stability_df["document_name"] == document_name
        ]
        raw_group = raw_df[raw_df["document_name"] == document_name]
        rows.append(
            {
                "document_name": document_name,
                "criteria_evaluated": int(answer_group["criterion_id"].nunique()),
                "raw_runs": int(len(raw_group)),
                "unstable_answer_criteria": int(answer_group["needs_review"].sum()),
                "avg_answer_agreement_rate": round(answer_group["answer_agreement_rate"].mean(), 4),
                "conditions_evaluated": int(len(condition_group)),
                "unstable_conditions": int(condition_group["needs_review"].sum()),
                "avg_condition_agreement_rate": round(condition_group["condition_agreement_rate"].mean(), 4)
                if not condition_group.empty
                else pd.NA,
                "error_runs": int((raw_group["status"] == "Error").sum()),
            }
        )
    return pd.DataFrame(rows)


def write_final_outputs(
    *,
    output_dir: Path,
    manifest_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    answer_stability_df: pd.DataFrame,
    condition_stability_df: pd.DataFrame,
    document_summary_df: pd.DataFrame,
) -> None:
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)
    raw_df.to_csv(output_dir / "raw_results.csv", index=False)
    condition_df.to_csv(output_dir / "condition_results.csv", index=False)
    answer_stability_df.to_csv(output_dir / "answer_stability.csv", index=False)
    condition_stability_df.to_csv(output_dir / "condition_stability.csv", index=False)
    document_summary_df.to_csv(output_dir / "document_summary.csv", index=False)

    workbook_path = output_dir / "stability_audit_v3.xlsx"
    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        document_summary_df.to_excel(writer, index=False, sheet_name="summary")
        answer_stability_df.to_excel(writer, index=False, sheet_name="answer_stability")
        condition_stability_df.to_excel(writer, index=False, sheet_name="condition_stability")
        raw_df.to_excel(writer, index=False, sheet_name="raw_results")
        condition_df.to_excel(writer, index=False, sheet_name="condition_results")
        manifest_df.to_excel(writer, index=False, sheet_name="manifest")

        workbook = writer.book
        header_fmt = workbook.add_format(
            {"bold": True, "text_wrap": True, "valign": "top", "fg_color": "#D9EAF7", "border": 1}
        )
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        for sheet_name, worksheet in writer.sheets.items():
            df_sheet = {
                "summary": document_summary_df,
                "answer_stability": answer_stability_df,
                "condition_stability": condition_stability_df,
                "raw_results": raw_df,
                "condition_results": condition_df,
                "manifest": manifest_df,
            }[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, max(len(df_sheet), 1), max(len(df_sheet.columns) - 1, 0))
            for col_num, value in enumerate(df_sheet.columns):
                worksheet.write(0, col_num, value, header_fmt)
            worksheet.set_column(0, max(len(df_sheet.columns) - 1, 0), 22, wrap_fmt)


# %%
# 7. Run audit

if RUN_AUDIT:
    audit_outputs = run_stability_audit()
else:
    print("RUN_AUDIT is False. Review setup above, then set RUN_AUDIT = True and run this cell.")

