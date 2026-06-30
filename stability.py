"""Generic N-run stability aggregation shared by the GPT-action rubric cores.

Tab 1 (tab1_v3_core) keeps its own copy for now; tab2_core and
sustainability_core use this. Converge tab1 onto this later.

A "run" is one evaluation result dict. Runs are collapsed into a single modal
value plus a stability percentage (how many runs agreed with the mode). A run is
treated as the error sentinel when its ``Status`` == ``"Error"``.
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Hashable, Sequence

# Spanish column names produced for every aggregated row (consistent with tab1).
STABILITY_COLUMNS = [
    "N corridas",
    "Conteo modal",
    "Estabilidad (%)",
    "Estable (>=80%)",
    "Deriva principal (si inestable)",
    "Distribución de respuestas",
    "Corridas con error",
]

DEFAULT_THRESHOLD_PCT = 80.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_SECONDS = 1.5


def _format_pct(value: float) -> str:
    return f"{value:.0f}%" if float(value).is_integer() else f"{value:.1f}%"


def _order_index(value: Hashable, value_order: list[Hashable]) -> int:
    try:
        return value_order.index(value)
    except ValueError:
        return len(value_order)


def _distribution_text(counts: Counter, value_order: list[Hashable]) -> str:
    parts = [f"{v}={counts[v]}" for v in value_order if counts.get(v, 0)]
    parts += [f"{v}={counts[v]}" for v in sorted(set(counts) - set(value_order), key=str)]
    return "; ".join(parts)


def _drift_text(
    counts: Counter,
    modal: Hashable,
    total: int,
    stable: bool,
    value_order: list[Hashable],
) -> str:
    if stable:
        return ""
    remaining = total - counts.get(modal, 0)
    non_modal = {v: c for v, c in counts.items() if v != modal and c > 0}
    if remaining <= 0 or not non_modal:
        return ""
    max_count = max(non_modal.values())
    top = sorted(
        (v for v, c in non_modal.items() if c == max_count),
        key=lambda v: _order_index(v, value_order),
    )
    label = " / ".join(str(v) for v in top)
    tie_prefix = "empate: " if len(top) > 1 else ""
    total_pct = 100 * max_count / total
    return f"{label} ({tie_prefix}{max_count}/{remaining} restantes; {_format_pct(total_pct)} total)"


def aggregate_runs(
    repeated_results: list[dict[str, Any]],
    *,
    value_key: str,
    value_order: Sequence[Hashable],
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    evidence_key: str = "Evidencia",
    error_label: Hashable = "Error",
) -> dict[str, Any]:
    """Collapse repeated runs into a modal value plus stability metrics.

    Returns a dict with: ``modal``, ``representative`` (the run kept for its
    analysis/evidence), ``reasoning_prefix`` (Spanish stability summary),
    ``columns`` (the STABILITY_COLUMNS dict), ``stable``, ``error_count``.
    Mapping ``modal`` back onto a score/answer column is left to the caller.
    """
    value_order = list(value_order)
    total = len(repeated_results)
    if total == 0:
        return {
            "modal": error_label,
            "representative": {},
            "reasoning_prefix": "No se recibieron corridas para este criterio.",
            "stable": False,
            "error_count": 0,
            "columns": {
                "N corridas": 0,
                "Conteo modal": 0,
                "Estabilidad (%)": 0.0,
                "Estable (>=80%)": "No",
                "Deriva principal (si inestable)": "",
                "Distribución de respuestas": "",
                "Corridas con error": 0,
            },
        }

    def label_of(result: dict[str, Any]) -> Hashable:
        return error_label if result.get("Status") == "Error" else result.get(value_key)

    repeated_results = sorted(repeated_results, key=lambda r: int(r.get("repeat", 0) or 0))
    counts: Counter = Counter(label_of(r) for r in repeated_results)
    modal = min(counts, key=lambda v: (-counts[v], _order_index(v, value_order)))
    modal_count = counts[modal]
    stability_pct = round(100 * modal_count / total, 1)
    stable = stability_pct >= threshold_pct
    drift = _drift_text(counts, modal, total, stable, value_order)
    distribution = _distribution_text(counts, value_order)
    error_count = sum(1 for r in repeated_results if r.get("Status") == "Error")

    candidates = [r for r in repeated_results if label_of(r) == modal] or repeated_results
    representative = max(
        candidates,
        key=lambda r: (
            r.get("Status") == "Success",
            len(str(r.get(evidence_key, ""))),
            -int(r.get("repeat", 0) or 0),
        ),
    )

    prefix = (
        f"Resultado modal tras {total} corridas independientes: {modal} "
        f"({modal_count}/{total}; {_format_pct(stability_pct)} de estabilidad). "
        f"Distribución: {distribution}."
    )
    if not stable:
        prefix += f" Resultado inestable (<{_format_pct(threshold_pct)})."
        if drift:
            prefix += f" Deriva principal: {drift}."

    return {
        "modal": modal,
        "representative": representative,
        "reasoning_prefix": prefix,
        "stable": stable,
        "error_count": error_count,
        "columns": {
            "N corridas": total,
            "Conteo modal": modal_count,
            "Estabilidad (%)": stability_pct,
            "Estable (>=80%)": "Sí" if stable else "No",
            "Deriva principal (si inestable)": drift,
            "Distribución de respuestas": distribution,
            "Corridas con error": error_count,
        },
    }


def run_with_retries(
    eval_once: Callable[[], dict[str, Any]],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """Call ``eval_once`` once, retrying only results whose Status is 'Error'."""
    result = eval_once()
    for attempt in range(1, max_retries + 1):
        if result.get("Status") != "Error":
            break
        time.sleep(backoff_seconds * attempt)
        result = eval_once()
    return result


def evaluate_with_stability(
    items: Sequence[tuple[Hashable, Any]],
    eval_one: Callable[[Any], dict[str, Any]],
    aggregate_one: Callable[[Hashable, list[dict[str, Any]]], dict[str, Any]],
    *,
    repeats: int,
    max_workers: int,
    progress_callback: Callable[[int, int], None] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> list[dict[str, Any]]:
    """Run each item ``repeats`` times concurrently, then aggregate per item.

    ``items`` is a list of (key, payload). ``eval_one(payload)`` returns one run
    dict. ``aggregate_one(key, runs)`` collapses that item's runs into a final
    row. Results are returned in the original item order.
    """
    items = list(items)
    if not items:
        return []

    total_calls = len(items) * repeats
    repeated_by_key: dict[Hashable, list[dict[str, Any]]] = {key: [] for key, _ in items}
    workers = max(1, min(int(max_workers), total_calls))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for key, payload in items:
            for repeat in range(1, repeats + 1):
                fut = ex.submit(
                    run_with_retries,
                    lambda p=payload: eval_one(p),
                    max_retries=max_retries,
                    backoff_seconds=backoff_seconds,
                )
                futures[fut] = (key, repeat)
        done = 0
        for fut in as_completed(futures):
            key, repeat = futures[fut]
            result = fut.result()
            result["repeat"] = repeat
            repeated_by_key[key].append(result)
            done += 1
            if progress_callback:
                progress_callback(done, total_calls)

    return [aggregate_one(key, repeated_by_key[key]) for key, _ in items]


def _demo() -> None:
    """Self-check for the modal/stability/drift math."""
    runs = [
        {"Score": 3, "Status": "Success", "Evidencia": "aaa", "repeat": 1},
        {"Score": 3, "Status": "Success", "Evidencia": "a", "repeat": 2},
        {"Score": 3, "Status": "Success", "Evidencia": "aa", "repeat": 3},
        {"Score": 2, "Status": "Success", "Evidencia": "b", "repeat": 4},
        {"Score": 0, "Status": "Error", "Evidencia": "", "repeat": 5},
    ]
    agg = aggregate_runs(runs, value_key="Score", value_order=[0, 1, 2, 3, "Error"])
    assert agg["modal"] == 3, agg["modal"]
    assert agg["columns"]["Estabilidad (%)"] == 60.0, agg["columns"]
    assert agg["stable"] is False
    assert agg["error_count"] == 1
    assert "2" in agg["columns"]["Deriva principal (si inestable)"]
    # Representative is a modal (Score==3) run with the longest evidence.
    assert agg["representative"]["repeat"] == 1, agg["representative"]

    # Unanimous -> stable, no drift.
    unanimous = [{"Score": 2, "Status": "Success", "repeat": i} for i in range(1, 6)]
    agg2 = aggregate_runs(unanimous, value_key="Score", value_order=[0, 1, 2, 3, "Error"])
    assert agg2["columns"]["Estabilidad (%)"] == 100.0
    assert agg2["stable"] is True
    assert agg2["columns"]["Deriva principal (si inestable)"] == ""
    print("stability._demo OK")


if __name__ == "__main__":
    _demo()
