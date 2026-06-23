"""Configuration and checkpoint helpers for refit-robust selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feature_method_sweep import MethodSpec, TrackSpec


def parse_csv_floats(value: str) -> list[float]:
    """Parse a comma-separated list of ridge alpha values.

    Empty fields are ignored so values such as "0.1, 1, 10," are accepted.

    Args:
        value: Comma-separated numeric string.

    Returns:
        Parsed float values in their input order.

    """
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def load_existing_indices(method_dir: Path) -> list[int] | None:
    """Load selected image indices from a run checkpoint.

    Args:
        method_dir: Payload directory containing selected_indices.npy.

    Returns:
        List of integer pool indices, or None when the checkpoint is absent.

    """
    path = method_dir / "selected_indices.npy"
    if not path.exists():
        return None
    return [int(x) for x in np.load(path).tolist()]


def load_existing_rows(path: Path, *, max_iteration: int | None = None) -> list[dict[str, Any]]:
    """Load a checkpoint CSV as dictionaries.

    Missing and empty files are treated as no completed rows.  When
    max_iteration is supplied and the CSV has an iteration column, rows after
    that completed iteration are ignored so partially written resume state
    cannot get ahead of selected_indices.npy.

    Args:
        path: CSV checkpoint path.
        max_iteration: Highest completed greedy iteration to keep.

    Returns:
        CSV rows converted to plain dictionaries with NaN replaced by None.

    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        rows = pd.read_csv(path).replace({np.nan: None}).to_dict("records")
    except pd.errors.EmptyDataError:
        return []
    if not rows or max_iteration is None or "iteration" not in rows[0]:
        return [dict(row) for row in rows]
    return [
        dict(row)
        for row in rows
        if row.get("iteration") is not None and int(row["iteration"]) <= max_iteration
    ]


def load_resume_state(
    *,
    method_dir: Path,
    target_size: int,
    init_size: int,
    pool_size: int,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Validate and load all state needed to resume selection.

    The selected-index checkpoint is authoritative.  Trace and candidate
    score rows are loaded only through the number of completed greedy
    iterations implied by selected_indices.npy.

    Args:
        method_dir: Method payload directory.
        target_size: Requested final number of selected images.
        init_size: Number of initial seed images.
        pool_size: Number of candidate rows available in the loaded pool.

    Returns:
        Tuple of selected indices, trace rows, and candidate-score rows, or
        None when no selected-index checkpoint exists.

    Raises:
        ValueError: If the checkpoint is internally inconsistent or refers
        to indices outside the current candidate pool.

    """
    selected = load_existing_indices(method_dir)
    if selected is None:
        return None
    if len(selected) < init_size:
        raise ValueError(
            f"Cannot resume {method_dir}: selected_indices.npy has only "
            f"{len(selected)} entries, expected at least init_size={init_size}"
        )
    if len(selected) > target_size:
        raise ValueError(
            f"Cannot resume {method_dir}: selected_indices.npy has "
            f"{len(selected)} entries, exceeding target_size={target_size}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError(f"Cannot resume {method_dir}: selected indices contain duplicates")
    bad = [idx for idx in selected if idx < 0 or idx >= pool_size]
    if bad:
        raise ValueError(
            f"Cannot resume {method_dir}: selected index outside pool_size={pool_size}: "
            f"{bad[:5]}"
        )

    completed_iterations = len(selected) - init_size
    trace_rows = load_existing_rows(
        method_dir / "selection_trace.csv",
        max_iteration=completed_iterations,
    )
    if len(trace_rows) < completed_iterations:
        raise ValueError(
            f"Cannot resume {method_dir}: selection_trace.csv has {len(trace_rows)} "
            f"completed rows, expected {completed_iterations}"
        )
    if len(trace_rows) > completed_iterations:
        print(
            f"Truncating resume trace from {len(trace_rows)} to "
            f"{completed_iterations} completed iterations",
            flush=True,
        )
        trace_rows = trace_rows[:completed_iterations]

    candidate_rows = load_existing_rows(
        method_dir / "candidate_scores.csv",
        max_iteration=completed_iterations,
    )
    return selected, trace_rows, candidate_rows


def save_filter_records(method_dir: Path, records: list[dict[str, Any]]) -> None:
    """Write image-filter audit records and summary counts.

    Args:
        method_dir: Method payload directory to receive filter_records.csv
            and filter_summary.json.
        records: Filter audit rows produced during initialization or greedy
            candidate validation.

    Side effects:
        Writes CSV/JSON files when records is non-empty.

    """
    if not records:
        return
    filter_df = pd.DataFrame(records)
    filter_df.to_csv(method_dir / "filter_records.csv", index=False)
    passed_series = (
        filter_df["passed"].map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})
        if "passed" in filter_df
        else pd.Series([], dtype=bool)
    )
    filter_summary = {
        "n_records": int(len(filter_df)),
        "n_passed": int(passed_series.sum()) if "passed" in filter_df else 0,
        "n_failed": int((~passed_series).sum()) if "passed" in filter_df else 0,
        "reason_counts": (
            filter_df["reason"].value_counts(dropna=False).to_dict()
            if "reason" in filter_df
            else {}
        ),
    }
    with (method_dir / "filter_summary.json").open("w") as f:
        json.dump(filter_summary, f, indent=2, default=str)


def format_seconds(seconds: float) -> str:
    """Format an elapsed duration for progress logs.

    Args:
        seconds: Duration in seconds.  Negative values are clamped to zero.

    Returns:
        Human-readable duration string in seconds, minutes, or hours.

    """
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{sec:04.1f}s"


def make_method(method_id: str, track: str) -> MethodSpec:
    """Build manifest metadata for the refit-robust method.

    Args:
        method_id: Stable payload identifier for the run.
        track: Encoding track used for proxy scoring and refit targets.

    Returns:
        MethodSpec compatible with feature_method_sweep payload writers.

    """
    return MethodSpec(
        method_id=method_id,
        label=f"{track} eval-augmented LOO refit robust",
        tracks=(TrackSpec(name=track, type="encoding", encoding_name=track),),
        track_agg_method="identity",
        track_norm_method="none",
        within="mean",
        across="min",
        summary_weights={track: 1.0},
        description=(
            "Experimental greedy selector. Candidate shortlist is formed by the "
            "attenuated fixed-RDM track objective, then reranked by "
            "eval-set-augmented LOO teacher/student recovery accuracy, with "
            "RDM margin used as a tie-breaker."
        ),
    )
