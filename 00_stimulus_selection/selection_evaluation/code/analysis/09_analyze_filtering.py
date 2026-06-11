#!/usr/bin/env python3
"""Analyze image filtering results from stimulus selection.

Outputs:
- filter_records.csv: Full record of all filter evaluations
- filter_summary.json: Aggregate statistics about filtering
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.utils import (
    add_standard_args,
    setup_from_args,
)


def compute_filter_summary(records: list[dict]) -> dict:
    """Compute aggregate statistics from filter records.

    Args:
        records: List of filter record dictionaries

    Returns:
        Summary statistics dictionary
    """
    if not records:
        return {"total_evaluated": 0, "error": "No filter records found"}

    df = pd.DataFrame(records)

    total = len(df)
    passed = int(df["passed"].sum())
    failed = int(total - passed)

    # Rejection breakdown by reason
    failed_df = df[~df["passed"]]
    rejection_counts = Counter(failed_df["reason"].tolist())

    # Rank distribution (how often was 1st, 2nd, 3rd+ choice selected)
    passed_df = df[df["passed"]]
    rank_counts = Counter(passed_df["rank"].tolist())
    rank_distribution = {
        "1": int(rank_counts.get(1, 0)),
        "2": int(rank_counts.get(2, 0)),
        "3": int(rank_counts.get(3, 0)),
        "4+": int(sum(v for k, v in rank_counts.items() if k >= 4)),
    }

    # Resolution statistics (for records that have resolution data)
    has_resolution = df["width"].notna() & df["height"].notna()
    if has_resolution.any():
        df_with_res = df[has_resolution].copy()
        df_with_res["max_dim"] = df_with_res[["width", "height"]].max(axis=1)

        passed_res = df_with_res[df_with_res["passed"]]["max_dim"]
        failed_res = df_with_res[~df_with_res["passed"]]["max_dim"]

        resolution_stats = {
            "passed": {
                "count": int(len(passed_res)),
                "mean": float(passed_res.mean()) if len(passed_res) > 0 else None,
                "median": float(passed_res.median()) if len(passed_res) > 0 else None,
                "min": int(passed_res.min()) if len(passed_res) > 0 else None,
                "max": int(passed_res.max()) if len(passed_res) > 0 else None,
            },
            "failed": {
                "count": int(len(failed_res)),
                "mean": float(failed_res.mean()) if len(failed_res) > 0 else None,
                "median": float(failed_res.median()) if len(failed_res) > 0 else None,
                "min": int(failed_res.min()) if len(failed_res) > 0 else None,
                "max": int(failed_res.max()) if len(failed_res) > 0 else None,
            },
        }
    else:
        resolution_stats = None

    # Classifier probability statistics
    has_prob = df["natural_prob"].notna()
    if has_prob.any():
        df_with_prob = df[has_prob]

        passed_prob = df_with_prob[df_with_prob["passed"]]["natural_prob"]
        failed_prob = df_with_prob[~df_with_prob["passed"]]["natural_prob"]

        classifier_stats = {
            "passed": {
                "count": int(len(passed_prob)),
                "mean": float(passed_prob.mean()) if len(passed_prob) > 0 else None,
                "median": float(passed_prob.median()) if len(passed_prob) > 0 else None,
                "min": float(passed_prob.min()) if len(passed_prob) > 0 else None,
                "max": float(passed_prob.max()) if len(passed_prob) > 0 else None,
            },
            "failed": {
                "count": int(len(failed_prob)),
                "mean": float(failed_prob.mean()) if len(failed_prob) > 0 else None,
                "median": float(failed_prob.median()) if len(failed_prob) > 0 else None,
                "min": float(failed_prob.min()) if len(failed_prob) > 0 else None,
                "max": float(failed_prob.max()) if len(failed_prob) > 0 else None,
            },
        }
    else:
        classifier_stats = None

    # Phase breakdown
    phase_counts = Counter(df["phase"].tolist())

    # Score statistics for passed vs failed
    score_stats = {
        "passed": {
            "mean": float(passed_df["score"].mean()) if len(passed_df) > 0 else None,
            "std": float(passed_df["score"].std()) if len(passed_df) > 0 else None,
        },
        "failed": {
            "mean": float(failed_df["score"].mean()) if len(failed_df) > 0 else None,
            "std": float(failed_df["score"].std()) if len(failed_df) > 0 else None,
        },
    }

    # Compute score penalty (difference between rank-1 score and selected score per iteration)
    score_penalties = []
    greedy_df = df[df["phase"] == "greedy"]
    for iteration in greedy_df["iteration"].unique():
        iter_df = greedy_df[greedy_df["iteration"] == iteration].sort_values("rank")
        if len(iter_df) > 0:
            rank1_score = iter_df.iloc[0]["score"]
            selected_row = iter_df[iter_df["passed"]]
            if len(selected_row) > 0:
                selected_score = selected_row.iloc[0]["score"]
                penalty = rank1_score - selected_score
                score_penalties.append(penalty)

    if score_penalties:
        score_penalty_stats = {
            "mean": float(np.mean(score_penalties)),
            "median": float(np.median(score_penalties)),
            "max": float(np.max(score_penalties)),
            "total": float(np.sum(score_penalties)),
        }
    else:
        score_penalty_stats = None

    return {
        "total_evaluated": int(total),
        "total_passed": int(passed),
        "total_failed": int(failed),
        "pass_rate": float(passed / total) if total > 0 else 0.0,
        "rejection_breakdown": {k: int(v) for k, v in rejection_counts.items()},
        "rank_distribution": rank_distribution,
        "phase_breakdown": {k: int(v) for k, v in phase_counts.items()},
        "resolution_stats": resolution_stats,
        "classifier_stats": classifier_stats,
        "score_stats": score_stats,
        "score_penalty_stats": score_penalty_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze image filtering results from stimulus selection"
    )
    add_standard_args(parser)
    args = parser.parse_args()

    # Setup
    payload, output_dir, _ = setup_from_args(args)

    # Get filter records from payload
    filter_records = payload.get("filter_records")
    if filter_records is None:
        print("No filter_records found in payload.")
        print("This may indicate filtering was not enabled during selection.")
        return

    print(f"Found {len(filter_records)} filter records")

    # Save full records to CSV
    records_df = pd.DataFrame(filter_records)
    records_path = output_dir / "filter_records.csv"
    records_df.to_csv(records_path, index=False)
    print(f"Saved filter records to {records_path}")

    # Compute and save summary
    summary = compute_filter_summary(filter_records)
    summary_path = output_dir / "filter_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved filter summary to {summary_path}")

    # Print summary
    print("\n=== Filter Summary ===")
    print(f"  Total evaluated: {summary['total_evaluated']}")
    print(f"  Passed: {summary['total_passed']}")
    print(f"  Failed: {summary['total_failed']}")
    print(f"  Pass rate: {summary['pass_rate']:.1%}")

    print("\n  Rejection breakdown:")
    for reason, count in sorted(
        summary["rejection_breakdown"].items(), key=lambda x: -x[1]
    ):
        print(f"    {reason}: {count}")

    print("\n  Rank distribution (of selected images):")
    for rank, count in summary["rank_distribution"].items():
        print(f"    Rank {rank}: {count}")

    if summary.get("resolution_stats"):
        print("\n  Resolution stats (max dimension):")
        res = summary["resolution_stats"]
        if res["passed"]["count"] > 0:
            print(
                f"    Passed: mean={res['passed']['mean']:.0f}, "
                f"median={res['passed']['median']:.0f}, "
                f"range=[{res['passed']['min']}, {res['passed']['max']}]"
            )
        if res["failed"]["count"] > 0:
            print(
                f"    Failed: mean={res['failed']['mean']:.0f}, "
                f"median={res['failed']['median']:.0f}, "
                f"range=[{res['failed']['min']}, {res['failed']['max']}]"
            )

    if summary.get("classifier_stats"):
        print("\n  Classifier probability stats:")
        clf = summary["classifier_stats"]
        if clf["passed"]["count"] > 0:
            print(
                f"    Passed: mean={clf['passed']['mean']:.3f}, "
                f"median={clf['passed']['median']:.3f}"
            )
        if clf["failed"]["count"] > 0:
            print(
                f"    Failed: mean={clf['failed']['mean']:.3f}, "
                f"median={clf['failed']['median']:.3f}"
            )

    if summary.get("score_penalty_stats"):
        print("\n  Score penalty from filtering:")
        pen = summary["score_penalty_stats"]
        print(f"    Mean: {pen['mean']:.4f}")
        print(f"    Median: {pen['median']:.4f}")
        print(f"    Max: {pen['max']:.4f}")
        print(f"    Total: {pen['total']:.4f}")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
