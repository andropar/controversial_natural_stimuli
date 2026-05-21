#!/usr/bin/env python3
"""Extract and analyze selection scores from stimulus selection results.

Outputs:
- greedy_scores.csv: Score progression during greedy selection phase
- refinement.csv: Refinement pass details (if refinement was used)
- summary.csv: High-level selection summary statistics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.utils import (
    load_selection_payload,
    get_output_dir,
    get_all_tracks_for_evaluation,
)


def compute_raw_combined_score(
    per_track_scores: dict[str, list[float]],
) -> list[float]:
    """Compute raw combined score as simple average of per-track scores.

    Args:
        per_track_scores: Dict mapping track name to list of scores (one per iteration)

    Returns:
        List of raw combined scores (one per iteration)
    """
    if not per_track_scores:
        return []

    track_names = list(per_track_scores.keys())
    n_iterations = len(next(iter(per_track_scores.values())))

    raw_combined = []
    for i in range(n_iterations):
        scores_at_iter = [per_track_scores[name][i] for name in track_names]
        combined = sum(scores_at_iter) / len(scores_at_iter)
        raw_combined.append(combined)

    return raw_combined


def extract_greedy_scores(payload: dict) -> pd.DataFrame:
    """Extract greedy phase scores from payload.

    Args:
        payload: Selection payload dictionary

    Returns:
        DataFrame with columns: iteration, score_combined, score_combined_raw, score_<track>, n_selected
    """
    scores_combined = payload.get("scores", [])
    init_size = payload.get("config", {}).get("init_size", 3)

    rows = []
    for i, score in enumerate(scores_combined):
        row = {
            "iteration": i + 1,
            "score_combined": score,
            "n_selected": init_size + i + 1,
        }
        rows.append(row)

    # Add per-track scores if available
    scores_per_view = payload.get("scores_per_view_history") or {}
    scores_per_rep = payload.get("scores_per_rep_history") or {}

    # Merge both sources
    all_track_scores = {**scores_per_view, **scores_per_rep}

    for track_name, track_scores in all_track_scores.items():
        col_name = f"score_{track_name}"
        for i, score in enumerate(track_scores):
            if i < len(rows):
                rows[i][col_name] = score

    # Compute raw combined score (simple average of per-track scores)
    if all_track_scores:
        raw_combined = compute_raw_combined_score(all_track_scores)
        for i, score in enumerate(raw_combined):
            if i < len(rows):
                rows[i]["score_combined_raw"] = score

    return pd.DataFrame(rows)


def extract_refinement_history(payload: dict) -> pd.DataFrame | None:
    """Extract refinement history from payload.

    Args:
        payload: Selection payload dictionary

    Returns:
        DataFrame with refinement details, or None if no refinement was done
    """
    refinement_history = payload.get("refinement_history")
    if not refinement_history:
        return None

    rows = []
    for record in refinement_history:
        row = {
            "pass_num": record.get("pass", record.get("pass_num", 0)),
            "position": record.get("position", 0),
            "old_idx": record.get("old_idx", -1),
            "new_idx": record.get("new_idx", -1),
            "score": record.get("score", 0.0),
            "replaced": record.get("replaced", False),
        }
        # Add per-track scores if available
        scores_per_track = record.get("scores_per_track")
        if scores_per_track:
            for track_name, track_score in scores_per_track.items():
                row[f"score_{track_name}"] = track_score
            # Compute raw combined score (simple average)
            per_track_dict = {k: [v] for k, v in scores_per_track.items()}
            raw_combined = compute_raw_combined_score(per_track_dict)
            if raw_combined:
                row["score_combined_raw"] = raw_combined[0]
        rows.append(row)

    return pd.DataFrame(rows)


def compute_summary_stats(
    payload: dict,
    greedy_df: pd.DataFrame,
    refinement_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compute high-level summary statistics.

    Args:
        payload: Selection payload dictionary
        greedy_df: Greedy scores DataFrame
        refinement_df: Refinement DataFrame (or None)

    Returns:
        DataFrame with key-value pairs
    """
    config = payload.get("config", {})
    model_names = payload.get("model_names", [])
    selected_indices = payload.get("selected_global_indices", [])

    stats = {
        "n_selected": len(selected_indices),
        "n_models": len(model_names),
        "init_size": config.get("init_size", 3),
        "target_size": config.get("target_size", 100),
        "method": config.get("method_name", "unknown"),
        "model_set": config.get("model_set_name", "unknown"),
        "metric": config.get("metric", "cosine"),
        "corr_type": config.get("corr_type", "correlation"),
        "noise_ceiling_target": config.get("noise_ceiling_target", 0.46),
        "seed": config.get("seed", 42),
    }

    # Greedy phase stats
    if not greedy_df.empty:
        stats["final_score"] = greedy_df["score_combined"].iloc[-1]
        stats["initial_score"] = greedy_df["score_combined"].iloc[0]
        stats["score_improvement"] = stats["final_score"] - stats["initial_score"]
        stats["n_greedy_iterations"] = len(greedy_df)

    # Refinement stats
    if refinement_df is not None and not refinement_df.empty:
        stats["n_refinement_passes"] = refinement_df["pass_num"].max() + 1
        stats["n_replacements"] = refinement_df["replaced"].sum()
        stats["refinement_enabled"] = True

        # Per-pass breakdown
        for pass_num in refinement_df["pass_num"].unique():
            pass_df = refinement_df[refinement_df["pass_num"] == pass_num]
            stats[f"pass_{pass_num}_replacements"] = pass_df["replaced"].sum()
    else:
        stats["refinement_enabled"] = False
        stats["n_refinement_passes"] = 0
        stats["n_replacements"] = 0

    # Track aggregation info
    track_agg = payload.get("track_aggregation", {})
    stats["track_agg_method"] = track_agg.get("agg_method", "unknown")
    stats["track_norm_method"] = track_agg.get("norm_method", "unknown")
    if track_agg.get("raw_weight") is not None:
        stats["raw_weight"] = track_agg["raw_weight"]

    # Track info
    tracks = get_all_tracks_for_evaluation(payload)
    stats["n_tracks"] = len(tracks)
    stats["track_names"] = ",".join(t["name"] for t in tracks)

    # Convert to DataFrame
    rows = [{"key": k, "value": v} for k, v in stats.items()]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze selection scores"
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Path to selection result directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <result-dir>/eval_pipeline/)",
    )
    args = parser.parse_args()

    # Load payload
    print(f"Loading payload from {args.result_dir}")
    payload = load_selection_payload(args.result_dir)

    # Setup output directory
    output_dir = args.output_dir or get_output_dir(args.result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract greedy scores
    print("Extracting greedy phase scores...")
    greedy_df = extract_greedy_scores(payload)
    greedy_path = output_dir / "greedy_scores.csv"
    greedy_df.to_csv(greedy_path, index=False)
    print(f"  Saved {len(greedy_df)} iterations to {greedy_path}")

    # Extract refinement history
    print("Extracting refinement history...")
    refinement_df = extract_refinement_history(payload)
    refinement_path = output_dir / "refinement.csv"
    if refinement_df is not None:
        refinement_df.to_csv(refinement_path, index=False)
        n_replaced = refinement_df["replaced"].sum()
        n_passes = refinement_df["pass_num"].max() + 1
        print(f"  Saved {len(refinement_df)} records ({n_replaced} replacements, {n_passes} passes) to {refinement_path}")
    else:
        # Write empty file with headers
        pd.DataFrame(columns=["pass_num", "position", "old_idx", "new_idx", "score", "replaced"]).to_csv(
            refinement_path, index=False
        )
        print(f"  No refinement history found, wrote empty file to {refinement_path}")

    # Compute summary stats
    print("Computing summary statistics...")
    summary_df = compute_summary_stats(payload, greedy_df, refinement_df)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved {len(summary_df)} stats to {summary_path}")

    # Print key stats
    print("\n=== Summary ===")
    for _, row in summary_df.iterrows():
        print(f"  {row['key']}: {row['value']}")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
