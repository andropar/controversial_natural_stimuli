#!/usr/bin/env python3
"""Compute ablation analyses for stimulus selection.

Analyzes:
1. Effect of refinement phase
2. Per-track contributions to selection
3. Score decomposition

Outputs:
- ablations.csv: Summary of ablation results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analyze_refinement_effect(
    greedy_df: pd.DataFrame,
    refinement_df: pd.DataFrame,
) -> dict:
    """Analyze the effect of refinement phase.

    Args:
        greedy_df: Greedy phase scores DataFrame
        refinement_df: Refinement phase DataFrame

    Returns:
        Dict with refinement analysis results
    """
    results = {}

    # Get final greedy score
    if "score_combined_raw" in greedy_df.columns:
        greedy_final = greedy_df["score_combined_raw"].iloc[-1]
        results["greedy_final_score"] = greedy_final
    else:
        greedy_final = None

    # Get refinement data
    if refinement_df is not None and not refinement_df.empty:
        replaced = refinement_df[refinement_df["replaced"]]

        if not replaced.empty and "score_combined_raw" in replaced.columns:
            refinement_final = replaced["score_combined_raw"].iloc[-1]
            results["refinement_final_score"] = refinement_final

            if greedy_final is not None:
                improvement = refinement_final - greedy_final
                pct_improvement = improvement / greedy_final * 100 if greedy_final > 0 else 0
                results["refinement_improvement"] = improvement
                results["refinement_improvement_pct"] = pct_improvement

        # Analyze convergence
        n_replacements = len(replaced)
        results["n_replacements"] = n_replacements

        if "pass_num" in refinement_df.columns:
            passes = refinement_df["pass_num"].unique()
            n_passes = len(passes)
            results["n_passes"] = n_passes

            # Replacements per pass
            for p in sorted(passes):
                pass_replaced = refinement_df[
                    (refinement_df["pass_num"] == p) &
                    (refinement_df["replaced"])
                ]
                results[f"pass_{p}_replacements"] = len(pass_replaced)

            # Convergence rate (exponential decay fit)
            if n_passes > 1:
                pass_counts = [
                    len(refinement_df[
                        (refinement_df["pass_num"] == p) &
                        (refinement_df["replaced"])
                    ])
                    for p in sorted(passes)
                ]
                # Simple decay ratio
                if pass_counts[0] > 0:
                    decay_ratio = pass_counts[-1] / pass_counts[0] if pass_counts[0] > 0 else 0
                    results["convergence_ratio"] = decay_ratio

    return results


def analyze_track_contributions(
    greedy_df: pd.DataFrame,
) -> list[dict]:
    """Analyze per-track score contributions.

    Args:
        greedy_df: Greedy phase scores DataFrame

    Returns:
        List of dicts with per-track analysis
    """
    results = []

    # Find track score columns
    score_cols = [c for c in greedy_df.columns if c.startswith("score_") and c != "score_combined_raw"]

    for col in score_cols:
        track_name = col.replace("score_", "")
        initial = greedy_df[col].iloc[0]
        final = greedy_df[col].iloc[-1]

        improvement = initial - final  # Lower is better
        pct_improvement = improvement / initial * 100 if initial > 0 else 0

        results.append({
            "track": track_name,
            "initial_score": initial,
            "final_score": final,
            "improvement": improvement,
            "improvement_pct": pct_improvement,
        })

    return results


def analyze_selection_trajectory(
    greedy_df: pd.DataFrame,
) -> dict:
    """Analyze the selection trajectory.

    Args:
        greedy_df: Greedy phase scores DataFrame

    Returns:
        Dict with trajectory analysis
    """
    results = {}

    if "score_combined_raw" not in greedy_df.columns:
        return results

    scores = greedy_df["score_combined_raw"].values
    n_iter = len(scores)

    results["n_iterations"] = n_iter
    results["initial_score"] = scores[0]
    results["final_score"] = scores[-1]
    results["total_improvement"] = scores[0] - scores[-1]

    # Find when most improvement happened
    improvements = np.diff(scores) * -1  # Negative because lower is better
    cumulative = np.cumsum(improvements)
    total = cumulative[-1] if len(cumulative) > 0 else 0

    if total > 0:
        # Find iteration where we hit 50%, 90% of improvement
        for pct, label in [(0.5, "50pct"), (0.9, "90pct")]:
            threshold = pct * total
            idx = np.searchsorted(cumulative, threshold)
            results[f"iter_at_{label}_improvement"] = min(idx + 1, n_iter)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute ablation analyses"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing greedy_scores.csv and refinement.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input-dir>)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    all_results = []

    # Load greedy scores
    greedy_path = input_dir / "greedy_scores.csv"
    greedy_df = None
    if greedy_path.exists():
        print(f"Loading {greedy_path}...")
        greedy_df = pd.read_csv(greedy_path)

    # Load refinement data
    refinement_path = input_dir / "refinement.csv"
    refinement_df = None
    if refinement_path.exists():
        print(f"Loading {refinement_path}...")
        refinement_df = pd.read_csv(refinement_path)

    # Analyze refinement effect
    print("\n=== Refinement Ablation ===")
    if greedy_df is not None:
        refinement_results = analyze_refinement_effect(greedy_df, refinement_df)
        for k, v in refinement_results.items():
            all_results.append({"analysis": "refinement", "metric": k, "value": v})
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Analyze per-track contributions
    print("\n=== Per-Track Contributions ===")
    if greedy_df is not None:
        track_results = analyze_track_contributions(greedy_df)
        print(f"{'Track':<20} {'Initial':>10} {'Final':>10} {'Improvement':>12} {'% Improv':>10}")
        print("-" * 65)
        for tr in track_results:
            print(f"{tr['track']:<20} {tr['initial_score']:>10.3f} {tr['final_score']:>10.3f} {tr['improvement']:>12.3f} {tr['improvement_pct']:>9.1f}%")
            for k, v in tr.items():
                all_results.append({"analysis": f"track_{tr['track']}", "metric": k, "value": v})

    # Analyze trajectory
    print("\n=== Selection Trajectory ===")
    if greedy_df is not None:
        traj_results = analyze_selection_trajectory(greedy_df)
        for k, v in traj_results.items():
            all_results.append({"analysis": "trajectory", "metric": k, "value": v})
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Save results
    if all_results:
        ablations_df = pd.DataFrame(all_results)
        ablations_path = output_dir / "ablations.csv"
        ablations_df.to_csv(ablations_path, index=False)
        print(f"\nSaved ablation results to {ablations_path}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
