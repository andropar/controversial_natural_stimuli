#!/usr/bin/env python3
"""Compute statistical significance tests for evaluation results.

Reads:
- discriminability.csv

Outputs:
- statistics.csv: Statistical test results (t-test, effect size, p-values)
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


def compute_statistics_for_track(
    discrim_df: pd.DataFrame,
    track: str,
) -> dict:
    """Compute statistical tests for a single track.

    Args:
        discrim_df: Discriminability DataFrame
        track: Track name

    Returns:
        Dict with statistical test results
    """
    track_data = discrim_df[discrim_df["track"] == track]

    # Get AUC values
    selected_auc = track_data[track_data["subset_type"] == "selected"]["auc"].iloc[0]
    random_auc = track_data[track_data["subset_type"] == "random"]["auc"].iloc[0]

    # Get error probabilities at each noise level for paired comparison
    selected_data = track_data[track_data["subset_type"] == "selected"].sort_values("noise_mult")
    random_data = track_data[track_data["subset_type"] == "random"].sort_values("noise_mult")

    selected_errors = selected_data["error_prob"].values
    random_errors = random_data["error_prob"].values
    random_stds = random_data["error_prob_std"].values if "error_prob_std" in random_data.columns else None

    results = {
        "track": track,
        "selected_auc": selected_auc,
        "random_auc": random_auc,
        "auc_diff": random_auc - selected_auc,
        "auc_improvement_pct": (random_auc - selected_auc) / random_auc * 100 if random_auc > 0 else 0,
    }

    # Paired t-test on error probabilities across noise levels
    if len(selected_errors) == len(random_errors) and len(selected_errors) > 1:
        t_stat, p_value = stats.ttest_rel(random_errors, selected_errors)
        results["ttest_t"] = t_stat
        results["ttest_p"] = p_value
        results["ttest_significant"] = p_value < 0.05

        # Effect size (Cohen's d for paired samples)
        diff = random_errors - selected_errors
        cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
        results["cohens_d"] = cohens_d

        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            w_stat, w_p = stats.wilcoxon(random_errors, selected_errors, alternative='greater')
            results["wilcoxon_w"] = w_stat
            results["wilcoxon_p"] = w_p
        except ValueError:
            # Can fail if all differences are zero
            results["wilcoxon_w"] = np.nan
            results["wilcoxon_p"] = np.nan

    # If we have std from random subsets, compute z-score at target noise ceiling
    if random_stds is not None:
        # Find the noise level closest to NC=0.46 (multiplier ~1.0)
        noise_mults = random_data["noise_mult"].values
        target_idx = np.argmin(np.abs(noise_mults - 1.0))

        selected_at_target = selected_errors[target_idx]
        random_at_target = random_errors[target_idx]
        std_at_target = random_stds[target_idx] if not np.isnan(random_stds[target_idx]) else 0.1

        if std_at_target > 0:
            z_score = (random_at_target - selected_at_target) / std_at_target
            # One-sided p-value (selected should be lower)
            p_value_z = 1 - stats.norm.cdf(z_score)
            results["z_score_at_target"] = z_score
            results["p_value_at_target"] = p_value_z

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistical significance tests"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing discriminability.csv",
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

    # Load discriminability data
    discrim_path = input_dir / "discriminability.csv"
    if not discrim_path.exists():
        print(f"Error: {discrim_path} not found")
        return

    print(f"Loading {discrim_path}...")
    discrim_df = pd.read_csv(discrim_path)

    # Compute statistics for each track
    tracks = discrim_df["track"].unique()
    results = []

    for track in tracks:
        print(f"Computing statistics for track '{track}'...")
        track_results = compute_statistics_for_track(discrim_df, track)
        results.append(track_results)

    # Compute aggregate statistics
    all_selected_errors = []
    all_random_errors = []

    for track in tracks:
        track_data = discrim_df[discrim_df["track"] == track]
        selected_data = track_data[track_data["subset_type"] == "selected"].sort_values("noise_mult")
        random_data = track_data[track_data["subset_type"] == "random"].sort_values("noise_mult")
        all_selected_errors.extend(selected_data["error_prob"].values)
        all_random_errors.extend(random_data["error_prob"].values)

    all_selected_errors = np.array(all_selected_errors)
    all_random_errors = np.array(all_random_errors)

    # Aggregate t-test
    t_stat, p_value = stats.ttest_rel(all_random_errors, all_selected_errors)
    diff = all_random_errors - all_selected_errors
    cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0

    aggregate_results = {
        "track": "AGGREGATE",
        "selected_auc": np.mean([r["selected_auc"] for r in results]),
        "random_auc": np.mean([r["random_auc"] for r in results]),
        "auc_diff": np.mean([r["auc_diff"] for r in results]),
        "auc_improvement_pct": np.mean([r["auc_improvement_pct"] for r in results]),
        "ttest_t": t_stat,
        "ttest_p": p_value,
        "ttest_significant": p_value < 0.05,
        "cohens_d": cohens_d,
    }
    results.append(aggregate_results)

    # Save results
    stats_df = pd.DataFrame(results)
    stats_path = output_dir / "statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to {stats_path}")

    # Print summary
    print("\n=== Statistical Summary ===")
    print(f"{'Track':<20} {'AUC Diff':>10} {'t-stat':>10} {'p-value':>12} {'Cohen d':>10} {'Sig?':>6}")
    print("-" * 70)
    for r in results:
        sig = "***" if r.get("ttest_p", 1) < 0.001 else ("**" if r.get("ttest_p", 1) < 0.01 else ("*" if r.get("ttest_p", 1) < 0.05 else ""))
        print(f"{r['track']:<20} {r['auc_diff']:>10.3f} {r.get('ttest_t', np.nan):>10.2f} {r.get('ttest_p', np.nan):>12.2e} {r.get('cohens_d', np.nan):>10.2f} {sig:>6}")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
