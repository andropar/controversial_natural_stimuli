#!/usr/bin/env python3
"""Generate filtering analysis plots from computed CSV/JSON results.

Reads:
- filter_records.csv
- filter_summary.json

Outputs:
- plots/filtering_rejection_reasons.pdf
- plots/filtering_resolution_dist.pdf
- plots/filtering_classifier_dist.pdf
- plots/filtering_rank_distribution.pdf
- plots/filtering_over_iterations.pdf
- plots/filtering_score_penalty.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def plot_rejection_reasons(
    summary: dict,
    output_path: Path,
):
    """Plot bar chart of rejection reasons.

    Args:
        summary: Filter summary dictionary
        output_path: Path to save plot
    """
    breakdown = summary.get("rejection_breakdown", {})
    if not breakdown:
        print("No rejection breakdown data, skipping plot")
        return

    # Sort by count descending
    items = sorted(breakdown.items(), key=lambda x: -x[1])
    reasons = [item[0] for item in items]
    counts = [item[1] for item in items]

    # Shorten reason names for display
    display_names = []
    for r in reasons:
        if r == "resolution_too_low":
            display_names.append("Resolution\nToo Low")
        elif r == "download_failed":
            display_names.append("Download\nFailed")
        elif r == "classifier_below_threshold":
            display_names.append("Classifier\nBelow Threshold")
        elif r == "url_not_found":
            display_names.append("URL Not\nFound")
        elif r == "image_name_not_found":
            display_names.append("Image Name\nNot Found")
        else:
            display_names.append(r.replace("_", "\n"))

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(reasons)))
    bars = ax.bar(range(len(reasons)), counts, color=colors, edgecolor="black", alpha=0.8)

    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_title("Image Rejection Reasons")
    ax.set_xticks(range(len(reasons)))
    ax.set_xticklabels(display_names, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.annotate(
            f"{count}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Add total in corner
    total_failed = summary.get("total_failed", sum(counts))
    ax.text(
        0.98, 0.95,
        f"Total failed: {total_failed}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved rejection reasons plot to {output_path}")


def plot_resolution_distribution(
    records_df: pd.DataFrame,
    output_path: Path,
):
    """Plot resolution distribution histogram comparing passed vs failed.

    Args:
        records_df: DataFrame with filter records
        output_path: Path to save plot
    """
    # Filter to records with resolution data
    has_res = records_df["width"].notna() & records_df["height"].notna()
    if not has_res.any():
        print("No resolution data, skipping plot")
        return

    df = records_df[has_res].copy()
    df["max_dim"] = df[["width", "height"]].max(axis=1)

    passed = df[df["passed"]]["max_dim"]
    failed = df[~df["passed"]]["max_dim"]

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(0, max(df["max_dim"].max(), 4096), 40)

    if len(passed) > 0:
        ax.hist(passed, bins=bins, alpha=0.6, label=f"Passed (n={len(passed)})", color="C0", edgecolor="black")
    if len(failed) > 0:
        ax.hist(failed, bins=bins, alpha=0.6, label=f"Failed (n={len(failed)})", color="C3", edgecolor="black")

    # Add vertical line for typical threshold
    ax.axvline(1000, color="red", linestyle="--", linewidth=2, label="Typical threshold (1000px)")

    ax.set_xlabel("Max Dimension (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Resolution Distribution: Passed vs Failed Images")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved resolution distribution to {output_path}")


def plot_classifier_distribution(
    records_df: pd.DataFrame,
    output_path: Path,
):
    """Plot classifier probability distribution histogram.

    Args:
        records_df: DataFrame with filter records
        output_path: Path to save plot
    """
    has_prob = records_df["natural_prob"].notna()
    if not has_prob.any():
        print("No classifier probability data, skipping plot")
        return

    df = records_df[has_prob]

    passed = df[df["passed"]]["natural_prob"]
    failed = df[~df["passed"]]["natural_prob"]

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(0, 1, 30)

    if len(passed) > 0:
        ax.hist(passed, bins=bins, alpha=0.6, label=f"Passed (n={len(passed)})", color="C0", edgecolor="black")
    if len(failed) > 0:
        ax.hist(failed, bins=bins, alpha=0.6, label=f"Failed (n={len(failed)})", color="C3", edgecolor="black")

    # Add vertical line for typical threshold
    ax.axvline(0.85, color="red", linestyle="--", linewidth=2, label="Typical threshold (0.85)")

    ax.set_xlabel("P(Natural)")
    ax.set_ylabel("Count")
    ax.set_title("Classifier Probability Distribution: Passed vs Failed")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved classifier distribution to {output_path}")


def plot_rank_distribution(
    summary: dict,
    output_path: Path,
):
    """Plot bar chart showing rank distribution of selected images.

    Args:
        summary: Filter summary dictionary
        output_path: Path to save plot
    """
    rank_dist = summary.get("rank_distribution", {})
    if not rank_dist or all(v == 0 for v in rank_dist.values()):
        print("No rank distribution data, skipping plot")
        return

    ranks = ["1", "2", "3", "4+"]
    counts = [rank_dist.get(r, 0) for r in ranks]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["C2", "C1", "C3", "C4"]
    bars = ax.bar(ranks, counts, color=colors, edgecolor="black", alpha=0.8)

    ax.set_xlabel("Rank of Selected Candidate")
    ax.set_ylabel("Count")
    ax.set_title("How Often Was the Top-Ranked Candidate Selected?")
    ax.grid(True, alpha=0.3, axis="y")

    # Add count labels
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.annotate(
                f"{count}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

    # Add percentage annotation
    total = sum(counts)
    if total > 0 and counts[0] > 0:
        pct = counts[0] / total * 100
        ax.text(
            0.98, 0.95,
            f"1st choice selected: {pct:.1f}%",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved rank distribution to {output_path}")


def plot_filtering_over_iterations(
    records_df: pd.DataFrame,
    output_path: Path,
):
    """Plot cumulative failures over iterations.

    Args:
        records_df: DataFrame with filter records
        output_path: Path to save plot
    """
    greedy_df = records_df[records_df["phase"] == "greedy"]
    if greedy_df.empty:
        print("No greedy phase data, skipping iterations plot")
        return

    # Count failures per iteration by reason
    failed_df = greedy_df[~greedy_df["passed"]]

    if failed_df.empty:
        print("No failures in greedy phase, skipping iterations plot")
        return

    # Get unique reasons and iterations
    reasons = failed_df["reason"].unique()
    max_iter = int(greedy_df["iteration"].max())

    fig, ax = plt.subplots(figsize=(12, 5))

    # Cumulative count by reason
    for i, reason in enumerate(reasons):
        reason_df = failed_df[failed_df["reason"] == reason]
        cumulative = []
        for it in range(max_iter + 1):
            count = (reason_df["iteration"] <= it).sum()
            cumulative.append(count)

        # Shorten label
        label = reason.replace("_", " ").title()
        if len(label) > 20:
            label = label[:17] + "..."

        ax.plot(range(max_iter + 1), cumulative, label=label, linewidth=2, marker="o", markersize=3)

    ax.set_xlabel("Greedy Iteration")
    ax.set_ylabel("Cumulative Failures")
    ax.set_title("Cumulative Filter Failures Over Greedy Selection")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved filtering over iterations to {output_path}")


def _compute_per_track_penalty(rank1_scores_per_track: dict, selected_scores_per_track: dict) -> float:
    """Compute penalty as sum of per-track differences.

    Args:
        rank1_scores_per_track: Dict of track_name -> score for rank-1 candidate
        selected_scores_per_track: Dict of track_name -> score for selected candidate

    Returns:
        Sum of (rank1_track_i - selected_track_i) across all tracks
    """
    penalty = 0.0
    for track_name in rank1_scores_per_track:
        if track_name in selected_scores_per_track:
            penalty += rank1_scores_per_track[track_name] - selected_scores_per_track[track_name]
    return penalty


def plot_score_penalty(
    records_df: pd.DataFrame,
    output_path: Path,
):
    """Plot score penalty from filtering (diff between rank-1 and selected score).

    Uses raw per-track scores when available for interpretable penalties,
    falling back to combined z-scored scores for backward compatibility.

    Args:
        records_df: DataFrame with filter records
        output_path: Path to save plot
    """
    greedy_df = records_df[records_df["phase"] == "greedy"]
    if greedy_df.empty:
        print("No greedy phase data, skipping score penalty plot")
        return

    # Check if per-track scores are available
    has_per_track = (
        "scores_per_track" in greedy_df.columns
        and greedy_df["scores_per_track"].notna().any()
    )
    use_raw_scores = has_per_track

    if use_raw_scores:
        print("Using raw per-track scores for penalty computation")
    else:
        print("Using combined z-scored scores (per-track scores not available)")

    # Compute penalty per iteration
    penalties = []
    iterations = []

    for iteration in sorted(greedy_df["iteration"].unique()):
        iter_df = greedy_df[greedy_df["iteration"] == iteration].sort_values("rank")
        if len(iter_df) > 0:
            selected_row = iter_df[iter_df["passed"]]
            if len(selected_row) > 0:
                if use_raw_scores:
                    # Use raw per-track scores
                    rank1_scores_per_track = iter_df.iloc[0]["scores_per_track"]
                    selected_scores_per_track = selected_row.iloc[0]["scores_per_track"]
                    if rank1_scores_per_track and selected_scores_per_track:
                        penalty = _compute_per_track_penalty(
                            rank1_scores_per_track, selected_scores_per_track
                        )
                        penalties.append(penalty)
                        iterations.append(iteration)
                else:
                    # Fall back to combined z-scored scores
                    rank1_score = iter_df.iloc[0]["score"]
                    selected_score = selected_row.iloc[0]["score"]
                    penalty = rank1_score - selected_score
                    penalties.append(penalty)
                    iterations.append(iteration)

    if not penalties:
        print("No score penalty data computed, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Penalty over iterations
    ax1 = axes[0]
    ax1.scatter(iterations, penalties, alpha=0.6, c="C3", s=30)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Add trend line
    if len(iterations) > 1:
        z = np.polyfit(iterations, penalties, 1)
        p = np.poly1d(z)
        ax1.plot(iterations, p(iterations), "r--", alpha=0.8, label="Trend")

    ax1.set_xlabel("Greedy Iteration")
    ylabel = "Raw Score Penalty (sum of per-track)" if use_raw_scores else "Score Penalty (z-scored)"
    ax1.set_ylabel(ylabel)
    title = "Raw Score Penalty from Filtering" if use_raw_scores else "Score Penalty from Filtering (z-scored)"
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Histogram of penalties
    ax2 = axes[1]
    ax2.hist(penalties, bins=30, color="C3", alpha=0.7, edgecolor="black")
    ax2.axvline(np.mean(penalties), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(penalties):.4f}")
    ax2.axvline(np.median(penalties), color="orange", linestyle=":", linewidth=2, label=f"Median: {np.median(penalties):.4f}")

    ax2.set_xlabel(ylabel)
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Score Penalties")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Add total penalty
    total = sum(penalties)
    ax2.text(
        0.98, 0.95,
        f"Total penalty: {total:.4f}",
        transform=ax2.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved score penalty plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate filtering analysis plots"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing filter_records.csv and filter_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <input-dir>/plots/)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format (default: pdf)",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = args.output_dir or (args.input_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load filter records CSV
    records_path = args.input_dir / "filter_records.csv"
    if not records_path.exists():
        print(f"Error: {records_path} not found")
        print("Run 09_analyze_filtering.py first to generate filter records")
        return

    print(f"Loading {records_path}...")
    records_df = pd.read_csv(records_path)
    print(f"Loaded {len(records_df)} filter records")

    # Load filter summary JSON
    summary_path = args.input_dir / "filter_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        print(f"Warning: {summary_path} not found, some plots may be skipped")
        summary = {}

    # Generate plots
    fmt = args.format

    # 1. Rejection reasons
    plot_rejection_reasons(summary, output_dir / f"filtering_rejection_reasons.{fmt}")

    # 2. Resolution distribution
    plot_resolution_distribution(records_df, output_dir / f"filtering_resolution_dist.{fmt}")

    # 3. Classifier probability distribution
    plot_classifier_distribution(records_df, output_dir / f"filtering_classifier_dist.{fmt}")

    # 4. Rank distribution
    plot_rank_distribution(summary, output_dir / f"filtering_rank_distribution.{fmt}")

    # 5. Filtering over iterations
    plot_filtering_over_iterations(records_df, output_dir / f"filtering_over_iterations.{fmt}")

    # 6. Score penalty
    plot_score_penalty(records_df, output_dir / f"filtering_score_penalty.{fmt}")

    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
