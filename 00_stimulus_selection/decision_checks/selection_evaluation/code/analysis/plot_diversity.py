#!/usr/bin/env python3
"""Generate diversity plots from computed CSV results.

Reads:
- diversity.csv

Outputs:
- plots/diversity_histogram.pdf: Histogram of pairwise similarities
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def plot_diversity_summary(
    diversity_df: pd.DataFrame,
    output_path: Path,
):
    """Plot diversity summary bar chart.

    Args:
        diversity_df: DataFrame with diversity metrics
        output_path: Path to save PDF
    """
    if diversity_df.empty:
        print("No diversity data, skipping plot")
        return

    row = diversity_df.iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Pairwise similarity comparison
    ax1 = axes[0]
    labels = ["Selected"]
    values = [row["mean_pairwise_sim"]]
    errors = [row["std_pairwise_sim"]]
    colors = ["C0"]

    if "random_mean_pairwise_sim" in row and pd.notna(row["random_mean_pairwise_sim"]):
        labels.append("Random")
        values.append(row["random_mean_pairwise_sim"])
        errors.append(row.get("random_std_pairwise_sim", 0))
        colors.append("C1")

    x = np.arange(len(labels))
    bars = ax1.bar(x, values, yerr=errors, color=colors, alpha=0.8, capsize=5)

    ax1.set_xlabel("")
    ax1.set_ylabel("Mean Pairwise Cosine Similarity")
    ax1.set_title("Image Similarity Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, values):
        ax1.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    # Panel 2: Feature spread metrics
    ax2 = axes[1]
    spread_metrics = {
        "Feature\nVariance": row.get("mean_feature_variance", 0),
        "Feature\nEntropy": row.get("feature_entropy", 0),
    }

    x2 = np.arange(len(spread_metrics))
    bars2 = ax2.bar(x2, list(spread_metrics.values()), color="C2", alpha=0.8)

    ax2.set_xlabel("")
    ax2.set_ylabel("Value")
    ax2.set_title("Feature Spread Metrics")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(list(spread_metrics.keys()))
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars2, spread_metrics.values()):
        ax2.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diversity summary to {output_path}")


def plot_pairwise_histogram(
    pairwise_df: pd.DataFrame,
    output_path: Path,
):
    """Plot histogram of pairwise similarities.

    Args:
        pairwise_df: DataFrame with pairwise similarities
        output_path: Path to save PDF
    """
    if pairwise_df.empty:
        print("No pairwise data, skipping histogram")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    similarities = pairwise_df["similarity"].values

    ax.hist(similarities, bins=30, color="C0", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(similarities), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(similarities):.3f}")
    ax.axvline(np.median(similarities), color="orange", linestyle=":", linewidth=2, label=f"Median: {np.median(similarities):.3f}")

    ax.set_xlabel("Pairwise Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Pairwise Similarities (Selected Stimuli)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pairwise histogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate diversity plots from CSVs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing diversity.csv",
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

    # Load and plot diversity summary
    diversity_path = args.input_dir / "diversity.csv"
    if diversity_path.exists():
        print(f"Loading {diversity_path}...")
        diversity_df = pd.read_csv(diversity_path)
        plot_diversity_summary(
            diversity_df,
            output_dir / f"diversity_summary.{args.format}",
        )
    else:
        print(f"Warning: {diversity_path} not found, skipping diversity summary plot")

    # Load and plot pairwise histogram if available
    pairwise_path = args.input_dir / "pairwise_similarities.csv"
    if pairwise_path.exists():
        print(f"Loading {pairwise_path}...")
        pairwise_df = pd.read_csv(pairwise_path)
        plot_pairwise_histogram(
            pairwise_df,
            output_dir / f"pairwise_histogram.{args.format}",
        )
    else:
        print(f"Note: {pairwise_path} not found (run with --save-pairwise to generate)")

    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
