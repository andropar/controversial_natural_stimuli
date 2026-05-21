#!/usr/bin/env python3
"""Generate discriminability plots from computed CSV results.

Reads:
- discriminability.csv
- correlation_matrices.csv

Outputs:
- plots/discriminability_curves.pdf: Error prob vs noise level (selected vs random)
- plots/auc_comparison.pdf: Bar chart of AUC across tracks
- plots/correlation_matrices_<track>.pdf: Heatmaps per track
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def plot_discriminability_curves(
    discrim_df: pd.DataFrame,
    output_path: Path,
    n_models: int = 6,
):
    """Plot error probability vs noise ceiling curves.

    Args:
        discrim_df: Discriminability DataFrame with columns:
            track, noise_ceiling, subset_type, error_prob, error_prob_std
        output_path: Path to save PDF
        n_models: Number of models (for computing chance level)
    """
    tracks = discrim_df["track"].unique()
    n_tracks = len(tracks)

    # Chance level for M-class discrimination: (M-1)/M
    chance_level = (n_models - 1) / n_models

    # Determine x-axis column (prefer noise_ceiling if available)
    x_col = "noise_ceiling" if "noise_ceiling" in discrim_df.columns else "noise_mult"
    x_label = "Noise Ceiling" if x_col == "noise_ceiling" else "Noise Multiplier"

    # Determine grid layout
    n_cols = min(3, n_tracks)
    n_rows = (n_tracks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()

    for idx, track in enumerate(tracks):
        ax = axes[idx]
        track_data = discrim_df[discrim_df["track"] == track]

        # Plot selected
        selected = track_data[track_data["subset_type"] == "selected"]
        selected = selected.sort_values(x_col)
        ax.plot(
            selected[x_col],
            selected["error_prob"],
            "o-",
            color="C0",
            linewidth=2,
            label="Selected",
            markersize=4,
        )

        # Plot random with error bars if available
        random_data = track_data[track_data["subset_type"] == "random"]
        random_data = random_data.sort_values(x_col)
        ax.plot(
            random_data[x_col],
            random_data["error_prob"],
            "s--",
            color="C1",
            linewidth=2,
            label="Random",
            markersize=4,
            alpha=0.7,
        )

        # Add error band for random baseline if std available
        if "error_prob_std" in random_data.columns:
            std = random_data["error_prob_std"].fillna(0)
            if std.sum() > 0:  # Only plot if we have actual std values
                ax.fill_between(
                    random_data[x_col],
                    random_data["error_prob"] - std,
                    random_data["error_prob"] + std,
                    color="C1",
                    alpha=0.2,
                )

        # Reference line at chance level: (n_models-1)/n_models
        ax.axhline(chance_level, color="gray", linestyle=":", alpha=0.5, label=f"Chance ({chance_level:.2f})")

        # Use log scale only for noise_mult, not noise_ceiling
        if x_col == "noise_mult":
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Error Probability")
        ax.set_title(track)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    # Hide unused axes
    for idx in range(n_tracks, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved discriminability curves to {output_path}")


def plot_auc_comparison(
    discrim_df: pd.DataFrame,
    output_path: Path,
):
    """Plot AUC comparison bar chart.

    Args:
        discrim_df: Discriminability DataFrame
        output_path: Path to save PDF
    """
    # Get AUC values (one per track/subset_type)
    auc_data = discrim_df.groupby(["track", "subset_type"])["auc"].first().reset_index()

    # Pivot for grouped bar chart
    auc_pivot = auc_data.pivot(index="track", columns="subset_type", values="auc")

    # Reorder columns if both exist
    cols = []
    if "selected" in auc_pivot.columns:
        cols.append("selected")
    if "random" in auc_pivot.columns:
        cols.append("random")
    auc_pivot = auc_pivot[cols]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(max(6, len(auc_pivot) * 1.2), 5))

    x = np.arange(len(auc_pivot))
    width = 0.35

    colors = {"selected": "C0", "random": "C1"}
    for i, col in enumerate(auc_pivot.columns):
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            auc_pivot[col].values,
            width,
            label=col.capitalize(),
            color=colors.get(col, f"C{i}"),
            alpha=0.8,
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Track")
    ax.set_ylabel("AUC (Error Probability)")
    ax.set_title("Discriminability AUC by Track")
    ax.set_xticks(x)
    ax.set_xticklabels(auc_pivot.index, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Lower is better for error probability AUC
    ax.set_ylim(0, max(auc_pivot.values.max() * 1.2, 0.5))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC comparison to {output_path}")


def plot_correlation_matrices(
    corr_df: pd.DataFrame,
    output_dir: Path,
):
    """Plot correlation matrix heatmaps for each track.

    Args:
        corr_df: Correlation DataFrame with columns:
            track, matrix_type, model_i, model_j, correlation
        output_dir: Directory to save PDFs
    """
    tracks = corr_df["track"].unique()

    for track in tracks:
        track_data = corr_df[corr_df["track"] == track]
        matrix_types = track_data["matrix_type"].unique()
        model_names = track_data["model_i"].unique()
        n_models = len(model_names)

        # Create subplots for each matrix type
        n_types = len(matrix_types)
        fig, axes = plt.subplots(
            1, n_types,
            figsize=(4 * n_types + 1, 4),
            squeeze=False,
        )
        axes = axes.flatten()

        for idx, matrix_type in enumerate(sorted(matrix_types)):
            ax = axes[idx]
            type_data = track_data[track_data["matrix_type"] == matrix_type]

            # Reconstruct matrix
            matrix = np.zeros((n_models, n_models))
            model_to_idx = {m: i for i, m in enumerate(model_names)}

            for _, row in type_data.iterrows():
                i = model_to_idx[row["model_i"]]
                j = model_to_idx[row["model_j"]]
                matrix[i, j] = row["correlation"]

            # Plot heatmap
            sns.heatmap(
                matrix,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="RdYlBu_r",
                vmin=0,
                vmax=1,
                xticklabels=[m.split("_")[-1][:8] for m in model_names],
                yticklabels=[m.split("_")[-1][:8] for m in model_names],
                cbar=idx == len(matrix_types) - 1,
            )
            ax.set_title(matrix_type.replace("_", " ").title())

        fig.suptitle(f"Correlation Matrices: {track}", fontsize=12)
        plt.tight_layout()

        output_path = output_dir / f"correlation_matrices_{track}.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved correlation matrices for '{track}' to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate discriminability plots from CSVs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing discriminability.csv and correlation_matrices.csv",
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

    # Check for input files
    discrim_path = args.input_dir / "discriminability.csv"
    corr_path = args.input_dir / "correlation_matrices.csv"

    # Load correlation matrices first to get n_models
    n_models = 6  # Default
    corr_df = None
    if corr_path.exists():
        print(f"Loading {corr_path}...")
        corr_df = pd.read_csv(corr_path)
        n_models = corr_df["model_i"].nunique()
        print(f"  Found {n_models} models")

    # Plot discriminability curves and AUC
    if discrim_path.exists():
        print(f"Loading {discrim_path}...")
        discrim_df = pd.read_csv(discrim_path)

        plot_discriminability_curves(
            discrim_df,
            output_dir / f"discriminability_curves.{args.format}",
            n_models=n_models,
        )

        plot_auc_comparison(
            discrim_df,
            output_dir / f"auc_comparison.{args.format}",
        )
    else:
        print(f"Warning: {discrim_path} not found, skipping discriminability plots")

    # Plot correlation matrices
    if corr_df is not None:
        plot_correlation_matrices(corr_df, output_dir)
    elif corr_path.exists():
        print(f"Loading {corr_path}...")
        corr_df = pd.read_csv(corr_path)
        plot_correlation_matrices(corr_df, output_dir)
    else:
        print(f"Warning: {corr_path} not found, skipping correlation plots")

    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
