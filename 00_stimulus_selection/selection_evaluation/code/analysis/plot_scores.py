#!/usr/bin/env python3
"""Generate score trajectory and refinement plots from computed CSV results.

Reads:
- greedy_scores.csv
- refinement.csv
- summary.csv

Outputs:
- plots/score_trajectory_raw.pdf: Raw score vs iteration
- plots/refinement_analysis.pdf: Refinement pass analysis
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


def plot_refinement_analysis(
    refinement_df: pd.DataFrame,
    output_path: Path,
    greedy_df: pd.DataFrame | None = None,
):
    """Plot refinement analysis.

    Args:
        refinement_df: DataFrame with columns: pass_num, position, old_idx, new_idx, score, replaced
        output_path: Path to save PDF
        greedy_df: Optional DataFrame with greedy scores for before/after comparison
    """
    if refinement_df.empty:
        print("No refinement data, skipping refinement plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Score improvement from refinement (deltas)
    ax1 = axes[0]

    # Get score columns (tracks) - exclude combined scores
    score_cols = [c for c in refinement_df.columns if c.startswith("score_") and c not in ("score_combined_raw", "score_combined", "score")]

    # Find the combined score column in each dataframe
    greedy_score_col = None
    refine_score_col = None
    for col in ["score_combined_raw", "score_combined", "score"]:
        if greedy_df is not None and col in greedy_df.columns:
            greedy_score_col = col
            break
    for col in ["score_combined_raw", "score_combined", "score"]:
        if col in refinement_df.columns:
            refine_score_col = col
            break

    has_greedy_data = greedy_df is not None and not greedy_df.empty and greedy_score_col is not None
    has_refine_data = refine_score_col is not None

    if has_greedy_data and has_refine_data:
        # "Before" = end of greedy phase
        greedy_last = greedy_df.iloc[-1]

        # "After" = last replacement in refinement
        replaced_only = refinement_df[refinement_df["replaced"]]
        if not replaced_only.empty:
            refine_last = replaced_only.iloc[-1]
        else:
            refine_last = refinement_df.iloc[-1]

        # Build data for delta bar chart
        track_names = []
        deltas = []
        pct_improvements = []

        # Add combined first
        before = greedy_last[greedy_score_col]
        after = refine_last[refine_score_col]
        track_names.append("combined")
        deltas.append(after - before)
        pct_improvements.append((after - before) / before * 100 if before > 0 else 0)

        # Add per-track scores
        for col in sorted(score_cols):
            if col in greedy_df.columns and col in refinement_df.columns:
                track_name = col.replace("score_", "")
                before = greedy_last[col]
                after = refine_last[col]
                track_names.append(track_name)
                deltas.append(after - before)
                pct_improvements.append((after - before) / before * 100 if before > 0 else 0)

        if track_names:
            x = np.arange(len(track_names))
            colors = ["C0" if d >= 0 else "C3" for d in deltas]
            bars = ax1.bar(x, deltas, color=colors, alpha=0.8)

            # Add % improvement labels on bars
            for i, (bar, pct) in enumerate(zip(bars, pct_improvements)):
                height = bar.get_height()
                label = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
                ax1.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=8,
                    fontweight="bold",
                )

            ax1.set_xlabel("Track", fontsize=11)
            ax1.set_ylabel("Score Improvement", fontsize=11)
            ax1.set_title("Score Improvement from Refinement", fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(track_names, rotation=45, ha="right", fontsize=9)
            ax1.axhline(y=0, color="black", linewidth=0.5)
            ax1.grid(True, alpha=0.3, axis="y")
        else:
            ax1.text(0.5, 0.5, "Only combined score available\n(no per-track breakdown)", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_title("Score Improvement from Refinement", fontsize=12)
    else:
        msg = "Missing data for comparison"
        if not has_greedy_data:
            msg = "No greedy score data"
        elif not has_refine_data:
            msg = "No refinement score data"
        ax1.text(0.5, 0.5, msg, ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Score Improvement from Refinement", fontsize=12)

    # Plot 2: Replacements per pass
    ax2 = axes[1]
    replacements_per_pass = refinement_df.groupby("pass_num")["replaced"].sum()
    pass_nums = replacements_per_pass.index.astype(int)
    bars = ax2.bar(
        pass_nums,
        replacements_per_pass.values,
        color="C0",
        alpha=0.8,
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax2.set_xlabel("Pass Number", fontsize=11)
    ax2.set_ylabel("Number of Replacements", fontsize=11)
    ax2.set_title("Replacements per Refinement Pass", fontsize=12)
    ax2.set_xticks(pass_nums)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved refinement analysis to {output_path}")


def plot_score_trajectory_raw(
    greedy_df: pd.DataFrame,
    output_path: Path,
    refinement_df: pd.DataFrame | None = None,
):
    """Plot raw combined score trajectory over iterations with per-track scores.

    All scores plotted on the same axis so you can see combined = average of tracks.

    Args:
        greedy_df: DataFrame with columns: iteration, score_combined_raw OR score_combined, score_<track>, n_selected
        output_path: Path to save PDF
        refinement_df: Optional DataFrame with refinement history
    """
    # Accept either score_combined_raw or score_combined
    score_col = "score_combined_raw" if "score_combined_raw" in greedy_df.columns else "score_combined"
    if score_col not in greedy_df.columns:
        print("No score_combined_raw or score_combined column, skipping raw score trajectory plot")
        return

    # Identify per-track score columns (exclude combined scores)
    greedy_track_cols = [
        c for c in greedy_df.columns
        if c.startswith("score_") and c not in ("score_combined", "score_combined_raw")
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Build color map for tracks
    track_color_map = {}
    if greedy_track_cols:
        colors = plt.cm.tab10(np.linspace(0, 1, len(greedy_track_cols)))
        for idx, col in enumerate(sorted(greedy_track_cols)):
            track_color_map[col] = colors[idx]

    # Plot per-track scores first (so combined is on top)
    for col in sorted(greedy_track_cols):
        track_name = col.replace("score_", "")
        ax.plot(
            greedy_df["iteration"],
            greedy_df[col],
            "--",
            color=track_color_map[col],
            linewidth=1,
            alpha=0.6,
            label=track_name,
        )

    # Plot combined score (average of tracks) - thicker black line
    ax.plot(
        greedy_df["iteration"],
        greedy_df[score_col],
        "o-",
        color="black",
        linewidth=2.5,
        markersize=4,
        label="combined (avg)",
    )

    # Add refinement phase if available
    last_greedy_iter = greedy_df["iteration"].max() if not greedy_df.empty else 0
    has_refinement = False

    # Check for score column in refinement_df (accept score_combined_raw, score_combined, or score)
    refine_score_col = None
    if refinement_df is not None and not refinement_df.empty:
        for col_name in ["score_combined_raw", "score_combined", "score"]:
            if col_name in refinement_df.columns:
                refine_score_col = col_name
                break

    if refine_score_col is not None:
        replaced = refinement_df[refinement_df["replaced"]].copy()
        if not replaced.empty:
            has_refinement = True
            replaced = replaced.sort_values(["pass_num", "position"]).reset_index(drop=True)
            replaced["iteration"] = range(
                last_greedy_iter + 1, last_greedy_iter + 1 + len(replaced)
            )

            # Vertical line at greedy/refinement boundary
            ax.axvline(
                x=last_greedy_iter + 0.5,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

            # Plot refinement per-track scores (same colors, no legend duplicates)
            for col in sorted(greedy_track_cols):
                if col in replaced.columns:
                    ax.plot(
                        replaced["iteration"],
                        replaced[col],
                        "--",
                        color=track_color_map[col],
                        linewidth=1,
                        alpha=0.6,
                    )

            # Plot refinement combined scores
            ax.plot(
                replaced["iteration"],
                replaced[refine_score_col],
                "o-",
                color="black",
                linewidth=2.5,
                markersize=4,
            )

    # Add "Refinement" label after all plotting (so y-limits are set)
    if has_refinement:
        ylim = ax.get_ylim()
        ax.text(
            last_greedy_iter + 0.7,
            ylim[0] + 0.02 * (ylim[1] - ylim[0]),
            "Refinement",
            fontsize=9,
            color="gray",
            rotation=90,
            va="bottom",
        )

    # Add vertical line at best raw combined score
    all_scores = list(zip(greedy_df["iteration"].tolist(), greedy_df[score_col].tolist()))
    if has_refinement and refine_score_col is not None:
        replaced = refinement_df[refinement_df["replaced"]].copy()
        replaced = replaced.sort_values(["pass_num", "position"]).reset_index(drop=True)
        refine_iters = range(last_greedy_iter + 1, last_greedy_iter + 1 + len(replaced))
        all_scores.extend(zip(refine_iters, replaced[refine_score_col].tolist()))

    if all_scores:
        best_iter, best_score = max(all_scores, key=lambda x: x[1])
        ax.axvline(
            x=best_iter,
            color="green",
            linestyle=":",
            linewidth=2,
            alpha=0.8,
        )
        ylim = ax.get_ylim()
        ax.text(
            best_iter + 0.3,
            ylim[1] - 0.02 * (ylim[1] - ylim[0]),
            f"best ({best_score:.4f})",
            fontsize=8,
            color="green",
            va="top",
        )

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    title = "Selection Score Trajectory (Raw)"
    if has_refinement:
        n_replacements = len(refinement_df[refinement_df["replaced"]])
        title += f" (+ {n_replacements} refinement replacements)"
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved raw score trajectory to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate score trajectory and refinement plots"
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
    greedy_path = args.input_dir / "greedy_scores.csv"
    refinement_path = args.input_dir / "refinement.csv"

    # Load refinement data first (needed for combined trajectory plot)
    refinement_df = None
    if refinement_path.exists():
        print(f"Loading {refinement_path}...")
        refinement_df = pd.read_csv(refinement_path)
        if refinement_df.empty or refinement_df["replaced"].sum() == 0:
            print("  No refinement replacements found")
            refinement_df = None
        else:
            n_replacements = refinement_df["replaced"].sum()
            print(f"  Found {n_replacements} refinement replacements")

    # Load greedy scores
    greedy_df = None
    if greedy_path.exists():
        print(f"Loading {greedy_path}...")
        greedy_df = pd.read_csv(greedy_path)

        plot_score_trajectory_raw(
            greedy_df,
            output_dir / f"score_trajectory_raw.{args.format}",
            refinement_df=refinement_df,
        )
    else:
        print(f"Warning: {greedy_path} not found, skipping score trajectory plots")

    # Plot refinement analysis (detailed breakdown)
    if refinement_df is not None:
        plot_refinement_analysis(
            refinement_df,
            output_dir / f"refinement_analysis.{args.format}",
            greedy_df=greedy_df,
        )
    else:
        print("Skipping refinement analysis plot (no refinement data)")

    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
