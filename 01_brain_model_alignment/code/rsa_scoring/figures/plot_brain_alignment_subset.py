#!/usr/bin/env python3
"""
Plot brain alignment comparison: full stimulus set vs. first-K subset.

Shows that the first ~20 greedy-selected stimuli produce similar brain
alignment patterns to the full 100-stimulus set.

Inputs:
    03_alignment_inference/results/subset_scores_K20.csv       (subset scores)
    01_brain_model_alignment/results/rsa_scores/{subject}/crsa_scores.csv   (full scores, fRSA)
    01_brain_model_alignment/results/rsa_scores/{subject}/wrsa_transfer_scores.csv  (full scores, mRSA)

Outputs:
    figures/brain_alignment_subset.pdf/png

Usage:
    python plot_brain_alignment_subset.py [--max-stim 20]
"""

import argparse
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[3]
SHARE_ROOT = STAGE.parent
sys.path.insert(0, str(SHARE_ROOT / "src"))
from cstims.paper import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

MODEL_SETS = config.MODEL_SETS
MODEL_DISPLAY_NAMES = config.MODEL_DISPLAY_NAMES
SUBJECTS = config.SUBJECTS
STATS_DATA_DIR = config.STATS_DATA_DIR
RSA_DATA_DIR = config.RSA_DATA_DIR
FIGURES_DIR = STAGE / "figures" / "rsa_scores"

from cstims.paper.style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

# Full set: red/blue. Subset: darker shades with hatching
COLOR_FULL_VERSA = "#D64541"
COLOR_SUBSET_VERSA = "#922B2B"
COLOR_FULL_CRSA = "#2980B9"
COLOR_SUBSET_CRSA = "#1A5276"
COLOR_BASELINE_FULL = "#95A5A6"
COLOR_BASELINE_SUBSET = "#616A6B"

TITLE_MAP = {
    "training_objective": "Training Objective",
    "sota": "State of the Art",
    "architecture": "Architecture",
    "dataset": "Dataset",
    "all_models": "All Models",
}

PANEL_LABELS = ["a", "b", "c", "d", "e"]


def load_full_scores() -> pd.DataFrame:
    """Load original full-set scores from per-subject CSVs."""
    dfs = []
    for subject in SUBJECTS:
        data_dir = RSA_DATA_DIR / subject
        for filename, (col, method) in {
            "wrsa_transfer_scores.csv": ("wrsa_transfer", "wrsa_transfer"),
            "crsa_scores.csv": ("crsa", "crsa"),
        }.items():
            path = data_dir / filename
            if path.exists():
                df = pd.read_csv(path)
                df = df.rename(columns={col: "score"})
                df["method"] = method
                if "subject" not in df.columns:
                    df["subject"] = subject
                dfs.append(df[["subject", "model_set", "model", "display_name",
                               "stimulus_type", "bootstrap_idx", "method", "score"]])
    return pd.concat(dfs, ignore_index=True)


def load_subset_scores(max_stim: int) -> pd.DataFrame:
    """Load subset scores and reshape to long format."""
    path = STATS_DATA_DIR / f"subset_scores_K{max_stim}.csv"
    df = pd.read_csv(path)

    # Reshape: separate crsa and wrsa_transfer into method column
    records = []
    for _, row in df.iterrows():
        base = {
            "subject": row["subject"],
            "model_set": row["model_set"],
            "model": row["model"],
            "display_name": row["display_name"],
            "stimulus_type": row["stimulus_type"],
            "bootstrap_idx": row["bootstrap_idx"],
        }
        records.append({**base, "method": "crsa", "score": row["crsa"]})
        if not np.isnan(row["wrsa_transfer"]):
            records.append({**base, "method": "wrsa_transfer", "score": row["wrsa_transfer"]})
    return pd.DataFrame(records)


def prepare_paired_data(df: pd.DataFrame, model_set: str, method: str):
    """Cross-subject averaged paired data (controversial only)."""
    models = MODEL_SETS[model_set]
    subset = df[(df["model_set"] == model_set) & (df["method"] == method)]
    if len(subset) == 0:
        return None

    cstim = subset[subset["stimulus_type"] == "controversial"]
    per_subj = cstim.groupby(["subject", "model"])["score"].mean().reset_index()
    mean = per_subj.groupby("model")["score"].mean()
    sem = per_subj.groupby("model")["score"].sem()

    common = [m for m in models if m in mean.index]
    if not common:
        return None

    return {
        "models": common,
        "display_names": [MODEL_DISPLAY_NAMES.get(m, m) for m in common],
        "scores": [mean[m] for m in common],
        "sem": [sem.get(m, 0) for m in common],
    }


def plot_comparison_figure(df_full: pd.DataFrame, df_subset: pd.DataFrame, max_stim: int):
    """
    Scatter comparison: full-set score (x) vs subset score (y) for all models.
    One panel per method (mRSA, fRSA), colored by model set.
    """
    ms_colors = {
        "sota": "#E74C3C",
        "architecture": "#3498DB",
        "training_objective": "#2ECC71",
        "dataset": "#F39C12",
        "all_models": "#9B59B6",
    }

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.60, 3.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.14, wspace=0.22)

    for ax, (method, method_label) in zip(axes, [
        ("wrsa_transfer", "mixed RSA"),
        ("crsa", "fixed RSA"),
    ]):
        all_full = []
        all_sub = []

        for model_set in ["sota", "training_objective", "architecture", "dataset", "all_models"]:
            full_data = prepare_paired_data(df_full, model_set, method)
            sub_data = prepare_paired_data(df_subset, model_set, method)

            if full_data is None or sub_data is None:
                continue

            # Align models
            common = [m for m in full_data["models"] if m in sub_data["models"]]
            full_scores = [full_data["scores"][full_data["models"].index(m)] for m in common]
            sub_scores = [sub_data["scores"][sub_data["models"].index(m)] for m in common]
            full_sem = [full_data["sem"][full_data["models"].index(m)] for m in common]
            sub_sem = [sub_data["sem"][sub_data["models"].index(m)] for m in common]

            ax.errorbar(
                full_scores, sub_scores,
                xerr=full_sem, yerr=sub_sem,
                fmt="o", color=ms_colors[model_set],
                markersize=4, alpha=0.85, capsize=1.5,
                elinewidth=0.6, linewidth=0,
                label=TITLE_MAP.get(model_set, model_set),
            )

            all_full.extend(full_scores)
            all_sub.extend(sub_scores)

        # Identity line
        if all_full:
            lo = min(min(all_full), min(all_sub))
            hi = max(max(all_full), max(all_sub))
            margin = (hi - lo) * 0.1
            lims = [lo - margin, hi + margin]
            ax.plot(lims, lims, color="#444444", linestyle="--",
                    linewidth=0.6, alpha=0.5, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            # Compute correlation
            from scipy import stats
            r, p = stats.pearsonr(all_full, all_sub)
            ax.text(0.04, 0.96, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=FONT["annotation"], va="top",
                    color="#333333")

        ax.set_xlabel(f"{method_label} (N=100)", fontsize=FONT["axis_label"])
        ax.set_ylabel(f"{method_label} (N={max_stim})", fontsize=FONT["axis_label"])
        ax.set_title(method_label, fontweight="bold",
                     pad=4, fontsize=FONT["title"])
        ax.tick_params(labelsize=FONT["tick"])
        ax.set_aspect("equal")

    axes[0].legend(frameon=True, framealpha=0.9, edgecolor="none",
                   fontsize=FONT["legend"], loc="lower right",
                   handletextpad=0.4, handlelength=1.0)

    return fig


def plot_overlay_bars(df_full: pd.DataFrame, df_subset: pd.DataFrame, max_stim: int):
    """
    Bar chart overlay: full set (solid) vs subset (hatched) for all-models panel.
    Two-row layout like brain_alignment.py: mRSA top, fRSA bottom.
    """
    fig = plt.figure(figsize=(7.2, 5.5))

    h = 0.18
    gap = 0.14
    top = 0.93

    method_configs = [
        ("wrsa_transfer", "mRSA", COLOR_FULL_VERSA, COLOR_SUBSET_VERSA, top - h),
        ("crsa", "fRSA", COLOR_FULL_CRSA, COLOR_SUBSET_CRSA, top - h - gap - h),
    ]

    for method, method_label, c_full, c_sub, bot in method_configs:
        full_data = prepare_paired_data(df_full, "all_models", method)
        sub_data = prepare_paired_data(df_subset, "all_models", method)
        if full_data is None or sub_data is None:
            continue

        # Sort by full-set controversial score
        order = sorted(range(len(full_data["models"])),
                       key=lambda i: full_data["scores"][i], reverse=True)

        models = [full_data["models"][i] for i in order]
        names = [full_data["display_names"][i] for i in order]
        full_scores = [full_data["scores"][i] for i in order]
        full_sem = [full_data["sem"][i] for i in order]

        # Align subset to same order
        sub_map = dict(zip(sub_data["models"], zip(sub_data["scores"], sub_data["sem"])))
        sub_scores = [sub_map[m][0] if m in sub_map else 0 for m in models]
        sub_sem = [sub_map[m][1] if m in sub_map else 0 for m in models]

        ax = fig.add_axes([0.08, bot, 0.88, h])
        n = len(models)
        x = np.arange(n)
        width = 0.38

        ax.bar(x - width / 2, full_scores, width,
               color=c_full, alpha=0.9, label=f"N=100",
               yerr=full_sem, capsize=2,
               error_kw=dict(linewidth=0.8, color="black"))
        ax.bar(x + width / 2, sub_scores, width,
               color=c_sub, alpha=0.7, label=f"N={max_stim}",
               yerr=sub_sem, capsize=2,
               error_kw=dict(linewidth=0.8, color="black"),
               hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(f"{method_label} ($r$)")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="upper right")

        y_max = max(max(f + e for f, e in zip(full_scores, full_sem)),
                    max(s + e for s, e in zip(sub_scores, sub_sem)))
        ax.set_ylim(0, y_max * 1.15)

    fig.suptitle(f"All models: N=100 vs. first N={max_stim} stimuli",
                 fontweight="bold", y=0.98)
    return fig


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stim", type=int, default=20)
    args = parser.parse_args()
    max_stim = args.max_stim

    print("Loading full-set scores...")
    df_full = load_full_scores()
    print(f"  {len(df_full)} records")

    print(f"Loading subset scores (K={max_stim})...")
    df_subset = load_subset_scores(max_stim)
    print(f"  {len(df_subset)} records")

    # Scatter comparison
    print("\nPlotting scatter comparison...")
    fig = plot_comparison_figure(df_full, df_subset, max_stim)
    for fmt in ["pdf", "png"]:
        out = FIGURES_DIR / f"brain_alignment_subset_scatter.{fmt}"
        fig.savefig(out)
        print(f"  Saved {out}")
    plt.close(fig)

    # Overlay bars for all_models
    print("\nPlotting overlay bars (all_models)...")
    fig = plot_overlay_bars(df_full, df_subset, max_stim)
    for fmt in ["pdf", "png"]:
        out = FIGURES_DIR / f"brain_alignment_subset_bars.{fmt}"
        fig.savefig(out)
        print(f"  Saved {out}")
    plt.close(fig)

    print("\nDone!")


if __name__ == "__main__":
    main()
