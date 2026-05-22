#!/usr/bin/env python3
"""
Visualise the within-baseline subsampling control (06_ood/02).

Three figures:

1. baseline_subsampling_match_quality.{pdf,png}
   Per-cell distribution of OOD-match quality (gap in SD units), by model_set.
   Shows where the baseline pool can and cannot reach cstim OOD.

2. baseline_subsampling_matched_only.{pdf,png}
   For cells where match_quality == 'matched' (gap < 0.5 SD), bar comparison
   of cstim wRSA vs matched-K vs all-boot, per model_set. Cells with no
   matched data are flagged.

3. baseline_subsampling_scatter.{pdf,png}
   Per (model_set × model) scatter of bootstraps in (mean_OOD, wRSA) space
   with the cstim point, the matched-K subset, and the all-boot mean.
   Cell-by-cell match quality is annotated.

The scatter plot's purpose: show whether the cstim point lies on the natural
(OOD, wRSA) trend defined by the bootstraps, or off it.
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers"))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers" / "figures"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config

DATA_DIR = _PAPER / "results"
PER     = DATA_DIR / "baseline_subsampling.csv"
SUMMARY = DATA_DIR / "baseline_subsampling_summary.csv"
FIGS    = Path(__file__).resolve().parents[2] / "figures" / "supplementary"
PNG_DIR = FIGS / "png"

GROUPS = ["all_models", "architecture", "training_objective", "sota", "dataset"]
GROUP_LABELS = {
    "all_models": "All models",
    "architecture": "Architecture",
    "training_objective": "Training obj.",
    "sota": "SOTA",
    "dataset": "Dataset",
}
QUALITY_COLOURS = {
    "matched":      "#2ca02c",
    "weak":         "#ff7f0e",
    "out_of_range": "#d62728",
}


# --------------------------------------------------------------------------------
# Figure 1 — match-quality landscape
# --------------------------------------------------------------------------------

def fig_match_quality(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0), constrained_layout=True)

    # (a) gap_sd distribution per model_set, coloured by class
    ax = axes[0]
    positions = np.arange(len(GROUPS))
    for k, model_set in enumerate(GROUPS):
        sub = summary[summary["model_set"] == model_set]
        if sub.empty:
            continue
        rng = np.random.default_rng(0)
        for q, c in QUALITY_COLOURS.items():
            cell = sub[sub["match_quality"] == q]
            if cell.empty:
                continue
            xj = k + rng.uniform(-0.18, 0.18, size=len(cell))
            ax.scatter(xj, cell["gap_sd"].values, color=c, alpha=0.7, s=22,
                       edgecolor="none", label=q if k == 0 else None)
    ax.axhline(0.5, color="black", lw=0.6, ls="--")
    ax.axhline(1.5, color="black", lw=0.6, ls="--")
    ax.text(len(GROUPS) - 0.4, 0.6, "0.5 SD", fontsize=7, color="black")
    ax.text(len(GROUPS) - 0.4, 1.6, "1.5 SD", fontsize=7, color="black")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=8, rotation=10, ha="right")
    ax.set_ylabel("gap in SD units\n|cstim_OOD − matched_K_mean| / SD(bootstrap_OOD)", fontsize=8)
    ax.set_title("OOD-match quality per (subject × model) cell", fontsize=10)
    ax.tick_params(labelsize=8)
    handles = [mpatches.Patch(color=c, label=q) for q, c in QUALITY_COLOURS.items()]
    ax.legend(handles=handles, fontsize=7, loc="upper left")

    # (b) bar plot of class counts per model_set
    ax = axes[1]
    qs = list(QUALITY_COLOURS.keys())
    counts = (summary.groupby(["model_set", "match_quality"]).size().unstack(fill_value=0)
              .reindex(index=GROUPS, columns=qs, fill_value=0))
    bottom = np.zeros(len(GROUPS))
    for q in qs:
        ax.bar(positions, counts[q].values, bottom=bottom,
               color=QUALITY_COLOURS[q], label=q, edgecolor="white", lw=0.4)
        bottom += counts[q].values
    ax.set_xticks(positions)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=8, rotation=10, ha="right")
    ax.set_ylabel("# (subject × model) cells", fontsize=8)
    ax.set_title("Match-quality cell counts (out of 25–100 per set)", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Within-baseline OOD matching: when is the comparison meaningful?",
                 fontsize=11)
    return fig


# --------------------------------------------------------------------------------
# Figure 2 — matched-only bar comparison
# --------------------------------------------------------------------------------

def fig_matched_only(summary):
    matched = summary[summary["match_quality"] == "matched"]
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(len(GROUPS) * 3.5, 4.4),
                             constrained_layout=True, sharey=True)

    for ax, model_set in zip(axes, GROUPS):
        cell = matched[matched["model_set"] == model_set]
        all_for_set = summary[summary["model_set"] == model_set]
        n_total = len(all_for_set)
        if cell.empty:
            ax.text(0.5, 0.5,
                    f"No matched cells\n(0 / {n_total} cells\nhave gap < 0.5 SD)\n\n"
                    "Baseline pool cannot\nreach this set's OOD level.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9,
                    color="#d62728")
            ax.set_title(GROUP_LABELS[model_set], fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.tick_params(left=False, labelleft=False)
            continue

        # Aggregate per-model means within matched cells (subject-mean)
        per_model = (cell.groupby("model")
                     [["cstim_wrsa", "matched_wrsa", "all_boot_wrsa"]]
                     .mean().reset_index())
        x = np.arange(len(per_model))
        w = 0.27
        ax.bar(x - w, per_model["all_boot_wrsa"], w, color="#9bc4dc",
               label="all 1000 boots")
        ax.bar(x,     per_model["matched_wrsa"],  w, color="#2166ac",
               label=f"matched K=50")
        ax.bar(x + w, per_model["cstim_wrsa"],    w, color="#d6604d",
               label="cstim")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [config.MODEL_DISPLAY_NAMES.get(m, m) for m in per_model["model"]],
            rotation=60, ha="right", fontsize=6,
        )
        ax.set_title(f"{GROUP_LABELS[model_set]}\n({len(cell)} / {n_total} cells matched)",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("wRSA", fontsize=8) if model_set == GROUPS[0] else None
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Matched-only comparison — only cells where cstim OOD is within 0.5 SD of "
        "the closest 50 bootstrap mean OODs", fontsize=10,
    )
    return fig


# --------------------------------------------------------------------------------
# Figure 3 — scatter overlays per (model_set × model)
# --------------------------------------------------------------------------------

def fig_scatter(per, summary, model_set):
    """Per model in a model_set: scatter of bootstraps in (OOD, wRSA) space,
    overlaying cstim, matched-K, and all-boot mean. Coloured by match quality."""
    models = config.MODEL_SETS[model_set]
    n = len(models)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.6),
                             constrained_layout=True)
    axes = np.atleast_2d(axes).flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        # Subject-mean across subjects per bootstrap
        sub = per[(per["stim_type"] == "vicco") & (per["model_set"] == model_set)
                  & (per["model"] == model)]
        if sub.empty:
            ax.set_visible(False)
            continue
        boot_avg = (sub.groupby("bootstrap_idx")[["mean_loglik_pred_z", "wrsa"]]
                    .mean().reset_index())

        # Subject-mean cstim
        cstim = (per[(per["stim_type"] == "controversial")
                     & (per["model_set"] == model_set)
                     & (per["model"] == model)]
                 [["mean_loglik_pred_z", "wrsa"]].mean())

        # Match quality from summary (subject-mean of per-subject quality
        # is hard to reduce; use the dominant class)
        s = summary[(summary["model_set"] == model_set) & (summary["model"] == model)]
        dom = s["match_quality"].mode().iloc[0] if not s.empty else "unknown"
        qcolor = QUALITY_COLOURS.get(dom, "gray")

        ax.scatter(boot_avg["mean_loglik_pred_z"], boot_avg["wrsa"],
                   s=4, color="#2166ac", alpha=0.18)

        # Highlight K=50 closest bootstraps
        order = np.argsort(np.abs(boot_avg["mean_loglik_pred_z"].values
                                   - cstim["mean_loglik_pred_z"]))[:50]
        ax.scatter(boot_avg["mean_loglik_pred_z"].iloc[order],
                   boot_avg["wrsa"].iloc[order],
                   s=10, color="#08306b", alpha=0.85,
                   edgecolor="none")
        ax.scatter(cstim["mean_loglik_pred_z"], cstim["wrsa"],
                   s=85, color="#d6604d", marker="*", edgecolor="black", lw=0.6, zorder=5)
        ax.axhline(boot_avg["wrsa"].mean(), color="#2166ac", lw=0.6, ls="--", alpha=0.6)
        ax.axvline(cstim["mean_loglik_pred_z"], color="#d6604d", lw=0.6, ls="--", alpha=0.6)

        ax.set_title(f"{config.MODEL_DISPLAY_NAMES.get(model, model)} ({dom})",
                     fontsize=8, color=qcolor, fontweight="bold")
        ax.set_xlabel("mean loglik z (pred)", fontsize=7)
        ax.set_ylabel("wRSA", fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Vicco bootstraps vs controversial — model_set = {GROUP_LABELS[model_set]}\n"
        f"big blue = K=50 closest-OOD boots; red ★ = cstim; "
        f"colour-coded title = dominant match quality",
        fontsize=10,
    )
    return fig


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGS / f"{name}.pdf"
    png_path = PNG_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {pdf_path}")
    print(f"Saved → {png_path}")
    plt.close(fig)


def main():
    print("Loading data...")
    per     = pd.read_csv(PER)
    summary = pd.read_csv(SUMMARY)

    save(fig_match_quality(summary), "baseline_subsampling_match_quality")
    save(fig_matched_only(summary),  "baseline_subsampling_matched_only")
    save(fig_scatter(per, summary, "all_models"),  "baseline_subsampling_scatter_all_models")

    print("\nNumeric summary by model_set (matched cells only):")
    matched = summary[summary["match_quality"] == "matched"]
    if matched.empty:
        print("  No matched cells in any set.")
    else:
        agg = (matched.groupby("model_set")
               [["cstim_wrsa", "matched_wrsa", "drop_vs_matched", "gap_sd"]]
               .agg(["mean", "std", "count"]).round(3))
        print(agg.to_string())


if __name__ == "__main__":
    main()
