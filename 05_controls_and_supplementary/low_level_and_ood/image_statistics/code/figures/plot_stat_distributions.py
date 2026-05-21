#!/usr/bin/env python3
"""
Plot distributions of low-level image statistics across stimulus sets.

For each stat, one subplot: boxplot (+ strip) per stimulus set along the
x-axis. The `vicco` baseline is colored separately and its median drawn as
a dashed reference line across the panel so deviations of the controversial
sets are immediately visible.

Reads: ../data/image_stats.csv
Writes: stat_distributions.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
from style import apply_style, FONT  # noqa: E402

apply_style()

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "data" / "image_stats.csv"
OUT_PDF = HERE / "stat_distributions.pdf"
OUT_PNG = HERE / "stat_distributions.png"

STAT_ORDER = [
    ("lum_mean", "Luminance (mean)"),
    ("lum_rms", "Contrast (RMS)"),
    ("colorfulness", "Colorfulness"),
    ("lab_chroma_mean", "LAB chroma"),
    ("hue_entropy", "Hue entropy"),
    ("sf_slope", "1/f slope"),
    ("sf_high_low_ratio", "High/low SF (log10)"),
    ("edge_mag_mean", "Edge magnitude"),
    ("orient_anisotropy", "Orientation anisotropy\n(log2 H/V)"),
    ("edge_com_x", "Edge horizontal CoM\n(offset from center)"),
    ("symmetry_lr", "L–R symmetry"),
    ("entropy", "Shannon entropy"),
    ("jpeg_ratio", "JPEG ratio"),
]

# Reference distributions on the left, controversial sets on the right.
SET_ORDER = ["deepvision_train", "vicco",
             "all_models", "architecture", "dataset", "sota", "training_objective"]
SET_LABELS = {
    "deepvision_train": "train",
    "vicco": "base",
    "all_models": "all",
    "architecture": "arch",
    "dataset": "data",
    "sota": "sota",
    "training_objective": "obj",
}
REFERENCE_SET = "deepvision_train"
N_PERM = 5000
SIG_ALPHA = 0.05
TRAIN_COLOR = "#2c7fb8"      # encoder training distribution
BASELINE_COLOR = "#444444"   # vicco
CSTIM_COLOR = "#c0392b"      # controversial


def _color_for(s: str) -> str:
    if s == "deepvision_train":
        return TRAIN_COLOR
    if s == "vicco":
        return BASELINE_COLOR
    return CSTIM_COLOR


def _bh_fdr(pvals: np.ndarray, alpha: float = SIG_ALPHA) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values. NaNs pass through.

    Returns adjusted p-values on the same array shape; compare directly to alpha.
    """
    p = np.asarray(pvals, dtype=float)
    mask = np.isfinite(p)
    m = int(mask.sum())
    adj = np.full_like(p, np.nan)
    if m == 0:
        return adj
    order = np.argsort(p[mask])
    ranked = p[mask][order]
    # BH: p_adj[(k)] = min over i >= k of (m / i) * p[(i)]
    adj_sorted = ranked * m / (np.arange(m) + 1)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    adj[mask] = out
    return adj


def main():
    df = pd.read_csv(CSV)
    sets = [s for s in SET_ORDER if s in df["stimulus_set"].unique()]
    xs = np.arange(len(sets))

    ncols = 4
    nrows = int(np.ceil(len(STAT_ORDER) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 1.7))
    axes = np.atleast_2d(axes).ravel()

    rng = np.random.default_rng(0)

    # Precompute data per statistic
    stat_data: dict = {}
    for col, _label in STAT_ORDER:
        data = [df.loc[df["stimulus_set"] == s, col].dropna().values for s in sets]
        stat_data[col] = {"data": data}

    for ax, (col, label) in zip(axes, STAT_ORDER):
        data = stat_data[col]["data"]
        colors = [_color_for(s) for s in sets]

        # reference median lines (training + baseline)
        if "deepvision_train" in sets:
            t_med = np.median(data[sets.index("deepvision_train")])
            ax.axhline(t_med, color=TRAIN_COLOR, lw=0.6, ls="--", alpha=0.55, zorder=0)
        if "vicco" in sets:
            v_med = np.median(data[sets.index("vicco")])
            ax.axhline(v_med, color=BASELINE_COLOR, lw=0.6, ls="--", alpha=0.55, zorder=0)

        # strip (jittered points)
        for xi, vals, c in zip(xs, data, colors):
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(
                xi + jitter, vals, s=1.8, color=c, alpha=0.25,
                linewidths=0, zorder=1,
            )

        # boxplot on top
        bp = ax.boxplot(
            data, positions=xs, widths=0.55, showfliers=False,
            patch_artist=True, zorder=2,
            medianprops=dict(color="white", lw=1.0),
            whiskerprops=dict(lw=0.6),
            capprops=dict(lw=0.6),
            boxprops=dict(lw=0.6),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
            patch.set_edgecolor(c)

        ax.set_title(label, fontsize=FONT["title"])
        ax.set_xticks(xs)
        ax.set_xticklabels([SET_LABELS[s] for s in sets], rotation=0,
                           fontsize=FONT["tick"])
        ax.tick_params(axis="y", labelsize=FONT["tick"])

    for ax in axes[len(STAT_ORDER):]:
        ax.axis("off")

    # legend
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=TRAIN_COLOR,
                   markersize=6, label="train = LAION-fMRI shared (encoder training)"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=BASELINE_COLOR,
                   markersize=6, label="base = baseline"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=CSTIM_COLOR,
                   markersize=6, label="controversial sets (all, arch, data, sota, obj)"),
        plt.Line2D([0], [0], color=TRAIN_COLOR, lw=0.8, ls="--",
                   label="train median"),
        plt.Line2D([0], [0], color=BASELINE_COLOR, lw=0.8, ls="--",
                   label="base median"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=FONT["legend"], frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
