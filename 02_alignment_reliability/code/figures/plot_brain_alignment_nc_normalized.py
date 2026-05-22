#!/usr/bin/env python3
"""
Plot NC-normalized brain alignment: per-subject dot + connecting-line visualization.

Same layout as plot_brain_alignment.py but using noise-ceiling-normalized scores
(score / sqrt(NC) per subject). The NC reference is shown as a dashed line at 1.0.

Inputs:
    data/nc_normalized_scores.csv

Outputs:
    figures/brain_alignment_nc_normalized.pdf/png

Usage:
    python plot_brain_alignment_nc_normalized.py
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root
sys.path.insert(0, str(_PAPER / "figures"))  # for shared figure style
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers"))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers" / "figures"))
import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

MODEL_SETS = config.MODEL_SETS
MODEL_DISPLAY_NAMES = config.MODEL_DISPLAY_NAMES
SUBJECTS = config.SUBJECTS
DATA_DIR = Path(__file__).resolve().parents[2] / "results"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
PNG_DIR = FIGURES_DIR / "png"

from style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

COLOR_CSTIM = "#D64541"
COLOR_CSTIM_CRSA = "#B03A38"
COLOR_BASE = "#2980B9"
COLOR_BASE_CRSA = "#1F6699"

TITLE_MAP = {
    "training_objective": "Training Objective",
    "sota": "State of the Art",
    "architecture": "Architecture",
    "dataset": "Dataset",
    "all_models": "All Models",
}

SHORT_DISPLAY_NAMES = {
    "torchvision_vgg16_imagenet1k_v1": "VGG-16",
    "torchvision_resnet50_imagenet1k_v1": "ResNet-50",
    "torchvision_convnext_base_imagenet1k_v1": "ConvNeXt-B",
    "torchvision_vit_l_16_imagenet1k_v1": "ViT-L/16",
    "cornet_s": "CORnet-S",
    "vissl_resnet50_supervised": "Supervised",
    "vissl_resnet50_barlowtwins": "BarlowTwins",
    "vissl_resnet50_mocov2": "MoCoV2",
    "vicreg_resnet50": "VICReg",
    "robustness_imagenet_l2_eps3": "Robust-L2",
    "slip_vit_l_slip": "SLIP",
    "slip_vit_l_simclr": "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b": "CLIP-L2B",
    "dinov2_vitl14": "DINOv2",
    "openclip_vit_so400m_14_siglip_webli": "SigLIP",
    "openclip_vit_l_14_quickgelu_metaclip_400m": "MC-400M",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc": "MC-Full",
    "timm_vit_large_patch14_clip_224_dfn2b": "DFN-2B",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "CLIP-OAI",
    "openclip_vit_l_14_laion400m_e31": "CLIP-L400",
}

PANEL_LABELS = list("abcde")
VERSA_EXCLUDED = set()


# ===========================================================================
# Data loading
# ===========================================================================

def load_all_scores() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "nc_normalized_scores.csv")
    # Map method names to match plot_brain_alignment conventions
    method_map = {"wrsa_transfer": "mRSA", "crsa": "fRSA"}
    df["method"] = df["method"].map(method_map)
    return df


# ===========================================================================
# Drawing
# ===========================================================================

def draw_box(ax, x, mean, sem, width=0.30, filled=True, color="#D64541"):
    """Colin-style box: midline +/- SEM."""
    lo = mean - sem
    hi = mean + sem
    h = hi - lo
    if h < 1e-4:
        h = 0.003
        lo = mean - h / 2

    ec = color
    fc = color if filled else "white"
    lw = 0.6 if filled else 0.8

    rect = mpatches.FancyBboxPatch(
        (x - width / 2, lo), width, h,
        boxstyle="square,pad=0",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        alpha=0.75 if filled else 0.85,
        zorder=3 if filled else 4,
    )
    ax.add_patch(rect)

    ml_color = "white" if filled else ec
    ax.hlines(mean, x - width / 2, x + width / 2,
              colors=ml_color, linewidth=0.5, zorder=5)


# ===========================================================================
# Data extraction — per-subject scores
# ===========================================================================

def get_per_subject_scores(df, model_set):
    """
    Returns:
        data: {model: {method: {stim_type: {subject: score}}}}
        subjects: sorted list of subjects
    """
    models = MODEL_SETS[model_set]
    subset = df[df["model_set"] == model_set]
    subjects = sorted(subset["subject"].unique())

    data = {}
    for model in models:
        data[model] = {}
        for method in ["mRSA", "fRSA"]:
            if method == "mRSA" and model in VERSA_EXCLUDED:
                continue
            data[model][method] = {}
            m_sub = subset[(subset["model"] == model) & (subset["method"] == method)]

            for stim_type in ["controversial", "vicco"]:
                scores = {}
                for subj in subjects:
                    v = m_sub[(m_sub["subject"] == subj) &
                              (m_sub["stimulus_type"] == stim_type)]["nc_normalized"]
                    if len(v) > 0:
                        scores[subj] = v.mean()
                if scores:
                    data[model][method][stim_type] = scores

    data = {m: d for m, d in data.items() if d}
    return data, subjects


# ===========================================================================
# Panel plotting
# ===========================================================================

def plot_panel(ax, df, model_set,
               panel_label=None, title="",
               show_ylabel=True, show_legend=False,
               use_short_names=False):
    """
    Per-subject dots with connecting lines (NC-normalized scores).
    Left = controversial (red), right = baseline (blue).
    Filled dots + solid lines = mRSA, open dots + dashed lines = fRSA.
    NC reference shown as dashed line at 1.0.
    """
    data, subjects = get_per_subject_scores(df, model_set)
    if not data:
        ax.set_visible(False)
        return

    # Sort by mRSA controversial mean descending
    def sort_key(m):
        if "mRSA" in data[m] and "controversial" in data[m]["mRSA"]:
            return np.mean(list(data[m]["mRSA"]["controversial"].values()))
        return -999
    order = sorted(data.keys(), key=sort_key, reverse=True)
    n = len(order)

    # --- NC reference line at 1.0 (normalized ceiling) ---
    ax.axhline(1.0, color="#666666", linewidth=0.7, linestyle="--", alpha=0.5, zorder=0,
               label="Noise ceiling")

    # --- Plot per model ---
    x = np.arange(n)
    offset = 0.20
    box_w = 0.30
    dot_size = 8

    for i, m in enumerate(order):
        md = data[m]

        for method in ["mRSA", "fRSA"]:
            if method not in md:
                continue

            is_versa = method == "mRSA"
            cstim_scores = md[method].get("controversial", {})
            base_scores = md[method].get("vicco", {})

            c_color = COLOR_CSTIM if is_versa else COLOR_CSTIM_CRSA
            b_color = COLOR_BASE if is_versa else COLOR_BASE_CRSA

            # --- Boxes (mean +/- SEM) ---
            if cstim_scores:
                c_vals = np.array(list(cstim_scores.values()))
                c_mean = c_vals.mean()
                c_sem = c_vals.std(ddof=1) / np.sqrt(len(c_vals)) if len(c_vals) > 1 else 0
                draw_box(ax, x[i] - offset, c_mean, c_sem, width=box_w,
                         filled=is_versa, color=c_color)

            if base_scores:
                b_vals = np.array(list(base_scores.values()))
                b_mean = b_vals.mean()
                b_sem = b_vals.std(ddof=1) / np.sqrt(len(b_vals)) if len(b_vals) > 1 else 0
                draw_box(ax, x[i] + offset, b_mean, b_sem, width=box_w,
                         filled=is_versa, color=b_color)

            # --- Per-subject dots ---
            for subj in subjects:
                c_val = cstim_scores.get(subj)
                b_val = base_scores.get(subj)

                if c_val is not None and b_val is not None:
                    ls = "-" if is_versa else "--"
                    ax.plot([x[i] - offset, x[i] + offset], [c_val, b_val],
                            color="#999999", linewidth=0.3, linestyle=ls,
                            alpha=0.4, zorder=2)

                if c_val is not None:
                    if is_versa:
                        ax.scatter(x[i] - offset, c_val, s=dot_size,
                                   facecolors=c_color, edgecolors="white",
                                   linewidths=0.4, zorder=6, marker="o", alpha=0.85)
                    else:
                        ax.scatter(x[i] - offset, c_val, s=dot_size * 1.2,
                                   facecolors="none", edgecolors=c_color,
                                   linewidths=0.6, zorder=7, marker="D")
                if b_val is not None:
                    if is_versa:
                        ax.scatter(x[i] + offset, b_val, s=dot_size,
                                   facecolors=b_color, edgecolors="white",
                                   linewidths=0.4, zorder=6, marker="o", alpha=0.85)
                    else:
                        ax.scatter(x[i] + offset, b_val, s=dot_size * 1.2,
                                   facecolors="none", edgecolors=b_color,
                                   linewidths=0.6, zorder=7, marker="D")

    # --- X-axis labels ---
    ax.set_xticks(x)
    if use_short_names:
        labels = [SHORT_DISPLAY_NAMES.get(m, MODEL_DISPLAY_NAMES.get(m, m)) for m in order]
    else:
        labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in order]
    ax.set_xticklabels(labels, rotation=45, ha="right")

    if show_ylabel:
        ax.set_ylabel("Score (frac. NC)")
    if title:
        ax.set_title(title, fontweight="bold", pad=4)
    if panel_label:
        ax.text(-0.06, 1.12, panel_label, transform=ax.transAxes,
                fontsize=FONT["panel_label"], fontweight="bold", va="top")

    # Y-limits
    all_vals = [1.0]
    for m in order:
        md = data[m]
        for method in ["mRSA", "fRSA"]:
            if method not in md:
                continue
            for st in ["controversial", "vicco"]:
                if st in md[method]:
                    all_vals.extend(md[method][st].values())
    y_max = max(all_vals) * 1.08
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.6, n - 0.4)

    # --- Legend ---
    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_CSTIM,
                   markeredgecolor=COLOR_CSTIM, markersize=4, label="Controversial"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_BASE,
                   markeredgecolor=COLOR_BASE, markersize=4, label="Baseline"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#888888",
                   markeredgecolor="#888888", markersize=4, label="mRSA"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
                   markeredgecolor="#888888", markersize=4, markeredgewidth=0.8,
                   label="fRSA"),
            Line2D([0], [0], color="#666666", linewidth=0.7, linestyle="--",
                   alpha=0.5, label="Noise ceiling"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True,
                  framealpha=0.95, edgecolor="none", ncol=3,
                  columnspacing=0.5, handletextpad=0.3, handlelength=1.2)


# ===========================================================================
# Figure assembly
# ===========================================================================

def plot_figure(df):
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(W_DOUBLE, 7.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.12,
                             left=0.05, right=0.97, top=0.90, bottom=0.18)

    # Left: all_models
    ax_all = fig.add_subplot(outer[0])
    plot_panel(ax_all, df, "all_models",
               panel_label="a", title="All Models",
               show_ylabel=True, show_legend=True,
               use_short_names=True)

    # Right: 2×2 grid of controlled sets
    gs_right = outer[1].subgridspec(2, 2, wspace=0.25, hspace=0.55)
    ctrl_sets = ["sota", "training_objective", "architecture", "dataset"]

    for idx, ms in enumerate(ctrl_sets):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs_right[row, col])
        plot_panel(ax, df, ms,
                   panel_label=PANEL_LABELS[idx + 1],
                   title=TITLE_MAP.get(ms, ms),
                   show_ylabel=(col == 0))

    return fig


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NC-normalized scores...")
    df = load_all_scores()
    print(f"  {df['subject'].nunique()} subjects, {len(df)} score records")

    print("Plotting NC-normalized brain alignment figure...")
    fig = plot_figure(df)
    fig.savefig(FIGURES_DIR / "brain_alignment_nc_normalized.pdf")
    fig.savefig(PNG_DIR / "brain_alignment_nc_normalized.png", dpi=DPI)
    print("  Saved: brain_alignment_nc_normalized.pdf, brain_alignment_nc_normalized.png")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
