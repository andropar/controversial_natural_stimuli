#!/usr/bin/env python3
"""
Brain alignment for controlled stimulus sets — all 20 models.

Like plot_brain_alignment.py but for each controlled set shows ALL models
(in-set and out-of-set), using:
  - controversial: cross_set_wrsa_scores.csv (all models on controlled stimuli)
  - baseline (vicco): wrsa_transfer_scores.csv, all_models set

In-set models are marked with a star on the x-axis label.
X sorted by controlled-cstim mRSA mean descending.

Outputs:
    figures/controlled_all_models.pdf/png
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_SHARE_ROOT / "src"))
from cstims.paper import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from cstims.paper.style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

STAGE_DIR = Path(__file__).resolve().parents[3]
SUBJECTS    = config.SUBJECTS
MODEL_SETS  = config.MODEL_SETS
MODEL_DISPLAY_NAMES = config.MODEL_DISPLAY_NAMES
RSA_DATA_DIR = STAGE_DIR / "results" / "rsa_scores"
FIGURES_DIR = STAGE_DIR / "figures" / "rsa_scores" / "supplementary"
PNG_DIR = FIGURES_DIR / "png"

SHORT_DISPLAY_NAMES = {
    "torchvision_vgg16_imagenet1k_v1":                 "VGG-16",
    "torchvision_resnet50_imagenet1k_v1":              "ResNet-50",
    "torchvision_convnext_base_imagenet1k_v1":         "ConvNeXt-B",
    "torchvision_vit_l_16_imagenet1k_v1":              "ViT-L/16",
    "cornet_s":                                         "CORnet-S",
    "vissl_resnet50_supervised":                        "Supervised",
    "vissl_resnet50_barlowtwins":                       "BarlowTwins",
    "vissl_resnet50_mocov2":                            "MoCoV2",
    "vicreg_resnet50":                                  "VICReg",
    "robustness_imagenet_l2_eps3":                      "Robust-L2",
    "slip_vit_l_slip":                                  "SLIP",
    "slip_vit_l_simclr":                                "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b":          "CLIP-L2B",
    "dinov2_vitl14":                                    "DINOv2",
    "openclip_vit_so400m_14_siglip_webli":              "SigLIP",
    "openclip_vit_l_14_quickgelu_metaclip_400m":        "MC-400M",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc":      "MC-Full",
    "timm_vit_large_patch14_clip_224_dfn2b":            "DFN-2B",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "CLIP-OAI",
    "openclip_vit_l_14_laion400m_e31":                  "CLIP-L400",
}

TITLE_MAP = {
    "training_objective": "Training Objective",
    "sota":               "State of the Art",
    "architecture":       "Architecture",
    "dataset":            "Dataset",
}

COLOR_CSTIM = "#D64541"
COLOR_BASE  = "#2980B9"


# ===========================================================================
# Data loading
# ===========================================================================

def load_scores():
    """Returns (cross_df, vicco_df) both with subject/model columns."""
    cross_dfs = []
    vicco_dfs = []
    for subject in SUBJECTS:
        # Cross-set controversial scores (all 20 models × all groups)
        p = RSA_DATA_DIR / subject / "cross_set_wrsa_scores.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["subject"] = subject
            cross_dfs.append(df)

        # Vicco scores from all_models set (same images regardless of set)
        p2 = RSA_DATA_DIR / subject / "wrsa_transfer_scores.csv"
        if p2.exists():
            df2 = pd.read_csv(p2)
            df2 = df2[(df2["model_set"] == "all_models") &
                      (df2["stimulus_type"] == "vicco")]
            vicco_dfs.append(df2)

    return pd.concat(cross_dfs, ignore_index=True), pd.concat(vicco_dfs, ignore_index=True)


# ===========================================================================
# Drawing helpers (from plot_brain_alignment.py)
# ===========================================================================

def draw_box(ax, x, mean, sem, width=0.28, filled=True, color="#D64541"):
    lo, hi = mean - sem, mean + sem
    h = hi - lo
    if h < 1e-4:
        h = 0.003; lo = mean - h / 2
    fc = color if filled else "white"
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, lo), width, h,
        boxstyle="square,pad=0",
        facecolor=fc, edgecolor=color,
        linewidth=0.6 if filled else 0.8,
        alpha=0.75 if filled else 0.85, zorder=3 if filled else 4,
    )
    ax.add_patch(rect)
    ax.hlines(mean, x - width / 2, x + width / 2,
              colors="white" if filled else color, linewidth=0.5, zorder=5)


# ===========================================================================
# Panel
# ===========================================================================

def plot_panel(ax, cross_df, vicco_df, controlled_set,
               panel_label=None, show_ylabel=True):
    all_models = MODEL_SETS["all_models"]
    in_set     = set(MODEL_SETS[controlled_set])

    # Per-subject controversial scores on controlled stimuli
    cstim = cross_df[cross_df["stimulus_group"] == controlled_set].copy()

    # Sort models by mean cstim score descending
    model_means = cstim.groupby("model")["wrsa_transfer"].mean()
    order = model_means.sort_values(ascending=False).index.tolist()
    # Keep only models in all_models
    order = [m for m in order if m in all_models]
    n = len(order)

    # Per-subject vicco means (averaged over bootstrap samples)
    vicco_by_model_subj = (
        vicco_df.groupby(["model", "subject"])["wrsa_transfer"].mean()
    )

    offset  = 0.20
    box_w   = 0.28
    dot_size = 8

    for i, model in enumerate(order):
        for stim_type, x_pos, color in [
            ("controversial", i - offset, COLOR_CSTIM),
            ("vicco",         i + offset, COLOR_BASE),
        ]:
            scores = {}
            for subj in SUBJECTS:
                if stim_type == "controversial":
                    v = cstim[(cstim["model"] == model) &
                              (cstim["subject"] == subj)]["wrsa_transfer"]
                    if len(v):
                        scores[subj] = float(v.mean())
                else:
                    key = (model, subj)
                    if key in vicco_by_model_subj.index:
                        scores[subj] = float(vicco_by_model_subj[key])

            if not scores:
                continue

            vals = np.array(list(scores.values()))
            sem  = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            draw_box(ax, x_pos, vals.mean(), sem,
                     width=box_w, filled=True, color=color)

            for subj in SUBJECTS:
                v = scores.get(subj)
                if v is not None:
                    ax.scatter(x_pos, v, s=dot_size,
                               facecolors=color, edgecolors="white",
                               linewidths=0.4, zorder=6, marker="o", alpha=0.85)

        # Connecting lines per subject
        for subj in SUBJECTS:
            c_sub = cstim[(cstim["model"] == model) & (cstim["subject"] == subj)]
            key = (model, subj)
            if len(c_sub) and key in vicco_by_model_subj.index:
                ax.plot([i - offset, i + offset],
                        [float(c_sub["wrsa_transfer"].mean()),
                         float(vicco_by_model_subj[key])],
                        color="#999999", linewidth=0.3, alpha=0.4, zorder=2)

    # X labels — star for in-set models
    labels = []
    for m in order:
        name = SHORT_DISPLAY_NAMES.get(m, MODEL_DISPLAY_NAMES.get(m, m))
        labels.append(f"{name}★" if m in in_set else name)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT.get("tick", 6))

    # Color in-set tick labels
    for tick, model in zip(ax.get_xticklabels(), order):
        tick.set_color("#8B0000" if model in in_set else "black")

    # --- CV ratio annotation ---
    cstim_means = np.array([
        cstim[cstim["model"] == m]["wrsa_transfer"].mean()
        for m in order
        if len(cstim[cstim["model"] == m]) > 0
    ])
    vicco_means = np.array([
        float(vicco_by_model_subj[m].mean())
        for m in order
        if m in vicco_by_model_subj.index.get_level_values("model")
    ])
    if len(cstim_means) >= 2 and len(vicco_means) >= 2 and vicco_means.mean() != 0:
        cv_cstim = cstim_means.std(ddof=1) / cstim_means.mean()
        cv_vicco = vicco_means.std(ddof=1) / vicco_means.mean()
        cv_ratio = cv_cstim / cv_vicco if cv_vicco != 0 else float("nan")
        ax.text(0.97, 0.03, f"CV {cv_ratio:.1f}×",
                transform=ax.transAxes, fontsize=FONT["annotation"],
                ha="right", va="bottom", color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                zorder=10)

    if show_ylabel:
        ax.set_ylabel("mRSA score ($r_s$)")
    ax.set_title(TITLE_MAP.get(controlled_set, controlled_set),
                 fontweight="bold", pad=4)
    if panel_label:
        ax.text(-0.06, 1.12, panel_label, transform=ax.transAxes,
                fontsize=FONT["panel_label"], fontweight="bold", va="top")

    all_vals = []
    for m in order:
        for subj in SUBJECTS:
            v = cstim[(cstim["model"] == m) & (cstim["subject"] == subj)]["wrsa_transfer"]
            if len(v): all_vals.append(float(v.mean()))
            key = (m, subj)
            if key in vicco_by_model_subj.index:
                all_vals.append(float(vicco_by_model_subj[key]))
    if all_vals:
        ax.set_ylim(0, max(all_vals) * 1.10)
    ax.set_xlim(-0.6, n - 0.4)


# ===========================================================================
# Figure
# ===========================================================================

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scores...")
    cross_df, vicco_df = load_scores()
    print(f"  cross: {len(cross_df)} rows, vicco: {len(vicco_df)} rows")

    ctrl_sets = ["sota", "training_objective", "architecture", "dataset"]

    fig = plt.figure(figsize=(W_DOUBLE, 7.0))
    gs  = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.70,
                            left=0.06, right=0.97, top=0.90, bottom=0.20)

    for idx, (ms, lbl) in enumerate(zip(ctrl_sets, "abcd")):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        plot_panel(ax, cross_df, vicco_df, ms,
                   panel_label=lbl, show_ylabel=(col == 0))

    # Legend
    handles = [
        mpatches.Patch(facecolor=COLOR_CSTIM, edgecolor="none",
                       label="Controversial (controlled stimuli)"),
        mpatches.Patch(facecolor=COLOR_BASE,  edgecolor="none",
                       label="Baseline (vicco)"),
        plt.scatter([], [], marker="$★$", s=40, color="#8B0000",
                    label="In-set model"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=True, framealpha=0.95, edgecolor="none",
               columnspacing=1.0, handletextpad=0.5,
               fontsize=FONT.get("legend", 7))

    fig.savefig(FIGURES_DIR / "controlled_all_models.pdf")
    fig.savefig(PNG_DIR / "controlled_all_models.png", dpi=DPI)
    print("  Saved: controlled_all_models.pdf/png")
    plt.close(fig)


if __name__ == "__main__":
    main()
