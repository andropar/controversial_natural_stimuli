#!/usr/bin/env python3
"""
Cross-set mRSA: in-set vs out-of-set models on controlled stimuli.

For each controlled set, compares mRSA scores when evaluated on:
  - all_models stimuli (left)
  - controlled stimuli (right)

Separately for in-set models (optimized for) vs out-of-set models (not
optimized for). If both groups increase equally, the effect is a property
of the images. If only in-set increases, it's optimization-specific.

Layout: 2×2 grid, one panel per controlled set.
Each panel:
  - X: all_models stimuli | controlled stimuli
  - Y: mRSA score
  - Per-model thin lines (one per model, colored by in/out-of-set)
  - Thick line = mean across models × subjects

Outputs:
    figures/cross_set_out_of_set.pdf/png

Usage:
    python plot_cross_set_out_of_set.py
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
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from cstims.paper.style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

STAGE_DIR = Path(__file__).resolve().parents[3]
SUBJECTS = config.SUBJECTS
MODEL_SETS = config.MODEL_SETS
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

COLOR_IN  = "#D64541"   # red — in-set
COLOR_OUT = "#2980B9"   # blue — out-of-set

TITLE_MAP = {
    "training_objective": "Training Objective",
    "sota":               "State of the Art",
    "architecture":       "Architecture",
    "dataset":            "Dataset",
}

X_ALL  = 0
X_CTRL = 1
X_LABELS = ["all_models", "controlled"]


def load_data() -> pd.DataFrame:
    dfs = []
    for subject in SUBJECTS:
        p = RSA_DATA_DIR / subject / "cross_set_wrsa_scores.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True)


def plot_panel(ax, df, controlled_set, panel_label=None, show_ylabel=True):
    # Subset to relevant groups
    sub = df[df["stimulus_group"].isin(["all_models", controlled_set])].copy()

    # Per-model mean across subjects
    model_means = (
        sub.groupby(["model", "stimulus_group", "in_set"])["wrsa_transfer"]
        .mean()
        .reset_index()
    )

    in_set_models  = model_means[model_means["in_set"]  == True]["model"].unique()
    out_set_models = model_means[model_means["in_set"]  == False]["model"].unique()

    def get_score(model, group):
        v = model_means[
            (model_means["model"] == model) &
            (model_means["stimulus_group"] == group)
        ]["wrsa_transfer"]
        return float(v.iloc[0]) if len(v) else None

    # --- Per-model thin lines ---
    for model, color, alpha, lw in [
        *[(m, COLOR_IN,  0.5, 0.8) for m in in_set_models],
        *[(m, COLOR_OUT, 0.35, 0.6) for m in out_set_models],
    ]:
        y_all  = get_score(model, "all_models")
        y_ctrl = get_score(model, controlled_set)
        if y_all is None or y_ctrl is None:
            continue
        ax.plot([X_ALL, X_CTRL], [y_all, y_ctrl],
                color=color, linewidth=lw, alpha=alpha, zorder=2)
        ax.scatter([X_ALL, X_CTRL], [y_all, y_ctrl],
                   s=10, color=color, alpha=alpha + 0.2, zorder=3, linewidths=0)

    # --- Mean lines (per group) ---
    for models, color in [(in_set_models, COLOR_IN), (out_set_models, COLOR_OUT)]:
        ys_all  = [get_score(m, "all_models")   for m in models]
        ys_ctrl = [get_score(m, controlled_set) for m in models]
        ys_all  = [y for y in ys_all  if y is not None]
        ys_ctrl = [y for y in ys_ctrl if y is not None]
        if not ys_all or not ys_ctrl:
            continue
        mu_all  = np.mean(ys_all)
        mu_ctrl = np.mean(ys_ctrl)
        sem_all  = np.std(ys_all,  ddof=1) / np.sqrt(len(ys_all))
        sem_ctrl = np.std(ys_ctrl, ddof=1) / np.sqrt(len(ys_ctrl))

        ax.plot([X_ALL, X_CTRL], [mu_all, mu_ctrl],
                color=color, linewidth=2.0, zorder=5)
        for x, mu, sem in [(X_ALL, mu_all, sem_all), (X_CTRL, mu_ctrl, sem_ctrl)]:
            ax.errorbar(x, mu, yerr=sem, fmt="o", color=color,
                        markersize=5, linewidth=1.5, capsize=3, zorder=6)

        # Delta annotation next to each mean line
        delta = mu_ctrl - mu_all
        ax.annotate(f"{delta:+.3f}",
                    xy=(X_CTRL, mu_ctrl),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=FONT["annotation"], color=color, va="center")

    # --- Formatting ---
    ax.set_xticks([X_ALL, X_CTRL])
    ax.set_xticklabels(["all\nmodels", TITLE_MAP.get(controlled_set, controlled_set)],
                       fontsize=FONT.get("tick", 7))
    ax.set_xlim(-0.4, 1.6)

    all_vals = model_means["wrsa_transfer"].values
    ax.set_ylim(max(0, all_vals.min() * 0.9), all_vals.max() * 1.12)

    if show_ylabel:
        ax.set_ylabel("mRSA score ($r_s$)")
    ax.set_title(TITLE_MAP.get(controlled_set, controlled_set),
                 fontweight="bold", pad=4)
    if panel_label:
        ax.text(-0.12, 1.12, panel_label, transform=ax.transAxes,
                fontsize=FONT["panel_label"], fontweight="bold", va="top")


def make_legend(fig):
    handles = [
        Line2D([0], [0], color=COLOR_IN,  linewidth=2, label="In-set models"),
        Line2D([0], [0], color=COLOR_OUT, linewidth=2, label="Out-of-set models"),
        Line2D([0], [0], color="#888888", linewidth=0.8, alpha=0.6,
               label="Individual models (thin)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=True, framealpha=0.95, edgecolor="none",
               columnspacing=1.0, handletextpad=0.5,
               fontsize=FONT.get("legend", 7))


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cross-set scores...")
    df = load_data()
    print(f"  {df['subject'].nunique()} subjects, {len(df)} rows")

    ctrl_sets = ["sota", "training_objective", "architecture", "dataset"]
    fig = plt.figure(figsize=(W_DOUBLE * 0.7, 5.5))
    gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.55,
                           left=0.10, right=0.90, top=0.88, bottom=0.12)

    for idx, (ms, lbl) in enumerate(zip(ctrl_sets, "abcd")):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        plot_panel(ax, df, ms, panel_label=lbl, show_ylabel=(col == 0))

    make_legend(fig)

    fig.savefig(FIGURES_DIR / "cross_set_out_of_set.pdf")
    fig.savefig(PNG_DIR / "cross_set_out_of_set.png", dpi=DPI)
    print("  Saved: cross_set_out_of_set.pdf/png")
    plt.close(fig)


if __name__ == "__main__":
    main()
