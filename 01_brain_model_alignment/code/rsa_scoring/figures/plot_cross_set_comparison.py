#!/usr/bin/env python3
"""
Cross-set score comparison: all_models vs. controlled stimulus sets.

For each of the 4 controlled sets (sota, training_objective, architecture,
dataset), compare per-model RSA scores when stimuli were selected by:
  - all_models (left side, lighter)
  - the controlled set itself (right side, darker)

Layout: 2×2 grid, one panel per controlled set.
X-axis: models shared between all_models and the controlled set,
        sorted by their all_models mRSA-cstim rank (descending).
Y-axis: score (r_s)

Per model per panel, 4 sub-positions:
  [all_models cstim] [all_models vicco] | [controlled cstim] [controlled vicco]
  Red = cstim, Blue = vicco.  Filled/solid = mRSA, open/dashed = fRSA.
  Connecting lines link the same subject's all_models vs controlled score
  for matching cstim/vicco and method.

CV annotation: coefficient of variation across models (mean/std ratio),
  shown for all_models vs controlled, per method.

Outputs:
    figures/cross_set_comparison.pdf/png

Usage:
    python plot_cross_set_comparison.py
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_SHARE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))
sys.path.insert(0, str(_PAPER / "figures"))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers"))
sys.path.insert(0, str(_SHARE_ROOT / "shared" / "code" / "paper_helpers" / "figures"))
import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats

from style_improved import apply_style, FONT, DPI, W_DOUBLE

apply_style()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parents[3]
MODEL_SETS = config.MODEL_SETS
MODEL_DISPLAY_NAMES = config.MODEL_DISPLAY_NAMES
SUBJECTS = config.SUBJECTS
RSA_DATA_DIR = STAGE_DIR / "results" / "rsa_scores"
FIGURES_DIR = STAGE_DIR / "figures" / "rsa_scores" / "supplementary"
PNG_DIR = FIGURES_DIR / "png"

# Match brain_alignment color scheme
COLOR_CSTIM_ALL  = "#E8837F"   # all_models cstim  — lighter red
COLOR_CSTIM_CTRL = "#D64541"   # controlled  cstim — darker red
COLOR_BASE_ALL   = "#7FB6D9"   # all_models vicco  — lighter blue
COLOR_BASE_CTRL  = "#2980B9"   # controlled  vicco — darker blue

TITLE_MAP = {
    "training_objective": "Training Objective",
    "sota":               "State of the Art",
    "architecture":       "Architecture",
    "dataset":            "Dataset",
}

SHORT_DISPLAY_NAMES = {
    "torchvision_vgg16_imagenet1k_v1":                    "VGG-16",
    "torchvision_resnet50_imagenet1k_v1":                 "ResNet-50",
    "torchvision_convnext_base_imagenet1k_v1":            "ConvNeXt-B",
    "torchvision_vit_l_16_imagenet1k_v1":                 "ViT-L/16",
    "cornet_s":                                            "CORnet-S",
    "vissl_resnet50_supervised":                           "Supervised",
    "vissl_resnet50_barlowtwins":                          "BarlowTwins",
    "vissl_resnet50_mocov2":                               "MoCoV2",
    "vicreg_resnet50":                                     "VICReg",
    "robustness_imagenet_l2_eps3":                         "Robust-L2",
    "slip_vit_l_slip":                                     "SLIP",
    "slip_vit_l_simclr":                                   "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b":             "CLIP-L2B",
    "dinov2_vitl14":                                       "DINOv2",
    "openclip_vit_so400m_14_siglip_webli":                 "SigLIP",
    "openclip_vit_l_14_quickgelu_metaclip_400m":           "MC-400M",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc":         "MC-Full",
    "timm_vit_large_patch14_clip_224_dfn2b":               "DFN-2B",
    "timm_vit_large_patch14_clip_quickgelu_224_openai":    "CLIP-OAI",
    "openclip_vit_l_14_laion400m_e31":                     "CLIP-L400",
}


# ===========================================================================
# Data loading
# ===========================================================================

def load_all_scores() -> pd.DataFrame:
    dfs = []
    score_files = {
        "crsa_scores.csv":          ("crsa",          "fRSA"),
        "wrsa_transfer_scores.csv": ("wrsa_transfer",  "mRSA"),
    }
    for subject in SUBJECTS:
        data_dir = RSA_DATA_DIR / subject
        for filename, (col_name, method_name) in score_files.items():
            path = data_dir / filename
            if path.exists():
                df = pd.read_csv(path)
                df = df.rename(columns={col_name: "score"})
                df["method"] = method_name
                if "subject" not in df.columns:
                    df["subject"] = subject
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ===========================================================================
# Drawing helpers
# ===========================================================================

def draw_box(ax, x, mean, sem, width=0.20, filled=True, color="#D64541"):
    """Mean ± SEM box, matching brain_alignment style."""
    lo = mean - sem
    hi = mean + sem
    h = hi - lo
    if h < 1e-4:
        h = 0.003
        lo = mean - h / 2

    fc = color if filled else "white"
    lw = 0.6 if filled else 0.8

    rect = mpatches.FancyBboxPatch(
        (x - width / 2, lo), width, h,
        boxstyle="square,pad=0",
        facecolor=fc, edgecolor=color, linewidth=lw,
        alpha=0.75 if filled else 0.85,
        zorder=3 if filled else 4,
    )
    ax.add_patch(rect)
    ml_color = "white" if filled else color
    ax.hlines(mean, x - width / 2, x + width / 2,
              colors=ml_color, linewidth=0.5, zorder=5)


# ===========================================================================
# Per-subject data extraction
# ===========================================================================

def extract_scores(df, model_set, models):
    """
    Returns nested dict:
        {model: {method: {stim_type: {subject: score}}}}
    stim_type: "controversial" | "vicco"
    Only includes models in `models` list.
    """
    subset = df[df["model_set"] == model_set]
    data = {}
    for model in models:
        data[model] = {}
        for method in ["mRSA", "fRSA"]:
            data[model][method] = {}
            m_sub = subset[(subset["model"] == model) & (subset["method"] == method)]
            for stim_type in ["controversial", "vicco"]:
                scores = {}
                for subj in SUBJECTS:
                    v = m_sub[
                        (m_sub["subject"] == subj) &
                        (m_sub["stimulus_type"] == stim_type)
                    ]["score"]
                    if len(v) > 0:
                        scores[subj] = float(v.mean())
                if scores:
                    data[model][method][stim_type] = scores
    return data


def all_models_rank_order(df, models):
    """Sort `models` by descending mRSA-cstim mean across subjects (all_models set)."""
    all_data = extract_scores(df, "all_models", models)

    def sort_key(m):
        d = all_data.get(m, {}).get("mRSA", {}).get("controversial", {})
        return np.mean(list(d.values())) if d else -999.0

    return sorted(models, key=sort_key, reverse=True)


# ===========================================================================
# Panel
# ===========================================================================

def plot_panel(ax, df, controlled_set, panel_label=None, show_ylabel=True):
    """
    One panel comparing all_models vs `controlled_set` for the models in
    `controlled_set`.  X-axis sorted by all_models mRSA-cstim rank.
    """
    models_in_ctrl = MODEL_SETS[controlled_set]
    # Only models present in both all_models and controlled_set data
    all_models_in_data = df[df["model_set"] == "all_models"]["model"].unique()
    ctrl_in_data = df[df["model_set"] == controlled_set]["model"].unique()
    shared = [m for m in models_in_ctrl
              if m in all_models_in_data and m in ctrl_in_data]
    if not shared:
        ax.set_visible(False)
        return

    order = all_models_rank_order(df, shared)
    n = len(order)

    all_data  = extract_scores(df, "all_models",   order)
    ctrl_data = extract_scores(df, controlled_set, order)

    # Sub-position offsets within each model slot
    # Layout per model: [all_cstim, all_vicco] | [ctrl_cstim, ctrl_vicco]
    # Gap between all and ctrl groups is wider.
    off = 0.13   # spacing between adjacent sub-positions
    gap = 0.10   # extra gap between all and ctrl halves
    # Positions relative to model centre:
    pos_all_cstim  = -(1.5 * off + gap / 2)
    pos_all_vicco  = -(0.5 * off + gap / 2)
    pos_ctrl_cstim = +(0.5 * off + gap / 2)
    pos_ctrl_vicco = +(1.5 * off + gap / 2)

    sub_positions = {
        ("all_models",   "controversial"): pos_all_cstim,
        ("all_models",   "vicco"):         pos_all_vicco,
        (controlled_set, "controversial"): pos_ctrl_cstim,
        (controlled_set, "vicco"):         pos_ctrl_vicco,
    }
    colors = {
        ("all_models",   "controversial"): COLOR_CSTIM_ALL,
        ("all_models",   "vicco"):         COLOR_BASE_ALL,
        (controlled_set, "controversial"): COLOR_CSTIM_CTRL,
        (controlled_set, "vicco"):         COLOR_BASE_CTRL,
    }
    data_lookup = {
        "all_models":   all_data,
        controlled_set: ctrl_data,
    }

    dot_size = 7
    box_w    = 0.20

    # Vertical separator between all and ctrl halves
    for xi in range(n):
        ax.axvline(xi, color="#cccccc", linewidth=0.4, linestyle=":", zorder=0)

    for i, model in enumerate(order):
        # For connecting lines: link same subject across all_models↔ctrl for
        # same stim_type and method
        for method in ["mRSA", "fRSA"]:
            is_versa = method == "mRSA"
            for stim_type in ["controversial", "vicco"]:
                xA  = i + pos_all_cstim  if stim_type == "controversial" else i + pos_all_vicco
                xC  = i + pos_ctrl_cstim if stim_type == "controversial" else i + pos_ctrl_vicco
                cA  = colors[("all_models",   stim_type)]
                cC  = colors[(controlled_set, stim_type)]

                a_scores = all_data[model].get(method, {}).get(stim_type, {})
                c_scores = ctrl_data[model].get(method, {}).get(stim_type, {})

                # Mean box for all_models side
                if a_scores:
                    vals = np.array(list(a_scores.values()))
                    draw_box(ax, xA, vals.mean(),
                             vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0,
                             width=box_w, filled=is_versa, color=cA)

                # Mean box for controlled side
                if c_scores:
                    vals = np.array(list(c_scores.values()))
                    draw_box(ax, xC, vals.mean(),
                             vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0,
                             width=box_w, filled=is_versa, color=cC)

                # Per-subject dots and connecting lines
                for subj in SUBJECTS:
                    a_val = a_scores.get(subj)
                    c_val = c_scores.get(subj)

                    if a_val is not None and c_val is not None:
                        ls = "-" if is_versa else "--"
                        ax.plot([xA, xC], [a_val, c_val],
                                color="#999999", linewidth=0.3, linestyle=ls,
                                alpha=0.35, zorder=2)

                    for x_pos, val, color in [(xA, a_val, cA), (xC, c_val, cC)]:
                        if val is None:
                            continue
                        if is_versa:
                            ax.scatter(x_pos, val, s=dot_size,
                                       facecolors=color, edgecolors="white",
                                       linewidths=0.4, zorder=6, marker="o",
                                       alpha=0.85)
                        else:
                            ax.scatter(x_pos, val, s=dot_size * 1.2,
                                       facecolors="none", edgecolors=color,
                                       linewidths=0.6, zorder=7, marker="D")

    # --- Annotation: CV(cstim)/CV(baseline) for each set + Spearman ρ ---
    # CV ratio: how much more spread cstim scores are vs baseline scores.
    # Spearman ρ: correspondence of model score orderings between sets.
    def _model_means(data, method, stim_type):
        return {m: np.mean(list(data[m].get(method, {}).get(stim_type, {}).values()))
                for m in order if data[m].get(method, {}).get(stim_type, {})}

    def _cv(means_dict):
        vals = np.array(list(means_dict.values()))
        return vals.std(ddof=1) / vals.mean() if len(vals) >= 2 and vals.mean() != 0 else float("nan")

    ann_lines = []
    for method in ["mRSA", "fRSA"]:
        cstim_all  = _model_means(all_data,  method, "controversial")
        base_all   = _model_means(all_data,  method, "vicco")
        cstim_ctrl = _model_means(ctrl_data, method, "controversial")
        base_ctrl  = _model_means(ctrl_data, method, "vicco")

        cv_ratio_all  = _cv(cstim_all)  / _cv(base_all)  if _cv(base_all)  != 0 else float("nan")
        cv_ratio_ctrl = _cv(cstim_ctrl) / _cv(base_ctrl) if _cv(base_ctrl) != 0 else float("nan")

        # Spearman ρ between all_models and controlled cstim scores
        shared = [m for m in order if m in cstim_all and m in cstim_ctrl]
        if len(shared) >= 2:
            rho, p = stats.spearmanr(
                [cstim_all[m] for m in shared],
                [cstim_ctrl[m] for m in shared],
            )
            p_str = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            rho_str = f"  ρ={rho:.2f}{p_str}"
        else:
            rho_str = ""

        # Mean magnitude shift: cstim only (baseline is identical across sets)
        shared_cstim = [m for m in order if m in cstim_all and m in cstim_ctrl]
        mu_cstim_all  = np.mean([cstim_all[m]  for m in shared_cstim]) if shared_cstim else float("nan")
        mu_cstim_ctrl = np.mean([cstim_ctrl[m] for m in shared_cstim]) if shared_cstim else float("nan")
        d_cstim = mu_cstim_ctrl - mu_cstim_all

        parts = []
        if not np.isnan(cv_ratio_all):
            parts.append(f"all×{cv_ratio_all:.2f}")
        if not np.isnan(cv_ratio_ctrl):
            parts.append(f"ctrl×{cv_ratio_ctrl:.2f}")
        if parts:
            line = f"{method}: CV " + " / ".join(parts) + rho_str
            if not np.isnan(d_cstim):
                line += f"  Δμ{d_cstim:+.3f}"
            ann_lines.append(line)

    if ann_lines:
        ax.text(0.5, -0.38, "\n".join(ann_lines),
                transform=ax.transAxes,
                fontsize=FONT["annotation"] - 0.5,
                ha="center", va="top", color="#555555",
                family="monospace",
                zorder=10)

    # --- Formatting ---
    labels = [SHORT_DISPLAY_NAMES.get(m, MODEL_DISPLAY_NAMES.get(m, m))
              for m in order]
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    if show_ylabel:
        ax.set_ylabel("Score ($r_s$)")

    title = TITLE_MAP.get(controlled_set, controlled_set)
    ax.set_title(title, fontweight="bold", pad=4)

    if panel_label:
        ax.text(-0.08, 1.12, panel_label, transform=ax.transAxes,
                fontsize=FONT["panel_label"], fontweight="bold", va="top")

    # Y-limits
    all_vals = []
    for model in order:
        for src_data in [all_data, ctrl_data]:
            for method in ["mRSA", "fRSA"]:
                for st in ["controversial", "vicco"]:
                    all_vals.extend(src_data[model].get(method, {}).get(st, {}).values())
    if all_vals:
        ax.set_ylim(0, max(all_vals) * 1.10)
    ax.set_xlim(-0.6, n - 0.4)


# ===========================================================================
# Legend
# ===========================================================================

def make_legend(fig):
    handles = [
        # Stimulus type
        mpatches.Patch(facecolor=COLOR_CSTIM_CTRL, edgecolor="none",
                       label="Cstim (controlled)", alpha=0.85),
        mpatches.Patch(facecolor=COLOR_CSTIM_ALL,  edgecolor="none",
                       label="Cstim (all models)", alpha=0.85),
        mpatches.Patch(facecolor=COLOR_BASE_CTRL,  edgecolor="none",
                       label="Baseline (controlled)", alpha=0.85),
        mpatches.Patch(facecolor=COLOR_BASE_ALL,   edgecolor="none",
                       label="Baseline (all models)", alpha=0.85),
        # Method
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#888888",
               markeredgecolor="#888888", markersize=4, label="mRSA (filled)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="none",
               markeredgecolor="#888888", markersize=4, markeredgewidth=0.8,
               label="fRSA (open)"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.99),
               ncol=6, frameon=True, framealpha=0.95, edgecolor="none",
               columnspacing=0.8, handletextpad=0.4, handlelength=1.2,
               fontsize=FONT.get("legend", 7))


# ===========================================================================
# Figure assembly
# ===========================================================================

def plot_figure(df):
    ctrl_sets = ["sota", "training_objective", "architecture", "dataset"]
    panel_labels = list("abcd")

    fig = plt.figure(figsize=(W_DOUBLE, 6.5))
    gs = gridspec.GridSpec(
        2, 2,
        wspace=0.30, hspace=0.90,
        left=0.06, right=0.97, top=0.88, bottom=0.18,
    )

    for idx, (ms, lbl) in enumerate(zip(ctrl_sets, panel_labels)):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        plot_panel(ax, df, ms,
                   panel_label=lbl,
                   show_ylabel=(col == 0))

    make_legend(fig)
    return fig


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scores...")
    df = load_all_scores()
    print(f"  {df['subject'].nunique()} subjects, {len(df)} records")

    print("Plotting cross-set comparison...")
    fig = plot_figure(df)
    fig.savefig(FIGURES_DIR / "cross_set_comparison.pdf")
    fig.savefig(PNG_DIR / "cross_set_comparison.png", dpi=DPI)
    print("  Saved: cross_set_comparison.pdf, cross_set_comparison.png")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
