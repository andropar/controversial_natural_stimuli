#!/usr/bin/env python3
"""Replica of 02_rsa_scores/figures/brain_alignment.pdf using best layers.

For each available metric, produce a two-row figure:
    Row 1 (top):    metric at the best layer per (subject, model, cstim_set)
    Row 2 (bottom): metric at the paper layer (existing 02_rsa_scores values)

Same per-subject dot + connecting-line visualization, same colors, same NC
ribbons. The contrast between rows shows how much per-subject layer choice
recovers cstim alignment vs the single paper-layer choice.

Outputs:
    figures/brain_alignment_best_layer.pdf/png       (fRSA)
    figures/brain_alignment_best_layer_mrsa.pdf/png  (mRSA-transfer)
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from config import (
    PAPER_ROOT, MODEL_SETS, MODEL_DISPLAY_NAMES, SUBJECTS,
    STATS_DATA_DIR, RSA_DATA_DIR,
)
from style import apply_style, FONT, DPI

apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "results"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"
BEST_CSV = DATA_DIR / "best_layer_crsa_scores.csv"
BEST_WRSA_CSV = DATA_DIR / "best_layer_wrsa_scores.csv"

COLOR_CSTIM = "#D64541"
COLOR_BASE = "#2980B9"

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


def load_paper_layer_scores(filename: str) -> pd.DataFrame:
    """Load existing per-subject paper-layer scores from 02_rsa_scores."""
    dfs = []
    for subject in SUBJECTS:
        path = RSA_DATA_DIR / subject / filename
        if path.exists():
            df = pd.read_csv(path)
            if "subject" not in df.columns:
                df["subject"] = subject
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_best_layer_scores(path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_noise_ceilings() -> pd.DataFrame:
    return pd.read_csv(STATS_DATA_DIR / "rdm_noise_ceilings.csv")


def get_per_subject_scores(df, model_set, metric_col):
    """{model: {stim_type: {subject: score}}}."""
    models = MODEL_SETS[model_set]
    subset = df[df["model_set"] == model_set]
    subjects = sorted(subset["subject"].unique())

    data = {}
    for model in models:
        m_sub = subset[subset["model"] == model]
        if m_sub.empty:
            continue
        per_method = {}
        for stim_type in ["controversial", "vicco"]:
            scores = {}
            for subj in subjects:
                v = m_sub[(m_sub["subject"] == subj)
                         & (m_sub["stimulus_type"] == stim_type)][metric_col]
                if len(v) > 0:
                    scores[subj] = v.mean()
            if scores:
                per_method[stim_type] = scores
        if per_method:
            data[model] = per_method
    return data, subjects


def draw_box(ax, x, mean, sem, width=0.30, color=COLOR_CSTIM, filled=False):
    lo = mean - sem
    hi = mean + sem
    h = max(hi - lo, 0.003)
    if hi - lo < 1e-4:
        lo = mean - h / 2
    facecolor = color if filled else "white"
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, lo), width, h,
        boxstyle="square,pad=0",
        facecolor=facecolor, edgecolor=color, linewidth=0.8,
        alpha=0.75 if filled else 0.85, zorder=4,
    )
    ax.add_patch(rect)
    mean_color = "white" if filled else color
    ax.hlines(mean, x - width / 2, x + width / 2,
              colors=mean_color, linewidth=0.5, zorder=5)


def plot_panel(ax, scores_data, model_set, nc_df, fixed_order, is_mrsa=False,
               title="", show_ylabel=True, ylim=None,
               show_xticklabels=True, show_legend=False):
    """Per-subject dots + connecting lines."""
    data, subjects = scores_data
    if not data:
        ax.set_visible(False)
        return

    order = [m for m in fixed_order if m in data]
    n = len(order)

    # NC ribbons (split-half Spearman-Brown; same as plot_brain_alignment.py)
    nc_cstim = nc_df[(nc_df["group"] == model_set) & (nc_df["stimulus_type"] == "controversial")]
    if len(nc_cstim) > 0:
        nc_cstim_vals = np.sqrt(nc_cstim.set_index("subject")["noise_ceiling_spearman"])
    else:
        nc_cstim_vals = np.sqrt(
            nc_df[nc_df["group"] == "vicco"].groupby("subject")["noise_ceiling_spearman"].mean()
        )
    nc_c_mean = nc_cstim_vals.mean()
    nc_c_sem = nc_cstim_vals.std(ddof=1) / np.sqrt(len(nc_cstim_vals)) if len(nc_cstim_vals) > 1 else 0
    nc_vicco_vals = np.sqrt(
        nc_df[nc_df["group"] == "vicco"].groupby("subject")["noise_ceiling_spearman"].mean()
    )
    nc_v_mean = nc_vicco_vals.mean()
    nc_v_sem = nc_vicco_vals.std(ddof=1) / np.sqrt(len(nc_vicco_vals)) if len(nc_vicco_vals) > 1 else 0

    ax.axhspan(nc_c_mean - nc_c_sem, nc_c_mean + nc_c_sem,
               color=COLOR_CSTIM, alpha=0.08, zorder=0, linewidth=0)
    ax.axhline(nc_c_mean, color=COLOR_CSTIM, linewidth=0.4, alpha=0.3, zorder=0)
    ax.axhspan(nc_v_mean - nc_v_sem, nc_v_mean + nc_v_sem,
               color=COLOR_BASE, alpha=0.08, zorder=0, linewidth=0)
    ax.axhline(nc_v_mean, color=COLOR_BASE, linewidth=0.4, alpha=0.3, zorder=0)

    x = np.arange(n)
    offset = 0.20
    box_w = 0.30
    dot_size = 8
    marker = "o" if is_mrsa else "D"
    linestyle = "-" if is_mrsa else "--"
    dot_face = "filled" if is_mrsa else "none"

    for i, m in enumerate(order):
        cstim_scores = data[m].get("controversial", {})
        base_scores = data[m].get("vicco", {})

        if cstim_scores:
            v = np.array(list(cstim_scores.values()))
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            draw_box(ax, x[i] - offset, v.mean(), sem,
                     width=box_w, color=COLOR_CSTIM, filled=is_mrsa)
        if base_scores:
            v = np.array(list(base_scores.values()))
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            draw_box(ax, x[i] + offset, v.mean(), sem,
                     width=box_w, color=COLOR_BASE, filled=is_mrsa)

        for subj in subjects:
            c_val = cstim_scores.get(subj)
            b_val = base_scores.get(subj)
            if c_val is not None and b_val is not None:
                ax.plot([x[i] - offset, x[i] + offset], [c_val, b_val],
                        color="#999999", linewidth=0.3, linestyle=linestyle,
                        alpha=0.4, zorder=2)
            if c_val is not None:
                ax.scatter(x[i] - offset, c_val, s=dot_size * 1.2,
                           facecolors=COLOR_CSTIM if dot_face == "filled" else "none",
                           edgecolors=COLOR_CSTIM,
                           linewidths=0.4 if is_mrsa else 0.6,
                           zorder=6, marker=marker)
            if b_val is not None:
                ax.scatter(x[i] + offset, b_val, s=dot_size * 1.2,
                           facecolors=COLOR_BASE if dot_face == "filled" else "none",
                           edgecolors=COLOR_BASE,
                           linewidths=0.4 if is_mrsa else 0.6,
                           zorder=6, marker=marker)

    ax.set_xticks(x)
    labels = [SHORT_DISPLAY_NAMES.get(m, MODEL_DISPLAY_NAMES.get(m, m)) for m in order]
    if show_xticklabels:
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    if show_ylabel:
        ax.set_ylabel("Score ($r_s$)")
    else:
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)

    if title:
        ax.set_title(title, fontweight="bold", pad=4)

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(-0.6, n - 0.4)

    if show_legend:
        handles = [
            Line2D([0], [0], marker=marker, color="none",
                   markerfacecolor=COLOR_CSTIM if is_mrsa else "none",
                   markeredgecolor=COLOR_CSTIM,
                   markersize=4, markeredgewidth=0.6, label="Controversial stimuli"),
            Line2D([0], [0], marker=marker, color="none",
                   markerfacecolor=COLOR_BASE if is_mrsa else "none",
                   markeredgecolor=COLOR_BASE,
                   markersize=4, markeredgewidth=0.6, label="Baseline stimuli"),
            mpatches.Patch(facecolor=COLOR_CSTIM, alpha=0.20,
                           edgecolor="none", label="Noise ceiling (contr.)"),
            mpatches.Patch(facecolor=COLOR_BASE, alpha=0.20,
                           edgecolor="none", label="Noise ceiling (baseline)"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True,
                  framealpha=0.95, edgecolor="none", ncol=2,
                  columnspacing=0.6, handletextpad=0.3, handlelength=1.2)


def compute_model_order(df, model_set, metric_col):
    """Sort models by cstim mean (descending). Per-set order."""
    data, _ = get_per_subject_scores(df, model_set, metric_col)
    def key(m):
        return np.mean(list(data[m]["controversial"].values())) if "controversial" in data[m] else -999
    return sorted(data.keys(), key=key, reverse=True)


def compute_row_ylim(df, all_sets, nc_df, metric_col):
    all_vals = []
    for ms in all_sets:
        data, _ = get_per_subject_scores(df, ms, metric_col)
        for md in data.values():
            for st in ["controversial", "vicco"]:
                if st in md:
                    all_vals.extend(md[st].values())
    all_vals.extend(np.sqrt(nc_df["noise_ceiling_spearman"].clip(0).values))
    return (0, max(all_vals) * 1.08) if all_vals else (0, 1.0)


def make_figure(df_best, df_paper, nc_df, metric_col, output_stem, row_label, is_mrsa):
    ctrl_sets = ["sota", "training_objective", "architecture", "dataset"]
    all_sets = ["all_models"] + ctrl_sets
    width_ratios = [4, 1, 1, 1, 1]

    # Use the best-layer df to define per-set model order (stable across rows).
    orders = {ms: compute_model_order(df_best, ms, metric_col) for ms in all_sets}

    # Shared ylim across BOTH rows + sets so the visual comparison is honest.
    ylim_best = compute_row_ylim(df_best, all_sets, nc_df, metric_col)
    ylim_paper = compute_row_ylim(df_paper, all_sets, nc_df, metric_col)
    ylim = (0, max(ylim_best[1], ylim_paper[1]))

    # Try to import W_DOUBLE from style if available
    try:
        from style import W_DOUBLE
        figw = W_DOUBLE
    except ImportError:
        figw = 14.0

    fig = plt.figure(figsize=(figw, 7.5))
    gs = fig.add_gridspec(2, 5, width_ratios=width_ratios,
                          wspace=0.06, hspace=0.12,
                          left=0.06, right=0.98, top=0.93, bottom=0.14)

    top_y, bot_y = 0.93, 0.14
    row_centers = [top_y - 0.25 * (top_y - bot_y), bot_y + 0.25 * (top_y - bot_y)]
    row_labels = [(f"Best-layer {row_label}", df_best), (f"Paper-layer {row_label}", df_paper)]

    for row, (label, src_df) in enumerate(row_labels):
        is_bottom = (row == 1)
        fig.text(0.005, row_centers[row], label,
                 va="center", ha="left", fontsize=8, fontweight="bold",
                 rotation=90, transform=fig.transFigure)

        ax_all = fig.add_subplot(gs[row, 0])
        plot_panel(ax_all,
                   get_per_subject_scores(src_df, "all_models", metric_col),
                   "all_models", nc_df,
                   fixed_order=orders["all_models"],
                   is_mrsa=is_mrsa,
                   title="All Models" if row == 0 else "",
                   show_ylabel=True, ylim=ylim,
                   show_xticklabels=is_bottom,
                   show_legend=(row == 0))

        for col, ms in enumerate(ctrl_sets, start=1):
            ax = fig.add_subplot(gs[row, col])
            plot_panel(ax,
                       get_per_subject_scores(src_df, ms, metric_col),
                       ms, nc_df,
                       fixed_order=orders[ms],
                       is_mrsa=is_mrsa,
                       title=TITLE_MAP.get(ms, ms) if row == 0 else "",
                       show_ylabel=False, ylim=ylim,
                       show_xticklabels=is_bottom)

    out_pdf = FIG_DIR / f"{output_stem}.pdf"
    out_png = FIG_DIR / f"{output_stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    nc_df = load_noise_ceilings()

    if BEST_CSV.exists():
        make_figure(
            load_best_layer_scores(BEST_CSV),
            load_paper_layer_scores("crsa_scores.csv"),
            nc_df,
            metric_col="crsa",
            output_stem="brain_alignment_best_layer",
            row_label="fRSA",
            is_mrsa=False,
        )
    else:
        print(f"Skipping fRSA figure; missing {BEST_CSV}")

    if BEST_WRSA_CSV.exists():
        make_figure(
            load_best_layer_scores(BEST_WRSA_CSV),
            load_paper_layer_scores("wrsa_transfer_scores.csv"),
            nc_df,
            metric_col="wrsa_transfer",
            output_stem="brain_alignment_best_layer_mrsa",
            row_label="mRSA",
            is_mrsa=True,
        )
    else:
        print(f"Skipping mRSA figure; missing {BEST_WRSA_CSV}")


if __name__ == "__main__":
    main()
