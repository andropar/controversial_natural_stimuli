#!/usr/bin/env python3
"""Full replica of 02_rsa_scores/figures/brain_alignment.pdf with both
mRSA-transfer (top row) and fRSA (bottom row) at the BEST per-(subject,
model, set) layer.

Layout matches the original brain_alignment.pdf:
    Row 1 (top):    mRSA at best layer (filled circles, solid lines)
    Row 2 (bottom): fRSA at best layer (open diamonds, dashed lines)

Same color scheme: red=controversial, blue=baseline.
NC ribbons from rdm_noise_ceilings.csv.

Outputs:
    figures/brain_alignment_best_layer_full.pdf/png
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

DATA_DIR = LAYER_SWEEP_ROOT / "data"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"

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


def load_scores():
    """Load both metrics' best-layer csvs and convert to a unified frame."""
    out = []
    crsa_path = DATA_DIR / "best_layer_crsa_scores.csv"
    if crsa_path.exists():
        d = pd.read_csv(crsa_path).rename(columns={"crsa": "score"})
        d["method"] = "fRSA"
        out.append(d)
    wrsa_path = DATA_DIR / "best_layer_wrsa_scores.csv"
    if wrsa_path.exists():
        d = pd.read_csv(wrsa_path).rename(columns={"wrsa_transfer": "score"})
        d["method"] = "mRSA"
        out.append(d)
    if not out:
        raise FileNotFoundError("No best-layer csvs found")
    return pd.concat(out, ignore_index=True)


def load_noise_ceilings():
    return pd.read_csv(STATS_DATA_DIR / "rdm_noise_ceilings.csv")


def get_per_subject_scores(df, model_set):
    models = MODEL_SETS[model_set]
    subset = df[df["model_set"] == model_set]
    subjects = sorted(subset["subject"].unique())
    data = {}
    for model in models:
        m_sub = subset[subset["model"] == model]
        if m_sub.empty:
            continue
        per_method = {}
        for method in ["mRSA", "fRSA"]:
            per_method[method] = {}
            for stim_type in ["controversial", "vicco"]:
                scores = {}
                for subj in subjects:
                    v = m_sub[(m_sub["subject"] == subj)
                             & (m_sub["method"] == method)
                             & (m_sub["stimulus_type"] == stim_type)]["score"]
                    if len(v) > 0:
                        scores[subj] = v.mean()
                if scores:
                    per_method[method][stim_type] = scores
            if not per_method[method]:
                del per_method[method]
        if per_method:
            data[model] = per_method
    return data, subjects


def draw_box(ax, x, mean, sem, width=0.30, filled=True, color=COLOR_CSTIM):
    lo = mean - sem
    hi = mean + sem
    h = max(hi - lo, 0.003)
    if hi - lo < 1e-4:
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


def plot_panel(ax, scores_data, model_set, nc_df, fixed_order, method,
               title="", show_ylabel=True, ylim=None,
               show_xticklabels=True, show_legend=False):
    data, subjects = scores_data
    if not data:
        ax.set_visible(False)
        return

    is_mrsa = (method == "mRSA")
    order = [m for m in fixed_order if m in data and method in data[m]]
    n = len(order)
    if n == 0:
        ax.set_visible(False)
        return

    # NC ribbons
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

    for i, m in enumerate(order):
        cstim_scores = data[m][method].get("controversial", {})
        base_scores = data[m][method].get("vicco", {})

        if cstim_scores:
            v = np.array(list(cstim_scores.values()))
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            draw_box(ax, x[i] - offset, v.mean(), sem,
                     width=box_w, filled=is_mrsa, color=COLOR_CSTIM)
        if base_scores:
            v = np.array(list(base_scores.values()))
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            draw_box(ax, x[i] + offset, v.mean(), sem,
                     width=box_w, filled=is_mrsa, color=COLOR_BASE)

        for subj in subjects:
            c_val = cstim_scores.get(subj)
            b_val = base_scores.get(subj)
            ls = "-" if is_mrsa else "--"
            if c_val is not None and b_val is not None:
                ax.plot([x[i] - offset, x[i] + offset], [c_val, b_val],
                        color="#999999", linewidth=0.3, linestyle=ls,
                        alpha=0.4, zorder=2)
            mk = "o" if is_mrsa else "D"
            ms_scale = 1.0 if is_mrsa else 1.2
            if c_val is not None:
                ax.scatter(x[i] - offset, c_val, s=dot_size * ms_scale,
                           facecolors=COLOR_CSTIM if is_mrsa else "none",
                           edgecolors=COLOR_CSTIM,
                           linewidths=0.4 if is_mrsa else 0.6,
                           zorder=6, marker=mk, alpha=0.85 if is_mrsa else 1.0)
            if b_val is not None:
                ax.scatter(x[i] + offset, b_val, s=dot_size * ms_scale,
                           facecolors=COLOR_BASE if is_mrsa else "none",
                           edgecolors=COLOR_BASE,
                           linewidths=0.4 if is_mrsa else 0.6,
                           zorder=6, marker=mk, alpha=0.85 if is_mrsa else 1.0)

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
        mk = "o" if is_mrsa else "D"
        lw = 0.4 if is_mrsa else 0.6
        handles = [
            Line2D([0], [0], marker=mk, color="none",
                   markerfacecolor=COLOR_CSTIM if is_mrsa else "none",
                   markeredgecolor=COLOR_CSTIM, markersize=4,
                   markeredgewidth=lw, label="Controversial stimuli"),
            Line2D([0], [0], marker=mk, color="none",
                   markerfacecolor=COLOR_BASE if is_mrsa else "none",
                   markeredgecolor=COLOR_BASE, markersize=4,
                   markeredgewidth=lw, label="Baseline stimuli"),
            mpatches.Patch(facecolor=COLOR_CSTIM, alpha=0.20,
                           edgecolor="none", label="Noise ceiling (contr.)"),
            mpatches.Patch(facecolor=COLOR_BASE, alpha=0.20,
                           edgecolor="none", label="Noise ceiling (baseline)"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True,
                  framealpha=0.95, edgecolor="none", ncol=2,
                  columnspacing=0.6, handletextpad=0.3, handlelength=1.2)


def compute_model_order(df, model_set, method):
    data, _ = get_per_subject_scores(df, model_set)
    def key(m):
        if method in data[m] and "controversial" in data[m][method]:
            return np.mean(list(data[m][method]["controversial"].values()))
        return -999
    return sorted(data.keys(), key=key, reverse=True)


def compute_row_ylim(df, all_sets, method, nc_df):
    all_vals = []
    for ms in all_sets:
        data, _ = get_per_subject_scores(df, ms)
        for md in data.values():
            if method in md:
                for st in ["controversial", "vicco"]:
                    if st in md[method]:
                        all_vals.extend(md[method][st].values())
    all_vals.extend(np.sqrt(nc_df["noise_ceiling_spearman"].clip(0).values))
    return (0, max(all_vals) * 1.08) if all_vals else (0, 1.0)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_scores()
    nc_df = load_noise_ceilings()

    methods = sorted(df["method"].unique(), key=lambda m: 0 if m == "mRSA" else 1)
    print("Methods present:", methods)

    ctrl_sets = ["sota", "training_objective", "architecture", "dataset"]
    all_sets = ["all_models"] + ctrl_sets
    width_ratios = [4, 1, 1, 1, 1]

    # Per-(set, method) order: use method-specific cstim mean for sorting
    primary_method = methods[0]
    orders = {ms: compute_model_order(df, ms, primary_method) for ms in all_sets}
    ylims = {meth: compute_row_ylim(df, all_sets, meth, nc_df) for meth in methods}

    try:
        from style import W_DOUBLE
        figw = W_DOUBLE
    except ImportError:
        figw = 14.0

    n_rows = len(methods)
    fig = plt.figure(figsize=(figw, 4.2 * n_rows))
    gs = fig.add_gridspec(n_rows, 5, width_ratios=width_ratios,
                          wspace=0.06, hspace=0.12,
                          left=0.06, right=0.98, top=0.93, bottom=0.14)

    top_y, bot_y = 0.93, 0.14
    if n_rows == 1:
        row_centers = [(top_y + bot_y) / 2]
    else:
        row_centers = [top_y - 0.25 * (top_y - bot_y),
                       bot_y + 0.25 * (top_y - bot_y)]

    method_labels = {"mRSA": "Mixed RSA", "fRSA": "Fixed RSA"}

    for row, method in enumerate(methods):
        is_bottom = (row == n_rows - 1)
        fig.text(0.005, row_centers[row], method_labels.get(method, method),
                 va="center", ha="left", fontsize=8, fontweight="bold",
                 rotation=90, transform=fig.transFigure)
        ylim = ylims[method]

        ax_all = fig.add_subplot(gs[row, 0])
        plot_panel(ax_all,
                   get_per_subject_scores(df, "all_models"),
                   "all_models", nc_df,
                   fixed_order=orders["all_models"], method=method,
                   title="All Models" if row == 0 else "",
                   show_ylabel=True, ylim=ylim,
                   show_xticklabels=is_bottom,
                   show_legend=(row == 0))

        for col, ms in enumerate(ctrl_sets, start=1):
            ax = fig.add_subplot(gs[row, col])
            plot_panel(ax,
                       get_per_subject_scores(df, ms),
                       ms, nc_df,
                       fixed_order=orders[ms], method=method,
                       title=TITLE_MAP.get(ms, ms) if row == 0 else "",
                       show_ylabel=False, ylim=ylim,
                       show_xticklabels=is_bottom)

    out_pdf = FIG_DIR / "brain_alignment_best_layer_full.pdf"
    out_png = FIG_DIR / "brain_alignment_best_layer_full.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


if __name__ == "__main__":
    main()
