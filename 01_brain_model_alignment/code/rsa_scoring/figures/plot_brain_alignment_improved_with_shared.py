#!/usr/bin/env python3
"""Brain-alignment figure variant that adds the DeepVision shared-stimulus
results next to the existing controversial / vicco-baseline boxes.

Layout per model (3 boxes side by side):
    cstim   (vermillion, left)
    vicco   (blue,       middle)   <- existing baseline
    shared  (green,      right)    <- new: DeepVision shared brain data,
                                       unique-trained encoding applied OOD

The shared box sits in the same x-axis cluster as the vicco baseline so the
"baseline-ish" comparison stays grouped on the right of each model triple.

NC bands are drawn for cstim + vicco only (no NC available for shared yet).
"""
from __future__ import annotations

import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[3]
SHARE_ROOT = STAGE.parent
sys.path.insert(0, str(SHARE_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from cstims.paper import config
from cstims.paper.style_improved import (
    apply_style, FONT, DPI, W_DOUBLE, OKABE_ITO,
    COLOR_CSTIM, COLOR_BASELINE,
    add_panel_label,
)

apply_style()

MODEL_SETS = config.MODEL_SETS
MODEL_DISPLAY = config.MODEL_DISPLAY_NAMES
SUBJECTS = config.SUBJECTS
STATS_DATA_DIR = config.STATS_DATA_DIR
RSA_DATA_DIR = config.RSA_DATA_DIR
FIGURES_DIR = STAGE / "figures" / "rsa_scores"

COLOR_SHARED = OKABE_ITO["bluish_green"]   # #009E73 — distinct from blue + vermillion

SHORT_NAMES = {
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

TITLE = {
    "training_objective": "Training Objective",
    "sota":               "State of the Art",
    "architecture":       "Architecture",
    "dataset":            "Dataset",
    "all_models":         "All Models",
}

PANEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]

# 3-box layout: cstim left, vicco middle, shared right.
# Cluster spans roughly [x-0.37, x+0.37].
OFFSET_CSTIM  = -0.27
OFFSET_VICCO  =  0.0
OFFSET_SHARED = +0.27
BOX_WIDTH = 0.20

STIM_TYPES = ("controversial", "vicco", "deepvision_shared")
STIM_DISPLAY = {
    "controversial":     "Controversial",
    "vicco":             "Baseline (vicco)",
    "deepvision_shared": "DeepVision shared",
}
STIM_COLOR = {
    "controversial":     COLOR_CSTIM,
    "vicco":             COLOR_BASELINE,
    "deepvision_shared": COLOR_SHARED,
}
STIM_OFFSET = {
    "controversial":     OFFSET_CSTIM,
    "vicco":             OFFSET_VICCO,
    "deepvision_shared": OFFSET_SHARED,
}


def load_scores():
    dfs = []
    score_files = {
        "crsa_scores.csv":          ("crsa",          "fRSA"),
        "wrsa_transfer_scores.csv": ("wrsa_transfer", "mRSA"),
    }
    for sub in SUBJECTS:
        d = RSA_DATA_DIR / sub
        for fname, (col, method) in score_files.items():
            p = d / fname
            if p.exists():
                df = pd.read_csv(p).rename(columns={col: "score"})
                df["method"] = method
                if "subject" not in df.columns:
                    df["subject"] = sub
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_nc():
    return pd.read_csv(STATS_DATA_DIR / "rdm_noise_ceilings.csv")


def load_bs_nc():
    p = SHARE_ROOT / "02_alignment_reliability" / "results" / "between_subject_noise_ceilings.csv"
    return pd.read_csv(p)


def load_perm():
    p = STATS_DATA_DIR / "permutation_test_results.csv"
    return pd.read_csv(p) if p.exists() else None


def get_per_subject_scores(df, model_set):
    """Return {model: {method: {stim_type: {subject: mean_score}}}}, subjects."""
    models = MODEL_SETS[model_set]
    sub = df[df["model_set"] == model_set]
    subjects = sorted(sub["subject"].unique())
    out = {}
    for m in models:
        out[m] = {}
        for method in ("mRSA", "fRSA"):
            block = sub[(sub["model"] == m) & (sub["method"] == method)]
            for st in STIM_TYPES:
                d = (block[block["stimulus_type"] == st]
                     .groupby("subject")["score"].mean().to_dict())
                if d:
                    out[m].setdefault(method, {})[st] = d
    return {m: d for m, d in out.items() if d}, subjects


def lookup_spread(perm_df, model_set, method):
    if perm_df is None:
        return None
    csv_method = {"mRSA": "wrsa_transfer", "fRSA": "crsa"}[method]
    rows = perm_df[
        (perm_df["model_set"] == model_set)
        & (perm_df["method"] == csv_method)
        & (perm_df["metric"] == "median_pairwise_diff")
    ]
    if rows.empty:
        return None
    return float(rows["observed_ratio"].iloc[0])


def draw_box(ax, x, mean, sem, width=BOX_WIDTH, filled=True, color=COLOR_CSTIM):
    lo, hi = mean - sem, mean + sem
    if hi - lo < 1e-4:
        lo, hi = mean - 0.0015, mean + 0.0015
    fc = color if filled else "white"
    ec = color
    rect = mpatches.Rectangle((x - width/2, lo), width, hi - lo,
                              facecolor=fc, edgecolor=ec, linewidth=0.6,
                              alpha=0.75 if filled else 0.95, zorder=3)
    ax.add_patch(rect)
    ml_color = "white" if filled else ec
    ax.hlines(mean, x - width/2, x + width/2, colors=ml_color,
              linewidth=0.7, zorder=5)


def plot_method_panel(ax, df, nc_df, bs_nc_df, perm_df, model_set, method,
                      show_xticks, panel_label=None, show_legend=False,
                      is_largest_amp=False):
    data, subjects = get_per_subject_scores(df, model_set)
    if not data:
        ax.set_visible(False); return

    is_mrsa = (method == "mRSA")

    # Sort models by mean mixed-RSA score on the controversial stimuli.
    # Keep that order for both rows so fixed RSA is directly comparable to
    # the mixed-RSA ordering used in the paper.
    def key(m):
        s = data[m].get("mRSA", {}).get("controversial", {})
        return np.mean(list(s.values())) if s else -999
    order = [m for m in data if method in data[m]]
    order.sort(key=key, reverse=True)

    # ---- Noise-ceiling bands (cstim + vicco only — none for shared yet) ----
    if method == "mRSA":
        nc_ctrl = (nc_df[(nc_df["group"] == model_set)
                          & (nc_df["stimulus_type"] == "controversial")]
                   .groupby("subject")["noise_ceiling_spearman"].mean())
        nc_base = (nc_df[nc_df["group"] == "vicco"]
                   .groupby("subject")["noise_ceiling_spearman"].mean())
        nc_ctrl_v = np.sqrt(nc_ctrl.values) if len(nc_ctrl) else np.array([np.nan])
        nc_base_v = np.sqrt(nc_base.values)
    else:
        nc_ctrl_rows = bs_nc_df[(bs_nc_df["group"] == model_set)
                                 & (bs_nc_df["stimulus_type"] == "controversial")]
        nc_base_rows = bs_nc_df[bs_nc_df["group"] == "vicco"]
        nc_ctrl_v = nc_ctrl_rows["nc_mid"].values if len(nc_ctrl_rows) else \
                    nc_base_rows["nc_mid"].values
        nc_base_v = nc_base_rows["nc_mid"].values

    def _band(vals):
        m = np.nanmean(vals); s = np.nanstd(vals, ddof=1) / np.sqrt(max(len(vals), 1))
        return m - s, m + s, m

    nc_drawn = []
    nc_ctrl_mean = float(np.nanmean(nc_ctrl_v)) if len(nc_ctrl_v) else np.nan
    nc_base_mean = float(np.nanmean(nc_base_v)) if len(nc_base_v) else np.nan

    if is_mrsa:
        if not np.isnan(nc_ctrl_mean):
            lo, hi, m = _band(nc_ctrl_v)
            ax.axhspan(lo, hi, color=COLOR_CSTIM, alpha=0.10, zorder=0, linewidth=0)
            ax.axhline(m, color=COLOR_CSTIM, linewidth=0.9, alpha=0.7, zorder=0,
                       linestyle="-")
        if not np.isnan(nc_base_mean):
            lo, hi, m = _band(nc_base_v)
            ax.axhspan(lo, hi, color=COLOR_BASELINE, alpha=0.10, zorder=0, linewidth=0)
            ax.axhline(m, color=COLOR_BASELINE, linewidth=0.9, alpha=0.7, zorder=0,
                       linestyle="--")
    else:
        nc_drawn = [("base",  nc_base_mean, COLOR_BASELINE),
                    ("cstim", nc_ctrl_mean, COLOR_CSTIM)]

    # ---- Per-model: 3 boxes + per-subject dots ----
    n = len(order)
    x = np.arange(n)
    for i, m in enumerate(order):
        per_method = data[m].get(method, {})
        per_subject_vals = {}  # st -> {subject: val}
        for st in STIM_TYPES:
            vals_dict = per_method.get(st, {})
            if not vals_dict:
                continue
            v = np.array(list(vals_dict.values()))
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            draw_box(ax, x[i] + STIM_OFFSET[st], v.mean(), sem,
                     width=BOX_WIDTH, filled=is_mrsa, color=STIM_COLOR[st])
            per_subject_vals[st] = vals_dict

        # Per-subject dots + connecting lines.
        # Connect cstim<->vicco and vicco<->shared with light gray lines so the
        # eye can follow each subject across the three conditions.
        for s in subjects:
            xs, ys = [], []
            for st in STIM_TYPES:
                d = per_subject_vals.get(st, {})
                if s in d:
                    xs.append(x[i] + STIM_OFFSET[st])
                    ys.append(d[s])
            if len(xs) >= 2:
                ax.plot(xs, ys, color="#888888", linewidth=0.4, alpha=0.4, zorder=2)
            mk = "o" if is_mrsa else "D"
            for st in STIM_TYPES:
                d = per_subject_vals.get(st, {})
                if s in d:
                    ax.scatter(x[i] + STIM_OFFSET[st], d[s], s=8,
                               facecolors=STIM_COLOR[st] if is_mrsa else "none",
                               edgecolors=STIM_COLOR[st],
                               linewidths=0.5, marker=mk, zorder=6, alpha=0.9)

    # ---- Axis cosmetics ----
    ax.set_xticks(x)
    if show_xticks:
        ax.set_xticklabels([SHORT_NAMES.get(m, MODEL_DISPLAY.get(m, m))
                            for m in order],
                           rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    # ---- Spread-ratio header strip ----
    spread = lookup_spread(perm_df, model_set, method)
    if spread is not None:
        col = "#9A4500" if spread >= 1.0 else "#666666"
        if is_largest_amp and method == "mRSA":
            label = f"★ spread ratio: {spread:.2f}×"
            font_size = FONT["annotation"] + 1
        else:
            label = f"spread ratio: {spread:.2f}×"
            font_size = FONT["annotation"]
        ax.text(0.5, 1.005, label,
                transform=ax.transAxes, fontsize=font_size,
                fontweight="bold", color=col, ha="center", va="bottom",
                zorder=10)

    # ---- Off-panel NC indicator (fRSA only) ----
    if nc_drawn:
        nc_map = {kind: (val, col) for kind, val, col in nc_drawn}
        nc_b_val, _ = nc_map.get("base",  (np.nan, COLOR_BASELINE))
        nc_c_val, _ = nc_map.get("cstim", (np.nan, COLOR_CSTIM))
        ax.text(0.985, 0.965, "NC (off-panel)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT["small"] - 1, color="#555", fontweight="bold")
        ax.text(0.985, 0.875, f"base  {nc_b_val:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT["small"] - 1, color=COLOR_BASELINE)
        ax.text(0.985, 0.795, f"cstim {nc_c_val:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT["small"] - 1, color=COLOR_CSTIM)

    if panel_label is not None:
        add_panel_label(ax, panel_label, x=-0.04, y=1.05)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_CSTIM, markeredgecolor=COLOR_CSTIM,
                   markersize=4, label="Controversial"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_BASELINE, markeredgecolor=COLOR_BASELINE,
                   markersize=4, label="Baseline (vicco)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_SHARED, markeredgecolor=COLOR_SHARED,
                   markersize=4, label="DeepVision shared"),
            Line2D([0], [0], color=COLOR_CSTIM, linewidth=0.9, linestyle="-",
                   label="NC (cstim)"),
            Line2D([0], [0], color=COLOR_BASELINE, linewidth=0.9, linestyle="--",
                   label="NC (base)"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True,
                  framealpha=0.92, edgecolor="none", fontsize=FONT["small"],
                  ncol=2, columnspacing=0.8, handletextpad=0.4, handlelength=1.4)


def main():
    df = load_scores()
    nc_df = load_nc()
    bs_nc_df = load_bs_nc()
    perm_df = load_perm()

    ratios = []
    for ms in PANEL_ORDER:
        n_models = len([m for m in MODEL_SETS[ms]])
        ratios.append(max(n_models, 4))

    fig = plt.figure(figsize=(W_DOUBLE, 9.0))
    gs = fig.add_gridspec(
        2, len(PANEL_ORDER),
        width_ratios=ratios,
        wspace=0.06, hspace=0.18,
        left=0.05, right=0.99, top=0.94, bottom=0.16,
    )

    # Y-limits anchored on data (and NC for mRSA, since they overlap).
    # Now data also includes deepvision_shared, which can be higher than vicco.
    def row_ylim(method):
        data_max = df[df["method"] == method]["score"].max()
        if method == "mRSA":
            nc_max = float(np.sqrt(nc_df["noise_ceiling_spearman"].clip(0).max()))
            upper = max(data_max, nc_max) * 1.10
        else:
            upper = data_max * 1.10
        return 0, upper

    ylims = {"mRSA": row_ylim("mRSA"), "fRSA": row_ylim("fRSA")}

    mrsa_ratios = {}
    for ms in PANEL_ORDER:
        r = lookup_spread(perm_df, ms, "mRSA")
        if r is not None:
            mrsa_ratios[ms] = r
    largest_ms = max(mrsa_ratios, key=mrsa_ratios.get) if mrsa_ratios else None

    panel_letters = list("abcde")
    for r, method in enumerate(["mRSA", "fRSA"]):
        for c, ms in enumerate(PANEL_ORDER):
            ax = fig.add_subplot(gs[r, c])
            ax.set_ylim(*ylims[method])
            plot_method_panel(
                ax, df, nc_df, bs_nc_df, perm_df, ms, method,
                show_xticks=(r == 1),
                panel_label=panel_letters[c] if r == 0 else None,
                show_legend=(r == 0 and c == 0),
                is_largest_amp=(ms == largest_ms),
            )
            if c == 0:
                ax.set_ylabel(("Mixed RSA " if method == "mRSA"
                               else "Fixed RSA ") + r"($r_s$)")
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            if r == 0:
                ax.set_title(TITLE[ms], y=1.08)

    out_pdf = FIGURES_DIR / "brain_alignment_improved_with_shared.pdf"
    out_png = FIGURES_DIR / "brain_alignment_improved_with_shared.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
