#!/usr/bin/env python3
"""
Improved brain alignment figure.

Fixes vs. original:
- Okabe-Ito palette (vermillion / blue) — colour-blind safe.
- Noise-ceiling bands use the same controversial/baseline colour encoding as
  the score distributions.
- Per-panel spread-ratio text removed; spread is shown in a companion summary.
- Panel labels (a..e) for the five model sets.
- Mixed and Fixed RSA share the same colour, distinguished by marker
  fill (mixed = filled, fixed = open) and line style — so a single legend
  works.
- Mixed and fixed RSA rows share the same y-limit and both show noise ceiling
  bands in-panel.
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
    apply_style, FONT, DPI, W_DOUBLE,
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
PNG_DIR = FIGURES_DIR / "png"
SPREAD_STATS_PATH = STATS_DATA_DIR / "spread_statistics.csv"

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
    "training_objective": "Train. Objective",
    "sota":               "State of the Art",
    "architecture":       "Architecture",
    "dataset":            "Dataset",
    "all_models":         "All Models",
}

# Order of panels left → right
PANEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
NC_GRAY = "#9A9A9A"


def load_scores():
    dfs = []
    score_files = {
        "crsa_scores.csv":          ("crsa", "fRSA"),
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
    paths = [
        STATS_DATA_DIR / "rdm_noise_ceilings.csv",
        SHARE_ROOT / "02_alignment_reliability" / "results" / "rdm_noise_ceilings.csv",
    ]
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"Could not find rdm_noise_ceilings.csv in {paths}")


def load_bs_nc():
    paths = [
        STATS_DATA_DIR / "between_subject_noise_ceilings.csv",
        SHARE_ROOT / "02_alignment_reliability" / "results" / "between_subject_noise_ceilings.csv",
    ]
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"Could not find between_subject_noise_ceilings.csv in {paths}")


def load_perm():
    p = STATS_DATA_DIR / "permutation_test_results.csv"
    return pd.read_csv(p) if p.exists() else None


def get_per_subject_scores(df, model_set):
    models = MODEL_SETS[model_set]
    sub = df[df["model_set"] == model_set]
    subjects = sorted(sub["subject"].unique())
    out = {}
    for m in models:
        out[m] = {}
        for method in ("mRSA", "fRSA"):
            block = sub[(sub["model"] == m) & (sub["method"] == method)]
            for st in ("controversial", "vicco"):
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


def draw_box(ax, x, mean, sem, width=0.30, filled=True, color="#D55E00"):
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

    # ---- Noise ceiling band: neutral gray ----
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

    nc_ctrl_mean = float(np.nanmean(nc_ctrl_v)) if len(nc_ctrl_v) else np.nan
    nc_base_mean = float(np.nanmean(nc_base_v)) if len(nc_base_v) else np.nan

    # Noise ceiling: draw both rows in-panel with the same colour/line encoding.
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

    # ---- Per-model boxes + per-subject dots ----
    n = len(order)
    x = np.arange(n)
    offset = 0.20
    cstim_panel_values = []
    base_panel_values = []
    for i, m in enumerate(order):
        c = data[m][method].get("controversial", {})
        b = data[m][method].get("vicco", {})
        if c:
            v = np.array(list(c.values()))
            cstim_panel_values.extend(v)
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            draw_box(ax, x[i] - offset, v.mean(), sem,
                     width=0.30, filled=is_mrsa, color=COLOR_CSTIM)
        if b:
            v = np.array(list(b.values()))
            base_panel_values.extend(v)
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            draw_box(ax, x[i] + offset, v.mean(), sem,
                     width=0.30, filled=is_mrsa, color=COLOR_BASELINE)

        for s in subjects:
            cv, bv = c.get(s), b.get(s)
            if cv is not None and bv is not None:
                ax.plot([x[i] - offset, x[i] + offset], [cv, bv],
                        color="#888888", linewidth=0.4, alpha=0.4, zorder=2)
            mk = "o" if is_mrsa else "D"
            if cv is not None:
                ax.scatter(x[i] - offset, cv, s=8,
                           facecolors=COLOR_CSTIM if is_mrsa else "none",
                           edgecolors=COLOR_CSTIM,
                           linewidths=0.5, marker=mk, zorder=6,
                           alpha=0.9)
            if bv is not None:
                ax.scatter(x[i] + offset, bv, s=8,
                           facecolors=COLOR_BASELINE if is_mrsa else "none",
                           edgecolors=COLOR_BASELINE,
                           linewidths=0.5, marker=mk, zorder=6,
                           alpha=0.9)

    def draw_score_interval(vals, xpos, color):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return
        lo, mid, hi = np.percentile(vals, [2.5, 50.0, 97.5])
        cap = 0.055
        ax.vlines(xpos, lo, hi, color=color, linewidth=0.8, alpha=0.95, zorder=7)
        ax.hlines([lo, hi], xpos - cap, xpos + cap, color=color,
                  linewidth=0.8, alpha=0.95, zorder=7)
        ax.hlines(mid, xpos - cap * 1.25, xpos + cap * 1.25, color=color,
                  linewidth=1.1, alpha=0.95, zorder=8)

    draw_score_interval(cstim_panel_values, n - 0.02, COLOR_CSTIM)
    draw_score_interval(base_panel_values, n + 0.16, COLOR_BASELINE)

    # ---- Axis cosmetics ----
    ax.set_xticks(x)
    ax.set_xlim(-0.65, n + 0.32)
    if show_xticks:
        ax.set_xticklabels([SHORT_NAMES.get(m, MODEL_DISPLAY.get(m, m))
                            for m in order],
                           rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    if panel_label is not None:
        ax.text(
            0.015, 0.98, panel_label,
            transform=ax.transAxes,
            fontsize=FONT["panel_label"],
            fontweight="bold",
            ha="left", va="top",
            zorder=10,
        )

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_CSTIM, markeredgecolor=COLOR_CSTIM,
                   markersize=4, label="Controversial"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_BASELINE, markeredgecolor=COLOR_BASELINE,
                   markersize=4, label="Baseline"),
            Line2D([0], [0], color=COLOR_CSTIM, linewidth=0.9,
                   linestyle="-",  label="Noise ceiling (cstim)"),
            Line2D([0], [0], color=COLOR_BASELINE, linewidth=0.9,
                   linestyle="--", label="Noise ceiling (base)"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True,
                  framealpha=0.92, edgecolor="none", fontsize=FONT["small"],
                  ncol=2, columnspacing=0.8, handletextpad=0.4, handlelength=1.4)


def plot_median_spread_summary():
    """Two-panel companion: absolute median model-score spread."""
    if not SPREAD_STATS_PATH.exists():
        print(f"[WARN] Missing {SPREAD_STATS_PATH}; skipping median-spread summary")
        return

    spread_df = pd.read_csv(SPREAD_STATS_PATH)
    method_specs = [
        ("crsa", "fRSA"),
        ("wrsa_transfer", "mRSA"),
    ]
    labels = {
        "all_models": "All",
        "sota": "SOTA",
        "training_objective": "Train.\nObj.",
        "architecture": "Arch.",
        "dataset": "Dataset",
    }

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 4.4), sharey=True)
    x = np.arange(len(PANEL_ORDER), dtype=float)
    width = 0.34
    err_kw = dict(ecolor="0.30", elinewidth=0.8, capsize=2)

    for ax, (method, title) in zip(axes, method_specs):
        baseline_means, baseline_sems = [], []
        cstim_means, cstim_sems = [], []
        for model_set in PANEL_ORDER:
            rows = spread_df[
                (spread_df["method"] == method)
                & (spread_df["model_set"] == model_set)
            ]
            base_vals = rows["vicco_median_pairwise_diff_mean"].astype(float)
            cstim_vals = rows["cstim_median_pairwise_diff"].astype(float)
            baseline_means.append(base_vals.mean())
            baseline_sems.append(base_vals.sem())
            cstim_means.append(cstim_vals.mean())
            cstim_sems.append(cstim_vals.sem())

        ax.bar(
            x - width / 2,
            baseline_means,
            width,
            yerr=baseline_sems,
            color=COLOR_BASELINE,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
            label="Baseline",
            error_kw=err_kw,
            zorder=3,
        )
        ax.bar(
            x + width / 2,
            cstim_means,
            width,
            yerr=cstim_sems,
            color=COLOR_CSTIM,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
            label="Controversial",
            error_kw=err_kw,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[ms] for ms in PANEL_ORDER])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Median pairwise score difference")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.92,
                   edgecolor="none", fontsize=FONT["small"])

    out_pdf = FIGURES_DIR / "median_spread_by_model_set.pdf"
    out_png = PNG_DIR / "median_spread_by_model_set.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


def main():
    df = load_scores()
    nc_df = load_nc()
    bs_nc_df = load_bs_nc()
    perm_df = load_perm()

    # Width ratios proportional to model count
    ratios = []
    for ms in PANEL_ORDER:
        n_models = len([m for m in MODEL_SETS[ms]])
        ratios.append(max(n_models, 4))

    fig = plt.figure(figsize=(W_DOUBLE, 10.6))
    gs = fig.add_gridspec(
        2, len(PANEL_ORDER),
        width_ratios=ratios,
        wspace=0.06, hspace=0.14,
        left=0.05, right=0.99, top=0.94, bottom=0.14,
    )

    # Shared y-limits across rows; include both mRSA and fRSA noise ceilings.
    data_max = float(df["score"].max())
    mrsa_nc_max = float(np.sqrt(nc_df["noise_ceiling_spearman"].clip(0).max()))
    frsa_nc_max = float(bs_nc_df["nc_mid"].max())
    shared_ylim = (0, max(data_max, mrsa_nc_max, frsa_nc_max) * 1.10)

    panel_letters = list("abcde")
    for r, method in enumerate(["mRSA", "fRSA"]):
        for c, ms in enumerate(PANEL_ORDER):
            ax = fig.add_subplot(gs[r, c])
            ax.set_ylim(*shared_ylim)
            plot_method_panel(
                ax, df, nc_df, bs_nc_df, perm_df, ms, method,
                show_xticks=(r == 1),
                panel_label=panel_letters[c] if r == 0 else None,
                show_legend=(r == 0 and c == 0),
            )
            if c == 0:
                ax.set_ylabel(("Mixed RSA " if method == "mRSA"
                               else "Fixed RSA ") + r"($r_s$)")
            else:
                ax.set_ylabel("")
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", left=False, labelleft=False)
            if r == 0:
                ax.set_title(TITLE[ms], y=1.03)

    out_pdf = FIGURES_DIR / "brain_alignment_improved.pdf"
    out_png = PNG_DIR / "brain_alignment_improved.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)
    plot_median_spread_summary()


if __name__ == "__main__":
    main()
