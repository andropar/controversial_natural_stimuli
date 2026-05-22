#!/usr/bin/env python3
"""Brain-alignment-style mRSA plot: paper layer vs best-on-shared layer.

This mirrors the mRSA row of 01_brain_model_alignment's
brain_alignment_improved.pdf, but the two rows juxtapose:

    1. paper_layer
    2. best_on_shared, selected on DeepVision shared images

Within each row/panel, orange is cstim and blue is Vicco baseline.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT, SOURCE_PAPER_ROOT

sys.path.insert(0, str(SOURCE_PAPER_ROOT / "figures"))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from config import MODEL_DISPLAY_NAMES, MODEL_SETS, STATS_DATA_DIR, SUBJECTS
from style_improved import (
    COLOR_BASELINE,
    COLOR_CSTIM,
    DPI,
    FONT,
    W_DOUBLE,
    add_panel_label,
    apply_style,
)


apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "results"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"
PNG_DIR = FIG_DIR / "png"

TRANSFER_CSV = DATA_DIR / "mrsa_dense_layer_selection_transfer.csv"
SUMMARY_CSV = DATA_DIR / "mrsa_brain_alignment_paper_vs_shared_summary.csv"
SUMMARY_NC_CSV = DATA_DIR / "mrsa_brain_alignment_paper_vs_shared_nc_normalized_summary.csv"

PANEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
PANEL_TITLE = {
    "all_models": "All Models",
    "sota": "State of the Art",
    "training_objective": "Training Objective",
    "architecture": "Architecture",
    "dataset": "Dataset",
}
ROW_RULES = [
    ("paper_layer", "paper_layer", "Paper layer"),
    ("best_on_shared", "deepvision_shared", "Best on shared layer"),
]
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


def sem(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return 0.0
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def draw_box(ax, x, vals, *, width=0.30, color="#D55E00") -> None:
    vals = np.asarray(vals, dtype=float)
    mean = float(vals.mean())
    err = sem(vals)
    lo, hi = mean - err, mean + err
    if hi - lo < 1e-4:
        lo, hi = mean - 0.0015, mean + 0.0015
    rect = mpatches.Rectangle(
        (x - width / 2, lo),
        width,
        hi - lo,
        facecolor=color,
        edgecolor=color,
        linewidth=0.6,
        alpha=0.75,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.hlines(mean, x - width / 2, x + width / 2, colors="white",
              linewidth=0.7, zorder=5)


def median_pairwise_diff(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan
    return float(np.median([abs(a - b) for a, b in itertools.combinations(vals, 2)]))


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(TRANSFER_CSV)
    keep = df[
        df["selection_rule"].isin(["paper_layer", "best_on_shared"])
        & df["eval_target"].isin(["cstim", "vicco"])
    ].copy()
    keep = keep[
        keep["selection_model_set"].isin(["paper_layer", "deepvision_shared"])
    ]
    return keep


def panel_data(df: pd.DataFrame, model_set: str, rule: str, selection_model_set: str) -> pd.DataFrame:
    models = set(MODEL_SETS[model_set])
    cstim = df[
        df["selection_rule"].eq(rule)
        & df["selection_model_set"].eq(selection_model_set)
        & df["eval_target"].eq("cstim")
        & df["eval_model_set"].eq(model_set)
        & df["model"].isin(models)
    ][["subject", "model", "display_name", "mrsa_mean"]].rename(columns={"mrsa_mean": "cstim"})
    vicco = df[
        df["selection_rule"].eq(rule)
        & df["selection_model_set"].eq(selection_model_set)
        & df["eval_target"].eq("vicco")
        & df["eval_model_set"].eq("vicco")
        & df["model"].isin(models)
    ][["subject", "model", "display_name", "mrsa_mean"]].rename(columns={"mrsa_mean": "vicco"})
    out = cstim.merge(vicco, on=["subject", "model", "display_name"],
                      how="inner", validate="one_to_one")
    return out


def model_order_from_panel(panel: pd.DataFrame) -> list[str]:
    order = (
        panel.groupby("model")["cstim"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    return order


def load_noise_ceilings() -> pd.DataFrame:
    return pd.read_csv(STATS_DATA_DIR / "rdm_noise_ceilings.csv")


def noise_ceiling_maps(nc: pd.DataFrame, model_set: str) -> tuple[dict[str, float], dict[str, float]]:
    ctrl = (
        nc[(nc["group"].eq(model_set)) & nc["stimulus_type"].eq("controversial")]
        .groupby("subject")["noise_ceiling_spearman"]
        .mean()
    )
    base = (
        nc[nc["group"].eq("vicco")]
        .groupby("subject")["noise_ceiling_spearman"]
        .mean()
    )
    cstim_nc = np.sqrt(ctrl.clip(lower=0)).to_dict()
    vicco_nc = np.sqrt(base.clip(lower=0)).to_dict()
    return cstim_nc, vicco_nc


def normalize_panel(panel: pd.DataFrame, nc: pd.DataFrame, model_set: str) -> pd.DataFrame:
    cstim_nc, vicco_nc = noise_ceiling_maps(nc, model_set)
    out = panel.copy()
    out["cstim"] = [
        val / cstim_nc.get(sub, np.nan)
        for sub, val in zip(out["subject"], out["cstim"])
    ]
    out["vicco"] = [
        val / vicco_nc.get(sub, np.nan)
        for sub, val in zip(out["subject"], out["vicco"])
    ]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["cstim", "vicco"])


def add_noise_ceiling(ax, nc: pd.DataFrame, model_set: str) -> None:
    ctrl = (
        nc[(nc["group"].eq(model_set)) & nc["stimulus_type"].eq("controversial")]
        .groupby("subject")["noise_ceiling_spearman"]
        .mean()
    )
    base = (
        nc[nc["group"].eq("vicco")]
        .groupby("subject")["noise_ceiling_spearman"]
        .mean()
    )
    for vals, color, linestyle in [
        (np.sqrt(ctrl.values), COLOR_CSTIM, "-"),
        (np.sqrt(base.values), COLOR_BASELINE, "--"),
    ]:
        if len(vals) == 0:
            continue
        mean = float(np.nanmean(vals))
        err = sem(vals)
        ax.axhspan(mean - err, mean + err, color=color, alpha=0.10,
                   zorder=0, linewidth=0)
        ax.axhline(mean, color=color, linewidth=0.9, alpha=0.7,
                   linestyle=linestyle, zorder=0)


def spread_ratio(panel: pd.DataFrame) -> float:
    model_means = panel.groupby("model", as_index=False).agg(
        cstim=("cstim", "mean"),
        vicco=("vicco", "mean"),
    )
    cstim_spread = median_pairwise_diff(model_means["cstim"].to_numpy())
    vicco_spread = median_pairwise_diff(model_means["vicco"].to_numpy())
    return cstim_spread / vicco_spread if vicco_spread > 0 else np.nan


def collect_summary(df: pd.DataFrame, nc: pd.DataFrame, *, normalized: bool = False) -> pd.DataFrame:
    rows = []
    for model_set in PANEL_ORDER:
        for rule, selection_model_set, row_label in ROW_RULES:
            panel = panel_data(df, model_set, rule, selection_model_set)
            if normalized:
                panel = normalize_panel(panel, nc, model_set)
            model_means = panel.groupby("model", as_index=False).agg(
                cstim=("cstim", "mean"),
                vicco=("vicco", "mean"),
            )
            rows.append({
                "model_set": model_set,
                "selection_rule": rule,
                "selection_label": row_label,
                "n_models": model_means["model"].nunique(),
                "mean_cstim": float(model_means["cstim"].mean()),
                "mean_vicco": float(model_means["vicco"].mean()),
                "mean_drop": float((model_means["cstim"] - model_means["vicco"]).mean()),
                "spread_ratio": spread_ratio(panel),
            })
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_NC_CSV if normalized else SUMMARY_CSV, index=False)
    return out


def plot_panel(ax, panel: pd.DataFrame, model_order: list[str], nc: pd.DataFrame,
               model_set: str, *, normalized: bool, show_legend: bool) -> None:
    if normalized:
        ax.axhline(1.0, color="#555555", linewidth=0.7, linestyle="-", zorder=0)
    else:
        add_noise_ceiling(ax, nc, model_set)

    subjects = sorted(panel["subject"].unique())
    x = np.arange(len(model_order))
    offset = 0.20
    for i, model in enumerate(model_order):
        block = panel[panel["model"].eq(model)]
        c = block.set_index("subject")["cstim"].to_dict()
        b = block.set_index("subject")["vicco"].to_dict()
        if c:
            draw_box(ax, x[i] - offset, list(c.values()), color=COLOR_CSTIM)
        if b:
            draw_box(ax, x[i] + offset, list(b.values()), color=COLOR_BASELINE)
        for sub in subjects:
            cv = c.get(sub)
            bv = b.get(sub)
            if cv is not None and bv is not None:
                ax.plot([x[i] - offset, x[i] + offset], [cv, bv],
                        color="#888888", linewidth=0.4, alpha=0.4, zorder=2)
            if cv is not None:
                ax.scatter(x[i] - offset, cv, s=8, facecolors=COLOR_CSTIM,
                           edgecolors=COLOR_CSTIM, linewidths=0.5, zorder=6,
                           alpha=0.9)
            if bv is not None:
                ax.scatter(x[i] + offset, bv, s=8, facecolors=COLOR_BASELINE,
                           edgecolors=COLOR_BASELINE, linewidths=0.5, zorder=6,
                           alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [SHORT_NAMES.get(m, MODEL_DISPLAY_NAMES.get(m, m)) for m in model_order],
        rotation=45,
        ha="right",
    )

    ratio = spread_ratio(panel)
    if np.isfinite(ratio):
        color = "#9A4500" if ratio >= 1 else "#666666"
        ax.text(0.5, 1.005, f"spread ratio: {ratio:.2f}x",
                transform=ax.transAxes, fontsize=FONT["annotation"],
                fontweight="bold", color=color, ha="center", va="bottom")

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_CSTIM, markeredgecolor=COLOR_CSTIM,
                   markersize=4, label="Controversial"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_BASELINE, markeredgecolor=COLOR_BASELINE,
                   markersize=4, label="Baseline"),
        ]
        if normalized:
            handles.append(Line2D([0], [0], color="#555555", linewidth=0.7,
                                  linestyle="-", label="Noise ceiling"))
        else:
            handles.extend([
                Line2D([0], [0], color=COLOR_CSTIM, linewidth=0.9,
                       linestyle="-", label="Noise ceiling (cstim)"),
                Line2D([0], [0], color=COLOR_BASELINE, linewidth=0.9,
                       linestyle="--", label="Noise ceiling (base)"),
            ])
        ax.legend(handles=handles, loc="upper right", frameon=True,
                  framealpha=0.92, edgecolor="none", fontsize=FONT["small"],
                  ncol=2, columnspacing=0.8, handletextpad=0.4, handlelength=1.4)


def render(df: pd.DataFrame, nc: pd.DataFrame, *, normalized: bool) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    collect_summary(df, nc, normalized=normalized)
    ratios = [max(len(MODEL_SETS[ms]), 4) for ms in PANEL_ORDER]

    fig = plt.figure(figsize=(W_DOUBLE, 11.2))
    gs = fig.add_gridspec(
        2, len(PANEL_ORDER),
        width_ratios=ratios,
        wspace=0.06,
        hspace=0.58,
        left=0.05,
        right=0.99,
        top=0.91,
        bottom=0.14,
    )

    if normalized:
        panel_max = 0.0
        for rule, selection_model_set, _ in ROW_RULES:
            for model_set in PANEL_ORDER:
                p = panel_data(df, model_set, rule, selection_model_set)
                p = normalize_panel(p, nc, model_set)
                panel_max = max(panel_max, float(np.nanmax(p[["cstim", "vicco"]].to_numpy())))
        ymax = max(1.05, panel_max * 1.08)
    else:
        ymax = max(
            df["mrsa_mean"].max(),
            np.sqrt(nc["noise_ceiling_spearman"].clip(0).max()),
        ) * 1.08

    for row_i, (rule, selection_model_set, row_label) in enumerate(ROW_RULES):
        for col_i, model_set in enumerate(PANEL_ORDER):
            ax = fig.add_subplot(gs[row_i, col_i])
            panel = panel_data(df, model_set, rule, selection_model_set)
            if normalized:
                panel = normalize_panel(panel, nc, model_set)
            order = model_order_from_panel(panel)
            ax.set_ylim(0, ymax)
            plot_panel(
                ax,
                panel,
                order,
                nc,
                model_set,
                normalized=normalized,
                show_legend=(row_i == 0 and col_i == 0),
            )
            if col_i == 0:
                ylabel = (
                    f"{row_label}\nNC-normalized mRSA"
                    if normalized
                    else f"{row_label}\nMixed RSA ($r_s$)"
                )
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            if row_i == 0:
                ax.set_title(PANEL_TITLE[model_set], y=1.08)
                add_panel_label(ax, chr(ord("a") + col_i), x=-0.04, y=1.05)

    suffix = "_nc_normalized" if normalized else ""
    out_pdf = FIG_DIR / f"mrsa_brain_alignment_paper_vs_shared{suffix}.pdf"
    out_png = PNG_DIR / f"mrsa_brain_alignment_paper_vs_shared{suffix}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Wrote {SUMMARY_NC_CSV if normalized else SUMMARY_CSV}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    plt.close(fig)


def main() -> None:
    df = load_scores()
    nc = load_noise_ceilings()
    render(df, nc, normalized=False)
    render(df, nc, normalized=True)


if __name__ == "__main__":
    main()
