#!/usr/bin/env python3
"""Plot 1:1 mixed-RSA comparisons for original vs CSTIM-weighted refits."""

from __future__ import annotations

import argparse
from pathlib import Path

import _paths  # noqa: F401
from _paths import FIGURES_DIR, PNG_DIR, RESULTS_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
from style_improved import (
    COLOR_BASELINE,
    COLOR_CSTIM,
    DPI,
    FONT,
    W_DOUBLE,
    apply_style,
)


apply_style()

SCORE_CSV = RESULTS_DIR / "cstim_loso_weighted_scores.csv"
PANEL_SPECS = [
    (
        "original_gap",
        "Original encoding",
        "original_vicco",
        "original_cstim",
        "Original Vicco mRSA",
        "Original CSTIM mRSA",
    ),
    (
        "refit_gap",
        "CSTIM-weighted encoding",
        "refit_vicco",
        "refit_cstim_loso",
        "Weighted Vicco mRSA",
        "Weighted CSTIM LOSO mRSA",
    ),
    (
        "cstim_shift",
        "CSTIM shift",
        "original_cstim",
        "refit_cstim_loso",
        "Original CSTIM mRSA",
        "Weighted CSTIM LOSO mRSA",
    ),
    (
        "vicco_shift",
        "Vicco shift",
        "original_vicco",
        "refit_vicco",
        "Original Vicco mRSA",
        "Weighted Vicco mRSA",
    ),
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
FONT_TINY = FONT.get("tiny", FONT.get("small", FONT["tick"]))


def load_wide(weight: float, model_set: str) -> pd.DataFrame:
    df = pd.read_csv(SCORE_CSV)
    if "eval_target" not in df.columns:
        raise ValueError(
            f"{SCORE_CSV} does not contain eval_target. Re-run "
            "code/analysis/01_compute_weighted_loso.py after the Vicco update."
        )
    df["cstim_weight"] = df["cstim_weight"].astype(float)
    endpoint = df[
        np.isclose(df["cstim_weight"], weight)
        & df["model_set"].eq(model_set)
    ].copy()
    keys = ["subject", "model", "display_name", "selected_layer", "model_set"]
    cstim = endpoint[endpoint["eval_target"].eq("cstim_loso")][
        keys + ["mrsa_loso", "original_best_shared_mrsa"]
    ].rename(
        columns={
            "mrsa_loso": "refit_cstim_loso",
            "original_best_shared_mrsa": "original_cstim",
        }
    )
    vicco = endpoint[endpoint["eval_target"].eq("vicco")][
        keys + ["mrsa_loso", "original_best_shared_mrsa"]
    ].rename(
        columns={
            "mrsa_loso": "refit_vicco",
            "original_best_shared_mrsa": "original_vicco",
        }
    )
    wide = cstim.merge(vicco, on=keys, how="inner", validate="one_to_one")
    if wide.empty:
        raise RuntimeError(f"No matched CSTIM/Vicco rows for {model_set}, w={weight:g}")
    wide = wide.sort_values(["model", "subject"]).reset_index(drop=True)
    out_csv = RESULTS_DIR / f"cstim_loso_condition_comparison_{model_set}_w{weight:g}.csv"
    wide.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return wide


def finite_limits(df: pd.DataFrame) -> tuple[float, float]:
    vals = []
    for _, _, xcol, ycol, _, _ in PANEL_SPECS:
        vals.extend(df[xcol].to_numpy(dtype=float))
        vals.extend(df[ycol].to_numpy(dtype=float))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    lo = float(vals.min())
    hi = float(vals.max())
    pad = max((hi - lo) * 0.07, 0.015)
    return lo - pad, hi + pad


def draw_panel(
    ax,
    df: pd.DataFrame,
    spec: tuple[str, str, str, str, str, str],
    *,
    label_models: bool,
) -> None:
    _name, title, xcol, ycol, xlabel, ylabel = spec
    points = df[[xcol, ycol]].to_numpy(dtype=float)
    finite = np.isfinite(points).all(axis=1)
    sub = df.loc[finite].copy()
    ax.scatter(
        sub[xcol],
        sub[ycol],
        s=16,
        color="0.35",
        alpha=0.22,
        linewidths=0,
        zorder=2,
        label="Subject-model",
    )
    means = (
        sub.groupby(["model", "display_name"], as_index=False)[[xcol, ycol]]
        .mean()
        .sort_values(ycol, ascending=False)
    )
    ax.scatter(
        means[xcol],
        means[ycol],
        s=42,
        facecolor=COLOR_CSTIM,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.92,
        zorder=4,
        label="Model mean",
    )
    if label_models:
        for row in means.itertuples(index=False):
            label = SHORT_NAMES.get(row.model, config.MODEL_DISPLAY_NAMES.get(row.model, row.model))
            ax.text(
                getattr(row, xcol),
                getattr(row, ycol),
                f" {label}",
                fontsize=FONT_TINY,
                color="0.18",
                alpha=0.76,
                ha="left",
                va="center",
                zorder=5,
            )
    if len(sub) > 2:
        rho = stats.spearmanr(sub[xcol], sub[ycol]).statistic
    else:
        rho = np.nan
    delta = float(np.nanmean(sub[ycol] - sub[xcol]))
    ax.text(
        0.03,
        0.97,
        f"rho={rho:.2f}\nmean delta={delta:+.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_TINY,
        color="0.25",
    )
    ax.set_title(title, fontsize=FONT["small"], fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22, linewidth=0.45)


def plot_comparison(
    df: pd.DataFrame,
    *,
    model_set: str,
    weight: float,
    label_models: bool,
) -> None:
    lo, hi = finite_limits(df)
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE * 0.72, 7.4), constrained_layout=True)
    for ax, spec in zip(axes.ravel(), PANEL_SPECS):
        draw_panel(ax, df, spec, label_models=label_models)
        ax.plot([lo, hi], [lo, hi], color="0.55", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        f"All-model mixed RSA comparisons, CSTIM weight {weight:g}",
        fontsize=FONT["title"],
        fontweight="bold",
    )
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=FONT["small"],
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_labeled" if label_models else ""
    stem = f"cstim_vicco_original_refit_1to1_{model_set}_w{weight:g}{suffix}_cached"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = PNG_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    print(f"Saved {pdf}")
    print(f"Saved {png}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--model-set", default="all_models")
    parser.add_argument("--label-models", action="store_true")
    args = parser.parse_args()

    wide = load_wide(args.weight, args.model_set)
    plot_comparison(
        wide,
        model_set=args.model_set,
        weight=args.weight,
        label_models=args.label_models,
    )


if __name__ == "__main__":
    main()
