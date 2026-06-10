#!/usr/bin/env python3
"""Brain-alignment-style CSTIM/Vicco comparison with all subject points."""

from __future__ import annotations

import argparse

import _paths  # noqa: F401
from _paths import FIGURES_DIR, PNG_DIR, RESULTS_DIR, SHARE_ROOT

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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
STATS_DATA_DIR = config.STATS_DATA_DIR
PANEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
TITLE = {
    "training_objective": "Train. Objective",
    "sota": "State of the Art",
    "architecture": "Architecture",
    "dataset": "Dataset",
    "all_models": "All Models",
}
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
ROW_SPECS = [
    ("original", "Original best-shared", "original_cstim", "original_vicco"),
    ("refit", "CSTIM-weighted refit", "refit_cstim_loso", "refit_vicco"),
]
PAIR_LINE_ALPHA = 0.18
PAIR_LINE_WIDTH = 0.28
NC_BAND_ALPHA = 0.032
NC_LINE_ALPHA = 0.36
DELTA_SUMMARY_OFFSET = 0.54
RANGE_SUMMARY_OFFSET = 1.12
SUMMARY_END_PAD = 0.36
SPREAD_RANGE_GAP = 0.26
SPREAD_CAP_WIDTH = 0.11
POINT_SIZE = 11


def sem(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return 0.0
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def load_scores(weight: float) -> pd.DataFrame:
    df = pd.read_csv(SCORE_CSV)
    if "eval_target" not in df.columns:
        raise ValueError(
            f"{SCORE_CSV} does not contain eval_target. Re-run the weighted LOSO analysis."
        )
    df["cstim_weight"] = df["cstim_weight"].astype(float)
    df = df[np.isclose(df["cstim_weight"], weight)].copy()
    keys = ["subject", "model", "display_name", "selected_layer", "model_set"]
    cstim = df[df["eval_target"].eq("cstim_loso")][
        keys + ["mrsa_loso", "original_best_shared_mrsa"]
    ].rename(
        columns={
            "mrsa_loso": "refit_cstim_loso",
            "original_best_shared_mrsa": "original_cstim",
        }
    )
    vicco = df[df["eval_target"].eq("vicco")][
        keys + ["mrsa_loso", "original_best_shared_mrsa"]
    ].rename(
        columns={
            "mrsa_loso": "refit_vicco",
            "original_best_shared_mrsa": "original_vicco",
        }
    )
    wide = cstim.merge(vicco, on=keys, how="inner", validate="one_to_one")
    if wide.empty:
        raise RuntimeError(f"No matched CSTIM/Vicco rows for cstim_weight={weight:g}")
    return wide


def load_noise_ceilings() -> pd.DataFrame:
    paths = [
        STATS_DATA_DIR / "rdm_noise_ceilings.csv",
        SHARE_ROOT / "02_alignment_reliability" / "results" / "rdm_noise_ceilings.csv",
    ]
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find rdm_noise_ceilings.csv in {paths}")


def add_noise_ceiling(ax, nc: pd.DataFrame, model_set: str) -> None:
    ctrl = (
        nc[nc["group"].eq(model_set) & nc["stimulus_type"].eq("controversial")]
        .groupby("subject")["noise_ceiling_spearman"]
        .mean()
    )
    base = nc[nc["group"].eq("vicco")].groupby("subject")["noise_ceiling_spearman"].mean()
    for vals, color, linestyle in [
        (np.sqrt(ctrl.clip(lower=0).to_numpy(dtype=float)), COLOR_CSTIM, "-"),
        (np.sqrt(base.clip(lower=0).to_numpy(dtype=float)), COLOR_BASELINE, "--"),
    ]:
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        mean = float(vals.mean())
        err = sem(vals)
        ax.axhspan(
            mean - err,
            mean + err,
            color=color,
            alpha=NC_BAND_ALPHA,
            zorder=0,
            linewidth=0,
        )
        ax.axhline(
            mean,
            color=color,
            linewidth=0.85,
            alpha=NC_LINE_ALPHA,
            linestyle=linestyle,
            zorder=0,
        )


def draw_box(
    ax,
    x: float,
    vals: np.ndarray,
    *,
    color: str,
    width: float = 0.30,
) -> None:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
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
    ax.hlines(mean, x - width / 2, x + width / 2, colors="white", linewidth=0.7, zorder=5)


def panel_data(df: pd.DataFrame, model_set: str) -> pd.DataFrame:
    models = set(config.MODEL_SETS[model_set])
    return df[df["model_set"].eq(model_set) & df["model"].isin(models)].copy()


def model_order(panel: pd.DataFrame) -> list[str]:
    return (
        panel.groupby("model")["original_cstim"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )


def panel_model_mean_values(panel: pd.DataFrame, condition: str) -> np.ndarray:
    values = panel.groupby("model")[condition].mean().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def draw_delta_summary(ax, panel: pd.DataFrame, c_col: str, v_col: str, xpos: float) -> None:
    vals = (
        panel.assign(delta=panel[v_col] - panel[c_col])
        .groupby("subject")["delta"]
        .mean()
        .to_numpy(dtype=float)
    )
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    mean = float(vals.mean())
    err = sem(vals)
    ax.vlines(xpos, mean - err, mean + err, colors="0.12", linewidth=0.9, zorder=7)
    ax.hlines(
        [mean - err, mean + err],
        xpos - 0.07,
        xpos + 0.07,
        colors="0.12",
        linewidth=0.75,
        zorder=7,
    )
    ax.scatter(xpos, mean, s=16, color="0.12", zorder=8)
    ax.hlines(0, xpos - 0.10, xpos + 0.10, colors="0.65", linewidth=0.55, zorder=6)


def draw_spread_ranges(ax, panel: pd.DataFrame, c_col: str, v_col: str, x_center: float) -> None:
    for condition, color, xpos in [
        (c_col, COLOR_CSTIM, x_center - SPREAD_RANGE_GAP / 2),
        (v_col, COLOR_BASELINE, x_center + SPREAD_RANGE_GAP / 2),
    ]:
        vals = panel_model_mean_values(panel, condition)
        if len(vals) == 0:
            continue
        lo = float(vals.min())
        hi = float(vals.max())
        mid = float(np.median(vals))
        ax.vlines(xpos, lo, hi, colors=color, linewidth=1.0, zorder=4)
        ax.hlines(
            [lo, mid, hi],
            xpos - SPREAD_CAP_WIDTH / 2,
            xpos + SPREAD_CAP_WIDTH / 2,
            colors=color,
            linewidth=[0.75, 1.0, 0.75],
            zorder=6,
        )


def plot_panel(
    ax,
    panel: pd.DataFrame,
    order: list[str],
    nc: pd.DataFrame,
    model_set: str,
    row_key: str,
    c_col: str,
    v_col: str,
    *,
    show_xticks: bool,
    panel_label: str | None = None,
    show_legend: bool = False,
) -> None:
    add_noise_ceiling(ax, nc, model_set)

    subjects = sorted(panel["subject"].unique())
    x = np.arange(len(order))
    offset = 0.20
    delta_x = len(order) + DELTA_SUMMARY_OFFSET
    range_center = len(order) + RANGE_SUMMARY_OFFSET
    draw_delta_summary(ax, panel, c_col, v_col, delta_x)
    draw_spread_ranges(ax, panel, c_col, v_col, range_center)

    for i, model in enumerate(order):
        block = panel[panel["model"].eq(model)]
        cstim = block.set_index("subject")[c_col].to_dict()
        vicco = block.set_index("subject")[v_col].to_dict()
        draw_box(ax, x[i] - offset, np.array(list(cstim.values())), color=COLOR_CSTIM)
        draw_box(ax, x[i] + offset, np.array(list(vicco.values())), color=COLOR_BASELINE)

        for subject in subjects:
            c_val = cstim.get(subject)
            b_val = vicco.get(subject)
            if c_val is not None and b_val is not None:
                ax.plot(
                    [x[i] - offset, x[i] + offset],
                    [c_val, b_val],
                    color="#777777",
                    linewidth=PAIR_LINE_WIDTH,
                    alpha=PAIR_LINE_ALPHA,
                    zorder=2,
                )
            if c_val is not None:
                ax.scatter(
                    x[i] - offset,
                    c_val,
                    s=POINT_SIZE,
                    facecolors=COLOR_CSTIM,
                    edgecolors="white",
                    linewidths=0.25,
                    marker="o",
                    zorder=6,
                    alpha=0.92,
                )
            if b_val is not None:
                ax.scatter(
                    x[i] + offset,
                    b_val,
                    s=POINT_SIZE,
                    facecolors=COLOR_BASELINE,
                    edgecolors="white",
                    linewidths=0.25,
                    marker="o",
                    zorder=6,
                    alpha=0.92,
                )

    ax.set_xlim(-0.65, range_center + SPREAD_RANGE_GAP / 2 + SUMMARY_END_PAD)
    if show_xticks:
        xticks = [*x, delta_x, range_center]
        labels = [SHORT_NAMES.get(m, config.MODEL_DISPLAY_NAMES.get(m, m)) for m in order] + [
            r"$\Delta$",
            "range",
        ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    if panel_label is not None:
        ax.text(
            0.015,
            1.125,
            panel_label,
            transform=ax.transAxes,
            fontsize=FONT["panel_label"],
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False,
            zorder=10,
        )

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLOR_CSTIM,
                markeredgecolor=COLOR_CSTIM,
                markersize=4,
                label="Controversial",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLOR_BASELINE,
                markeredgecolor=COLOR_BASELINE,
                markersize=4,
                label="Baseline",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            fontsize=FONT["small"],
            ncol=1,
            columnspacing=0.8,
            handletextpad=0.4,
            handlelength=1.4,
        )


def row_noise_upper_values(nc: pd.DataFrame) -> list[float]:
    values = []
    base = nc[nc["group"].eq("vicco")].groupby("subject")["noise_ceiling_spearman"].mean()
    for model_set in PANEL_ORDER:
        ctrl = (
            nc[nc["group"].eq(model_set) & nc["stimulus_type"].eq("controversial")]
            .groupby("subject")["noise_ceiling_spearman"]
            .mean()
        )
        for vals in [ctrl, base]:
            vals = np.sqrt(vals.clip(lower=0).to_numpy(dtype=float))
            vals = vals[np.isfinite(vals)]
            if len(vals):
                values.append(float(vals.mean()) + sem(vals))
    return values


def row_y_limits(df: pd.DataFrame, nc: pd.DataFrame, c_col: str, v_col: str) -> tuple[float, float]:
    vals = []
    for model_set in PANEL_ORDER:
        panel = panel_data(df, model_set)
        if not panel.empty:
            vals.extend(panel[[c_col, v_col]].to_numpy(dtype=float).ravel())
    vals.extend(row_noise_upper_values(nc))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (-0.03, 1.0)
    return (-0.04, float(vals.max()) * 1.025)


def write_summary(df: pd.DataFrame, weight: float) -> None:
    rows = []
    for model_set in PANEL_ORDER:
        panel = panel_data(df, model_set)
        for row_key, label, c_col, v_col in ROW_SPECS:
            rows.append(
                {
                    "model_set": model_set,
                    "protocol": row_key,
                    "label": label,
                    "cstim_weight": weight,
                    "n": len(panel),
                    "n_models": panel["model"].nunique(),
                    "n_subjects": panel["subject"].nunique(),
                    "mean_cstim": float(panel[c_col].mean()),
                    "mean_vicco": float(panel[v_col].mean()),
                    "mean_cstim_minus_vicco": float((panel[c_col] - panel[v_col]).mean()),
                }
            )
    out = RESULTS_DIR / f"cstim_loso_brain_alignment_style_allpoints_w{weight:g}_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=2.0)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scores(args.weight)
    nc = load_noise_ceilings()
    write_summary(df, args.weight)
    ratios = [
        len(panel_data(df, model_set)["model"].unique()) + (2.8 if model_set == "all_models" else 1.9)
        for model_set in PANEL_ORDER
    ]
    ylims = {
        row_key: row_y_limits(df, nc, c_col, v_col)
        for row_key, _label, c_col, v_col in ROW_SPECS
    }
    orders = {model_set: model_order(panel_data(df, model_set)) for model_set in PANEL_ORDER}

    fig = plt.figure(figsize=(W_DOUBLE, 10.1))
    gs = fig.add_gridspec(
        len(ROW_SPECS),
        len(PANEL_ORDER),
        width_ratios=ratios,
        wspace=0.06,
        hspace=0.15,
        left=0.05,
        right=0.99,
        top=0.93,
        bottom=0.14,
    )

    panel_letters = list("abcde")
    for row, (row_key, row_label, c_col, v_col) in enumerate(ROW_SPECS):
        for col, model_set in enumerate(PANEL_ORDER):
            ax = fig.add_subplot(gs[row, col])
            panel = panel_data(df, model_set)
            ax.set_ylim(*ylims[row_key])
            plot_panel(
                ax,
                panel,
                orders[model_set],
                nc,
                model_set,
                row_key,
                c_col,
                v_col,
                show_xticks=(row == len(ROW_SPECS) - 1),
                panel_label=panel_letters[col] if row == 0 else None,
                show_legend=(row == 0 and col == 0),
            )
            if col == 0:
                ax.set_ylabel(f"{row_label}\nMixed RSA ($r_s$)")
            else:
                ax.set_ylabel("")
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", left=False, labelleft=False)
            if row == 0:
                ax.set_title(TITLE[model_set], y=0.995, fontsize=FONT["small"], fontweight="bold")

    stem = f"cstim_loso_brain_alignment_style_allpoints_w{args.weight:g}_cached"
    out_pdf = FIGURES_DIR / f"{stem}.pdf"
    out_png = PNG_DIR / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
