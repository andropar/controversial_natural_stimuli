#!/usr/bin/env python3
"""Per-model mRSA comparison of shared-selected vs cstim-selected layers."""

from __future__ import annotations

import sys
from pathlib import Path

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

from cstims import paths
from cstims.constants import MODEL_DISPLAY_NAMES, MODEL_SETS
STATS_DATA_DIR = paths.stats_data_dir()
from cstims.paper.style_improved import (
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
SUMMARY_CSV = DATA_DIR / "mrsa_brain_alignment_best_shared_vs_cstim_layers_summary.csv"
SUMMARY_NC_CSV = DATA_DIR / "mrsa_brain_alignment_best_shared_vs_cstim_layers_nc_normalized_summary.csv"

PANEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
PANEL_TITLE = {
    "all_models": "All Models",
    "sota": "State of the Art",
    "training_objective": "Training Objective",
    "architecture": "Architecture",
    "dataset": "Dataset",
}
RULE_ORDER = ("best_on_shared", "best_on_cstim")
RULE_LABEL = {
    "best_on_shared": "Best on shared layer",
    "best_on_cstim": "Best on cstim layer",
}
CONDITION_ORDER = ("cstim", "vicco")
CONDITION_LABEL = {
    "cstim": "cstim",
    "vicco": "Vicco",
}
CONDITION_COLOR = {
    "cstim": COLOR_CSTIM,
    "vicco": COLOR_BASELINE,
}
RULE_BASE_OFFSET = {
    "best_on_shared": -0.20,
    "best_on_cstim": 0.20,
}
CONDITION_OFFSET = {
    "cstim": -0.075,
    "vicco": 0.075,
}
BOX_WIDTH = 0.12
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


def box_position(x: float, rule: str, condition: str) -> float:
    return x + RULE_BASE_OFFSET[rule] + CONDITION_OFFSET[condition]


def draw_box(ax, x, vals, *, color: str, rule: str) -> None:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    mean = float(vals.mean())
    err = sem(vals)
    lo, hi = mean - err, mean + err
    if hi - lo < 1e-4:
        lo, hi = mean - 0.0015, mean + 0.0015

    face_alpha = 0.78 if rule == "best_on_shared" else 0.30
    hatch = None if rule == "best_on_shared" else "////"
    rect = mpatches.Rectangle(
        (x - BOX_WIDTH / 2, lo),
        BOX_WIDTH,
        hi - lo,
        facecolor=mcolors.to_rgba(color, face_alpha),
        edgecolor=color,
        linewidth=0.7,
        hatch=hatch,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.hlines(mean, x - BOX_WIDTH / 2, x + BOX_WIDTH / 2, colors="white",
              linewidth=0.7, zorder=5)


def load_transfer() -> pd.DataFrame:
    return pd.read_csv(TRANSFER_CSV)


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
    return np.sqrt(ctrl.clip(lower=0)).to_dict(), np.sqrt(base.clip(lower=0)).to_dict()


def normalize_panel(panel: pd.DataFrame, nc: pd.DataFrame, model_set: str) -> pd.DataFrame:
    cstim_nc, vicco_nc = noise_ceiling_maps(nc, model_set)
    out = panel.copy()
    for rule in RULE_ORDER:
        out[f"{rule}_cstim"] = [
            val / cstim_nc.get(sub, np.nan)
            for sub, val in zip(out["subject"], out[f"{rule}_cstim"])
        ]
        out[f"{rule}_vicco"] = [
            val / vicco_nc.get(sub, np.nan)
            for sub, val in zip(out["subject"], out[f"{rule}_vicco"])
        ]
    value_cols = [f"{rule}_{condition}" for rule in RULE_ORDER for condition in CONDITION_ORDER]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=value_cols)


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


def selection_model_set(rule: str, model_set: str) -> str:
    if rule == "best_on_shared":
        return "deepvision_shared"
    if rule == "best_on_cstim":
        return model_set
    raise ValueError(rule)


def eval_model_set(condition: str, model_set: str) -> str:
    if condition == "cstim":
        return model_set
    if condition == "vicco":
        return "vicco"
    raise ValueError(condition)


def panel_data(df: pd.DataFrame, model_set: str) -> pd.DataFrame:
    models = set(MODEL_SETS[model_set])
    out = None
    keys = ["subject", "model", "display_name"]
    for rule in RULE_ORDER:
        for condition in CONDITION_ORDER:
            block = df[
                df["selection_rule"].eq(rule)
                & df["selection_model_set"].eq(selection_model_set(rule, model_set))
                & df["eval_target"].eq(condition)
                & df["eval_model_set"].eq(eval_model_set(condition, model_set))
                & df["model"].isin(models)
            ][keys + ["mrsa_mean"]].rename(
                columns={"mrsa_mean": f"{rule}_{condition}"}
            )
            if out is None:
                out = block
            else:
                out = out.merge(block, on=keys, how="inner", validate="one_to_one")
    if out is None:
        return pd.DataFrame(columns=keys)
    return out


def model_order_from_panel(panel: pd.DataFrame) -> list[str]:
    return (
        panel.groupby("model")["best_on_shared_cstim"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )


def collect_summary(
    df: pd.DataFrame,
    nc: pd.DataFrame,
    *,
    normalized: bool = False,
) -> pd.DataFrame:
    rows = []
    for model_set in PANEL_ORDER:
        panel = panel_data(df, model_set)
        if normalized:
            panel = normalize_panel(panel, nc, model_set)
        agg = {
            f"{rule}_{condition}": (f"{rule}_{condition}", "mean")
            for rule in RULE_ORDER
            for condition in CONDITION_ORDER
        }
        model_means = panel.groupby("model", as_index=False).agg(**agg)
        rows.append({
            "model_set": model_set,
            "n_models": model_means["model"].nunique(),
            "mean_cstim_best_on_shared": float(model_means["best_on_shared_cstim"].mean()),
            "mean_vicco_best_on_shared": float(model_means["best_on_shared_vicco"].mean()),
            "mean_cstim_best_on_cstim": float(model_means["best_on_cstim_cstim"].mean()),
            "mean_vicco_best_on_cstim": float(model_means["best_on_cstim_vicco"].mean()),
            "delta_cstim_best_cstim_minus_shared": float(
                (model_means["best_on_cstim_cstim"] - model_means["best_on_shared_cstim"]).mean()
            ),
            "delta_vicco_best_cstim_minus_shared": float(
                (model_means["best_on_cstim_vicco"] - model_means["best_on_shared_vicco"]).mean()
            ),
        })
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_NC_CSV if normalized else SUMMARY_CSV, index=False)
    return out


def plotted_max(df: pd.DataFrame, nc: pd.DataFrame, *, normalized: bool) -> float:
    panel_max = 0.0
    value_cols = [f"{rule}_{condition}" for rule in RULE_ORDER for condition in CONDITION_ORDER]
    for model_set in PANEL_ORDER:
        panel = panel_data(df, model_set)
        if normalized:
            panel = normalize_panel(panel, nc, model_set)
        vals = panel[value_cols].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            panel_max = max(panel_max, float(vals.max()))
    return panel_max


def plot_panel(
    ax,
    panel: pd.DataFrame,
    model_order: list[str],
    nc: pd.DataFrame,
    model_set: str,
    *,
    normalized: bool,
) -> None:
    if normalized:
        ax.axhline(1.0, color="#555555", linewidth=0.7, linestyle="-", zorder=0)
    else:
        add_noise_ceiling(ax, nc, model_set)

    subjects = sorted(panel["subject"].unique())
    x = np.arange(len(model_order))
    for i, model in enumerate(model_order):
        block = panel[panel["model"].eq(model)]
        by_col = {
            f"{rule}_{condition}": block.set_index("subject")[f"{rule}_{condition}"].to_dict()
            for rule in RULE_ORDER
            for condition in CONDITION_ORDER
        }
        for rule in RULE_ORDER:
            for condition in CONDITION_ORDER:
                values = by_col[f"{rule}_{condition}"]
                draw_box(
                    ax,
                    box_position(x[i], rule, condition),
                    list(values.values()),
                    color=CONDITION_COLOR[condition],
                    rule=rule,
                )
            for subject in subjects:
                xs, ys = [], []
                for condition in CONDITION_ORDER:
                    value = by_col[f"{rule}_{condition}"].get(subject)
                    if value is not None and np.isfinite(value):
                        xs.append(box_position(x[i], rule, condition))
                        ys.append(value)
                if len(xs) == 2:
                    ax.plot(xs, ys, color="#888888", linewidth=0.4,
                            alpha=0.45 if rule == "best_on_shared" else 0.30,
                            linestyle="-" if rule == "best_on_shared" else "--",
                            zorder=2)
        for subject in subjects:
            for rule in RULE_ORDER:
                for condition in CONDITION_ORDER:
                    value = by_col[f"{rule}_{condition}"].get(subject)
                    if value is None or not np.isfinite(value):
                        continue
                    if rule == "best_on_shared":
                        face = CONDITION_COLOR[condition]
                        alpha = 0.90
                    else:
                        face = "white"
                        alpha = 0.80
                    ax.scatter(
                        box_position(x[i], rule, condition),
                        value,
                        s=8,
                        facecolors=face,
                        edgecolors=CONDITION_COLOR[condition],
                        linewidths=0.55,
                        zorder=6,
                        alpha=alpha,
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [SHORT_NAMES.get(m, MODEL_DISPLAY_NAMES.get(m, m)) for m in model_order],
        rotation=45,
        ha="right",
    )


def legend_handles(*, normalized: bool) -> list:
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COLOR_CSTIM, markeredgecolor=COLOR_CSTIM,
               markersize=4, label="cstim"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COLOR_BASELINE, markeredgecolor=COLOR_BASELINE,
               markersize=4, label="Vicco"),
        mpatches.Patch(facecolor=mcolors.to_rgba("0.5", 0.60),
                       edgecolor="0.5", label="Best on shared layer"),
        mpatches.Patch(facecolor=mcolors.to_rgba("0.9", 0.30),
                       edgecolor="0.5", hatch="////", label="Best on cstim layer"),
    ]
    if normalized:
        handles.append(Line2D([0], [0], color="#555555", linewidth=0.7,
                              linestyle="-", label="NC = 1.0"))
    else:
        handles.extend([
            Line2D([0], [0], color=COLOR_CSTIM, linewidth=0.9,
                   linestyle="-", label="NC (cstim)"),
            Line2D([0], [0], color=COLOR_BASELINE, linewidth=0.9,
                   linestyle="--", label="NC (base)"),
        ])
    return handles


def render(df: pd.DataFrame, nc: pd.DataFrame, *, normalized: bool = False) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    collect_summary(df, nc, normalized=normalized)

    ratios = [max(len(MODEL_SETS[ms]), 4) for ms in PANEL_ORDER]
    fig = plt.figure(figsize=(W_DOUBLE, 5.8))
    gs = fig.add_gridspec(
        1,
        len(PANEL_ORDER),
        width_ratios=ratios,
        wspace=0.06,
        left=0.05,
        right=0.99,
        top=0.76,
        bottom=0.26,
    )

    panel_max = plotted_max(df, nc, normalized=normalized)
    if normalized:
        ymax = max(1.05, panel_max * 1.08)
    else:
        ymax = max(panel_max, np.sqrt(nc["noise_ceiling_spearman"].clip(0).max())) * 1.08

    for col_i, model_set in enumerate(PANEL_ORDER):
        ax = fig.add_subplot(gs[0, col_i])
        panel = panel_data(df, model_set)
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
        )
        if col_i == 0:
            ylabel = "NC-normalized mRSA" if normalized else "Mixed RSA ($r_s$)"
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(PANEL_TITLE[model_set], y=1.08)
        add_panel_label(ax, chr(ord("a") + col_i), x=-0.04, y=1.05)

    fig.legend(
        handles=legend_handles(normalized=normalized),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.98),
        frameon=True,
        framealpha=0.92,
        edgecolor="none",
        fontsize=FONT["small"],
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.4,
    )

    suffix = "_nc_normalized" if normalized else ""
    out_pdf = FIG_DIR / f"mrsa_brain_alignment_best_shared_vs_cstim_layers{suffix}.pdf"
    out_png = PNG_DIR / f"mrsa_brain_alignment_best_shared_vs_cstim_layers{suffix}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Wrote {SUMMARY_NC_CSV if normalized else SUMMARY_CSV}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    plt.close(fig)


def main() -> None:
    df = load_transfer()
    nc = load_noise_ceilings()
    render(df, nc, normalized=False)
    render(df, nc, normalized=True)


if __name__ == "__main__":
    main()
