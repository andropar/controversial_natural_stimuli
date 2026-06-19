#!/usr/bin/env python3
"""Brain-alignment-style plot using best-on-shared layer-sweep scores.

This mirrors brain_alignment_improved.pdf, but evaluates both mRSA and fRSA at
the layer selected on DeepVision shared images in the dense mRSA layer sweep.
"""
from __future__ import annotations

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from cstims import constants, paths
from cstims.paper.style_improved import (
    COLOR_BASELINE,
    COLOR_CSTIM,
    DPI,
    FONT,
    W_DOUBLE,
    apply_style,
)


apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "results"
FIGURES_DIR = LAYER_SWEEP_ROOT / "figures"
PNG_DIR = FIGURES_DIR / "png"
MRSA_TRANSFER_CSV = DATA_DIR / "mrsa_dense_layer_selection_transfer.csv"
FRSA_TRANSFER_CSV = DATA_DIR / "frsa_best_shared_layer_transfer.csv"

MODEL_SETS = constants.MODEL_SETS
MODEL_DISPLAY = constants.MODEL_DISPLAY_NAMES
STATS_DATA_DIR = paths.stats_data_dir()

PANEL_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
METHOD_ORDER = ["mRSA", "fRSA"]
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
NC_BAND_ALPHA = 0.032
NC_LINE_ALPHA = 0.36
PAIR_LINE_ALPHA = 0.18
PAIR_LINE_WIDTH = 0.28
DELTA_SUMMARY_OFFSET = 0.54
RANGE_SUMMARY_OFFSET = 1.12
SUMMARY_END_PAD = 0.36
SPREAD_RANGE_GAP = 0.26
SPREAD_CAP_WIDTH = 0.11
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260528


def sem(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return 0.0
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def bootstrap_ci(vals: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(vals, size=(BOOTSTRAP_N, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return mean, float(lo), float(hi)


def draw_box(
    ax,
    x: float,
    vals: np.ndarray,
    *,
    color: str,
    filled: bool,
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
        facecolor=color if filled else "white",
        edgecolor=color,
        linewidth=0.6,
        alpha=0.75 if filled else 0.95,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.hlines(
        mean,
        x - width / 2,
        x + width / 2,
        colors="white" if filled else color,
        linewidth=0.7,
        zorder=5,
    )


def load_mrsa_scores() -> pd.DataFrame:
    df = pd.read_csv(MRSA_TRANSFER_CSV)
    out = df[
        df["selection_rule"].eq("best_on_shared")
        & df["selection_model_set"].eq("deepvision_shared")
        & df["eval_target"].isin(["cstim", "vicco"])
    ].copy()
    out["score"] = out["mrsa_mean"].astype(float)
    out["method"] = "mRSA"
    return out


def load_frsa_scores() -> pd.DataFrame:
    if not FRSA_TRANSFER_CSV.exists():
        raise FileNotFoundError(
            f"Missing {FRSA_TRANSFER_CSV}. Run "
            "code/analysis/15_compute_frsa_best_shared_layer_transfer.py first."
        )
    df = pd.read_csv(FRSA_TRANSFER_CSV)
    out = df[
        df["selection_rule"].eq("best_on_shared")
        & df["selection_model_set"].eq("deepvision_shared")
        & df["eval_target"].isin(["cstim", "vicco"])
    ].copy()
    out["score"] = out["frsa_mean"].astype(float)
    out["method"] = "fRSA"
    return out


def load_scores() -> pd.DataFrame:
    cols = [
        "subject",
        "model",
        "display_name",
        "selected_layer",
        "selected_layer_frac",
        "eval_target",
        "eval_model_set",
        "score",
        "method",
    ]
    return pd.concat([load_mrsa_scores()[cols], load_frsa_scores()[cols]], ignore_index=True)


def load_noise_ceilings() -> pd.DataFrame:
    candidate_paths = [
        STATS_DATA_DIR / "rdm_noise_ceilings.csv",
        paths.reliability_data_dir() / "rdm_noise_ceilings.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find rdm_noise_ceilings.csv in {candidate_paths}")


def load_between_subject_noise_ceilings() -> pd.DataFrame:
    candidate_paths = [
        STATS_DATA_DIR / "between_subject_noise_ceilings.csv",
        paths.reliability_data_dir() / "between_subject_noise_ceilings.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Could not find between_subject_noise_ceilings.csv in {candidate_paths}"
    )


def eval_model_set(condition: str, model_set: str) -> str:
    if condition == "cstim":
        return model_set
    if condition == "vicco":
        return "vicco"
    raise ValueError(condition)


def panel_data(df: pd.DataFrame, model_set: str, method: str) -> pd.DataFrame:
    keys = ["subject", "model", "display_name"]
    models = set(MODEL_SETS[model_set])
    out = None
    method_df = df[df["method"].eq(method)]
    for condition in ("cstim", "vicco"):
        block = method_df[
            method_df["eval_target"].eq(condition)
            & method_df["eval_model_set"].eq(eval_model_set(condition, model_set))
            & method_df["model"].isin(models)
        ][keys + ["score"]].rename(columns={"score": condition})
        if out is None:
            out = block
        else:
            out = out.merge(block, on=keys, how="inner", validate="one_to_one")
    if out is None:
        return pd.DataFrame(columns=[*keys, "cstim", "vicco"])
    return out


def model_order(mrsa_panel: pd.DataFrame) -> list[str]:
    return (
        mrsa_panel.groupby("model")["cstim"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )


def panel_model_mean_values(panel: pd.DataFrame, condition: str) -> np.ndarray:
    values = panel.groupby("model")[condition].mean().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def panel_subject_delta_values(panel: pd.DataFrame) -> np.ndarray:
    deltas = panel.assign(delta=panel["vicco"] - panel["cstim"])
    values = deltas.groupby("subject")["delta"].mean().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def draw_delta_summary(ax, panel: pd.DataFrame, xpos: float) -> None:
    vals = panel_subject_delta_values(panel)
    mean, lo, hi = bootstrap_ci(vals)
    if not np.isfinite(mean):
        return
    ax.vlines(xpos, lo, hi, colors="0.12", linewidth=0.9, zorder=7)
    ax.hlines([lo, hi], xpos - 0.07, xpos + 0.07, colors="0.12", linewidth=0.75, zorder=7)
    ax.scatter(xpos, mean, s=14, color="0.12", zorder=8)
    ax.hlines(0, xpos - 0.10, xpos + 0.10, colors="0.65", linewidth=0.55, zorder=6)


def draw_spread_ranges(ax, panel: pd.DataFrame, x_center: float) -> float:
    """Draw min-max ranges of model-level boxplot means."""
    specs = [
        ("cstim", COLOR_CSTIM, x_center - SPREAD_RANGE_GAP / 2),
        ("vicco", COLOR_BASELINE, x_center + SPREAD_RANGE_GAP / 2),
    ]
    for condition, color, xpos in specs:
        vals = panel_model_mean_values(panel, condition)
        if len(vals) == 0:
            continue
        lo = float(vals.min())
        hi = float(vals.max())
        mid = float(np.median(vals))
        ax.vlines(
            xpos,
            lo,
            hi,
            colors=color,
            linewidth=1.0,
            zorder=4,
        )
        ax.hlines(
            [lo, mid, hi],
            xpos - SPREAD_CAP_WIDTH / 2,
            xpos + SPREAD_CAP_WIDTH / 2,
            colors=color,
            linewidth=[0.75, 1.0, 0.75],
            zorder=6,
        )
    return x_center


def add_mrsa_noise_ceiling(ax, nc: pd.DataFrame, model_set: str) -> None:
    ctrl = (
        nc[nc["group"].eq(model_set) & nc["stimulus_type"].eq("controversial")]
        .groupby("subject")["noise_ceiling_spearman"]
        .mean()
    )
    base = nc[nc["group"].eq("vicco")].groupby("subject")["noise_ceiling_spearman"].mean()
    for vals, color, linestyle in [
        (np.sqrt(ctrl.clip(lower=0).values), COLOR_CSTIM, "-"),
        (np.sqrt(base.clip(lower=0).values), COLOR_BASELINE, "--"),
    ]:
        _add_band(ax, vals, color, linestyle)


def add_frsa_noise_ceiling(ax, bs_nc: pd.DataFrame, model_set: str) -> None:
    ctrl = bs_nc[bs_nc["group"].eq(model_set) & bs_nc["stimulus_type"].eq("controversial")]
    base = bs_nc[bs_nc["group"].eq("vicco")]
    ctrl_vals = ctrl["nc_mid"].values if len(ctrl) else base["nc_mid"].values
    base_vals = base["nc_mid"].values
    for vals, color, linestyle in [
        (ctrl_vals, COLOR_CSTIM, "-"),
        (base_vals, COLOR_BASELINE, "--"),
    ]:
        _add_band(ax, vals, color, linestyle)


def _add_band(ax, vals, color: str, linestyle: str) -> None:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
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


def add_noise_ceiling(ax, nc: pd.DataFrame, bs_nc: pd.DataFrame, model_set: str, method: str) -> None:
    if method == "mRSA":
        add_mrsa_noise_ceiling(ax, nc, model_set)
    else:
        add_frsa_noise_ceiling(ax, bs_nc, model_set)


def plot_panel(
    ax,
    panel: pd.DataFrame,
    order: list[str],
    nc: pd.DataFrame,
    bs_nc: pd.DataFrame,
    model_set: str,
    method: str,
    *,
    show_xticks: bool,
    panel_label: str | None = None,
    show_legend: bool = False,
) -> None:
    add_noise_ceiling(ax, nc, bs_nc, model_set, method)

    is_mrsa = method == "mRSA"
    marker = "o" if is_mrsa else "D"
    subjects = sorted(panel["subject"].unique())
    x = np.arange(len(order))
    offset = 0.20
    delta_x = len(order) + DELTA_SUMMARY_OFFSET
    range_center = len(order) + RANGE_SUMMARY_OFFSET
    draw_delta_summary(ax, panel, delta_x)
    draw_spread_ranges(ax, panel, range_center)
    for i, model in enumerate(order):
        block = panel[panel["model"].eq(model)]
        cstim = block.set_index("subject")["cstim"].to_dict()
        vicco = block.set_index("subject")["vicco"].to_dict()
        if cstim:
            draw_box(
                ax,
                x[i] - offset,
                np.array(list(cstim.values())),
                color=COLOR_CSTIM,
                filled=is_mrsa,
            )
        if vicco:
            draw_box(
                ax,
                x[i] + offset,
                np.array(list(vicco.values())),
                color=COLOR_BASELINE,
                filled=is_mrsa,
            )

        for subject in subjects:
            c_val = cstim.get(subject)
            b_val = vicco.get(subject)
            if c_val is not None and b_val is not None:
                ax.plot(
                    [x[i] - offset, x[i] + offset],
                    [c_val, b_val],
                    color="#888888",
                    linewidth=PAIR_LINE_WIDTH,
                    alpha=PAIR_LINE_ALPHA,
                    zorder=2,
                )
            if c_val is not None:
                ax.scatter(
                    x[i] - offset,
                    c_val,
                    s=8,
                    facecolors=COLOR_CSTIM if is_mrsa else "none",
                    edgecolors=COLOR_CSTIM,
                    linewidths=0.5,
                    marker=marker,
                    zorder=6,
                    alpha=0.9,
                )
            if b_val is not None:
                ax.scatter(
                    x[i] + offset,
                    b_val,
                    s=8,
                    facecolors=COLOR_BASELINE if is_mrsa else "none",
                    edgecolors=COLOR_BASELINE,
                    linewidths=0.5,
                    marker=marker,
                    zorder=6,
                    alpha=0.9,
                )

    ax.set_xlim(-0.65, range_center + SPREAD_RANGE_GAP / 2 + SUMMARY_END_PAD)
    if show_xticks:
        xticks = [*x, delta_x, range_center]
        labels = [SHORT_NAMES.get(m, MODEL_DISPLAY.get(m, m)) for m in order] + [
            r"$\Delta$",
            "range",
        ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
        )
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


def row_noise_upper_values(nc: pd.DataFrame, bs_nc: pd.DataFrame, method: str) -> list[float]:
    values = []
    if method == "mRSA":
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
    else:
        base = bs_nc[bs_nc["group"].eq("vicco")]
        for model_set in PANEL_ORDER:
            ctrl = bs_nc[
                bs_nc["group"].eq(model_set)
                & bs_nc["stimulus_type"].eq("controversial")
            ]
            for vals in [
                ctrl["nc_mid"].to_numpy(dtype=float),
                base["nc_mid"].to_numpy(dtype=float),
            ]:
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    values.append(float(vals.mean()) + sem(vals))
    return values


def row_y_limits(df: pd.DataFrame, nc: pd.DataFrame, bs_nc: pd.DataFrame, method: str) -> tuple[float, float]:
    vals = []
    for model_set in PANEL_ORDER:
        panel = panel_data(df, model_set, method)
        if not panel.empty:
            vals.extend(panel[["cstim", "vicco"]].to_numpy(dtype=float).ravel())
    vals.extend(row_noise_upper_values(nc, bs_nc, method))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (-0.03, 1.0)
    headroom = 1.025 if method == "mRSA" else 1.035
    return (-0.04, float(vals.max()) * headroom)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scores()
    nc = load_noise_ceilings()
    bs_nc = load_between_subject_noise_ceilings()
    ratios = [
        len(MODEL_SETS[model_set]) + (2.8 if model_set == "all_models" else 1.9)
        for model_set in PANEL_ORDER
    ]
    ylims = {method: row_y_limits(df, nc, bs_nc, method) for method in METHOD_ORDER}

    fig = plt.figure(figsize=(W_DOUBLE, 10.1))
    gs = fig.add_gridspec(
        len(METHOD_ORDER),
        len(PANEL_ORDER),
        width_ratios=ratios,
        wspace=0.06,
        hspace=0.15,
        left=0.05,
        right=0.99,
        top=0.93,
        bottom=0.14,
    )

    orders = {
        model_set: model_order(panel_data(df, model_set, "mRSA"))
        for model_set in PANEL_ORDER
    }
    panel_letters = list("abcde")
    for row, method in enumerate(METHOD_ORDER):
        for col, model_set in enumerate(PANEL_ORDER):
            ax = fig.add_subplot(gs[row, col])
            panel = panel_data(df, model_set, method)
            ax.set_ylim(*ylims[method])
            plot_panel(
                ax,
                panel,
                orders[model_set],
                nc,
                bs_nc,
                model_set,
                method,
                show_xticks=(row == len(METHOD_ORDER) - 1),
                panel_label=panel_letters[col] if row == 0 else None,
                show_legend=(row == 0 and col == 0),
            )
            if col == 0:
                ax.set_ylabel(("Mixed RSA " if method == "mRSA" else "Fixed RSA ") + r"($r_s$)")
            else:
                ax.set_ylabel("")
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", left=False, labelleft=False)
            if row == 0:
                ax.set_title(TITLE[model_set], y=0.995, fontsize=FONT["small"], fontweight="bold")

    out_pdf = FIGURES_DIR / "brain_alignment_best_shared_layer_improved.pdf"
    out_png = PNG_DIR / "brain_alignment_best_shared_layer_improved.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
