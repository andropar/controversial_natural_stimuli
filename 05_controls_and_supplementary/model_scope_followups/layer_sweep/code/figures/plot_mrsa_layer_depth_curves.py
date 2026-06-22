#!/usr/bin/env python3
"""Plot dense-layer mRSA curves with layer-selection markers."""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cstims.constants import MODEL_DISPLAY_NAMES, MODEL_SETS
from cstims.paper.style_improved import apply_style, DPI, FONT
from layers_config import MAIN_LAYER, get_layer_set


apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "results"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"
PNG_DIR = FIG_DIR / "png"

LAYER_SCORES_CSV = DATA_DIR / "mrsa_dense_all_eval_layer_scores.csv"
CURVE_SUMMARY_CSV = DATA_DIR / "mrsa_layer_depth_curve_summary.csv"
SMOOTH_FRACTION = 0.05

CSTIM_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
CSTIM_LABELS = {
    "all_models": "All Models",
    "sota": "State of the Art",
    "training_objective": "Training Objective",
    "architecture": "Architecture",
    "dataset": "Dataset",
}
LINE_COLORS = {
    "shared": "#555555",
    "vicco": "#0072B2",
    "cstim": "#D55E00",
}
MARKER_COLORS = {
    "paper_layer": "#333333",
    "best_on_shared": "#2A9D8F",
    "best_on_cstim": "#E76F51",
}


def model_label(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def layer_maps():
    specs = get_layer_set("dense")
    idx = {}
    frac = {}
    for model, layers in specs.items():
        names = [name for name, _ in layers]
        denom = max(len(names) - 1, 1)
        idx[model] = {name: i for i, name in enumerate(names)}
        frac[model] = {name: i / denom for i, name in enumerate(names)}
    return specs, idx, frac


def build_curve_summary(force: bool = False) -> pd.DataFrame:
    specs, idx, frac = layer_maps()
    if CURVE_SUMMARY_CSV.exists() and not force:
        return pd.read_csv(CURVE_SUMMARY_CSV)

    usecols = ["model", "layer", "eval_target", "model_set", "mrsa"]
    scores = pd.read_csv(LAYER_SCORES_CSV, usecols=usecols)
    summary = (
        scores.groupby(["model", "layer", "eval_target", "model_set"], as_index=False)["mrsa"]
        .mean()
    )
    summary["layer_index"] = [
        idx[m][layer] for m, layer in zip(summary["model"], summary["layer"])
    ]
    summary["layer_frac"] = [
        frac[m][layer] for m, layer in zip(summary["model"], summary["layer"])
    ]
    summary["display_name"] = summary["model"].map(MODEL_DISPLAY_NAMES).fillna(summary["model"])
    summary.to_csv(CURVE_SUMMARY_CSV, index=False)
    return summary


def get_curve(summary: pd.DataFrame, model: str, target: str, model_set: str) -> pd.DataFrame:
    return summary[
        summary["model"].eq(model)
        & summary["eval_target"].eq(target)
        & summary["model_set"].eq(model_set)
    ].sort_values("layer_index")


def smooth_curve(curve: pd.DataFrame) -> pd.DataFrame:
    curve = curve.copy()
    if len(curve) >= 5:
        window = max(3, int(round(len(curve) * SMOOTH_FRACTION)))
        if window % 2 == 0:
            window += 1
        curve["mrsa_plot"] = (
            curve["mrsa"]
            .rolling(window, center=True, min_periods=1)
            .mean()
        )
    else:
        curve["mrsa_plot"] = curve["mrsa"]
    return curve


def best_frac(curve: pd.DataFrame) -> float:
    if curve.empty:
        return np.nan
    value_col = "mrsa_plot" if "mrsa_plot" in curve.columns else "mrsa"
    return float(curve.loc[curve[value_col].idxmax(), "layer_frac"])


def plot_curve(ax, curve: pd.DataFrame, *, color: str, label: str) -> None:
    ax.plot(
        curve["layer_frac"].to_numpy(dtype=float),
        curve["mrsa_plot"].to_numpy(dtype=float),
        color=color,
        linewidth=1.0,
        label=label,
    )


def plot_for_set(summary: pd.DataFrame, cstim_set: str) -> None:
    specs = get_layer_set("dense")
    model_order = [m for m in MODEL_SETS[cstim_set] if m in specs]
    n_models = len(model_order)
    ncols = min(4, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.8 * ncols, 1.75 * nrows + 0.6),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, model in zip(axes, model_order):
        shared = get_curve(summary, model, "shared", "deepvision_shared")
        vicco = get_curve(summary, model, "vicco", "vicco")
        cstim = get_curve(summary, model, "cstim", cstim_set)
        shared_plot = smooth_curve(shared)
        vicco_plot = smooth_curve(vicco)
        cstim_plot = smooth_curve(cstim)

        plot_curve(ax, shared_plot, color=LINE_COLORS["shared"], label="Shared")
        plot_curve(ax, vicco_plot, color=LINE_COLORS["vicco"], label="Vicco")
        plot_curve(ax, cstim_plot, color=LINE_COLORS["cstim"], label="Cstim")

        paper_frac = summary[
            summary["model"].eq(model)
            & summary["layer"].eq(MAIN_LAYER[model])
        ]["layer_frac"].iloc[0]
        shared_frac = best_frac(shared_plot)
        cstim_frac = best_frac(cstim_plot)
        ax.axvline(paper_frac, color=MARKER_COLORS["paper_layer"], linewidth=0.7,
                   linestyle="--", alpha=0.8, label="Paper layer")
        ax.axvline(shared_frac, color=MARKER_COLORS["best_on_shared"], linewidth=0.8,
                   linestyle=":", alpha=0.95, label="Best shared")
        ax.axvline(cstim_frac, color=MARKER_COLORS["best_on_cstim"], linewidth=0.8,
                   linestyle=":", alpha=0.95, label="Best cstim")

        ax.set_title(model_label(model), fontsize=FONT["small"])
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.08, 0.80)
        ax.grid(axis="y", color="0.92", linewidth=0.4)

    for ax in axes[n_models:]:
        ax.set_visible(False)
    for ax in axes[max(0, (nrows - 1) * ncols):]:
        if ax.get_visible():
            ax.set_xlabel("layer depth", fontsize=FONT["axis_label"])
    for i, ax in enumerate(axes):
        if ax.get_visible() and i % ncols == 0:
            ax.set_ylabel("mRSA", fontsize=FONT["axis_label"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:6], labels[:6], loc="upper center", ncol=6,
               frameon=False, fontsize=FONT["legend"], bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Dense-layer mRSA curves: {CSTIM_LABELS[cstim_set]}",
                 fontsize=FONT["title"], y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_pdf = FIG_DIR / f"mrsa_layer_depth_curves_{cstim_set}.pdf"
    out_png = PNG_DIR / f"mrsa_layer_depth_curves_{cstim_set}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    # Rebuild because layer ordering lives in layers_config.py and can change
    # without the raw score table changing.
    summary = build_curve_summary(force=True)
    for cstim_set in CSTIM_SETS:
        plot_for_set(summary, cstim_set)
    print(f"Wrote {CURVE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
