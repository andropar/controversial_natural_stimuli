#!/usr/bin/env python3
"""Plot the standalone 100k-pool selected-vs-random diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch


SCRIPT = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT.parent
RESULTS_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"
SHARE_ROOT = ANALYSIS_DIR.parents[2]

sys.path.insert(0, str(SHARE_ROOT / "src"))

from cstims.paper.style_improved import (  # noqa: E402
    COLOR_BASELINE,
    COLOR_CSTIM,
    DPI,
    FONT,
    MODEL_SET_DISPLAY_SHORT,
    MODEL_SET_ORDER,
    W_DOUBLE,
    W_1_5COL,
    add_panel_label,
    apply_style,
)


TRACKS = ["raw", "sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
ENCODING_TRACKS = TRACKS[1:]
TRACK_LABELS = {
    "raw": "Raw",
    "sub-01": "S01",
    "sub-03": "S03",
    "sub-05": "S05",
    "sub-06": "S06",
    "sub-07": "S07",
}
SUBSET_ORDER = ["random", "selected"]
SUBSET_LABELS = {"random": "Random", "selected": "Selected"}
SUBSET_COLORS = {"random": COLOR_BASELINE, "selected": COLOR_CSTIM}
MATRIX_LABELS = {"clean": "Clean RDMs", "noisy_by_clean": "Noisy-by-clean"}


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = pd.read_csv(RESULTS_DIR / "model_components.csv")
    recovery = pd.read_csv(RESULTS_DIR / "model_recovery.csv")
    track_summary = pd.read_csv(RESULTS_DIR / "track_summary.csv")
    return components, recovery, track_summary


def combined_components(components: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_set, subset_type, matrix_type), group in components.groupby(
        ["model_set", "subset_type", "matrix_type"], sort=False
    ):
        pivot = group.pivot_table(
            index="model",
            columns="track",
            values=["self_corr", "mean_other_corr", "margin"],
            aggfunc="first",
        )
        needed = [("margin", "raw"), *[("margin", track) for track in ENCODING_TRACKS]]
        common = pivot.dropna(subset=needed)
        if common.empty:
            continue

        combined_by_metric = {}
        for metric in ["self_corr", "mean_other_corr", "margin"]:
            encoding_mean = sum(common[(metric, track)] for track in ENCODING_TRACKS) / len(
                ENCODING_TRACKS
            )
            combined_by_metric[metric] = 0.5 * common[(metric, "raw")] + 0.5 * encoding_mean

        rec_mean = np.nan
        rec_min = np.nan
        if matrix_type == "noisy_by_clean":
            rec_sub = recovery[
                (recovery["model_set"] == model_set)
                & (recovery["subset_type"] == subset_type)
            ]
            rec_pivot = rec_sub.pivot_table(
                index="model", columns="track", values="recovery_accuracy", aggfunc="first"
            )
            rec_common = rec_pivot.dropna(subset=TRACKS)
            if not rec_common.empty:
                rec_combined = 0.5 * rec_common["raw"] + 0.5 * rec_common[ENCODING_TRACKS].mean(
                    axis=1
                )
                rec_mean = float(rec_combined.mean())
                rec_min = float(rec_combined.min())

        rows.append(
            {
                "model_set": model_set,
                "subset_type": subset_type,
                "matrix_type": matrix_type,
                "track": "combined_raw_plus_encoding",
                "n_models": int(len(common)),
                "self_corr_mean": float(combined_by_metric["self_corr"].mean()),
                "self_corr_min": float(combined_by_metric["self_corr"].min()),
                "mean_other_corr_mean": float(combined_by_metric["mean_other_corr"].mean()),
                "mean_other_corr_min": float(combined_by_metric["mean_other_corr"].min()),
                "margin_mean": float(combined_by_metric["margin"].mean()),
                "margin_min": float(combined_by_metric["margin"].min()),
                "recovery_accuracy_mean": rec_mean,
                "recovery_accuracy_min": rec_min,
            }
        )
    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "combined_summary.csv"
    out.to_csv(out_path, index=False)
    return out


def ordered_model_sets(df: pd.DataFrame) -> list[str]:
    present = set(df["model_set"])
    return [model_set for model_set in MODEL_SET_ORDER if model_set in present]


def plot_grouped_bars(
    ax,
    df: pd.DataFrame,
    matrix_type: str,
    value_col: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    model_sets = ordered_model_sets(df)
    sub = df[df["matrix_type"] == matrix_type]
    x = np.arange(len(model_sets))
    width = 0.34

    for offset, subset_type in [(-width / 2, "random"), (width / 2, "selected")]:
        vals = []
        for model_set in model_sets:
            row = sub[(sub["model_set"] == model_set) & (sub["subset_type"] == subset_type)]
            vals.append(float(row[value_col].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=SUBSET_COLORS[subset_type],
            edgecolor="white",
            linewidth=0.6,
            label=SUBSET_LABELS[subset_type],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SET_DISPLAY_SHORT[m] for m in model_sets], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.grid(axis="y", alpha=0.45)
    if ylim is not None:
        ax.set_ylim(*ylim)


def make_objective_figure(combined: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 4.1), constrained_layout=False)

    plot_grouped_bars(
        axes[0],
        combined,
        "clean",
        "margin_min",
        "Worst-model margin",
        "Clean combined objective",
        ylim=(0, 0.85),
    )
    plot_grouped_bars(
        axes[1],
        combined,
        "noisy_by_clean",
        "margin_min",
        "Worst-model margin",
        "Noise-aware combined objective",
        ylim=(0, 0.45),
    )
    plot_grouped_bars(
        axes[2],
        combined,
        "noisy_by_clean",
        "recovery_accuracy_mean",
        "Model recovery accuracy",
        "Noisy-by-clean recovery",
        ylim=(0.88, 1.01),
    )

    for label, ax in zip("abc", axes):
        add_panel_label(ax, label, x=-0.08, y=1.06)

    fig.legend(
        handles=[
            Patch(facecolor=SUBSET_COLORS["random"], label=SUBSET_LABELS["random"]),
            Patch(facecolor=SUBSET_COLORS["selected"], label=SUBSET_LABELS["selected"]),
        ],
        loc="lower center",
        bbox_to_anchor=(0.53, 0.02),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("100k-pool selected-vs-random objective diagnostics", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.30, wspace=0.34)
    save_figure(fig, "pool100k_combined_objective")


def make_absolute_self_other_figure(combined: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 4.3), constrained_layout=False)

    plot_grouped_bars(
        axes[0],
        combined,
        "noisy_by_clean",
        "self_corr_mean",
        "Self correlation",
        "Absolute self-correlation",
        ylim=(0, 0.58),
    )
    plot_grouped_bars(
        axes[1],
        combined,
        "noisy_by_clean",
        "mean_other_corr_mean",
        "Mean other correlation",
        "Absolute other-correlation",
        ylim=(0, 0.34),
    )

    for label, ax in zip("ab", axes):
        add_panel_label(ax, label, x=-0.10, y=1.06)

    fig.legend(
        handles=[
            Patch(facecolor=SUBSET_COLORS["random"], label=SUBSET_LABELS["random"]),
            Patch(facecolor=SUBSET_COLORS["selected"], label=SUBSET_LABELS["selected"]),
        ],
        loc="lower center",
        bbox_to_anchor=(0.53, 0.02),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Absolute 100k-pool noisy-by-clean correlations", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.30, wspace=0.26)
    save_figure(fig, "pool100k_absolute_self_other")


def pivot_delta(
    track_summary: pd.DataFrame,
    matrix_type: str,
    value_col: str,
) -> pd.DataFrame:
    sub = track_summary[track_summary["matrix_type"] == matrix_type]
    rows = []
    for (model_set, track), group in sub.groupby(["model_set", "track"], sort=False):
        vals = dict(zip(group["subset_type"], group[value_col]))
        if {"selected", "random"} <= set(vals):
            rows.append(
                {
                    "model_set": model_set,
                    "track": track,
                    "delta": float(vals["selected"] - vals["random"]),
                }
            )
    df = pd.DataFrame(rows)
    return df.pivot(index="track", columns="model_set", values="delta").reindex(
        index=TRACKS, columns=ordered_model_sets(track_summary)
    )


def draw_heatmap(ax, data: pd.DataFrame, title: str, cmap: str, center_zero: bool) -> None:
    values = data.to_numpy(dtype=float)
    if center_zero:
        vmax = np.nanmax(np.abs(values))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        norm = None
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title, pad=4)
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels([MODEL_SET_DISPLAY_SHORT[m] for m in data.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels([TRACK_LABELS[t] for t in data.index])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=FONT["small"])
    return im


def make_breakdown_figure(track_summary: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(W_1_5COL, 8.2), constrained_layout=False)
    axes = axes.ravel()

    specs = [
        ("margin_min", "Delta worst-model margin", "RdBu_r", True),
        ("self_corr_mean", "Delta attenuated self-correlation", "RdBu_r", True),
        ("mean_other_corr", "Delta attenuated other-correlation", "RdBu_r", True),
        ("recovery_accuracy", "Delta recovery accuracy", "RdBu_r", True),
    ]
    images = []
    for ax, (value_col, title, cmap, center_zero) in zip(axes, specs):
        data = pivot_delta(track_summary, "noisy_by_clean", value_col)
        images.append(draw_heatmap(ax, data, title, cmap, center_zero))

    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label, x=-0.10, y=1.05)

    for ax, im in zip(axes, images):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cbar.ax.tick_params(labelsize=FONT["small"])

    fig.suptitle("Selected minus random, 100k-pool noisy-by-clean diagnostics", y=0.98)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.12, wspace=0.40, hspace=0.58)
    save_figure(fig, "pool100k_noisy_breakdown")


def save_figure(fig, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=DPI)
        print(f"Saved {out}")
    plt.close(fig)


def main() -> None:
    components, recovery, track_summary = load_tables()
    combined = combined_components(components, recovery)
    make_objective_figure(combined)
    make_absolute_self_other_figure(combined)
    make_breakdown_figure(track_summary)


if __name__ == "__main__":
    main()
