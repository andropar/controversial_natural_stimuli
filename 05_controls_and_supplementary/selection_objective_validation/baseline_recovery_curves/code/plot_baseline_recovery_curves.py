#!/usr/bin/env python3
"""Plot the VICCO-baseline recovery-curve supplement."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT = Path(__file__).resolve()
SHARE = SCRIPT.parents[4]
SOURCE_PLOT = (
    SHARE
    / "00_stimulus_selection"
    / "decision_checks"
    / "selection_evaluation"
    / "code"
    / "figures"
    / "plot_insilico_evaluation_unique_improved.py"
)
RESULTS = SCRIPT.parents[1] / "results"
FIGURES = SCRIPT.parents[1] / "figures"


def load_source_plot():
    spec = importlib.util.spec_from_file_location("insilico_plot", SOURCE_PLOT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SOURCE_PLOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = load_source_plot()
plot.config.EVAL_DATA_DIR = RESULTS
plot.DATA_SUFFIX = "_baseline_boot"
plot.OUTPUT_TARGETS = [(FIGURES, FIGURES / "png")]


def plot_faceted_curve(
    ax,
    model_set: str,
    track_key: str,
    title: str,
    show_ylabel: bool,
    show_xlabel: bool,
    show_legend: bool = False,
    show_empirical_label: bool = True,
) -> None:
    df = plot.load_discriminability(model_set)
    tracks = ["raw"] if track_key == "raw" else plot.ENCODING_TRACKS
    sel, base, base_std, x = plot.get_sel_rand(df, tracks)
    col = plot.model_set_color(model_set)

    if sel is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(x, base, color="#6F6F6F", lw=1.0, ls="--", alpha=0.75, label="Baseline")
        ax.plot(x, sel, color=col, lw=1.35, ls="-", alpha=0.96, label="Controversial")
        if base_std is not None:
            ax.fill_between(
                x,
                base - base_std,
                base + base_std,
                color="#6F6F6F",
                alpha=0.10,
                linewidth=0,
            )

    plot._style_curve_axis(
        ax,
        title,
        show_xlabel=False,
        show_empirical_label=show_empirical_label,
    )
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(["0.01", "0.1", "1", "10"])
    if not show_xlabel:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel("")
    if not show_ylabel:
        plt.setp(ax.get_yticklabels(), visible=False)
    if show_legend:
        ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            fontsize=plot.FONT["small"],
            handlelength=1.5,
            handletextpad=0.4,
        )


def plot_summary_bars(
    ax,
    data: dict[str, list],
    title: str,
    ylabel: str,
    show_xticklabels: bool,
    show_legend: bool,
):
    if not data["labels"]:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = np.arange(len(data["labels"]), dtype=float)
    width = 0.34
    err_kw = dict(ecolor="0.3", capsize=2, elinewidth=0.8)

    ax.bar(
        x - width / 2,
        data["random"],
        width,
        yerr=data["random_err"],
        color="#DDDDDD",
        alpha=0.85,
        edgecolor="#666666",
        linewidth=0.5,
        hatch="//",
        label="Baseline",
        error_kw=err_kw,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        data["selected"],
        width,
        yerr=data["selected_err"],
        color=data["colors"],
        alpha=0.90,
        edgecolor="white",
        linewidth=0.5,
        label="Controversial",
        error_kw=err_kw,
        zorder=3,
    )

    bottom, top = plot._bar_ylim(data)
    y_span = top - bottom
    for i, color in enumerate(data["colors"]):
        bar_top = max(
            data["random"][i] + data["random_err"][i],
            data["selected"][i] + data["selected_err"][i],
        )
        pct = (
            (data["selected"][i] - data["random"][i])
            / max(data["random"][i], 1e-6)
            * 100
        )
        ax.text(
            x[i],
            bar_top + 0.05 * y_span,
            f"{pct:+.0f}%",
            fontsize=plot.FONT["small"],
            ha="center",
            va="bottom",
            color=color,
            alpha=0.95,
        )

    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(data["labels"]) - 0.5)
    if show_xticklabels:
        ax.set_xticklabels(data["labels"], rotation=25, ha="right")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom, top)
    if title:
        ax.set_title(title, pad=3)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    if show_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, -0.02),
            ncol=2,
            frameon=True,
            framealpha=0.88,
            edgecolor="none",
            columnspacing=0.6,
            handlelength=1.2,
            handletextpad=0.3,
            borderpad=0.25,
            labelspacing=0.25,
            fontsize=plot.FONT["small"],
        )


plot.plot_faceted_curve = plot_faceted_curve
plot.plot_summary_bars = plot_summary_bars


def main() -> None:
    plot.make_figure("auc", "baseline_recovery_curves")


if __name__ == "__main__":
    main()
