#!/usr/bin/env python3
"""Plot analytical-vs-MC objective validation."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
PAPER = SECTION.parent
DATA = SECTION / "data"

sys.path.insert(0, str(PAPER / "figures"))
from style_improved import (  # noqa: E402
    DPI,
    FONT,
    OKABE_ITO,
    W_DOUBLE,
    apply_style,
)


COLOR_ANALYTICAL = OKABE_ITO["blue"]
COLOR_MC = OKABE_ITO["vermillion"]
COLOR_AGREE = OKABE_ITO["bluish_green"]
COLOR_BIAS = OKABE_ITO["orange"]
GRAY = "#666666"


def panel_label(ax, label: str):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=FONT["panel_label"],
        fontweight="bold",
        ha="left",
        va="top",
        color="#111111",
    )


def read_csv(path: Path) -> list[dict[str, float | str]]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def values(rows: list[dict[str, float | str]], column: str) -> np.ndarray:
    return np.array([float(row[column]) for row in rows])


def plot_scatter(ax, rows: list[dict[str, float | str]], summary: dict[str, float | str]):
    analytical = values(rows, "utility_analytical")
    mc = values(rows, "utility_mc")
    sem = values(rows, "utility_mc_sem")

    lo = min(analytical.min(), mc.min())
    hi = max(analytical.max(), mc.max())
    pad = (hi - lo) * 0.08
    lo -= pad
    hi += pad

    ax.errorbar(
        analytical,
        mc,
        yerr=sem,
        fmt="o",
        ms=2.4,
        lw=0,
        elinewidth=0.35,
        alpha=0.55,
        color=COLOR_ANALYTICAL,
        ecolor="#999999",
        rasterized=True,
    )
    ax.plot([lo, hi], [lo, hi], color="#222222", lw=0.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Analytical utility")
    ax.set_ylabel("MC utility")
    ax.set_title("Paper objective at NC = 0.46", fontsize=FONT["title"], pad=3)
    ax.grid(alpha=0.35)
    ax.text(
        0.12,
        0.95,
        f"rank $\\rho$ = {float(summary['spearman_rho']):.2f}\n"
        f"top-20 = {100 * float(summary['top20_overlap']):.0f}%\n"
        f"regret = {1000 * float(summary['mc_regret_analytical_choice']):.2f} $\\times 10^{{-3}}$\n"
        f"bias = {1000 * float(summary['mean_bias_mc_minus_analytical']):.2f} $\\times 10^{{-3}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT["small"],
        color=GRAY,
    )
    panel_label(ax, "a")


def plot_noise_agreement(ax, rows: list[dict[str, float | str]]):
    x = np.arange(len(rows))
    nc = values(rows, "nc_target")
    ax.plot(
        x,
        values(rows, "spearman_rho"),
        marker="o",
        ms=3,
        lw=1.2,
        color=COLOR_AGREE,
        label="rank rho",
    )
    ax.plot(
        x,
        values(rows, "top20_overlap"),
        marker="s",
        ms=3,
        lw=1.2,
        color=COLOR_MC,
        label="top-20 overlap",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in nc])
    ax.set_xlabel("Target noise ceiling")
    ax.set_ylabel("Agreement")
    ax.set_title("Hard-min objective agreement", fontsize=FONT["title"], pad=3)
    ax.set_ylim(0.45, 1.02)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=FONT["small"], loc="lower right")
    panel_label(ax, "b")


def plot_aggregation_bias(ax, rows: list[dict[str, float | str]]):
    labels = [str(row["label"]).replace(" / ", "\n") for row in rows]
    x = np.arange(len(rows))
    bias = 1000 * values(rows, "mean_bias_mc_minus_analytical")
    mae = 1000 * values(rows, "mae")

    ax.axhline(0, color="#222222", lw=0.7)
    ax.bar(x, bias, width=0.58, color=COLOR_BIAS, edgecolor="none", label="signed bias")
    ax.plot(x, mae, color=COLOR_ANALYTICAL, marker="o", ms=3, lw=1.1, label="mean |error|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Utility difference\n($\\times 10^{-3}$)")
    ax.set_title("Aggregation-dependent bias", fontsize=FONT["title"], pad=3)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=FONT["small"], loc="lower right")
    panel_label(ax, "c")


def plot_aggregation_agreement(ax, rows: list[dict[str, float | str]]):
    labels = [str(row["label"]).replace(" / ", "\n") for row in rows]
    x = np.arange(len(rows))
    width = 0.34

    ax.bar(
        x - width / 2,
        values(rows, "spearman_rho"),
        width,
        color=COLOR_AGREE,
        edgecolor="none",
        label="rank rho",
    )
    ax.bar(
        x + width / 2,
        values(rows, "top20_overlap"),
        width,
        color=COLOR_MC,
        edgecolor="none",
        label="top-20 overlap",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Agreement")
    ax.set_title("Aggregation controls at NC = 0.46", fontsize=FONT["title"], pad=3)
    ax.set_ylim(0.68, 1.02)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=FONT["small"], loc="lower right")
    panel_label(ax, "d")


def main():
    apply_style()
    candidate = read_csv(DATA / "candidate_utilities_nc0.46_mean_min.csv")
    by_noise = read_csv(DATA / "summary_by_noise_ceiling.csv")
    by_agg = read_csv(DATA / "summary_by_aggregation.csv")
    default_summary = next(
        row for row in by_noise if abs(float(row["nc_target"]) - 0.46) < 1e-8
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(W_DOUBLE, 5.25),
        gridspec_kw=dict(wspace=0.36, hspace=0.76),
    )

    plot_scatter(axes[0, 0], candidate, default_summary)
    plot_noise_agreement(axes[0, 1], by_noise)
    plot_aggregation_bias(axes[1, 0], by_agg)
    plot_aggregation_agreement(axes[1, 1], by_agg)

    fig.savefig(HERE / "objective_mc_validation.pdf", dpi=DPI)
    fig.savefig(HERE / "objective_mc_validation.png", dpi=DPI)
    print(f"Wrote {HERE / 'objective_mc_validation.pdf'}")


if __name__ == "__main__":
    main()
