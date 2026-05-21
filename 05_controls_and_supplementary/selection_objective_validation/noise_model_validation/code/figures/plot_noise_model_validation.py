#!/usr/bin/env python3
"""Plot the supplemental RDM-space vs feature-space noise validation."""

from __future__ import annotations

import sys
import csv
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


COLOR_RDM = OKABE_ITO["blue"]
COLOR_FEAT = OKABE_ITO["vermillion"]
COLOR_DIFF = OKABE_ITO["bluish_green"]
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


def load_data():
    nc = read_csv(DATA / "noise_ceiling_results.csv")
    corr = read_csv(DATA / "correlation_matrix_results.csv")
    disc = read_csv(DATA / "discriminability_results.csv")
    rank = read_csv(DATA / "ranking_results.csv")
    return nc, corr, disc, rank


def group_noise_ceiling(nc: list[dict[str, float | str]]) -> dict[str, np.ndarray]:
    targets = sorted({float(row["nc_target"]) for row in nc})
    out = {"nc_target": np.array(targets)}
    for prefix, column in [("rdm", "achieved_nc_rdm"), ("feat", "achieved_nc_feat")]:
        means = []
        sems = []
        for target in targets:
            vals = np.array([float(row[column]) for row in nc if float(row["nc_target"]) == target])
            means.append(vals.mean())
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))
        out[f"{prefix}_mean"] = np.array(means)
        out[f"{prefix}_sem"] = np.array(sems)
    return out


def values(rows: list[dict[str, float | str]], column: str) -> np.ndarray:
    return np.array([float(row[column]) for row in rows])


def plot_noise_ceiling(ax, nc: list[dict[str, float | str]]):
    summary = group_noise_ceiling(nc)
    x = np.arange(len(summary["nc_target"]))
    width = 0.34
    ax.bar(
        x - width / 2,
        summary["rdm_mean"],
        width,
        yerr=summary["rdm_sem"],
        color=COLOR_RDM,
        edgecolor="none",
        label="RDM-space",
        error_kw=dict(lw=0.6, capsize=2, capthick=0.6),
    )
    ax.bar(
        x + width / 2,
        summary["feat_mean"],
        width,
        yerr=summary["feat_sem"],
        color=COLOR_FEAT,
        edgecolor="none",
        label="Feature-space",
        error_kw=dict(lw=0.6, capsize=2, capthick=0.6),
    )
    ax.plot(x, summary["nc_target"], color="#222222", lw=0.7, marker="o", ms=2.5, label="Target")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in summary["nc_target"]])
    ax.set_xlabel("Target noise ceiling")
    ax.set_ylabel("Achieved NC")
    ax.set_title("Noise calibration", fontsize=FONT["title"], pad=3)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=FONT["small"], ncol=1, loc="upper left", bbox_to_anchor=(0.08, 1.0))
    ax.grid(axis="y", alpha=0.35)
    panel_label(ax, "a")


def plot_correlation_structure(ax, corr: list[dict[str, float | str]]):
    nc_target = values(corr, "nc_target")
    x = np.arange(len(corr))
    ax.plot(
        x,
        values(corr, "upper_tri_pearson_r"),
        color=COLOR_DIFF,
        marker="o",
        ms=3,
        lw=1.2,
        label="model-correlation matrix",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in nc_target])
    ax.set_xlabel("Target noise ceiling")
    ax.set_ylabel("Matrix r")
    ax.set_title("Model-correlation matrices\nRDM vs feature-space", fontsize=FONT["title"], pad=3)
    ax.set_ylim(0.94, 1.005)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=FONT["small"], loc="lower right")
    panel_label(ax, "b")


def plot_downstream(
    ax,
    disc: list[dict[str, float | str]],
):
    x = np.arange(len(disc))
    nc_target = values(disc, "nc_target")
    ax.plot(
        x,
        100 * (1.0 - values(disc, "error_prob_rdm")),
        color=COLOR_RDM,
        marker="o",
        ms=3,
        lw=1.2,
        label="RDM-space",
    )
    ax.plot(
        x,
        100 * (1.0 - values(disc, "error_prob_feat")),
        color=COLOR_FEAT,
        marker="s",
        ms=3,
        lw=1.2,
        label="Feature-space",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in nc_target])
    ax.set_xlabel("Target noise ceiling")
    ax.set_ylabel("ID accuracy (%)")
    ax.set_title("Model-identification accuracy", fontsize=FONT["title"], pad=3)
    ax.set_ylim(99.74, 100.03)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=FONT["small"], loc="lower right")
    ax.text(
        0.04,
        0.10,
        "Max difference: 0.2 pp",
        transform=ax.transAxes,
        fontsize=FONT["small"],
        color=GRAY,
        ha="left",
        va="bottom",
    )
    panel_label(ax, "c")


def plot_candidate_ranking(ax, rank: list[dict[str, float | str]]):
    x = np.arange(len(rank))
    nc_target = values(rank, "nc_target")

    ax.plot(
        x,
        values(rank, "top20_overlap"),
        color=COLOR_FEAT,
        marker="s",
        ms=3,
        lw=1.2,
        label="top-20 overlap",
    )
    ax.plot(
        x,
        values(rank, "spearman_rho"),
        color=COLOR_DIFF,
        marker="o",
        ms=3,
        lw=1.2,
        label="rank rho",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in nc_target])
    ax.set_xlabel("Target noise ceiling")
    ax.set_ylabel("Rank agreement")
    ax.set_title("Candidate rankings\nRDM vs feature-space", fontsize=FONT["title"], pad=3)
    ax.set_ylim(0, 0.82)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=FONT["small"], loc="upper left", bbox_to_anchor=(0.08, 1.0))
    panel_label(ax, "d")


def main():
    apply_style()
    nc, corr, disc, rank = load_data()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(W_DOUBLE, 5.25),
        gridspec_kw=dict(wspace=0.34, hspace=0.72),
    )

    plot_noise_ceiling(axes[0, 0], nc)
    plot_correlation_structure(axes[0, 1], corr)
    plot_downstream(axes[1, 0], disc)
    plot_candidate_ranking(axes[1, 1], rank)

    fig.savefig(HERE / "noise_model_validation.pdf", dpi=DPI)
    fig.savefig(HERE / "noise_model_validation.png", dpi=DPI)
    print(f"Wrote {HERE / 'noise_model_validation.pdf'}")


if __name__ == "__main__":
    main()
