#!/usr/bin/env python3
"""Subject-level median deltas for the broad benchmark figure."""
from __future__ import annotations

import math
import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cstims.paper import config
from cstims.paper.style_improved import apply_style, FONT, DPI, W_DOUBLE, OKABE_ITO, add_panel_label

apply_style()

FIGURES_DIR = Path(__file__).resolve().parent
SCORES_PATH = config.RSA_DATA_DIR / "rsa_large_benchmark_scores.csv"
NC_PATH = config.STATS_DATA_DIR / "rdm_noise_ceilings.csv"

MODEL_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
SET_LABELS = {
    "all_models": "All",
    "sota": "SOTA",
    "training_objective": "Train.",
    "architecture": "Arch.",
    "dataset": "Dataset",
}
METHODS = [
    ("wrsa_transfer", "mixed RSA", OKABE_ITO["blue"]),
    ("crsa", "fixed RSA", OKABE_ITO["sky_blue"]),
]


def load_nc_lookup() -> pd.DataFrame:
    nc = pd.read_csv(NC_PATH)
    rows = []
    for _, row in nc.iterrows():
        group = "same_session" if row["stimulus_type"] == "vicco" else row["group"]
        rows.append(
            {
                "subject": row["subject"],
                "group": group,
                "stimulus_type": row["stimulus_type"],
                "bootstrap_idx": row["bootstrap_idx"],
                "within_sqrt_rSB": math.sqrt(max(float(row["noise_ceiling_spearman"]), 0.0)),
            }
        )
    out = pd.DataFrame(rows)
    base = (
        out[out["stimulus_type"] == "vicco"]
        .groupby("subject", as_index=False)
        .agg(within_sqrt_rSB=("within_sqrt_rSB", "mean"))
    )
    base["group"] = "same_session"
    cstim = out[(out["stimulus_type"] == "controversial") & (out["bootstrap_idx"] == 0)]
    return pd.concat([cstim[["subject", "group", "within_sqrt_rSB"]], base], ignore_index=True)


def subject_median_deltas(scores: pd.DataFrame, nc: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for subject in sorted(scores["subject"].unique()):
        base = scores[(scores["subject"] == subject) & (scores["stimulus_type"] == "vicco")]
        if base.empty:
            continue
        nc_base = nc[(nc["subject"] == subject) & (nc["group"] == "same_session")]["within_sqrt_rSB"].mean()
        if not np.isfinite(nc_base) or nc_base <= 0:
            continue
        base_model = base.groupby("model")[score_col].mean() / nc_base
        for model_set in MODEL_SETS:
            cstim = scores[
                (scores["subject"] == subject)
                & (scores["stimulus_type"] == "controversial")
                & (scores["group"] == model_set)
            ]
            if cstim.empty:
                continue
            nc_cstim = nc[(nc["subject"] == subject) & (nc["group"] == model_set)]["within_sqrt_rSB"].mean()
            if not np.isfinite(nc_cstim) or nc_cstim <= 0:
                continue
            cstim_model = cstim.groupby("model")[score_col].mean() / nc_cstim
            common = base_model.index.intersection(cstim_model.index)
            if len(common) < 5:
                continue
            deltas = cstim_model.loc[common] - base_model.loc[common]
            rows.append(
                {
                    "subject": subject,
                    "model_set": model_set,
                    "n_models": len(common),
                    "median_delta": float(np.median(deltas)),
                }
            )
    return pd.DataFrame(rows)


def draw():
    scores = pd.read_csv(SCORES_PATH)
    nc = load_nc_lookup()

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.78, 3.4), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.24, wspace=0.10)

    for ax, (score_col, title, color), panel_label in zip(axes, METHODS, ["a", "b"]):
        deltas = subject_median_deltas(scores, nc, score_col)
        x = np.arange(len(MODEL_SETS))
        means = []
        cis = []
        for model_set in MODEL_SETS:
            vals = deltas[deltas["model_set"] == model_set]["median_delta"].to_numpy()
            means.append(np.mean(vals))
            cis.append(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            jitter = np.linspace(-0.09, 0.09, len(vals))
            ax.scatter(
                np.full(len(vals), x[len(means) - 1]) + jitter,
                vals,
                s=18,
                facecolor="white",
                edgecolor="#333",
                linewidth=0.6,
                zorder=4,
            )
        ax.errorbar(
            x,
            means,
            yerr=cis,
            fmt="o",
            markersize=5.2,
            markerfacecolor=color,
            markeredgecolor="#222222",
            markeredgewidth=0.5,
            color="#222222",
            ecolor="#222222",
            capsize=3,
            linewidth=0.9,
            zorder=5,
        )
        ax.axhline(0, color="#222", linewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([SET_LABELS[m] for m in MODEL_SETS], rotation=0)
        ax.set_title(title, fontweight="bold", pad=4)
        ax.grid(axis="y", alpha=0.22, linewidth=0.4, zorder=0)
        add_panel_label(ax, panel_label, x=-0.08, y=1.05)
        if panel_label == "a":
            ax.set_ylabel("median delta\n(cstim - baseline, NC-norm.)", labelpad=5)

    ymins = [ax.get_ylim()[0] for ax in axes]
    ymaxs = [ax.get_ylim()[1] for ax in axes]
    ymin = min(min(ymins), -0.30)
    ymax = max(max(ymaxs), 0.08)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"benchmark_median_deltas.{ext}"
        fig.savefig(out, dpi=DPI)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    draw()
