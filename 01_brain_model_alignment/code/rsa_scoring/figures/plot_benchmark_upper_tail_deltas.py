#!/usr/bin/env python3
"""Upper-tail broad-benchmark deltas for the Fig. 4 top-of-distribution claim."""
from __future__ import annotations

import math
import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

PAPER = Path(__file__).resolve().parents[2]
PROJECT = PAPER.parents[1]
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PROJECT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cstims.paper import config  # noqa: E402
from cstims.paper.style_improved import DPI, FONT, OKABE_ITO, W_DOUBLE, add_panel_label, apply_style  # noqa: E402


FIGURES_DIR = Path(__file__).resolve().parent
SCORES_PATH = config.RSA_DATA_DIR / "rsa_large_benchmark_scores.csv"
NC_PATH = config.STATS_DATA_DIR / "rdm_noise_ceilings.csv"
OUT_CSV = config.RSA_DATA_DIR / "benchmark_upper_tail_deltas.csv"
SUMMARY_CSV = config.RSA_DATA_DIR / "benchmark_upper_tail_deltas_summary.csv"

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
TOP_K = 10


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


def upper_tail_deltas(scores: pd.DataFrame, nc: pd.DataFrame, score_col: str) -> pd.DataFrame:
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
            top = base_model.loc[common].sort_values(ascending=False).head(min(TOP_K, len(common))).index
            deltas = cstim_model.loc[top] - base_model.loc[top]
            rows.append(
                {
                    "subject": subject,
                    "model_set": model_set,
                    "score_col": score_col,
                    "n_common_models": len(common),
                    "n_top_models": len(top),
                    "top_k_requested": TOP_K,
                    "top_delta_mean": float(deltas.mean()),
                    "top_delta_median": float(deltas.median()),
                    "top_baseline_mean": float(base_model.loc[top].mean()),
                    "top_cstim_mean": float(cstim_model.loc[top].mean()),
                    "top_models": ";".join(top.tolist()),
                }
            )
    return pd.DataFrame(rows)


def draw() -> None:
    apply_style()
    scores = pd.read_csv(SCORES_PATH)
    nc = load_nc_lookup()
    all_deltas = []
    for score_col, _, _ in METHODS:
        deltas = upper_tail_deltas(scores, nc, score_col)
        all_deltas.append(deltas)
    out = pd.concat(all_deltas, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    summary = (
        out.groupby(["score_col", "model_set"], as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            mean_top_delta=("top_delta_mean", "mean"),
            sem_top_delta=("top_delta_mean", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            median_top_delta=("top_delta_median", "mean"),
        )
    )
    summary["ci95_top_delta"] = 1.96 * summary["sem_top_delta"]
    summary.to_csv(SUMMARY_CSV, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.78, 3.35), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.24, wspace=0.10)
    for ax, (score_col, title, color), panel in zip(axes, METHODS, ["a", "b"]):
        sub = out[out["score_col"] == score_col]
        x = np.arange(len(MODEL_SETS))
        means = []
        cis = []
        for i, model_set in enumerate(MODEL_SETS):
            vals = sub[sub["model_set"] == model_set]["top_delta_mean"].to_numpy()
            means.append(float(np.mean(vals)))
            cis.append(float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            jitter = np.linspace(-0.09, 0.09, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=18,
                facecolor="white",
                edgecolor="#333333",
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
        ax.axhline(0, color="#222222", linewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([SET_LABELS[m] for m in MODEL_SETS], fontsize=FONT["tick"])
        ax.set_title(title, fontsize=FONT["title"], fontweight="bold", pad=4)
        ax.grid(axis="y", alpha=0.22, linewidth=0.4, zorder=0)
        add_panel_label(ax, panel, x=-0.08, y=1.05)
        if panel == "a":
            ax.set_ylabel("top-10 baseline-ranked delta\n(cstim - baseline, NC-norm.)", fontsize=FONT["axis_label"])
        ax.tick_params(axis="y", labelsize=FONT["tick"])

    ymin = min(-0.36, *(ax.get_ylim()[0] for ax in axes))
    ymax = max(0.08, *(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    for ext in ("pdf", "png"):
        out_path = FIGURES_DIR / f"benchmark_upper_tail_deltas.{ext}"
        fig.savefig(out_path, dpi=DPI)
        print(f"Saved {out_path}")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    draw()
