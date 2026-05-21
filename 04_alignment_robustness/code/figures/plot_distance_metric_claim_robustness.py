#!/usr/bin/env python3
"""Downstream claim robustness under correlation vs cosine RDM distance."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PAPER = Path(__file__).resolve().parents[2]
PROJECT = PAPER.parents[1]
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PAPER / "figures"))

import config  # noqa: E402
from style_improved import (  # noqa: E402
    DPI,
    FONT,
    MODEL_SET_COLORS,
    MODEL_SET_DISPLAY_SHORT,
    MODEL_SET_ORDER,
    W_DOUBLE,
    add_panel_label,
    apply_style,
)

MODEL_SET_MARKERS = {
    "all_models": "o",
    "sota": "s",
    "training_objective": "^",
    "architecture": "D",
    "dataset": "P",
}


FIGURES_DIR = Path(__file__).resolve().parent
FIXED_PATH = config.STATS_DATA_DIR / "distance_metric_robustness.csv"
MIXED_PATH = config.STATS_DATA_DIR / "mixed_distance_metric_robustness.csv"
OUT_CSV = config.STATS_DATA_DIR / "distance_metric_claim_robustness.csv"
SUMMARY_CSV = config.STATS_DATA_DIR / "distance_metric_claim_robustness_summary.csv"


def median_pairwise_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    diffs = np.abs(values[:, None] - values[None, :])
    tri = np.triu_indices(len(values), k=1)
    return float(np.median(diffs[tri]))


def summarize_endpoint(df: pd.DataFrame, method: str, corr_col: str, cosine_col: str) -> pd.DataFrame:
    rows = []
    for (subject, model_set), grp in df.groupby(["subject", "model_set"]):
        for distance_label, col in [("correlation", corr_col), ("cosine", cosine_col)]:
            cstim = grp[(grp["stimulus_type"] == "controversial") & (grp["bootstrap_idx"] == 0)]
            base = grp[grp["stimulus_type"] == "vicco"]
            common = sorted(set(cstim["model"]).intersection(base["model"]))
            if len(common) < 2:
                continue
            cstim_scores = cstim.set_index("model").loc[common, col].astype(float)
            base_model_mean = base.groupby("model")[col].mean().loc[common].astype(float)
            base_spreads = []
            for _, bgrp in base.groupby("bootstrap_idx"):
                bvals = bgrp.set_index("model").reindex(common)[col].to_numpy(dtype=float)
                base_spreads.append(median_pairwise_abs(bvals))
            cstim_spread = median_pairwise_abs(cstim_scores.to_numpy(dtype=float))
            base_spread = float(np.nanmean(base_spreads))
            spread_ratio = cstim_spread / base_spread if np.isfinite(base_spread) and base_spread > 0 else np.nan
            rows.append(
                {
                    "subject": subject,
                    "model_set": model_set,
                    "method": method,
                    "distance_metric": distance_label,
                    "n_models": len(common),
                    "mean_delta": float(cstim_scores.mean() - base_model_mean.mean()),
                    "cstim_mean": float(cstim_scores.mean()),
                    "baseline_mean": float(base_model_mean.mean()),
                    "cstim_spread": cstim_spread,
                    "baseline_spread": base_spread,
                    "spread_ratio": spread_ratio,
                    "log2_spread_ratio": float(np.log2(spread_ratio)) if spread_ratio > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_summary() -> pd.DataFrame:
    fixed = pd.read_csv(FIXED_PATH)
    mixed = pd.read_csv(MIXED_PATH)
    out = pd.concat(
        [
            summarize_endpoint(fixed, "fixed RSA", "crsa_correlation", "crsa_cosine"),
            summarize_endpoint(mixed, "mixed RSA", "mixed_rsa_correlation", "mixed_rsa_cosine"),
        ],
        ignore_index=True,
    )
    out.to_csv(OUT_CSV, index=False)
    rows = []
    for (method, statistic), grp in out.groupby(["method", "distance_metric"]):
        rows.append(
            {
                "method": method,
                "distance_metric": statistic,
                "mean_delta_mean": grp["mean_delta"].mean(),
                "mean_delta_sem": grp["mean_delta"].std(ddof=1) / np.sqrt(len(grp)),
                "log2_spread_ratio_mean": grp["log2_spread_ratio"].mean(),
                "log2_spread_ratio_sem": grp["log2_spread_ratio"].std(ddof=1) / np.sqrt(len(grp)),
                "n_endpoints": len(grp),
            }
        )
    summary = pd.DataFrame(rows)
    # Pairwise correspondence of the downstream endpoint statistics.
    for method in ["fixed RSA", "mixed RSA"]:
        wide = out[out["method"] == method].pivot_table(
            index=["subject", "model_set"],
            columns="distance_metric",
            values=["mean_delta", "log2_spread_ratio"],
        )
        for metric in ["mean_delta", "log2_spread_ratio"]:
            paired = wide[metric].dropna()
            rho = spearmanr(paired["correlation"], paired["cosine"]).statistic if len(paired) > 1 else np.nan
            same_sign = np.mean(np.sign(paired["correlation"]) == np.sign(paired["cosine"])) if len(paired) else np.nan
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        [
                            {
                                "method": method,
                                "distance_metric": f"paired_{metric}",
                                "mean_delta_mean": np.nan,
                                "mean_delta_sem": np.nan,
                                "log2_spread_ratio_mean": np.nan,
                                "log2_spread_ratio_sem": np.nan,
                                "n_endpoints": len(paired),
                                "spearman_correlation_vs_cosine": rho,
                                "same_sign_fraction": same_sign,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    summary.to_csv(SUMMARY_CSV, index=False)
    return out


def plot_metric(ax, data: pd.DataFrame, method: str, value_col: str, title: str, panel: str) -> None:
    wide = data[data["method"] == method].pivot_table(
        index=["subject", "model_set"],
        columns="distance_metric",
        values=value_col,
    ).reset_index()
    for model_set in MODEL_SET_ORDER:
        sub = wide[wide["model_set"] == model_set]
        ax.scatter(
            sub["correlation"],
            sub["cosine"],
            s=28,
            color=MODEL_SET_COLORS[model_set],
            marker=MODEL_SET_MARKERS[model_set],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            label=MODEL_SET_DISPLAY_SHORT[model_set] if panel == "a" else None,
            zorder=3,
        )
    vals = np.concatenate([wide["correlation"].to_numpy(), wide["cosine"].to_numpy()])
    vals = vals[np.isfinite(vals)]
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.10 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#333333", lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color="#BBBBBB", lw=0.6, zorder=0)
    ax.axvline(0, color="#BBBBBB", lw=0.6, zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    rho = spearmanr(wide["correlation"], wide["cosine"]).statistic
    ax.text(0.04, 0.94, f"$\\rho$ = {rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=FONT["tick"])
    ax.set_title(f"{method}: {title}", fontsize=FONT["title"], pad=4)
    ax.set_xlabel("correlation distance", fontsize=FONT["axis_label"])
    ax.set_ylabel("cosine distance", fontsize=FONT["axis_label"])
    ax.tick_params(axis="both", labelsize=FONT["tick"])
    ax.grid(True, color="#E6E6E6", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(ax, panel, x=-0.12, y=1.05)


def draw() -> None:
    apply_style()
    data = build_summary()
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE * 0.86, 6.35))
    plot_metric(axes[0, 0], data, "mixed RSA", "mean_delta", "mean alignment delta", "a")
    plot_metric(axes[0, 1], data, "mixed RSA", "log2_spread_ratio", "spread ratio", "b")
    plot_metric(axes[1, 0], data, "fixed RSA", "mean_delta", "mean alignment delta", "c")
    plot_metric(axes[1, 1], data, "fixed RSA", "log2_spread_ratio", "spread ratio", "d")
    axes[0, 0].legend(
        frameon=False,
        loc="lower right",
        fontsize=FONT["legend"] - 1,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.94, wspace=0.28, hspace=0.48)
    for ext in ("pdf", "png"):
        out = FIGURES_DIR / f"distance_metric_claim_robustness.{ext}"
        fig.savefig(out, dpi=DPI)
        print(f"Saved {out}")
    plt.close(fig)
    print(pd.read_csv(SUMMARY_CSV).to_string(index=False))


if __name__ == "__main__":
    draw()
