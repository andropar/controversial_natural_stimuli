#!/usr/bin/env python3
"""Make the four-panel explanation-analysis summary figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[2]
DATA = PAPER / "18_explain_alignment_effect" / "results"
FIGURES = PAPER / "18_explain_alignment_effect" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MODEL_SET_ORDER = ["all_models", "architecture", "dataset", "sota", "training_objective"]
FAMILY_ORDER = ["low_level", "ood", "semantic_embedding", "model_disagreement", "model_space"]


def main() -> None:
    import matplotlib.pyplot as plt

    ladder = pd.read_csv(DATA / "matched_counterfactual_ladder_summary.csv")
    reliability = pd.read_csv(DATA / "reliability_control_summary.csv")
    pair = pd.read_csv(DATA / "pair_variance_partition_summary.csv")
    residual = pd.read_csv(DATA / "residual_readout_contrasts.csv")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
    ax = axes[0, 0]
    rel = reliability.set_index("model_set").reindex(MODEL_SET_ORDER)
    x = np.arange(len(rel))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(x - 0.18, rel["delta_mean"], width=0.34, color="#0072B2", label="Raw")
    ax.bar(x + 0.18, rel["delta_NCnorm_mean"], width=0.34, color="#D55E00", label="NC-normalized")
    ax.set_xticks(x)
    ax.set_xticklabels(["all", "arch", "data", "sota", "train"])
    ax.set_ylabel("Controversial - baseline")
    ax.set_title("A  Reliability", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    pooled = ladder[ladder["scope"] == "pooled"].sort_values("order")
    y = np.arange(len(pooled))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.plot(
        pooled["mean_delta"],
        y,
        color="#555555",
        linewidth=1.0,
        zorder=1,
    )
    ax.scatter(pooled["mean_delta"], y, color="#009E73", edgecolor="black", linewidth=0.5, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(pooled["ladder_step"] + "  " + pooled["baseline_label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mixed-RSA delta")
    ax.set_title("B  Matched counterfactuals", loc="left", fontsize=10)

    ax = axes[1, 0]
    pv = pair[pair["scope"] == "pooled"].set_index("family").reindex(FAMILY_ORDER)
    x = np.arange(len(pv))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(x, pv["mean_unique_r2_drop"], color="#0072B2", width=0.55)
    ax.errorbar(
        x,
        pv["mean_unique_r2_drop"],
        yerr=1.96 * pv["sem_unique_r2_drop"],
        fmt="none",
        color="black",
        linewidth=0.8,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["low", "OOD", "embed", "disagree", "models"])
    ax.set_ylabel("Unique blocked-CV R2 drop")
    ax.set_title("C  Pair-level follow-up", loc="left", fontsize=10)

    ax = axes[1, 1]
    rr = residual[residual["metric"].isin(["r_ensemble_cv", "loso_residual_fraction"])].copy()
    rr = rr.pivot(index="stimulus_group", columns="metric", values="mean_diff_diagnostic_minus_baseline")
    rr = rr.reindex(MODEL_SET_ORDER)
    x = np.arange(len(rr))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(x - 0.18, rr["r_ensemble_cv"], width=0.34, color="#CC79A7", label="Ensemble RSA")
    ax.bar(x + 0.18, rr["loso_residual_fraction"], width=0.34, color="#E69F00", label="Residual fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(["all", "arch", "data", "sota", "train"])
    ax.set_ylabel("Delta vs vicco")
    ax.set_title("D  Residual structure", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.45)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES / f"explanation_summary.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {FIGURES / 'explanation_summary.pdf'}")


if __name__ == "__main__":
    main()
