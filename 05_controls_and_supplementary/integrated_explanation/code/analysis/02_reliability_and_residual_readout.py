#!/usr/bin/env python3
"""Reliability and residual-structure readout for the explanation analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[2]
DATA = PAPER / "18_explain_alignment_effect" / "results"
FIGURES = PAPER / "18_explain_alignment_effect" / "figures"
PRIMARY = PAPER / "03_statistics" / "results" / "primary_endpoint_summary.csv"
RESIDUAL_SUMMARY = PAPER / "10_residual_reliability" / "results" / "residual_decomposition_summary.csv"
RESIDUAL_CONTRASTS = PAPER / "10_residual_reliability" / "results" / "residual_decomposition_contrasts.csv"

DATA.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

MODEL_SET_ORDER = ["all_models", "architecture", "dataset", "sota", "training_objective"]
RESIDUAL_METRICS = [
    "correlation_ceiling",
    "r_ensemble_cv",
    "ensemble_gap_to_correlation_ceiling",
    "residual_split_half_reliability",
    "within_residual_fraction",
    "r_loso_residual",
    "loso_residual_fraction",
]


def _sem(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def _summarize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for model_set, grp in df.groupby("model_set", sort=False):
        row = {
            "model_set": model_set,
            "n_subjects": int(grp["subject"].nunique()),
            "n_rows": int(len(grp)),
        }
        for col in cols:
            if col not in grp:
                continue
            row[f"{col}_mean"] = float(grp[col].mean())
            row[f"{col}_sem"] = _sem(grp[col])
            row[f"{col}_n_negative"] = int((grp[col] < 0).sum()) if col.startswith("delta") else np.nan
            row[f"{col}_n_positive"] = int((grp[col] > 0).sum()) if col.startswith("delta") else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_control() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PRIMARY)
    df = df[
        (df["metric"] == "mixed_RSA")
        & (df["baseline_type"] == "same_session_unselected")
        & (df["primary_alignment_endpoint"].fillna(False).astype(bool))
    ].copy()
    df = df.sort_values(["model_set", "subject"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No same-session mixed-RSA rows found in {PRIMARY}")

    df["nc_gap_cstim_minus_baseline"] = df["NC_cstim_mean"] - df["NC_baseline_mean"]
    df["nc_ratio_cstim_over_baseline"] = df["NC_cstim_mean"] / df["NC_baseline_mean"]
    df["raw_delta_negative"] = df["delta"] < 0
    df["nc_norm_delta_negative"] = df["delta_NCnorm"] < 0
    df["reliability_read"] = np.select(
        [
            (df["delta"] < 0) & (df["delta_NCnorm"] >= 0),
            (df["delta"] < 0) & (df["delta_NCnorm"] < 0),
            (df["delta"] >= 0),
        ],
        [
            "raw drop removed after NC normalization",
            "drop persists after NC normalization",
            "no raw drop",
        ],
        default="unclassified",
    )

    summary = _summarize(
        df,
        [
            "delta",
            "delta_NCnorm",
            "NC_cstim_mean",
            "NC_baseline_mean",
            "nc_gap_cstim_minus_baseline",
            "nc_ratio_cstim_over_baseline",
        ],
    )
    return df, summary


def residual_readout() -> pd.DataFrame:
    summary = pd.read_csv(RESIDUAL_SUMMARY)
    summary = summary[
        (summary["rsa_type"] == "mixed")
        & (summary["stimulus_group"].isin(MODEL_SET_ORDER + ["vicco"]))
    ].copy()
    summary.to_csv(DATA / "residual_readout_summary.csv", index=False)

    contrasts = pd.read_csv(RESIDUAL_CONTRASTS)
    contrasts = contrasts[
        (contrasts["rsa_type"] == "mixed")
        & (contrasts["stimulus_group"].isin(MODEL_SET_ORDER))
        & (contrasts["metric"].isin(RESIDUAL_METRICS))
    ].copy()
    contrasts.to_csv(DATA / "residual_readout_contrasts.csv", index=False)
    return contrasts


def plot_reliability(df: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))

    means = (
        df.groupby("model_set", as_index=False)[["delta", "delta_NCnorm"]]
        .mean()
        .set_index("model_set")
        .reindex(MODEL_SET_ORDER)
    )
    x = np.arange(len(means))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].bar(x - 0.18, means["delta"], width=0.34, label="Raw", color="#0072B2")
    axes[0].bar(x + 0.18, means["delta_NCnorm"], width=0.34, label="NC-normalized", color="#D55E00")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["all", "arch", "data", "sota", "train"], rotation=0)
    axes[0].set_ylabel("Controversial - same-session baseline")
    axes[0].set_title("A  Reliability control", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    resid = contrasts[contrasts["metric"] == "loso_residual_fraction"].copy()
    resid = resid.set_index("stimulus_group").reindex(MODEL_SET_ORDER)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].bar(
        np.arange(len(resid)),
        resid["mean_diff_diagnostic_minus_baseline"],
        color="#009E73",
        width=0.55,
    )
    axes[1].set_xticks(np.arange(len(resid)))
    axes[1].set_xticklabels(["all", "arch", "data", "sota", "train"], rotation=0)
    axes[1].set_ylabel("Delta vs vicco")
    axes[1].set_title("B  Residual fraction after model removal", loc="left", fontsize=10)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.45)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES / f"reliability_and_residual_readout.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    by_cell, rel_summary = reliability_control()
    by_cell.to_csv(DATA / "reliability_control_by_cell.csv", index=False)
    rel_summary.to_csv(DATA / "reliability_control_summary.csv", index=False)
    contrasts = residual_readout()
    plot_reliability(by_cell, contrasts)
    print(f"wrote {DATA / 'reliability_control_by_cell.csv'}")
    print(f"wrote {DATA / 'reliability_control_summary.csv'}")
    print(f"wrote {DATA / 'residual_readout_contrasts.csv'}")
    print(rel_summary.to_string(index=False))


if __name__ == "__main__":
    main()
