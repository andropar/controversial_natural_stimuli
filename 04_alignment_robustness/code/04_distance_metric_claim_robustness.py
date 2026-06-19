#!/usr/bin/env python3
"""Summarize downstream claim robustness under correlation vs cosine RDM distance.

Inputs:
  04_alignment_robustness/results/distance_metric_robustness.csv
  04_alignment_robustness/results/mixed_distance_metric_robustness.csv

Outputs:
  04_alignment_robustness/results/distance_metric_claim_robustness.csv
  04_alignment_robustness/results/distance_metric_claim_robustness_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parent
PAPER_HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(PAPER_HELPERS))

from cstims import constants, paths


DATA_DIR = paths.robustness_data_dir()
FIXED_PATH = DATA_DIR / "distance_metric_robustness.csv"
MIXED_PATH = DATA_DIR / "mixed_distance_metric_robustness.csv"
OUT_CSV = DATA_DIR / "distance_metric_claim_robustness.csv"
SUMMARY_CSV = DATA_DIR / "distance_metric_claim_robustness_summary.csv"


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
                    "spread_ratio": spread_ratio if np.isfinite(spread_ratio) else np.nan,
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
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
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
    for method in ["fixed RSA", "mixed RSA"]:
        wide = out[out["method"] == method].pivot_table(
            index=["subject", "model_set"],
            columns="distance_metric",
            values=["mean_delta", "log2_spread_ratio"],
        )
        for metric in ["mean_delta", "log2_spread_ratio"]:
            paired = wide[metric].dropna()
            rho = spearmanr(paired["correlation"], paired["cosine"]).statistic if len(paired) > 1 else np.nan
            same_sign = (
                np.mean(np.sign(paired["correlation"]) == np.sign(paired["cosine"]))
                if len(paired)
                else np.nan
            )
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


def main() -> None:
    out = build_summary()
    summary = pd.read_csv(SUMMARY_CSV)
    print(f"Saved {len(out)} rows to {OUT_CSV}")
    print(f"Saved {len(summary)} rows to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
