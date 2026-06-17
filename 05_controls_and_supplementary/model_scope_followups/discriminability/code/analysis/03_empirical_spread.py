#!/usr/bin/env python3
"""Result 4(c): Empirical spread consolidation.

Loads existing per-subject scores and computes between-model spread metrics
per (metric, set), comparing cstim and vicco baseline:
    - median absolute pairwise difference of model means
    - range
    - IQR
    - normalized variance (CV²)

Reports the same set-by-set spread numbers that already appear in the paper,
plus the spread ratio (cstim/vicco) per metric. Honest framing for the
'spread widens on cstim' claim — note where it does and doesn't hold.

Output:
    data/empirical_spread.csv
"""

import sys
from itertools import combinations
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[4]
PAPER = PROJECT / "experiments" / "cstim_paper"
sys.path.insert(0, str(PAPER))

from cstims.paper.config import MODEL_SETS, RSA_DATA_DIR, SUBJECTS  # noqa

DATA_DIR = Path(__file__).resolve().parents[1] / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]


def median_pairwise_diff(x):
    x = np.asarray(x)
    if len(x) < 2:
        return np.nan
    diffs = np.abs(x[:, None] - x[None, :])
    iu, ju = np.triu_indices_from(diffs, k=1)
    return float(np.median(diffs[iu, ju]))


def load_metric(metric):
    fname = "wrsa_transfer_scores.csv" if metric == "mRSA" else "crsa_scores.csv"
    score_col = "wrsa_transfer" if metric == "mRSA" else "crsa"
    dfs = []
    for s in SUBJECTS:
        p = RSA_DATA_DIR / s / fname
        if p.exists():
            d = pd.read_csv(p)
            if "subject" not in d.columns:
                d["subject"] = s
            d = d.rename(columns={score_col: "score"})
            dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def per_model_means(df, model_set, stim_type, models):
    """Returns array of per-model subject-averaged scores."""
    sub = df[(df["model_set"] == model_set) & (df["stimulus_type"] == stim_type)]
    out = []
    for m in models:
        v = sub[sub["model"] == m]["score"]
        if len(v) > 0:
            out.append(v.mean())
    return np.array(out)


def main():
    rows = []
    models = MODEL_SETS["all_models"]
    for metric in ["mRSA", "fRSA"]:
        df = load_metric(metric)
        # vicco model means: per-model average over (subjects × bootstraps)
        vicco_means = per_model_means(df, "vicco", "vicco", models)
        # Vicco rows in csv are duplicated across all model_sets — pick all_models
        # for vicco_means (or any set) — same models, same scores. Above call
        # filters df by model_set == 'vicco' which never matches; need fix.
        # Actually for vicco, model_set IS the cstim set the row is grouped under
        # (e.g., 'vicco' vicco rows live under model_set='all_models', etc.).
        # Use stimulus_type alone:
        vicco_rows = df[df["stimulus_type"] == "vicco"]
        vicco_means = []
        for m in models:
            v = vicco_rows[vicco_rows["model"] == m]["score"]
            if len(v) > 0:
                vicco_means.append(v.mean())
        vicco_means = np.array(vicco_means)
        spread_v = median_pairwise_diff(vicco_means)

        for st in CSTIM_SETS:
            cstim_means = per_model_means(df, st, "controversial", models)
            spread_c = median_pairwise_diff(cstim_means)
            rows.append({
                "metric": metric, "model_set": st,
                "n_models": len(cstim_means),
                "vicco_median_pairwise_diff": spread_v,
                "cstim_median_pairwise_diff": spread_c,
                "spread_ratio_cstim_vs_vicco": (spread_c / spread_v) if spread_v else np.nan,
                "vicco_range": float(vicco_means.max() - vicco_means.min()) if len(vicco_means) else np.nan,
                "cstim_range": float(cstim_means.max() - cstim_means.min()) if len(cstim_means) else np.nan,
                "vicco_iqr": float(np.subtract(*np.percentile(vicco_means, [75, 25]))) if len(vicco_means) else np.nan,
                "cstim_iqr": float(np.subtract(*np.percentile(cstim_means, [75, 25]))) if len(cstim_means) else np.nan,
            })

    out = pd.DataFrame(rows)
    out_path = DATA_DIR / "empirical_spread.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
