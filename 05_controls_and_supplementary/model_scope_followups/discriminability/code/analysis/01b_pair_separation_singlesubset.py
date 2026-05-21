#!/usr/bin/env python3
"""Sanity check for 01_pair_separation: use a SINGLE random 100-image vicco
subset (not the mean over 1000 bootstraps) as the baseline.

This equalizes sampling noise between cstim and vicco baseline (cstim is a
single 100-image set; vicco-mean of 1000 bootstraps has effectively-zero
sampling noise, which gives baseline an unfair advantage in distinguishing
model pairs). Repeat for K random subsets and average.

Output:
    data/pair_separation_singlesubset_{mrsa,frsa}.csv
    data/pair_separation_singlesubset_summary.csv
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests

PROJECT = Path(__file__).resolve().parents[4]
PAPER = PROJECT / "experiments" / "cstim_paper"
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PROJECT))

from config import MODEL_SETS, RSA_DATA_DIR, SUBJECTS  # noqa

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
NEAR_CEILING_THRESH_MRSA = 0.30
NEAR_CEILING_THRESH_FRSA = 0.15
ALPHA = 0.05
N_SUBSETS = 100  # vicco subsets to average over


def paired_t(deltas):
    deltas = np.asarray(deltas)
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) < 3 or np.std(deltas) == 0:
        return np.nan
    t = np.mean(deltas) / (np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
    return float(2 * student_t.sf(abs(t), df=len(deltas) - 1))


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


def run(metric, near_thresh):
    df = load_metric(metric)
    models = MODEL_SETS["all_models"]
    subjects = sorted(df["subject"].unique())

    # vicco rows: per (subject, model, bootstrap_idx)
    vicco = df[df["stimulus_type"] == "vicco"][[
        "subject", "model", "bootstrap_idx", "score"]].copy()
    n_vicco_boots = vicco["bootstrap_idx"].nunique()
    print(f"  {metric}: {n_vicco_boots} vicco bootstraps available")

    # cstim rows
    cstim = df[df["stimulus_type"] == "controversial"]

    # Pre-pivot vicco: (subject, model) -> dict[boot_idx] -> score
    vicco_idx = {(s, m): {} for s in subjects for m in models}
    for _, r in vicco.iterrows():
        if r["model"] in models:
            vicco_idx[(r["subject"], r["model"])][r["bootstrap_idx"]] = r["score"]

    # Pre-pivot cstim: (subject, set, model) -> score
    cstim_idx = {}
    for _, r in cstim.iterrows():
        if r["model"] in models:
            cstim_idx[(r["subject"], r["model_set"], r["model"])] = r["score"]

    # Pick K random vicco bootstrap indices
    rng = np.random.default_rng(0)
    chosen_boots = rng.choice(n_vicco_boots, size=min(N_SUBSETS, n_vicco_boots),
                              replace=False)

    # For each chosen vicco subset, run pair separation analysis
    pair_results_per_subset = []
    for boot_k in chosen_boots:
        rows = []
        for A, B in combinations(models, 2):
            # vicco deltas at this bootstrap
            deltas_v, vA, vB = [], [], []
            for s in subjects:
                a = vicco_idx[(s, A)].get(boot_k)
                b = vicco_idx[(s, B)].get(boot_k)
                if a is not None and b is not None:
                    deltas_v.append(a - b)
                    vA.append(a)
                    vB.append(b)
            if len(deltas_v) < 3:
                continue
            p_v = paired_t(np.array(deltas_v))
            mean_vA = float(np.mean(vA))
            mean_vB = float(np.mean(vB))
            near_ceil = (mean_vA > near_thresh) and (mean_vB > near_thresh)

            for st in CSTIM_SETS:
                deltas_c = []
                for s in subjects:
                    a = cstim_idx.get((s, st, A))
                    b = cstim_idx.get((s, st, B))
                    if a is not None and b is not None:
                        deltas_c.append(a - b)
                if len(deltas_c) < 3:
                    continue
                rows.append({
                    "boot_k": int(boot_k),
                    "model_A": A, "model_B": B, "model_set": st,
                    "p_vicco": p_v, "p_cstim": paired_t(np.array(deltas_c)),
                    "mean_vicco_A": mean_vA, "mean_vicco_B": mean_vB,
                    "near_ceiling": near_ceil,
                })
        df_pairs = pd.DataFrame(rows)
        # FDR within boot_k for vicco (unique pairs) and within (boot_k, set) for cstim
        unique_v = df_pairs.drop_duplicates(["model_A", "model_B"])[
            ["model_A", "model_B", "p_vicco"]]
        if unique_v["p_vicco"].notna().any():
            mask = unique_v["p_vicco"].notna()
            _, q_v, _, _ = multipletests(unique_v.loc[mask, "p_vicco"].values,
                                          alpha=ALPHA, method="fdr_bh")
            qmap = dict(zip(unique_v.loc[mask].set_index(["model_A", "model_B"]).index, q_v))
            df_pairs["q_vicco"] = [qmap.get((a, b), np.nan)
                                    for a, b in zip(df_pairs["model_A"], df_pairs["model_B"])]
        df_pairs["q_cstim"] = np.nan
        for st in CSTIM_SETS:
            mask = (df_pairs["model_set"] == st) & df_pairs["p_cstim"].notna()
            if mask.sum() == 0:
                continue
            _, q, _, _ = multipletests(df_pairs.loc[mask, "p_cstim"].values,
                                        alpha=ALPHA, method="fdr_bh")
            df_pairs.loc[mask, "q_cstim"] = q
        df_pairs["tied_baseline"] = df_pairs["q_vicco"] > ALPHA
        df_pairs["separated_cstim"] = df_pairs["q_cstim"] < ALPHA
        df_pairs["tied_and_near_ceiling"] = df_pairs["tied_baseline"] & df_pairs["near_ceiling"]
        df_pairs["converted"] = df_pairs["tied_and_near_ceiling"] & df_pairs["separated_cstim"]
        pair_results_per_subset.append(df_pairs)

    full = pd.concat(pair_results_per_subset, ignore_index=True)
    full.to_csv(DATA_DIR / f"pair_separation_singlesubset_{metric.lower()}.csv",
                index=False)

    # Aggregate across bootstrap subsets for the summary
    summary = []
    for st in CSTIM_SETS:
        sub = full[full["model_set"] == st]
        agg = sub.groupby("boot_k").agg(
            n_tied_strict=("tied_baseline", "sum"),
            n_tied_near_ceiling=("tied_and_near_ceiling", "sum"),
            n_separated=("separated_cstim", "sum"),
            n_converted_near_ceiling=("converted", "sum"),
        )
        agg["conversion_rate_pct"] = 100 * agg["n_converted_near_ceiling"] / agg["n_tied_near_ceiling"].replace(0, np.nan)
        summary.append({
            "metric": metric, "model_set": st,
            "n_subsets": len(agg),
            "n_tied_strict_mean": float(agg["n_tied_strict"].mean()),
            "n_tied_strict_std": float(agg["n_tied_strict"].std(ddof=1)),
            "n_tied_near_ceiling_mean": float(agg["n_tied_near_ceiling"].mean()),
            "n_separated_cstim_mean": float(agg["n_separated"].mean()),
            "n_converted_near_ceiling_mean": float(agg["n_converted_near_ceiling"].mean()),
            "n_converted_near_ceiling_std": float(agg["n_converted_near_ceiling"].std(ddof=1)),
            "conversion_rate_pct_mean": float(agg["conversion_rate_pct"].mean()),
            "conversion_rate_pct_std": float(agg["conversion_rate_pct"].std(ddof=1)),
        })
    return summary


def main():
    all_summary = []
    for metric, thresh in [("mRSA", NEAR_CEILING_THRESH_MRSA),
                            ("fRSA", NEAR_CEILING_THRESH_FRSA)]:
        print(f"\n=== {metric} ===")
        s = run(metric, thresh)
        all_summary.extend(s)
        for row in s:
            print(f"  {row['model_set']:<22} "
                  f"tied(NC) avg = {row['n_tied_near_ceiling_mean']:5.1f}  "
                  f"converted avg = {row['n_converted_near_ceiling_mean']:5.1f} ± "
                  f"{row['n_converted_near_ceiling_std']:.1f}  "
                  f"({row['conversion_rate_pct_mean']:.1f}% ± {row['conversion_rate_pct_std']:.1f}%)")

    df = pd.DataFrame(all_summary)
    out = DATA_DIR / "pair_separation_singlesubset_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
