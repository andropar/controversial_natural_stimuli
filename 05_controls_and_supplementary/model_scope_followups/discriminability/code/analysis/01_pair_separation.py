#!/usr/bin/env python3
"""Result 4(a): Baseline-tied pair separation.

For each model pair (A, B) and each cstim set:
    Δ_subj_baseline = mean(vicco_RSA_A) − mean(vicco_RSA_B) per subject
                      where mean is over 1000 vicco bootstraps at n=100.
    Δ_subj_cstim    = cstim_RSA_A − cstim_RSA_B per subject (n=100, single value).

Run paired Wilcoxon signed-rank test across the 5 subjects on Δ.
FDR-correct across pairs (within set, within metric).
A pair is "tied on baseline" if q_vicco > 0.05.
A pair is "near-ceiling" if both models have mean(vicco_RSA) > NEAR_CEILING_THRESH.
A pair is "separated on cstim" if q_cstim < 0.05.

Headline: of N pairs tied-and-near-ceiling on baseline, what fraction become
separated on cstim?

Done for both mRSA-transfer and fRSA, using the paper-layer values from
02_rsa_scores/data/{subject}/{wrsa_transfer,crsa}_scores.csv.

Output:
    scripts/claude/discriminability/data/pair_separation_{mrsa,frsa}.csv
    scripts/claude/discriminability/data/pair_separation_summary.csv
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

PROJECT = Path(__file__).resolve().parents[4]
PAPER = PROJECT / "experiments" / "cstim_paper"
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PROJECT))

from config import MODEL_SETS, MODEL_DISPLAY_NAMES, RSA_DATA_DIR, SUBJECTS  # noqa

DATA_DIR = Path(__file__).resolve().parents[1] / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
NEAR_CEILING_THRESH_MRSA = 0.30  # mean vicco mRSA threshold to call a model "near ceiling"
NEAR_CEILING_THRESH_FRSA = 0.15  # fRSA is on a tighter scale; pick threshold from data
ALPHA = 0.05


def load_scores_per_metric(metric: str) -> pd.DataFrame:
    """metric in {'mRSA', 'fRSA'}.

    Returns a long DataFrame with columns:
        subject, model, model_set, stimulus_type, bootstrap_idx, score
    """
    file_map = {"mRSA": ("wrsa_transfer_scores.csv", "wrsa_transfer"),
                "fRSA": ("crsa_scores.csv", "crsa")}
    fname, score_col = file_map[metric]
    dfs = []
    for s in SUBJECTS:
        p = RSA_DATA_DIR / s / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "subject" not in d.columns:
            d["subject"] = s
        d = d.rename(columns={score_col: "score"})
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def per_subject_alignments(df: pd.DataFrame, models: list) -> dict:
    """Returns:
        vicco_means[subject][model] = mean vicco RSA over 1000 bootstraps
        cstim[subject][set][model] = single cstim RSA (1 value)
    """
    vicco_means = {s: {} for s in df["subject"].unique()}
    cstim = {s: {st: {} for st in CSTIM_SETS} for s in df["subject"].unique()}
    for (s, m), sub in df.groupby(["subject", "model"]):
        if m not in models:
            continue
        # Vicco rows
        v_rows = sub[sub["stimulus_type"] == "vicco"]
        if len(v_rows) > 0:
            vicco_means[s][m] = v_rows["score"].mean()
        # Cstim rows by model_set
        c_rows = sub[sub["stimulus_type"] == "controversial"]
        for st in CSTIM_SETS:
            cs_rows = c_rows[c_rows["model_set"] == st]
            if len(cs_rows) > 0:
                cstim[s][st][m] = cs_rows["score"].iloc[0]  # single value
    return vicco_means, cstim


def pairwise_test(deltas: np.ndarray) -> float:
    """Paired t-test against zero (one-sample t-test on paired differences).

    Wilcoxon signed-rank cannot achieve p < 0.05 two-sided at n=5 (minimum
    achievable p is 0.0625), so it nullifies the analysis under FDR. Paired
    t is the standard for cohort-level neuroimaging comparisons at small n
    and produces arbitrary p-values. Two-sided. Returns NaN if len < 3 or
    all deltas zero.
    """
    deltas = np.asarray(deltas)
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) < 3:
        return np.nan
    if np.std(deltas) == 0:
        return 1.0
    try:
        # one-sample t against zero
        t = np.mean(deltas) / (np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
        from scipy.stats import t as student_t
        p = 2 * student_t.sf(abs(t), df=len(deltas) - 1)
    except Exception:
        return np.nan
    return float(p)


def run_pair_separation(metric: str, near_ceil_thresh: float) -> pd.DataFrame:
    df = load_scores_per_metric(metric)
    models = MODEL_SETS["all_models"]  # 20 models
    vicco_means, cstim = per_subject_alignments(df, models)

    rows = []
    for A, B in combinations(models, 2):
        # vicco deltas (using bootstrap means per subject)
        deltas_v = []
        means_v_A = []
        means_v_B = []
        for s in vicco_means:
            if A in vicco_means[s] and B in vicco_means[s]:
                deltas_v.append(vicco_means[s][A] - vicco_means[s][B])
                means_v_A.append(vicco_means[s][A])
                means_v_B.append(vicco_means[s][B])
        deltas_v = np.array(deltas_v)
        if len(deltas_v) == 0:
            continue
        p_vicco = pairwise_test(deltas_v)
        mean_v_A = np.mean(means_v_A)
        mean_v_B = np.mean(means_v_B)
        near_ceil = (mean_v_A > near_ceil_thresh) and (mean_v_B > near_ceil_thresh)

        for st in CSTIM_SETS:
            deltas_c = []
            for s in vicco_means:
                if A in cstim[s][st] and B in cstim[s][st]:
                    deltas_c.append(cstim[s][st][A] - cstim[s][st][B])
            deltas_c = np.array(deltas_c)
            if len(deltas_c) < 3:
                continue
            p_cstim = pairwise_test(deltas_c)
            rows.append({
                "metric": metric,
                "model_A": A, "model_B": B,
                "model_set": st,
                "n_subjects": len(deltas_c),
                "mean_vicco_A": mean_v_A,
                "mean_vicco_B": mean_v_B,
                "mean_delta_vicco": np.mean(deltas_v),
                "mean_delta_cstim": np.mean(deltas_c),
                "p_vicco": p_vicco,
                "p_cstim": p_cstim,
                "near_ceiling": near_ceil,
            })
    out = pd.DataFrame(rows)

    # FDR-correct within (metric, set) for vicco and cstim independently
    out["q_vicco"] = np.nan
    out["q_cstim"] = np.nan
    for st in CSTIM_SETS:
        mask = (out["model_set"] == st)
        if mask.sum() == 0:
            continue
        # vicco p-values are duplicated across cstim_sets within a metric;
        # use unique-per-pair vicco p-values for FDR correction (do once per metric)
        pass  # handled below with a unique-pair pass

    # Compute q_vicco once per metric using unique pairs (vicco doesn't depend on set)
    pair_to_pvicco = (out.drop_duplicates(subset=["model_A", "model_B"])
                        .set_index(["model_A", "model_B"])["p_vicco"])
    valid = pair_to_pvicco.dropna()
    if len(valid) > 0:
        _, q_vicco_arr, _, _ = multipletests(valid.values, alpha=ALPHA, method="fdr_bh")
        q_vicco_map = dict(zip(valid.index, q_vicco_arr))
        out["q_vicco"] = [q_vicco_map.get((a, b), np.nan) for a, b in zip(out["model_A"], out["model_B"])]

    for st in CSTIM_SETS:
        mask = (out["model_set"] == st) & out["p_cstim"].notna()
        if mask.sum() == 0:
            continue
        _, q, _, _ = multipletests(out.loc[mask, "p_cstim"].values, alpha=ALPHA, method="fdr_bh")
        out.loc[mask, "q_cstim"] = q

    out["tied_baseline"] = out["q_vicco"] > ALPHA
    out["separated_cstim"] = out["q_cstim"] < ALPHA
    out["tied_and_near_ceiling"] = out["tied_baseline"] & out["near_ceiling"]
    out["converted"] = out["tied_and_near_ceiling"] & out["separated_cstim"]
    return out


def main():
    summary_rows = []
    for metric, thresh in [("mRSA", NEAR_CEILING_THRESH_MRSA),
                            ("fRSA", NEAR_CEILING_THRESH_FRSA)]:
        print(f"\n=== {metric} (near-ceiling threshold = {thresh}) ===")
        out = run_pair_separation(metric, thresh)
        out_path = DATA_DIR / f"pair_separation_{metric.lower()}.csv"
        out.to_csv(out_path, index=False)
        print(f"  wrote {len(out)} rows -> {out_path}")

        for st in CSTIM_SETS:
            sub = out[out["model_set"] == st]
            n_pairs = len(sub)
            n_tied_strict = int(sub["tied_baseline"].sum())
            n_tied_nc = int(sub["tied_and_near_ceiling"].sum())
            n_sep_cstim = int(sub["separated_cstim"].sum())
            n_converted = int(sub["converted"].sum())
            row = {
                "metric": metric, "model_set": st,
                "n_pairs_total": n_pairs,
                "n_tied_strict": n_tied_strict,
                "n_tied_near_ceiling": n_tied_nc,
                "n_separated_cstim_total": n_sep_cstim,
                "n_converted_strict": int((sub["tied_baseline"] & sub["separated_cstim"]).sum()),
                "n_converted_near_ceiling": n_converted,
                "conversion_rate_strict_pct": (
                    100 * (sub["tied_baseline"] & sub["separated_cstim"]).sum() / max(n_tied_strict, 1)
                ),
                "conversion_rate_near_ceiling_pct": 100 * n_converted / max(n_tied_nc, 1),
            }
            summary_rows.append(row)
            print(f"  {st:<22} pairs={n_pairs:>3} | tied(strict)={n_tied_strict:>3} "
                  f"| tied(near-ceil)={n_tied_nc:>3} | sep(cstim)={n_sep_cstim:>3} "
                  f"| converted(near-ceil)={n_converted:>3} ({row['conversion_rate_near_ceiling_pct']:.1f}%)")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = DATA_DIR / "pair_separation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary -> {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
