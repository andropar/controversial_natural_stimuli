"""
06_compute_spread_statistics.py

Formal statistical tests on whether controversial stimuli amplify the spread
(variance/range) of model scores compared to baseline (vicco) stimuli.

Approach:
- For each model set × subject × method: compute score range and SD for
  controversial, and for each of 10 vicco bootstrap samples.
- Bootstrap CI on the spread difference (controversial - mean vicco spread).
- Levene's test for equality of variances.
- Report per-set and aggregated across subjects.

Inputs:
  experiments/cstim_fmri_analysis/data/{sub}/wrsa_transfer_scores.csv
  experiments/cstim_fmri_analysis/data/{sub}/crsa_scores.csv

Outputs:
  experiments/cstim_fmri_analysis/data/spread_statistics.csv
  experiments/cstim_fmri_analysis/data/spread_statistics_summary.csv

Usage:
  python 06_compute_spread_statistics.py
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))
from scipy import stats
from itertools import combinations


# Setup imports from cstim_paper root
_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root for cstims

from cstims.paper import config


METHODS = {
    "wrsa_transfer": "wrsa_transfer_scores.csv",
    "crsa": "crsa_scores.csv",
}


def load_scores(subject: str, method: str) -> pd.DataFrame:
    """Load score CSV for a subject and method."""
    filename = METHODS[method]
    path = config.RSA_DATA_DIR / subject / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_spread_metrics(scores: np.ndarray) -> dict:
    """Compute spread metrics for a vector of model scores."""
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    # Mean pairwise absolute difference (Gini mean difference)
    n = len(scores)
    pairwise_diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_diffs.append(abs(scores[i] - scores[j]))
    mean_pairwise_diff = np.mean(pairwise_diffs) if pairwise_diffs else 0.0
    median_pairwise_diff = np.median(pairwise_diffs) if pairwise_diffs else 0.0
    return {
        "range": np.max(scores) - np.min(scores),
        "std": std,
        "iqr": np.subtract(*np.percentile(scores, [75, 25])),
        "cv": std / mean if mean > 0 else np.nan,
        "mean_pairwise_diff": mean_pairwise_diff,
        "median_pairwise_diff": median_pairwise_diff,
    }


def analyze_spread(df: pd.DataFrame, method: str, score_col: str) -> list[dict]:
    """
    Analyze spread for all model sets in a single subject's data.

    Returns list of result dicts.
    """
    results = []
    subject = df["subject"].iloc[0]

    for model_set in df["model_set"].unique():
        ms_df = df[df["model_set"] == model_set]

        # Controversial scores (single set, bootstrap_idx=0)
        cstim_df = ms_df[ms_df["stimulus_type"] == "controversial"]
        if cstim_df.empty:
            continue
        cstim_scores = cstim_df[score_col].values
        cstim_spread = compute_spread_metrics(cstim_scores)

        # Vicco bootstrap scores
        vicco_df = ms_df[ms_df["stimulus_type"] == "vicco"]
        if vicco_df.empty:
            continue

        bootstrap_indices = sorted(vicco_df["bootstrap_idx"].unique())
        vicco_spreads = []
        vicco_all_scores = []
        for bidx in bootstrap_indices:
            boot_scores = vicco_df[vicco_df["bootstrap_idx"] == bidx][score_col].values
            vicco_spreads.append(compute_spread_metrics(boot_scores))
            vicco_all_scores.append(boot_scores)

        # Mean vicco spread
        vicco_ranges = [s["range"] for s in vicco_spreads]
        vicco_stds = [s["std"] for s in vicco_spreads]

        mean_vicco_range = np.mean(vicco_ranges)
        mean_vicco_std = np.mean(vicco_stds)

        # IQR, CV, pairwise diff
        vicco_iqrs = [s["iqr"] for s in vicco_spreads]
        vicco_cvs = [s["cv"] for s in vicco_spreads]
        vicco_pairwise_diffs = [s["mean_pairwise_diff"] for s in vicco_spreads]
        vicco_median_pairwise_diffs = [s["median_pairwise_diff"] for s in vicco_spreads]
        mean_vicco_iqr = np.mean(vicco_iqrs)
        mean_vicco_cv = np.nanmean(vicco_cvs)
        mean_vicco_pairwise_diff = np.mean(vicco_pairwise_diffs)
        mean_vicco_median_pairwise_diff = np.mean(vicco_median_pairwise_diffs)

        # Spread ratio
        range_ratio = cstim_spread["range"] / mean_vicco_range if mean_vicco_range > 0 else np.nan
        std_ratio = cstim_spread["std"] / mean_vicco_std if mean_vicco_std > 0 else np.nan
        iqr_ratio = cstim_spread["iqr"] / mean_vicco_iqr if mean_vicco_iqr > 0 else np.nan
        cv_ratio = cstim_spread["cv"] / mean_vicco_cv if mean_vicco_cv > 0 else np.nan
        pairwise_diff_ratio = (cstim_spread["mean_pairwise_diff"] / mean_vicco_pairwise_diff
                               if mean_vicco_pairwise_diff > 0 else np.nan)
        median_pairwise_diff_ratio = (cstim_spread["median_pairwise_diff"] / mean_vicco_median_pairwise_diff
                                      if mean_vicco_median_pairwise_diff > 0 else np.nan)

        # Levene's test: controversial scores vs mean vicco scores
        vicco_mean_scores = np.mean(np.array(vicco_all_scores), axis=0)
        levene_stat, levene_p = stats.levene(cstim_scores, vicco_mean_scores)

        # Per-metric one-sample t-tests on whether diffs > 0
        def _onesided_ttest(diffs):
            if len(diffs) > 1:
                t, p = stats.ttest_1samp(diffs, 0)
                return t, p / 2 if t > 0 else 1 - p / 2
            return np.nan, np.nan

        range_diffs = [cstim_spread["range"] - vr for vr in vicco_ranges]
        iqr_diffs = [cstim_spread["iqr"] - vi for vi in vicco_iqrs]
        cv_diffs = [cstim_spread["cv"] - vc for vc in vicco_cvs if not np.isnan(vc)]
        pairwise_diffs_metric = [cstim_spread["mean_pairwise_diff"] - vp
                                 for vp in vicco_pairwise_diffs]
        median_pairwise_diffs_metric = [cstim_spread["median_pairwise_diff"] - vp
                                        for vp in vicco_median_pairwise_diffs]

        ttest_range_t, ttest_range_p = _onesided_ttest(range_diffs)
        ttest_iqr_t, ttest_iqr_p = _onesided_ttest(iqr_diffs)
        ttest_cv_t, ttest_cv_p = _onesided_ttest(cv_diffs)
        ttest_pairwise_t, ttest_pairwise_p = _onesided_ttest(pairwise_diffs_metric)
        ttest_median_pairwise_t, ttest_median_pairwise_p = _onesided_ttest(median_pairwise_diffs_metric)

        n_models = len(cstim_scores)

        results.append({
            "subject": subject,
            "model_set": model_set,
            "method": method,
            "n_models": n_models,
            # Controversial spread
            "cstim_range": cstim_spread["range"],
            "cstim_std": cstim_spread["std"],
            "cstim_iqr": cstim_spread["iqr"],
            "cstim_cv": cstim_spread["cv"],
            "cstim_mean_pairwise_diff": cstim_spread["mean_pairwise_diff"],
            "cstim_median_pairwise_diff": cstim_spread["median_pairwise_diff"],
            # Vicco spread (mean across bootstraps)
            "vicco_range_mean": mean_vicco_range,
            "vicco_range_std": np.std(vicco_ranges, ddof=1),
            "vicco_std_mean": mean_vicco_std,
            "vicco_std_std": np.std(vicco_stds, ddof=1),
            "vicco_iqr_mean": mean_vicco_iqr,
            "vicco_iqr_std": np.std(vicco_iqrs, ddof=1),
            "vicco_cv_mean": mean_vicco_cv,
            "vicco_cv_std": np.std(vicco_cvs, ddof=1),
            "vicco_pairwise_diff_mean": mean_vicco_pairwise_diff,
            "vicco_pairwise_diff_std": np.std(vicco_pairwise_diffs, ddof=1),
            "vicco_median_pairwise_diff_mean": mean_vicco_median_pairwise_diff,
            "vicco_median_pairwise_diff_std": np.std(vicco_median_pairwise_diffs, ddof=1),
            # Ratios
            "range_ratio": range_ratio,
            "std_ratio": std_ratio,
            "iqr_ratio": iqr_ratio,
            "cv_ratio": cv_ratio,
            "pairwise_diff_ratio": pairwise_diff_ratio,
            "median_pairwise_diff_ratio": median_pairwise_diff_ratio,
            # Statistical tests
            "levene_stat": levene_stat,
            "levene_p": levene_p,
            "ttest_range_t": ttest_range_t,
            "ttest_range_p_onesided": ttest_range_p,
            "ttest_iqr_t": ttest_iqr_t,
            "ttest_iqr_p_onesided": ttest_iqr_p,
            "ttest_cv_t": ttest_cv_t,
            "ttest_cv_p_onesided": ttest_cv_p,
            "ttest_pairwise_t": ttest_pairwise_t,
            "ttest_pairwise_p_onesided": ttest_pairwise_p,
            "ttest_median_pairwise_t": ttest_median_pairwise_t,
            "ttest_median_pairwise_p_onesided": ttest_median_pairwise_p,
        })

    return results


def _cross_subject_ttest(ratios: np.ndarray) -> float:
    """One-sample one-sided t-test on per-subject ratios vs 1.0.

    Tests H1: ratio > 1 (controversial spread > baseline spread).
    Returns one-sided p-value.
    """
    ratios = ratios[~np.isnan(ratios)]
    if len(ratios) < 2:
        return np.nan
    t, p = stats.ttest_1samp(ratios, 1.0)
    return p / 2 if t > 0 else 1 - p / 2


def compute_cross_subject_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate spread statistics across subjects.

    For each model_set × method, compute:
    - Mean spread ratio across subjects
    - Per-metric one-sample t-test on per-subject ratios vs 1.0
      (tests whether ratio is consistently > 1 across subjects)
    """
    summary_rows = []

    for (model_set, method), grp in results_df.groupby(["model_set", "method"]):
        n_subjects = len(grp)

        ratio_cols = {
            "range": "range_ratio",
            "iqr": "iqr_ratio",
            "cv": "cv_ratio",
            "pairwise": "pairwise_diff_ratio",
            "median_pairwise": "median_pairwise_diff_ratio",
        }

        row = {
            "model_set": model_set,
            "method": method,
            "n_subjects": n_subjects,
            "mean_range_ratio": grp["range_ratio"].mean(),
            "mean_std_ratio": grp["std_ratio"].mean(),
            "mean_pairwise_diff_ratio": grp["pairwise_diff_ratio"].mean(),
            "mean_median_pairwise_diff_ratio": grp["median_pairwise_diff_ratio"].mean(),
        }

        for metric_name, col in ratio_cols.items():
            row[f"ttest_{metric_name}_p"] = _cross_subject_ttest(grp[col].values)

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description="Compute spread statistics")
    parser.add_argument("--subjects", nargs="+", default=config.SUBJECTS)
    args = parser.parse_args()

    all_results = []

    for subject in args.subjects:
        print(f"\n{'='*60}")
        print(f"Processing {subject}")
        print(f"{'='*60}")

        for method, score_col in [("wrsa_transfer", "wrsa_transfer"), ("crsa", "crsa")]:
            df = load_scores(subject, method)
            if df.empty:
                print(f"  No {method} data for {subject}, skipping")
                continue

            results = analyze_spread(df, method, score_col)
            all_results.extend(results)

            for r in results:
                print(f"  {method} | {r['model_set']:25s} | "
                      f"range ratio={r['range_ratio']:.2f} | "
                      f"cv ratio={r['cv_ratio']:.2f} | "
                      f"iqr ratio={r['iqr_ratio']:.2f}")

    # Save detailed results
    results_df = pd.DataFrame(all_results)
    output_path = config.STATS_DATA_DIR / "spread_statistics.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to {output_path}")

    # Compute cross-subject summary
    summary_df = compute_cross_subject_summary(results_df)
    summary_path = config.STATS_DATA_DIR / "spread_statistics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("CROSS-SUBJECT SUMMARY")
    print(f"{'='*80}")
    for _, row in summary_df.iterrows():
        metrics = ["range", "iqr", "cv", "pairwise"]
        sigs = []
        for m in metrics:
            p = row[f"ttest_{m}_p"]
            s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            sigs.append(f"{m}={s}(p={p:.3f})")
        print(f"  {row['method']:15s} | {row['model_set']:25s} | "
              f"range ratio={row['mean_range_ratio']:.2f} | "
              f"{', '.join(sigs)}")


if __name__ == "__main__":
    main()
