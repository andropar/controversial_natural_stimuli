#!/usr/bin/env python3
"""
Supplementary statistics for the results summary:

1. Within-set discrimination using all-models stimuli (Section 3 addition):
   Tests whether all-models simulation distances predict within-set fMRI
   pairwise score differences when measured on all-models stimuli.

2. Effect sizes (Cohen's d) for brain alignment drop (Section 2 addition):
   Paired effect sizes for controversial vs. baseline score differences.

3. Multiple comparisons summary for permutation tests.

Inputs:
    experiments/simulation_to_fmri/data/option_b_pairwise.csv
    experiments/cstim_fmri_analysis/data/{subject}/wrsa_transfer_scores.csv
    experiments/cstim_fmri_analysis/data/{subject}/crsa_scores.csv
    experiments/cstim_fmri_analysis/data/permutation_test_results.csv

Outputs:
    data/within_set_allmodels_stimuli.csv
    data/effect_sizes_brain_alignment.csv

Usage:
    python 12_supplementary_statistics.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Setup imports from cstim_paper root
_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root for cstims

import config

SIM_DATA_DIR = config.SIM_DATA_DIR

PER_SET_NAMES = ["sota", "architecture", "training_objective", "dataset"]


# =============================================================================
# 1. Within-set discrimination using all-models stimuli
# =============================================================================

def within_set_allmodels_analysis():
    """
    For each per-set model group, test whether all-models simulation distances
    predict within-set pairwise fMRI score differences when measured on
    all-models stimuli.
    """
    pairwise_path = SIM_DATA_DIR / "option_b_pairwise.csv"
    pairwise_df = pd.read_csv(pairwise_path)

    # All-models encoding-track data
    allm = pairwise_df[
        (pairwise_df["model_set"] == "all_models") &
        (pairwise_df["track_type"] == "encoding")
    ].copy()

    # Parse pairs into (model_i, model_j)
    pair_split = allm["pair"].str.split(r"\|\|", regex=True)
    allm["model_i"] = pair_split.str[0]
    allm["model_j"] = pair_split.str[1]

    # Per-set own-stimuli data (for comparison)
    per_set_own = pairwise_df[
        pairwise_df["model_set"].isin(PER_SET_NAMES) &
        (pairwise_df["track_type"] == "encoding")
    ].copy()

    results = []

    for model_set in PER_SET_NAMES:
        set_models = set(config.MODEL_SETS[model_set])

        for subject in config.SUBJECTS:
            # All-models stimuli: subset to within-set pairs
            subj_allm = allm[allm["subject"] == subject]
            within_set = subj_allm[
                subj_allm["model_i"].isin(set_models) &
                subj_allm["model_j"].isin(set_models)
            ]

            if len(within_set) < 3:
                continue

            r_allm, p_allm = stats.spearmanr(
                within_set["sim_distance"], within_set["fmri_score_diff"]
            )

            # Per-set own stimuli (for comparison)
            subj_own = per_set_own[
                (per_set_own["model_set"] == model_set) &
                (per_set_own["subject"] == subject)
            ]
            r_own, p_own = np.nan, np.nan
            if len(subj_own) >= 3:
                r_own, p_own = stats.spearmanr(
                    subj_own["sim_distance"], subj_own["fmri_score_diff"]
                )

            results.append({
                "model_set": model_set,
                "subject": subject,
                "n_pairs": len(within_set),
                "rho_allmodels_stim": r_allm,
                "p_allmodels_stim": p_allm,
                "rho_own_stim": r_own,
                "p_own_stim": p_own,
            })

    df = pd.DataFrame(results)
    if df.empty:
        print("WARNING: No within-set pairs found")
        return df

    # Add cross-subject aggregates
    agg_rows = []
    for model_set in PER_SET_NAMES:
        ms_df = df[df["model_set"] == model_set]
        if len(ms_df) == 0:
            continue

        # Fisher's method for combining p-values
        for stim_type, r_col, p_col in [
            ("allmodels_stim", "rho_allmodels_stim", "p_allmodels_stim"),
            ("own_stim", "rho_own_stim", "p_own_stim"),
        ]:
            ps = ms_df[p_col].dropna().values
            rs = ms_df[r_col].dropna().values
            if len(ps) == 0:
                continue
            ps_clipped = np.clip(ps, 1e-300, 1.0)
            chi2 = -2 * np.sum(np.log(ps_clipped))
            fisher_p = 1 - stats.chi2.cdf(chi2, df=2 * len(ps))

            agg_rows.append({
                "model_set": model_set,
                "subject": "AGGREGATE",
                "n_pairs": int(ms_df["n_pairs"].iloc[0]),
                "rho_allmodels_stim": np.mean(rs) if stim_type == "allmodels_stim" else np.nan,
                "p_allmodels_stim": fisher_p if stim_type == "allmodels_stim" else np.nan,
                "rho_own_stim": np.mean(rs) if stim_type == "own_stim" else np.nan,
                "p_own_stim": fisher_p if stim_type == "own_stim" else np.nan,
            })

    if agg_rows:
        # Merge the two aggregate rows per model_set
        agg_df = pd.DataFrame(agg_rows)
        merged_aggs = []
        for model_set in PER_SET_NAMES:
            ms_agg = agg_df[agg_df["model_set"] == model_set]
            if len(ms_agg) == 0:
                continue
            row = {"model_set": model_set, "subject": "AGGREGATE"}
            for _, r in ms_agg.iterrows():
                for col in ["n_pairs", "rho_allmodels_stim", "p_allmodels_stim",
                            "rho_own_stim", "p_own_stim"]:
                    if not pd.isna(r[col]):
                        row[col] = r[col]
            merged_aggs.append(row)
        df = pd.concat([df, pd.DataFrame(merged_aggs)], ignore_index=True)

    return df


# =============================================================================
# 2. Effect sizes for brain alignment drop
# =============================================================================

def compute_effect_sizes():
    """
    For each model_set × method, compute paired Cohen's d for the
    controversial vs. baseline score drop.

    Unit of analysis: models. For each model, compute cross-subject mean
    controversial score and cross-subject mean baseline score (mean of 10
    Vicco bootstraps). Cohen's d is paired across models.
    """
    results = []

    for method, score_col, filename in [
        ("mRSA", "wrsa_transfer", "wrsa_transfer_scores.csv"),
        ("fRSA", "crsa", "crsa_scores.csv"),
    ]:
        # Load all subjects
        dfs = []
        for subject in config.SUBJECTS:
            path = config.get_subject_data_dir(subject) / filename
            if path.exists():
                dfs.append(pd.read_csv(path))
        if not dfs:
            continue
        all_scores = pd.concat(dfs, ignore_index=True)

        for model_set in ["sota", "architecture", "training_objective", "dataset", "all_models"]:
            ms_data = all_scores[all_scores["model_set"] == model_set]
            if ms_data.empty:
                continue

            models = config.MODEL_SETS[model_set]

            # For each model: cross-subject mean controversial and baseline
            cstim_means = []
            vicco_means = []
            for model in models:
                m_data = ms_data[ms_data["model"] == model]
                if m_data.empty:
                    continue

                # Controversial: single value per subject, average across subjects
                cstim = m_data[m_data["stimulus_type"] == "controversial"][score_col]
                if cstim.empty:
                    continue

                # Vicco: mean of 10 bootstraps per subject, then average across subjects
                vicco = m_data[m_data["stimulus_type"] == "vicco"].groupby("subject")[score_col].mean()

                cstim_mean = cstim.mean()
                vicco_mean = vicco.mean()

                cstim_means.append(cstim_mean)
                vicco_means.append(vicco_mean)

            if len(cstim_means) < 3:
                continue

            cstim_arr = np.array(cstim_means)
            vicco_arr = np.array(vicco_means)
            diffs = cstim_arr - vicco_arr  # negative = drop

            # Paired Cohen's d
            d = np.mean(diffs) / np.std(diffs, ddof=1)

            # Percent drop
            pct_drop = 100 * (1 - np.mean(cstim_arr) / np.mean(vicco_arr))

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(cstim_arr, vicco_arr)

            results.append({
                "model_set": model_set,
                "method": method,
                "n_models": len(cstim_means),
                "mean_controversial": np.mean(cstim_arr),
                "mean_baseline": np.mean(vicco_arr),
                "pct_drop": pct_drop,
                "cohens_d": d,
                "t_stat": t_stat,
                "p_value": p_val,
            })

    return pd.DataFrame(results)


# =============================================================================
# 3. Multiple comparisons check
# =============================================================================

def check_multiple_comparisons():
    """Check if permutation test p-values survive Bonferroni correction."""
    perm_path = config.STATS_DATA_DIR / "permutation_test_results.csv"
    if not perm_path.exists():
        print("No permutation test results found.")
        return None

    df = pd.read_csv(perm_path)
    n_tests = len(df)

    # Bonferroni threshold
    alpha = 0.05
    bonf_threshold = alpha / n_tests

    df["bonferroni_threshold"] = bonf_threshold
    df["survives_bonferroni"] = df["p_perm"] < bonf_threshold

    n_sig_uncorrected = (df["p_perm"] < 0.05).sum()
    n_sig_bonferroni = df["survives_bonferroni"].sum()

    print(f"\nMultiple comparisons ({n_tests} tests):")
    print(f"  Bonferroni threshold: {bonf_threshold:.5f}")
    print(f"  Significant (uncorrected p < 0.05): {n_sig_uncorrected}/{n_tests}")
    print(f"  Significant (Bonferroni): {n_sig_bonferroni}/{n_tests}")

    return df


# =============================================================================
# Main
# =============================================================================

def main():
    config.STATS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Within-set discrimination using all-models stimuli
    print("=" * 60)
    print("1. Within-set discrimination using all-models stimuli")
    print("=" * 60)
    within_df = within_set_allmodels_analysis()
    out = config.STATS_DATA_DIR / "within_set_allmodels_stimuli.csv"
    within_df.to_csv(out, index=False)
    print(f"Saved {out}")

    print("\nResults (encoding track, within-set pairs from all-models data):")
    print(f"{'Set':<22} {'Subject':<12} {'n':>4} "
          f"{'ρ(allm stim)':>14} {'p':>10} "
          f"{'ρ(own stim)':>14} {'p':>10}")
    print("-" * 90)
    for _, row in within_df.sort_values(["model_set", "subject"]).iterrows():
        rho_a = f"{row['rho_allmodels_stim']:.3f}" if not pd.isna(row['rho_allmodels_stim']) else "N/A"
        p_a = f"{row['p_allmodels_stim']:.3e}" if not pd.isna(row['p_allmodels_stim']) else "N/A"
        rho_o = f"{row['rho_own_stim']:.3f}" if not pd.isna(row['rho_own_stim']) else "N/A"
        p_o = f"{row['p_own_stim']:.3e}" if not pd.isna(row['p_own_stim']) else "N/A"
        print(f"{row['model_set']:<22} {row['subject']:<12} {int(row['n_pairs']):>4} "
              f"{rho_a:>14} {p_a:>10} "
              f"{rho_o:>14} {p_o:>10}")

    # 2. Effect sizes
    print("\n" + "=" * 60)
    print("2. Effect sizes for brain alignment drop")
    print("=" * 60)
    effect_df = compute_effect_sizes()
    out = config.STATS_DATA_DIR / "effect_sizes_brain_alignment.csv"
    effect_df.to_csv(out, index=False)
    print(f"Saved {out}")

    print(f"\n{'Set':<22} {'Method':<8} {'n':>4} "
          f"{'% drop':>8} {'Cohen d':>10} {'t':>8} {'p':>10}")
    print("-" * 80)
    for _, row in effect_df.iterrows():
        print(f"{row['model_set']:<22} {row['method']:<8} {int(row['n_models']):>4} "
              f"{row['pct_drop']:>7.1f}% {row['cohens_d']:>10.2f} "
              f"{row['t_stat']:>8.2f} {row['p_value']:>10.3e}")

    # 3. Multiple comparisons
    print("\n" + "=" * 60)
    print("3. Multiple comparisons check")
    print("=" * 60)
    perm_df = check_multiple_comparisons()
    if perm_df is not None:
        # Show which tests survive
        sig = perm_df[perm_df["survives_bonferroni"]]
        nonsig = perm_df[~perm_df["survives_bonferroni"] & (perm_df["p_perm"] < 0.05)]
        print(f"\n  Tests surviving Bonferroni:")
        for _, row in sig.iterrows():
            print(f"    {row['model_set']:>22} {row['method']:>15} {row['metric']:>20} "
                  f"ratio={row['observed_ratio']:.2f} p={row['p_perm']:.5f}")
        if len(nonsig) > 0:
            print(f"\n  Significant uncorrected but NOT after Bonferroni:")
            for _, row in nonsig.iterrows():
                print(f"    {row['model_set']:>22} {row['method']:>15} {row['metric']:>20} "
                      f"ratio={row['observed_ratio']:.2f} p={row['p_perm']:.5f}")


if __name__ == "__main__":
    main()
