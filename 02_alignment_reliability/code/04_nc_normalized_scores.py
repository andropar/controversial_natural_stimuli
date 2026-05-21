"""
10_compute_nc_normalized_scores.py

Compute noise-ceiling-normalized RSA scores.

For each subject × model_set × model, divide the RSA score by the
corresponding RDM noise ceiling (Spearman-Brown corrected split-half).
Then average across subjects.

Inputs:
    experiments/cstim_fmri_analysis/data/{sub}/crsa_scores.csv
    experiments/cstim_fmri_analysis/data/{sub}/wrsa_transfer_scores.csv
    experiments/cstim_fmri_analysis/data/rdm_noise_ceilings.csv

Outputs:
    experiments/cstim_fmri_analysis/data/nc_normalized_scores.csv
    experiments/cstim_fmri_analysis/data/nc_normalized_summary.csv

Usage:
    python 10_compute_nc_normalized_scores.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys

# Setup imports from cstim_paper root
_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER.parents[1]))  # project root for cstims

import config


METHODS = {
    "wrsa_transfer": ("wrsa_transfer_scores.csv", "wrsa_transfer"),
    "crsa": ("crsa_scores.csv", "crsa"),
}


def load_noise_ceilings() -> pd.DataFrame:
    """Load noise ceilings, computing mean vicco NC per subject."""
    nc = pd.read_csv(config.STATS_DATA_DIR / "rdm_noise_ceilings.csv")

    # Controversial: one NC per (subject, group)
    cstim_nc = nc[nc["stimulus_type"] == "controversial"][
        ["subject", "group", "noise_ceiling_spearman"]
    ].copy()
    cstim_nc = cstim_nc.rename(columns={"group": "model_set",
                                         "noise_ceiling_spearman": "nc"})

    # Vicco: mean NC across 10 bootstraps per subject
    vicco_nc = nc[nc["stimulus_type"] == "vicco"].groupby("subject").agg(
        nc=("noise_ceiling_spearman", "mean")
    ).reset_index()
    vicco_nc["model_set"] = "vicco"

    return pd.concat([cstim_nc, vicco_nc], ignore_index=True)


def main():
    nc_df = load_noise_ceilings()
    all_rows = []

    for method, (filename, score_col) in METHODS.items():
        for subject in config.SUBJECTS:
            path = config.RSA_DATA_DIR / subject / filename
            if not path.exists():
                continue
            scores = pd.read_csv(path)

            for _, row in scores.iterrows():
                model_set = row["model_set"]
                stim_type = row["stimulus_type"]

                # Look up NC: for controversial, use the model_set NC;
                # for vicco, use the mean vicco NC
                if stim_type == "controversial":
                    nc_match = nc_df[(nc_df["subject"] == subject) &
                                     (nc_df["model_set"] == model_set)]
                else:
                    nc_match = nc_df[(nc_df["subject"] == subject) &
                                     (nc_df["model_set"] == "vicco")]

                if nc_match.empty or nc_match["nc"].iloc[0] <= 0:
                    nc_val = np.nan
                    normalized = np.nan
                else:
                    nc_val = nc_match["nc"].iloc[0]
                    # The NC is reliability (ρ_xx); the upper bound on
                    # model-brain correlation is sqrt(ρ_xx) (Spearman 1904).
                    normalized = row[score_col] / np.sqrt(nc_val)

                all_rows.append({
                    "subject": subject,
                    "model_set": model_set,
                    "model": row["model"],
                    "display_name": row["display_name"],
                    "stimulus_type": stim_type,
                    "bootstrap_idx": row["bootstrap_idx"],
                    "method": method,
                    "score": row[score_col],
                    "noise_ceiling": nc_val,
                    "nc_normalized": normalized,
                })

    # Save full per-subject results
    full_df = pd.DataFrame(all_rows)
    full_path = config.STATS_DATA_DIR / "nc_normalized_scores.csv"
    full_df.to_csv(full_path, index=False)
    print(f"Saved {len(full_df)} rows to {full_path}")

    # Compute summary: cross-subject mean ± SEM for controversial scores
    # (one row per model × model_set × method)
    cstim_df = full_df[full_df["stimulus_type"] == "controversial"]
    summary = cstim_df.groupby(["model_set", "model", "display_name", "method"]).agg(
        score_mean=("score", "mean"),
        score_sem=("score", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        nc_normalized_mean=("nc_normalized", "mean"),
        nc_normalized_sem=("nc_normalized", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        noise_ceiling_mean=("noise_ceiling", "mean"),
        n_subjects=("subject", "nunique"),
    ).reset_index()

    # Also compute vicco baseline summary (mean across bootstraps first, then subjects)
    vicco_df = full_df[full_df["stimulus_type"] == "vicco"]
    if not vicco_df.empty:
        # Average across bootstraps per subject × model × method
        vicco_by_subj = vicco_df.groupby(
            ["subject", "model_set", "model", "display_name", "method"]
        ).agg(
            score=("score", "mean"),
            nc_normalized=("nc_normalized", "mean"),
            noise_ceiling=("noise_ceiling", "first"),
        ).reset_index()

        vicco_summary = vicco_by_subj.groupby(
            ["model_set", "model", "display_name", "method"]
        ).agg(
            score_mean=("score", "mean"),
            score_sem=("score", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            nc_normalized_mean=("nc_normalized", "mean"),
            nc_normalized_sem=("nc_normalized", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            noise_ceiling_mean=("noise_ceiling", "mean"),
            n_subjects=("subject", "nunique"),
        ).reset_index()

        vicco_summary["stimulus_type"] = "vicco"
        summary["stimulus_type"] = "controversial"
        summary = pd.concat([summary, vicco_summary], ignore_index=True)
    else:
        summary["stimulus_type"] = "controversial"

    summary_path = config.STATS_DATA_DIR / "nc_normalized_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary ({len(summary)} rows) to {summary_path}")

    # Print overview
    print(f"\n{'='*80}")
    print("NC-NORMALIZED SCORES (controversial, cross-subject mean)")
    print(f"{'='*80}")
    cstim_summary = summary[summary["stimulus_type"] == "controversial"]
    for method in ["wrsa_transfer", "crsa"]:
        print(f"\n  {method}:")
        method_df = cstim_summary[cstim_summary["method"] == method]
        for model_set in config.MODEL_SETS:
            ms_df = method_df[method_df["model_set"] == model_set].sort_values(
                "nc_normalized_mean", ascending=False
            )
            if ms_df.empty:
                continue
            print(f"    {model_set}:")
            for _, r in ms_df.iterrows():
                print(f"      {r['display_name']:20s} raw={r['score_mean']:.3f} "
                      f"normalized={r['nc_normalized_mean']:.3f} "
                      f"(NC={r['noise_ceiling_mean']:.3f})")


if __name__ == "__main__":
    main()
