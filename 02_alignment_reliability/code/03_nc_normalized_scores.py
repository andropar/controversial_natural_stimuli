"""
03_nc_normalized_scores.py

Compute noise-ceiling-normalized RSA scores.

For each subject × model_set × model, divide the RSA score by the
corresponding RDM noise ceiling (Spearman-Brown corrected split-half).
Then average across subjects.

Inputs:
    01_brain_model_alignment/results/rsa_scores/{sub}/crsa_scores.csv
    01_brain_model_alignment/results/rsa_scores/{sub}/wrsa_transfer_scores.csv
    02_alignment_reliability/results/rdm_noise_ceilings.csv

Outputs:
    02_alignment_reliability/results/nc_normalized_scores.csv
    02_alignment_reliability/results/nc_normalized_summary.csv

Usage:
    python 02_alignment_reliability/code/03_nc_normalized_scores.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parent
PAPER_HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(PAPER_HELPERS))

from cstims import constants, paths


METHODS = {
    "wrsa_transfer": ("wrsa_transfer_scores.csv", "wrsa_transfer"),
    "crsa": ("crsa_scores.csv", "crsa"),
}
STIMULUS_TYPES = ["controversial", "vicco"]
N_VICCO_SCORE_BOOTSTRAPS = 10


def load_noise_ceilings() -> pd.DataFrame:
    """Load noise ceilings, computing mean vicco NC per subject."""
    nc = pd.read_csv(paths.reliability_data_dir() / "rdm_noise_ceilings.csv")

    # Controversial: one NC per (subject, group)
    cstim_nc = nc[nc["stimulus_type"] == "controversial"][
        ["subject", "group", "noise_ceiling_spearman"]
    ].copy()
    cstim_nc = cstim_nc.rename(columns={"group": "model_set",
                                         "noise_ceiling_spearman": "nc"})

    # Vicco: mean NC across available reliability bootstraps per subject.
    vicco_nc = nc[nc["stimulus_type"] == "vicco"].groupby("subject").agg(
        nc=("noise_ceiling_spearman", "mean")
    ).reset_index()
    vicco_nc["model_set"] = "vicco"

    return pd.concat([cstim_nc, vicco_nc], ignore_index=True)


def normalize_score_table(
    scores: pd.DataFrame,
    subject: str,
    method: str,
    score_col: str,
    nc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach subject-level noise ceilings and compute normalized scores."""
    scores = scores[scores["stimulus_type"].isin(STIMULUS_TYPES)].copy()
    scores = scores[
        (scores["stimulus_type"] != "vicco")
        | (scores["bootstrap_idx"] < N_VICCO_SCORE_BOOTSTRAPS)
    ].copy()
    if scores.empty:
        return pd.DataFrame()

    scores["subject"] = subject
    scores["nc_model_set"] = np.where(
        scores["stimulus_type"] == "controversial",
        scores["model_set"],
        "vicco",
    )

    merged = scores.merge(
        nc_df.rename(columns={"model_set": "nc_model_set"}),
        on=["subject", "nc_model_set"],
        how="left",
    )
    valid_nc = merged["nc"].gt(0)
    merged["score"] = merged[score_col]
    merged["noise_ceiling"] = merged["nc"].where(valid_nc)
    merged["nc_normalized"] = np.where(
        valid_nc,
        merged["score"] / np.sqrt(merged["nc"]),
        np.nan,
    )
    merged["method"] = method

    return merged[
        [
            "subject",
            "model_set",
            "model",
            "display_name",
            "stimulus_type",
            "bootstrap_idx",
            "method",
            "score",
            "noise_ceiling",
            "nc_normalized",
        ]
    ]


def main():
    nc_df = load_noise_ceilings()
    frames = []

    for method, (filename, score_col) in METHODS.items():
        for subject in constants.SUBJECTS:
            path = paths.rsa_data_dir() / subject / filename
            if not path.exists():
                continue
            scores = pd.read_csv(path)
            frames.append(normalize_score_table(scores, subject, method, score_col, nc_df))

    # Save full per-subject results
    full_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    full_path = paths.reliability_data_dir() / "nc_normalized_scores.csv"
    full_path.parent.mkdir(parents=True, exist_ok=True)
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

    summary_path = paths.reliability_data_dir() / "nc_normalized_summary.csv"
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
        for model_set in constants.MODEL_SETS:
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
