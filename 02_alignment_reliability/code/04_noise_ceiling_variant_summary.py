"""
04_noise_ceiling_variant_summary.py

Build a compact subject-level summary comparing raw alignment scores with
multiple noise-ceiling normalizations.

Inputs:
  01_brain_model_alignment/results/rsa_scores/{sub}/crsa_scores.csv
  01_brain_model_alignment/results/rsa_scores/{sub}/wrsa_transfer_scores.csv
  02_alignment_reliability/results/rdm_noise_ceilings.csv
  02_alignment_reliability/results/between_subject_noise_ceilings.csv

Outputs:
  02_alignment_reliability/results/noise_ceiling_variant_summary.csv

Usage:
  python 02_alignment_reliability/code/04_noise_ceiling_variant_summary.py
"""

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

STAGE = Path(__file__).resolve().parents[1]
SHARE_ROOT = STAGE.parent
PAPER_HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(PAPER_HELPERS))

from cstims import constants, paths


DATA = paths.reliability_data_dir()
RSA = paths.rsa_data_dir()
SUBJECTS = constants.SUBJECTS
MODEL_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
METHOD_LABEL = {"wrsa_transfer": "mixed_RSA", "crsa": "fixed_RSA"}
SCORE_COL = {"wrsa_transfer": "wrsa_transfer", "crsa": "crsa"}
STIMULUS_TYPES = ["controversial", "vicco"]


def load_score_table(method: str) -> pd.DataFrame:
    frames = []
    for subject in SUBJECTS:
        path = RSA / subject / f"{method}_scores.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["subject"] = subject
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_nc_lookup() -> pd.DataFrame:
    nc = pd.read_csv(DATA / "rdm_noise_ceilings.csv")
    rows = []
    for _, r in nc.iterrows():
        group = "same_session" if r["stimulus_type"] == "vicco" else r["group"]
        within_rsb = float(r["noise_ceiling_spearman"])
        rows.append(
            {
                "subject": r["subject"],
                "group": group,
                "stimulus_type": r["stimulus_type"],
                "bootstrap_idx": r["bootstrap_idx"],
                "within_rSB": within_rsb,
                "within_sqrt_rSB": math.sqrt(max(within_rsb, 0.0)),
            }
        )
    out = pd.DataFrame(rows)
    base = (
        out[out["stimulus_type"] == "vicco"]
        .groupby("subject", as_index=False)
        .agg(within_rSB=("within_rSB", "mean"), within_sqrt_rSB=("within_sqrt_rSB", "mean"))
    )
    base["group"] = "same_session"
    base["stimulus_type"] = "vicco"

    cstim = out[out["stimulus_type"] == "controversial"].copy()
    cstim = cstim[cstim["bootstrap_idx"] == 0]
    return pd.concat([cstim, base], ignore_index=True)


def build_noise_ceiling_variant_summary() -> pd.DataFrame:
    rows = []
    nc_lookup = load_nc_lookup()
    between = pd.read_csv(DATA / "between_subject_noise_ceilings.csv")
    between = between.rename(columns={"group": "between_group", "stimulus_type": "between_type"})

    score_frames = []
    for method in ["wrsa_transfer", "crsa"]:
        scores = load_score_table(method)
        if scores.empty:
            continue

        score_col = SCORE_COL[method]
        scores = scores[
            scores["model_set"].isin(MODEL_SETS)
            & scores["stimulus_type"].isin(STIMULUS_TYPES)
        ].copy()
        scores["baseline_or_stimulus"] = np.where(
            scores["stimulus_type"] == "vicco", "same_session", "controversial"
        )
        grouped = (
            scores.groupby(["subject", "model_set", "model", "baseline_or_stimulus"], as_index=False)
            .agg(score=(score_col, "mean"))
        )
        grouped["metric"] = METHOD_LABEL[method]
        score_frames.append(grouped)

    score_df = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    if score_df.empty:
        return score_df

    score_df["nc_group"] = np.where(
        score_df["baseline_or_stimulus"] == "same_session",
        "same_session",
        score_df["model_set"],
    )
    score_df = score_df.merge(
        nc_lookup[["subject", "group", "within_rSB", "within_sqrt_rSB"]].rename(
            columns={"group": "nc_group"}
        ),
        on=["subject", "nc_group"],
        how="left",
    )
    score_df["between_group"] = np.where(
        score_df["baseline_or_stimulus"] == "same_session",
        "vicco",
        score_df["model_set"],
    )
    score_df["between_type"] = np.where(
        score_df["baseline_or_stimulus"] == "same_session",
        "vicco",
        "controversial",
    )
    score_df = score_df.merge(
        between[["subject", "between_group", "between_type", "nc_mid"]].rename(
            columns={"nc_mid": "between_nc_mid"}
        ),
        on=["subject", "between_group", "between_type"],
        how="left",
    )

    score_df["score_raw"] = score_df["score"]
    score_df["score_within_sqrt_rSB"] = score_df["score"] / score_df["within_sqrt_rSB"]
    score_df["score_within_rSB"] = score_df["score"] / score_df["within_rSB"]
    score_df["score_between_mid"] = score_df["score"] / score_df["between_nc_mid"]

    variants = [
        ("none_raw_score", "score_raw"),
        ("within_subject_sqrt_rSB", "score_within_sqrt_rSB"),
        ("within_subject_rSB", "score_within_rSB"),
        ("between_subject_mid", "score_between_mid"),
    ]
    for (subject, model_set, metric), grp in score_df.groupby(["subject", "model_set", "metric"]):
        cstim = grp[grp["baseline_or_stimulus"] == "controversial"]
        base = grp[grp["baseline_or_stimulus"] == "same_session"]
        if cstim.empty or base.empty:
            continue

        base_model = base.groupby("model", as_index=True).mean(numeric_only=True)
        cstim_model = cstim.groupby("model", as_index=True).mean(numeric_only=True)
        common = sorted(set(base_model.index) & set(cstim_model.index))
        if not common:
            continue

        for variant, col in variants:
            score_cstim = float(cstim_model.loc[common, col].mean())
            score_baseline = float(base_model.loc[common, col].mean())
            delta = score_cstim - score_baseline
            rows.append(
                {
                    "subject": subject,
                    "model_set": model_set,
                    "metric": metric,
                    "normalization_variant": variant,
                    "score_cstim": score_cstim,
                    "score_baseline": score_baseline,
                    "delta": delta,
                    "direction_negative": bool(delta < 0),
                    "n_models": len(common),
                }
            )

    return pd.DataFrame(rows).sort_values(["metric", "model_set", "subject", "normalization_variant"])


def main():
    out = build_noise_ceiling_variant_summary()
    out_path = DATA / "noise_ceiling_variant_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
