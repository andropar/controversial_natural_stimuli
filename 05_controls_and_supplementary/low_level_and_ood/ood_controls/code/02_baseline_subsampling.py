#!/usr/bin/env python3
"""
Within-baseline subsampling control for the OOD analysis.

Question: among baseline (vicco) subsets that are *as OOD* as the controversial
sets, do we still see the alignment drop?

Design (per subject × model_set × model):
  1. Reconstruct the 1000 vicco bootstraps used by 02_compute_wrsa_transfer.py
     (deterministic from bootstrap_sample_indices(292, 100, n_boots=1000, seed=0)).
  2. For each bootstrap, compute its mean prediction-space PPCA loglik z
     (mean over the 100 stimuli in that bootstrap, for this subject × model).
  3. Compute the cstim mean OOD on its 100 stimuli, same subject × model.
  4. Quantify *match quality*: gap_in_sd = (cstim_ood - matched_K_mean_ood) /
     std(bootstrap_ood). Categorise:
         "matched"      gap < 0.5 SD       → OOD-equating is real
         "weak"         0.5 ≤ gap < 1.5 SD → conservative (baseline somewhat lower-OOD)
         "out_of_range" gap ≥ 1.5 SD       → baseline pool can't reach cstim OOD
  5. Within the K=50 OOD-closest bootstraps, take their mean wRSA.
  6. Compare:
       - cstim wRSA
       - all-boot mean wRSA
       - matched-K mean wRSA
     The matched-vs-cstim comparison is only interpretable as an OOD-equating
     test where match_quality == "matched".

Aggregation across subjects:
  - Per-subject results saved.
  - Per (model_set, model) summary: subject-mean of each metric, plus the
    per-subject match-quality distribution.

Two outputs:
  data/baseline_subsampling.csv          - per (subj × set × model × bootstrap),
                                           one row per cstim point (boot_idx=-1)
                                           and one per bootstrap.
  data/baseline_subsampling_summary.csv  - per (subj × set × model) match
                                           quality and aggregate scores.

The summary CSV is the canonical result for the figure / paper claims.
"""

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))

import numpy as np
import pandas as pd
from tqdm import tqdm

from cstims.paper import config
from cstims.paper.utils import bootstrap_sample_indices

# --------------------------------------------------------------------------------
# Constants matching 02_compute_wrsa_transfer.py
# --------------------------------------------------------------------------------

N_VICCO_BOOTSTRAPS = 1000
N_VICCO_SAMPLE     = 100
N_VICCO_TOTAL      = 292
BOOTSTRAP_SEED     = 0
K_MATCH            = 50    # K nearest-OOD bootstraps used as the matched comparator
SD_GOOD_MATCH      = 0.5   # gap thresholds (in SD of bootstrap OOD distribution)
SD_OUT_OF_RANGE    = 1.5

OOD_DATA = config.OOD_DATA_DIR / "pca_loglik.csv"
RSA_DIR  = config.RSA_DATA_DIR
OUT_PER  = config.OOD_DATA_DIR / "baseline_subsampling.csv"
OUT_SUM  = config.OOD_DATA_DIR / "baseline_subsampling_summary.csv"


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------

def load_vicco_stim_idx(subject: str) -> np.ndarray:
    """Per-subject ordering of vicco stimuli (file_idx 0..291), as iterated by
    02_compute_wrsa_transfer.py."""
    info = pd.read_csv(config.get_subject_data_dir(subject) / "cstim_stimulus_info.csv")
    vicco = info[info["group"] == "vicco"].reset_index(drop=True)
    file_idx = vicco["stim_idx"].astype(int).values - 1
    if len(file_idx) != N_VICCO_TOTAL:
        raise RuntimeError(f"{subject}: expected {N_VICCO_TOTAL} vicco stims, got {len(file_idx)}")
    return file_idx


def load_wrsa(subjects):
    dfs = []
    for s in subjects:
        p = RSA_DIR / s / "wrsa_transfer_scores.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def classify_match(gap_sd: float) -> str:
    if gap_sd < SD_GOOD_MATCH:
        return "matched"
    if gap_sd < SD_OUT_OF_RANGE:
        return "weak"
    return "out_of_range"


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    print("Loading OOD data and wRSA scores...")
    ood  = pd.read_csv(OOD_DATA)
    wrsa = load_wrsa(config.SUBJECTS)
    print(f"  OOD: {len(ood):,} rows;  wRSA: {len(wrsa):,} rows")

    bootstrap_subsets = bootstrap_sample_indices(
        N_VICCO_TOTAL, N_VICCO_SAMPLE,
        n_bootstrap=N_VICCO_BOOTSTRAPS, seed=BOOTSTRAP_SEED,
    )
    subj_vicco_file_idx = {s: load_vicco_stim_idx(s) for s in config.SUBJECTS}

    # ---- Vicco mean OOD per (subject, model, bootstrap_idx) ----
    print("\nComputing per-bootstrap mean OOD for vicco...")
    vicco_ood = ood[ood["stimulus_group"] == "vicco"]

    boot_ood_rows = []
    for (subject, model), grp in tqdm(
        vicco_ood.groupby(["subject", "model"], sort=False),
        total=vicco_ood[["subject", "model"]].drop_duplicates().shape[0],
    ):
        arr_pred = np.full(N_VICCO_TOTAL, np.nan)
        idx = grp["stimulus_idx"].astype(int).values
        arr_pred[idx] = grp["loglik_pred_z"].values

        file_idx_order = subj_vicco_file_idx[subject]
        for b, subset_pos in enumerate(bootstrap_subsets):
            subset_file_idx = file_idx_order[subset_pos]
            boot_ood_rows.append({
                "subject":            subject,
                "model":              model,
                "bootstrap_idx":      b,
                "mean_loglik_pred_z": float(np.nanmean(arr_pred[subset_file_idx])),
            })
    boot_ood_df = pd.DataFrame(boot_ood_rows)

    # ---- Cstim mean OOD per (subject, model, model_set) ----
    cstim_groups = list(config.MODEL_SETS.keys())
    cstim_ood_df = (
        ood[ood["stimulus_group"].isin(cstim_groups)]
        .groupby(["subject", "model", "stimulus_group"])["loglik_pred_z"]
        .mean().reset_index()
        .rename(columns={"stimulus_group": "model_set",
                         "loglik_pred_z": "cstim_mean_loglik_pred_z"})
    )

    # ---- Per-row long table: cstim and per-bootstrap rows ----
    print("Building per-row long table...")
    wrsa_vicco = (
        wrsa[wrsa["stimulus_type"] == "vicco"]
        [["subject", "model_set", "model", "bootstrap_idx", "wrsa_transfer"]]
        .rename(columns={"wrsa_transfer": "wrsa"})
    )
    vicco_long = wrsa_vicco.merge(boot_ood_df, on=["subject", "model", "bootstrap_idx"])
    vicco_long["stim_type"] = "vicco"

    wrsa_cstim = (
        wrsa[wrsa["stimulus_type"] == "controversial"]
        [["subject", "model_set", "model", "wrsa_transfer"]]
        .rename(columns={"wrsa_transfer": "wrsa"})
    )
    cstim_long = wrsa_cstim.merge(cstim_ood_df, on=["subject", "model_set", "model"])
    cstim_long = cstim_long.rename(columns={"cstim_mean_loglik_pred_z": "mean_loglik_pred_z"})
    cstim_long["bootstrap_idx"] = -1
    cstim_long["stim_type"] = "controversial"

    long_cols = ["subject", "model_set", "model", "stim_type", "bootstrap_idx",
                 "mean_loglik_pred_z", "wrsa"]
    long = pd.concat([vicco_long[long_cols], cstim_long[long_cols]], ignore_index=True)
    long.to_csv(OUT_PER, index=False)
    print(f"  Saved {len(long):,} rows → {OUT_PER}")

    # ---- Per (subject, model_set, model) summary ----
    print("\nBuilding per-subject summary with match quality...")
    summary_rows = []
    for (subject, model_set, model), boots in tqdm(
        vicco_long.groupby(["subject", "model_set", "model"]),
        total=vicco_long[["subject","model_set","model"]].drop_duplicates().shape[0],
    ):
        boot_oods  = boots["mean_loglik_pred_z"].values
        boot_wrsas = boots["wrsa"].values
        cstim_row = cstim_long[
            (cstim_long["subject"] == subject)
            & (cstim_long["model_set"] == model_set)
            & (cstim_long["model"] == model)
        ]
        if cstim_row.empty:
            continue
        cstim_ood  = float(cstim_row["mean_loglik_pred_z"].iloc[0])
        cstim_wrsa = float(cstim_row["wrsa"].iloc[0])

        order = np.argsort(np.abs(boot_oods - cstim_ood))[:K_MATCH]
        matched_oods  = boot_oods[order]
        matched_wrsas = boot_wrsas[order]

        boot_sd = float(boot_oods.std()) if boot_oods.std() > 0 else np.nan
        gap_raw = cstim_ood - matched_oods.mean()
        gap_sd  = abs(gap_raw) / boot_sd if boot_sd > 0 else np.nan
        # Out-of-range = cstim OOD beyond bootstrap OOD distribution edge
        out_of_range = (cstim_ood < boot_oods.min() - 1e-9) or (cstim_ood > boot_oods.max() + 1e-9)

        summary_rows.append({
            "subject":         subject,
            "model_set":       model_set,
            "model":           model,
            "cstim_ood":       cstim_ood,
            "cstim_wrsa":      cstim_wrsa,
            "all_boot_ood":    float(boot_oods.mean()),
            "all_boot_wrsa":   float(boot_wrsas.mean()),
            "boot_ood_sd":     boot_sd,
            "boot_ood_min":    float(boot_oods.min()),
            "boot_ood_max":    float(boot_oods.max()),
            "matched_K":       K_MATCH,
            "matched_ood":     float(matched_oods.mean()),
            "matched_wrsa":    float(matched_wrsas.mean()),
            "gap_raw":         float(gap_raw),
            "gap_sd":          float(gap_sd) if np.isfinite(gap_sd) else np.nan,
            "out_of_range":    bool(out_of_range),
            "match_quality":   classify_match(gap_sd) if np.isfinite(gap_sd) else "unknown",
            "drop_vs_all":     cstim_wrsa - float(boot_wrsas.mean()),
            "drop_vs_matched": cstim_wrsa - float(matched_wrsas.mean()),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUM, index=False)
    print(f"  Saved {len(summary):,} (subject × set × model) rows → {OUT_SUM}")

    # ---- Diagnostic printout ----
    print("\nMatch quality by (model_set), counted across subjects × models:")
    qual = (summary.groupby(["model_set", "match_quality"]).size()
            .unstack(fill_value=0))
    print(qual.to_string())

    print("\nPer (model_set) subject-mean (across model × subject):")
    by_set = (summary.groupby("model_set")
              [["cstim_wrsa", "all_boot_wrsa", "matched_wrsa",
                "cstim_ood", "all_boot_ood", "matched_ood",
                "gap_sd", "drop_vs_all", "drop_vs_matched"]]
              .mean().round(3))
    print(by_set.to_string())

    n_matched = (summary["match_quality"] == "matched").sum()
    print(f"\n→ Of {len(summary):,} (subject × set × model) cells, {n_matched} qualify as "
          f"'matched' (gap < {SD_GOOD_MATCH} SD of bootstrap OOD).")
    if n_matched > 0:
        print("\nDrop_vs_matched on 'matched' cells only (subject-mean per model_set):")
        m = summary[summary["match_quality"] == "matched"]
        print(m.groupby("model_set")
              [["cstim_wrsa", "matched_wrsa", "drop_vs_matched", "gap_sd"]]
              .agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
