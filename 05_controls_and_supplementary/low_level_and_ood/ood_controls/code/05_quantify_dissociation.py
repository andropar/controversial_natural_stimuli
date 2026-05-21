#!/usr/bin/env python3
"""
05 — Quantitative summary for the OOD dissociation panel.

Two questions, one CSV:
  (1) "Shift exists?" — Cohen's d for cstim vs baseline on three axes:
        - low-level Mahalanobis distance (per-stimulus, two-sample d)
        - PPCA log-lik z, predicted-response space (per-stimulus, two-sample d)
        - PPCA log-lik z, raw-feature space          (per-stimulus, two-sample d)
        - brain RDM noise ceiling                    (per-subject, paired d_z)
      Sign convention: positive d ⇒ cstim is more OOD / less reliable than baseline.

  (2) "Shift doesn't drive the alignment drop?" — paired Δwrsa per
      (subject × model) cell between cstim and four control baselines:
        - match_<set>      (mean-matched baseline, narrow spread, n=100)
        - dist_match_<set> (distribution-shape-matched baseline: mean + spread
                             + shape match, n=100; primary control)
        - top100           (highest-low-level baseline, n=100)
        - full baseline    (all baseline images for that set's roster)
      And the same on noise-ceiling-normalized wRSA. Subject-cluster
      bootstrap (1000 resamples) gives the 95% CI on the mean Δ.

Output: data/dissociation_summary.csv (long format, one row per axis × set
× comparison).
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))

import numpy as np
import pandas as pd

import config


CSTIM_SETS = ["all_models", "architecture", "training_objective", "sota", "dataset"]
OOD = config.OOD_DATA_DIR
N_BOOT = 1000
RNG_SEED = 42


def cohens_d(a, b):
    """Two-sample Cohen's d with pooled SD."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * s1 ** 2 + (len(b) - 1) * s2 ** 2)
                     / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def cohens_dz(diffs):
    """Paired (within-subject) effect size."""
    diffs = np.asarray(diffs, float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return np.nan
    sd = diffs.std(ddof=1)
    return diffs.mean() / sd if sd > 0 else np.nan


def subject_cluster_bootstrap_ci(df, value_col, subject_col="subject",
                                 n_boot=N_BOOT, seed=RNG_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    subjects = np.asarray(df[subject_col].unique())
    grouped = {s: df.loc[df[subject_col] == s, value_col].to_numpy()
               for s in subjects}
    point = float(df[value_col].mean())
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(subjects, size=len(subjects), replace=True)
        vals = np.concatenate([grouped[s] for s in sample])
        boot_means[i] = vals.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def load_wrsa():
    dfs = []
    for s in config.SUBJECTS:
        p = config.RSA_DATA_DIR / s / "wrsa_transfer_scores.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# (1) "Shift exists" — Cohen's d per axis × set
# ---------------------------------------------------------------------------

def shift_low_level():
    df = pd.read_csv(OOD / "low_level_robustness_per_image_distances.csv")
    base = df[df["stim_set"] == "vicco"]["mahal_distance"].to_numpy()
    rows = []
    for s in CSTIM_SETS:
        cs = df[df["stim_set"] == s]["mahal_distance"].to_numpy()
        rows.append({
            "block": "shift_exists",
            "axis": "low_level_distance",
            "set": s,
            "comparison": "cstim_vs_baseline_per_stimulus",
            "n_cstim": len(cs),
            "n_baseline": len(base),
            "mean_cstim": cs.mean(),
            "mean_baseline": base.mean(),
            "delta_mean": cs.mean() - base.mean(),
            "cohens_d_oriented": cohens_d(cs, base),
            "direction_note": "positive d ⇒ cstim more low-level-distant",
        })
    return pd.DataFrame(rows)


def shift_ppca():
    df = pd.read_csv(OOD / "pca_loglik.csv")
    per_stim = (df.groupby(["stimulus_group", "stimulus_idx"])
                  [["loglik_pred_z", "loglik_feature_z"]]
                  .mean()
                  .reset_index())
    rows = []
    axes = [
        ("loglik_pred_z",   "ppca_pred_loglik_z"),
        ("loglik_feature_z", "ppca_feature_loglik_z"),
    ]
    for col, axis_label in axes:
        base = per_stim[per_stim["stimulus_group"] == "vicco"][col].to_numpy()
        for s in CSTIM_SETS:
            cs = per_stim[per_stim["stimulus_group"] == s][col].to_numpy()
            d_signed = cohens_d(cs, base)
            d_oriented = -d_signed if np.isfinite(d_signed) else np.nan
            rows.append({
                "block": "shift_exists",
                "axis": axis_label,
                "set": s,
                "comparison": "cstim_vs_baseline_per_stimulus_pooled_subject_model",
                "n_cstim": len(cs),
                "n_baseline": len(base),
                "mean_cstim": cs.mean(),
                "mean_baseline": base.mean(),
                "delta_mean": cs.mean() - base.mean(),
                "cohens_d_signed": d_signed,
                "cohens_d_oriented": d_oriented,
                "direction_note": "positive d_oriented ⇒ cstim more OOD (lower log-lik)",
            })
    return pd.DataFrame(rows)


def shift_noise_ceiling():
    nc = pd.read_csv(config.STATS_DATA_DIR / "rdm_noise_ceilings.csv")
    rows = []
    for s in CSTIM_SETS:
        cstim_nc = (nc[(nc["stimulus_type"] == "controversial")
                       & (nc["group"] == s)]
                    [["subject", "noise_ceiling_spearman"]]
                    .rename(columns={"noise_ceiling_spearman": "cstim_nc"}))
        base_nc = (nc[nc["stimulus_type"] == "vicco"]
                   .groupby("subject")["noise_ceiling_spearman"].mean()
                   .rename("base_nc").reset_index())
        merged = cstim_nc.merge(base_nc, on="subject")
        if merged.empty:
            continue
        diffs = (merged["cstim_nc"] - merged["base_nc"]).to_numpy()
        dz_signed = cohens_dz(diffs)
        dz_oriented = -dz_signed if np.isfinite(dz_signed) else np.nan
        rows.append({
            "block": "shift_exists",
            "axis": "noise_ceiling",
            "set": s,
            "comparison": "cstim_vs_baseline_paired_per_subject",
            "n_subjects": len(merged),
            "mean_cstim": merged["cstim_nc"].mean(),
            "mean_baseline": merged["base_nc"].mean(),
            "delta_mean": diffs.mean(),
            "cohens_dz_signed": dz_signed,
            "cohens_d_oriented": dz_oriented,
            "direction_note": "positive d_oriented ⇒ cstim less reliable",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (2) Paired Δwrsa: cstim vs control baseline subsets
# ---------------------------------------------------------------------------

def _per_cell_paired(s, baseline_kind, wrsa_full, det):
    """Per (subject, model) Δ between cstim wRSA and a baseline subset's wRSA.

    baseline_kind: "mean_match" | "dist_match" | "top100" | "full"
        - mean_match: vicco subset whose mean low-level matches cstim's
        - dist_match: vicco subset whose full distribution (mean + spread + shape)
                       matches cstim's
        - top100   : vicco subset of 100 highest-low-level images
        - full     : full vicco baseline (n=292) with same model roster
    """
    models = config.MODEL_SETS[s]
    cstim = (wrsa_full[(wrsa_full["stimulus_type"] == "controversial")
                       & (wrsa_full["model_set"] == s)]
             .groupby(["subject", "model"])["wrsa_transfer"].mean()
             .rename("cstim_wrsa").reset_index())

    if baseline_kind == "full":
        bs = (wrsa_full[(wrsa_full["stimulus_type"] == "vicco")
                        & (wrsa_full["model_set"] == s)]
              .groupby(["subject", "model"])["wrsa_transfer"].mean()
              .rename("baseline_wrsa").reset_index())
    else:
        if baseline_kind == "mean_match":
            subset = f"match_{s}"
        elif baseline_kind == "dist_match":
            subset = f"dist_match_{s}"
        elif baseline_kind == "top100":
            subset = "top100"
        else:
            raise ValueError(f"unknown baseline_kind: {baseline_kind}")
        bs = (det[(det["subset"] == subset) & (det["model"].isin(models))]
              .groupby(["subject", "model"])["wrsa"].mean()
              .rename("baseline_wrsa").reset_index())

    merged = cstim.merge(bs, on=["subject", "model"])
    merged["delta"] = merged["cstim_wrsa"] - merged["baseline_wrsa"]
    merged["set"] = s
    return merged


def paired_dwrsa(baseline_kind):
    wrsa = load_wrsa()
    det = pd.read_csv(OOD / "wrsa_low_level_subsets.csv")
    rows = []
    pieces = []
    for s in CSTIM_SETS:
        cells = _per_cell_paired(s, baseline_kind, wrsa, det)
        pieces.append(cells)
        if len(cells) == 0:
            continue
        mean, lo, hi = subject_cluster_bootstrap_ci(cells, "delta")
        rows.append({
            "block": "drop_persists",
            "axis": f"wrsa_minus_{baseline_kind}_baseline",
            "set": s,
            "comparison": "paired_cstim_minus_baseline_per_subject_model",
            "n_cells": len(cells),
            "delta_mean": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "direction_note": "delta < 0 ⇒ cstim drop persists vs control",
        })
    combined = pd.concat(pieces, ignore_index=True)
    if len(combined):
        mean, lo, hi = subject_cluster_bootstrap_ci(combined, "delta")
        rows.append({
            "block": "drop_persists",
            "axis": f"wrsa_minus_{baseline_kind}_baseline",
            "set": "combined",
            "comparison": "paired_cstim_minus_baseline_per_subject_model_pooled",
            "n_cells": len(combined),
            "delta_mean": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "direction_note": "delta < 0 ⇒ cstim drop persists vs control",
        })
    return pd.DataFrame(rows)


def paired_dwrsa_nc_normalized():
    wrsa = load_wrsa()
    nc = pd.read_csv(config.STATS_DATA_DIR / "rdm_noise_ceilings.csv")
    cstim_nc = (nc[nc["stimulus_type"] == "controversial"]
                [["subject", "group", "noise_ceiling_spearman"]]
                .rename(columns={"group": "model_set",
                                 "noise_ceiling_spearman": "cstim_nc"}))
    base_nc = (nc[nc["stimulus_type"] == "vicco"]
               .groupby("subject")["noise_ceiling_spearman"].mean()
               .rename("base_nc").reset_index())

    rows = []
    pieces = []
    for s in CSTIM_SETS:
        cstim = (wrsa[(wrsa["stimulus_type"] == "controversial")
                      & (wrsa["model_set"] == s)]
                 .merge(cstim_nc[cstim_nc["model_set"] == s],
                        on=["subject", "model_set"]))
        cstim = cstim[cstim["cstim_nc"] > 0].copy()
        cstim["norm"] = cstim["wrsa_transfer"] / np.sqrt(cstim["cstim_nc"])
        cstim_agg = (cstim.groupby(["subject", "model"])["norm"].mean()
                     .rename("cstim_norm").reset_index())

        base = (wrsa[(wrsa["stimulus_type"] == "vicco")
                     & (wrsa["model_set"] == s)]
                .merge(base_nc, on="subject"))
        base = base[base["base_nc"] > 0].copy()
        base["norm"] = base["wrsa_transfer"] / np.sqrt(base["base_nc"])
        base_agg = (base.groupby(["subject", "model"])["norm"].mean()
                    .rename("baseline_norm").reset_index())

        merged = cstim_agg.merge(base_agg, on=["subject", "model"])
        merged["delta"] = merged["cstim_norm"] - merged["baseline_norm"]
        merged["set"] = s
        pieces.append(merged)
        if len(merged) == 0:
            continue
        mean, lo, hi = subject_cluster_bootstrap_ci(merged, "delta")
        rows.append({
            "block": "drop_persists_nc_normalized",
            "axis": "nc_normalized_wrsa_minus_baseline",
            "set": s,
            "comparison": "paired_cstim_minus_baseline_per_subject_model",
            "n_cells": len(merged),
            "delta_mean": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "direction_note": "delta < 0 ⇒ cstim drop persists after NC normalization",
        })
    combined = pd.concat(pieces, ignore_index=True)
    if len(combined):
        mean, lo, hi = subject_cluster_bootstrap_ci(combined, "delta")
        rows.append({
            "block": "drop_persists_nc_normalized",
            "axis": "nc_normalized_wrsa_minus_baseline",
            "set": "combined",
            "comparison": "paired_cstim_minus_baseline_per_subject_model_pooled",
            "n_cells": len(combined),
            "delta_mean": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "direction_note": "delta < 0 ⇒ cstim drop persists after NC normalization",
        })
    return pd.DataFrame(rows)


def main():
    parts = [
        shift_low_level(),
        shift_ppca(),
        shift_noise_ceiling(),
        paired_dwrsa("mean_match"),
        paired_dwrsa("dist_match"),
        paired_dwrsa("top100"),
        paired_dwrsa("full"),
        paired_dwrsa_nc_normalized(),
    ]
    out = pd.concat(parts, ignore_index=True, sort=False)
    out_path = OOD / "dissociation_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    cols_display = ["block", "axis", "set",
                    "delta_mean", "cohens_d_oriented",
                    "ci95_lo", "ci95_hi"]
    cols_display = [c for c in cols_display if c in out.columns]
    print(out[cols_display].to_string(index=False))


if __name__ == "__main__":
    main()
