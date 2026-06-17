#!/usr/bin/env python3
"""
Are controversial stimuli simply OOD images, or do they carry independent
model-disagreement signal?

Methodological choices made for this rewrite:

  (1) PER-SET MODELS — Disagreement on each stim_set is computed using only
      the models that drove that set's selection. The vicco reference for
      that set is computed using THE SAME model list, so disagreement
      magnitudes are directly comparable (same model count, same roster).

  (2) PAIR OOD — We report two definitions side by side:
        pair_ood_mean = (ood_i + ood_j) / 2     (symmetric)
        pair_ood_max  = max(ood_i, ood_j)       (driven by more-OOD endpoint)
      Spearman r is reported separately for each.

  (3) RESIDUAL REFERENCE — We fit `image_disagreement = a + b·image_OOD` on
      vicco using THE SAME model roster used for the cstim set. Cstim
      residuals are then computed against this matched reference. By
      construction vicco residuals computed against their own roster's
      reference are zero-mean.

  (4) VICReg sensitivity — for sets that include VICReg (training_objective,
      all_models), both variants are computed (`all`, `no_vicreg`).

Implementation: load each (model, subject) encoding ONCE, predict on every
stim_set's features, cache the subject-mean predicted RDM vector per
(model, stim_set). Then assembling per-set / per-variant stacks is a slice.

Outputs:
  data/disagreement_vs_ood_pairs.csv     per (set × variant × pair)
  data/disagreement_vs_ood_images.csv    per (set × variant × image, with residual)
  data/disagreement_vs_ood_summary.csv   per (set × variant) Spearman + residual
"""

import sys
from pathlib import Path
_CSTIMS_SHARE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists())
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER))

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from cstims.paper import config
from cstims.paper.utils import (
    load_encoding_model, predict_voxel_responses,
    compute_rdm_correlation, rdm_to_vector,
)

OUT_DIR = config.OOD_DATA_DIR

STIM_SETS = ["all_models", "architecture", "training_objective", "sota", "dataset", "vicco"]
ALL_MODELS = config.MODEL_SETS["all_models"]


# --------------------------------------------------------------------------------
# (variant, stim_set) → list of models for disagreement / reference
# --------------------------------------------------------------------------------

def models_for_set(stim_set: str, variant: str) -> list:
    """Models to use when measuring disagreement on `stim_set` under `variant`.
    For vicco, we use the same models requested by the caller — vicco does not
    have a single canonical roster (it serves as reference for any set)."""
    if stim_set == "vicco":
        # caller picks the roster — we should never reach here directly
        raise ValueError("Use models_for_set(cstim_set, variant) and apply that "
                         "list to the vicco reference too.")
    if stim_set in config.MODEL_SETS:
        models = list(config.MODEL_SETS[stim_set])
    else:
        raise ValueError(stim_set)
    if variant == "no_vicreg":
        models = [m for m in models if m != "vicreg_resnet50"]
    return models


# --------------------------------------------------------------------------------
# Build per (model, stim_set) subject-mean predicted RDM cache
# --------------------------------------------------------------------------------

def build_rdm_cache():
    """For each (model, stim_set) where features and encodings exist, compute
    the subject-mean predicted RDM vector. Heavy step: encodings loaded once
    per (model, subject)."""
    cache = {}        # (model, stim_set) -> subject-mean rdm vector
    n_imgs_per_set = {}

    for model in tqdm(ALL_MODELS, desc="Models"):
        feat_path = config.CSTIM_FEATURE_CACHE / f"{model}.npz"
        if not feat_path.exists():
            print(f"  SKIP {model}: missing feature cache")
            continue
        feats = np.load(feat_path)
        # Pre-load every set's features
        X_sets = {ss: feats[ss].astype(np.float32) for ss in STIM_SETS if ss in feats}
        for ss, X in X_sets.items():
            n_imgs_per_set[ss] = X.shape[0]

        # For each subject, load encoding once, predict on all sets
        per_subj_rdm_vecs = {ss: [] for ss in X_sets}
        for subject in config.SUBJECTS:
            try:
                enc = load_encoding_model(model, subject)
            except FileNotFoundError:
                continue
            for ss, X in X_sets.items():
                pred = predict_voxel_responses(X, enc)
                pred_hlvis = pred[:, enc["roi_hlvis"]]
                rdm = compute_rdm_correlation(pred_hlvis)
                per_subj_rdm_vecs[ss].append(rdm_to_vector(rdm))

        # Subject-mean per (model, stim_set)
        for ss, vecs in per_subj_rdm_vecs.items():
            if not vecs:
                continue
            cache[(model, ss)] = np.mean(np.stack(vecs, axis=0), axis=0)

    return cache, n_imgs_per_set


# --------------------------------------------------------------------------------
# Build (n_models, n_pairs) z-scored RDM stack from cache
# --------------------------------------------------------------------------------

def build_stack(cache: dict, models: list, stim_set: str):
    """Returns (rdm_z stack, used_models). z-score within model. None if empty."""
    used = [m for m in models if (m, stim_set) in cache]
    if not used:
        return None, []
    rdm_z = []
    for m in used:
        v = cache[(m, stim_set)]
        rdm_z.append((v - v.mean()) / (v.std() + 1e-10))
    return np.stack(rdm_z, axis=0), used


# --------------------------------------------------------------------------------
# Per-image OOD across THE SAME models
# --------------------------------------------------------------------------------

def per_image_ood(stim_set: str, models: list) -> np.ndarray:
    df = pd.read_csv(config.OOD_DATA_DIR / "pca_loglik.csv")
    sub = df[(df["stimulus_group"] == stim_set) & (df["model"].isin(models))]
    return (sub.groupby("stimulus_idx")["loglik_pred_z"]
            .mean().sort_index().values)


# --------------------------------------------------------------------------------
# Pair / image stats
# --------------------------------------------------------------------------------

def compute_stats(rdm_z: np.ndarray, ood_per_img: np.ndarray):
    n_imgs = len(ood_per_img)
    iu = np.triu_indices(n_imgs, k=1)
    pair_disagreement = rdm_z.var(axis=0)
    pair_ood_mean = (ood_per_img[iu[0]] + ood_per_img[iu[1]]) / 2.0
    pair_ood_max  = np.maximum(ood_per_img[iu[0]], ood_per_img[iu[1]])
    var_mat = np.zeros((n_imgs, n_imgs))
    var_mat[iu] = pair_disagreement
    var_mat = var_mat + var_mat.T
    np.fill_diagonal(var_mat, np.nan)
    image_disagreement = np.nanmean(var_mat, axis=1)
    return {
        "iu":                 iu,
        "pair_disagreement":  pair_disagreement,
        "pair_ood_mean":      pair_ood_mean,
        "pair_ood_max":       pair_ood_max,
        "image_disagreement": image_disagreement,
    }


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    print("Building per (model, stim_set) RDM cache (heavy — one encoding load per "
          "(model, subject), all sets predicted from the loaded encoding)...")
    cache, n_imgs_per_set = build_rdm_cache()
    print(f"  Cached {len(cache):,} (model, stim_set) RDMs.")

    pair_rows, image_rows, summary_rows = [], [], []

    cstim_sets = [s for s in STIM_SETS if s != "vicco"]
    for stim_set in cstim_sets:
        variants = ["all"]
        if "vicreg_resnet50" in config.MODEL_SETS[stim_set]:
            variants.append("no_vicreg")

        for variant in variants:
            models = models_for_set(stim_set, variant)

            # ---- vicco reference using THE SAME models ----
            v_stack, v_used = build_stack(cache, models, "vicco")
            if v_stack is None:
                print(f"  SKIP {stim_set}/{variant}: no vicco coverage")
                continue
            v_ood = per_image_ood("vicco", v_used)
            if len(v_ood) != n_imgs_per_set["vicco"]:
                print(f"  WARN vicco OOD len {len(v_ood)} ≠ n_imgs {n_imgs_per_set['vicco']}")
                continue
            v_stats = compute_stats(v_stack, v_ood)
            slope, intercept, *_ = stats.linregress(v_ood, v_stats["image_disagreement"])

            # ---- cstim disagreement using same model roster ----
            c_stack, c_used = build_stack(cache, models, stim_set)
            if c_stack is None:
                print(f"  SKIP {stim_set}/{variant}: no cstim coverage")
                continue
            c_ood = per_image_ood(stim_set, c_used)
            if len(c_ood) != n_imgs_per_set[stim_set]:
                print(f"  WARN {stim_set} OOD len {len(c_ood)} ≠ n_imgs {n_imgs_per_set[stim_set]}")
                continue
            c_stats = compute_stats(c_stack, c_ood)
            c_residuals = c_stats["image_disagreement"] - (slope * c_ood + intercept)

            # Sanity: vicco residuals against its own ref are zero-mean (check)
            v_residuals = v_stats["image_disagreement"] - (slope * v_ood + intercept)

            r_pm_v, _ = stats.spearmanr(v_stats["pair_ood_mean"], v_stats["pair_disagreement"])
            r_px_v, _ = stats.spearmanr(v_stats["pair_ood_max"],  v_stats["pair_disagreement"])
            r_im_v, _ = stats.spearmanr(v_ood, v_stats["image_disagreement"])
            r_pm_c, _ = stats.spearmanr(c_stats["pair_ood_mean"], c_stats["pair_disagreement"])
            r_px_c, _ = stats.spearmanr(c_stats["pair_ood_max"],  c_stats["pair_disagreement"])
            r_im_c, _ = stats.spearmanr(c_ood, c_stats["image_disagreement"])

            print(f"  {stim_set:<22s} variant={variant:<10s}  "
                  f"n_models={len(c_used):>2d}  "
                  f"vicco_ref(intercept={intercept:+.4f}, slope={slope:+.4f})  "
                  f"cstim r(pair,ood_mean)={r_pm_c:+.3f}  "
                  f"r(pair,ood_max)={r_px_c:+.3f}  "
                  f"mean_residual_cstim={c_residuals.mean():+.4f}  "
                  f"mean_residual_vicco={v_residuals.mean():+.4f}")

            # Save vicco rows under (vicco, stim_set, variant)  — multi-keyed
            # because vicco's reference is per-(stim_set, variant)
            for k in range(len(v_stats["pair_disagreement"])):
                pair_rows.append({
                    "stim_set":          "vicco",
                    "ref_for_set":       stim_set,
                    "variant":           variant,
                    "i":                 int(v_stats["iu"][0][k]),
                    "j":                 int(v_stats["iu"][1][k]),
                    "pair_disagreement": float(v_stats["pair_disagreement"][k]),
                    "pair_ood_mean":     float(v_stats["pair_ood_mean"][k]),
                    "pair_ood_max":      float(v_stats["pair_ood_max"][k]),
                })
            for ii in range(len(v_ood)):
                image_rows.append({
                    "stim_set":           "vicco",
                    "ref_for_set":        stim_set,
                    "variant":            variant,
                    "image_idx":          ii,
                    "image_disagreement": float(v_stats["image_disagreement"][ii]),
                    "image_ood":          float(v_ood[ii]),
                    "residual":           float(v_residuals[ii]),
                })

            # Save cstim rows
            for k in range(len(c_stats["pair_disagreement"])):
                pair_rows.append({
                    "stim_set":          stim_set,
                    "ref_for_set":       stim_set,
                    "variant":           variant,
                    "i":                 int(c_stats["iu"][0][k]),
                    "j":                 int(c_stats["iu"][1][k]),
                    "pair_disagreement": float(c_stats["pair_disagreement"][k]),
                    "pair_ood_mean":     float(c_stats["pair_ood_mean"][k]),
                    "pair_ood_max":      float(c_stats["pair_ood_max"][k]),
                })
            for ii in range(len(c_ood)):
                image_rows.append({
                    "stim_set":           stim_set,
                    "ref_for_set":        stim_set,
                    "variant":            variant,
                    "image_idx":          ii,
                    "image_disagreement": float(c_stats["image_disagreement"][ii]),
                    "image_ood":          float(c_ood[ii]),
                    "residual":           float(c_residuals[ii]),
                })

            summary_rows.append({
                "stim_set":               "vicco",
                "ref_for_set":            stim_set,
                "variant":                variant,
                "n_models":               len(v_used),
                "n_pairs":                int(len(v_stats["pair_disagreement"])),
                "n_imgs":                 int(len(v_ood)),
                "spearman_pair_ood_mean": float(r_pm_v),
                "spearman_pair_ood_max":  float(r_px_v),
                "spearman_image_ood":     float(r_im_v),
                "mean_residual":          float(v_residuals.mean()),
                "ref_intercept":          float(intercept),
                "ref_slope":              float(slope),
            })
            summary_rows.append({
                "stim_set":               stim_set,
                "ref_for_set":            stim_set,
                "variant":                variant,
                "n_models":               len(c_used),
                "n_pairs":                int(len(c_stats["pair_disagreement"])),
                "n_imgs":                 int(len(c_ood)),
                "spearman_pair_ood_mean": float(r_pm_c),
                "spearman_pair_ood_max":  float(r_px_c),
                "spearman_image_ood":     float(r_im_c),
                "mean_residual":          float(c_residuals.mean()),
                "ref_intercept":          float(intercept),
                "ref_slope":              float(slope),
            })

    pd.DataFrame(pair_rows).to_csv(OUT_DIR / "disagreement_vs_ood_pairs.csv", index=False)
    pd.DataFrame(image_rows).to_csv(OUT_DIR / "disagreement_vs_ood_images.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "disagreement_vs_ood_summary.csv", index=False)

    print("\nSaved pairs   →", OUT_DIR / "disagreement_vs_ood_pairs.csv")
    print("Saved images  →", OUT_DIR / "disagreement_vs_ood_images.csv")
    print("Saved summary →", OUT_DIR / "disagreement_vs_ood_summary.csv")

    print("\nFinal summary:")
    print(pd.DataFrame(summary_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
