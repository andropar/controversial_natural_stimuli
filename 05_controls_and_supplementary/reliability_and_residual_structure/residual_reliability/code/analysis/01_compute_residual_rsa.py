#!/usr/bin/env python3
"""
Residual reliability of the brain RDM after accounting for the full vision-model set.

For each (subject, stimulus group, rsa_type):
  - brain RDM (hlvis, correlation distance), + split-half halves for NC.
  - 20 model RDMs (raw features for fixed RSA; predicted-voxel RDMs for mixed).
  - Single-best: max over models of Spearman(brain, model).
  - Ensemble: ridge on rank-transformed model RDMs predicting rank-transformed
    brain RDM, stimulus-level 10-fold CV.
  - Noise ceiling reliability: SB-corrected split-half reliability of brain RDM.
  - Correlation ceiling: sqrt(noise ceiling reliability), the relevant ceiling
    for model-brain correlations.
  - Residual reliability: SB-corrected split-half reliability of
    (brain - ensemble prediction), where the ensemble is fit on each half
    independently using the same CV design.
  - LOSO residual RSA: residualize each subject's full brain RDM against that
    subject's model-RDM space, average residuals across the other subjects, and
    correlate subject residuals with the leave-one-subject-out residual mean.

Optimisation: per (subject, group, rsa_type) we compute the *full* pairwise RDM
matrix once, then subset for vicco bootstraps (a bootstrap is just row/column
indexing). Encoders are cached per (subject, model).

Vicco baseline: bootstrap resamples of N=100 from the 292 images.

Output: results/residual_rsa.csv
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parents[2]
sys.path.insert(0, str(SHARE_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from cstims import constants, paths
from cstims.cache import load_cstim_features, load_cstim_repetition_cache
from cstims.rdm import compute_rdm_correlation
from cstims.sampling import bootstrap_sample_indices, stimulus_cv_splits
from cstims.subjects import parse_subject_arg
from cstims.paper.utils import load_encoding_model, predict_voxel_responses


N_VICCO_SUBSAMPLES = 10
VICCO_SAMPLE_SIZE = 100
N_CV_FOLDS = 10
N_CV_REPEATS = 50
CV_RANDOM_STATE = 42
RIDGE_ALPHAS = np.logspace(-2, 6, 30)
GROUPS_CONTROVERSIAL = ["all_models", "architecture", "dataset", "sota", "training_objective"]
ALL_MODELS = constants.MODEL_SETS["all_models"]


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------

def _load_subject_reps(subject: str):
    """Load per-rep hlvis betas + stim info, indexed by group."""
    cache = load_cstim_repetition_cache(subject)
    group_keys: dict[str, list[str]] = {}
    group_stim_idx: dict[str, np.ndarray] = {}
    for group in cache.available_groups:
        group_stim_idx[group] = cache.feature_indices(group, sort_by_stim_idx=True)
        group_keys[group] = cache.stim_keys_for_group(
            group, sort_by_stim_idx=True
        )

    return {
        "betas_by_rep": cache.betas_by_rep,
        "group_keys": group_keys,
        "group_stim_idx": group_stim_idx,
    }


def _brain_rdm_matrices(
    stim_keys: list[str], betas_by_rep: dict
) -> dict[str, np.ndarray]:
    """Return full brain RDMs {full, even, odd} as N×N matrices."""
    n = len(stim_keys)
    n_vox = betas_by_rep[stim_keys[0]].shape[0]

    avg_full = np.zeros((n, n_vox), dtype=np.float32)
    avg_even = np.zeros((n, n_vox), dtype=np.float32)
    avg_odd = np.zeros((n, n_vox), dtype=np.float32)

    for i, k in enumerate(stim_keys):
        reps = betas_by_rep[k]
        nreps = reps.shape[1]
        avg_full[i] = reps.mean(axis=1)
        if nreps == 1:
            avg_even[i] = reps[:, 0]
            avg_odd[i] = reps[:, 0]
        else:
            ei = np.arange(0, nreps, 2)
            oi = np.arange(1, nreps, 2)
            avg_even[i] = reps[:, ei].mean(axis=1)
            avg_odd[i] = reps[:, oi].mean(axis=1)

    return {
        "full": compute_rdm_correlation(avg_full),
        "even": compute_rdm_correlation(avg_even),
        "odd":  compute_rdm_correlation(avg_odd),
    }


def _model_rdm_matrices(
    group: str, stim_idx: np.ndarray, subject: str, rsa_type: str,
    encoding_cache: dict | None,
) -> dict[str, np.ndarray]:
    """Return {model_name: N×N RDM} for the given group subset."""
    rdms = {}
    for model in ALL_MODELS:
        feats = load_cstim_features(model, group)[stim_idx]
        if rsa_type == "fixed":
            rdms[model] = compute_rdm_correlation(feats)
        elif rsa_type == "mixed":
            key = (subject, model)
            if encoding_cache is None:
                enc = load_encoding_model(model, subject)
            else:
                if key not in encoding_cache:
                    encoding_cache[key] = load_encoding_model(model, subject)
                enc = encoding_cache[key]
            pred = predict_voxel_responses(feats, enc)
            pred_hlvis = pred[:, enc["roi_hlvis"]]
            rdms[model] = compute_rdm_correlation(pred_hlvis)
            if encoding_cache is None:
                del enc, pred, pred_hlvis
                gc.collect()
        else:
            raise ValueError(rsa_type)
    return rdms


def _triu_vec(rdm_matrix: np.ndarray, sub_idx: np.ndarray | None = None) -> np.ndarray:
    """Upper-triangular vector of an RDM matrix (optionally subset by sub_idx)."""
    if sub_idx is not None:
        rdm_matrix = rdm_matrix[np.ix_(sub_idx, sub_idx)]
    n = rdm_matrix.shape[0]
    i, j = np.triu_indices(n, k=1)
    return rdm_matrix[i, j]


# ---------------------------------------------------------------------------
# Ensemble ridge with stimulus-level CV
# ---------------------------------------------------------------------------

def _ridge_oof_ranked(
    brain_vec: np.ndarray,
    model_vecs: list[np.ndarray],
    n_stim: int,
    alphas: np.ndarray = RIDGE_ALPHAS,
    n_folds: int = N_CV_FOLDS,
    n_repeats: int = N_CV_REPEATS,
    random_state: int = CV_RANDOM_STATE,
    return_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """OOF ridge predictions on rank-transformed RDMs with repeated image-blocked CV.

    A single image-blocked K-fold partition can only score pairs whose two
    images land in the same held-out fold. Repeating the partition gives most
    image pairs at least one no-shared-image out-of-fold prediction while
    preserving the strict train/test image separation.
    """
    y = stats.rankdata(brain_vec).astype(np.float64)
    X = np.column_stack([stats.rankdata(v).astype(np.float64) for v in model_vecs])

    n_repeats = max(1, int(n_repeats))
    pred_sum = np.zeros_like(y, dtype=np.float64)
    pred_count = np.zeros_like(y, dtype=np.int32)

    for rep in range(n_repeats):
        splits = stimulus_cv_splits(
            n_stim, n_splits=n_folds, random_state=random_state + rep
        )
        for train_idx, test_idx in splits:
            if len(test_idx) == 0:
                continue
            mean = X[train_idx].mean(axis=0)
            std = X[train_idx].std(axis=0) + 1e-8
            X_tr = (X[train_idx] - mean) / std
            X_te = (X[test_idx] - mean) / std

            ridge = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
            ridge.fit(X_tr, y[train_idx])
            pred_sum[test_idx] += ridge.predict(X_te)
            pred_count[test_idx] += 1

    y_pred = np.full_like(y, np.nan)
    covered = pred_count > 0
    y_pred[covered] = pred_sum[covered] / pred_count[covered]
    if return_counts:
        return y_pred, pred_count
    return y_pred


def _ridge_residual_ranked(
    brain_vec: np.ndarray,
    model_vecs: list[np.ndarray],
    alphas: np.ndarray = RIDGE_ALPHAS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Full-fit ridge residual on rank-transformed RDM vectors.

    This is intentionally an in-sample residualization step: it gives the model
    RDM space every chance to remove structure before the cross-subject residual
    correlation is computed.
    """
    y = stats.rankdata(brain_vec).astype(np.float64)
    X = np.column_stack([stats.rankdata(v).astype(np.float64) for v in model_vecs])

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_std = (X - mean) / std

    ridge = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
    ridge.fit(X_std, y)
    pred = ridge.predict(X_std)
    return y - pred, pred, float(ridge.alpha_)


def _rank_zscore(vec: np.ndarray) -> np.ndarray:
    """Rank-transform and z-score a vector for leave-one-subject-out averaging."""
    ranks = stats.rankdata(vec).astype(np.float64)
    std = ranks.std()
    if std == 0:
        return ranks - ranks.mean()
    return (ranks - ranks.mean()) / std


def _zscore(vec: np.ndarray) -> np.ndarray:
    std = np.nanstd(vec)
    if std == 0 or not np.isfinite(std):
        return vec - np.nanmean(vec)
    return (vec - np.nanmean(vec)) / std


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.spearmanr(a[mask], b[mask])
    return float(r)


def _sb_correct(r: float) -> float:
    if not np.isfinite(r) or (1 + r) == 0:
        return np.nan
    return float(2 * r / (1 + r))


# ---------------------------------------------------------------------------
# Per-cell analysis
# ---------------------------------------------------------------------------

def analyze_cell(
    brain_rdms: dict[str, np.ndarray],   # N×N matrices: full, even, odd
    model_rdms: dict[str, np.ndarray],   # {model: N×N}
    sub_idx: np.ndarray | None = None,
    n_stim_override: int | None = None,
    cv_repeats: int = N_CV_REPEATS,
    cv_seed: int = CV_RANDOM_STATE,
) -> dict:
    """Compute single-best, ensemble, NC, residual reliability for one cell.

    If `sub_idx` is given, all RDM matrices are subset to those stimulus
    indices before computing triu vectors. `n_stim_override` sets the number
    of stimuli for CV splits (defaults to matrix size or len(sub_idx))."""
    # Pull triu vectors in matching stimulus order
    def vec(m): return _triu_vec(m, sub_idx)
    brain_full = vec(brain_rdms["full"])
    brain_even = vec(brain_rdms["even"])
    brain_odd  = vec(brain_rdms["odd"])
    model_vecs = {m: vec(r) for m, r in model_rdms.items()}

    n_stim = n_stim_override if n_stim_override is not None else (
        len(sub_idx) if sub_idx is not None else brain_rdms["full"].shape[0]
    )

    # Single-best model
    per_model_r = {m: _spearman(brain_full, v) for m, v in model_vecs.items()}
    best_model = max(per_model_r, key=lambda k: per_model_r[k])
    r_single_best = per_model_r[best_model]

    # Ensemble OOF on full-rep brain
    model_vec_list = [model_vecs[m] for m in ALL_MODELS]
    oof_full, oof_counts = _ridge_oof_ranked(
        brain_full,
        model_vec_list,
        n_stim,
        n_repeats=cv_repeats,
        random_state=cv_seed,
        return_counts=True,
    )
    brain_full_r = stats.rankdata(brain_full)
    r_ensemble = _spearman(brain_full_r, oof_full)
    n_pairs_total = int(len(brain_full))
    n_pairs_oof = int((oof_counts > 0).sum())
    pair_coverage = n_pairs_oof / n_pairs_total if n_pairs_total else np.nan

    # Noise ceiling from split-half
    r_halves = _spearman(brain_even, brain_odd)
    noise_ceiling = _sb_correct(r_halves)
    correlation_ceiling = np.sqrt(noise_ceiling) if noise_ceiling > 0 else np.nan

    # Residual reliability: fit ensemble on each half, correlate residuals
    oof_even = _ridge_oof_ranked(
        brain_even,
        model_vec_list,
        n_stim,
        n_repeats=cv_repeats,
        random_state=cv_seed,
    )
    oof_odd = _ridge_oof_ranked(
        brain_odd,
        model_vec_list,
        n_stim,
        n_repeats=cv_repeats,
        random_state=cv_seed,
    )
    resid_even = stats.rankdata(brain_even) - oof_even
    resid_odd  = stats.rankdata(brain_odd)  - oof_odd
    residual_reliability = _sb_correct(_spearman(resid_even, resid_odd))

    # Full residual used for LOSO residual RSA. This is not itself reported as
    # a model score; it is the subject-specific residual after removing the
    # full model-RDM space as generously as possible.
    residual_full, full_fit_pred, residual_alpha = _ridge_residual_ranked(
        brain_full, model_vec_list
    )
    r_ensemble_full_fit = _spearman(brain_full_r, full_fit_pred)

    return {
        "r_single_best": r_single_best,
        "r_single_best_model": best_model,
        "r_ensemble": r_ensemble,  # Backwards-compatible alias.
        "r_ensemble_cv": r_ensemble,
        "r_ensemble_full_fit": r_ensemble_full_fit,
        "cv_folds_image_blocked": int(N_CV_FOLDS),
        "cv_repeats_image_blocked": int(cv_repeats),
        "cv_random_state": int(cv_seed),
        "n_pairs_total": n_pairs_total,
        "n_pairs_oof_predicted": n_pairs_oof,
        "oof_pair_coverage": pair_coverage,
        "oof_pair_prediction_count_min": int(oof_counts[oof_counts > 0].min()) if n_pairs_oof else 0,
        "oof_pair_prediction_count_median": (
            float(np.median(oof_counts[oof_counts > 0])) if n_pairs_oof else np.nan
        ),
        "oof_pair_prediction_count_max": int(oof_counts.max()) if len(oof_counts) else 0,
        "noise_ceiling": noise_ceiling,  # Backwards-compatible alias.
        "noise_ceiling_reliability": noise_ceiling,
        "correlation_ceiling": correlation_ceiling,
        "ensemble_pct_correlation_ceiling": (
            r_ensemble / correlation_ceiling
            if np.isfinite(correlation_ceiling) and correlation_ceiling > 0
            else np.nan
        ),
        "ensemble_gap_to_correlation_ceiling": (
            correlation_ceiling - r_ensemble
            if np.isfinite(correlation_ceiling) else np.nan
        ),
        "residual_reliability": residual_reliability,  # Backwards-compatible alias.
        "residual_split_half_reliability": residual_reliability,
        "residual_alpha": residual_alpha,
        "_brain_full_vec": brain_full,
        "_residual_full_vec": residual_full,
    }


def _add_loso_metrics(cell_rows: list[dict]) -> None:
    """Add LOSO brain/residual correlations to rows from one analysis cell."""
    for row in cell_rows:
        others = [r for r in cell_rows if r["subject"] != row["subject"]]
        if not others:
            row["r_loso_brain"] = np.nan
            row["r_loso_residual"] = np.nan
            row["loso_residual_fraction"] = np.nan
            continue

        loso_brain = np.mean(
            [_rank_zscore(other["_brain_full_vec"]) for other in others], axis=0
        )
        loso_residual = np.mean(
            [_zscore(other["_residual_full_vec"]) for other in others], axis=0
        )

        row["r_loso_brain"] = _spearman(row["_brain_full_vec"], loso_brain)
        row["r_loso_residual"] = _spearman(row["_residual_full_vec"], loso_residual)
        row["loso_residual_fraction"] = (
            row["r_loso_residual"] / row["r_loso_brain"]
            if np.isfinite(row["r_loso_brain"]) and row["r_loso_brain"] > 0
            else np.nan
        )


def _public_row(row: dict) -> dict:
    """Drop in-memory vectors before writing CSV."""
    return {k: v for k, v in row.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--n-vicco-bootstraps",
        type=int,
        default=None,
        help="Deprecated name. Samples are N-matched draws without replacement.",
    )
    parser.add_argument(
        "--n-vicco-subsamples",
        type=int,
        default=N_VICCO_SUBSAMPLES,
        help="Number of N=100 Vicco baseline subsamples drawn without replacement.",
    )
    parser.add_argument("--cv-repeats", type=int, default=N_CV_REPEATS)
    parser.add_argument("--cv-seed", type=int, default=CV_RANDOM_STATE)
    parser.add_argument(
        "--rsa-type", choices=["all", "fixed", "mixed"], default="all",
        help="Run one RSA type or both (default: all).",
    )
    parser.add_argument(
        "--groups", default=None,
        help="Comma-separated stimulus groups to run. Use 'vicco' for baseline.",
    )
    args = parser.parse_args()

    subjects = parse_subject_arg(args.subject)
    rsa_types = ["fixed", "mixed"] if args.rsa_type == "all" else [args.rsa_type]
    n_vicco_subsamples = (
        args.n_vicco_bootstraps
        if args.n_vicco_bootstraps is not None
        else args.n_vicco_subsamples
    )
    requested_groups = (
        set(g.strip() for g in args.groups.split(",") if g.strip())
        if args.groups else set(GROUPS_CONTROVERSIAL + ["vicco"])
    )
    out_path = Path(args.output) if args.output else (
        STAGE / "results" / "residual_rsa.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    print(f"Loading subject data: {subjects}", flush=True)
    subject_data = {subject: _load_subject_reps(subject) for subject in subjects}

    rows: list[dict] = []
    def save_progress() -> None:
        if rows:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"  Saved {len(rows)} rows -> {out_path}", flush=True)

    for rsa_type in rsa_types:
        print(f"\n== {rsa_type} RSA ==", flush=True)

        # Controversial groups: analyze all subjects for one cell, then add
        # leave-one-subject-out residual correlations across those subjects.
        for group in GROUPS_CONTROVERSIAL:
            if group not in requested_groups:
                continue
            print(f"  {group}", flush=True)
            cell_rows: list[dict] = []

            for subject, sdata in subject_data.items():
                if group not in sdata["group_keys"]:
                    continue
                keys = sdata["group_keys"][group]
                stim_idx = sdata["group_stim_idx"][group]
                if len(keys) == 0:
                    continue

                brain_rdms = _brain_rdm_matrices(keys, sdata["betas_by_rep"])
                # Encoding models are large; keep this cache local so the mixed
                # RSA pass does not retain all subject/model encoders in memory.
                encoding_cache: dict | None = {} if rsa_type == "fixed" else None
                model_rdms = _model_rdm_matrices(
                    group, stim_idx, subject, rsa_type, encoding_cache
                )
                result = analyze_cell(
                    brain_rdms,
                    model_rdms,
                    cv_repeats=args.cv_repeats,
                    cv_seed=args.cv_seed,
                )
                del model_rdms, encoding_cache
                gc.collect()
                cell_rows.append({
                    "subject": subject,
                    "stimulus_group": group,
                    "stimulus_type": "controversial",
                    "rsa_type": rsa_type,
                    "bootstrap_idx": 0,
                    "subsample_idx": 0,
                    "baseline_sampling": "none",
                    "n_stimuli": len(keys),
                    **result,
                })

            _add_loso_metrics(cell_rows)
            for row in cell_rows:
                print(f"    {row['subject']:6s}  "
                      f"single={row['r_single_best']:.3f}  "
                      f"ens_cv={row['r_ensemble_cv']:.3f}  "
                      f"ceil={row['correlation_ceiling']:.3f}  "
                      f"loso_resid={row['r_loso_residual']:.3f}", flush=True)
                rows.append(_public_row(row))
            save_progress()

        # Vicco: precompute full 292×292 RDMs once per subject, then subset per
        # bootstrap. The same bootstrap indices are used across subjects so LOSO
        # residual correlations compare the same image pairs.
        if "vicco" not in requested_groups:
            continue
        print("  vicco", flush=True)
        vicco_precomp: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
        n_vicco = None
        for subject, sdata in subject_data.items():
            if "vicco" not in sdata["group_keys"]:
                continue
            vicco_keys = sdata["group_keys"]["vicco"]
            vicco_stim_idx = sdata["group_stim_idx"]["vicco"]
            if len(vicco_keys) < VICCO_SAMPLE_SIZE:
                print(f"    {subject}: only {len(vicco_keys)} vicco images, skipping")
                continue
            if n_vicco is None:
                n_vicco = len(vicco_keys)
            elif n_vicco != len(vicco_keys):
                raise ValueError("Subjects have different vicco stimulus counts")

            print(f"    {subject}: precomputing full {len(vicco_keys)}×{len(vicco_keys)} RDMs...",
                  flush=True)
            brain_rdms_full = _brain_rdm_matrices(vicco_keys, sdata["betas_by_rep"])
            encoding_cache: dict | None = {} if rsa_type == "fixed" else None
            model_rdms_full = _model_rdm_matrices(
                "vicco", vicco_stim_idx, subject, rsa_type, encoding_cache
            )
            del encoding_cache
            gc.collect()
            vicco_precomp[subject] = (brain_rdms_full, model_rdms_full)

        if n_vicco is None:
            continue

        # The shared helper samples without replacement; this is an N-matched
        # baseline subsampling scheme, not a bootstrap in the statistical sense.
        boots = bootstrap_sample_indices(
            n_total=n_vicco, n_sample=VICCO_SAMPLE_SIZE,
            n_bootstrap=n_vicco_subsamples, seed=0,
        )
        for bidx, sub_idx in enumerate(tqdm(boots, desc=f"  vicco [{rsa_type}]")):
            cell_rows = []
            for subject, (brain_rdms_full, model_rdms_full) in vicco_precomp.items():
                result = analyze_cell(
                    brain_rdms_full, model_rdms_full,
                    sub_idx=sub_idx, n_stim_override=VICCO_SAMPLE_SIZE,
                    cv_repeats=args.cv_repeats,
                    cv_seed=args.cv_seed,
                )
                cell_rows.append({
                    "subject": subject,
                    "stimulus_group": "vicco",
                    "stimulus_type": "vicco",
                    "rsa_type": rsa_type,
                    "bootstrap_idx": bidx,
                    "subsample_idx": bidx,
                    "baseline_sampling": "n_matched_without_replacement",
                    "n_stimuli": VICCO_SAMPLE_SIZE,
                    **result,
                })

            _add_loso_metrics(cell_rows)
            rows.extend(_public_row(row) for row in cell_rows)
        save_progress()

    print(f"\nDone. Wrote {out_path}")


if __name__ == "__main__":
    main()
