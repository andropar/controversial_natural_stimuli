#!/usr/bin/env python3
"""Compute set-specific target-adaptation encoding scores.

This is the revised version of the quick pooled-CSTIM LOSO analysis.  The
selected layer is still fixed to the dense layer-sweep best-on-shared choice,
but target samples are included only from the target set being evaluated:

    DeepVision unique + one 100-image CSTIM target set -> CSTIM analytic LOSO

For the baseline diagnostic, Vicco is handled symmetrically in a separate fit:

    DeepVision unique + Vicco -> Vicco analytic LOSO

The script also scores held-out Vicco predictions from each CSTIM-adapted fit,
so baseline behavior can be compared under the same fitted models that produced
the CSTIM LOSO predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import _paths  # noqa: F401
from _paths import CACHE_DIR, LAYER_SWEEP_ROOT, RESULTS_DIR, SHARE_ROOT, SOURCE_LAYER_SWEEP_ROOT

import numpy as np
import pandas as pd
import scipy.linalg
from scipy import stats

from cstims.paper import config
from cstims.datasets.deepvision import DeepVisionBenchmark


CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
BASELINE_SET = "vicco"
DEFAULT_WEIGHTS = "0,0.25,0.5,1,2,4,8"

SELECTION_CSV = LAYER_SWEEP_ROOT / "results" / "mrsa_dense_layer_selection_transfer.csv"
SOURCE_FEATURE_DIR = SOURCE_LAYER_SWEEP_ROOT / "cache" / "features"
SOURCE_DV_FEATURE_DIR = SOURCE_LAYER_SWEEP_ROOT / "cache" / "dv_features"
SOURCE_ENCODING_DIR = SOURCE_LAYER_SWEEP_ROOT / "cache" / "encodings"
LOCAL_SELECTED_CACHE_DIR = CACHE_DIR / "selected_layer_features"
LOCAL_FEATURE_DIR = LOCAL_SELECTED_CACHE_DIR / "features"
LOCAL_DV_FEATURE_DIR = LOCAL_SELECTED_CACHE_DIR / "dv_features"
FEATURE_DIRS = (LOCAL_FEATURE_DIR, SOURCE_FEATURE_DIR)
DV_FEATURE_DIRS = (LOCAL_DV_FEATURE_DIR, SOURCE_DV_FEATURE_DIR)

DV_CACHE_ROOT = SHARE_ROOT / "01_brain_model_alignment" / "cache_or_heavy" / "deepvision_benchmark_cache"
BRAIN_CACHE_ROOT = (
    SHARE_ROOT / "01_brain_model_alignment" / "cache_or_heavy" / "cstim_brain_response_cache" / "data"
)

SCORE_CSV = RESULTS_DIR / "target_adaptation_weighted_scores.csv"
SUMMARY_CSV = RESULTS_DIR / "target_adaptation_weighted_summary.csv"
AUDIT_CSV = RESULTS_DIR / "target_adaptation_cached_selection_audit.csv"
RUN_META_JSON = RESULTS_DIR / "target_adaptation_run_metadata.json"


@dataclass(frozen=True)
class Selection:
    subject: str
    model: str
    display_name: str
    layer: str


def sanitize_layer_name(layer: str) -> str:
    return (
        str(layer)
        .replace(".", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("/", "_")
    )


def stable_seed(*parts: str, base: int = 20260607) -> int:
    text = "::".join(str(p) for p in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return (int.from_bytes(digest, "little") + int(base)) % (2**31 - 1)


def parse_weights(spec: str) -> list[float]:
    values = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(float(chunk))
    if not values:
        raise ValueError("No target weights provided")
    return values


def load_best_shared_selections() -> pd.DataFrame:
    df = pd.read_csv(SELECTION_CSV)
    rows = df[
        df["selection_rule"].eq("best_on_shared")
        & df["selection_model_set"].eq("deepvision_shared")
        & df["eval_target"].eq("shared")
    ].copy()
    rows = rows[
        [
            "subject",
            "model",
            "display_name",
            "selected_layer",
            "selected_layer_index",
            "selected_layer_frac",
            "selection_mrsa",
        ]
    ].drop_duplicates(["subject", "model"])
    return rows.rename(columns={"selected_layer": "layer"})


def load_original_reference() -> pd.DataFrame:
    df = pd.read_csv(SELECTION_CSV)
    rows = df[
        df["selection_rule"].eq("best_on_shared")
        & df["selection_model_set"].eq("deepvision_shared")
        & df["eval_target"].isin(["cstim", BASELINE_SET])
    ].copy()
    out = rows[
        [
            "subject",
            "model",
            "eval_target",
            "eval_model_set",
            "mrsa_mean",
            "mrsa_sem",
            "selected_layer",
        ]
    ].rename(
        columns={
            "eval_model_set": "model_set",
            "mrsa_mean": "original_best_shared_mrsa",
            "mrsa_sem": "original_best_shared_mrsa_sem",
            "selected_layer": "layer",
        }
    )
    out.loc[out["eval_target"].eq(BASELINE_SET), "model_set"] = BASELINE_SET
    return out


def npz_has_key(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            return key in z.files
    except Exception:
        return False


def cstim_feature_paths(model: str, stimulus_set: str) -> list[Path]:
    return [root / model / f"{stimulus_set}.npz" for root in FEATURE_DIRS]


def deepvision_feature_paths(subject: str, model: str) -> list[Path]:
    return [root / subject / f"{model}.npz" for root in DV_FEATURE_DIRS]


def npz_has_key_any(paths: list[Path], key: str) -> bool:
    return any(npz_has_key(path, key) for path in paths)


def find_npz_with_key(paths: list[Path], key: str) -> Path:
    for path in paths:
        if npz_has_key(path, key):
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find key {key!r} in any of: {joined}")


def encoding_path(subject: str, model: str, layer: str) -> Path:
    return (
        SOURCE_ENCODING_DIR
        / subject
        / f"{model}.layer{sanitize_layer_name(layer)}"
        / "encoding_model.npz"
    )


def audit_selection_cache(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in selections.itertuples(index=False):
        feature_reasons = []
        for cset in [*CSTIM_SETS, BASELINE_SET]:
            if not npz_has_key_any(cstim_feature_paths(row.model, cset), row.layer):
                feature_reasons.append(f"missing_target_feature:{cset}")
        if not npz_has_key_any(deepvision_feature_paths(row.subject, row.model), row.layer):
            feature_reasons.append("missing_deepvision_unique_feature")
        alpha_ready = encoding_path(row.subject, row.model, row.layer).exists()
        reasons = list(feature_reasons)
        if not alpha_ready:
            reasons.append("missing_existing_alpha_cache")
        rows.append(
            {
                "subject": row.subject,
                "model": row.model,
                "display_name": row.display_name,
                "selected_layer": row.layer,
                "feature_cache_ready": len(feature_reasons) == 0,
                "alpha_cache_ready": bool(alpha_ready),
                "cache_ready": len(feature_reasons) == 0,
                "missing_reason": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def load_feature(path: Path, layer: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        arr = z[layer]
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_feature_from_paths(paths: list[Path], layer: str) -> np.ndarray:
    return load_feature(find_npz_with_key(paths, layer), layer)


def project_if_needed(
    arrays: list[np.ndarray],
    *,
    model: str,
    layer: str,
    target_dim: int,
) -> tuple[list[np.ndarray], int, bool]:
    dim = int(arrays[0].shape[1])
    if target_dim <= 0 or dim <= target_dim:
        return [np.ascontiguousarray(a, dtype=np.float32) for a in arrays], dim, False

    rng = np.random.default_rng(stable_seed(model, layer, "target_adaptation_projection"))
    proj = rng.normal(0.0, 1.0 / math.sqrt(target_dim), size=(dim, target_dim)).astype(
        np.float32
    )
    out = [np.ascontiguousarray(a @ proj, dtype=np.float32) for a in arrays]
    return out, target_dim, True


def load_deepvision_responses(subject: str) -> tuple[np.ndarray, np.ndarray]:
    bench = DeepVisionBenchmark(
        cache_root=DV_CACHE_ROOT,
        subject=subject,
        voxel_set="visual",
        input_source="finalinterp",
        image_set="unique",
        n_jobs=1,
    )
    responses = bench.response_data.to_numpy(dtype=np.float32)
    hlvis = bench.get_roi_mask("hlvis").astype(bool)
    return responses[hlvis].T.astype(np.float32), hlvis


def load_cstim_subject_data(subject: str) -> dict:
    data_dir = BRAIN_CACHE_ROOT / subject
    betas_data = np.load(data_dir / "cstim_betas_averaged.npz", allow_pickle=True)
    voxel_data = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")

    hlvis = voxel_data["hlvis_mask"].astype(bool)
    betas_hlvis = betas_data["betas"][hlvis, :].astype(np.float32)
    key_to_idx = {k: i for i, k in enumerate(betas_data["stim_keys"])}

    group_brain_idx = {}
    group_file_idx = {}
    for group in [*CSTIM_SETS, BASELINE_SET]:
        block = stim_info[stim_info["group"].eq(group)]
        keys = block["stim_key"].to_numpy()
        group_brain_idx[group] = np.array([key_to_idx[k] for k in keys], dtype=int)
        file_idx = block["stim_idx"].to_numpy(dtype=int)
        if group == BASELINE_SET:
            file_idx = file_idx - 1
        group_file_idx[group] = file_idx

    return {
        "betas_hlvis": betas_hlvis,
        "group_brain_idx": group_brain_idx,
        "group_file_idx": group_file_idx,
    }


def standardize_from_reference(
    train: np.ndarray, *others: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, eps)
    train_z = ((train - mean) / scale).astype(np.float32)
    other_z = [((arr - mean) / scale).astype(np.float32) for arr in others]
    return train_z, other_z, mean, scale


def rdm_corr(features: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(np.asarray(features, dtype=np.float64))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    rdm = 1.0 - corr
    np.fill_diagonal(rdm, 0.0)
    return rdm


def rsa_spearman(pred: np.ndarray, brain: np.ndarray) -> float:
    pr = rdm_corr(pred)
    br = rdm_corr(brain)
    idx = np.triu_indices(pr.shape[0], k=1)
    val = stats.spearmanr(pr[idx], br[idx]).statistic
    return float(val) if np.isfinite(val) else np.nan


def bootstrap_rsa_spearman(
    pred: np.ndarray,
    brain: np.ndarray,
    *,
    n_bootstrap: int,
    n_sample: int,
    seed: int,
) -> tuple[float, float, int, int]:
    n_total = pred.shape[0]
    n_sample = min(int(n_sample), int(n_total))
    if n_bootstrap <= 0 or n_total <= n_sample:
        score = rsa_spearman(pred, brain)
        return score, np.nan, 1, int(n_total)

    vals = []
    for i in range(n_bootstrap):
        rng = np.random.default_rng(seed + i)
        idx = np.sort(rng.choice(n_total, size=n_sample, replace=False))
        vals.append(rsa_spearman(pred[idx], brain[idx]))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, 0, n_sample
    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
    return float(vals.mean()), float(sem), int(len(vals)), n_sample


def fit_weighted_loso_predictions(
    X_dv: np.ndarray,
    Y_dv: np.ndarray,
    X_target: np.ndarray,
    Y_target: np.ndarray,
    *,
    alpha: float,
    target_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit weighted ridge and return analytic leave-one-target-out predictions."""
    n_dv = X_dv.shape[0]
    n_target = X_target.shape[0]
    X_all = np.vstack([X_dv, X_target]).astype(np.float64, copy=False)
    Y_all = np.vstack([Y_dv, Y_target]).astype(np.float64, copy=False)

    design = np.empty((X_all.shape[0], X_all.shape[1] + 1), dtype=np.float64)
    design[:, :-1] = X_all
    design[:, -1] = 1.0

    weights = np.ones(X_all.shape[0], dtype=np.float64)
    weights[n_dv:] = float(target_weight)
    sqrt_w = np.sqrt(weights)
    Xw = design * sqrt_w[:, None]
    Yw = Y_all * sqrt_w[:, None]

    gram = Xw.T @ Xw
    diag = np.ones(gram.shape[0], dtype=np.float64)
    diag[-1] = 0.0
    gram.flat[:: gram.shape[0] + 1] += float(alpha) * diag
    rhs = Xw.T @ Yw

    try:
        c_factor = scipy.linalg.cho_factor(gram, lower=True, check_finite=False)
        coef = scipy.linalg.cho_solve(c_factor, rhs, check_finite=False)
        Xt = design[n_dv:]
        yhat = Xt @ coef
        if target_weight > 0:
            inv_x = scipy.linalg.cho_solve(c_factor, Xt.T, check_finite=False)
            h = target_weight * np.einsum("ij,ji->i", Xt, inv_x, optimize=True)
        else:
            h = np.zeros(n_target, dtype=np.float64)
    except scipy.linalg.LinAlgError:
        coef = np.linalg.pinv(gram, rcond=1e-6) @ rhs
        Xt = design[n_dv:]
        yhat = Xt @ coef
        if target_weight > 0:
            inv_x = np.linalg.pinv(gram, rcond=1e-6) @ Xt.T
            h = target_weight * np.einsum("ij,ji->i", Xt, inv_x, optimize=True)
        else:
            h = np.zeros(n_target, dtype=np.float64)

    denom = np.maximum(1.0 - h, 1e-4)
    loo = (yhat - h[:, None] * Y_target) / denom[:, None]
    return loo.astype(np.float32), h.astype(np.float32), coef.astype(np.float32)


def predict_with_coef(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.empty((X.shape[0], X.shape[1] + 1), dtype=np.float32)
    design[:, :-1] = X.astype(np.float32, copy=False)
    design[:, -1] = 1.0
    return np.ascontiguousarray(design @ coef, dtype=np.float32)


def _finite_positive_alphas(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        alphas = np.asarray(z["alphas"], dtype=np.float64)
    return alphas[np.isfinite(alphas) & (alphas > 0)]


def load_global_alpha(selections: pd.DataFrame) -> float:
    medians = []
    for row in selections.itertuples(index=False):
        path = encoding_path(row.subject, row.model, row.layer)
        if not path.exists():
            continue
        try:
            alphas = _finite_positive_alphas(path)
        except Exception:
            continue
        if len(alphas):
            medians.append(float(np.median(alphas)))
    if medians:
        return float(np.median(medians))
    return 1.0e3


def load_alpha(
    subject: str,
    model: str,
    layer: str,
    *,
    fallback_alpha: float,
) -> tuple[float, float, float, str]:
    path = encoding_path(subject, model, layer)
    if path.exists():
        try:
            alphas = _finite_positive_alphas(path)
        except Exception:
            alphas = np.asarray([], dtype=np.float64)
        if len(alphas):
            return (
                float(np.median(alphas)),
                float(np.mean(alphas)),
                float(np.std(alphas)),
                "median_existing_dense_sweep_alpha",
            )
    return (
        float(fallback_alpha),
        float("nan"),
        float("nan"),
        "global_median_existing_dense_sweep_alpha",
    )


def target_sets_for_model(model: str, *, score_membership_only: bool) -> list[str]:
    if not score_membership_only:
        return list(CSTIM_SETS)
    return [model_set for model_set in CSTIM_SETS if model in set(config.MODEL_SETS[model_set])]


def original_ref_map(original_ref: pd.DataFrame, sel: Selection) -> dict:
    ref_rows = original_ref[
        original_ref["subject"].eq(sel.subject)
        & original_ref["model"].eq(sel.model)
        & original_ref["layer"].eq(sel.layer)
    ]
    return {
        (row.eval_target, row.model_set): (
            float(row.original_best_shared_mrsa),
            float(row.original_best_shared_mrsa_sem)
            if np.isfinite(row.original_best_shared_mrsa_sem)
            else np.nan,
        )
        for row in ref_rows.itertuples(index=False)
    }


def add_score_row(
    out: list[dict],
    *,
    sel: Selection,
    model_set: str,
    adaptation_target: str,
    eval_target: str,
    stimulus_type: str,
    target_weight: float,
    score: float,
    score_sem: float,
    original_score: float,
    original_sem: float,
    n_deepvision_train: int,
    n_target_train: int,
    n_stimuli_scored: int,
    n_score_bootstrap: int,
    score_sample_size: int,
    feature_dim_original: int,
    feature_dim_analysis: int,
    feature_projected: bool,
    alpha_rule: str,
    alpha_median: float,
    alpha_mean: float,
    alpha_std: float,
    hat_diag_mean: float,
    hat_diag_max: float,
    training_target_scope: str,
    runtime_seconds: float = np.nan,
) -> None:
    out.append(
        {
            "subject": sel.subject,
            "model": sel.model,
            "display_name": sel.display_name,
            "selected_layer": sel.layer,
            "model_set": model_set,
            "adaptation_target": adaptation_target,
            "eval_target": eval_target,
            "stimulus_type": stimulus_type,
            "target_weight": float(target_weight),
            "cstim_weight": float(target_weight),
            "mrsa_loso": float(score),
            "mrsa_loso_sem": float(score_sem) if np.isfinite(score_sem) else np.nan,
            "original_best_shared_mrsa": original_score,
            "original_best_shared_mrsa_sem": original_sem,
            "delta_vs_original": score - original_score
            if np.isfinite(score) and np.isfinite(original_score)
            else np.nan,
            "n_deepvision_train": int(n_deepvision_train),
            "n_target_train": int(n_target_train),
            "n_stimuli_scored": int(n_stimuli_scored),
            "n_score_bootstrap": int(n_score_bootstrap),
            "score_sample_size": int(score_sample_size),
            "feature_dim_original": int(feature_dim_original),
            "feature_dim_analysis": int(feature_dim_analysis),
            "feature_projected": bool(feature_projected),
            "feature_protocol": "selected_layer_compact_or_source_native_projected",
            "alpha_rule": alpha_rule,
            "alpha_median": float(alpha_median),
            "alpha_mean": float(alpha_mean) if np.isfinite(alpha_mean) else np.nan,
            "alpha_std": float(alpha_std) if np.isfinite(alpha_std) else np.nan,
            "hat_diag_mean": float(hat_diag_mean) if np.isfinite(hat_diag_mean) else np.nan,
            "hat_diag_max": float(hat_diag_max) if np.isfinite(hat_diag_max) else np.nan,
            "target_zscore_reference": "deepvision_unique",
            "feature_zscore_reference": "deepvision_unique",
            "training_target_scope": training_target_scope,
            "runtime_seconds_subject_model": runtime_seconds,
        }
    )


def compute_one_selection(
    sel: Selection,
    *,
    target_weights: list[float],
    analysis_dim: int,
    fallback_alpha: float,
    original_ref: pd.DataFrame,
    score_membership_only: bool,
    n_vicco_boot: int,
) -> list[dict]:
    t0 = time.time()
    X_dv_raw = load_feature_from_paths(deepvision_feature_paths(sel.subject, sel.model), sel.layer)
    Y_dv_raw, _hlvis = load_deepvision_responses(sel.subject)
    if X_dv_raw.shape[0] != Y_dv_raw.shape[0]:
        raise ValueError(
            f"DeepVision feature/response length mismatch for {sel.subject}/{sel.model}: "
            f"{X_dv_raw.shape[0]} vs {Y_dv_raw.shape[0]}"
        )

    cstim_data = load_cstim_subject_data(sel.subject)
    Y_dv_z, _unused, _y_mean, _y_scale = standardize_from_reference(Y_dv_raw)
    alpha_median, alpha_mean, alpha_std, alpha_rule = load_alpha(
        sel.subject,
        sel.model,
        sel.layer,
        fallback_alpha=fallback_alpha,
    )
    refs = original_ref_map(original_ref, sel)

    X_vicco_all = load_feature_from_paths(cstim_feature_paths(sel.model, BASELINE_SET), sel.layer)
    vicco_file_idx = cstim_data["group_file_idx"][BASELINE_SET]
    vicco_brain_idx = cstim_data["group_brain_idx"][BASELINE_SET]
    X_vicco_raw = X_vicco_all[vicco_file_idx].astype(np.float32)
    Y_vicco_raw = cstim_data["betas_hlvis"][:, vicco_brain_idx].T.astype(np.float32)
    vicco_ref, vicco_ref_sem = refs.get((BASELINE_SET, BASELINE_SET), (np.nan, np.nan))

    target_sets = target_sets_for_model(sel.model, score_membership_only=score_membership_only)
    out = []

    cstim_blocks = {}
    for group in target_sets:
        X_group_all = load_feature_from_paths(cstim_feature_paths(sel.model, group), sel.layer)
        file_idx = cstim_data["group_file_idx"][group]
        brain_idx = cstim_data["group_brain_idx"][group]
        cstim_blocks[group] = {
            "X_raw": X_group_all[file_idx].astype(np.float32),
            "Y_raw": cstim_data["betas_hlvis"][:, brain_idx].T.astype(np.float32),
            "feature_dim_original": int(X_group_all.shape[1]),
        }

    for group, block in cstim_blocks.items():
        [X_dv, X_target, X_vicco], final_dim, projected = project_if_needed(
            [X_dv_raw, block["X_raw"], X_vicco_raw],
            model=sel.model,
            layer=sel.layer,
            target_dim=analysis_dim,
        )
        X_dv_z, [X_target_z, X_vicco_z], _x_mean, _x_scale = standardize_from_reference(
            X_dv, X_target, X_vicco
        )
        Y_target_z = ((block["Y_raw"] - _y_mean) / _y_scale).astype(np.float32)
        cstim_ref, cstim_ref_sem = refs.get(("cstim", group), (np.nan, np.nan))

        for weight in target_weights:
            pred_loso, hat, coef = fit_weighted_loso_predictions(
                X_dv_z,
                Y_dv_z,
                X_target_z,
                Y_target_z,
                alpha=alpha_median,
                target_weight=weight,
            )
            cstim_score = rsa_spearman(pred_loso, block["Y_raw"])
            pred_vicco = predict_with_coef(X_vicco_z, coef)
            vicco_score, vicco_sem, vicco_n_boot, vicco_sample = bootstrap_rsa_spearman(
                pred_vicco,
                Y_vicco_raw,
                n_bootstrap=n_vicco_boot,
                n_sample=block["Y_raw"].shape[0],
                seed=stable_seed(sel.subject, sel.model, group, "heldout_vicco"),
            )
            add_score_row(
                out,
                sel=sel,
                model_set=group,
                adaptation_target=group,
                eval_target="cstim_loso",
                stimulus_type="controversial",
                target_weight=weight,
                score=cstim_score,
                score_sem=np.nan,
                original_score=cstim_ref,
                original_sem=cstim_ref_sem,
                n_deepvision_train=X_dv_z.shape[0],
                n_target_train=X_target_z.shape[0],
                n_stimuli_scored=X_target_z.shape[0],
                n_score_bootstrap=1,
                score_sample_size=X_target_z.shape[0],
                feature_dim_original=block["feature_dim_original"],
                feature_dim_analysis=final_dim,
                feature_projected=projected,
                alpha_rule=alpha_rule,
                alpha_median=alpha_median,
                alpha_mean=alpha_mean,
                alpha_std=alpha_std,
                hat_diag_mean=float(np.mean(hat)),
                hat_diag_max=float(np.max(hat)),
                training_target_scope=f"deepvision_unique_plus_{group}_target_loso",
            )
            add_score_row(
                out,
                sel=sel,
                model_set=group,
                adaptation_target=group,
                eval_target="vicco_heldout",
                stimulus_type="baseline",
                target_weight=weight,
                score=vicco_score,
                score_sem=vicco_sem,
                original_score=vicco_ref,
                original_sem=vicco_ref_sem,
                n_deepvision_train=X_dv_z.shape[0],
                n_target_train=X_target_z.shape[0],
                n_stimuli_scored=Y_vicco_raw.shape[0],
                n_score_bootstrap=vicco_n_boot,
                score_sample_size=vicco_sample,
                feature_dim_original=int(X_vicco_all.shape[1]),
                feature_dim_analysis=final_dim,
                feature_projected=projected,
                alpha_rule=alpha_rule,
                alpha_median=alpha_median,
                alpha_mean=alpha_mean,
                alpha_std=alpha_std,
                hat_diag_mean=np.nan,
                hat_diag_max=np.nan,
                training_target_scope=f"deepvision_unique_plus_{group}_target_vicco_held_out",
            )

    [X_dv, X_vicco], final_dim, projected = project_if_needed(
        [X_dv_raw, X_vicco_raw],
        model=sel.model,
        layer=sel.layer,
        target_dim=analysis_dim,
    )
    X_dv_z, [X_vicco_z], _x_mean, _x_scale = standardize_from_reference(X_dv, X_vicco)
    Y_vicco_z = ((Y_vicco_raw - _y_mean) / _y_scale).astype(np.float32)
    for weight in target_weights:
        pred_loso, hat, _coef = fit_weighted_loso_predictions(
            X_dv_z,
            Y_dv_z,
            X_vicco_z,
            Y_vicco_z,
            alpha=alpha_median,
            target_weight=weight,
        )
        vicco_score, vicco_sem, vicco_n_boot, vicco_sample = bootstrap_rsa_spearman(
            pred_loso,
            Y_vicco_raw,
            n_bootstrap=n_vicco_boot,
            n_sample=100,
            seed=stable_seed(sel.subject, sel.model, "vicco_loso"),
        )
        add_score_row(
            out,
            sel=sel,
            model_set=BASELINE_SET,
            adaptation_target=BASELINE_SET,
            eval_target="vicco_loso",
            stimulus_type="baseline",
            target_weight=weight,
            score=vicco_score,
            score_sem=vicco_sem,
            original_score=vicco_ref,
            original_sem=vicco_ref_sem,
            n_deepvision_train=X_dv_z.shape[0],
            n_target_train=X_vicco_z.shape[0],
            n_stimuli_scored=Y_vicco_raw.shape[0],
            n_score_bootstrap=vicco_n_boot,
            score_sample_size=vicco_sample,
            feature_dim_original=int(X_vicco_all.shape[1]),
            feature_dim_analysis=final_dim,
            feature_projected=projected,
            alpha_rule=alpha_rule,
            alpha_median=alpha_median,
            alpha_mean=alpha_mean,
            alpha_std=alpha_std,
            hat_diag_mean=float(np.mean(hat)),
            hat_diag_max=float(np.max(hat)),
            training_target_scope="deepvision_unique_plus_vicco_target_loso",
        )

    elapsed = time.time() - t0
    for row in out:
        row["runtime_seconds_subject_model"] = elapsed
    return out


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_summary(scores: pd.DataFrame) -> None:
    rows = []
    group_cols = ["model_set", "adaptation_target", "eval_target", "target_weight"]
    for keys, block in scores.groupby(group_cols):
        vals = block["mrsa_loso"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "mean_mrsa": float(vals.mean()) if len(vals) else np.nan,
                "sem_mrsa": float(sem) if np.isfinite(sem) else np.nan,
                "n": int(len(block)),
                "n_models": int(block["model"].nunique()),
                "n_subjects": int(block["subject"].nunique()),
            }
        )
    atomic_write_csv(pd.DataFrame(rows), SUMMARY_CSV)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--analysis-dim", type=int, default=512)
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument(
        "--score-all-model-sets",
        action="store_true",
        help="Score every cached model on every CSTIM set. Default keeps only model-set memberships.",
    )
    parser.add_argument("--n-vicco-boot", type=int, default=1000)
    parser.add_argument("--max-selections", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    weights = parse_weights(args.weights)
    selections = load_best_shared_selections()
    if args.subject != "all":
        selections = selections[selections["subject"].eq(args.subject)].copy()
    if args.models:
        selections = selections[selections["model"].isin(args.models)].copy()
    if args.max_selections:
        selections = selections.head(args.max_selections).copy()

    audit = audit_selection_cache(selections)
    atomic_write_csv(audit, AUDIT_CSV)
    ready = audit[audit["cache_ready"]].copy()
    print(
        f"Cache-ready selections: {len(ready)}/{len(audit)} "
        f"({ready['model'].nunique()} models)",
        flush=True,
    )
    if ready.empty:
        raise RuntimeError(f"No cache-ready selections. See {AUDIT_CSV}")
    fallback_alpha = load_global_alpha(selections)
    print(f"Global fallback alpha: {fallback_alpha:.6g}", flush=True)

    if SCORE_CSV.exists() and not args.overwrite:
        print(f"[cached] {SCORE_CSV} exists; use --overwrite to recompute", flush=True)
        return

    original_ref = load_original_reference()
    rows = []
    total = len(ready)
    for idx, row in enumerate(ready.itertuples(index=False), start=1):
        sel = Selection(
            subject=row.subject,
            model=row.model,
            display_name=row.display_name,
            layer=row.selected_layer,
        )
        target_sets = target_sets_for_model(
            sel.model, score_membership_only=not args.score_all_model_sets
        )
        print(
            f"[{idx:03d}/{total:03d}] {sel.subject} {sel.model} "
            f"layer={sel.layer} targets={','.join(target_sets)}",
            flush=True,
        )
        rows.extend(
            compute_one_selection(
                sel,
                target_weights=weights,
                analysis_dim=args.analysis_dim,
                fallback_alpha=fallback_alpha,
                original_ref=original_ref,
                score_membership_only=not args.score_all_model_sets,
                n_vicco_boot=args.n_vicco_boot,
            )
        )
        if idx % 5 == 0:
            scores = pd.DataFrame(rows)
            atomic_write_csv(scores, SCORE_CSV)
            write_summary(scores)
            print(f"  wrote checkpoint rows={len(rows)} -> {SCORE_CSV}", flush=True)

    scores = pd.DataFrame(rows)
    atomic_write_csv(scores, SCORE_CSV)
    write_summary(scores)
    meta = {
        "weights": weights,
        "analysis_dim": args.analysis_dim,
        "selection_csv": str(SELECTION_CSV),
        "local_feature_dir": str(LOCAL_FEATURE_DIR),
        "local_dv_feature_dir": str(LOCAL_DV_FEATURE_DIR),
        "source_feature_dir": str(SOURCE_FEATURE_DIR),
        "source_dv_feature_dir": str(SOURCE_DV_FEATURE_DIR),
        "source_encoding_dir": str(SOURCE_ENCODING_DIR),
        "fallback_alpha": fallback_alpha,
        "alpha_rule": "median existing dense-sweep alpha when present; otherwise global median",
        "n_cache_ready_selections": int(len(ready)),
        "n_total_selections_considered": int(len(audit)),
        "score_membership_only": not args.score_all_model_sets,
        "target_adaptation_scheme": "deepvision_unique_plus_one_target_set_with_analytic_loso",
        "baseline_adaptation_scheme": "deepvision_unique_plus_vicco_with_analytic_loso",
        "heldout_baseline_scheme": "vicco scored from each cstim-target-adapted model",
        "n_vicco_boot": int(args.n_vicco_boot),
        "feature_protocol_note": (
            "Uses the existing selected-layer compact/source-native cache, with optional "
            "deterministic down-projection to analysis_dim for tractability."
        ),
        "python": os.sys.executable,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    RUN_META_JSON.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(scores)} rows -> {SCORE_CSV}", flush=True)
    print(f"Wrote summary -> {SUMMARY_CSV}", flush=True)
    print(f"Wrote audit -> {AUDIT_CSV}", flush=True)


if __name__ == "__main__":
    main()
