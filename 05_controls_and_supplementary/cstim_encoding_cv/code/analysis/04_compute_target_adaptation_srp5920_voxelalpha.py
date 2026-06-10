#!/usr/bin/env python3
"""Compute true SRP5920/per-voxel-alpha target adaptation scores.

This overwrites the main target-adaptation result files with the intended
protocol:

  - exact dense best-on-shared selected layers
  - fresh flatten_srp5920_v1 features
  - per-voxel alpha selection from DeepVision unique
  - target-set-specific weighted refits
  - exact analytic LOSO for the target samples

For a fixed alpha, target weighting is computed as a Woodbury update from the
DeepVision-only ridge fit, reducing each target refit to small target-space
linear algebra while remaining exact for the fixed per-voxel alpha.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import _paths  # noqa: F401
from _paths import CACHE_DIR, LAYER_SWEEP_ROOT, RESULTS_DIR, SHARE_ROOT

import numpy as np
import pandas as pd
import scipy.linalg
from scipy import stats

import config
from cstims.datasets.deepvision import DeepVisionBenchmark
from cstims.encoding.fitting import fit_voxelwise_ridgecv_fast
from srp_utils import FEATURE_PROTOCOL, SRP_TARGET_DIM, cached_layer_current


CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
BASELINE_SET = "vicco"
DEFAULT_WEIGHTS = "0,0.25,0.5,1,2,4,8"
ALPHA_GRID = np.logspace(np.log10(0.1), np.log10(1.0e7), 20).astype(np.float32)
DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_N_VICCO_BOOT = 1000
VICCO_BOOT_SAMPLE_SIZE = 100
VICCO_BOOT_SEED = 0
RESPONSE_ZSCORE_EPS = 1e-6
FEATURE_ZSCORE_EPS = 1e-6

SELECTION_CSV = LAYER_SWEEP_ROOT / "results" / "mrsa_dense_layer_selection_transfer.csv"
LOCAL_SELECTED_CACHE_DIR = CACHE_DIR / "selected_layer_features_srp5920"
LOCAL_FEATURE_DIR = LOCAL_SELECTED_CACHE_DIR / "features"
LOCAL_DV_FEATURE_DIR = LOCAL_SELECTED_CACHE_DIR / "dv_features"
LOCAL_ALPHA_DIR = CACHE_DIR / "target_adaptation_srp5920" / "alphas"

DV_CACHE_ROOT = SHARE_ROOT / "01_brain_model_alignment" / "cache_or_heavy" / "brain_data"
BRAIN_CACHE_ROOT = (
    SHARE_ROOT / "01_brain_model_alignment" / "cache_or_heavy" / "brain_data_cache" / "data"
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


def feature_path(model: str, stimulus_set: str) -> Path:
    return LOCAL_FEATURE_DIR / model / f"{stimulus_set}.npz"


def dv_feature_path(subject: str, model: str) -> Path:
    return LOCAL_DV_FEATURE_DIR / subject / f"{model}.npz"


def alpha_path(subject: str, model: str, layer: str) -> Path:
    return LOCAL_ALPHA_DIR / subject / f"{model}.layer{sanitize_layer_name(layer)}.npz"


def cached_feature_ready(path: Path, layer: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            return cached_layer_current(z, layer, target_dim=SRP_TARGET_DIM)
    except Exception:
        return False


def alpha_cache_current(path: Path, *, n_voxels: int, n_folds: int, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            if str(np.asarray(z["feature_protocol"]).item()) != FEATURE_PROTOCOL:
                return False
            if int(np.asarray(z["feature_dim"]).item()) != SRP_TARGET_DIM:
                return False
            if int(np.asarray(z["n_folds"]).item()) != int(n_folds):
                return False
            if int(np.asarray(z["seed"]).item()) != int(seed):
                return False
            if z["alphas"].shape[0] != n_voxels:
                return False
            return np.allclose(z["alpha_grid"], ALPHA_GRID)
    except Exception:
        return False


def audit_selection_cache(
    selections: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
    n_voxels_by_subject: dict[str, int] | None = None,
) -> pd.DataFrame:
    rows = []
    for row in selections.itertuples(index=False):
        feature_reasons = []
        for stim_set in [*CSTIM_SETS, BASELINE_SET]:
            if not cached_feature_ready(feature_path(row.model, stim_set), row.layer):
                feature_reasons.append(f"missing_srp5920_target_feature:{stim_set}")
        if not cached_feature_ready(dv_feature_path(row.subject, row.model), row.layer):
            feature_reasons.append("missing_srp5920_deepvision_unique_feature")
        alpha_ready = False
        if n_voxels_by_subject and row.subject in n_voxels_by_subject:
            alpha_ready = alpha_cache_current(
                alpha_path(row.subject, row.model, row.layer),
                n_voxels=n_voxels_by_subject[row.subject],
                n_folds=n_folds,
                seed=seed,
            )
        rows.append(
            {
                "subject": row.subject,
                "model": row.model,
                "display_name": row.display_name,
                "selected_layer": row.layer,
                "feature_cache_ready": len(feature_reasons) == 0,
                "alpha_cache_ready": bool(alpha_ready),
                "cache_ready": len(feature_reasons) == 0,
                "missing_reason": ";".join(feature_reasons),
            }
        )
    return pd.DataFrame(rows)


def load_feature(path: Path, layer: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        if not cached_layer_current(z, layer, target_dim=SRP_TARGET_DIM):
            raise RuntimeError(f"Stale/non-SRP5920 feature cache: {path} layer={layer}")
        arr = z[layer]
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] != SRP_TARGET_DIM:
        raise ValueError(f"{path} {layer}: expected {SRP_TARGET_DIM}, got {arr.shape[1]}")
    return np.ascontiguousarray(arr, dtype=np.float32)


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


def bootstrap_sample_indices(
    n_total: int,
    n_sample: int,
    *,
    n_bootstrap: int,
    seed: int = VICCO_BOOT_SEED,
) -> list[np.ndarray]:
    """Match the manuscript/layer-sweep bootstrap convention exactly."""
    samples = []
    for i in range(n_bootstrap):
        rng = np.random.default_rng(seed + i)
        idx = rng.choice(n_total, size=n_sample, replace=False)
        samples.append(np.sort(idx))
    return samples


def zscore_targets_by_voxel(targets: np.ndarray):
    mean = targets.mean(axis=0, dtype=np.float64)
    std = targets.std(axis=0, dtype=np.float64, ddof=0)
    std = np.maximum(std, RESPONSE_ZSCORE_EPS)
    standardized = (targets - mean) / std
    return standardized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def zscore_features_from_deepvision(features: np.ndarray):
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64, ddof=0).astype(np.float32)
    scale = np.maximum(scale, FEATURE_ZSCORE_EPS)
    standardized = (features - mean) / scale
    return standardized.astype(np.float32), mean, scale


def apply_feature_zscore(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((features - mean) / scale).astype(np.float32)


def layer_sweep_eval_design(
    features_z: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    """Evaluation design used by the dense layer-sweep stream scorer.

    The dense layer-sweep fitter stores raw-space ridge weights after fitting in
    standardized feature space. The stream scorer then calls the shared
    prediction helper, which standardizes the evaluation features before
    applying those stored raw-space weights. Algebraically, for a standardized
    coefficient vector this is equivalent to evaluating with
    ``features_z / feature_scale - feature_mean / feature_scale``.
    """
    scale = np.maximum(np.asarray(feature_scale, dtype=np.float64), FEATURE_ZSCORE_EPS)
    mean_over_scale = np.asarray(feature_mean, dtype=np.float64) / scale
    return np.asarray(features_z, dtype=np.float64) / scale[None, :] - mean_over_scale[None, :]


def atomic_savez_compressed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def fit_or_load_alpha_cache(
    *,
    sel: Selection,
    X_dv_raw: np.ndarray,
    Y_dv_raw: np.ndarray,
    n_folds: int,
    seed: int,
    overwrite: bool,
) -> dict:
    path = alpha_path(sel.subject, sel.model, sel.layer)
    n_voxels = Y_dv_raw.shape[1]
    if not overwrite and alpha_cache_current(path, n_voxels=n_voxels, n_folds=n_folds, seed=seed):
        with np.load(path, allow_pickle=True) as z:
            return {key: z[key] for key in z.files}

    t0 = time.time()
    Y_z, Y_mean, Y_std = zscore_targets_by_voxel(Y_dv_raw)
    _X_z, feature_mean, feature_scale = zscore_features_from_deepvision(X_dv_raw)

    rng = np.random.RandomState(seed)
    fold_alphas = []
    n_images = X_dv_raw.shape[0]
    for fold in range(n_folds):
        indices = rng.permutation(n_images)
        train_idx = indices[: n_images // 2]
        test_idx = indices[n_images // 2 :]
        print(
            f"    alpha fold {fold + 1}/{n_folds}: "
            f"train={len(train_idx)} test={len(test_idx)}",
            flush=True,
        )
        _pred, alphas, _scaler = fit_voxelwise_ridgecv_fast(
            X_train=X_dv_raw[train_idx],
            Y_train=Y_z[train_idx],
            X_test=X_dv_raw[test_idx],
            alphas=ALPHA_GRID,
            verbose=False,
            scale_features=True,
        )
        fold_alphas.append(alphas.astype(np.float32))

    fold_alphas_array = np.stack(fold_alphas, axis=0).astype(np.float32)
    chosen_alphas = np.median(fold_alphas_array, axis=0).astype(np.float32)
    payload = {
        "alphas": chosen_alphas,
        "fold_alphas": fold_alphas_array,
        "alpha_grid": ALPHA_GRID.astype(np.float32),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "Y_mean_zscore": Y_mean.astype(np.float32),
        "Y_std_zscore": Y_std.astype(np.float32),
        "feature_protocol": np.array(FEATURE_PROTOCOL),
        "feature_dim": np.array(SRP_TARGET_DIM, dtype=np.int32),
        "fit_protocol": np.array("hydra_random_half_split_ridgecv_v1"),
        "n_folds": np.array(n_folds, dtype=np.int32),
        "seed": np.array(seed, dtype=np.int32),
        "alpha_aggregation": np.array("median"),
        "n_train": np.array(n_images, dtype=np.int32),
        "runtime_seconds": np.array(time.time() - t0, dtype=np.float32),
    }
    atomic_savez_compressed(path, payload)
    print(f"    wrote alpha cache {path} ({time.time() - t0:.1f}s)", flush=True)
    return payload


def rdm_corr(features: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(np.asarray(features, dtype=np.float64))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    rdm = 1.0 - corr
    np.fill_diagonal(rdm, 0.0)
    return rdm


def rdm_upper_vec(rdm: np.ndarray) -> np.ndarray:
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def rank_vector(vec: np.ndarray) -> np.ndarray:
    return stats.rankdata(vec, method="average").astype(np.float32)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    den = float(np.sqrt(np.dot(xm, xm) * np.dot(ym, ym)))
    if den <= 0:
        return float("nan")
    return float(np.dot(xm, ym) / den)


def rsa_spearman(pred: np.ndarray, brain: np.ndarray) -> float:
    pr = rdm_corr(pred)
    br = rdm_corr(brain)
    idx = np.triu_indices(pr.shape[0], k=1)
    val = stats.spearmanr(pr[idx], br[idx]).statistic
    return float(val) if np.isfinite(val) else np.nan


def load_or_compute_vicco_bootstrap(
    *,
    subject: str,
    betas_hlvis: np.ndarray,
    vicco_brain_idx: np.ndarray,
    n_bootstrap: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Return Vicco bootstrap indices and pre-ranked brain RDMs.

    The dense layer-sweep cache is preferred so the target-adaptation run uses
    the same bootstrap samples and brain-rank vectors as the Fig. 2 source
    table.
    """
    n_vicco = int(len(vicco_brain_idx))
    n_sample = min(VICCO_BOOT_SAMPLE_SIZE, n_vicco)
    if n_bootstrap <= 0 or n_sample <= 0:
        return [], [], n_sample
    cache_path = (
        LAYER_SWEEP_ROOT
        / "cache_or_heavy"
        / "brain_ranks"
        / f"cstim_{subject}_vicco{n_bootstrap}.npz"
    )
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as z:
                boot = [idx.astype(int) for idx in z["vicco_bootstrap"]]
                ranks = [r.astype(np.float32) for r in z["vicco_ranks"]]
                cached_n = int(np.asarray(z["n_vicco_sample"]).item())
            if len(boot) == n_bootstrap and len(ranks) == n_bootstrap and cached_n == n_sample:
                return boot, ranks, cached_n
        except Exception:
            pass

    boot = bootstrap_sample_indices(
        n_vicco,
        n_sample,
        n_bootstrap=n_bootstrap,
        seed=VICCO_BOOT_SEED,
    )
    ranks = []
    for idx in boot:
        brain = betas_hlvis[:, vicco_brain_idx[idx]].T
        ranks.append(rank_vector(rdm_upper_vec(rdm_corr(brain))))
    return boot, ranks, n_sample


def rsa_spearman_bootstrap_mean(
    pred: np.ndarray,
    *,
    boot: list[np.ndarray],
    brain_ranks: list[np.ndarray],
) -> tuple[float, float]:
    vals = []
    for idx, brain_rank in zip(boot, brain_ranks):
        pred_rank = rank_vector(rdm_upper_vec(rdm_corr(pred[idx])))
        vals.append(pearson_r(pred_rank, brain_rank))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
    return float(vals.mean()), sem


def decompose_deepvision_kernel(X_dv_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    X = np.asarray(X_dv_z, dtype=np.float64)
    kernel = X @ X.T
    kernel = (kernel + kernel.T) * 0.5
    eigvals, eigvecs = scipy.linalg.eigh(kernel, overwrite_a=True, check_finite=False)
    eigvals = np.maximum(eigvals.astype(np.float64), 0.0)
    print(f"    deepvision kernel eigendecomp {len(eigvals)}x{len(eigvals)} in {time.time() - t0:.1f}s", flush=True)
    return eigvals, eigvecs.astype(np.float64, copy=False)


def alpha_groups(alphas: np.ndarray) -> list[tuple[float, np.ndarray]]:
    groups = []
    for alpha in np.unique(alphas):
        mask = np.isclose(alphas, alpha)
        idx = np.where(mask)[0]
        if len(idx):
            groups.append((float(alpha), idx))
    return groups


def weighted_predictions_for_target(
    *,
    X_dv_z: np.ndarray,
    Y_dv_z: np.ndarray,
    X_target_z: np.ndarray,
    Y_target_z: np.ndarray,
    X_eval_z: np.ndarray | None,
    alphas: np.ndarray,
    weights: list[float],
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    layer_sweep_eval: bool,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray] | None, dict[float, np.ndarray]]:
    """Return target LOSO predictions and optional held-out eval predictions."""
    n_dv = X_dv_z.shape[0]
    n_voxels = Y_dv_z.shape[1]
    target_pred = {
        weight: np.empty((X_target_z.shape[0], n_voxels), dtype=np.float32)
        for weight in weights
    }
    eval_pred = None
    if X_eval_z is not None:
        eval_pred = {
            weight: np.empty((X_eval_z.shape[0], n_voxels), dtype=np.float32)
            for weight in weights
        }
    hats = {weight: np.empty(X_target_z.shape[0], dtype=np.float32) for weight in weights}

    X_dv64 = np.asarray(X_dv_z, dtype=np.float64)
    Xt = np.asarray(X_target_z, dtype=np.float64)
    Yt = np.asarray(Y_target_z, dtype=np.float64)
    Rt = Xt @ X_dv64.T
    Rtq = Rt @ eigvecs
    TT = Xt @ Xt.T
    if layer_sweep_eval:
        Xt_eval = layer_sweep_eval_design(Xt, feature_mean, feature_scale)
    else:
        Xt_eval = Xt
    Rt_eval = Xt_eval @ X_dv64.T
    Rtq_eval = Rt_eval @ eigvecs
    TT_eval = Xt_eval @ Xt.T
    if X_eval_z is not None:
        Xe = np.asarray(X_eval_z, dtype=np.float64)
        if layer_sweep_eval:
            Xe_eval = layer_sweep_eval_design(Xe, feature_mean, feature_scale)
        else:
            Xe_eval = Xe
        Re_eval = Xe_eval @ X_dv64.T
        Req_eval = Re_eval @ eigvecs
        ET_eval = Xe_eval @ Xt.T
    else:
        Req_eval = None
        ET_eval = None

    for alpha, vox_idx in alpha_groups(alphas):
        inv = 1.0 / (eigvals + float(alpha))
        Y_group = np.asarray(Y_dv_z[:, vox_idx], dtype=np.float64)
        Yq = eigvecs.T @ Y_group
        base_t_train = Rtq @ (Yq * inv[:, None])
        base_t_eval = Rtq_eval @ (Yq * inv[:, None])
        K = (TT - (Rtq * inv[None, :]) @ Rtq.T) / float(alpha)
        K += 1.0 / float(n_dv)
        K = (K + K.T) * 0.5
        K_eval = (TT_eval - (Rtq_eval * inv[None, :]) @ Rtq.T) / float(alpha)
        K_eval += 1.0 / float(n_dv)
        if X_eval_z is not None and Req_eval is not None and ET_eval is not None:
            base_e = Req_eval @ (Yq * inv[:, None])
            C = (ET_eval - (Req_eval * inv[None, :]) @ Rtq.T) / float(alpha)
            C += 1.0 / float(n_dv)
        else:
            base_e = None
            C = None

        residual = Yt[:, vox_idx] - base_t_train
        for weight in weights:
            if weight <= 0:
                pred_t = base_t_eval
                h = np.zeros(X_target_z.shape[0], dtype=np.float64)
                if eval_pred is not None and base_e is not None:
                    eval_pred[weight][:, vox_idx] = base_e.astype(np.float32)
            else:
                B = K + np.eye(K.shape[0], dtype=np.float64) / float(weight)
                try:
                    cf = scipy.linalg.cho_factor(B, lower=True, check_finite=False)
                except scipy.linalg.LinAlgError:
                    B = B + np.eye(B.shape[0], dtype=np.float64) * 1e-8
                    cf = scipy.linalg.cho_factor(B, lower=True, check_finite=False)
                solved_residual = scipy.linalg.cho_solve(cf, residual, check_finite=False)
                pred_t = base_t_eval + K_eval @ solved_residual
                solved_eye = scipy.linalg.cho_solve(
                    cf, np.eye(K.shape[0], dtype=np.float64), check_finite=False
                )
                solved_k = scipy.linalg.cho_solve(cf, K, check_finite=False)
                h = np.clip(np.diag(solved_k), 0.0, 0.999999)
                eval_leverage = np.sum(K_eval * solved_eye.T, axis=1)
                loo_scale = eval_leverage / np.maximum(np.diag(solved_eye), 1e-12)
                if eval_pred is not None and C is not None and base_e is not None:
                    eval_pred[weight][:, vox_idx] = (base_e + C @ solved_residual).astype(
                        np.float32
                    )
            if weight <= 0:
                loo = pred_t
            else:
                loo = pred_t - loo_scale[:, None] * solved_residual
            target_pred[weight][:, vox_idx] = loo.astype(np.float32)
            hats[weight] = h.astype(np.float32)
    return target_pred, eval_pred, hats


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
    alpha_values: np.ndarray,
    hat_diag: np.ndarray | None,
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
            "feature_dim_original": int(SRP_TARGET_DIM),
            "feature_dim_analysis": int(SRP_TARGET_DIM),
            "feature_projected": True,
            "feature_protocol": FEATURE_PROTOCOL,
            "alpha_rule": "per_voxel_deepvision_unique_ridgecv",
            "alpha_median": float(np.median(alpha_values)),
            "alpha_mean": float(np.mean(alpha_values)),
            "alpha_std": float(np.std(alpha_values)),
            "n_alpha_unique": int(len(np.unique(alpha_values))),
            "hat_diag_mean": float(np.mean(hat_diag)) if hat_diag is not None else np.nan,
            "hat_diag_max": float(np.max(hat_diag)) if hat_diag is not None else np.nan,
            "target_zscore_reference": "deepvision_unique",
            "feature_zscore_reference": "deepvision_unique",
            "prediction_protocol": "layer_sweep_stream_predict_v1",
            "training_target_scope": training_target_scope,
            "runtime_seconds_subject_model": runtime_seconds,
        }
    )


def compute_one_selection(
    sel: Selection,
    *,
    target_weights: list[float],
    original_ref: pd.DataFrame,
    score_membership_only: bool,
    n_folds: int,
    seed: int,
    overwrite_alpha: bool,
    n_vicco_boot: int,
) -> list[dict]:
    t0 = time.time()
    X_dv_raw = load_feature(dv_feature_path(sel.subject, sel.model), sel.layer)
    Y_dv_raw, _hlvis = load_deepvision_responses(sel.subject)
    if X_dv_raw.shape[0] != Y_dv_raw.shape[0]:
        raise ValueError(
            f"DeepVision feature/response length mismatch for {sel.subject}/{sel.model}: "
            f"{X_dv_raw.shape[0]} vs {Y_dv_raw.shape[0]}"
        )

    alpha_payload = fit_or_load_alpha_cache(
        sel=sel,
        X_dv_raw=X_dv_raw,
        Y_dv_raw=Y_dv_raw,
        n_folds=n_folds,
        seed=seed,
        overwrite=overwrite_alpha,
    )
    alphas = np.asarray(alpha_payload["alphas"], dtype=np.float32)
    feature_mean = np.asarray(alpha_payload["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(alpha_payload["feature_scale"], dtype=np.float32)
    Y_mean = np.asarray(alpha_payload["Y_mean_zscore"], dtype=np.float32)
    Y_std = np.asarray(alpha_payload["Y_std_zscore"], dtype=np.float32)
    X_dv_z = apply_feature_zscore(X_dv_raw, feature_mean, feature_scale)
    Y_dv_z = ((Y_dv_raw - Y_mean) / Y_std).astype(np.float32)
    eigvals, eigvecs = decompose_deepvision_kernel(X_dv_z)

    cstim_data = load_cstim_subject_data(sel.subject)
    refs = original_ref_map(original_ref, sel)
    X_vicco_all = load_feature(feature_path(sel.model, BASELINE_SET), sel.layer)
    vicco_file_idx = cstim_data["group_file_idx"][BASELINE_SET]
    vicco_brain_idx = cstim_data["group_brain_idx"][BASELINE_SET]
    X_vicco_raw = X_vicco_all[vicco_file_idx].astype(np.float32)
    X_vicco_z = apply_feature_zscore(X_vicco_raw, feature_mean, feature_scale)
    Y_vicco_raw = cstim_data["betas_hlvis"][:, vicco_brain_idx].T.astype(np.float32)
    Y_vicco_z = ((Y_vicco_raw - Y_mean) / Y_std).astype(np.float32)
    vicco_ref, vicco_ref_sem = refs.get((BASELINE_SET, BASELINE_SET), (np.nan, np.nan))
    vicco_boot, vicco_brain_ranks, vicco_sample_size = load_or_compute_vicco_bootstrap(
        subject=sel.subject,
        betas_hlvis=cstim_data["betas_hlvis"],
        vicco_brain_idx=vicco_brain_idx,
        n_bootstrap=n_vicco_boot,
    )

    out = []
    for group in target_sets_for_model(sel.model, score_membership_only=score_membership_only):
        X_group_all = load_feature(feature_path(sel.model, group), sel.layer)
        file_idx = cstim_data["group_file_idx"][group]
        brain_idx = cstim_data["group_brain_idx"][group]
        X_target_raw = X_group_all[file_idx].astype(np.float32)
        X_target_z = apply_feature_zscore(X_target_raw, feature_mean, feature_scale)
        Y_target_raw = cstim_data["betas_hlvis"][:, brain_idx].T.astype(np.float32)
        Y_target_z = ((Y_target_raw - Y_mean) / Y_std).astype(np.float32)
        cstim_ref, cstim_ref_sem = refs.get(("cstim", group), (np.nan, np.nan))
        pred_loso, pred_vicco, hats = weighted_predictions_for_target(
            X_dv_z=X_dv_z,
            Y_dv_z=Y_dv_z,
            X_target_z=X_target_z,
            Y_target_z=Y_target_z,
            X_eval_z=X_vicco_z,
            alphas=alphas,
            weights=target_weights,
            eigvals=eigvals,
            eigvecs=eigvecs,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            layer_sweep_eval=True,
        )
        assert pred_vicco is not None
        for weight in target_weights:
            cstim_score = rsa_spearman(pred_loso[weight], Y_target_raw)
            if vicco_boot:
                vicco_score, vicco_sem = rsa_spearman_bootstrap_mean(
                    pred_vicco[weight],
                    boot=vicco_boot,
                    brain_ranks=vicco_brain_ranks,
                )
                vicco_n_scored = vicco_sample_size
                vicco_n_boot = len(vicco_boot)
            else:
                vicco_score = rsa_spearman(pred_vicco[weight], Y_vicco_raw)
                vicco_sem = np.nan
                vicco_n_scored = Y_vicco_raw.shape[0]
                vicco_n_boot = 1
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
                alpha_values=alphas,
                hat_diag=hats[weight],
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
                n_stimuli_scored=vicco_n_scored,
                n_score_bootstrap=vicco_n_boot,
                score_sample_size=vicco_n_scored,
                alpha_values=alphas,
                hat_diag=None,
                training_target_scope=f"deepvision_unique_plus_{group}_target_vicco_held_out",
            )

    vicco_loso_pred, _unused_eval, vicco_hats = weighted_predictions_for_target(
        X_dv_z=X_dv_z,
        Y_dv_z=Y_dv_z,
        X_target_z=X_vicco_z,
        Y_target_z=Y_vicco_z,
        X_eval_z=None,
        alphas=alphas,
        weights=target_weights,
        eigvals=eigvals,
        eigvecs=eigvecs,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        layer_sweep_eval=True,
    )
    for weight in target_weights:
        if vicco_boot:
            vicco_score, vicco_sem = rsa_spearman_bootstrap_mean(
                vicco_loso_pred[weight],
                boot=vicco_boot,
                brain_ranks=vicco_brain_ranks,
            )
            vicco_n_scored = vicco_sample_size
            vicco_n_boot = len(vicco_boot)
        else:
            vicco_score = rsa_spearman(vicco_loso_pred[weight], Y_vicco_raw)
            vicco_sem = np.nan
            vicco_n_scored = Y_vicco_raw.shape[0]
            vicco_n_boot = 1
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
            n_stimuli_scored=vicco_n_scored,
            n_score_bootstrap=vicco_n_boot,
            score_sample_size=vicco_n_scored,
            alpha_values=alphas,
            hat_diag=vicco_hats[weight],
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


def expected_rows_for_selection(model: str, *, n_weights: int, score_membership_only: bool) -> int:
    n_target_sets = len(
        target_sets_for_model(model, score_membership_only=score_membership_only)
    )
    return n_weights * (2 * n_target_sets + 1)


def completed_resume_keys(
    existing: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    weights: list[float],
    score_membership_only: bool,
) -> set[tuple[str, str]]:
    if existing.empty:
        return set()
    required = {
        "subject",
        "model",
        "target_weight",
        "feature_dim_analysis",
        "feature_protocol",
    }
    missing = required.difference(existing.columns)
    if missing:
        raise ValueError(f"Existing score CSV missing resume columns: {sorted(missing)}")

    weight_set = {float(w) for w in weights}
    complete = set()
    for row in selections.itertuples(index=False):
        key = (row.subject, row.model)
        block = existing[existing["subject"].eq(row.subject) & existing["model"].eq(row.model)]
        if block.empty:
            continue
        expected = expected_rows_for_selection(
            row.model,
            n_weights=len(weights),
            score_membership_only=score_membership_only,
        )
        has_current_features = (
            block["feature_dim_analysis"].eq(SRP_TARGET_DIM).all()
            and block["feature_protocol"].eq(FEATURE_PROTOCOL).all()
        )
        if (
            len(block) == expected
            and set(block["target_weight"].astype(float)) == weight_set
            and has_current_features
        ):
            complete.add(key)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--score-all-model-sets", action="store_true")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-vicco-boot", type=int, default=DEFAULT_N_VICCO_BOOT)
    parser.add_argument("--max-selections", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-alpha", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep complete subject/model rows already present in the score CSV and compute only missing selections.",
    )
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
    if selections.empty:
        raise RuntimeError("No selected subject/model/layer rows after filtering")

    n_voxels_by_subject = {}
    for subject in sorted(selections["subject"].unique()):
        Y, _ = load_deepvision_responses(subject)
        n_voxels_by_subject[subject] = Y.shape[1]

    audit = audit_selection_cache(
        selections,
        n_folds=args.n_folds,
        seed=args.seed,
        n_voxels_by_subject=n_voxels_by_subject,
    )
    atomic_write_csv(audit, AUDIT_CSV)
    ready = audit[audit["feature_cache_ready"]].copy()
    print(
        f"SRP5920 feature-ready selections: {len(ready)}/{len(audit)} "
        f"({ready['model'].nunique()} models)",
        flush=True,
    )
    if ready.empty:
        raise RuntimeError(
            f"No SRP5920 feature-ready selections. Run 00_fill_selected_layer_srp5920_cache.py. "
            f"See {AUDIT_CSV}"
        )

    score_membership_only = not args.score_all_model_sets
    if SCORE_CSV.exists() and not args.overwrite and not args.resume:
        print(f"[cached] {SCORE_CSV} exists; use --overwrite to recompute", flush=True)
        return

    original_ref = load_original_reference()
    rows = []
    completed_keys: set[tuple[str, str]] = set()
    if args.resume and SCORE_CSV.exists():
        existing_scores = pd.read_csv(SCORE_CSV)
        completed_keys = completed_resume_keys(
            existing_scores,
            ready,
            weights=weights,
            score_membership_only=score_membership_only,
        )
        if completed_keys:
            keep = existing_scores.apply(
                lambda r: (r["subject"], r["model"]) in completed_keys,
                axis=1,
            )
            rows = existing_scores.loc[keep].to_dict("records")
        print(
            f"[resume] keeping {len(rows)} rows from {len(completed_keys)} complete selections; "
            f"computing {len(ready) - len(completed_keys)} missing selections",
            flush=True,
        )

    total = len(ready)
    for idx, row in enumerate(ready.itertuples(index=False), start=1):
        sel = Selection(
            subject=row.subject,
            model=row.model,
            display_name=row.display_name,
            layer=row.selected_layer,
        )
        if (sel.subject, sel.model) in completed_keys:
            continue
        target_sets = target_sets_for_model(
            sel.model, score_membership_only=score_membership_only
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
                original_ref=original_ref,
                score_membership_only=score_membership_only,
                n_folds=args.n_folds,
                seed=args.seed,
                overwrite_alpha=args.overwrite_alpha,
                n_vicco_boot=args.n_vicco_boot,
            )
        )
        if idx % 2 == 0:
            scores = pd.DataFrame(rows)
            atomic_write_csv(scores, SCORE_CSV)
            write_summary(scores)
            print(f"  wrote checkpoint rows={len(rows)} -> {SCORE_CSV}", flush=True)

    scores = pd.DataFrame(rows)
    atomic_write_csv(scores, SCORE_CSV)
    write_summary(scores)
    audit = audit_selection_cache(
        selections,
        n_folds=args.n_folds,
        seed=args.seed,
        n_voxels_by_subject=n_voxels_by_subject,
    )
    atomic_write_csv(audit, AUDIT_CSV)
    meta = {
        "weights": weights,
        "analysis_dim": SRP_TARGET_DIM,
        "feature_protocol": FEATURE_PROTOCOL,
        "selection_csv": str(SELECTION_CSV),
        "local_feature_dir": str(LOCAL_FEATURE_DIR),
        "local_dv_feature_dir": str(LOCAL_DV_FEATURE_DIR),
        "local_alpha_dir": str(LOCAL_ALPHA_DIR),
        "alpha_grid": [float(x) for x in ALPHA_GRID],
        "n_folds": int(args.n_folds),
        "seed": int(args.seed),
        "alpha_rule": "per-voxel median across DeepVision unique random half-split RidgeCV alphas",
        "n_feature_ready_selections": int(len(ready)),
        "n_total_selections_considered": int(len(audit)),
        "score_membership_only": score_membership_only,
        "resumed_from_existing_scores": bool(args.resume),
        "n_resume_complete_selections": int(len(completed_keys)),
        "target_adaptation_scheme": "deepvision_unique_plus_one_target_set_weighted_exact_analytic_loso",
        "baseline_adaptation_scheme": "deepvision_unique_plus_vicco_weighted_exact_analytic_loso",
        "heldout_baseline_scheme": "vicco scored from each cstim-target-adapted model with layer-sweep bootstrap scoring",
        "prediction_protocol": "layer_sweep_stream_predict_v1",
        "n_vicco_boot": int(args.n_vicco_boot),
        "vicco_boot_sample_size": int(VICCO_BOOT_SAMPLE_SIZE),
        "vicco_boot_seed": int(VICCO_BOOT_SEED),
        "python": os.sys.executable,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    RUN_META_JSON.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(scores)} rows -> {SCORE_CSV}", flush=True)
    print(f"Wrote summary -> {SUMMARY_CSV}", flush=True)
    print(f"Wrote audit -> {AUDIT_CSV}", flush=True)


if __name__ == "__main__":
    main()
