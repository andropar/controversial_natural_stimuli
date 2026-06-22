#!/usr/bin/env python3
"""Fit encoding models at every (subject, model, sweep-layer).

This is the missing piece needed to compute mRSA-transfer at non-paper layers.
The existing UNIQUE_ENCODING_DIRS only have encodings at the single paper
layer per model.

Pipeline per (subject, model):
    1. Load DeepVision unique image set for the subject (n=4712)
    2. Extract features at all sweep layers in ONE forward pass per batch
       (cached to cache/dv_features/{subject}/{model}.npz)
    3. For each layer, fit using the same random-kfold alpha-selection protocol
       as experiments/encoding_fitting/fit_encoding_hydra.py:
          - response z-scoring per voxel
          - 5 random 50/50 splits with seed=42
          - 20 alphas from 0.1 to 1e7
          - median alpha aggregation across folds
       Then refit_with_chosen_alphas_fast on all images.
    4. Save encoding_model.npz per (subject, model, layer):
        - weights, intercept, feature_mean, feature_scale, roi_hlvis

Outputs:
    11_layer_sweep/cache/encodings/{subject}/{model}_{layer_safe}/encoding_model.npz
    11_layer_sweep/cache/dv_features/{subject}/{model}.npz   (multi-layer)

Idempotent: skips models with all layers already fit.
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gc
import glob
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from joblib import Parallel, delayed
from scipy.stats import rankdata
from tqdm import tqdm

from cstims import paths
from cstims.cache import load_cstim_brain_cache
from cstims.constants import MODEL_DISPLAY_NAMES
get_brain_input_dir = paths.get_brain_input_dir
from cstims.rdm import compute_rdm_correlation
from cstims.sampling import bootstrap_sample_indices
from cstims.subjects import parse_subject_arg
from batch_tuning import (
    parse_batch_candidates,
    parse_batch_size,
    records_to_array,
    tune_batch_size,
)
from layers_config import MODEL_LAYERS, MODEL_SOURCE, STIMULUS_SETS, get_layer_set
from multilayer_extractor import MultiLayerExtractor
from srp_utils import (
    cached_layer_current,
    FEATURE_PROTOCOL,
    metadata_arrays,
    read_layer_metadata,
    SRPProjectorCache,
    SRP_SEED,
    SRP_TARGET_DIM,
)

from cstims.datasets.deepvision import DeepVisionBenchmark
from cstims.encoding.fitting import (
    fit_voxelwise_ridgecv_fast, refit_with_chosen_alphas_fast,
)


CACHE_DIR = LAYER_SWEEP_ROOT / "cache_or_heavy"
DV_FEAT_CACHE = CACHE_DIR / "dv_features"
ENC_CACHE = CACHE_DIR / "encodings"
LOCK_DIR = CACHE_DIR / "locks"
DEEPVISION_CACHE = paths.deepvision_cache_root()
DV_BENCHMARK_CACHE = DEEPVISION_CACHE
DATA_DIR = LAYER_SWEEP_ROOT / "results"
STREAM_WRSA_CSV = DATA_DIR / "wrsa_dense_layer_sweep.csv"
STREAM_SHARED_CSV = DATA_DIR / "wrsa_dense_shared_layer_sweep.csv"
STREAM_PART_DIR = DATA_DIR / "stream_parts"
STREAM_WRSA_PART_DIR = STREAM_PART_DIR / "wrsa_dense_layer_sweep"
STREAM_SHARED_PART_DIR = STREAM_PART_DIR / "wrsa_dense_shared_layer_sweep"
STREAM_SHARED_STIMULUS_TYPE = "deepvision_shared"
DEFAULT_LAYERS_PER_CHUNK = 16
DEFAULT_N_FIT_JOBS = 3
DEFAULT_MAX_LAYERS_PER_CHUNK = 128
DEFAULT_MAX_FEATURE_GB_PER_CHUNK = 40.0

ENCODING_PROTOCOL = "hydra_random_kfold_v1"
DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_ALPHA_AGGREGATION = "median"
ALPHA_GRID = np.logspace(np.log10(0.1), np.log10(1.0e7), 20)
RESPONSE_ZSCORE_EPS = 1e-6
LOCK_POLL_SECONDS = 30
LOCK_STALE_SECONDS = 24 * 3600


def sanitize_layer_name(layer: str) -> str:
    return (str(layer).replace(".", "_").replace(":", "_")
            .replace("[", "_").replace("]", "_").replace("/", "_"))


def parse_subject_list(subject_arg: str):
    values = [v.strip() for v in str(subject_arg).split(",") if v.strip()]
    if not values or values == ["all"]:
        return parse_subject_arg("all")
    subjects = []
    for value in values:
        for subject in parse_subject_arg(value):
            if subject not in subjects:
                subjects.append(subject)
    return subjects


def parse_layers_per_chunk(value):
    value = str(value).strip().lower()
    if value == "auto":
        return "auto"
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--layers-per-chunk must be 'auto' or a positive integer")
    return parsed


def encoding_path(subject, model, layer):
    layer_safe = sanitize_layer_name(layer)
    return ENC_CACHE / subject / f"{model}.layer{layer_safe}" / "encoding_model.npz"


def lock_path(subject, model):
    return LOCK_DIR / f"encoding_{subject}_{model}.lock"


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _read_lock_pid(path):
    try:
        for line in path.read_text().splitlines():
            if line.startswith("pid="):
                return int(line.split("=", 1)[1])
    except Exception:
        return None
    return None


def acquire_model_lock(subject, model, *, n_folds, seed, alpha_aggregation, alpha_grid):
    """Serialize subject/model fits so parallel dispatchers do not race."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = lock_path(subject, model)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if is_complete(
                subject,
                model,
                n_folds=n_folds,
                seed=seed,
                alpha_aggregation=alpha_aggregation,
                alpha_grid=alpha_grid,
            ):
                print(f"  [cached] {model} (completed while locked)", flush=True)
                return None

            pid = _read_lock_pid(path)
            age = time.time() - path.stat().st_mtime if path.exists() else 0
            if pid is not None and not _pid_alive(pid) and age > LOCK_STALE_SECONDS:
                print(f"  [stale-lock] removing {path}", flush=True)
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue

            print(f"  [locked] {model}; waiting for {path.name}", flush=True)
            time.sleep(LOCK_POLL_SECONDS)
            continue

        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"host={socket.gethostname()}\n")
            f.write(f"subject={subject}\n")
            f.write(f"model={model}\n")
            f.write(f"started={time.time()}\n")
        return path


def release_model_lock(path):
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _scalar_str(value) -> str:
    if value is None:
        return ""
    return str(np.asarray(value).item())


def _zget(npz, key, default=None):
    return npz[key] if key in npz.files else default


def is_encoding_current(path, *, n_folds, seed, alpha_aggregation, alpha_grid):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            if _scalar_str(_zget(z, "fit_protocol")) != ENCODING_PROTOCOL:
                return False
            if int(np.asarray(z["n_folds"]).item()) != int(n_folds):
                return False
            if int(np.asarray(z["seed"]).item()) != int(seed):
                return False
            if _scalar_str(_zget(z, "alpha_aggregation")) != alpha_aggregation:
                return False
            if z["weights"].shape[0] != SRP_TARGET_DIM:
                return False
            if _scalar_str(_zget(z, "feature_protocol")) != FEATURE_PROTOCOL:
                return False
            return np.allclose(z["alpha_grid"], alpha_grid)
    except Exception:
        return False


def is_complete(subject, model, *, n_folds, seed, alpha_aggregation, alpha_grid):
    return all(
        is_encoding_current(
            encoding_path(subject, model, lyr),
            n_folds=n_folds,
            seed=seed,
            alpha_aggregation=alpha_aggregation,
            alpha_grid=alpha_grid,
        )
        for lyr, _ in MODEL_LAYERS[model]
    )


def _load_pil_images(paths):
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def load_dv_features(
    subject,
    model,
    image_paths,
    batch_size=64,
    batch_candidates=None,
    *,
    extract_missing=True,
):
    """Load or extract multi-layer features for the deepvision unique set."""
    cache_path = DV_FEAT_CACHE / subject / f"{model}.npz"
    needed_specs = list(MODEL_LAYERS[model])
    needed_layers = [n for n, _ in needed_specs]
    cached_payload = {}
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as d:
                cached_payload = {k: d[k] for k in d.files}
        except Exception:
            cached_payload = {}

    class _PayloadView:
        files = tuple(cached_payload.keys())
        def __getitem__(self, key):
            return cached_payload[key]

    payload_view = _PayloadView()
    if all(n in cached_payload and cached_layer_current(payload_view, n) for n in needed_layers):
        return {n: cached_payload[n] for n in needed_layers}

    missing_specs = [
        (n, agg)
        for n, agg in needed_specs
        if n not in cached_payload or not cached_layer_current(payload_view, n)
    ]
    if not missing_specs:
        return {n: cached_payload[n] for n in needed_layers}
    if not extract_missing:
        raise RuntimeError(
            f"DeepVision feature cache incomplete for {subject}/{model}: "
            f"{len(missing_specs)} missing or stale layers in {cache_path}"
        )
    print(f"  Extracting features ({len(image_paths)} images, "
          f"{len(missing_specs)} missing / {len(needed_specs)} dense layers)...", flush=True)
    t0 = time.time()
    ext = MultiLayerExtractor(model, MODEL_SOURCE[model], missing_specs)

    batch_tuning_records = []
    if batch_size == "auto":
        batch_candidates = parse_batch_candidates(batch_candidates)
        probe_paths = image_paths[:min(max(batch_candidates), len(image_paths))]
        probe_images = _load_pil_images(probe_paths)
        print("    tuning batch size...", flush=True)
        batch_size, batch_tuning_records = tune_batch_size(
            ext.extract,
            probe_images,
            candidates=batch_candidates,
            verbose=True,
        )
        print(f"    selected batch_size={batch_size}", flush=True)

    feats = {n: [] for n, _ in missing_specs}
    meta_by_layer = {}
    srp_cache = SRPProjectorCache()
    for s in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[s:s + batch_size]
        batch = _load_pil_images(batch_paths)
        out = ext.extract(batch)
        for n, _ in missing_specs:
            reduced, meta = srp_cache.transform(out[n])
            if n not in meta_by_layer:
                meta_by_layer[n] = meta
            elif int(meta_by_layer[n]["original_feature_dim"]) != int(meta["original_feature_dim"]):
                raise RuntimeError(f"Inconsistent feature dim for layer {n}")
            feats[n].append(reduced)
    feats = {n: np.concatenate(arr, axis=0).astype(np.float32) for n, arr in feats.items()}
    ext.free()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for layer_name, arr in feats.items():
        cached_payload[layer_name] = arr
        cached_payload.update(metadata_arrays(layer_name, meta_by_layer[layer_name]))
    cached_payload["_batch_size"] = np.array(batch_size, dtype=np.int32)
    if batch_tuning_records:
        cached_payload["_batch_tuning"] = records_to_array(batch_tuning_records)
    np.savez_compressed(cache_path, **cached_payload)
    print(f"    done in {time.time()-t0:.1f}s", flush=True)
    return {n: cached_payload[n] for n in needed_layers}


def load_cached_srp_metadata(subject, model, layer_name):
    cache_path = DV_FEAT_CACHE / subject / f"{model}.npz"
    if not cache_path.exists():
        return {}
    try:
        with np.load(cache_path, allow_pickle=True) as z:
            return read_layer_metadata(z, layer_name)
    except Exception:
        return {}


def zscore_targets_by_voxel(targets: np.ndarray):
    """Match fit_encoding_hydra.py z-scoring convention."""
    mean = targets.mean(axis=0, dtype=np.float64)
    std = targets.std(axis=0, dtype=np.float64, ddof=0)
    std = np.maximum(std, RESPONSE_ZSCORE_EPS)
    standardized = (targets - mean) / std
    return standardized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def aggregate_alphas(fold_alphas, method: str):
    fold_alphas_array = np.stack(fold_alphas, axis=0)
    if method == "median":
        return np.median(fold_alphas_array, axis=0), fold_alphas_array
    if method == "mean":
        return np.mean(fold_alphas_array, axis=0), fold_alphas_array
    raise ValueError(f"Unknown alpha aggregation: {method}")


def fit_layer_encoding(
    features,
    responses,
    hlvis_mask,
    *,
    n_folds,
    seed,
    alpha_aggregation,
    alpha_grid,
):
    """Fit paper-compatible RidgeCVFast on hlvis voxels."""
    Y_full = responses[hlvis_mask].T.astype(np.float32)  # (n_images, n_hlvis)
    Y_z, Y_mean, Y_std = zscore_targets_by_voxel(Y_full)
    n_images = features.shape[0]

    rng = np.random.RandomState(seed)
    fold_alphas = []
    for _ in range(n_folds):
        indices = rng.permutation(n_images)
        train_idx = indices[:n_images // 2]
        test_idx = indices[n_images // 2:]

        _, alphas, _ = fit_voxelwise_ridgecv_fast(
            X_train=features[train_idx],
            Y_train=Y_z[train_idx],
            X_test=features[test_idx],
            alphas=alpha_grid,
            verbose=False,
            scale_features=True,
        )
        fold_alphas.append(alphas)

    chosen_alphas, fold_alphas_array = aggregate_alphas(fold_alphas, alpha_aggregation)

    # Refit on full data with the fold-aggregated per-voxel alphas, matching
    # fit_encoding_hydra.py final-model fitting.
    W_raw, b_raw, feat_mean, feat_scale = refit_with_chosen_alphas_fast(
        X=features, Y=Y_z, alphas=chosen_alphas,
        verbose=False, scale_features=True,
    )
    # Compose into roi_hlvis full-vector representation matching existing format.
    # The existing encoding_model.npz stores arrays sized to a roi (here we save
    # ONLY hlvis since that's all we need for predict + RDM).
    return {
        "weights": W_raw.astype(np.float32),         # (n_features, n_hlvis)
        "intercept": b_raw.astype(np.float32),       # (n_hlvis,)
        "feature_mean": feat_mean.astype(np.float32),
        "feature_scale": feat_scale.astype(np.float32),
        "alphas": chosen_alphas.astype(np.float32),
        "fold_alphas": fold_alphas_array.astype(np.float32),
        "alpha_grid": alpha_grid.astype(np.float32),
        "fit_protocol": np.array(ENCODING_PROTOCOL),
        "n_folds": np.array(n_folds, dtype=np.int32),
        "seed": np.array(seed, dtype=np.int32),
        "alpha_aggregation": np.array(alpha_aggregation),
        "Y_mean_zscore": Y_mean.astype(np.float32),
        "Y_std_zscore": Y_std.astype(np.float32),
        "n_train": int(features.shape[0]),
    }


def _standardize_torch(x: torch.Tensor):
    mean = x.mean(dim=0)
    centered = x - mean
    scale = torch.sqrt(torch.mean(centered * centered, dim=0))
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return centered / scale, mean, scale


def _pearson_scores_torch(y_true: torch.Tensor, y_pred: torch.Tensor):
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    numerator = torch.sum(yt * yp, dim=0)
    denominator = torch.sqrt(torch.sum(yt * yt, dim=0) * torch.sum(yp * yp, dim=0))
    return numerator / denominator


def _select_alphas_gpu(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    alpha_grid: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    if X_train.shape[0] > X_train.shape[1]:
        raise ValueError(
            "GPU RidgeCV layer-sweep path currently implements sklearn's Gram "
            "GCV mode only. Expected n_train <= n_features after SRP."
        )

    X = torch.as_tensor(X_train, device=device, dtype=dtype)
    Y = torch.as_tensor(Y_train, device=device, dtype=dtype)

    X_scaled, _, _ = _standardize_torch(X)

    # Match sklearn RidgeCVFast/_RidgeGCV for fit_intercept=True. Dense X is
    # centered in preprocessing, then sklearn adds an explicit intercept
    # dimension to the Gram matrix and cancels its regularization.
    X_offset = X_scaled.mean(dim=0)
    Y_offset = Y.mean(dim=0)
    X_centered = X_scaled - X_offset
    Y_centered = Y - Y_offset

    sqrt_sw = torch.ones(X.shape[0], device=device, dtype=dtype)
    gram = X_centered @ X_centered.T
    gram = gram + torch.outer(sqrt_sw, sqrt_sw)
    eigvals, Q = torch.linalg.eigh(gram)
    eigvals = torch.clamp(eigvals, min=0)
    QT_Y = Q.T @ Y_centered
    normalized_sw = sqrt_sw / torch.linalg.vector_norm(sqrt_sw)
    intercept_dim = torch.argmax(torch.abs(normalized_sw @ Q))

    best_score = None
    best_alpha = None
    for alpha in np.asarray(alpha_grid, dtype=np.float64):
        w = 1.0 / (eigvals + float(alpha))
        w[intercept_dim] = 0.0
        dual_coef = Q @ (w[:, None] * QT_Y)
        g_inverse_diag = (Q * Q) @ w
        predictions = Y_centered - dual_coef / g_inverse_diag[:, None] + Y_offset
        alpha_score = _pearson_scores_torch(Y, predictions)
        if best_score is None:
            best_score = alpha_score
            best_alpha = torch.full_like(alpha_score, float(alpha))
        else:
            update = alpha_score > best_score
            best_score = torch.where(update, alpha_score, best_score)
            best_alpha = torch.where(
                update,
                torch.full_like(best_alpha, float(alpha)),
                best_alpha,
            )

    return best_alpha.detach().cpu().numpy().astype(np.float32)


def _refit_with_chosen_alphas_gpu(
    X_full: np.ndarray,
    Y_full: np.ndarray,
    chosen_alphas: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    X = torch.as_tensor(X_full, device=device, dtype=dtype)
    Y = torch.as_tensor(Y_full, device=device, dtype=dtype)

    X_scaled, feature_mean, feature_scale = _standardize_torch(X)
    X_offset = X_scaled.mean(dim=0)
    Y_offset = Y.mean(dim=0)
    X_centered = X_scaled - X_offset
    Y_centered = Y - Y_offset

    gram = X_centered @ X_centered.T
    eigvals, Q = torch.linalg.eigh(gram)
    eigvals = torch.clamp(eigvals, min=0)

    n_features = X.shape[1]
    n_voxels = Y.shape[1]
    W_raw = torch.empty((n_features, n_voxels), device=device, dtype=dtype)
    b_raw = torch.empty((n_voxels,), device=device, dtype=dtype)

    alphas = np.asarray(chosen_alphas)
    for alpha in np.unique(alphas):
        voxel_idx_np = np.flatnonzero(alphas == alpha)
        voxel_idx = torch.as_tensor(voxel_idx_np, device=device, dtype=torch.long)
        Y_sub = Y_centered.index_select(1, voxel_idx)
        denom = eigvals + float(alpha)
        dual_coef = Q @ ((Q.T @ Y_sub) / denom[:, None])
        W_scaled = X_centered.T @ dual_coef
        b_scaled = Y_offset.index_select(0, voxel_idx) - X_offset @ W_scaled
        W_raw[:, voxel_idx] = W_scaled / feature_scale[:, None]
        b_raw[voxel_idx] = b_scaled - (feature_mean / feature_scale) @ W_scaled

    return (
        W_raw.detach().cpu().numpy().astype(np.float32),
        b_raw.detach().cpu().numpy().astype(np.float32),
        feature_mean.detach().cpu().numpy().astype(np.float32),
        feature_scale.detach().cpu().numpy().astype(np.float32),
    )


def fit_layer_encoding_gpu(
    features,
    responses,
    hlvis_mask,
    *,
    n_folds,
    seed,
    alpha_aggregation,
    alpha_grid,
    dtype_name,
):
    """GPU implementation matched to fit_layer_encoding for SRP layer sweeps."""
    if not torch.cuda.is_available():
        raise RuntimeError("--fit-backend gpu requested but CUDA is not available")

    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    device = torch.device("cuda")
    Y_full = responses[hlvis_mask].T.astype(np.float32)
    Y_z, Y_mean, Y_std = zscore_targets_by_voxel(Y_full)
    n_images = features.shape[0]

    rng = np.random.RandomState(seed)
    fold_alphas = []
    for _ in range(n_folds):
        indices = rng.permutation(n_images)
        train_idx = indices[:n_images // 2]
        fold_alphas.append(
            _select_alphas_gpu(
                features[train_idx],
                Y_z[train_idx],
                alpha_grid,
                device=device,
                dtype=dtype,
            )
        )

    chosen_alphas, fold_alphas_array = aggregate_alphas(fold_alphas, alpha_aggregation)
    W_raw, b_raw, feat_mean, feat_scale = _refit_with_chosen_alphas_gpu(
        features,
        Y_z,
        chosen_alphas,
        device=device,
        dtype=dtype,
    )

    return {
        "weights": W_raw.astype(np.float32),
        "intercept": b_raw.astype(np.float32),
        "feature_mean": feat_mean.astype(np.float32),
        "feature_scale": feat_scale.astype(np.float32),
        "alphas": chosen_alphas.astype(np.float32),
        "fold_alphas": fold_alphas_array.astype(np.float32),
        "alpha_grid": alpha_grid.astype(np.float32),
        "fit_protocol": np.array(ENCODING_PROTOCOL),
        "n_folds": np.array(n_folds, dtype=np.int32),
        "seed": np.array(seed, dtype=np.int32),
        "alpha_aggregation": np.array(alpha_aggregation),
        "Y_mean_zscore": Y_mean.astype(np.float32),
        "Y_std_zscore": Y_std.astype(np.float32),
        "n_train": int(features.shape[0]),
        "fit_backend": np.array("gpu"),
        "gpu_fit_dtype": np.array(dtype_name),
    }


def _stable_layer_seed(model: str, layer: str, *, base_seed: int = SRP_SEED) -> int:
    """Stable per-layer SRP seed, so equal-width layers do not share a matrix."""
    digest = hashlib.blake2b(f"{model}::{layer}".encode("utf-8"), digest_size=4).digest()
    return (int.from_bytes(digest, byteorder="little", signed=False) + int(base_seed)) % (2**31 - 1)


def _rdm_upper_vec(rdm: np.ndarray) -> np.ndarray:
    n = rdm.shape[0]
    return rdm[np.triu_indices(n, k=1)]


def _rank_vector(vec: np.ndarray) -> np.ndarray:
    return rankdata(vec, method="average").astype(np.float32)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    den = float(np.sqrt(np.dot(xm, xm) * np.dot(ym, ym)))
    if den <= 0:
        return float("nan")
    return float(np.dot(xm, ym) / den)


def _ranked_rdm(features: np.ndarray) -> np.ndarray:
    return _rank_vector(_rdm_upper_vec(compute_rdm_correlation(features)))


def _load_mixed_items(items):
    images = []
    for item in items:
        if isinstance(item, Image.Image):
            images.append(item)
        else:
            with Image.open(item) as img:
                images.append(img.convert("RGB"))
    return images


def _atomic_savez_compressed(path: Path, **payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def _load_cstim_images(group: str):
    """Match 01_extract_layer_features.py stimulus-set loading exactly."""
    img_files = paths.cstim_image_paths(group, apply_architecture_dataset_swap=True)
    images = []
    for path in img_files:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images, [p.name for p in img_files]


def load_shared_image_paths():
    csv_path = DEEPVISION_CACHE / "image_sets" / "deepvision_shared.csv"
    image_dir = DEEPVISION_CACHE / "image_sets" / "deepvision_shared"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing shared image cache CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    return (image_dir.as_posix() + "/" + df["image_name"]).tolist()


def load_stream_eval_items():
    """Return one combined eval list plus slices for cstim/Vicco/shared subsets."""
    eval_items = []
    cstim_slices = {}
    cstim_counts = {}
    for group in STIMULUS_SETS:
        images, _filenames = _load_cstim_images(group)
        start = len(eval_items)
        eval_items.extend(images)
        cstim_slices[group] = slice(start, len(eval_items))
        cstim_counts[group] = len(images)

    shared_paths = load_shared_image_paths()
    shared_start = len(eval_items)
    eval_items.extend(shared_paths)
    shared_slice = slice(shared_start, len(eval_items))
    print(
        "[stream] eval images: "
        + ", ".join(f"{k}={v}" for k, v in cstim_counts.items())
        + f", shared={len(shared_paths)}",
        flush=True,
    )
    return eval_items, cstim_slices, shared_slice


def _load_cstim_subject_indices(subject: str):
    cache = load_cstim_brain_cache(subject)
    return (
        np.ascontiguousarray(cache.betas_roi, dtype=np.float32),
        cache.group_brain_indices(),
        cache.group_feature_indices(),
    )


def load_cstim_subject_ranks(subject: str, n_vicco_boot: int):
    betas_hlvis, group_indices, group_stim_idx = _load_cstim_subject_indices(subject)
    cache_dir = CACHE_DIR / "brain_ranks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"cstim_{subject}_vicco{n_vicco_boot}.npz"

    cstim_ranks = {}
    vicco_boot = []
    vicco_ranks = []
    n_vicco_sample = 0
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as z:
                groups = [str(g) for g in z["cstim_groups"].tolist()]
                ranks = z["cstim_ranks"].astype(np.float32)
                cstim_ranks = {g: ranks[i] for i, g in enumerate(groups)}
                if "vicco_bootstrap" in z.files and z["vicco_bootstrap"].size:
                    vicco_boot = [idx.astype(int) for idx in z["vicco_bootstrap"]]
                    vicco_ranks = [r.astype(np.float32) for r in z["vicco_ranks"]]
                    n_vicco_sample = int(np.asarray(z["n_vicco_sample"]).item())
                return {
                    "group_stim_idx": group_stim_idx,
                    "cstim_ranks": cstim_ranks,
                    "vicco_bootstrap": vicco_boot,
                    "vicco_ranks": vicco_ranks,
                    "n_vicco_sample": n_vicco_sample,
                }
        except Exception:
            pass

    for group, brain_idx in group_indices.items():
        if group == "vicco":
            continue
        cstim_ranks[group] = _ranked_rdm(betas_hlvis[:, brain_idx].T)

    if "vicco" in group_indices:
        n_vicco = len(group_indices["vicco"])
        n_vicco_sample = min(100, n_vicco)
        vicco_boot = bootstrap_sample_indices(
            n_vicco,
            n_vicco_sample,
            n_bootstrap=n_vicco_boot,
            seed=0,
        )
        vicco_brain_idx = group_indices["vicco"]
        for idx in tqdm(vicco_boot, desc=f"brain vicco ranks {subject}", leave=False):
            vicco_ranks.append(_ranked_rdm(betas_hlvis[:, vicco_brain_idx[idx]].T))

    _atomic_savez_compressed(
        cache_path,
        cstim_groups=np.array(list(cstim_ranks.keys()), dtype=object),
        cstim_ranks=np.stack(list(cstim_ranks.values()), axis=0).astype(np.float32)
        if cstim_ranks else np.empty((0, 0), dtype=np.float32),
        vicco_bootstrap=np.stack(vicco_boot, axis=0).astype(np.int32)
        if vicco_boot else np.empty((0, 0), dtype=np.int32),
        vicco_ranks=np.stack(vicco_ranks, axis=0).astype(np.float32)
        if vicco_ranks else np.empty((0, 0), dtype=np.float32),
        n_vicco_sample=np.array(n_vicco_sample, dtype=np.int32),
    )
    return {
        "group_stim_idx": group_stim_idx,
        "cstim_ranks": cstim_ranks,
        "vicco_bootstrap": vicco_boot,
        "vicco_ranks": vicco_ranks,
        "n_vicco_sample": n_vicco_sample,
    }


def load_shared_subject_ranks(subject: str, n_bootstrap: int, bootstrap_n: int, seed: int):
    cache_dir = CACHE_DIR / "brain_ranks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"shared_{subject}_boot{n_bootstrap}_n{bootstrap_n}_seed{seed}.npz"
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as z:
                return {
                    "boot": [idx.astype(int) for idx in z["boot"]],
                    "brain_ranks": [r.astype(np.float32) for r in z["brain_ranks"]],
                    "n_stimuli": int(np.asarray(z["n_stimuli"]).item()),
                }
        except Exception:
            pass

    root = (
        DEEPVISION_CACHE
        / "voxel_sets"
        / "deepvision_shared_visual_cve0p20"
        / "finalinterp"
        / subject
    )
    betas_path = root / "voxel_betas.npy"
    bsa_path = root / "brain_space_arrays.npz"
    if not betas_path.exists() or not bsa_path.exists():
        raise FileNotFoundError(f"{subject}: missing shared voxel cache under {root}")
    betas = np.load(betas_path).astype(np.float32)
    bsa = np.load(bsa_path)
    hlvis = np.asarray(bsa["hlvis_mask"], dtype=bool)
    betas_hlvis = np.ascontiguousarray(betas[hlvis, :], dtype=np.float32)
    n_stim = betas_hlvis.shape[1]
    n_sample = min(bootstrap_n, n_stim)
    boot = bootstrap_sample_indices(n_stim, n_sample, n_bootstrap=n_bootstrap, seed=seed)
    brain_ranks = []
    for idx in tqdm(boot, desc=f"brain shared ranks {subject}", leave=False):
        brain_ranks.append(_ranked_rdm(betas_hlvis[:, idx].T))

    _atomic_savez_compressed(
        cache_path,
        boot=np.stack(boot, axis=0).astype(np.int32),
        brain_ranks=np.stack(brain_ranks, axis=0).astype(np.float32),
        n_stimuli=np.array(n_sample, dtype=np.int32),
    )
    return {"boot": boot, "brain_ranks": brain_ranks, "n_stimuli": n_sample}


def tune_stream_batch_size(extractor, items, batch_candidates):
    candidates = parse_batch_candidates(batch_candidates)
    probe_items = items[:min(max(candidates), len(items))]
    probe_images = _load_mixed_items(probe_items)
    print(f"    tuning batch size among {candidates}...", flush=True)
    batch_size, _records = tune_batch_size(
        extractor.extract,
        probe_images,
        candidates=candidates,
        verbose=True,
    )
    print(f"    selected batch_size={batch_size}", flush=True)
    return batch_size


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cuda error: out of memory" in text
        or "cudnn_status_alloc_failed" in text
        or "defaultcpuallocator" in text and "can't allocate memory" in text
        or "no batch-size candidate completed successfully" in text
    )


def _cleanup_after_probe():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _layer_chunk_candidates(upper: int):
    if upper <= 1:
        return [1]
    anchors = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, upper]
    return sorted({min(int(v), upper) for v in anchors if int(v) > 0}, reverse=True)


def _feature_budget_layer_cap(n_items: int, max_feature_gb: float) -> int:
    if max_feature_gb <= 0:
        return 10**9
    bytes_per_layer = max(1, int(n_items)) * SRP_TARGET_DIM * np.dtype(np.float32).itemsize
    return max(1, int((float(max_feature_gb) * 1024**3) // bytes_per_layer))


def prepare_stream_extractor(
    *,
    model: str,
    layer_specs: list,
    items: list,
    batch_size_arg,
    batch_candidates,
    max_layers_per_chunk: int,
    max_feature_gb_per_chunk: float,
):
    """Create an extractor for the largest feasible prefix of ``layer_specs``."""
    if not layer_specs:
        raise ValueError("No layer specs left to extract")

    feature_cap = _feature_budget_layer_cap(len(items), max_feature_gb_per_chunk)
    hard_cap = len(layer_specs)
    if max_layers_per_chunk > 0:
        hard_cap = min(hard_cap, int(max_layers_per_chunk))
    hard_cap = min(hard_cap, feature_cap)
    candidates = _layer_chunk_candidates(hard_cap)
    probe_items = items[:min(max(parse_batch_candidates(batch_candidates)), len(items))]
    if not probe_items:
        raise ValueError("Need at least one image to probe extraction")

    last_oom = None
    for n_layers in candidates:
        chunk_specs = layer_specs[:n_layers]
        print(
            f"    probing layer chunk={n_layers} "
            f"(feature_budget_cap={feature_cap}, hard_cap={hard_cap})",
            flush=True,
        )
        extractor = None
        try:
            extractor = MultiLayerExtractor(model, MODEL_SOURCE[model], chunk_specs)
            if batch_size_arg == "auto":
                batch_size = tune_stream_batch_size(extractor, probe_items, batch_candidates)
            else:
                batch_size = int(batch_size_arg)
                probe_batch = _load_mixed_items(probe_items[:min(batch_size, len(probe_items))])
                out = extractor.extract(probe_batch)
                del out, probe_batch
            _cleanup_after_probe()
            return chunk_specs, extractor, batch_size
        except (RuntimeError, MemoryError) as exc:
            if extractor is not None:
                extractor.free()
            _cleanup_after_probe()
            if _is_oom_error(exc):
                last_oom = exc
                print(f"    layer chunk {n_layers}: OOM; trying smaller", flush=True)
                continue
            raise

    raise RuntimeError("No layer chunk candidate fit") from last_oom


def extract_reduced_stream_features(
    extractor,
    items,
    *,
    batch_size: int,
    model: str,
    progress_log: str | Path | None = None,
    chunk_idx: int | None = None,
    extract_prefetch_workers: int = 0,
):
    layer_names = [name for name, _ in extractor.layers]
    feats = {}
    srp_caches = {
        name: SRPProjectorCache(seed=_stable_layer_seed(model, name))
        for name in layer_names
    }
    meta_by_layer = {}
    total_batches = int(np.ceil(len(items) / max(batch_size, 1)))
    progress_every = max(1, total_batches // 100)

    def log_batch_stage(event: str, batch_idx: int, start_idx: int, stop_idx: int, **fields):
        if batch_idx != 0 and batch_idx + 1 != total_batches and (batch_idx + 1) % progress_every != 0:
            return
        append_progress_log(
            progress_log,
            event,
            model=model,
            chunk_idx=int(chunk_idx) if chunk_idx is not None else None,
            batch_num=int(batch_idx + 1),
            n_batches=int(total_batches),
            start_idx=int(start_idx),
            stop_idx=int(stop_idx),
            n_images=int(len(items)),
            batch_size=int(batch_size),
            n_layers=int(len(layer_names)),
            **fields,
        )

    def prepare_batch(start_idx: int, stop_idx: int):
        batch_t0 = time.time()
        images = _load_mixed_items(items[start_idx:stop_idx])
        batch = extractor.preprocess_images(images)
        return batch, time.time() - batch_t0

    def process_batch(batch_idx: int, start_idx: int, stop_idx: int, batch, load_elapsed: float):
        log_batch_stage("extract_batch_loaded", batch_idx, start_idx, stop_idx, elapsed_sec=float(load_elapsed))
        forward_t0 = time.time()
        raw = extractor.extract_tensor_batch(batch)
        forward_elapsed = time.time() - forward_t0
        log_batch_stage(
            "extract_batch_forward_done",
            batch_idx,
            start_idx,
            stop_idx,
            elapsed_sec=float(forward_elapsed),
        )
        srp_t0 = time.time()
        for name in layer_names:
            reduced, meta = srp_caches[name].transform(raw[name])
            if name not in meta_by_layer:
                meta_by_layer[name] = meta
            elif int(meta_by_layer[name]["original_feature_dim"]) != int(meta["original_feature_dim"]):
                raise RuntimeError(f"Inconsistent feature dim for layer {name}")
            reduced = np.ascontiguousarray(reduced, dtype=np.float32)
            if name not in feats:
                feats[name] = np.empty(
                    (len(items), reduced.shape[1]),
                    dtype=np.float32,
                )
            feats[name][start_idx:stop_idx] = reduced
        srp_elapsed = time.time() - srp_t0
        log_batch_stage("extract_batch_srp_done", batch_idx, start_idx, stop_idx, elapsed_sec=float(srp_elapsed))
        del raw, batch

    starts = list(range(0, len(items), batch_size))
    if extract_prefetch_workers <= 0:
        for batch_idx, start in enumerate(tqdm(starts, desc="    extract", leave=False)):
            stop = min(start + batch_size, len(items))
            log_batch_stage("extract_batch_start", batch_idx, start, stop)
            batch, load_elapsed = prepare_batch(start, stop)
            process_batch(batch_idx, start, stop, batch, load_elapsed)
    else:
        max_pending = max(1, int(extract_prefetch_workers))
        with ThreadPoolExecutor(max_workers=max_pending) as pool:
            pending = deque()
            next_batch_idx = 0

            def submit_until_full():
                nonlocal next_batch_idx
                while next_batch_idx < len(starts) and len(pending) < max_pending:
                    start_idx = starts[next_batch_idx]
                    stop_idx = min(start_idx + batch_size, len(items))
                    log_batch_stage("extract_batch_start", next_batch_idx, start_idx, stop_idx)
                    future = pool.submit(prepare_batch, start_idx, stop_idx)
                    pending.append((next_batch_idx, start_idx, stop_idx, future))
                    next_batch_idx += 1

            submit_until_full()
            with tqdm(total=len(starts), desc="    extract", leave=False) as pbar:
                while pending:
                    batch_idx, start, stop, future = pending.popleft()
                    batch, load_elapsed = future.result()
                    process_batch(batch_idx, start, stop, batch, load_elapsed)
                    pbar.update(1)
                    submit_until_full()
    return feats, meta_by_layer


def predict_stream(features: np.ndarray, enc: dict) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    weights = np.asarray(enc["weights"], dtype=np.float32)
    intercept = np.asarray(enc["intercept"], dtype=np.float32)
    pred = x @ weights + intercept
    roi = enc.get("roi_hlvis")
    if roi is not None:
        pred = pred[:, np.asarray(roi, dtype=bool)]
    return np.ascontiguousarray(pred, dtype=np.float32)


def score_ranked_prediction(pred: np.ndarray, brain_rank: np.ndarray) -> float:
    pred_rank = _ranked_rdm(pred)
    return _pearson_r(pred_rank, brain_rank)


def score_bootstrap_prediction(boot_idx, idx, pred_all, brain_rank, n_stimuli):
    return {
        "bootstrap_idx": boot_idx,
        "n_stimuli": n_stimuli,
        "rsa": score_ranked_prediction(pred_all[idx], brain_rank),
    }


def fit_stream_layer(layer_name, features, responses, hlvis_mask, args):
    if args.fit_backend == "gpu":
        enc = fit_layer_encoding_gpu(
            features,
            responses,
            hlvis_mask,
            n_folds=args.n_folds,
            seed=args.seed,
            alpha_aggregation=args.alpha_aggregation,
            alpha_grid=ALPHA_GRID,
            dtype_name=args.gpu_fit_dtype,
        )
    else:
        enc = fit_layer_encoding(
            features,
            responses,
            hlvis_mask,
            n_folds=args.n_folds,
            seed=args.seed,
            alpha_aggregation=args.alpha_aggregation,
            alpha_grid=ALPHA_GRID,
        )
    enc["layer"] = np.array(layer_name)
    enc["roi_hlvis"] = np.ones(enc["weights"].shape[1], dtype=bool)
    enc["feature_protocol"] = np.array(FEATURE_PROTOCOL)
    enc["srp_applied"] = np.array(True, dtype=bool)
    enc["srp_target_dim"] = np.array(SRP_TARGET_DIM, dtype=np.int32)
    enc["stored_feature_dim"] = np.array(SRP_TARGET_DIM, dtype=np.int32)
    return layer_name, enc


def stream_write_chunk_rows(path: Path, rows, layer_names):
    layer_names = set(str(layer) for layer in layer_names)
    if not rows and not path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    if path.exists():
        try:
            existing = pd.read_csv(path)
            if "layer" in existing.columns:
                existing = existing[~existing["layer"].astype(str).isin(layer_names)]
            frames.append(existing)
        except pd.errors.EmptyDataError:
            pass
    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


def stream_part_dirs(part_root: str | Path | None = None) -> tuple[Path, Path]:
    root = Path(part_root) if part_root else STREAM_PART_DIR
    return root / "wrsa_dense_layer_sweep", root / "wrsa_dense_shared_layer_sweep"


def stream_part_paths(subject: str, model: str, part_root: str | Path | None = None):
    safe_model = model.replace("/", "_")
    wrsa_dir, shared_dir = stream_part_dirs(part_root)
    return (
        wrsa_dir / f"{subject}_{safe_model}.csv",
        shared_dir / f"{subject}_{safe_model}.csv",
    )


def stream_encoding_path(root: str | Path | None, subject: str, model: str, layer: str):
    if root is None:
        return None
    root = Path(root)
    return root / subject / f"{model}.layer{sanitize_layer_name(layer)}" / "encoding_model.npz"


def append_progress_log(path: str | Path | None, event: str, **fields):
    if not path:
        return
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        **fields,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", buffering=1) as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def save_stream_encoding(path: Path | None, enc: dict):
    if path is None:
        return
    payload = dict(enc)
    payload["prediction_formula"] = np.array("features @ weights + intercept")
    payload["prediction_feature_space"] = np.array("raw_cached_srp_features")
    _atomic_savez_compressed(path, **payload)


def completed_stream_layers(
    wrsa_part: Path,
    shared_part: Path,
    *,
    expected_wrsa_rows: int,
    expected_shared_rows: int,
):
    if not wrsa_part.exists() or not shared_part.exists():
        return set()
    try:
        wrsa = pd.read_csv(wrsa_part, usecols=["layer"])
        shared = pd.read_csv(shared_part, usecols=["layer"])
    except Exception:
        return set()
    wrsa_counts = wrsa.groupby("layer").size()
    shared_counts = shared.groupby("layer").size()
    return {
        layer
        for layer in set(wrsa_counts.index).intersection(set(shared_counts.index))
        if wrsa_counts.get(layer, 0) == expected_wrsa_rows
        and shared_counts.get(layer, 0) == expected_shared_rows
    }


def score_stream_chunk(
    *,
    subject: str,
    model: str,
    display: str,
    features_by_layer: dict,
    enc_by_layer: dict,
    n_unique: int,
    cstim_slices: dict,
    shared_slice: slice,
    cstim_data: dict,
    shared_data: dict,
    n_score_jobs: int,
):
    wrsa_rows = []
    shared_rows = []

    for layer_name, enc in enc_by_layer.items():
        eval_feats = features_by_layer[layer_name][n_unique:]

        for group, brain_rank in cstim_data["cstim_ranks"].items():
            if group not in cstim_slices:
                continue
            features_group = eval_feats[cstim_slices[group]]
            file_idx = cstim_data["group_stim_idx"].get(group)
            if file_idx is None or len(file_idx) == 0 or len(features_group) == 0:
                continue
            pred = predict_stream(features_group[file_idx], enc)
            wrsa_rows.append({
                "subject": subject,
                "model": model,
                "display_name": display,
                "layer": layer_name,
                "model_set": group,
                "stimulus_type": "controversial",
                "bootstrap_idx": 0,
                "n_stimuli": int(len(file_idx)),
                "rsa": score_ranked_prediction(pred, brain_rank),
            })

        if "vicco" in cstim_slices and cstim_data["vicco_bootstrap"]:
            features_vicco = eval_feats[cstim_slices["vicco"]]
            file_idx = cstim_data["group_stim_idx"].get("vicco")
            if file_idx is not None and len(file_idx) and len(features_vicco):
                pred_vicco = predict_stream(features_vicco[file_idx], enc)
                jobs = (
                    delayed(score_bootstrap_prediction)(
                        boot_idx,
                        idx,
                        pred_vicco,
                        cstim_data["vicco_ranks"][boot_idx],
                        cstim_data["n_vicco_sample"],
                    )
                    for boot_idx, idx in enumerate(cstim_data["vicco_bootstrap"])
                )
                if n_score_jobs > 1:
                    res = Parallel(n_jobs=n_score_jobs, prefer="threads")(jobs)
                else:
                    res = [
                        score_bootstrap_prediction(
                            boot_idx,
                            idx,
                            pred_vicco,
                            cstim_data["vicco_ranks"][boot_idx],
                            cstim_data["n_vicco_sample"],
                        )
                        for boot_idx, idx in enumerate(cstim_data["vicco_bootstrap"])
                    ]
                for r in res:
                    wrsa_rows.append({
                        "subject": subject,
                        "model": model,
                        "display_name": display,
                        "layer": layer_name,
                        "model_set": "vicco",
                        "stimulus_type": "vicco",
                        **r,
                    })

        shared_feats = eval_feats[shared_slice]
        pred_shared = predict_stream(shared_feats, enc)
        shared_jobs = (
            delayed(score_bootstrap_prediction)(
                boot_idx,
                idx,
                pred_shared,
                shared_data["brain_ranks"][boot_idx],
                shared_data["n_stimuli"],
            )
            for boot_idx, idx in enumerate(shared_data["boot"])
        )
        if n_score_jobs > 1:
            shared_res = Parallel(n_jobs=n_score_jobs, prefer="threads")(shared_jobs)
        else:
            shared_res = [
                score_bootstrap_prediction(
                    boot_idx,
                    idx,
                    pred_shared,
                    shared_data["brain_ranks"][boot_idx],
                    shared_data["n_stimuli"],
                )
                for boot_idx, idx in enumerate(shared_data["boot"])
            ]
        for r in shared_res:
            shared_rows.append({
                "subject": subject,
                "model": model,
                "display_name": display,
                "layer": layer_name,
                "model_set": STREAM_SHARED_STIMULUS_TYPE,
                "stimulus_type": STREAM_SHARED_STIMULUS_TYPE,
                **r,
            })

    return wrsa_rows, shared_rows


def stream_subject_model(args, subject: str, model: str):
    if model not in MODEL_LAYERS:
        print(f"[skip] unknown model: {model}", flush=True)
        return

    print(f"\n=== stream {subject} / {model} ===", flush=True)
    bench = DeepVisionBenchmark(
        cache_root=str(DV_BENCHMARK_CACHE),
        subject=subject,
        voxel_set="visual",
        input_source="finalinterp",
        image_set="unique",
        n_jobs=args.deepvision_load_jobs,
    )
    responses = bench.response_data.to_numpy()
    hlvis_mask = bench.get_roi_mask("hlvis")
    unique_paths = bench.stimulus_data.image_path.tolist()
    eval_items, cstim_slices, shared_slice = load_stream_eval_items()
    all_items = list(unique_paths) + eval_items
    n_unique = len(unique_paths)

    cstim_data = load_cstim_subject_ranks(subject, args.n_vicco_boot)
    shared_data = load_shared_subject_ranks(
        subject,
        n_bootstrap=args.n_shared_boot,
        bootstrap_n=args.bootstrap_n,
        seed=args.shared_seed,
    )
    expected_wrsa = len(cstim_data["cstim_ranks"]) + len(cstim_data["vicco_bootstrap"])
    expected_shared = len(shared_data["boot"])
    wrsa_part, shared_part = stream_part_paths(subject, model, args.stream_part_root)
    completed = set() if args.overwrite else completed_stream_layers(
        wrsa_part,
        shared_part,
        expected_wrsa_rows=expected_wrsa,
        expected_shared_rows=expected_shared,
    )
    layer_specs = [(name, agg) for name, agg in MODEL_LAYERS[model] if name not in completed]
    if not layer_specs:
        print(f"[cached] {subject}/{model}: stream parts complete", flush=True)
        return

    print(
        f"[stream] unique={n_unique}, eval={len(eval_items)}, "
        f"layers={len(layer_specs)} remaining/{len(MODEL_LAYERS[model])}, "
        f"chunk={args.layers_per_chunk}, fit_jobs={args.n_fit_jobs}, "
        f"score_jobs={args.n_score_jobs}",
        flush=True,
    )
    append_progress_log(
        args.progress_log,
        "model_start",
        subject=subject,
        model=model,
        n_unique=int(n_unique),
        n_eval=int(len(eval_items)),
        n_layers_remaining=int(len(layer_specs)),
        n_layers_total=int(len(MODEL_LAYERS[model])),
        layers_per_chunk=int(args.layers_per_chunk),
        stream_part_root=str(args.stream_part_root or STREAM_PART_DIR),
        stream_encoding_root=str(args.stream_encoding_root or ""),
    )
    display = MODEL_DISPLAY_NAMES.get(model, model)
    n_chunks = int(np.ceil(len(layer_specs) / args.layers_per_chunk))

    for chunk_idx, start in enumerate(range(0, len(layer_specs), args.layers_per_chunk), start=1):
        chunk_specs = layer_specs[start:start + args.layers_per_chunk]
        layer_names = [name for name, _ in chunk_specs]
        chunk_t0 = time.time()
        print(
            f"  [chunk {chunk_idx}/{n_chunks}] extracting+fitting "
            f"{len(chunk_specs)} layers: {layer_names[0]} ... {layer_names[-1]}",
            flush=True,
        )
        append_progress_log(
            args.progress_log,
            "chunk_start",
            subject=subject,
            model=model,
            chunk_idx=int(chunk_idx),
            n_chunks=int(n_chunks),
            n_layers=int(len(chunk_specs)),
            first_layer=str(layer_names[0]),
            last_layer=str(layer_names[-1]),
        )
        extractor = MultiLayerExtractor(model, MODEL_SOURCE[model], chunk_specs)
        try:
            batch_size = args.batch_size
            if batch_size == "auto":
                batch_size = tune_stream_batch_size(extractor, all_items, args.batch_candidates)
            features_by_layer, _meta_by_layer = extract_reduced_stream_features(
                extractor,
                all_items,
                batch_size=batch_size,
                model=model,
                progress_log=args.progress_log,
                chunk_idx=chunk_idx,
                extract_prefetch_workers=args.extract_prefetch_workers,
            )
        finally:
            extractor.free()

        fit_jobs = (
            delayed(fit_stream_layer)(
                layer_name,
                features_by_layer[layer_name][:n_unique],
                responses,
                hlvis_mask,
                args,
            )
            for layer_name in layer_names
        )
        n_fit_jobs = 1 if args.fit_backend == "gpu" else args.n_fit_jobs
        if n_fit_jobs > 1 and len(layer_names) > 1:
            fitted = Parallel(n_jobs=n_fit_jobs, prefer="threads")(fit_jobs)
        else:
            fitted = [
                fit_stream_layer(
                    layer_name,
                    features_by_layer[layer_name][:n_unique],
                    responses,
                    hlvis_mask,
                    args,
                )
                for layer_name in layer_names
            ]
        enc_by_layer = {layer_name: enc for layer_name, enc in fitted}

        saved_count = 0
        if args.stream_encoding_root:
            for layer_name, enc in enc_by_layer.items():
                enc["layer"] = np.array(layer_name)
                enc["model"] = np.array(model)
                enc["subject"] = np.array(subject)
                enc["display_name"] = np.array(display)
                out_path = stream_encoding_path(
                    args.stream_encoding_root,
                    subject,
                    model,
                    layer_name,
                )
                save_stream_encoding(out_path, enc)
                saved_count += 1
            append_progress_log(
                args.progress_log,
                "encodings_saved",
                subject=subject,
                model=model,
                chunk_idx=int(chunk_idx),
                n_saved=int(saved_count),
            )

        wrsa_rows, shared_rows = score_stream_chunk(
            subject=subject,
            model=model,
            display=display,
            features_by_layer=features_by_layer,
            enc_by_layer=enc_by_layer,
            n_unique=n_unique,
            cstim_slices=cstim_slices,
            shared_slice=shared_slice,
            cstim_data=cstim_data,
            shared_data=shared_data,
            n_score_jobs=args.n_score_jobs,
        )
        stream_write_chunk_rows(wrsa_part, wrsa_rows, layer_names)
        stream_write_chunk_rows(shared_part, shared_rows, layer_names)
        elapsed = time.time() - chunk_t0
        print(
            f"  [chunk {chunk_idx}/{n_chunks}] wrote "
            f"{len(wrsa_rows)} cstim/vicco rows + {len(shared_rows)} shared rows "
            f"in {elapsed:.1f}s",
            flush=True,
        )
        append_progress_log(
            args.progress_log,
            "chunk_done",
            subject=subject,
            model=model,
            chunk_idx=int(chunk_idx),
            n_chunks=int(n_chunks),
            n_saved=int(saved_count),
            n_wrsa_rows=int(len(wrsa_rows)),
            n_shared_rows=int(len(shared_rows)),
            elapsed_sec=float(elapsed),
        )
        del features_by_layer, enc_by_layer, fitted, wrsa_rows, shared_rows
        gc.collect()

    append_progress_log(
        args.progress_log,
        "model_done",
        subject=subject,
        model=model,
        n_layers_total=int(len(MODEL_LAYERS[model])),
    )


def load_stream_subject_state(args, subject: str, model: str):
    bench = DeepVisionBenchmark(
        cache_root=str(DV_BENCHMARK_CACHE),
        subject=subject,
        voxel_set="visual",
        input_source="finalinterp",
        image_set="unique",
        n_jobs=args.deepvision_load_jobs,
    )
    responses = bench.response_data.to_numpy()
    hlvis_mask = bench.get_roi_mask("hlvis")
    unique_paths = bench.stimulus_data.image_path.tolist()
    cstim_data = load_cstim_subject_ranks(subject, args.n_vicco_boot)
    shared_data = load_shared_subject_ranks(
        subject,
        n_bootstrap=args.n_shared_boot,
        bootstrap_n=args.bootstrap_n,
        seed=args.shared_seed,
    )
    expected_wrsa = len(cstim_data["cstim_ranks"]) + len(cstim_data["vicco_bootstrap"])
    expected_shared = len(shared_data["boot"])
    wrsa_part, shared_part = stream_part_paths(subject, model, args.stream_part_root)
    completed = set() if args.overwrite else completed_stream_layers(
        wrsa_part,
        shared_part,
        expected_wrsa_rows=expected_wrsa,
        expected_shared_rows=expected_shared,
    )
    return {
        "subject": subject,
        "responses": responses,
        "hlvis_mask": hlvis_mask,
        "unique_paths": unique_paths,
        "cstim_data": cstim_data,
        "shared_data": shared_data,
        "wrsa_part": wrsa_part,
        "shared_part": shared_part,
        "completed": completed,
    }


def remaining_specs_for_states(model: str, states: list):
    return [
        (name, agg)
        for name, agg in MODEL_LAYERS[model]
        if any(name not in state["completed"] for state in states)
    ]


def build_model_stream_items(states: list, eval_items: list):
    items = []
    unique_slices = {}
    for state in states:
        start = len(items)
        items.extend(state["unique_paths"])
        unique_slices[state["subject"]] = slice(start, len(items))
    eval_start = len(items)
    items.extend(eval_items)
    return items, unique_slices, eval_start


def stream_model_all_subjects(args, model: str, subjects: list):
    if model not in MODEL_LAYERS:
        print(f"[skip] unknown model: {model}", flush=True)
        return

    print(f"\n=== stream model-major / {model} / {len(subjects)} subjects ===", flush=True)
    eval_items, cstim_slices, shared_slice = load_stream_eval_items()
    states = [load_stream_subject_state(args, subject, model) for subject in subjects]
    states = [
        state for state in states
        if len(state["completed"]) < len(MODEL_LAYERS[model])
    ]
    if not states:
        print(f"[cached] {model}: stream parts complete for all subjects", flush=True)
        return

    display = MODEL_DISPLAY_NAMES.get(model, model)
    append_progress_log(
        args.progress_log,
        "model_start",
        subject=",".join(state["subject"] for state in states),
        model=model,
        n_subjects=int(len(states)),
        n_eval=int(len(eval_items)),
        n_layers_total=int(len(MODEL_LAYERS[model])),
        layers_per_chunk=str(args.layers_per_chunk),
        stream_part_root=str(args.stream_part_root or STREAM_PART_DIR),
        stream_encoding_root=str(args.stream_encoding_root or ""),
    )

    chunk_idx = 0
    while True:
        remaining_specs = remaining_specs_for_states(model, states)
        if not remaining_specs:
            break
        states_with_remaining = [
            state for state in states
            if any(name not in state["completed"] for name, _agg in remaining_specs)
        ]
        probe_items, _probe_slices, _probe_eval_start = build_model_stream_items(
            states_with_remaining,
            eval_items,
        )

        chunk_specs, extractor, batch_size = prepare_stream_extractor(
            model=model,
            layer_specs=remaining_specs,
            items=probe_items,
            batch_size_arg=args.batch_size,
            batch_candidates=args.batch_candidates,
            max_layers_per_chunk=(
                args.max_layers_per_chunk
                if args.layers_per_chunk == "auto"
                else int(args.layers_per_chunk)
            ),
            max_feature_gb_per_chunk=args.max_feature_gb_per_chunk,
        )
        layer_names = [name for name, _agg in chunk_specs]
        active_states = [
            state for state in states
            if any(name not in state["completed"] for name in layer_names)
        ]
        all_items, unique_slices, eval_start = build_model_stream_items(
            active_states,
            eval_items,
        )
        chunk_idx += 1
        chunk_t0 = time.time()
        print(
            f"  [chunk {chunk_idx}] extracting {len(layer_names)} layers "
            f"for {len(active_states)} subjects, {len(all_items)} images: "
            f"{layer_names[0]} ... {layer_names[-1]}",
            flush=True,
        )
        append_progress_log(
            args.progress_log,
            "chunk_start",
            subject=",".join(state["subject"] for state in active_states),
            model=model,
            chunk_idx=int(chunk_idx),
            n_layers=int(len(layer_names)),
            n_subjects=int(len(active_states)),
            n_images=int(len(all_items)),
            batch_size=int(batch_size),
            first_layer=str(layer_names[0]),
            last_layer=str(layer_names[-1]),
        )

        try:
            features_by_layer, _meta_by_layer = extract_reduced_stream_features(
                extractor,
                all_items,
                batch_size=batch_size,
                model=model,
                progress_log=args.progress_log,
                chunk_idx=chunk_idx,
                extract_prefetch_workers=args.extract_prefetch_workers,
            )
        finally:
            extractor.free()

        eval_features_by_layer = {
            layer_name: arr[eval_start:]
            for layer_name, arr in features_by_layer.items()
        }
        total_fit = 0
        total_saved = 0
        total_wrsa_rows = 0
        total_shared_rows = 0

        for state in active_states:
            subject = state["subject"]
            subject_layers = [
                layer_name for layer_name in layer_names
                if layer_name not in state["completed"]
            ]
            if not subject_layers:
                continue
            subj_slice = unique_slices[subject]
            enc_by_layer = {}
            for layer_name in subject_layers:
                _name, enc = fit_stream_layer(
                    layer_name,
                    features_by_layer[layer_name][subj_slice],
                    state["responses"],
                    state["hlvis_mask"],
                    args,
                )
                enc_by_layer[layer_name] = enc
                total_fit += 1

            if args.stream_encoding_root:
                for layer_name, enc in enc_by_layer.items():
                    enc["layer"] = np.array(layer_name)
                    enc["model"] = np.array(model)
                    enc["subject"] = np.array(subject)
                    enc["display_name"] = np.array(display)
                    out_path = stream_encoding_path(
                        args.stream_encoding_root,
                        subject,
                        model,
                        layer_name,
                    )
                    save_stream_encoding(out_path, enc)
                    total_saved += 1

            wrsa_rows, shared_rows = score_stream_chunk(
                subject=subject,
                model=model,
                display=display,
                features_by_layer=eval_features_by_layer,
                enc_by_layer=enc_by_layer,
                n_unique=0,
                cstim_slices=cstim_slices,
                shared_slice=shared_slice,
                cstim_data=state["cstim_data"],
                shared_data=state["shared_data"],
                n_score_jobs=args.n_score_jobs,
            )
            stream_write_chunk_rows(state["wrsa_part"], wrsa_rows, subject_layers)
            stream_write_chunk_rows(state["shared_part"], shared_rows, subject_layers)
            state["completed"].update(subject_layers)
            total_wrsa_rows += len(wrsa_rows)
            total_shared_rows += len(shared_rows)
            del enc_by_layer, wrsa_rows, shared_rows
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed = time.time() - chunk_t0
        print(
            f"  [chunk {chunk_idx}] fit={total_fit}, saved={total_saved}, "
            f"rows={total_wrsa_rows}+{total_shared_rows} in {elapsed:.1f}s",
            flush=True,
        )
        append_progress_log(
            args.progress_log,
            "chunk_done",
            subject=",".join(state["subject"] for state in active_states),
            model=model,
            chunk_idx=int(chunk_idx),
            n_layers=int(len(layer_names)),
            n_subjects=int(len(active_states)),
            n_fit=int(total_fit),
            n_saved=int(total_saved),
            n_wrsa_rows=int(total_wrsa_rows),
            n_shared_rows=int(total_shared_rows),
            elapsed_sec=float(elapsed),
        )
        del features_by_layer, eval_features_by_layer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    append_progress_log(
        args.progress_log,
        "model_done",
        subject=",".join(subjects),
        model=model,
        n_layers_total=int(len(MODEL_LAYERS[model])),
    )


def stream_main(args):
    global MODEL_LAYERS
    MODEL_LAYERS = get_layer_set(args.layer_set)
    args.batch_size = parse_batch_size(args.batch_size)
    args.layers_per_chunk = parse_layers_per_chunk(args.layers_per_chunk)
    if args.layers_per_chunk == "auto":
        args.layers_per_chunk = DEFAULT_LAYERS_PER_CHUNK
    if args.n_score_jobs is None:
        args.n_score_jobs = args.n_fit_jobs

    wrsa_part_dir, shared_part_dir = stream_part_dirs(args.stream_part_root)
    wrsa_part_dir.mkdir(parents=True, exist_ok=True)
    shared_part_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subject_list(args.subject)
    for subject in subjects:
        for model in args.models:
            stream_subject_model(args, subject, model)
    print("[stream] done", flush=True)


def stream_model_main(args):
    global MODEL_LAYERS
    MODEL_LAYERS = get_layer_set(args.layer_set)
    args.batch_size = parse_batch_size(args.batch_size)
    args.layers_per_chunk = parse_layers_per_chunk(args.layers_per_chunk)
    if args.n_score_jobs is None:
        args.n_score_jobs = args.n_fit_jobs

    wrsa_part_dir, shared_part_dir = stream_part_dirs(args.stream_part_root)
    wrsa_part_dir.mkdir(parents=True, exist_ok=True)
    shared_part_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subject_list(args.subject)
    for model in args.models:
        stream_model_all_subjects(args, model, subjects)
    print("[stream-model] done", flush=True)


def validate_stream_parts_complete(args) -> None:
    subjects = parse_subject_list(args.subject)
    missing = []
    for subject in subjects:
        cstim_data = load_cstim_subject_ranks(subject, args.n_vicco_boot)
        shared_data = load_shared_subject_ranks(
            subject,
            n_bootstrap=args.n_shared_boot,
            bootstrap_n=args.bootstrap_n,
            seed=args.shared_seed,
        )
        expected_wrsa = len(cstim_data["cstim_ranks"]) + len(cstim_data["vicco_bootstrap"])
        expected_shared = len(shared_data["boot"])
        for model in args.models:
            if model not in MODEL_LAYERS:
                missing.append(f"{subject}/{model}: unknown model")
                continue
            wrsa_part, shared_part = stream_part_paths(subject, model, args.stream_part_root)
            complete_layers = completed_stream_layers(
                wrsa_part,
                shared_part,
                expected_wrsa_rows=expected_wrsa,
                expected_shared_rows=expected_shared,
            )
            expected_layers = [name for name, _ in MODEL_LAYERS[model]]
            missing_layers = [name for name in expected_layers if name not in complete_layers]
            if missing_layers:
                missing.append(
                    f"{subject}/{model}: {len(missing_layers)}/{len(expected_layers)} "
                    f"layers missing or incomplete"
                )
    if missing:
        preview = "\n".join(f"  - {item}" for item in missing[:40])
        extra = "" if len(missing) <= 40 else f"\n  ... {len(missing) - 40} more"
        raise RuntimeError(
            "Refusing to merge incomplete dense layer sweep stream parts:\n"
            f"{preview}{extra}"
        )


def merge_stream_parts(out_csv: Path, part_dir: Path):
    files = sorted(glob.glob(str(part_dir / "*.csv")))
    if not files:
        print(f"[merge] no parts in {part_dir}", flush=True)
        return
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        print(f"[merge] no readable parts in {part_dir}", flush=True)
        return
    df = pd.concat(frames, ignore_index=True)
    key_cols = ["subject", "model", "layer", "model_set", "stimulus_type", "bootstrap_idx"]
    df = df.drop_duplicates(subset=key_cols, keep="last")
    df = df.sort_values(key_cols, kind="stable")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[merge] wrote {len(df)} rows -> {out_csv}", flush=True)


def merge_stream_main(args):
    wrsa_csv = Path(args.out_csv) if args.out_csv else STREAM_WRSA_CSV
    shared_csv = Path(args.shared_out_csv) if args.shared_out_csv else STREAM_SHARED_CSV
    wrsa_part_dir, shared_part_dir = stream_part_dirs(args.stream_part_root)
    validate_stream_parts_complete(args)
    merge_stream_parts(wrsa_csv, wrsa_part_dir)
    merge_stream_parts(shared_csv, shared_part_dir)


def main():
    global MODEL_LAYERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--layer-set", choices=["configured", "dense", "cornet_decoder"], default="configured",
                        help="Layer inventory to fit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", default="auto",
                        help="Batch size or 'auto' to benchmark per model")
    parser.add_argument("--batch-candidates", default=None,
                        help="Comma-separated candidates for --batch-size auto")
    parser.add_argument("--extract-prefetch-workers", type=int, default=2,
                        help="Streaming mode: CPU worker threads per process for "
                             "loading and preprocessing upcoming image batches. "
                             "Use 0 to disable prefetching.")
    parser.add_argument("--deepvision-load-jobs", type=int, default=8,
                        help="Workers used to build missing DeepVision response caches. "
                             "Use 1 for memory-constrained Slurm retries.")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha-aggregation", choices=["median", "mean"],
                        default=DEFAULT_ALPHA_AGGREGATION)
    parser.add_argument("--mode", choices=["all", "features", "fit", "stream", "stream-model", "merge-stream"],
                        default="all",
                        help="'features' only prepares DeepVision feature caches; "
                             "'fit' requires those caches and only fits encodings; "
                             "'stream' extracts flattened+SRP layer chunks, fits, scores, "
                             "and writes result parts; 'stream-model' does the same "
                             "model-major across all requested subjects.")
    parser.add_argument("--layers-per-chunk", default=str(DEFAULT_LAYERS_PER_CHUNK),
                        help="Streaming mode: number of model layers extracted/fitted together, "
                             "or 'auto' for model-major GPU probing.")
    parser.add_argument("--max-layers-per-chunk", type=int, default=DEFAULT_MAX_LAYERS_PER_CHUNK,
                        help="stream-model auto mode: hard cap on layers per extraction chunk. "
                             "Use 0 for no explicit layer-count cap.")
    parser.add_argument("--max-feature-gb-per-chunk", type=float,
                        default=DEFAULT_MAX_FEATURE_GB_PER_CHUNK,
                        help="stream-model auto mode: approximate host-memory budget for the "
                             "reduced feature block per GPU worker. Use <=0 to disable.")
    parser.add_argument("--n-fit-jobs", type=int, default=DEFAULT_N_FIT_JOBS,
                        help="Streaming mode: parallel layer fits within each GPU worker.")
    parser.add_argument("--fit-backend", choices=["cpu", "gpu"], default="cpu",
                        help="Streaming mode: fit encoding models on CPU or with the "
                             "GPU RidgeCV/refit implementation.")
    parser.add_argument("--gpu-fit-dtype", choices=["float32", "float64"], default="float64",
                        help="Streaming mode: dtype for --fit-backend gpu. float64 "
                             "matches CPU RidgeCVFast alpha choices most closely.")
    parser.add_argument("--n-score-jobs", type=int, default=None,
                        help="Streaming mode: parallel bootstrap scoring jobs. "
                             "Default matches --n-fit-jobs.")
    parser.add_argument("--n-vicco-boot", type=int, default=1000)
    parser.add_argument("--n-shared-boot", type=int, default=1000)
    parser.add_argument("--bootstrap-n", type=int, default=100)
    parser.add_argument("--shared-seed", type=int, default=0)
    parser.add_argument("--stream-part-root", default=None,
                        help="Optional stream part root. Defaults to results/stream_parts.")
    parser.add_argument("--stream-encoding-root", default=None,
                        help="Streaming mode: optional root for saving fitted encoding_model.npz "
                             "files as {root}/{subject}/{model}.layer{layer}/encoding_model.npz.")
    parser.add_argument("--progress-log", default=None,
                        help="Streaming mode: append JSONL progress events to this file.")
    parser.add_argument("--out-csv", default=None,
                        help="merge-stream mode: cstim/Vicco output CSV.")
    parser.add_argument("--shared-out-csv", default=None,
                        help="merge-stream mode: DeepVision-shared output CSV.")
    args = parser.parse_args()

    MODEL_LAYERS = get_layer_set(args.layer_set)
    if args.models is None:
        args.models = list(MODEL_LAYERS.keys())

    if args.mode == "stream":
        stream_main(args)
        return
    if args.mode == "stream-model":
        stream_model_main(args)
        return
    if args.mode == "merge-stream":
        merge_stream_main(args)
        return

    args.batch_size = parse_batch_size(args.batch_size)
    batch_candidates = parse_batch_candidates(args.batch_candidates)

    subjects = parse_subject_arg(args.subject)
    print(f"Subjects: {subjects}")
    print(f"Models: {len(args.models)}")
    print(f"Encoding protocol: {ENCODING_PROTOCOL} "
          f"(n_folds={args.n_folds}, seed={args.seed}, "
          f"alpha_aggregation={args.alpha_aggregation}, "
          f"n_alphas={len(ALPHA_GRID)})")

    DV_FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    ENC_CACHE.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        print(f"\n=== {subject} ===")
        bench = DeepVisionBenchmark(
            cache_root=str(DV_BENCHMARK_CACHE),
            subject=subject,
            voxel_set="visual",
            input_source="finalinterp",
            image_set="unique",
            n_jobs=args.deepvision_load_jobs,
        )
        responses = bench.response_data.to_numpy()  # (n_voxels, n_images)
        n_images = bench.n_stimuli
        hlvis_mask = bench.get_roi_mask("hlvis")
        image_paths = bench.stimulus_data.image_path.tolist()
        print(f"  n_images={n_images}, n_hlvis={int(hlvis_mask.sum())}")

        for model in args.models:
            if model not in MODEL_LAYERS:
                continue
            model_lock = None
            try:
                if not args.overwrite and is_complete(
                    subject,
                    model,
                    n_folds=args.n_folds,
                    seed=args.seed,
                    alpha_aggregation=args.alpha_aggregation,
                    alpha_grid=ALPHA_GRID,
                ):
                    print(f"  [cached] {model}", flush=True)
                    continue

                if not args.overwrite:
                    model_lock = acquire_model_lock(
                        subject,
                        model,
                        n_folds=args.n_folds,
                        seed=args.seed,
                        alpha_aggregation=args.alpha_aggregation,
                        alpha_grid=ALPHA_GRID,
                    )
                    if model_lock is None:
                        continue

                    if is_complete(
                        subject,
                        model,
                        n_folds=args.n_folds,
                        seed=args.seed,
                        alpha_aggregation=args.alpha_aggregation,
                        alpha_grid=ALPHA_GRID,
                    ):
                        print(f"  [cached] {model}", flush=True)
                        continue

                print(f"\n  --- {model} ---", flush=True)
                feats_per_layer = load_dv_features(subject, model, image_paths,
                                                   batch_size=args.batch_size,
                                                   batch_candidates=batch_candidates,
                                                   extract_missing=(args.mode != "fit"))

                if args.mode == "features":
                    continue

                for layer_name, _ in MODEL_LAYERS[model]:
                    out_path = encoding_path(subject, model, layer_name)
                    if (
                        not args.overwrite
                        and is_encoding_current(
                            out_path,
                            n_folds=args.n_folds,
                            seed=args.seed,
                            alpha_aggregation=args.alpha_aggregation,
                            alpha_grid=ALPHA_GRID,
                        )
                    ):
                        continue
                    t0 = time.time()
                    X = feats_per_layer[layer_name]
                    if X.ndim != 2:
                        X = X.reshape(X.shape[0], -1)
                    X = X.astype(np.float32)

                    enc = fit_layer_encoding(
                        X,
                        responses,
                        hlvis_mask,
                        n_folds=args.n_folds,
                        seed=args.seed,
                        alpha_aggregation=args.alpha_aggregation,
                        alpha_grid=ALPHA_GRID,
                    )
                    enc["layer"] = np.array(layer_name)
                    enc["model"] = np.array(model)
                    enc["subject"] = np.array(subject)
                    srp_meta = load_cached_srp_metadata(subject, model, layer_name)
                    if srp_meta:
                        enc["feature_protocol"] = np.array(
                            srp_meta.get("feature_protocol", FEATURE_PROTOCOL)
                        )
                        enc["srp_applied"] = np.array(srp_meta["srp_applied"], dtype=bool)
                        enc["srp_target_dim"] = np.array(srp_meta["srp_target_dim"], dtype=np.int32)
                        enc["srp_seed"] = np.array(srp_meta["srp_seed"], dtype=np.int32)
                        enc["original_feature_dim"] = np.array(
                            srp_meta["original_feature_dim"], dtype=np.int32
                        )
                        enc["stored_feature_dim"] = np.array(
                            srp_meta["stored_feature_dim"], dtype=np.int32
                        )
                    enc["roi_hlvis"] = np.ones(enc["weights"].shape[1], dtype=bool)
                    # Save
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out_path, **enc)
                    print(f"    [layer={layer_name:<35} feat={X.shape[1]:5d}] "
                          f"fit done {time.time()-t0:5.1f}s -> {out_path.name}",
                          flush=True)
            finally:
                release_model_lock(model_lock)

    print("\nAll done.")


if __name__ == "__main__":
    main()
