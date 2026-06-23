"""Data loading, target preparation, and proxy-shortlist helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cstims.encoding.linear import encode_batch_for_all_encodings
from cstims.evaluation.noise_calibration import (
    calibrate_response_noise_for_rdm_reliability,
    multiplier_to_rdm_reliability,
    response_noise_std_from_mode,
)
from cstims.evaluation.ridge import ridge_ops_for_eval_sets
from cstims.evaluation.teacher_student import (
    detect_equivalent_models,
    pearson_columns,
    stable_seed,
)
from feature_method_sweep import (
    MethodRuntime,
    MethodSpec,
    build_runtime,
    compute_track_scores,
    get_track_candidate_features,
)


MAX_BASE_KERNEL_PRECOMPUTE_GIB = 16.0
MAX_BASE_KERNEL_PRECOMPUTE_RAM_FRACTION = 0.25
DEFAULT_ALPHA_TARGET_BATCH_SIZE = 4096
DEFAULT_SCORE_TARGET_BATCH_SIZE = 8192


@dataclass
class StudentOps:
    model: str
    x_train_raw: np.ndarray
    train_mean: np.ndarray
    train_scale: np.ndarray
    x_train: np.ndarray
    x_val: np.ndarray
    x_base: np.ndarray
    base_eigvals: np.ndarray
    base_eigvecs: np.ndarray
    k_base_pool: np.ndarray | None
    val_ops: dict[float, tuple[np.ndarray, dict[str, np.ndarray]]]


@dataclass
class TeacherTargets:
    model: str
    y_base_clean: np.ndarray
    base_indices: np.ndarray
    train_pos: np.ndarray
    val_pos: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    response_noise_std: float
    achieved_fit_rdm_reliability: float
    y_base_clean_path: str | None = None

    @property
    def n_targets(self) -> int:
        """Return the number of response target columns available for this teacher."""
        return int(self.y_base_clean.shape[1])


@dataclass
class TeacherNoiseState:
    y_base_fit: np.ndarray
    alpha_choices: dict[str, tuple[list[float], np.ndarray]]
    y_base_fit_path: str | None = None


@dataclass
class FitContext:
    student_ops: dict[str, StudentOps]
    teacher_targets: dict[str, TeacherTargets]
    equivalence_labels: list[int]
    target_cols: np.ndarray | None
    train_indices: np.ndarray
    val_indices: np.ndarray
    base_indices: np.ndarray
    fit_noise_calibration: str
    rdm_calibration_comparison: str


def raw_subset_tensors(
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    indices: np.ndarray | list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load raw feature rows for selected models as float32 tensors on the target device."""
    idx = np.asarray(indices, dtype=np.int64)
    return {
        model: torch.from_numpy(np.asarray(raw_features_np[model][idx])).to(
            device=device,
            dtype=torch.float32,
        )
        for model in model_names
    }


def feature_standardization_stats(
    train: np.ndarray,
    *,
    scale_by_sqrt_features: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-set feature centering and scaling statistics."""
    train = np.asarray(train, dtype=np.float32)
    mean = train.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    if scale_by_sqrt_features:
        scale *= np.float32(math.sqrt(train.shape[1]))
    return mean, scale


def apply_standardization(
    array: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply previously computed feature standardization to an array."""
    return np.asarray((np.asarray(array, dtype=np.float32) - mean) / scale, dtype=np.float32)


def encode_indices(
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    indices: np.ndarray | list[int],
    track: str,
    encoding_params: Any,
    device: torch.device,
    target_cols: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Encode selected natural-pool rows into the requested brain-response target space."""
    raw = raw_subset_tensors(raw_features_np, model_names, indices, device)
    encoded = encode_batch_for_all_encodings(raw, {track: encoding_params[track]})[track]
    out: dict[str, np.ndarray] = {}
    for model in model_names:
        arr = encoded[model].detach().cpu().numpy().astype(np.float32, copy=False)
        if target_cols is not None:
            arr = arr[:, target_cols]
        out[model] = arr
    del raw, encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def choose_target_columns(
    *,
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    track: str,
    encoding_params: Any,
    device: torch.device,
    target_dim: int | None,
    seed: int,
) -> np.ndarray | None:
    """Choose a reproducible target-column subset when dimensionality reduction is requested."""
    if target_dim is None or target_dim <= 0:
        return None
    probe_model = model_names[0]
    encoded_one = encode_indices(
        raw_features_np=raw_features_np,
        model_names=[probe_model],
        indices=[0],
        track=track,
        encoding_params=encoding_params,
        device=device,
    )
    full_dim = next(iter(encoded_one.values())).shape[1]
    if target_dim >= full_dim:
        return None
    rng = np.random.default_rng(seed + stable_seed(track, "target_cols", target_dim))
    return np.sort(rng.choice(full_dim, size=target_dim, replace=False)).astype(np.int64)


def build_proxy_runtime(
    *,
    method: MethodSpec,
    selected_indices: list[int],
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    encoding_params: Any,
    var_noise_by_track: dict[str, dict[str, float]],
    metric: str,
    device: torch.device,
    pool_size: int,
) -> MethodRuntime:
    """Construct the fixed-RDM proxy runtime for the current selected image set."""
    return build_runtime(
        method=method,
        selected_indices=selected_indices,
        raw_features_np=raw_features_np,
        model_names=model_names,
        encoding_params=encoding_params,
        var_noise_by_track=var_noise_by_track,
        metric=metric,
        device=device,
        pool_size=pool_size,
    )


def proxy_scores_for_pool(
    *,
    runtime: MethodRuntime,
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    encoding_params: Any,
    track: str,
    metric: str,
    corr_type: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Score all eligible pool candidates with the cheap proxy objective."""
    pool_size = len(runtime.pool_mask)
    scores = np.full(pool_size, -np.inf, dtype=np.float32)
    track_spec = runtime.spec.tracks[0]
    for start in range(0, pool_size, batch_size):
        end = min(start + batch_size, pool_size)
        batch_indices = np.arange(start, end, dtype=np.int64)
        valid_mask = runtime.pool_mask[batch_indices]
        if not valid_mask.any():
            continue
        valid_indices = batch_indices[valid_mask]
        raw_batch = raw_subset_tensors(raw_features_np, model_names, valid_indices, device)
        encoded_batch = encode_batch_for_all_encodings(
            raw_batch,
            {track: encoding_params[track]},
        )
        cand = get_track_candidate_features(track_spec, raw_batch, encoded_batch)
        batch_scores = compute_track_scores(
            candidate_features=cand,
            runtime=runtime.tracks[track],
            metric=metric,
            corr_type=corr_type,
            within=runtime.spec.within,
            across=runtime.spec.across,
        )
        scores[valid_indices] = batch_scores.detach().cpu().numpy().astype(np.float32)
        del raw_batch, encoded_batch, cand, batch_scores
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return scores


def topk_shortlist(
    *,
    proxy_scores: np.ndarray,
    pool_mask: np.ndarray,
    top_k: int,
    random_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Combine top proxy candidates with optional random exploration candidates."""
    eligible = np.flatnonzero(pool_mask)
    if len(eligible) == 0:
        return np.empty(0, dtype=np.int64)
    top_k = min(max(0, int(top_k)), len(eligible))
    if top_k:
        eligible_scores = proxy_scores[eligible]
        kth = min(top_k, len(eligible_scores))
        part = np.argpartition(-eligible_scores, kth - 1)[:kth]
        top = eligible[part[np.argsort(-eligible_scores[part])]]
    else:
        top = np.empty(0, dtype=np.int64)
    random_k = min(max(0, int(random_k)), len(eligible))
    random = rng.choice(eligible, size=random_k, replace=False) if random_k else np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate([top, random])).astype(np.int64)


def build_fit_context(
    *,
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    track: str,
    encoding_params: Any,
    device: torch.device,
    target_cols: np.ndarray | None,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    base_indices: np.ndarray,
    refit_train_pos: np.ndarray,
    refit_val_pos: np.ndarray,
    alphas: list[float],
    base_noise_ceiling: float,
    noise_mult: float,
    fit_noise_calibration: str,
    rdm_calibration_comparison: str,
    metric: str,
    corr_type: str,
    seed: int,
    calibration_images: int,
    calibration_noise_samples: int,
    calibration_max_iter: int,
    precompute_base_kernels: bool,
    kernel_batch_size: int,
    target_cache_dir: Path | None,
    target_batch_size: int,
) -> FitContext:
    """Prepare standardized student features, teacher targets, and refit calibration state."""
    student_ops: dict[str, StudentOps] = {}
    for model in model_names:
        x_train_raw = np.asarray(raw_features_np[model][train_indices], dtype=np.float32)
        x_val_raw = np.asarray(raw_features_np[model][val_indices], dtype=np.float32)
        x_base_raw = np.asarray(raw_features_np[model][base_indices], dtype=np.float32)
        train_mean, train_scale = feature_standardization_stats(
            x_train_raw,
            scale_by_sqrt_features=True,
        )
        x_train = apply_standardization(x_train_raw, train_mean, train_scale)
        x_val = apply_standardization(x_val_raw, train_mean, train_scale)
        x_base = apply_standardization(x_base_raw, train_mean, train_scale)
        val_ops = ridge_ops_for_eval_sets(x_train, x_val, {}, alphas)
        x_base64 = np.asarray(x_base, dtype=np.float64)
        k_base = x_base64 @ x_base64.T
        eigvals, eigvecs = np.linalg.eigh(k_base)
        eigvals = np.maximum(eigvals, 0.0)
        k_base_pool = None
        if precompute_base_kernels:
            n_pool = raw_features_np[model].shape[0]
            k_base_pool = np.empty((x_base.shape[0], n_pool), dtype=np.float32)
            for start in range(0, n_pool, kernel_batch_size):
                end = min(start + kernel_batch_size, n_pool)
                x_pool = apply_standardization(
                    raw_features_np[model][start:end],
                    train_mean,
                    train_scale,
                )
                k_base_pool[:, start:end] = np.asarray(x_base @ x_pool.T, dtype=np.float32)
            print(
                f"  precomputed base kernel for {model}: {k_base_pool.shape}",
                flush=True,
            )
        student_ops[model] = StudentOps(
            model=model,
            x_train_raw=x_train_raw,
            train_mean=train_mean,
            train_scale=train_scale,
            x_train=x_train,
            x_val=x_val,
            x_base=x_base,
            base_eigvals=np.asarray(eigvals, dtype=np.float64),
            base_eigvecs=np.asarray(eigvecs, dtype=np.float64),
            k_base_pool=k_base_pool,
            val_ops=val_ops,
        )

    teacher_targets: dict[str, TeacherTargets] = {}
    for model in model_names:
        print(f"  encoding and caching refit target for teacher={model}", flush=True)
        encoded_one = encode_indices(
            raw_features_np=raw_features_np,
            model_names=[model],
            indices=base_indices,
            track=track,
            encoding_params=encoding_params,
            device=device,
            target_cols=target_cols,
        )
        y_base_raw = np.asarray(encoded_one[model], dtype=np.float32)
        y_base_clean, y_base_clean_path = standardize_base_targets(
            y_base_raw=y_base_raw,
            train_pos=refit_train_pos,
            teacher=model,
            cache_dir=target_cache_dir,
            target_batch_size=target_batch_size,
        )
        achieved = np.nan
        if fit_noise_calibration == "rdm_empirical":
            calib_pos = refit_train_pos
            if 0 < calibration_images < len(refit_train_pos):
                calib_rng = np.random.default_rng(
                    seed + stable_seed(model, "selector_noise_calibration")
                )
                calib_local_idx = np.sort(
                    calib_rng.choice(len(refit_train_pos), size=calibration_images, replace=False)
                )
                calib_pos = refit_train_pos[calib_local_idx]
            y_calib = np.asarray(y_base_clean[calib_pos], dtype=np.float32)
            cal_rng = np.random.default_rng(
                seed
                + stable_seed(
                    model,
                    noise_mult,
                    "rdm_empirical",
                    rdm_calibration_comparison,
                )
            )
            response_noise_std, achieved = calibrate_response_noise_for_rdm_reliability(
                y_calib,
                target_reliability=multiplier_to_rdm_reliability(
                    noise_mult,
                    base_noise_ceiling,
                    rdm_calibration_comparison,
                ),
                metric=metric,
                corr_type=corr_type,
                rng=cal_rng,
                n_samples=calibration_noise_samples,
                max_iter=calibration_max_iter,
                comparison=rdm_calibration_comparison,
            )
        else:
            response_noise_std = response_noise_std_from_mode(
                noise_mult,
                base_noise_ceiling,
                fit_noise_calibration,
            )
        teacher_targets[model] = TeacherTargets(
            model=model,
            y_base_clean=y_base_clean,
            base_indices=np.asarray(base_indices, dtype=np.int64),
            train_pos=np.asarray(refit_train_pos, dtype=np.int64),
            val_pos=np.asarray(refit_val_pos, dtype=np.int64),
            train_indices=np.asarray(train_indices, dtype=np.int64),
            val_indices=np.asarray(val_indices, dtype=np.int64),
            response_noise_std=float(response_noise_std),
            achieved_fit_rdm_reliability=float(achieved),
            y_base_clean_path=y_base_clean_path,
        )
        del encoded_one, y_base_raw

    equivalence_labels = detect_equivalent_models(
        {model: np.asarray(raw_features_np[model][base_indices], dtype=np.float32) for model in model_names},
        model_names,
    )
    return FitContext(
        student_ops=student_ops,
        teacher_targets=teacher_targets,
        equivalence_labels=equivalence_labels,
        target_cols=target_cols,
        train_indices=train_indices,
        val_indices=val_indices,
        base_indices=base_indices,
        fit_noise_calibration=fit_noise_calibration,
        rdm_calibration_comparison=rdm_calibration_comparison,
    )


def resolve_base_kernel_precompute(
    *,
    requested: bool,
    pool_size: int,
    refit_pool_size: int,
    model_names: list[str],
    max_ram_gb: float,
) -> tuple[bool, dict[str, Any]]:
    """Enable base-kernel precompute only when its estimated memory cost fits the budget."""
    bytes_per_model = (
        int(refit_pool_size) * int(pool_size) * np.dtype(np.float32).itemsize
    )
    n_models = max(1, len(model_names))
    total_gib = (bytes_per_model * n_models) / (1024**3)
    per_model_gib = bytes_per_model / (1024**3)
    budget_gib = min(
        MAX_BASE_KERNEL_PRECOMPUTE_GIB,
        max(1.0, float(max_ram_gb) * MAX_BASE_KERNEL_PRECOMPUTE_RAM_FRACTION),
    )
    config = {
        "requested": bool(requested),
        "enabled": bool(requested),
        "estimated_gib_per_model": per_model_gib,
        "estimated_gib_total": total_gib,
        "budget_gib": budget_gib,
        "reason": None,
    }
    if requested and total_gib > budget_gib:
        config["enabled"] = False
        config["reason"] = "estimated_precompute_cache_exceeds_budget"
        print(
            "Disabling base-kernel precompute: "
            f"estimated {total_gib:.1f} GiB total "
            f"({per_model_gib:.1f} GiB/model) for pool_size={pool_size}, "
            f"refit_pool_size={refit_pool_size}, n_models={len(model_names)} "
            f"exceeds {budget_gib:.1f} GiB budget. "
            "Shortlist kernels will be computed on demand.",
            flush=True,
        )
    return bool(config["enabled"]), config


def response_noise_rows(
    *,
    image_indices: np.ndarray,
    n_targets: int,
    std: float,
    seed: int,
    parts: tuple[object, ...],
) -> np.ndarray:
    """Generate deterministic response-noise rows for complete target vectors."""
    rows = []
    for image_idx in image_indices:
        rng = np.random.default_rng(seed + stable_seed(*parts, int(image_idx)))
        rows.append(rng.normal(0.0, std, size=n_targets).astype(np.float32))
    return np.stack(rows, axis=0)


def response_noise_rows_chunk(
    *,
    image_indices: np.ndarray,
    n_targets: int,
    start: int,
    end: int,
    std: float,
    seed: int,
    parts: tuple[object, ...],
) -> np.ndarray:
    """Generate deterministic response-noise rows for a target-column chunk."""
    image_indices = np.asarray(image_indices, dtype=np.int64)
    width = int(end) - int(start)
    if width <= 0:
        return np.empty((len(image_indices), 0), dtype=np.float32)
    if float(std) == 0.0:
        return np.zeros((len(image_indices), width), dtype=np.float32)
    out = np.empty((len(image_indices), width), dtype=np.float32)
    for row_idx, image_idx in enumerate(image_indices):
        rng = np.random.default_rng(
            seed
            + stable_seed(
                *parts,
                int(image_idx),
                int(start),
                int(end),
                int(n_targets),
            )
        )
        out[row_idx] = rng.normal(0.0, std, size=width).astype(np.float32)
    return out


def safe_cache_component(value: str) -> str:
    """Sanitize a string for use as one component of a cache filename."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def target_rows_chunk(
    target: TeacherTargets,
    row_positions: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """Read a chunk of target columns for selected base-set positions."""
    return np.asarray(
        target.y_base_clean[np.asarray(row_positions, dtype=np.int64), start:end],
        dtype=np.float32,
    )


def target_standardization_stats(
    y_base_raw: np.ndarray,
    train_pos: np.ndarray,
    *,
    target_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute target centering and scaling statistics from refit-train rows in chunks."""
    n_targets = int(y_base_raw.shape[1])
    target_batch_size = min(max(1, int(target_batch_size)), n_targets)
    train_pos = np.asarray(train_pos, dtype=np.int64)
    mean = np.empty((1, n_targets), dtype=np.float32)
    scale = np.empty((1, n_targets), dtype=np.float32)
    for start in range(0, n_targets, target_batch_size):
        end = min(start + target_batch_size, n_targets)
        train_chunk = np.asarray(y_base_raw[train_pos, start:end], dtype=np.float32)
        mean[:, start:end] = train_chunk.mean(
            axis=0,
            dtype=np.float64,
            keepdims=True,
        ).astype(np.float32)
        scale_chunk = train_chunk.std(
            axis=0,
            dtype=np.float64,
            keepdims=True,
        ).astype(np.float32)
        scale_chunk[scale_chunk < 1e-6] = 1.0
        scale[:, start:end] = scale_chunk
    return mean, scale


def standardize_base_targets(
    *,
    y_base_raw: np.ndarray,
    train_pos: np.ndarray,
    teacher: str,
    cache_dir: Path | None,
    target_batch_size: int,
) -> tuple[np.ndarray, str | None]:
    """Standardize all base targets and optionally back them with a memory-mapped cache file."""
    n_base, n_targets = y_base_raw.shape
    target_batch_size = min(max(1, int(target_batch_size)), int(n_targets))
    mean, scale = target_standardization_stats(
        y_base_raw,
        train_pos,
        target_batch_size=target_batch_size,
    )
    path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{safe_cache_component(teacher)}_y_base_clean.npy"
        y_base_clean = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(int(n_base), int(n_targets)),
        )
    else:
        y_base_clean = np.empty((int(n_base), int(n_targets)), dtype=np.float32)
    for start in range(0, n_targets, target_batch_size):
        end = min(start + target_batch_size, n_targets)
        y_base_clean[:, start:end] = (
            np.asarray(y_base_raw[:, start:end], dtype=np.float32)
            - mean[:, start:end]
        ) / scale[:, start:end]
    if path is not None:
        y_base_clean.flush()
        del y_base_clean
        return np.load(path, mmap_mode="r"), str(path)
    return np.asarray(y_base_clean, dtype=np.float32), None


def encode_indices_modelwise_cached(
    *,
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    indices: np.ndarray | list[int],
    track: str,
    encoding_params: Any,
    device: torch.device,
    target_cols: np.ndarray | None,
    cache_dir: Path | None,
    cache_prefix: str,
) -> dict[str, np.ndarray]:
    """Encode rows one model at a time and optionally store each result as a memmap."""
    out: dict[str, np.ndarray] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for model in model_names:
        encoded_one = encode_indices(
            raw_features_np=raw_features_np,
            model_names=[model],
            indices=indices,
            track=track,
            encoding_params=encoding_params,
            device=device,
            target_cols=target_cols,
        )
        arr = np.asarray(encoded_one[model], dtype=np.float32)
        if cache_dir is not None:
            path = cache_dir / (
                f"{safe_cache_component(cache_prefix)}_"
                f"{safe_cache_component(model)}.npy"
            )
            cached = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float32,
                shape=arr.shape,
            )
            cached[:] = arr
            cached.flush()
            del cached
            out[model] = np.load(path, mmap_mode="r")
        else:
            out[model] = arr
        del encoded_one, arr
    return out


def select_targetwise_alpha_indices_for_target(
    alpha_ops: dict[float, tuple[np.ndarray, dict[str, np.ndarray]]],
    target: TeacherTargets,
    *,
    noise_sample_idx: int,
    seed: int,
    target_batch_size: int,
) -> tuple[list[float], np.ndarray]:
    """Select one ridge alpha per target column using validation Pearson correlation."""
    if not isinstance(alpha_ops, dict):
        raise TypeError("Chunked alpha selection requires dict ridge operators")

    alpha_values = list(alpha_ops)
    if not alpha_values:
        raise ValueError("No alpha operators available")
    n_targets = target.n_targets
    target_batch_size = min(max(1, int(target_batch_size)), n_targets)
    best_alpha_idx = np.empty(n_targets, dtype=np.int32)
    for start in range(0, n_targets, target_batch_size):
        end = min(start + target_batch_size, n_targets)
        y_train_chunk = target_rows_chunk(target, target.train_pos, start, end)
        y_train_chunk += response_noise_rows_chunk(
            image_indices=target.train_indices,
            n_targets=n_targets,
            start=start,
            end=end,
            std=target.response_noise_std,
            seed=seed,
            parts=("alpha_train_noise", target.model, noise_sample_idx),
        )
        y_val_chunk = target_rows_chunk(target, target.val_pos, start, end)
        y_val_chunk += response_noise_rows_chunk(
            image_indices=target.val_indices,
            n_targets=n_targets,
            start=start,
            end=end,
            std=target.response_noise_std,
            seed=seed,
            parts=("alpha_val_noise", target.model, noise_sample_idx),
        )
        scores = np.empty((len(alpha_values), end - start), dtype=np.float64)
        for alpha_idx, alpha in enumerate(alpha_values):
            val_op, _eval_ops = alpha_ops[alpha]
            pred_val = np.asarray(val_op @ y_train_chunk, dtype=np.float32)
            scores[alpha_idx] = pearson_columns(pred_val, y_val_chunk)
            del pred_val
        scores = np.nan_to_num(scores, nan=-np.inf)
        best_alpha_idx[start:end] = np.argmax(scores, axis=0).astype(np.int32)
    return alpha_values, best_alpha_idx


def write_noisy_base_fit_memmap(
    *,
    target: TeacherTargets,
    teacher: str,
    noise_sample_idx: int,
    seed: int,
    cache_dir: Path,
    target_batch_size: int,
) -> tuple[np.ndarray, str]:
    """Write one noisy teacher base-fit target matrix to a memory-mapped cache file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (
        f"{safe_cache_component(teacher)}_noise{int(noise_sample_idx):03d}_"
        "y_base_fit.npy"
    )
    target_batch_size = max(1, int(target_batch_size))
    n_targets = target.n_targets
    y_base_fit = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=target.y_base_clean.shape,
    )
    std = target.response_noise_std
    for start in range(0, n_targets, target_batch_size):
        end = min(start + target_batch_size, n_targets)
        y_base_fit[:, start:end] = np.asarray(
            target.y_base_clean[:, start:end],
            dtype=np.float32,
        ) + response_noise_rows_chunk(
            image_indices=target.base_indices,
            n_targets=n_targets,
            start=start,
            end=end,
            std=std,
            seed=seed,
            parts=("refit_base_fit_noise", teacher, noise_sample_idx),
        )
    y_base_fit.flush()
    del y_base_fit
    return np.load(path, mmap_mode="r"), str(path)


def build_noise_states(
    *,
    fit_context: FitContext,
    model_names: list[str],
    seed: int,
    alphas: list[float],
    n_noise_samples: int,
    alpha_target_batch_size: int,
    noise_cache_dir: Path | None,
) -> dict[str, list[TeacherNoiseState]]:
    """Precompute noisy teacher targets and target-wise alpha choices for all noise samples."""
    states: dict[str, list[TeacherNoiseState]] = {}
    for teacher in model_names:
        target = fit_context.teacher_targets[teacher]
        teacher_states: list[TeacherNoiseState] = []
        for noise_sample_idx in range(n_noise_samples):
            std = target.response_noise_std
            y_base_fit_path = None
            if noise_cache_dir is not None:
                y_base_fit, y_base_fit_path = write_noisy_base_fit_memmap(
                    target=target,
                    teacher=teacher,
                    noise_sample_idx=noise_sample_idx,
                    seed=seed,
                    cache_dir=noise_cache_dir,
                    target_batch_size=alpha_target_batch_size,
                )
            else:
                y_base_fit = np.empty_like(target.y_base_clean, dtype=np.float32)
                for start in range(0, target.n_targets, alpha_target_batch_size):
                    end = min(start + alpha_target_batch_size, target.n_targets)
                    y_base_fit[:, start:end] = np.asarray(
                        target.y_base_clean[:, start:end],
                        dtype=np.float32,
                    ) + response_noise_rows_chunk(
                        image_indices=target.base_indices,
                        n_targets=target.n_targets,
                        start=start,
                        end=end,
                        std=std,
                        seed=seed,
                        parts=("refit_base_fit_noise", teacher, noise_sample_idx),
                    )
            alpha_choices = {}
            for student in model_names:
                alpha_values, best_alpha_idx = select_targetwise_alpha_indices_for_target(
                    fit_context.student_ops[student].val_ops,
                    target,
                    noise_sample_idx=noise_sample_idx,
                    seed=seed,
                    target_batch_size=alpha_target_batch_size,
                )
                alpha_choices[student] = (alpha_values, best_alpha_idx)
            teacher_states.append(
                TeacherNoiseState(
                    y_base_fit=y_base_fit,
                    alpha_choices=alpha_choices,
                    y_base_fit_path=y_base_fit_path,
                )
            )
        states[teacher] = teacher_states
    return states


def build_refit_splits(
    *,
    pool_size: int,
    selected_initial: list[int],
    refit_pool_size: int,
    refit_val_size: int,
    seed: int,
    exclude_refit_from_selection: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct deterministic refit train/validation splits from the natural-pool prefix."""
    if refit_val_size <= 0 or refit_val_size >= refit_pool_size:
        raise ValueError("--refit-val-size must be between 1 and refit_pool_size - 1")
    rng = np.random.default_rng(seed + stable_seed("refit_robust_selector", "refit_pool"))
    order = rng.permutation(pool_size)
    if exclude_refit_from_selection:
        initial = set(int(x) for x in selected_initial)
        order = np.asarray([idx for idx in order if int(idx) not in initial], dtype=np.int64)
    if len(order) < refit_pool_size:
        raise ValueError("Not enough pool images for requested refit pool")
    base_indices = order[:refit_pool_size].astype(np.int64)
    split_rng = np.random.default_rng(seed + stable_seed("refit_robust_selector", "refit_split"))
    perm = split_rng.permutation(refit_pool_size)
    train_pos = perm[: refit_pool_size - refit_val_size].astype(np.int64)
    val_pos = perm[refit_pool_size - refit_val_size : refit_pool_size].astype(np.int64)
    train_indices = base_indices[train_pos]
    val_indices = base_indices[val_pos]
    return base_indices, train_indices, val_indices, order


def bound_natural_feature_pool(
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    *,
    max_images: int | None,
) -> tuple[dict[str, np.ndarray], int, dict[str, Any]]:
    """Trim loaded model feature arrays to a shared, requested natural-pool size."""
    feature_lengths = {
        model: int(raw_features_np[model].shape[0])
        for model in model_names
    }
    shared_available = min(feature_lengths.values())
    if shared_available <= 0:
        raise ValueError(f"No shared natural-pool rows available: {feature_lengths}")

    if max_images is None:
        pool_size = shared_available
    else:
        requested = int(max_images)
        if requested <= 0:
            raise ValueError(f"Invalid max_images={requested}")
        if requested > shared_available:
            raise ValueError(
                f"Requested max_images={requested}, but the shortest loaded feature "
                f"array has only {shared_available} rows: {feature_lengths}"
            )
        pool_size = requested

    if any(length != pool_size for length in feature_lengths.values()):
        raw_features_np = {
            model: raw_features_np[model][:pool_size]
            for model in model_names
        }
        print(
            f"Using natural-pool prefix of {pool_size} rows; "
            f"loaded feature lengths were {feature_lengths}",
            flush=True,
        )

    pool_info = {
        "pool_feature_dir": None,
        "n_loaded": int(pool_size),
        "requested_max_images": int(max_images) if max_images is not None else None,
        "shared_available": int(shared_available),
        "feature_lengths": feature_lengths,
        "natural_feature_loader": True,
    }
    return raw_features_np, pool_size, pool_info
