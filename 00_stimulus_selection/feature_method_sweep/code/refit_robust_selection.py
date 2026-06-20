#!/usr/bin/env python3
"""Greedy selection with eval-set-augmented teacher/student refit scoring.

This is an experimental selector for the question:

    Which images preserve teacher recovery after each candidate model is allowed
    to refit a readout on an independent refit set plus the selected images?

The expensive objective is only evaluated on a shortlist.  By default that
shortlist is formed with the existing attenuated fixed-RDM sub-01 objective plus
some random exploration candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration dependency
    njit = None

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
SRC_DIR = ROOT / "src"
for path in (SRC_DIR, SCRIPT.parent):
    sys.path.insert(0, str(path))

from cstims.data_loader import load_natural_features_with_metadata, max_images_for_ram  # noqa: E402
from cstims.encoding.linear import encode_batch_for_all_encodings  # noqa: E402
from cstims.evaluation.noise_calibration import (  # noqa: E402
    calibrate_response_noise_for_rdm_reliability,
    multiplier_to_noise_ceiling,
    response_noise_std_from_mode,
)
from cstims.evaluation.ridge import (  # noqa: E402
    ridge_ops_for_eval_sets,
)
from cstims.evaluation.teacher_student import (  # noqa: E402
    detect_equivalent_models,
    pearson_columns,
    stable_seed,
)

from feature_method_sweep import (  # noqa: E402
    MethodRuntime,
    MethodSpec,
    TrackSpec,
    build_runtime,
    calibrate_noise_by_track,
    compute_track_scores,
    exclude_failed_indices,
    filter_record_to_dict,
    get_track_candidate_features,
    load_encoding_params_for_sweep,
    load_env_paths,
    load_existing_filter_records,
    load_layer_names,
    load_model_set,
    load_npz_pool_features,
    make_image_filter,
    mark_filter_failures,
    save_manifest,
    save_runtime_progress,
    select_initial_indices,
)


MODEL_LIST_CSV = ROOT / "00_stimulus_selection" / "resources" / "model_list.csv"
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


@dataclass
class SelectedAlphaState:
    a_inv_u_selected: np.ndarray
    selected_inverse: np.ndarray


@dataclass
class V2StudentCache:
    selected_inverse: np.ndarray
    selected_inverse_diag: np.ndarray
    selected_base_numerator: np.ndarray
    candidate_q: np.ndarray
    candidate_delta: np.ndarray
    candidate_z: np.ndarray


@dataclass
class V2PredictionPath:
    order: np.ndarray
    offsets: np.ndarray


@dataclass
class V2RoundCache:
    student_caches: dict[str, V2StudentCache]
    paths: dict[str, V2PredictionPath]
    blocks: list[tuple[str, int]]
    candidate_pos: dict[int, int]


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def load_existing_indices(method_dir: Path) -> list[int] | None:
    path = method_dir / "selected_indices.npy"
    if not path.exists():
        return None
    return [int(x) for x in np.load(path).tolist()]


def load_existing_rows(path: Path, *, max_iteration: int | None = None) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        rows = pd.read_csv(path).replace({np.nan: None}).to_dict("records")
    except pd.errors.EmptyDataError:
        return []
    if not rows or max_iteration is None or "iteration" not in rows[0]:
        return [dict(row) for row in rows]
    return [
        dict(row)
        for row in rows
        if row.get("iteration") is not None and int(row["iteration"]) <= max_iteration
    ]


def load_resume_state(
    *,
    method_dir: Path,
    target_size: int,
    init_size: int,
    pool_size: int,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]] | None:
    selected = load_existing_indices(method_dir)
    if selected is None:
        return None
    if len(selected) < init_size:
        raise ValueError(
            f"Cannot resume {method_dir}: selected_indices.npy has only "
            f"{len(selected)} entries, expected at least init_size={init_size}"
        )
    if len(selected) > target_size:
        raise ValueError(
            f"Cannot resume {method_dir}: selected_indices.npy has "
            f"{len(selected)} entries, exceeding target_size={target_size}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError(f"Cannot resume {method_dir}: selected indices contain duplicates")
    bad = [idx for idx in selected if idx < 0 or idx >= pool_size]
    if bad:
        raise ValueError(
            f"Cannot resume {method_dir}: selected index outside pool_size={pool_size}: "
            f"{bad[:5]}"
        )

    completed_iterations = len(selected) - init_size
    trace_rows = load_existing_rows(
        method_dir / "selection_trace.csv",
        max_iteration=completed_iterations,
    )
    if len(trace_rows) < completed_iterations:
        raise ValueError(
            f"Cannot resume {method_dir}: selection_trace.csv has {len(trace_rows)} "
            f"completed rows, expected {completed_iterations}"
        )
    if len(trace_rows) > completed_iterations:
        print(
            f"Truncating resume trace from {len(trace_rows)} to "
            f"{completed_iterations} completed iterations",
            flush=True,
        )
        trace_rows = trace_rows[:completed_iterations]

    candidate_rows = load_existing_rows(
        method_dir / "candidate_scores.csv",
        max_iteration=completed_iterations,
    )
    return selected, trace_rows, candidate_rows


def save_filter_records(method_dir: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    filter_df = pd.DataFrame(records)
    filter_df.to_csv(method_dir / "filter_records.csv", index=False)
    passed_series = (
        filter_df["passed"].map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})
        if "passed" in filter_df
        else pd.Series([], dtype=bool)
    )
    filter_summary = {
        "n_records": int(len(filter_df)),
        "n_passed": int(passed_series.sum()) if "passed" in filter_df else 0,
        "n_failed": int((~passed_series).sum()) if "passed" in filter_df else 0,
        "reason_counts": (
            filter_df["reason"].value_counts(dropna=False).to_dict()
            if "reason" in filter_df
            else {}
        ),
    }
    with (method_dir / "filter_summary.json").open("w") as f:
        json.dump(filter_summary, f, indent=2, default=str)


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{sec:04.1f}s"


def make_method(method_id: str, track: str) -> MethodSpec:
    return MethodSpec(
        method_id=method_id,
        label=f"{track} eval-augmented LOO refit robust",
        tracks=(TrackSpec(name=track, type="encoding", encoding_name=track),),
        track_agg_method="identity",
        track_norm_method="none",
        within="mean",
        across="min",
        summary_weights={track: 1.0},
        description=(
            "Experimental greedy selector. Candidate shortlist is formed by the "
            "attenuated fixed-RDM track objective, then reranked by "
            "eval-set-augmented LOO teacher/student recovery accuracy, with "
            "RDM margin used as a tie-breaker."
        ),
    )


def raw_subset_tensors(
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    indices: np.ndarray | list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
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
    return np.asarray((np.asarray(array, dtype=np.float32) - mean) / scale, dtype=np.float32)


if njit is not None:

    @njit(cache=False, nogil=True)
    def _materialize_v2_ops_numba(
        selected_inverse: np.ndarray,
        selected_inverse_diag: np.ndarray,
        selected_base_numerator: np.ndarray,
        candidate_q: np.ndarray,
        candidate_delta: np.ndarray,
        candidate_z: np.ndarray,
        candidate_pos: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_alphas = selected_inverse.shape[0]
        n_selected = selected_inverse.shape[1]
        n_base = selected_base_numerator.shape[2]
        n_eval = n_selected + 1
        base_ops = np.empty((n_alphas, n_eval, n_base), dtype=np.float32)
        eval_ops = np.empty((n_alphas, n_eval, n_eval), dtype=np.float32)

        for alpha_idx in range(n_alphas):
            delta = candidate_delta[alpha_idx, candidate_pos]
            inv_delta = 1.0 / delta
            for row in range(n_selected):
                q_row = candidate_q[alpha_idx, row, candidate_pos]
                inverse_diag = (
                    selected_inverse_diag[alpha_idx, row]
                    + q_row * q_row * inv_delta
                )
                inv_inverse_diag = 1.0 / inverse_diag
                for base_idx in range(n_base):
                    numerator = (
                        selected_base_numerator[alpha_idx, row, base_idx]
                        - q_row
                        * candidate_z[alpha_idx, base_idx, candidate_pos]
                        * inv_delta
                    )
                    base_ops[alpha_idx, row, base_idx] = np.float32(
                        numerator * inv_inverse_diag
                    )
                for col in range(n_selected):
                    if row == col:
                        eval_ops[alpha_idx, row, col] = np.float32(0.0)
                    else:
                        inverse_value = (
                            selected_inverse[alpha_idx, row, col]
                            + q_row
                            * candidate_q[alpha_idx, col, candidate_pos]
                            * inv_delta
                        )
                        eval_ops[alpha_idx, row, col] = np.float32(
                            -inverse_value * inv_inverse_diag
                        )
                eval_ops[alpha_idx, row, n_selected] = np.float32(
                    q_row * inv_delta * inv_inverse_diag
                )

            for base_idx in range(n_base):
                base_ops[alpha_idx, n_selected, base_idx] = np.float32(
                    candidate_z[alpha_idx, base_idx, candidate_pos]
                )
            for col in range(n_selected):
                eval_ops[alpha_idx, n_selected, col] = np.float32(
                    candidate_q[alpha_idx, col, candidate_pos]
                )
            eval_ops[alpha_idx, n_selected, n_selected] = np.float32(0.0)
        return base_ops, eval_ops


    @njit(cache=False, nogil=True, fastmath=True)
    def _fast_response_rdm(response: np.ndarray) -> np.ndarray:
        n_eval = response.shape[0]
        n_targets = response.shape[1]
        n_pairs = n_eval * (n_eval - 1) // 2
        inverse_norms = np.empty(n_eval, dtype=np.float64)
        for row in range(n_eval):
            squared_norm = 0.0
            for target_idx in range(n_targets):
                value = float(response[row, target_idx])
                squared_norm += value * value
            if squared_norm < 1e-24:
                inverse_norms[row] = 1.0
            else:
                inverse_norms[row] = 1.0 / math.sqrt(squared_norm)

        rdm = np.empty(n_pairs, dtype=np.float32)
        pair_idx = 0
        for row in range(n_eval - 1):
            for col in range(row + 1, n_eval):
                dot = 0.0
                for target_idx in range(n_targets):
                    dot += (
                        float(response[row, target_idx])
                        * float(response[col, target_idx])
                    )
                rdm[pair_idx] = np.float32(
                    1.0 - dot * inverse_norms[row] * inverse_norms[col]
                )
                pair_idx += 1
        return rdm


    @njit(cache=False, nogil=True, fastmath=True)
    def _fast_response_ranks(response: np.ndarray) -> np.ndarray:
        rdm = _fast_response_rdm(response)
        order = np.argsort(rdm, kind="mergesort")
        ranks = np.empty(order.size, dtype=np.int64)
        for idx in range(order.size):
            ranks[order[idx]] = idx
        return ranks


    @njit(cache=False, nogil=True, fastmath=True)
    def _fast_spearman_scores_flat(
        predicted_flat: np.ndarray,
        teacher_ranks: np.ndarray,
        n_blocks: int,
        target_dim: int,
    ) -> np.ndarray:
        n_students = predicted_flat.shape[0]
        n_eval = predicted_flat.shape[1]
        n_pairs = n_eval * (n_eval - 1) // 2
        denominator = float(n_pairs * (n_pairs * n_pairs - 1))
        scores = np.empty((n_blocks, n_students), dtype=np.float32)
        inverse_norms = np.empty(n_eval, dtype=np.float64)
        rdm = np.empty(n_pairs, dtype=np.float32)

        for block_idx in range(n_blocks):
            target_offset = block_idx * target_dim
            for student_idx in range(n_students):
                for row in range(n_eval):
                    squared_norm = 0.0
                    for target_idx in range(target_dim):
                        value = float(
                            predicted_flat[
                                student_idx,
                                row,
                                target_offset + target_idx,
                            ]
                        )
                        squared_norm += value * value
                    if squared_norm < 1e-24:
                        inverse_norms[row] = 1.0
                    else:
                        inverse_norms[row] = 1.0 / math.sqrt(squared_norm)

                pair_idx = 0
                for row in range(n_eval - 1):
                    for col in range(row + 1, n_eval):
                        dot = 0.0
                        for target_idx in range(target_dim):
                            dot += (
                                float(
                                    predicted_flat[
                                        student_idx,
                                        row,
                                        target_offset + target_idx,
                                    ]
                                )
                                * float(
                                    predicted_flat[
                                        student_idx,
                                        col,
                                        target_offset + target_idx,
                                    ]
                                )
                            )
                        rdm[pair_idx] = np.float32(
                            1.0 - dot * inverse_norms[row] * inverse_norms[col]
                        )
                        pair_idx += 1

                order = np.argsort(rdm, kind="mergesort")
                squared_rank_difference = np.int64(0)
                for rank_idx in range(n_pairs):
                    difference = (
                        np.int64(rank_idx)
                        - teacher_ranks[block_idx, order[rank_idx]]
                    )
                    squared_rank_difference += difference * difference
                scores[block_idx, student_idx] = np.float32(
                    1.0 - 6.0 * float(squared_rank_difference) / denominator
                )
        return scores


else:  # pragma: no cover - exercised only when Numba is unavailable
    _materialize_v2_ops_numba = None
    _fast_response_rdm = None
    _fast_response_ranks = None
    _fast_spearman_scores_flat = None


@torch.no_grad()
def encode_indices(
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    indices: np.ndarray | list[int],
    track: str,
    encoding_params: Any,
    device: torch.device,
    target_cols: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
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


@torch.no_grad()
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
            cal_rng = np.random.default_rng(seed + stable_seed(model, noise_mult, "rdm_empirical"))
            response_noise_std, achieved = calibrate_response_noise_for_rdm_reliability(
                y_calib,
                target_reliability=multiplier_to_noise_ceiling(noise_mult, base_noise_ceiling),
                metric=metric,
                corr_type=corr_type,
                rng=cal_rng,
                n_samples=calibration_noise_samples,
                max_iter=calibration_max_iter,
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
    )


def resolve_base_kernel_precompute(
    *,
    requested: bool,
    pool_size: int,
    refit_pool_size: int,
    model_names: list[str],
    max_ram_gb: float,
) -> tuple[bool, dict[str, Any]]:
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
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def target_rows_chunk(
    target: TeacherTargets,
    row_positions: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
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


def build_v2_round_cache(
    *,
    selected_indices: list[int],
    shortlist: np.ndarray,
    raw_features_np: dict[str, np.ndarray],
    fit_context: FitContext,
    noise_states: dict[str, list[TeacherNoiseState]],
    model_names: list[str],
    alphas: list[float],
    build_paths: bool = True,
) -> V2RoundCache:
    selected_array = np.asarray(selected_indices, dtype=np.int64)
    shortlist = np.asarray(shortlist, dtype=np.int64)
    alpha_array = np.asarray(alphas, dtype=np.float64)
    student_caches: dict[str, V2StudentCache] = {}

    for student in model_names:
        ops = fit_context.student_ops[student]
        x_selected = apply_standardization(
            raw_features_np[student][selected_array],
            ops.train_mean,
            ops.train_scale,
        )
        x_candidates = apply_standardization(
            raw_features_np[student][shortlist],
            ops.train_mean,
            ops.train_scale,
        )
        x_selected64 = x_selected.astype(np.float64, copy=False)
        x_candidates64 = x_candidates.astype(np.float64, copy=False)
        x_base64 = ops.x_base.astype(np.float64, copy=False)

        if ops.k_base_pool is not None:
            k_base_selected = np.asarray(ops.k_base_pool[:, selected_array], dtype=np.float64)
            k_base_candidates = np.asarray(ops.k_base_pool[:, shortlist], dtype=np.float64)
        else:
            k_base_selected = x_base64 @ x_selected64.T
            k_base_candidates = x_base64 @ x_candidates64.T
        k_selected_selected = x_selected64 @ x_selected64.T
        k_selected_candidates = x_selected64 @ x_candidates64.T
        k_candidate_diagonal = np.einsum("cf,cf->c", x_candidates64, x_candidates64)

        qtu_selected = ops.base_eigvecs.T @ k_base_selected
        eye_selected = np.eye(len(selected_array), dtype=np.float64)
        alpha_states: list[SelectedAlphaState] = []
        for alpha in alphas:
            alpha = float(alpha)
            denom = ops.base_eigvals + alpha
            a_inv_u_selected = ops.base_eigvecs @ (qtu_selected / denom[:, None])
            schur_selected = (
                k_selected_selected
                + alpha * eye_selected
                - k_base_selected.T @ a_inv_u_selected
            )
            schur_selected = 0.5 * (schur_selected + schur_selected.T)
            selected_inverse = np.linalg.inv(schur_selected)
            alpha_states.append(
                SelectedAlphaState(
                    a_inv_u_selected=np.asarray(a_inv_u_selected, dtype=np.float64),
                    selected_inverse=np.asarray(selected_inverse, dtype=np.float64),
                )
            )
        qtu_candidates = ops.base_eigvecs.T @ k_base_candidates
        inverse_denominators = 1.0 / (ops.base_eigvals[None, :] + alpha_array[:, None])
        a_inv_u_candidates = np.matmul(
            ops.base_eigvecs[None, :, :],
            qtu_candidates[None, :, :] * inverse_denominators[:, :, None],
        )
        selected_inverse = np.stack(
            [state.selected_inverse for state in alpha_states],
            axis=0,
        )
        a_inv_u_selected = np.stack(
            [state.a_inv_u_selected for state in alpha_states],
            axis=0,
        )
        cross = (
            k_selected_candidates[None, :, :]
            - np.matmul(k_base_selected.T[None, :, :], a_inv_u_candidates)
        )
        schur_diagonal = (
            k_candidate_diagonal[None, :]
            + alpha_array[:, None]
            - np.sum(k_base_candidates[None, :, :] * a_inv_u_candidates, axis=1)
        )
        candidate_q = np.matmul(selected_inverse, cross)
        candidate_delta = schur_diagonal - np.sum(cross * candidate_q, axis=1)
        candidate_z = a_inv_u_candidates - np.matmul(a_inv_u_selected, candidate_q)
        selected_inverse_diag = np.diagonal(selected_inverse, axis1=1, axis2=2).copy()
        selected_base_numerator = np.matmul(
            selected_inverse,
            np.transpose(a_inv_u_selected, (0, 2, 1)),
        )
        student_caches[student] = V2StudentCache(
            selected_inverse=np.asarray(selected_inverse, dtype=np.float64),
            selected_inverse_diag=np.asarray(selected_inverse_diag, dtype=np.float64),
            selected_base_numerator=np.asarray(selected_base_numerator, dtype=np.float64),
            candidate_q=np.asarray(candidate_q, dtype=np.float64),
            candidate_delta=np.asarray(candidate_delta, dtype=np.float64),
            candidate_z=np.asarray(candidate_z, dtype=np.float64),
        )

    blocks = [
        (teacher, noise_idx)
        for teacher in model_names
        for noise_idx in range(len(noise_states[teacher]))
    ]
    paths: dict[str, V2PredictionPath] = {}
    if build_paths:
        target_dim = fit_context.teacher_targets[model_names[0]].y_base_clean.shape[1]
        for student in model_names:
            alpha_indices: list[np.ndarray] = []
            for alpha_idx in range(len(alphas)):
                packed_indices: list[np.ndarray] = []
                for block_idx, (teacher, noise_idx) in enumerate(blocks):
                    noise_state = noise_states[teacher][noise_idx]
                    _alpha_values, best_alpha_idx = noise_state.alpha_choices[student]
                    cols = np.flatnonzero(best_alpha_idx == alpha_idx)
                    packed_indices.append(block_idx * target_dim + cols)
                alpha_indices.append(np.concatenate(packed_indices))
            order = np.concatenate(alpha_indices).astype(np.int64, copy=False)
            offsets = np.zeros(len(alphas) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(
                np.asarray([indices.size for indices in alpha_indices], dtype=np.int64)
            )
            paths[student] = V2PredictionPath(
                order=order,
                offsets=offsets,
            )

    return V2RoundCache(
        student_caches=student_caches,
        paths=paths,
        blocks=blocks,
        candidate_pos={int(candidate): pos for pos, candidate in enumerate(shortlist)},
    )


def materialize_v2_ops(
    cache: V2StudentCache,
    candidate_pos: int,
    *,
    delta_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray] | None:
    delta = cache.candidate_delta[:, candidate_pos]
    if np.any(delta <= delta_tol) or not np.all(np.isfinite(delta)):
        return None
    assert _materialize_v2_ops_numba is not None
    return _materialize_v2_ops_numba(
        cache.selected_inverse,
        cache.selected_inverse_diag,
        cache.selected_base_numerator,
        cache.candidate_q,
        cache.candidate_delta,
        cache.candidate_z,
        candidate_pos,
    )


def encoded_eval_for_candidate(
    *,
    encoded_eval_pool: dict[str, np.ndarray],
    encoded_pos: dict[int, int],
    selected_indices: list[int],
    candidate_idx: int,
) -> dict[str, np.ndarray]:
    eval_positions = [encoded_pos[int(idx)] for idx in [*selected_indices, int(candidate_idx)]]
    return {
        model: arr[eval_positions].astype(np.float32, copy=False)
        for model, arr in encoded_eval_pool.items()
    }


def prepare_v2_eval_data(
    *,
    round_cache: V2RoundCache,
    encoded_eval_by_model: dict[str, np.ndarray],
    eval_indices: np.ndarray,
    fit_context: FitContext,
    noise_mult: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_blocks = len(round_cache.blocks)
    n_eval = len(eval_indices)
    target_dim = next(iter(encoded_eval_by_model.values())).shape[1]
    eval_fit = np.empty((n_blocks, n_eval, target_dim), dtype=np.float32)
    eval_score = np.empty((n_blocks, n_eval, target_dim), dtype=np.float32)
    for block_idx, (teacher, noise_sample_idx) in enumerate(round_cache.blocks):
        teacher_target = fit_context.teacher_targets[teacher]
        eval_y_clean = encoded_eval_by_model[teacher]
        std = teacher_target.response_noise_std
        eval_fit[block_idx] = eval_y_clean + response_noise_rows(
            image_indices=eval_indices,
            n_targets=target_dim,
            std=std,
            seed=seed,
            parts=("eval_fit_noise", teacher, noise_mult, noise_sample_idx),
        )
        eval_score[block_idx] = eval_y_clean + response_noise_rows(
            image_indices=eval_indices,
            n_targets=target_dim,
            std=std,
            seed=seed,
            parts=("eval_score_noise", teacher, noise_mult, noise_sample_idx),
        )
    return eval_fit, eval_score


def gather_y_base_fit_columns(
    *,
    noise_states: dict[str, list[TeacherNoiseState]],
    blocks: list[tuple[str, int]],
    flat_indices: np.ndarray,
    target_dim: int,
) -> np.ndarray:
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    first_teacher, first_noise_idx = blocks[0]
    n_base = noise_states[first_teacher][first_noise_idx].y_base_fit.shape[0]
    out = np.empty((n_base, len(flat_indices)), dtype=np.float32)
    block_indices = flat_indices // int(target_dim)
    target_indices = flat_indices % int(target_dim)
    for block_idx in np.unique(block_indices):
        teacher, noise_idx = blocks[int(block_idx)]
        positions = np.flatnonzero(block_indices == block_idx)
        out[:, positions] = noise_states[teacher][noise_idx].y_base_fit[
            :,
            target_indices[positions],
        ]
    return out


def predict_student_flat_v2(
    *,
    student_idx: int,
    student: str,
    base_ops: np.ndarray,
    eval_ops: np.ndarray,
    eval_fit_flat: np.ndarray,
    predicted_flat: np.ndarray,
    round_cache: V2RoundCache,
    noise_states: dict[str, list[TeacherNoiseState]],
    target_dim: int,
    target_batch_size: int,
) -> None:
    path = round_cache.paths[student]
    target_batch_size = max(1, int(target_batch_size))
    for alpha_idx in range(base_ops.shape[0]):
        lo = int(path.offsets[alpha_idx])
        hi = int(path.offsets[alpha_idx + 1])
        if hi <= lo:
            continue
        for start in range(lo, hi, target_batch_size):
            end = min(start + target_batch_size, hi)
            flat_cols = path.order[start:end]
            y_base_cols = gather_y_base_fit_columns(
                noise_states=noise_states,
                blocks=round_cache.blocks,
                flat_indices=flat_cols,
                target_dim=target_dim,
            )
            eval_cols = eval_fit_flat[:, flat_cols]
            predicted_flat[student_idx, :, flat_cols] = (
                base_ops[alpha_idx] @ y_base_cols
                + eval_ops[alpha_idx] @ eval_cols
            )


def pair_index_arrays(n_eval: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.triu_indices(int(n_eval), k=1)
    return rows.astype(np.int64, copy=False), cols.astype(np.int64, copy=False)


def accumulate_response_gram(
    *,
    response: np.ndarray,
    row_norms: np.ndarray,
    pair_dots: np.ndarray,
    pair_rows: np.ndarray,
    pair_cols: np.ndarray,
) -> None:
    response = np.asarray(response, dtype=np.float32)
    gram = np.asarray(response @ response.T, dtype=np.float64)
    row_norms += np.diag(gram)
    pair_dots += gram[pair_rows, pair_cols]


def ordinal_ranks_from_response_stats(
    *,
    row_norms: np.ndarray,
    pair_dots: np.ndarray,
    pair_rows: np.ndarray,
    pair_cols: np.ndarray,
) -> np.ndarray:
    denom = np.sqrt(row_norms[pair_rows] * row_norms[pair_cols])
    similarity = np.zeros_like(pair_dots, dtype=np.float64)
    valid = denom > 1e-24
    similarity[valid] = pair_dots[valid] / denom[valid]
    rdm = np.asarray(1.0 - similarity, dtype=np.float64)
    order = np.argsort(rdm, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size, dtype=np.int64)
    return ranks


def spearman_from_ordinal_ranks(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
) -> float:
    n_pairs = int(ranks_a.size)
    if n_pairs < 2:
        return float("nan")
    diff = ranks_a.astype(np.int64, copy=False) - ranks_b.astype(np.int64, copy=False)
    sum_sq = float(np.sum(diff * diff, dtype=np.int64))
    denominator = float(n_pairs * (n_pairs * n_pairs - 1))
    return float(1.0 - 6.0 * sum_sq / denominator)


def score_candidate_refit_robust_v2_streaming(
    *,
    candidate_idx: int,
    selected_indices: list[int],
    encoded_eval_pool: dict[str, np.ndarray],
    encoded_pos: dict[int, int],
    fit_context: FitContext,
    model_names: list[str],
    metric: str,
    corr_type: str,
    noise_mult: float,
    base_noise_ceiling: float,
    seed: int,
    aggregate_teachers: str,
    objective: str,
    round_cache: V2RoundCache,
    noise_states: dict[str, list[TeacherNoiseState]],
    score_target_batch_size: int,
) -> dict[str, Any]:
    if metric != "cosine" or corr_type != "spearman":
        raise ValueError("Refit-robust streaming scoring supports only cosine/Spearman")

    candidate_pos = round_cache.candidate_pos[int(candidate_idx)]
    eval_indices = np.asarray([*selected_indices, int(candidate_idx)], dtype=np.int64)
    eval_positions = np.asarray([encoded_pos[int(idx)] for idx in eval_indices], dtype=np.int64)
    n_eval = int(len(eval_indices))
    n_blocks = int(len(round_cache.blocks))
    n_students = int(len(model_names))
    pair_rows, pair_cols = pair_index_arrays(n_eval)
    n_pairs = int(len(pair_rows))
    scores = np.empty((n_blocks, n_students), dtype=np.float32)

    dense_ops_by_student: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for student in model_names:
        dense_ops = materialize_v2_ops(round_cache.student_caches[student], candidate_pos)
        if dense_ops is None:
            raise RuntimeError(
                f"Numerically unstable candidate delta for candidate {candidate_idx}, "
                f"student {student}"
            )
        dense_ops_by_student[student] = dense_ops

    for block_idx, (teacher, noise_sample_idx) in enumerate(round_cache.blocks):
        target = fit_context.teacher_targets[teacher]
        noise_state = noise_states[teacher][noise_sample_idx]
        target_dim = target.n_targets
        target_batch_size = min(max(1, int(score_target_batch_size)), target_dim)
        teacher_row_norms = np.zeros(n_eval, dtype=np.float64)
        teacher_pair_dots = np.zeros(n_pairs, dtype=np.float64)
        student_row_norms = np.zeros((n_students, n_eval), dtype=np.float64)
        student_pair_dots = np.zeros((n_students, n_pairs), dtype=np.float64)
        encoded_teacher = encoded_eval_pool[teacher]

        for start in range(0, target_dim, target_batch_size):
            end = min(start + target_batch_size, target_dim)
            eval_y_clean = np.asarray(
                encoded_teacher[eval_positions, start:end],
                dtype=np.float32,
            )
            eval_fit = eval_y_clean + response_noise_rows_chunk(
                image_indices=eval_indices,
                n_targets=target_dim,
                start=start,
                end=end,
                std=target.response_noise_std,
                seed=seed,
                parts=("eval_fit_noise", teacher, noise_mult, noise_sample_idx),
            )
            eval_score = eval_y_clean + response_noise_rows_chunk(
                image_indices=eval_indices,
                n_targets=target_dim,
                start=start,
                end=end,
                std=target.response_noise_std,
                seed=seed,
                parts=("eval_score_noise", teacher, noise_mult, noise_sample_idx),
            )
            accumulate_response_gram(
                response=eval_score,
                row_norms=teacher_row_norms,
                pair_dots=teacher_pair_dots,
                pair_rows=pair_rows,
                pair_cols=pair_cols,
            )

            y_base_fit_chunk = np.asarray(
                noise_state.y_base_fit[:, start:end],
                dtype=np.float32,
            )
            for student_idx, student in enumerate(model_names):
                base_ops, eval_ops = dense_ops_by_student[student]
                _alpha_values, best_alpha_idx = noise_state.alpha_choices[student]
                alpha_chunk = best_alpha_idx[start:end]
                pred_chunk = np.empty((n_eval, end - start), dtype=np.float32)
                for alpha_idx_raw in np.unique(alpha_chunk):
                    alpha_idx = int(alpha_idx_raw)
                    cols = np.flatnonzero(alpha_chunk == alpha_idx)
                    if cols.size == 0:
                        continue
                    pred_chunk[:, cols] = (
                        base_ops[alpha_idx] @ y_base_fit_chunk[:, cols]
                        + eval_ops[alpha_idx] @ eval_fit[:, cols]
                    )
                accumulate_response_gram(
                    response=pred_chunk,
                    row_norms=student_row_norms[student_idx],
                    pair_dots=student_pair_dots[student_idx],
                    pair_rows=pair_rows,
                    pair_cols=pair_cols,
                )

        teacher_ranks = ordinal_ranks_from_response_stats(
            row_norms=teacher_row_norms,
            pair_dots=teacher_pair_dots,
            pair_rows=pair_rows,
            pair_cols=pair_cols,
        )
        for student_idx in range(n_students):
            student_ranks = ordinal_ranks_from_response_stats(
                row_norms=student_row_norms[student_idx],
                pair_dots=student_pair_dots[student_idx],
                pair_rows=pair_rows,
                pair_cols=pair_cols,
            )
            scores[block_idx, student_idx] = np.float32(
                spearman_from_ordinal_ranks(student_ranks, teacher_ranks)
            )

    return aggregate_v2_scores(
        scores=scores,
        candidate_idx=candidate_idx,
        n_eval=n_eval,
        fit_context=fit_context,
        model_names=model_names,
        round_cache=round_cache,
        aggregate_teachers=aggregate_teachers,
        noise_mult=noise_mult,
        base_noise_ceiling=base_noise_ceiling,
        objective=objective,
        score_backend="v2-stream",
    )


def score_candidate_refit_robust_v2_chunked(
    *,
    candidate_idx: int,
    selected_indices: list[int],
    encoded_eval_by_model: dict[str, np.ndarray],
    fit_context: FitContext,
    model_names: list[str],
    metric: str,
    corr_type: str,
    noise_mult: float,
    base_noise_ceiling: float,
    seed: int,
    aggregate_teachers: str,
    objective: str,
    round_cache: V2RoundCache,
    noise_states: dict[str, list[TeacherNoiseState]],
    score_target_batch_size: int,
) -> dict[str, Any]:
    if metric != "cosine" or corr_type != "spearman":
        raise ValueError("Refit-robust chunked scoring supports only cosine/Spearman")

    candidate_pos = round_cache.candidate_pos[int(candidate_idx)]
    eval_indices = np.asarray([*selected_indices, int(candidate_idx)], dtype=np.int64)
    dense_ops_by_student: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for student in model_names:
        dense_ops = materialize_v2_ops(round_cache.student_caches[student], candidate_pos)
        if dense_ops is None:
            raise RuntimeError(
                f"Numerically unstable candidate delta for candidate {candidate_idx}, "
                f"student {student}"
            )
        dense_ops_by_student[student] = dense_ops

    eval_fit, eval_score = prepare_v2_eval_data(
        round_cache=round_cache,
        encoded_eval_by_model=encoded_eval_by_model,
        eval_indices=eval_indices,
        fit_context=fit_context,
        noise_mult=noise_mult,
        seed=seed,
    )
    n_eval = int(len(eval_indices))
    n_blocks = int(len(round_cache.blocks))
    n_students = int(len(model_names))
    target_dim = int(eval_fit.shape[2])
    total_targets = n_blocks * target_dim
    target_batch_size = min(max(1, int(score_target_batch_size)), total_targets)
    pair_rows, pair_cols = pair_index_arrays(n_eval)
    n_pairs = int(len(pair_rows))
    eval_fit_flat = np.ascontiguousarray(
        np.transpose(eval_fit, (1, 0, 2)).reshape(n_eval, total_targets)
    )

    assert _fast_response_ranks is not None
    teacher_ranks = np.empty((n_blocks, n_pairs), dtype=np.int64)
    for block_idx in range(n_blocks):
        teacher_ranks[block_idx] = _fast_response_ranks(eval_score[block_idx])

    student_row_norms = np.zeros((n_students, n_blocks, n_eval), dtype=np.float64)
    student_pair_dots = np.zeros((n_students, n_blocks, n_pairs), dtype=np.float64)
    for student_idx, student in enumerate(model_names):
        path = round_cache.paths[student]
        base_ops, eval_ops = dense_ops_by_student[student]
        for alpha_idx in range(base_ops.shape[0]):
            lo = int(path.offsets[alpha_idx])
            hi = int(path.offsets[alpha_idx + 1])
            if hi <= lo:
                continue
            for start in range(lo, hi, target_batch_size):
                end = min(start + target_batch_size, hi)
                flat_cols = path.order[start:end]
                y_base_cols = gather_y_base_fit_columns(
                    noise_states=noise_states,
                    blocks=round_cache.blocks,
                    flat_indices=flat_cols,
                    target_dim=target_dim,
                )
                pred_chunk = (
                    base_ops[alpha_idx] @ y_base_cols
                    + eval_ops[alpha_idx] @ eval_fit_flat[:, flat_cols]
                )
                block_indices = flat_cols // int(target_dim)
                for block_idx_raw in np.unique(block_indices):
                    block_idx = int(block_idx_raw)
                    cols = np.flatnonzero(block_indices == block_idx)
                    if cols.size == 0:
                        continue
                    accumulate_response_gram(
                        response=pred_chunk[:, cols],
                        row_norms=student_row_norms[student_idx, block_idx],
                        pair_dots=student_pair_dots[student_idx, block_idx],
                        pair_rows=pair_rows,
                        pair_cols=pair_cols,
                    )

    scores = np.empty((n_blocks, n_students), dtype=np.float32)
    for block_idx in range(n_blocks):
        for student_idx in range(n_students):
            student_ranks = ordinal_ranks_from_response_stats(
                row_norms=student_row_norms[student_idx, block_idx],
                pair_dots=student_pair_dots[student_idx, block_idx],
                pair_rows=pair_rows,
                pair_cols=pair_cols,
            )
            scores[block_idx, student_idx] = np.float32(
                spearman_from_ordinal_ranks(student_ranks, teacher_ranks[block_idx])
            )

    return aggregate_v2_scores(
        scores=scores,
        candidate_idx=candidate_idx,
        n_eval=n_eval,
        fit_context=fit_context,
        model_names=model_names,
        round_cache=round_cache,
        aggregate_teachers=aggregate_teachers,
        noise_mult=noise_mult,
        base_noise_ceiling=base_noise_ceiling,
        objective=objective,
        score_backend="v2-chunked",
    )


def aggregate_v2_scores(
    *,
    scores: np.ndarray,
    candidate_idx: int,
    n_eval: int,
    fit_context: FitContext,
    model_names: list[str],
    round_cache: V2RoundCache,
    aggregate_teachers: str,
    noise_mult: float,
    base_noise_ceiling: float,
    objective: str,
    score_backend: str = "v2-fast",
) -> dict[str, Any]:
    teacher_utilities: list[float] = []
    teacher_self_scores: list[float] = []
    teacher_other_scores: list[float] = []
    teacher_majority_correct: list[bool] = []
    all_sample_correct: list[bool] = []
    block_idx = 0
    for teacher_idx, teacher in enumerate(model_names):
        teacher_equiv_label = int(fit_context.equivalence_labels[teacher_idx])
        off_equiv = np.asarray(
            [label != teacher_equiv_label for label in fit_context.equivalence_labels],
            dtype=bool,
        )
        sample_utilities = []
        sample_self_scores = []
        sample_other_scores = []
        sample_correct = []
        n_teacher_samples = sum(1 for block_teacher, _ in round_cache.blocks if block_teacher == teacher)
        for _ in range(n_teacher_samples):
            row = np.nan_to_num(scores[block_idx], nan=-np.inf)
            self_score = float(row[teacher_idx])
            competitor_scores = row[off_equiv]
            other_score = float(np.max(competitor_scores)) if len(competitor_scores) else float("nan")
            recovered_idx = int(np.argmax(row))
            correct = int(fit_context.equivalence_labels[recovered_idx]) == teacher_equiv_label
            sample_self_scores.append(self_score)
            sample_other_scores.append(other_score)
            utility = float(self_score - other_score)
            sample_utilities.append(utility)
            sample_correct.append(bool(correct))
            all_sample_correct.append(bool(correct))
            block_idx += 1
        teacher_utilities.append(float(np.mean(sample_utilities)))
        teacher_self_scores.append(float(np.mean(sample_self_scores)))
        teacher_other_scores.append(float(np.mean(sample_other_scores)))
        teacher_majority_correct.append(bool(np.mean(sample_correct) >= 0.5))

    if aggregate_teachers == "mean":
        margin_score = float(np.mean(teacher_utilities))
    elif aggregate_teachers == "min":
        margin_score = float(np.min(teacher_utilities))
    else:
        raise ValueError(f"Unsupported teacher aggregation: {aggregate_teachers}")
    recovery_accuracy = float(np.mean(all_sample_correct))
    teacher_majority_recovery_accuracy = float(np.mean(teacher_majority_correct))
    if objective == "accuracy_margin":
        score = recovery_accuracy
        score_tie_breaker = margin_score
    elif objective == "margin":
        score = margin_score
        score_tie_breaker = recovery_accuracy
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    noise_ceiling = multiplier_to_noise_ceiling(noise_mult, base_noise_ceiling)
    return {
        "candidate_index": int(candidate_idx),
        "n_eval": int(n_eval),
        "score": score,
        "score_tie_breaker": score_tie_breaker,
        "score_objective": objective,
        "score_recovery_accuracy": recovery_accuracy,
        "score_margin": margin_score,
        "teacher_margin_mean": float(np.mean(teacher_utilities)),
        "teacher_margin_min": float(np.min(teacher_utilities)),
        "teacher_self_score_mean": float(np.mean(teacher_self_scores)),
        "teacher_other_score_mean": float(np.mean(teacher_other_scores)),
        "recovery_accuracy": recovery_accuracy,
        "teacher_majority_recovery_accuracy": teacher_majority_recovery_accuracy,
        "noise_mult": float(noise_mult),
        "noise_ceiling": float(noise_ceiling),
        "score_backend": score_backend,
    }


def score_candidate_refit_robust_v2_fast(
    *,
    candidate_idx: int,
    selected_indices: list[int],
    encoded_eval_by_model: dict[str, np.ndarray],
    fit_context: FitContext,
    model_names: list[str],
    alphas: list[float],
    metric: str,
    corr_type: str,
    noise_mult: float,
    base_noise_ceiling: float,
    seed: int,
    aggregate_teachers: str,
    objective: str,
    round_cache: V2RoundCache,
    noise_states: dict[str, list[TeacherNoiseState]],
    score_target_batch_size: int,
) -> dict[str, Any]:
    if njit is None:
        raise RuntimeError("V2-fast scoring requires numba")
    if metric != "cosine" or corr_type != "spearman":
        raise ValueError("Refit-robust V2-fast currently supports only cosine/Spearman")

    candidate_pos = round_cache.candidate_pos[int(candidate_idx)]
    eval_indices = np.asarray([*selected_indices, int(candidate_idx)], dtype=np.int64)
    dense_ops_by_student: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for student in model_names:
        dense_ops = materialize_v2_ops(round_cache.student_caches[student], candidate_pos)
        if dense_ops is None:
            raise RuntimeError(
                f"Numerically unstable candidate delta for candidate {candidate_idx}, "
                f"student {student}"
            )
        dense_ops_by_student[student] = dense_ops

    eval_fit, eval_score = prepare_v2_eval_data(
        round_cache=round_cache,
        encoded_eval_by_model=encoded_eval_by_model,
        eval_indices=eval_indices,
        fit_context=fit_context,
        noise_mult=noise_mult,
        seed=seed,
    )
    n_eval = len(eval_indices)
    n_blocks = len(round_cache.blocks)
    target_dim = eval_fit.shape[2]
    total_targets = n_blocks * target_dim
    eval_fit_flat = np.ascontiguousarray(
        np.transpose(eval_fit, (1, 0, 2)).reshape(n_eval, total_targets)
    )
    predicted_flat = np.empty((len(model_names), n_eval, total_targets), dtype=np.float32)

    for student_idx, student in enumerate(model_names):
        base_ops, eval_ops = dense_ops_by_student[student]
        predict_student_flat_v2(
            student_idx=student_idx,
            student=student,
            base_ops=base_ops,
            eval_ops=eval_ops,
            eval_fit_flat=eval_fit_flat,
            predicted_flat=predicted_flat,
            round_cache=round_cache,
            noise_states=noise_states,
            target_dim=target_dim,
            target_batch_size=score_target_batch_size,
        )

    assert _fast_response_ranks is not None
    assert _fast_spearman_scores_flat is not None
    n_pairs = n_eval * (n_eval - 1) // 2
    teacher_ranks = np.empty((n_blocks, n_pairs), dtype=np.int64)
    for block_idx in range(n_blocks):
        teacher_ranks[block_idx] = _fast_response_ranks(eval_score[block_idx])
    scores = _fast_spearman_scores_flat(
        predicted_flat,
        teacher_ranks,
        n_blocks,
        target_dim,
    )

    return aggregate_v2_scores(
        scores=scores,
        candidate_idx=candidate_idx,
        n_eval=n_eval,
        fit_context=fit_context,
        model_names=model_names,
        round_cache=round_cache,
        aggregate_teachers=aggregate_teachers,
        noise_mult=noise_mult,
        base_noise_ceiling=base_noise_ceiling,
        objective=objective,
    )


_V2_WORKER_ARGS: dict[str, Any] | None = None


def _score_v2_worker(candidate_idx: int) -> dict[str, Any]:
    assert _V2_WORKER_ARGS is not None
    kwargs = _V2_WORKER_ARGS
    encoded_eval_by_model = encoded_eval_for_candidate(
        encoded_eval_pool=kwargs["encoded_eval_pool"],
        encoded_pos=kwargs["encoded_pos"],
        selected_indices=kwargs["selected_indices"],
        candidate_idx=int(candidate_idx),
    )
    return score_candidate_refit_robust_v2_chunked(
        candidate_idx=int(candidate_idx),
        selected_indices=kwargs["selected_indices"],
        encoded_eval_by_model=encoded_eval_by_model,
        fit_context=kwargs["fit_context"],
        model_names=kwargs["model_names"],
        metric=kwargs["metric"],
        corr_type=kwargs["corr_type"],
        noise_mult=kwargs["noise_mult"],
        base_noise_ceiling=kwargs["base_noise_ceiling"],
        seed=kwargs["seed"],
        aggregate_teachers=kwargs["aggregate_teachers"],
        objective=kwargs["objective"],
        round_cache=kwargs["round_cache"],
        noise_states=kwargs["noise_states"],
        score_target_batch_size=kwargs["score_target_batch_size"],
    )


def warm_v2_fast(round_cache: V2RoundCache, model_names: list[str], alphas: list[float]) -> None:
    if njit is None:
        return
    first_student = model_names[0]
    cache = round_cache.student_caches[first_student]
    _materialize_v2_ops_numba(
        cache.selected_inverse,
        cache.selected_inverse_diag,
        cache.selected_base_numerator,
        cache.candidate_q,
        cache.candidate_delta,
        cache.candidate_z,
        0,
    )
    n_eval = cache.selected_inverse.shape[1] + 1
    target_dim = round_cache.paths[first_student].order.size // max(
        1, len(round_cache.blocks)
    )
    n_blocks = len(round_cache.blocks)
    n_pairs = n_eval * (n_eval - 1) // 2
    dummy_response = np.zeros((n_eval, target_dim), dtype=np.float32)
    dummy_predicted = np.zeros(
        (len(model_names), n_eval, n_blocks * target_dim),
        dtype=np.float32,
    )
    _fast_response_ranks(dummy_response)
    _fast_spearman_scores_flat(
        dummy_predicted,
        np.zeros((n_blocks, n_pairs), dtype=np.int64),
        n_blocks,
        target_dim,
    )


def score_shortlist_refit_robust_v2_fast(
    *,
    shortlist: np.ndarray,
    selected_indices: list[int],
    encoded_eval_pool: dict[str, np.ndarray],
    encoded_pos: dict[int, int],
    raw_features_np: dict[str, np.ndarray],
    fit_context: FitContext,
    noise_states: dict[str, list[TeacherNoiseState]],
    model_names: list[str],
    alphas: list[float],
    metric: str,
    corr_type: str,
    noise_mult: float,
    base_noise_ceiling: float,
    seed: int,
    aggregate_teachers: str,
    objective: str,
    workers: int,
    score_target_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if njit is None:
        raise RuntimeError("V2-fast scoring requires numba")
    if metric != "cosine" or corr_type != "spearman":
        raise ValueError("Refit-robust selection currently supports only cosine/Spearman")

    timing: dict[str, Any] = {"backend": "v2-chunked"}
    start = time.monotonic()
    round_cache = build_v2_round_cache(
        selected_indices=selected_indices,
        shortlist=shortlist,
        raw_features_np=raw_features_np,
        fit_context=fit_context,
        noise_states=noise_states,
        model_names=model_names,
        alphas=alphas,
        build_paths=True,
    )
    timing["cache_seconds"] = float(time.monotonic() - start)
    timing["minimum_delta"] = float(
        min(np.min(cache.candidate_delta) for cache in round_cache.student_caches.values())
    )
    timing["delta_fallback_student_candidate_pairs"] = int(
        sum(
            np.count_nonzero(
                np.any(
                    (cache.candidate_delta <= 1e-10) | ~np.isfinite(cache.candidate_delta),
                    axis=0,
                )
            )
            for cache in round_cache.student_caches.values()
        )
    )

    start = time.monotonic()
    first_student = model_names[0]
    cache = round_cache.student_caches[first_student]
    _materialize_v2_ops_numba(
        cache.selected_inverse,
        cache.selected_inverse_diag,
        cache.selected_base_numerator,
        cache.candidate_q,
        cache.candidate_delta,
        cache.candidate_z,
        0,
    )
    timing["warmup_seconds"] = float(time.monotonic() - start)

    common_kwargs = {
        "selected_indices": selected_indices,
        "encoded_eval_pool": encoded_eval_pool,
        "encoded_pos": encoded_pos,
        "fit_context": fit_context,
        "model_names": model_names,
        "alphas": alphas,
        "metric": metric,
        "corr_type": corr_type,
        "noise_mult": noise_mult,
        "base_noise_ceiling": base_noise_ceiling,
        "seed": seed,
        "aggregate_teachers": aggregate_teachers,
        "objective": objective,
        "round_cache": round_cache,
        "noise_states": noise_states,
        "score_target_batch_size": score_target_batch_size,
    }

    start = time.monotonic()
    workers = min(max(1, int(workers)), len(shortlist))
    if workers > 1:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("--refit-score-workers > 1 requires fork")
        global _V2_WORKER_ARGS
        _V2_WORKER_ARGS = common_kwargs
        context = mp.get_context("fork")
        with context.Pool(workers) as pool:
            rows = pool.map(_score_v2_worker, [int(x) for x in shortlist], chunksize=8)
    else:
        rows = []
        for candidate_idx in shortlist:
            encoded_eval_by_model = encoded_eval_for_candidate(
                encoded_eval_pool=encoded_eval_pool,
                encoded_pos=encoded_pos,
                selected_indices=selected_indices,
                candidate_idx=int(candidate_idx),
            )
            rows.append(
                score_candidate_refit_robust_v2_chunked(
                    candidate_idx=int(candidate_idx),
                    encoded_eval_by_model=encoded_eval_by_model,
                    **{
                        key: value
                        for key, value in common_kwargs.items()
                        if key not in {"encoded_eval_pool", "encoded_pos", "alphas"}
                    },
                )
            )
    timing["score_seconds"] = float(time.monotonic() - start)
    timing["workers"] = int(workers)
    return rows, timing


def build_refit_splits(
    *,
    pool_size: int,
    selected_initial: list[int],
    refit_pool_size: int,
    refit_val_size: int,
    seed: int,
    exclude_refit_from_selection: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def run_selection(args: argparse.Namespace) -> Path:
    paths = load_env_paths(args.env)
    local_encoding_root = (
        ROOT
        / "01_brain_model_alignment"
        / "results"
        / "encoding_models"
        / "shared_subject_encoding_models"
        / "encoding_20251222_141301"
    )
    encoding_root = Path(paths.get("encoding_root", ""))
    if not encoding_root.exists() and local_encoding_root.exists():
        paths["encoding_root"] = str(local_encoding_root)
        print(f"Using local encoding_root: {local_encoding_root}", flush=True)
    model_set_name, model_names = load_model_set(args.model_set)

    model_list_csv = Path(paths.get("model_list_csv") or MODEL_LIST_CSV)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payload_root = output_root / "payloads"
    method = make_method(args.method_id, args.track)
    method_dir = payload_root / args.method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    save_manifest([method], payload_root)

    pool_feature_dir = args.pool_feature_dir
    max_images_arg = args.max_images
    pool_records_by_index: list[dict[str, Any]] | None = None
    pool_info: dict[str, Any] | None = None
    if pool_feature_dir is not None:
        if not args.disable_image_filter:
            raise ValueError(
                "Image filtering matches feature_method_sweep.py and requires the "
                "natural LAION feature loader. Do not pass --pool-feature-dir for "
                "filtered refit runs; use --disable-image-filter for explicit "
                "unfiltered .npz-pool runs."
            )
        pool_feature_dir = Path(pool_feature_dir).resolve()
        max_images = int(max_images_arg) if max_images_arg is not None else None
        raw_features_np, pool_records_by_index, pool_info = load_npz_pool_features(
            pool_feature_dir=pool_feature_dir,
            model_names=model_names,
            max_images=max_images,
        )
        raw_shard_slices = []
        pool_size = int(next(iter(raw_features_np.values())).shape[0])
    else:
        layer_names = load_layer_names(model_list_csv, model_names)
        if max_images_arg is not None:
            max_images = int(max_images_arg)
        else:
            max_images = max_images_for_ram(
                subset_root=Path(paths["subset_root"]),
                model_names=model_names,
                max_ram_bytes=int(args.max_ram_gb * 1024**3),
                model_csv=model_list_csv,
            )
        print(f"Loading {max_images} natural-pool images for model_set={model_set_name}")
        raw_features_np, raw_shard_slices = load_natural_features_with_metadata(
            subset_root=Path(paths["subset_root"]),
            preprocessed_dir=Path(paths["preprocessed_dirs"]["raw"]),
            model_names=model_names,
            layer_names=layer_names,
            max_images=max_images,
            model_csv=model_list_csv,
        )
        raw_features_np, pool_size, pool_info = bound_natural_feature_pool(
            raw_features_np,
            model_names,
            max_images=max_images,
        )
    if pool_size <= args.init_size:
        raise ValueError(f"Candidate pool too small: pool_size={pool_size}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    encoding_params = load_encoding_params_for_sweep(
        paths=paths,
        model_list_csv=model_list_csv,
        encoding_names=[args.track],
        device=device,
        roi_subset=args.encoding_roi_subset,
        shared_encodings=not args.unique_encodings,
    )
    if args.track not in encoding_params:
        raise RuntimeError(f"Missing encoding params for {args.track}")

    image_filter, image_filter_config = make_image_filter(
        args=args,
        paths=paths,
        raw_shard_slices=raw_shard_slices,
        output_root=output_root,
    )
    print(f"Image filter enabled: {image_filter is not None}", flush=True)
    if image_filter is not None:
        print(f"Image filter config: {image_filter_config}", flush=True)

    existing_selected = load_existing_indices(method_dir) if args.resume else None
    if existing_selected is not None:
        if len(existing_selected) < args.init_size:
            raise ValueError(
                f"Cannot resume {method_dir}: selected_indices.npy has only "
                f"{len(existing_selected)} entries, expected at least init_size={args.init_size}"
            )
        initial_indices = [int(x) for x in existing_selected[: args.init_size]]
        initial_filter_records: list[dict[str, Any]] = []
        initialization_source = "resume_existing_selected_indices"
    else:
        rng = np.random.default_rng(args.seed)
        initial_array, initial_filter_records_raw = select_initial_indices(
            rng=rng,
            initial_pool_size=pool_size,
            init_size=args.init_size,
            image_filter=image_filter,
        )
        initial_indices = [int(x) for x in initial_array.tolist()]
        initial_filter_records = [
            {
                **record,
                "method_id": args.method_id,
                "pool_size": pool_size,
            }
            for record in initial_filter_records_raw
        ]
        initialization_source = (
            "filtered_random_order" if image_filter is not None else "random_choice"
        )

    var_noise_by_track = calibrate_noise_by_track(
        raw_features_np=raw_features_np,
        model_names=model_names,
        track_specs=list(method.tracks),
        encoding_params=encoding_params,
        metric=args.metric,
        corr_type=args.corr_type,
        target_nc=args.noise_ceiling,
        seed=args.seed,
        device=device,
        calib_n_examples=args.proxy_noise_calib_examples,
        n_repeats=args.proxy_noise_calib_repeats,
    )
    if args.no_proxy_attenuation:
        var_noise_by_track = {args.track: {model: 0.0 for model in model_names}}

    base_indices, train_indices, val_indices, refit_order = build_refit_splits(
        pool_size=pool_size,
        selected_initial=initial_indices,
        refit_pool_size=args.refit_pool_size,
        refit_val_size=args.refit_val_size,
        seed=args.seed,
        exclude_refit_from_selection=args.exclude_refit_from_selection,
    )
    refit_lookup = {int(idx): pos for pos, idx in enumerate(base_indices)}
    refit_train_pos = np.asarray([refit_lookup[int(idx)] for idx in train_indices], dtype=np.int64)
    refit_val_pos = np.asarray([refit_lookup[int(idx)] for idx in val_indices], dtype=np.int64)

    target_cols = choose_target_columns(
        raw_features_np=raw_features_np,
        model_names=model_names,
        track=args.track,
        encoding_params=encoding_params,
        device=device,
        target_dim=args.target_dim,
        seed=args.seed,
    )
    print(
        f"Encoding refit targets for {args.track}; "
        f"target_dim={len(target_cols) if target_cols is not None else 'all'}",
        flush=True,
    )
    alphas = parse_csv_floats(args.alphas)
    precompute_base_kernels, base_kernel_precompute = resolve_base_kernel_precompute(
        requested=args.precompute_base_kernels,
        pool_size=pool_size,
        refit_pool_size=args.refit_pool_size,
        model_names=model_names,
        max_ram_gb=args.max_ram_gb,
    )
    target_cache_dir = args.noise_cache_dir / "clean_targets" if args.noise_cache_dir else None
    fit_context = build_fit_context(
        raw_features_np=raw_features_np,
        model_names=model_names,
        track=args.track,
        encoding_params=encoding_params,
        device=device,
        target_cols=target_cols,
        train_indices=train_indices,
        val_indices=val_indices,
        base_indices=base_indices,
        refit_train_pos=refit_train_pos,
        refit_val_pos=refit_val_pos,
        alphas=alphas,
        base_noise_ceiling=args.noise_ceiling,
        noise_mult=args.noise_mult,
        fit_noise_calibration=args.fit_noise_calibration,
        metric=args.metric,
        corr_type=args.corr_type,
        seed=args.seed,
        calibration_images=args.calibration_images,
        calibration_noise_samples=args.calibration_noise_samples,
        calibration_max_iter=args.calibration_max_iter,
        precompute_base_kernels=precompute_base_kernels,
        kernel_batch_size=args.kernel_batch_size,
        target_cache_dir=target_cache_dir,
        target_batch_size=args.alpha_target_batch_size,
    )
    print("Precomputing noisy refit targets and target-wise alpha choices", flush=True)
    noise_states = build_noise_states(
        fit_context=fit_context,
        model_names=model_names,
        seed=args.seed + stable_seed(args.method_id),
        alphas=alphas,
        n_noise_samples=args.n_noise_samples,
        alpha_target_batch_size=args.alpha_target_batch_size,
        noise_cache_dir=args.noise_cache_dir,
    )

    resume_state = (
        load_resume_state(
            method_dir=method_dir,
            target_size=args.target_size,
            init_size=args.init_size,
            pool_size=pool_size,
        )
        if args.resume
        else None
    )
    if resume_state is not None:
        selected_indices, trace_rows, candidate_rows = resume_state
        if selected_indices[: args.init_size] != list(initial_indices):
            raise ValueError(
                "Resume initial indices differ from the current run initialization; "
                "use the same --seed/--init-size/filter settings as the original "
                "run or start a new --output-root."
            )
        print(
            f"Resuming {args.method_id}: n_selected={len(selected_indices)}/"
            f"{args.target_size}, completed_iterations={len(trace_rows)}",
            flush=True,
        )
        filter_records = (
            load_existing_filter_records(method_dir)
            if image_filter is not None
            else []
        )
    else:
        selected_indices = list(initial_indices)
        trace_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        filter_records = initial_filter_records
    if image_filter is not None:
        mark_filter_failures(image_filter, filter_records)

    run_config = {
        "refit_robust_selection_version": "self_initialized_feature_method",
        "script": str(SCRIPT),
        "argv": sys.argv,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method_name": args.method_id,
        "model_set_name": model_set_name,
        "model_names": model_names,
        "paths": paths,
        "feature_method_sweep": True,
        "refit_robust_selection": True,
        "target_size": args.target_size,
        "init_size": args.init_size,
        "initial_indices": selected_indices[: args.init_size],
        "initialization_source": initialization_source,
        "seed": args.seed,
        "metric": args.metric,
        "corr_type": args.corr_type,
        "track": args.track,
        "candidate_pool_size": pool_size,
        "pool_feature_dir": str(pool_feature_dir) if pool_feature_dir is not None else None,
        "pool_info": pool_info,
        "max_ram_gb": args.max_ram_gb,
        "max_loaded_images": pool_size,
        "image_filter": image_filter_config,
        "refit_pool_size": args.refit_pool_size,
        "refit_val_size": args.refit_val_size,
        "refit_train_n": args.refit_pool_size - args.refit_val_size,
        "noise_mult": args.noise_mult,
        "noise_ceiling_target": args.noise_ceiling,
        "proxy_noise_calib_examples": args.proxy_noise_calib_examples,
        "proxy_noise_calib_repeats": args.proxy_noise_calib_repeats,
        "proxy_attenuation_disabled": bool(args.no_proxy_attenuation),
        "fit_noise_calibration": args.fit_noise_calibration,
        "n_noise_samples": args.n_noise_samples,
        "alphas": alphas,
        "target_dim": len(target_cols) if target_cols is not None else None,
        "top_k_proxy": args.top_k_proxy,
        "random_shortlist": args.random_shortlist,
        "teacher_aggregation": args.teacher_aggregation,
        "refit_objective": args.refit_objective,
        "exclude_refit_from_selection": args.exclude_refit_from_selection,
        "precompute_base_kernels": precompute_base_kernels,
        "base_kernel_precompute": base_kernel_precompute,
        "kernel_batch_size": args.kernel_batch_size,
        "alpha_target_batch_size": args.alpha_target_batch_size,
        "score_target_batch_size": args.score_target_batch_size,
        "noise_cache_dir": str(args.noise_cache_dir) if args.noise_cache_dir else None,
        "target_cache_dir": str(target_cache_dir) if target_cache_dir else None,
        "refit_score_workers": args.refit_score_workers,
        "refit_indices": base_indices.tolist(),
        "refit_train_indices": train_indices.tolist(),
        "refit_val_indices": val_indices.tolist(),
        "feature_method_spec": asdict(method),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2, default=str)

    checkpoint_runtime = build_proxy_runtime(
        method=method,
        selected_indices=selected_indices,
        raw_features_np=raw_features_np,
        model_names=model_names,
        encoding_params=encoding_params,
        var_noise_by_track=var_noise_by_track,
        metric=args.metric,
        device=device,
        pool_size=pool_size,
    )
    checkpoint_runtime.trace_rows = trace_rows
    checkpoint_runtime.scores_combined = [
        float(row["score_combined"]) for row in trace_rows if row.get("score_combined") is not None
    ]
    checkpoint_runtime.scores_per_track_history[args.track] = list(
        checkpoint_runtime.scores_combined
    )
    checkpoint_runtime.filter_records = filter_records
    save_runtime_progress(
        checkpoint_runtime,
        payload_root,
        raw_features_np,
        raw_shard_slices=raw_shard_slices,
        model_names=model_names,
        run_config=run_config,
        pool_records_by_index=pool_records_by_index,
    )

    start_total = time.monotonic()
    while len(selected_indices) < args.target_size:
        step_start = time.monotonic()
        step = len(selected_indices) - args.init_size + 1
        runtime = build_proxy_runtime(
            method=method,
            selected_indices=selected_indices,
            raw_features_np=raw_features_np,
            model_names=model_names,
            encoding_params=encoding_params,
            var_noise_by_track=var_noise_by_track,
            metric=args.metric,
            device=device,
            pool_size=pool_size,
        )
        if args.exclude_refit_from_selection:
            runtime.pool_mask[base_indices] = False
        if image_filter is not None:
            exclude_failed_indices(runtime, image_filter)
        proxy_scores = proxy_scores_for_pool(
            runtime=runtime,
            raw_features_np=raw_features_np,
            model_names=model_names,
            encoding_params=encoding_params,
            track=args.track,
            metric=args.metric,
            corr_type="correlation",
            device=device,
            batch_size=args.proxy_batch_size,
        )
        shortlist = topk_shortlist(
            proxy_scores=proxy_scores,
            pool_mask=runtime.pool_mask,
            top_k=args.top_k_proxy,
            random_k=args.random_shortlist,
            rng=np.random.default_rng(
                args.seed + stable_seed(args.method_id, "shortlist", step)
            ),
        )
        print(
            f"[refit-robust] step {step}: selected={len(selected_indices)}, "
            f"shortlist={len(shortlist)}, proxy_best={float(np.max(proxy_scores[shortlist])):.4f}",
            flush=True,
        )

        eval_indices_for_encoding = np.unique(
            np.concatenate([np.asarray(selected_indices, dtype=np.int64), shortlist])
        )
        eval_cache_dir = (
            args.noise_cache_dir / "eval_targets" / f"step_{step:03d}"
            if args.noise_cache_dir
            else None
        )
        encoded_eval_pool = encode_indices_modelwise_cached(
            raw_features_np=raw_features_np,
            model_names=model_names,
            indices=eval_indices_for_encoding,
            track=args.track,
            encoding_params=encoding_params,
            device=device,
            target_cols=target_cols,
            cache_dir=eval_cache_dir,
            cache_prefix=f"eval_step_{step:03d}",
        )
        encoded_pos = {int(idx): pos for pos, idx in enumerate(eval_indices_for_encoding)}

        best: dict[str, Any] | None = None
        score_seed = args.seed + stable_seed(args.method_id, step)
        try:
            scored_rows, backend_timing = score_shortlist_refit_robust_v2_fast(
                shortlist=shortlist,
                selected_indices=selected_indices,
                encoded_eval_pool=encoded_eval_pool,
                encoded_pos=encoded_pos,
                raw_features_np=raw_features_np,
                fit_context=fit_context,
                noise_states=noise_states,
                model_names=model_names,
                alphas=alphas,
                metric=args.metric,
                corr_type=args.corr_type,
                noise_mult=args.noise_mult,
                base_noise_ceiling=args.noise_ceiling,
                seed=score_seed,
                aggregate_teachers=args.teacher_aggregation,
                objective=args.refit_objective,
                workers=args.refit_score_workers,
                score_target_batch_size=args.score_target_batch_size,
            )
        finally:
            if eval_cache_dir is not None:
                del encoded_eval_pool
                shutil.rmtree(eval_cache_dir, ignore_errors=True)
        print(
            "[refit-robust] "
            f"step {step}: backend={backend_timing.get('backend')} "
            f"objective={args.refit_objective} "
            f"workers={backend_timing.get('workers', 1)} "
            f"cache={format_seconds(float(backend_timing.get('cache_seconds', 0.0)))} "
            f"warmup={format_seconds(float(backend_timing.get('warmup_seconds', 0.0)))} "
            f"score={format_seconds(float(backend_timing.get('score_seconds', 0.0)))} "
            f"min_delta={float(backend_timing.get('minimum_delta', np.nan)):.3e}",
            flush=True,
        )

        for rank, row in enumerate(scored_rows, start=1):
            candidate_idx = int(row["candidate_index"])
            row["iteration"] = step
            row["shortlist_rank"] = rank
            row["proxy_score"] = float(proxy_scores[int(candidate_idx)])
            candidate_rows.append(row)
        if not scored_rows:
            raise RuntimeError("Shortlist was empty")
        ranked_rows = sorted(
            scored_rows,
            key=lambda row: (float(row["score"]), float(row["score_tie_breaker"])),
            reverse=True,
        )
        filter_attempts = 0
        filter_selected_passed = None
        filter_selected_reason = None
        if image_filter is not None:
            before = len(image_filter.filter_records)
            ranked_indices = np.asarray(
                [int(row["candidate_index"]) for row in ranked_rows],
                dtype=np.int64,
            )
            ranked_scores = np.asarray(
                [float(row["score"]) for row in ranked_rows],
                dtype=np.float32,
            )
            selected_idx, _filter_score, filter_attempts = image_filter.select_first_valid(
                ranked_indices,
                ranked_scores,
                candidate_scores_per_track={
                    "refit_accuracy": np.asarray(
                        [float(row["score_recovery_accuracy"]) for row in ranked_rows],
                        dtype=np.float32,
                    ),
                    "refit_margin": np.asarray(
                        [float(row["score_margin"]) for row in ranked_rows],
                        dtype=np.float32,
                    ),
                    "proxy": np.asarray(
                        [float(row["proxy_score"]) for row in ranked_rows],
                        dtype=np.float32,
                    ),
                },
                phase="greedy",
                iteration=step,
            )
            new_filter_records = [
                {
                    **filter_record_to_dict(
                        record,
                        method_id=args.method_id,
                        pool_size=pool_size,
                    )
                }
                for record in image_filter.filter_records[before:]
            ]
            filter_records.extend(new_filter_records)
            selected_filter_record = next(
                (
                    record
                    for record in new_filter_records
                    if int(record["global_idx"]) == int(selected_idx)
                ),
                None,
            )
            if selected_filter_record is not None:
                filter_selected_passed = bool(selected_filter_record["passed"])
                filter_selected_reason = selected_filter_record["reason"]
            if not args.allow_filter_fallback and not bool(filter_selected_passed):
                raise RuntimeError(
                    "Image filter did not find a passing refit-robust candidate within "
                    f"{image_filter.config.max_attempts_per_iteration} attempts "
                    f"at iteration={step}. Increase --filter-max-attempts-per-iteration "
                    "or pass --allow-filter-fallback for diagnostic runs."
                )
            best_matches = [
                row for row in ranked_rows if int(row["candidate_index"]) == int(selected_idx)
            ]
            if not best_matches:
                raise RuntimeError(
                    f"Image filter returned idx={selected_idx}, which was not in the scored shortlist"
                )
            best = best_matches[0]
        else:
            best = ranked_rows[0]

        selected_indices.append(int(best["candidate_index"]))
        trace_row = {
            "iteration": step,
            "n_selected": len(selected_indices),
            "selected_index": int(best["candidate_index"]),
            "score_combined": float(best["score"]),
            "score_objective": args.refit_objective,
            "score_tie_breaker": float(best["score_tie_breaker"]),
            "score_refit_accuracy": float(best["score_recovery_accuracy"]),
            "score_refit_margin": float(best["score_margin"]),
            "score_refit_margin_mean": float(best["teacher_margin_mean"]),
            "score_refit_margin_min": float(best["teacher_margin_min"]),
            "teacher_self_score_mean": float(best["teacher_self_score_mean"]),
            "teacher_other_score_mean": float(best["teacher_other_score_mean"]),
            "recovery_accuracy": float(best["recovery_accuracy"]),
            "teacher_majority_recovery_accuracy": float(best["teacher_majority_recovery_accuracy"]),
            "proxy_score": float(best["proxy_score"]),
            "shortlist_size": int(len(shortlist)),
            "method_id": args.method_id,
            "within": args.refit_objective,
            "across": args.teacher_aggregation,
            "filter_attempts": int(filter_attempts),
            "filter_selected_passed": filter_selected_passed,
            "filter_selected_reason": filter_selected_reason,
            "elapsed_seconds": float(time.monotonic() - step_start),
        }
        trace_rows.append(trace_row)
        print(
            f"[refit-robust] step {step}: selected idx={trace_row['selected_index']} "
            f"score={trace_row['score_combined']:.4f} "
            f"acc={trace_row['score_refit_accuracy']:.3f} "
            f"margin={trace_row['score_refit_margin']:.4f} "
            f"proxy={trace_row['proxy_score']:.4f} "
            f"elapsed={format_seconds(trace_row['elapsed_seconds'])}",
            flush=True,
        )

        np.save(method_dir / "selected_indices.npy", np.asarray(selected_indices, dtype=np.int64))
        pd.DataFrame(trace_rows).to_csv(method_dir / "selection_trace.csv", index=False)
        pd.DataFrame(candidate_rows).to_csv(method_dir / "candidate_scores.csv", index=False)
        save_filter_records(method_dir, filter_records)

    final_runtime = build_proxy_runtime(
        method=method,
        selected_indices=selected_indices,
        raw_features_np=raw_features_np,
        model_names=model_names,
        encoding_params=encoding_params,
        var_noise_by_track=var_noise_by_track,
        metric=args.metric,
        device=device,
        pool_size=pool_size,
    )
    final_runtime.trace_rows = trace_rows
    final_runtime.scores_combined = [float(row["score_combined"]) for row in trace_rows]
    final_runtime.scores_per_track_history[args.track] = [
        float(row["score_combined"]) for row in trace_rows
    ]
    final_runtime.filter_records = filter_records
    save_runtime_progress(
        final_runtime,
        payload_root,
        raw_features_np,
        raw_shard_slices=raw_shard_slices,
        model_names=model_names,
        run_config=run_config,
        pool_records_by_index=pool_records_by_index,
    )
    pd.DataFrame(candidate_rows).to_csv(method_dir / "candidate_scores.csv", index=False)
    with (method_dir / "refit_robust_summary.json").open("w") as f:
        json.dump(
            {
                "selected_indices": selected_indices,
                "n_selected": len(selected_indices),
                "elapsed_seconds": time.monotonic() - start_total,
                "elapsed": format_seconds(time.monotonic() - start_total),
            },
            f,
            indent=2,
        )
    return method_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--method-id", default="sub01_eval_augmented_loo_refit_robust")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--env", default="raven")
    parser.add_argument("--model-set", default="sota")
    parser.add_argument("--pool-feature-dir", type=Path, default=None)
    parser.add_argument("--max-ram-gb", type=float, default=300.0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--track", default="sub-01")
    parser.add_argument("--encoding-roi-subset", default="hlvis")
    parser.add_argument("--unique-encodings", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--corr-type", choices=["spearman"], default="spearman")
    parser.add_argument("--target-size", type=int, default=6)
    parser.add_argument("--init-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refit-pool-size", type=int, default=200)
    parser.add_argument("--refit-val-size", type=int, default=40)
    parser.add_argument("--exclude-refit-from-selection", action="store_true", default=True)
    parser.add_argument("--allow-refit-selection-overlap", dest="exclude_refit_from_selection", action="store_false")
    parser.add_argument("--top-k-proxy", type=int, default=8)
    parser.add_argument("--random-shortlist", type=int, default=0)
    parser.add_argument("--proxy-batch-size", type=int, default=2048)
    parser.add_argument("--proxy-noise-calib-examples", type=int, default=1000)
    parser.add_argument("--proxy-noise-calib-repeats", type=int, default=100)
    parser.add_argument("--no-proxy-attenuation", action="store_true")
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--noise-mult", type=float, default=1.0)
    parser.add_argument("--noise-ceiling", type=float, default=0.46)
    parser.add_argument(
        "--fit-noise-calibration",
        choices=["response", "rdm_analytic", "rdm_empirical"],
        default="rdm_empirical",
    )
    parser.add_argument("--calibration-images", type=int, default=100)
    parser.add_argument("--calibration-noise-samples", type=int, default=2)
    parser.add_argument("--calibration-max-iter", type=int, default=8)
    parser.add_argument("--n-noise-samples", type=int, default=1)
    parser.add_argument(
        "--target-dim",
        type=int,
        default=0,
        help="Target-response dimensions for selection; <=0 uses the full encoding target.",
    )
    parser.add_argument("--teacher-aggregation", choices=["mean", "min"], default="mean")
    parser.add_argument(
        "--refit-objective",
        choices=["accuracy_margin", "margin"],
        default="accuracy_margin",
        help=(
            "accuracy_margin selects by recovery accuracy and uses the RDM "
            "margin as a tie-breaker. margin keeps the old margin-only objective."
        ),
    )
    parser.add_argument("--precompute-base-kernels", action="store_true", default=True)
    parser.add_argument(
        "--no-precompute-base-kernels",
        dest="precompute_base_kernels",
        action="store_false",
    )
    parser.add_argument("--kernel-batch-size", type=int, default=4096)
    parser.add_argument(
        "--alpha-target-batch-size",
        type=int,
        default=DEFAULT_ALPHA_TARGET_BATCH_SIZE,
        help="Target columns per chunk when selecting validation alphas.",
    )
    parser.add_argument(
        "--score-target-batch-size",
        type=int,
        default=DEFAULT_SCORE_TARGET_BATCH_SIZE,
        help="Target columns per chunk when scoring refit predictions.",
    )
    parser.add_argument(
        "--noise-cache-dir",
        type=Path,
        default=None,
        help="Directory for disk-backed noisy base target caches.",
    )
    parser.add_argument(
        "--refit-score-workers",
        type=int,
        default=1,
        help=(
            "Forked candidate workers for shortlist scoring. Set native BLAS "
            "threads to 1 when using more than one worker."
        ),
    )
    parser.add_argument(
        "--disable-image-filter",
        action="store_true",
        help="Disable final-run image filtering. Required for arbitrary .npz feature pools.",
    )
    parser.add_argument("--filter-min-resolution", type=int, default=1000)
    parser.add_argument("--filter-natural-prob-threshold", type=float, default=0.85)
    parser.add_argument("--filter-download-timeout", type=float, default=10.0)
    parser.add_argument("--filter-max-attempts-per-iteration", type=int, default=1000)
    parser.add_argument("--filter-parallel-batch-size", type=int, default=1)
    parser.add_argument("--filter-classifier-path", type=Path, default=None)
    parser.add_argument("--disable-filter-image-save", action="store_true")
    parser.add_argument(
        "--allow-filter-fallback",
        action="store_true",
        help=(
            "Allow selecting the top scored candidate when none passes within the "
            "configured filter attempt window. Off by default for final runs."
        ),
    )
    args = parser.parse_args()
    if args.kernel_batch_size <= 0:
        raise ValueError("--kernel-batch-size must be positive")
    if args.alpha_target_batch_size <= 0:
        raise ValueError("--alpha-target-batch-size must be positive")
    if args.score_target_batch_size <= 0:
        raise ValueError("--score-target-batch-size must be positive")
    if args.refit_score_workers <= 0:
        raise ValueError("--refit-score-workers must be positive")
    if args.proxy_noise_calib_examples <= 0:
        raise ValueError("--proxy-noise-calib-examples must be positive")
    if args.proxy_noise_calib_repeats <= 0:
        raise ValueError("--proxy-noise-calib-repeats must be positive")
    if njit is None:
        raise RuntimeError("refit-robust selection requires numba")
    if args.metric != "cosine" or args.corr_type != "spearman":
        raise ValueError("refit-robust selection currently supports only cosine/Spearman")
    if args.output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = (
            SCRIPT.parents[1]
            / "results"
            / f"{args.model_set}_refit_robust_{stamp}"
        )
    return args


def main() -> None:
    args = parse_args()
    method_dir = run_selection(args)
    print(f"Done. Payload: {method_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
