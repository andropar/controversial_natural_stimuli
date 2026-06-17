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
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
SRC_DIR = ROOT / "src"
for path in (SRC_DIR, SCRIPT.parent):
    sys.path.insert(0, str(path))

from cstims.encoding.linear import encode_batch_for_all_encodings  # noqa: E402
from cstims.evaluation.noise_calibration import (  # noqa: E402
    calibrate_response_noise_for_rdm_reliability,
    multiplier_to_noise_ceiling,
    response_noise_std_from_mode,
)
from cstims.evaluation.ridge import (  # noqa: E402
    ridge_eval_augmented_loo_ops,
    ridge_ops_for_eval_sets,
    standardize_from_train,
)
from cstims.evaluation.teacher_student.independent_refit_rdm_recovery import (  # noqa: E402
    detect_equivalent_models,
    select_targetwise_alpha_indices,
    stable_seed,
)
from cstims.rdm_cuda import calculate_correlation_value, get_rdm_vector_np  # noqa: E402

from feature_method_sweep import (  # noqa: E402
    MethodRuntime,
    MethodSpec,
    TrackSpec,
    build_runtime,
    compute_track_scores,
    get_track_candidate_features,
    load_encoding_params_for_sweep,
    load_env_paths,
    load_model_set,
    load_npz_pool_features,
    save_manifest,
    save_runtime_progress,
)


DEFAULT_SOURCE_RUN = (
    ROOT
    / "00_stimulus_selection"
    / "feature_method_sweep"
    / "results"
    / "sota_pool100k_seed42_raw_sub01_rawenc_w05_meanmin_attenuation_20260615_153656"
)
MODEL_LIST_CSV = ROOT / "00_stimulus_selection" / "resources" / "model_list.csv"


@dataclass
class StudentOps:
    model: str
    x_train_raw: np.ndarray
    x_train: np.ndarray
    x_val: np.ndarray
    x_base: np.ndarray
    val_ops: dict[float, tuple[np.ndarray, dict[str, np.ndarray]]]


@dataclass
class TeacherTargets:
    model: str
    y_train_clean: np.ndarray
    y_val_clean: np.ndarray
    y_base_clean: np.ndarray
    response_noise_std: float
    achieved_fit_rdm_reliability: float


@dataclass
class TeacherNoiseState:
    y_base_fit: np.ndarray
    alpha_choices: dict[str, tuple[list[float], np.ndarray]]


@dataclass
class FitContext:
    student_ops: dict[str, StudentOps]
    teacher_targets: dict[str, TeacherTargets]
    equivalence_labels: list[int]
    target_cols: np.ndarray | None
    train_indices: np.ndarray
    val_indices: np.ndarray
    base_indices: np.ndarray


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def load_source_payload(source_run: Path, method_id: str) -> dict[str, Any]:
    path = source_run / "payloads" / method_id / "selected_stimuli_data.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Source payload not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


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
            "eval-set-augmented LOO teacher/student RDM margin."
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
    encoded_one = encode_indices(
        raw_features_np=raw_features_np,
        model_names=model_names,
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


def standardize_y_from_train(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_train_std, y_val_std, y_base_std = standardize_from_train(
        y_train,
        y_val,
        y_base,
    )
    return y_train_std, y_val_std, y_base_std


def build_fit_context(
    *,
    raw_features_np: dict[str, np.ndarray],
    encoded_refit: dict[str, np.ndarray],
    model_names: list[str],
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
) -> FitContext:
    student_ops: dict[str, StudentOps] = {}
    for model in model_names:
        x_train, x_val, x_base = standardize_from_train(
            np.asarray(raw_features_np[model][train_indices], dtype=np.float32),
            np.asarray(raw_features_np[model][val_indices], dtype=np.float32),
            np.asarray(raw_features_np[model][base_indices], dtype=np.float32),
            scale_by_sqrt_features=True,
        )
        val_ops = ridge_ops_for_eval_sets(x_train, x_val, {}, alphas)
        student_ops[model] = StudentOps(
            model=model,
            x_train_raw=np.asarray(raw_features_np[model][train_indices], dtype=np.float32),
            x_train=x_train,
            x_val=x_val,
            x_base=x_base,
            val_ops=val_ops,
        )

    teacher_targets: dict[str, TeacherTargets] = {}
    for model in model_names:
        y_base_raw = encoded_refit[model]
        y_train_raw = y_base_raw[refit_train_pos]
        y_val_raw = y_base_raw[refit_val_pos]
        y_train_clean, y_val_clean, y_base_clean = standardize_y_from_train(
            y_train_raw,
            y_val_raw,
            y_base_raw,
        )
        achieved = np.nan
        if fit_noise_calibration == "rdm_empirical":
            y_calib = y_train_clean
            if 0 < calibration_images < len(y_train_clean):
                calib_rng = np.random.default_rng(
                    seed + stable_seed(model, "selector_noise_calibration")
                )
                calib_idx = np.sort(
                    calib_rng.choice(len(y_train_clean), size=calibration_images, replace=False)
                )
                y_calib = y_train_clean[calib_idx]
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
            y_train_clean=y_train_clean,
            y_val_clean=y_val_clean,
            y_base_clean=y_base_clean,
            response_noise_std=float(response_noise_std),
            achieved_fit_rdm_reliability=float(achieved),
        )

    equivalence_labels = detect_equivalent_models(
        {model: np.asarray(raw_features_np[model][base_indices], dtype=np.float32) for model in model_names},
        model_names,
    )
    return FitContext(
        student_ops=student_ops,
        teacher_targets=teacher_targets,
        equivalence_labels=equivalence_labels,
        target_cols=None,
        train_indices=train_indices,
        val_indices=val_indices,
        base_indices=base_indices,
    )


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


def build_noise_states(
    *,
    fit_context: FitContext,
    model_names: list[str],
    seed: int,
    alphas: list[float],
    n_noise_samples: int,
) -> dict[str, list[TeacherNoiseState]]:
    states: dict[str, list[TeacherNoiseState]] = {}
    for teacher in model_names:
        target = fit_context.teacher_targets[teacher]
        teacher_states: list[TeacherNoiseState] = []
        for noise_sample_idx in range(n_noise_samples):
            rng = np.random.default_rng(
                seed + stable_seed("refit_robust_refit_noise", teacher, noise_sample_idx)
            )
            std = target.response_noise_std
            y_train = target.y_train_clean + rng.normal(
                0.0,
                std,
                target.y_train_clean.shape,
            ).astype(np.float32)
            y_val = target.y_val_clean + rng.normal(
                0.0,
                std,
                target.y_val_clean.shape,
            ).astype(np.float32)
            y_base_fit = target.y_base_clean + rng.normal(
                0.0,
                std,
                target.y_base_clean.shape,
            ).astype(np.float32)
            alpha_choices = {}
            for student in model_names:
                alpha_values, best_alpha_idx, _coeff_cache = select_targetwise_alpha_indices(
                    fit_context.student_ops[student].val_ops,
                    y_train,
                    y_val,
                )
                alpha_choices[student] = (alpha_values, best_alpha_idx)
            teacher_states.append(
                TeacherNoiseState(
                    y_base_fit=y_base_fit,
                    alpha_choices=alpha_choices,
                )
            )
        states[teacher] = teacher_states
    return states


def standardize_eval_x(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    scale_by_sqrt_features: bool = True,
) -> np.ndarray:
    train = np.asarray(x_train, dtype=np.float32)
    eval_arr = np.asarray(x_eval, dtype=np.float32)
    mean = train.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    if scale_by_sqrt_features:
        scale *= np.float32(math.sqrt(train.shape[1]))
    return np.asarray((eval_arr - mean) / scale, dtype=np.float32)


def predict_eval_augmented(
    *,
    alpha_values: list[float],
    best_alpha_idx: np.ndarray,
    loo_by_alpha: dict[float, tuple[np.ndarray, np.ndarray]],
    y_base_fit: np.ndarray,
    eval_y_fit: np.ndarray,
) -> np.ndarray:
    n_eval = eval_y_fit.shape[0]
    n_targets = eval_y_fit.shape[1]
    pred = np.empty((n_eval, n_targets), dtype=np.float32)
    for alpha_idx, alpha in enumerate(alpha_values):
        cols = np.flatnonzero(best_alpha_idx == alpha_idx)
        if cols.size == 0:
            continue
        base_op, eval_op = loo_by_alpha[float(alpha)]
        pred[:, cols] = base_op @ y_base_fit[:, cols] + eval_op @ eval_y_fit[:, cols]
    return pred


def score_candidate_refit_robust(
    *,
    candidate_idx: int,
    selected_indices: list[int],
    encoded_eval_by_model: dict[str, np.ndarray],
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
) -> dict[str, Any]:
    eval_indices = np.asarray([*selected_indices, int(candidate_idx)], dtype=np.int64)
    n_eval = len(eval_indices)
    eval_key = "eval"
    eval_raw_by_student = {
        model: np.asarray(raw_features_np[model][eval_indices], dtype=np.float32)
        for model in model_names
    }
    loo_ops_by_student: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    for student in model_names:
        ops = fit_context.student_ops[student]
        x_eval = standardize_eval_x(ops.x_train_raw, eval_raw_by_student[student])
        loo = ridge_eval_augmented_loo_ops(ops.x_base, {eval_key: x_eval}, alphas)
        loo_ops_by_student[student] = {
            float(alpha): loo[float(alpha)][eval_key] for alpha in alphas
        }

    teacher_utilities: list[float] = []
    teacher_self_scores: list[float] = []
    teacher_other_scores: list[float] = []
    recovered_correct: list[bool] = []
    noise_ceiling = multiplier_to_noise_ceiling(noise_mult, base_noise_ceiling)

    for teacher_idx, teacher in enumerate(model_names):
        teacher_target = fit_context.teacher_targets[teacher]
        teacher_equiv_label = int(fit_context.equivalence_labels[teacher_idx])
        off_equiv = np.asarray(
            [label != teacher_equiv_label for label in fit_context.equivalence_labels],
            dtype=bool,
        )
        teacher_sample_utilities = []
        teacher_sample_self_scores = []
        teacher_sample_other_scores = []
        teacher_sample_correct = []
        eval_y_clean = encoded_eval_by_model[teacher]
        for noise_sample_idx, noise_state in enumerate(noise_states[teacher]):
            std = teacher_target.response_noise_std
            eval_y_fit = eval_y_clean + response_noise_rows(
                image_indices=eval_indices,
                n_targets=eval_y_clean.shape[1],
                std=std,
                seed=seed,
                parts=("eval_fit_noise", teacher, noise_mult, noise_sample_idx),
            )
            y_eval_noisy = eval_y_clean + response_noise_rows(
                image_indices=eval_indices,
                n_targets=eval_y_clean.shape[1],
                std=std,
                seed=seed,
                parts=("eval_score_noise", teacher, noise_mult, noise_sample_idx),
            )
            noisy_teacher_rdm = get_rdm_vector_np(y_eval_noisy, metric)

            scores = np.full(len(model_names), np.nan, dtype=np.float32)
            for student_idx, student in enumerate(model_names):
                alpha_values, best_alpha_idx = noise_state.alpha_choices[student]
                pred = predict_eval_augmented(
                    alpha_values=alpha_values,
                    best_alpha_idx=best_alpha_idx,
                    loo_by_alpha=loo_ops_by_student[student],
                    y_base_fit=noise_state.y_base_fit,
                    eval_y_fit=eval_y_fit,
                )
                pred_rdm = get_rdm_vector_np(pred, metric)
                scores[student_idx] = calculate_correlation_value(
                    pred_rdm,
                    noisy_teacher_rdm,
                    corr_type,
                )

            scores = np.nan_to_num(scores, nan=-np.inf)
            self_score = float(scores[teacher_idx])
            competitor_scores = scores[off_equiv]
            other_score = float(np.max(competitor_scores)) if len(competitor_scores) else float("nan")
            utility = self_score - other_score
            recovered_idx = int(np.argmax(scores))
            recovered_correct_label = (
                int(fit_context.equivalence_labels[recovered_idx]) == teacher_equiv_label
            )
            teacher_sample_self_scores.append(self_score)
            teacher_sample_other_scores.append(other_score)
            teacher_sample_utilities.append(float(utility))
            teacher_sample_correct.append(bool(recovered_correct_label))
        teacher_utilities.append(float(np.mean(teacher_sample_utilities)))
        teacher_self_scores.append(float(np.mean(teacher_sample_self_scores)))
        teacher_other_scores.append(float(np.mean(teacher_sample_other_scores)))
        recovered_correct.append(bool(np.mean(teacher_sample_correct) >= 0.5))

    if aggregate_teachers == "mean":
        score = float(np.mean(teacher_utilities))
    elif aggregate_teachers == "min":
        score = float(np.min(teacher_utilities))
    else:
        raise ValueError(f"Unsupported teacher aggregation: {aggregate_teachers}")

    return {
        "candidate_index": int(candidate_idx),
        "n_eval": int(n_eval),
        "score": score,
        "teacher_margin_mean": float(np.mean(teacher_utilities)),
        "teacher_margin_min": float(np.min(teacher_utilities)),
        "teacher_self_score_mean": float(np.mean(teacher_self_scores)),
        "teacher_other_score_mean": float(np.mean(teacher_other_scores)),
        "recovery_accuracy": float(np.mean(recovered_correct)),
        "noise_mult": float(noise_mult),
        "noise_ceiling": float(noise_ceiling),
    }


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


def run_selection(args: argparse.Namespace) -> Path:
    source_run = args.source_run.resolve()
    source_config = load_json(source_run / "run_config.json")
    paths = load_env_paths(args.env)
    model_set_name, configured_model_names = load_model_set(args.model_set)
    model_names = list(source_config.get("model_names") or configured_model_names)
    if model_names != configured_model_names:
        print(
            "Using model_names from source run because they differ from model-set config",
            flush=True,
        )

    pool_feature_dir = Path(args.pool_feature_dir or source_config["pool_feature_dir"]).resolve()
    max_images = int(args.max_images or source_config.get("candidate_pool_size") or source_config["max_images"])
    raw_features_np, pool_records_by_index, pool_info = load_npz_pool_features(
        pool_feature_dir=pool_feature_dir,
        model_names=model_names,
        max_images=max_images,
    )
    pool_size = int(next(iter(raw_features_np.values())).shape[0])

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    encoding_params = load_encoding_params_for_sweep(
        paths=paths,
        model_list_csv=MODEL_LIST_CSV,
        encoding_names=[args.track],
        device=device,
        roi_subset=args.encoding_roi_subset,
        shared_encodings=not args.unique_encodings,
    )
    if args.track not in encoding_params:
        raise RuntimeError(f"Missing encoding params for {args.track}")

    source_payload = load_source_payload(source_run, args.initial_from_method)
    initial_indices = [
        int(x)
        for x in np.asarray(source_payload["selected_global_indices"], dtype=np.int64)[
            : args.init_size
        ]
    ]
    var_noise_by_track = {
        args.track: dict(
            (source_payload.get("var_noise_by_model") or {}).get(
                args.track,
                {model: 0.0 for model in model_names},
            )
        )
    }
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
    encoded_refit = encode_indices(
        raw_features_np=raw_features_np,
        model_names=model_names,
        indices=base_indices,
        track=args.track,
        encoding_params=encoding_params,
        device=device,
        target_cols=target_cols,
    )
    alphas = parse_csv_floats(args.alphas)
    fit_context = build_fit_context(
        raw_features_np=raw_features_np,
        encoded_refit=encoded_refit,
        model_names=model_names,
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
    )
    fit_context.target_cols = target_cols
    print("Precomputing noisy refit targets and target-wise alpha choices", flush=True)
    noise_states = build_noise_states(
        fit_context=fit_context,
        model_names=model_names,
        seed=args.seed + stable_seed(args.method_id),
        alphas=alphas,
        n_noise_samples=args.n_noise_samples,
    )

    output_root = args.output_root.resolve()
    payload_root = output_root / "payloads"
    method = make_method(args.method_id, args.track)
    method_dir = payload_root / args.method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    save_manifest([method], payload_root)

    selected_indices = list(initial_indices)
    trace_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed + stable_seed(args.method_id, "shortlist"))

    run_config = {
        **source_config,
        "method_name": args.method_id,
        "model_set_name": model_set_name,
        "model_names": model_names,
        "feature_method_sweep": True,
        "refit_robust_selection": True,
        "source_run": str(source_run),
        "initial_from_method": args.initial_from_method,
        "target_size": args.target_size,
        "init_size": args.init_size,
        "seed": args.seed,
        "metric": args.metric,
        "corr_type": args.corr_type,
        "track": args.track,
        "candidate_pool_size": pool_size,
        "pool_feature_dir": str(pool_feature_dir),
        "pool_info": pool_info,
        "refit_pool_size": args.refit_pool_size,
        "refit_val_size": args.refit_val_size,
        "refit_train_n": args.refit_pool_size - args.refit_val_size,
        "noise_mult": args.noise_mult,
        "noise_ceiling_target": args.noise_ceiling,
        "fit_noise_calibration": args.fit_noise_calibration,
        "n_noise_samples": args.n_noise_samples,
        "alphas": alphas,
        "target_dim": len(target_cols) if target_cols is not None else None,
        "top_k_proxy": args.top_k_proxy,
        "random_shortlist": args.random_shortlist,
        "teacher_aggregation": args.teacher_aggregation,
        "exclude_refit_from_selection": args.exclude_refit_from_selection,
        "refit_indices": base_indices.tolist(),
        "refit_train_indices": train_indices.tolist(),
        "refit_val_indices": val_indices.tolist(),
        "feature_method_spec": asdict(method),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2, default=str)

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
            rng=rng,
        )
        print(
            f"[refit-robust] step {step}: selected={len(selected_indices)}, "
            f"shortlist={len(shortlist)}, proxy_best={float(np.max(proxy_scores[shortlist])):.4f}",
            flush=True,
        )

        eval_indices_for_encoding = np.unique(
            np.concatenate([np.asarray(selected_indices, dtype=np.int64), shortlist])
        )
        encoded_eval_pool = encode_indices(
            raw_features_np=raw_features_np,
            model_names=model_names,
            indices=eval_indices_for_encoding,
            track=args.track,
            encoding_params=encoding_params,
            device=device,
            target_cols=target_cols,
        )
        encoded_pos = {int(idx): pos for pos, idx in enumerate(eval_indices_for_encoding)}

        best: dict[str, Any] | None = None
        for rank, candidate_idx in enumerate(shortlist, start=1):
            eval_positions = [encoded_pos[int(idx)] for idx in [*selected_indices, int(candidate_idx)]]
            encoded_eval_by_model = {
                model: arr[eval_positions].astype(np.float32, copy=False)
                for model, arr in encoded_eval_pool.items()
            }
            row = score_candidate_refit_robust(
                candidate_idx=int(candidate_idx),
                selected_indices=selected_indices,
                encoded_eval_by_model=encoded_eval_by_model,
                raw_features_np=raw_features_np,
                fit_context=fit_context,
                noise_states=noise_states,
                model_names=model_names,
                alphas=alphas,
                metric=args.metric,
                corr_type=args.corr_type,
                noise_mult=args.noise_mult,
                base_noise_ceiling=args.noise_ceiling,
                seed=args.seed + stable_seed(args.method_id, step),
                aggregate_teachers=args.teacher_aggregation,
            )
            row["iteration"] = step
            row["shortlist_rank"] = rank
            row["proxy_score"] = float(proxy_scores[int(candidate_idx)])
            candidate_rows.append(row)
            if best is None or float(row["score"]) > float(best["score"]):
                best = row
        if best is None:
            raise RuntimeError("Shortlist was empty")

        selected_indices.append(int(best["candidate_index"]))
        trace_row = {
            "iteration": step,
            "n_selected": len(selected_indices),
            "selected_index": int(best["candidate_index"]),
            "score_combined": float(best["score"]),
            "score_refit_margin_mean": float(best["teacher_margin_mean"]),
            "score_refit_margin_min": float(best["teacher_margin_min"]),
            "teacher_self_score_mean": float(best["teacher_self_score_mean"]),
            "teacher_other_score_mean": float(best["teacher_other_score_mean"]),
            "recovery_accuracy": float(best["recovery_accuracy"]),
            "proxy_score": float(best["proxy_score"]),
            "shortlist_size": int(len(shortlist)),
            "method_id": args.method_id,
            "within": "eval_augmented_loo",
            "across": args.teacher_aggregation,
            "elapsed_seconds": float(time.monotonic() - step_start),
        }
        trace_rows.append(trace_row)
        print(
            f"[refit-robust] step {step}: selected idx={trace_row['selected_index']} "
            f"score={trace_row['score_combined']:.4f} "
            f"acc={trace_row['recovery_accuracy']:.3f} "
            f"proxy={trace_row['proxy_score']:.4f} "
            f"elapsed={format_seconds(trace_row['elapsed_seconds'])}",
            flush=True,
        )

        np.save(method_dir / "selected_indices.npy", np.asarray(selected_indices, dtype=np.int64))
        pd.DataFrame(trace_rows).to_csv(method_dir / "selection_trace.csv", index=False)
        pd.DataFrame(candidate_rows).to_csv(method_dir / "candidate_scores.csv", index=False)

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
    save_runtime_progress(
        final_runtime,
        payload_root,
        raw_features_np,
        raw_shard_slices=[],
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
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--method-id", default="sub01_eval_augmented_loo_refit_robust")
    parser.add_argument("--initial-from-method", default="sub01_only_mean_min")
    parser.add_argument("--env", default="raven")
    parser.add_argument("--model-set", default="sota")
    parser.add_argument("--pool-feature-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--track", default="sub-01")
    parser.add_argument("--encoding-roi-subset", default="hlvis")
    parser.add_argument("--unique-encodings", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--corr-type", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--target-size", type=int, default=6)
    parser.add_argument("--init-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refit-pool-size", type=int, default=200)
    parser.add_argument("--refit-val-size", type=int, default=40)
    parser.add_argument("--exclude-refit-from-selection", action="store_true", default=True)
    parser.add_argument("--allow-refit-selection-overlap", dest="exclude_refit_from_selection", action="store_false")
    parser.add_argument("--top-k-proxy", type=int, default=64)
    parser.add_argument("--random-shortlist", type=int, default=16)
    parser.add_argument("--proxy-batch-size", type=int, default=2048)
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
    parser.add_argument("--target-dim", type=int, default=256)
    parser.add_argument("--teacher-aggregation", choices=["mean", "min"], default="mean")
    args = parser.parse_args()
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
