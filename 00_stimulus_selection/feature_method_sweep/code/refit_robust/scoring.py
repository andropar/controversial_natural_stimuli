"""Candidate and shortlist scoring for refit-robust selection."""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from typing import Any

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration dependency
    njit = None

NUMBA_AVAILABLE = njit is not None

from cstims.evaluation.noise_calibration import (
    multiplier_to_noise_ceiling,
    multiplier_to_rdm_reliability,
)

from .cache import RoundCache, build_round_cache, materialize_candidate_ops
from .data import FitContext, TeacherNoiseState, response_noise_rows


if njit is not None:

    @njit(cache=False, nogil=True, fastmath=True)
    def _fast_response_rdm(response: np.ndarray) -> np.ndarray:
        """Compute a cosine-distance RDM vector for one response matrix."""
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
        """Compute ordinal RDM ranks for Spearman scoring of one response matrix."""
        rdm = _fast_response_rdm(response)
        order = np.argsort(rdm, kind="mergesort")
        ranks = np.empty(order.size, dtype=np.int64)
        for idx in range(order.size):
            ranks[order[idx]] = idx
        return ranks


else:  # pragma: no cover - exercised only when Numba is unavailable
    _fast_response_rdm = None
    _fast_response_ranks = None


_WORKER_ARGS: dict[str, Any] | None = None


def encoded_eval_for_candidate(
    *,
    encoded_eval_pool: dict[str, np.ndarray],
    encoded_pos: dict[int, int],
    selected_indices: list[int],
    candidate_idx: int,
) -> dict[str, np.ndarray]:
    """Gather encoded selected-plus-candidate eval rows from a shared eval cache."""
    eval_positions = [encoded_pos[int(idx)] for idx in [*selected_indices, int(candidate_idx)]]
    return {
        model: arr[eval_positions].astype(np.float32, copy=False)
        for model, arr in encoded_eval_pool.items()
    }


def prepare_eval_data(
    *,
    round_cache: RoundCache,
    encoded_eval_by_model: dict[str, np.ndarray],
    eval_indices: np.ndarray,
    fit_context: FitContext,
    noise_mult: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build noisy eval-fit and eval-score target tensors for each teacher/noise block."""
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
    """Gather base-fit target columns indexed in flattened block-target coordinates."""
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


def pair_index_arrays(n_eval: int) -> tuple[np.ndarray, np.ndarray]:
    """Return upper-triangle row and column index arrays for an eval set."""
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
    """Accumulate row norms and pairwise dots for chunked cosine-RDM scoring."""
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
    """Convert accumulated cosine-RDM statistics into ordinal ranks."""
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
    """Compute Spearman correlation from two ordinal rank vectors without tie correction."""
    n_pairs = int(ranks_a.size)
    if n_pairs < 2:
        return float("nan")
    diff = ranks_a.astype(np.int64, copy=False) - ranks_b.astype(np.int64, copy=False)
    sum_sq = float(np.sum(diff * diff, dtype=np.int64))
    denominator = float(n_pairs * (n_pairs * n_pairs - 1))
    return float(1.0 - 6.0 * sum_sq / denominator)


def score_candidate_refit_robust_chunked(
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
    round_cache: RoundCache,
    noise_states: dict[str, list[TeacherNoiseState]],
    score_target_batch_size: int,
) -> dict[str, Any]:
    """Score one candidate with chunked eval-augmented LOO refit recovery."""
    if metric != "cosine" or corr_type != "spearman":
        raise ValueError("Refit-robust chunked scoring supports only cosine/Spearman")

    candidate_pos = round_cache.candidate_pos[int(candidate_idx)]
    eval_indices = np.asarray([*selected_indices, int(candidate_idx)], dtype=np.int64)
    dense_ops_by_student: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for student in model_names:
        dense_ops = materialize_candidate_ops(round_cache.student_caches[student], candidate_pos)
        if dense_ops is None:
            raise RuntimeError(
                f"Numerically unstable candidate delta for candidate {candidate_idx}, "
                f"student {student}"
            )
        dense_ops_by_student[student] = dense_ops

    eval_fit, eval_score = prepare_eval_data(
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

    return aggregate_candidate_scores(
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
        score_backend="chunked",
    )


def aggregate_candidate_scores(
    *,
    scores: np.ndarray,
    candidate_idx: int,
    n_eval: int,
    fit_context: FitContext,
    model_names: list[str],
    round_cache: RoundCache,
    aggregate_teachers: str,
    noise_mult: float,
    base_noise_ceiling: float,
    objective: str,
    score_backend: str = "chunked",
) -> dict[str, Any]:
    """Aggregate teacher/student recovery scores into the candidate objective row."""
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
    if fit_context.fit_noise_calibration == "rdm_empirical":
        noise_ceiling = multiplier_to_rdm_reliability(
            noise_mult,
            base_noise_ceiling,
            fit_context.rdm_calibration_comparison,
        )
    else:
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
        "rdm_calibration_comparison": fit_context.rdm_calibration_comparison,
        "score_backend": score_backend,
    }


def _score_worker(candidate_idx: int) -> dict[str, Any]:
    """Multiprocessing worker entry point for scoring one shortlist candidate."""
    assert _WORKER_ARGS is not None
    kwargs = _WORKER_ARGS
    encoded_eval_by_model = encoded_eval_for_candidate(
        encoded_eval_pool=kwargs["encoded_eval_pool"],
        encoded_pos=kwargs["encoded_pos"],
        selected_indices=kwargs["selected_indices"],
        candidate_idx=int(candidate_idx),
    )
    return score_candidate_refit_robust_chunked(
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


def score_shortlist_refit_robust(
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
    """Build round caches and score every candidate in the current shortlist."""
    if njit is None:
        raise RuntimeError("chunked scoring requires numba")
    if metric != "cosine" or corr_type != "spearman":
        raise ValueError("Refit-robust selection currently supports only cosine/Spearman")

    timing: dict[str, Any] = {"backend": "chunked"}
    start = time.monotonic()
    round_cache = build_round_cache(
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
    _materialize_candidate_ops_numba(
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
        global _WORKER_ARGS
        _WORKER_ARGS = common_kwargs
        context = mp.get_context("fork")
        with context.Pool(workers) as pool:
            rows = pool.map(_score_worker, [int(x) for x in shortlist], chunksize=8)
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
                score_candidate_refit_robust_chunked(
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
