#!/usr/bin/env python3
"""Independent-refit teacher/student recovery scored in RDM space.

Readouts are fitted from candidate raw features to teacher response targets on
an independent natural-image pool, and held-out recovery is scored using the RSA
observation space used by the stimulus-selection analyses:

    corr(RDM(candidate predicted responses), noisy RDM(teacher responses)).

For the main response-noise/RDM-calibrated run, Gaussian response noise is added
both while fitting candidate readouts and while evaluating the held-out teacher
responses.  The response-noise scale is inferred per teacher by matching an
empirical RDM reliability target; empirical calibration can target either
clean-vs-noisy or noisy-vs-noisy RDM correlations.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cstims import paths
from cstims.encoding.linear import load_encoding_params_by_encoding
from cstims.evaluation.noise_calibration import (
    calibrate_response_noise_for_rdm_reliability,
    multiplier_to_noise_ceiling,
    multiplier_to_rdm_reliability,
    rdm_noise_std_from_clean,
    response_noise_std_from_mode,
)
from cstims.evaluation.ridge import (
    EvalAugmentedLooRidgeOps,
    EvalAugmentedNestedLooKernelOps,
    IndependentRidgeOps,
    build_eval_augmented_loo_ops,
    build_eval_augmented_nested_loo_ops,
    build_independent_ridge_ops,
    standardize_from_train,
)
from cstims.rdm import calculate_correlation_value, get_rdm_vector_np
from cstims.rdm import correlate_vector_batches, get_rdm_vector, prepare_correlation_reference_batch


DEFAULT_RESULTS = (
    paths.find_share_root()
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "final_stimuli_recovery"
    / "teacher_student"
    / "results"
)

EVAL_AUGMENTED_MODES = {"eval_augmented_loo", "eval_augmented_nested_loo"}
VALIDATION_ALPHA_SELECTION = "targetwise_validation_pearson"
NESTED_ALPHA_SELECTION = "targetwise_eval_nested_loo_pearson"


def pearson_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-target Pearson correlations across samples."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch for column correlations: {x.shape} vs {y.shape}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {x.ndim}D")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    out = np.full(x.shape[1], np.nan, dtype=np.float64)
    ok = denom > 0
    out[ok] = np.sum(x[:, ok] * y[:, ok], axis=0) / denom[ok]
    return out


def select_targetwise_alpha_indices(
    alpha_ops: IndependentRidgeOps
    | EvalAugmentedLooRidgeOps
    | dict[float, tuple[np.ndarray, dict[str, np.ndarray]]],
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[list[float], np.ndarray, dict[float, np.ndarray] | None]:
    """Choose one ridge alpha per target dimension using validation Pearson r."""
    compact_ops = alpha_ops
    if isinstance(alpha_ops, EvalAugmentedLooRidgeOps):
        compact_ops = alpha_ops.alpha_selector
    alpha_values = compact_ops.alpha_values if isinstance(compact_ops, IndependentRidgeOps) else list(compact_ops)
    if not alpha_values:
        raise ValueError("No alpha operators available")
    scores = np.empty((len(alpha_values), y_train.shape[1]), dtype=np.float64)
    coefficient_cache: dict[float, np.ndarray] | None = (
        {} if isinstance(alpha_ops, IndependentRidgeOps) else None
    )
    projected_y = (
        compact_ops.project_targets(y_train)
        if isinstance(compact_ops, IndependentRidgeOps)
        else None
    )
    if isinstance(compact_ops, IndependentRidgeOps):
        coeffs_by_alpha = []
        for alpha in alpha_values:
            coeff = compact_ops.coefficients_from_projected_targets(alpha, projected_y)
            if coefficient_cache is not None:
                coefficient_cache[float(alpha)] = coeff
            coeffs_by_alpha.append(coeff)
        n_targets = y_train.shape[1]
        pred_val_all = compact_ops.validation_prediction_from_coefficients(
            np.concatenate(coeffs_by_alpha, axis=1)
        )
        for alpha_idx, _alpha in enumerate(alpha_values):
            start = alpha_idx * n_targets
            stop = start + n_targets
            scores[alpha_idx] = pearson_columns(pred_val_all[:, start:stop], y_val)
    else:
        for alpha_idx, alpha in enumerate(alpha_values):
            val_op, _eval_ops = alpha_ops[alpha]
            pred_val = val_op @ y_train
            scores[alpha_idx] = pearson_columns(pred_val, y_val)
    scores = np.nan_to_num(scores, nan=-np.inf)
    return alpha_values, np.argmax(scores, axis=0).astype(np.int32), coefficient_cache


def predict_eval_with_targetwise_alphas(
    *,
    alpha_ops: IndependentRidgeOps
    | EvalAugmentedLooRidgeOps
    | EvalAugmentedNestedLooKernelOps
    | dict[float, tuple[np.ndarray, dict[str, np.ndarray]]],
    alpha_values: list[float],
    best_alpha_idx: np.ndarray,
    eval_key: str,
    y_train: np.ndarray,
    eval_refit_mode: str,
    y_base_fit: np.ndarray | None = None,
    eval_y_fit: np.ndarray | None = None,
    coefficient_cache: dict[float, np.ndarray] | None = None,
) -> np.ndarray:
    """Predict one eval set after per-target alpha selection."""
    if isinstance(alpha_ops, IndependentRidgeOps):
        if eval_refit_mode != "independent":
            raise ValueError(
                "IndependentRidgeOps only supports eval_refit_mode='independent'"
            )
        first_eval = alpha_ops.eval_projected[eval_key]
        n_eval = first_eval.shape[0]
        n_targets = y_train.shape[1]
    elif isinstance(alpha_ops, EvalAugmentedLooRidgeOps):
        if eval_refit_mode != "eval_augmented_loo":
            raise ValueError(
                "EvalAugmentedLooRidgeOps only supports "
                "eval_refit_mode='eval_augmented_loo'"
            )
        if y_base_fit is None or eval_y_fit is None:
            raise RuntimeError(
                "eval_augmented_loo prediction requires noisy base and eval targets"
            )
        first_eval = alpha_ops.eval_projected[eval_key]
        n_eval = first_eval.shape[0]
        n_targets = y_base_fit.shape[1]
    elif isinstance(alpha_ops, EvalAugmentedNestedLooKernelOps):
        raise ValueError(
            "Nested eval-augmented LOO prediction must use "
            "predict_eval_augmented_nested_loo"
        )
    else:
        first_eval_op = alpha_ops[alpha_values[0]][1][eval_key]
        if eval_refit_mode == "eval_augmented_loo":
            if y_base_fit is None or eval_y_fit is None:
                raise RuntimeError(
                    "eval_augmented_loo prediction requires noisy base and eval targets"
                )
            base_op, _heldin_eval_op = first_eval_op
            n_eval = base_op.shape[0]
            n_targets = y_base_fit.shape[1]
        elif eval_refit_mode == "independent":
            n_eval = first_eval_op.shape[0]
            n_targets = y_train.shape[1]
        else:
            raise ValueError(f"Unsupported eval_refit_mode: {eval_refit_mode}")

    pred = np.empty((n_eval, n_targets), dtype=np.float32)
    for alpha_idx, alpha in enumerate(alpha_values):
        cols = np.flatnonzero(best_alpha_idx == alpha_idx)
        if cols.size == 0:
            continue
        if isinstance(alpha_ops, IndependentRidgeOps):
            if coefficient_cache is not None and float(alpha) in coefficient_cache:
                coeff = coefficient_cache[float(alpha)][:, cols]
                pred[:, cols] = alpha_ops.eval_prediction_from_coefficients(
                    eval_key,
                    coeff,
                )
            else:
                pred[:, cols] = alpha_ops.eval_prediction(
                    alpha,
                    eval_key,
                    y_train[:, cols],
                )
        elif isinstance(alpha_ops, EvalAugmentedLooRidgeOps):
            if y_base_fit is None or eval_y_fit is None:
                raise RuntimeError("Missing augmented LOO fit targets")
            pred[:, cols] = alpha_ops.eval_prediction(
                alpha,
                eval_key,
                y_base_fit[:, cols],
                eval_y_fit[:, cols],
            )
        else:
            eval_op = alpha_ops[alpha][1][eval_key]
            if eval_refit_mode == "eval_augmented_loo":
                base_op, heldin_eval_op = eval_op
                pred[:, cols] = (
                    base_op @ y_base_fit[:, cols]
                    + heldin_eval_op @ eval_y_fit[:, cols]
                )
            else:
                pred[:, cols] = eval_op @ y_train[:, cols]
    return pred


def eval_keys_for_ops(
    alpha_ops: IndependentRidgeOps
    | EvalAugmentedLooRidgeOps
    | EvalAugmentedNestedLooKernelOps
    | dict[float, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    alpha_values: list[float],
) -> list[str]:
    if isinstance(
        alpha_ops,
        (IndependentRidgeOps, EvalAugmentedLooRidgeOps, EvalAugmentedNestedLooKernelOps),
    ):
        return alpha_ops.eval_keys
    return list(alpha_ops[alpha_values[0]][1])


def _require_equal_eval_lengths(lengths: tuple[int, ...]) -> int:
    unique = set(int(length) for length in lengths)
    if len(unique) != 1:
        raise ValueError(
            "Batched teacher/student scoring requires equal-sized eval sets; "
            f"got lengths={sorted(unique)}"
        )
    return int(lengths[0])


def _pack_eval_targets(
    eval_y_clean: dict[str, np.ndarray],
    eval_keys: tuple[str, ...],
) -> np.ndarray:
    return np.stack([eval_y_clean[key] for key in eval_keys], axis=0).astype(
        np.float32,
        copy=False,
    )


def _pearson_scores_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = pred.float() - pred.float().mean(dim=-2, keepdim=True)
    y = target.float() - target.float().mean(dim=-2, keepdim=True)
    numerator = torch.sum(x * y, dim=-2)
    denom = torch.sqrt(torch.sum(x * x, dim=-2) * torch.sum(y * y, dim=-2))
    return torch.nan_to_num(numerator / (denom + 1e-9), nan=-float("inf"))


def _select_independent_coefficients_gpu(
    ops: IndependentRidgeOps,
    y_train_batch: torch.Tensor,
    y_val_batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    train_projected = torch.as_tensor(
        ops.train_projected.T,
        dtype=torch.float32,
        device=device,
    )
    val_projected = torch.as_tensor(
        ops.val_projected,
        dtype=torch.float32,
        device=device,
    )
    eigvals = torch.as_tensor(ops.eigvals, dtype=torch.float32, device=device)
    alphas = torch.as_tensor(ops.alphas, dtype=torch.float32, device=device)

    projected_y = torch.matmul(train_projected.unsqueeze(0), y_train_batch)
    alpha_scores = []
    for alpha in alphas:
        coeff = projected_y / (eigvals[None, :, None] + alpha)
        pred_val = torch.matmul(val_projected.unsqueeze(0), coeff)
        alpha_scores.append(_pearson_scores_torch(pred_val, y_val_batch))
    scores = torch.stack(alpha_scores, dim=0)
    best_alpha_idx = torch.argmax(scores, dim=0)
    selected_alpha = alphas[best_alpha_idx]
    return projected_y / (eigvals[None, :, None] + selected_alpha[:, None, :])


def _select_eval_augmented_alphas_gpu(
    alpha_ops: EvalAugmentedLooRidgeOps
    | dict[float, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    y_train_batch: torch.Tensor,
    y_val_batch: torch.Tensor,
    device: torch.device,
) -> tuple[list[float], torch.Tensor]:
    if isinstance(alpha_ops, EvalAugmentedLooRidgeOps):
        selector = alpha_ops.alpha_selector
        train_projected = torch.as_tensor(
            selector.train_projected.T,
            dtype=torch.float32,
            device=device,
        )
        val_projected = torch.as_tensor(
            selector.val_projected,
            dtype=torch.float32,
            device=device,
        )
        eigvals = torch.as_tensor(selector.eigvals, dtype=torch.float32, device=device)
        alpha_values = selector.alpha_values
        projected_y = torch.matmul(train_projected.unsqueeze(0), y_train_batch)
        alpha_scores = []
        for alpha in alpha_values:
            coeff = projected_y / (eigvals[None, :, None] + float(alpha))
            pred_val = torch.matmul(val_projected.unsqueeze(0), coeff)
            alpha_scores.append(_pearson_scores_torch(pred_val, y_val_batch))
        scores = torch.stack(alpha_scores, dim=0)
        return alpha_values, torch.argmax(scores, dim=0)

    alpha_values = list(alpha_ops)
    alpha_scores = []
    for alpha in alpha_values:
        val_op, _eval_ops = alpha_ops[alpha]
        val_op_t = torch.as_tensor(val_op, dtype=torch.float32, device=device)
        pred_val = torch.matmul(val_op_t.unsqueeze(0), y_train_batch)
        alpha_scores.append(_pearson_scores_torch(pred_val, y_val_batch))
    scores = torch.stack(alpha_scores, dim=0)
    return alpha_values, torch.argmax(scores, dim=0)


def _predict_independent_gpu(
    ops: IndependentRidgeOps,
    coeff_selected: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if ops.eval_projected_all is None:
        raise ValueError("IndependentRidgeOps does not contain eval projections")
    eval_projected = torch.as_tensor(
        ops.eval_projected_all,
        dtype=torch.float32,
        device=device,
    )
    return torch.bmm(
        eval_projected.unsqueeze(0).expand(coeff_selected.shape[0], -1, -1),
        coeff_selected,
    )


def _predict_eval_augmented_loo_gpu(
    alpha_ops: EvalAugmentedLooRidgeOps
    | dict[float, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    alpha_values: list[float],
    best_alpha_idx: torch.Tensor,
    eval_keys: tuple[str, ...],
    y_base_fit_batch: torch.Tensor,
    eval_y_fit_batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(alpha_ops, EvalAugmentedLooRidgeOps):
        base_projected_t = torch.as_tensor(
            alpha_ops.base_projected.T,
            dtype=torch.float32,
            device=device,
        )
        eigvals = torch.as_tensor(
            alpha_ops.base_eigvals,
            dtype=torch.float32,
            device=device,
        )
        projected_y_base = torch.matmul(
            base_projected_t.unsqueeze(0),
            y_base_fit_batch,
        )
        coeff_by_alpha = {
            float(alpha): projected_y_base / (eigvals[None, :, None] + float(alpha))
            for alpha in alpha_values
        }

        pred_by_key = []
        for eval_idx, key in enumerate(eval_keys):
            eval_fit = eval_y_fit_batch[:, eval_idx]
            eval_projected = torch.as_tensor(
                alpha_ops.eval_projected[key],
                dtype=torch.float32,
                device=device,
            )
            pred_by_alpha = []
            for alpha in alpha_values:
                alpha = float(alpha)
                base_pred = torch.matmul(
                    eval_projected.unsqueeze(0),
                    coeff_by_alpha[alpha],
                )
                s_inv = torch.as_tensor(
                    alpha_ops.eval_s_inv[alpha][key],
                    dtype=torch.float32,
                    device=device,
                )
                loo_denom = torch.diagonal(s_inv)
                pred_by_alpha.append(
                    eval_fit
                    + torch.matmul(
                        s_inv.unsqueeze(0),
                        base_pred - eval_fit,
                    )
                    / loo_denom[None, :, None]
                )
            pred_stack = torch.stack(pred_by_alpha, dim=0)
            gather_idx = best_alpha_idx[:, None, :].expand(
                best_alpha_idx.shape[0],
                pred_stack.shape[2],
                best_alpha_idx.shape[1],
            )
            pred_by_key.append(
                torch.gather(pred_stack, dim=0, index=gather_idx.unsqueeze(0)).squeeze(0)
            )
        return torch.cat(pred_by_key, dim=1)

    pred_by_key = []
    for eval_idx, key in enumerate(eval_keys):
        pred_by_alpha = []
        eval_fit = eval_y_fit_batch[:, eval_idx]
        for alpha_idx, alpha in enumerate(alpha_values):
            del alpha_idx
            _val_op, eval_ops = alpha_ops[alpha]
            base_op, heldin_eval_op = eval_ops[key]
            base_op_t = torch.as_tensor(base_op, dtype=torch.float32, device=device)
            eval_op_t = torch.as_tensor(
                heldin_eval_op,
                dtype=torch.float32,
                device=device,
            )
            pred_by_alpha.append(
                torch.matmul(base_op_t.unsqueeze(0), y_base_fit_batch)
                + torch.matmul(eval_op_t.unsqueeze(0), eval_fit)
            )
        pred_stack = torch.stack(pred_by_alpha, dim=0)
        gather_idx = best_alpha_idx[:, None, :].expand(
            best_alpha_idx.shape[0],
            pred_stack.shape[2],
            best_alpha_idx.shape[1],
        )
        pred_by_key.append(
            torch.gather(pred_stack, dim=0, index=gather_idx.unsqueeze(0)).squeeze(0)
        )
    return torch.cat(pred_by_key, dim=1)


def _feature_base_predictions_by_alpha_gpu(
    alpha_ops: EvalAugmentedLooRidgeOps,
    alpha_values: list[float],
    eval_key: str,
    projected_y_base: torch.Tensor,
    eigvals: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    eval_projected = torch.as_tensor(
        alpha_ops.eval_projected[eval_key],
        dtype=torch.float32,
        device=device,
    )
    return [
        torch.matmul(
            eval_projected.unsqueeze(0),
            projected_y_base / (eigvals[None, :, None] + float(alpha)),
        )
        for alpha in alpha_values
    ]


def _kernel_base_predictions_by_alpha_gpu(
    alpha_ops: EvalAugmentedNestedLooKernelOps,
    alpha_values: list[float],
    eval_key: str,
    y_base_fit_batch: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    return [
        torch.matmul(
            torch.as_tensor(
                alpha_ops.eval_base_ops[float(alpha)][eval_key],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
            y_base_fit_batch,
        )
        for alpha in alpha_values
    ]


def _nested_eval_predictions_from_alpha_stack_gpu(
    *,
    alpha_values: list[float],
    eval_fit: torch.Tensor,
    base_pred_by_alpha: list[torch.Tensor],
    s_inv_by_alpha: list[torch.Tensor],
) -> torch.Tensor:
    n_eval_images = eval_fit.shape[1]
    if n_eval_images < 3:
        raise ValueError("Nested eval LOO alpha selection requires at least 3 eval images")

    final_pred_by_alpha = []
    score_by_alpha = []
    all_idx = torch.arange(n_eval_images, dtype=torch.long, device=eval_fit.device)
    for alpha_idx, _alpha in enumerate(alpha_values):
        s_inv = s_inv_by_alpha[alpha_idx]
        residual = torch.matmul(
            s_inv.unsqueeze(0),
            eval_fit - base_pred_by_alpha[alpha_idx],
        )
        diag = torch.diagonal(s_inv)
        final_pred_by_alpha.append(eval_fit - residual / diag[None, :, None])

        scores_for_outer = []
        for outer_idx in range(n_eval_images):
            inner_idx = all_idx[all_idx != outer_idx]
            a = diag[outer_idx]
            b = s_inv[outer_idx, inner_idx]
            c = diag[inner_idx]
            det = a * c - b * b
            correction = (
                -b[None, :, None] * residual[:, outer_idx : outer_idx + 1, :]
                + a * residual[:, inner_idx, :]
            ) / det[None, :, None]
            inner_pred = eval_fit[:, inner_idx, :] - correction
            scores_for_outer.append(
                _pearson_scores_torch(inner_pred, eval_fit[:, inner_idx, :])
            )
        score_by_alpha.append(torch.stack(scores_for_outer, dim=1))

    score_stack = torch.stack(score_by_alpha, dim=0)
    best_alpha_idx = torch.argmax(score_stack, dim=0)
    pred_stack = torch.stack(final_pred_by_alpha, dim=0)
    return torch.gather(pred_stack, dim=0, index=best_alpha_idx.unsqueeze(0)).squeeze(0)


def predict_eval_augmented_nested_loo(
    *,
    alpha_ops: EvalAugmentedLooRidgeOps | EvalAugmentedNestedLooKernelOps,
    eval_key: str,
    y_base_fit: np.ndarray,
    eval_y_fit: np.ndarray,
) -> np.ndarray:
    """Strict nested alpha selection for one eval set in the scalar path."""
    alpha_values = alpha_ops.alpha_values
    eval_y = np.asarray(eval_y_fit, dtype=np.float32)
    base_y = np.asarray(y_base_fit, dtype=np.float32)
    n_eval_images, n_targets = eval_y.shape
    if n_eval_images < 3:
        raise ValueError("Nested eval LOO alpha selection requires at least 3 eval images")

    if isinstance(alpha_ops, EvalAugmentedLooRidgeOps):
        projected_y_base = alpha_ops.base_projected.T @ base_y
    else:
        projected_y_base = None

    pred_by_alpha = []
    score_by_alpha = []
    all_idx = np.arange(n_eval_images)
    for alpha in alpha_values:
        alpha = float(alpha)
        if isinstance(alpha_ops, EvalAugmentedLooRidgeOps):
            denom = alpha_ops.base_eigvals.astype(np.float32, copy=False) + np.float32(alpha)
            coeff = projected_y_base / denom[:, None]
            base_pred = alpha_ops.eval_projected[eval_key] @ coeff
        else:
            base_pred = alpha_ops.base_prediction(alpha, eval_key, base_y)
        s_inv = alpha_ops.eval_s_inv[alpha][eval_key]
        residual = s_inv @ (eval_y - base_pred)
        diag = np.diag(s_inv).astype(np.float32, copy=False)
        pred_by_alpha.append(eval_y - residual / diag[:, None])

        scores_for_outer = np.empty((n_eval_images, n_targets), dtype=np.float32)
        for outer_idx in range(n_eval_images):
            inner_idx = all_idx[all_idx != outer_idx]
            a = diag[outer_idx]
            b = s_inv[outer_idx, inner_idx]
            c = diag[inner_idx]
            det = a * c - b * b
            correction = (
                -b[:, None] * residual[outer_idx : outer_idx + 1, :]
                + a * residual[inner_idx, :]
            ) / det[:, None]
            inner_pred = eval_y[inner_idx] - correction
            scores_for_outer[outer_idx] = pearson_columns(
                inner_pred,
                eval_y[inner_idx],
            )
        score_by_alpha.append(scores_for_outer)

    score_stack = np.nan_to_num(np.stack(score_by_alpha, axis=0), nan=-np.inf)
    best_alpha_idx = np.argmax(score_stack, axis=0)
    pred_stack = np.stack(pred_by_alpha, axis=0)
    row_idx = np.arange(n_eval_images)[:, None]
    col_idx = np.arange(n_targets)[None, :]
    return np.asarray(pred_stack[best_alpha_idx, row_idx, col_idx], dtype=np.float32)


def _score_prediction_batch_gpu(
    pred_all: torch.Tensor,
    *,
    n_eval_sets: int,
    n_eval_images: int,
    teacher_reference_batch: torch.Tensor,
    metric: str,
    corr_type: str,
    device: torch.device,
) -> np.ndarray:
    pred_rdm = get_rdm_vector(
        pred_all.reshape(
            pred_all.shape[0] * n_eval_sets,
            n_eval_images,
            pred_all.shape[2],
        ),
        metric=metric,
    )
    scores = correlate_vector_batches(
        pred_rdm,
        teacher_reference_batch,
        corr_type,
    )
    return scores.reshape(pred_all.shape[0], n_eval_sets).detach().cpu().numpy()


def _score_eval_augmented_loo_feature_gpu(
    alpha_ops: EvalAugmentedLooRidgeOps,
    alpha_values: list[float],
    best_alpha_idx: torch.Tensor,
    eval_keys: tuple[str, ...],
    y_base_fit_batch: torch.Tensor,
    eval_y_fit_batch: torch.Tensor,
    teacher_reference_batch: torch.Tensor,
    *,
    metric: str,
    corr_type: str,
    device: torch.device,
    eval_set_batch_size: int = 8,
) -> np.ndarray:
    """Predict and score feature-space augmented LOO eval sets in GPU chunks."""
    n_noise_samples = y_base_fit_batch.shape[0]
    n_eval_sets = len(eval_keys)
    n_eval_images = eval_y_fit_batch.shape[2]
    base_projected_t = torch.as_tensor(
        alpha_ops.base_projected.T,
        dtype=torch.float32,
        device=device,
    )
    eigvals = torch.as_tensor(
        alpha_ops.base_eigvals,
        dtype=torch.float32,
        device=device,
    )
    projected_y_base = torch.matmul(
        base_projected_t.unsqueeze(0),
        y_base_fit_batch,
    )

    out = np.empty((n_noise_samples, n_eval_sets), dtype=np.float32)
    sample_offsets = torch.arange(
        n_noise_samples,
        dtype=torch.long,
        device=device,
    )[:, None] * n_eval_sets
    for start in range(0, n_eval_sets, eval_set_batch_size):
        stop = min(start + eval_set_batch_size, n_eval_sets)
        pred_by_key = []
        for eval_idx in range(start, stop):
            key = eval_keys[eval_idx]
            eval_fit = eval_y_fit_batch[:, eval_idx]
            eval_projected = torch.as_tensor(
                alpha_ops.eval_projected[key],
                dtype=torch.float32,
                device=device,
            )
            pred_selected = torch.empty_like(eval_fit)
            for alpha_idx, alpha in enumerate(alpha_values):
                selected = best_alpha_idx == alpha_idx
                if not torch.any(selected):
                    continue
                alpha = float(alpha)
                coeff = projected_y_base / (eigvals[None, :, None] + alpha)
                base_pred = torch.matmul(
                    eval_projected.unsqueeze(0),
                    coeff,
                )
                s_inv = torch.as_tensor(
                    alpha_ops.eval_s_inv[alpha][key],
                    dtype=torch.float32,
                    device=device,
                )
                loo_denom = torch.diagonal(s_inv)
                pred_alpha = (
                    eval_fit
                    + torch.matmul(
                        s_inv.unsqueeze(0),
                        base_pred - eval_fit,
                    )
                    / loo_denom[None, :, None]
                )
                selected_3d = selected[:, None, :].expand_as(pred_selected)
                pred_selected[selected_3d] = pred_alpha[selected_3d]
                del coeff, base_pred, pred_alpha
            pred_by_key.append(pred_selected)

        pred_all = torch.cat(pred_by_key, dim=1)
        eval_offsets = torch.arange(start, stop, dtype=torch.long, device=device)[None, :]
        reference_idx = (sample_offsets + eval_offsets).reshape(-1)
        reference_chunk = teacher_reference_batch[reference_idx]
        chunk_scores = _score_prediction_batch_gpu(
            pred_all,
            n_eval_sets=stop - start,
            n_eval_images=n_eval_images,
            teacher_reference_batch=reference_chunk,
            metric=metric,
            corr_type=corr_type,
            device=device,
        )
        out[:, start:stop] = chunk_scores
        del pred_all, pred_by_key
    return out


def _score_eval_augmented_nested_loo_gpu(
    alpha_ops: EvalAugmentedLooRidgeOps | EvalAugmentedNestedLooKernelOps,
    eval_keys: tuple[str, ...],
    y_base_fit_batch: torch.Tensor,
    eval_y_fit_batch: torch.Tensor,
    teacher_reference_batch: torch.Tensor,
    *,
    metric: str,
    corr_type: str,
    device: torch.device,
    eval_set_batch_size: int = 4,
) -> np.ndarray:
    """Predict and score strict nested eval-augmented LOO eval sets."""
    n_noise_samples = y_base_fit_batch.shape[0]
    n_eval_sets = len(eval_keys)
    n_eval_images = eval_y_fit_batch.shape[2]
    alpha_values = alpha_ops.alpha_values

    projected_y_base = None
    eigvals = None
    if isinstance(alpha_ops, EvalAugmentedLooRidgeOps):
        base_projected_t = torch.as_tensor(
            alpha_ops.base_projected.T,
            dtype=torch.float32,
            device=device,
        )
        eigvals = torch.as_tensor(
            alpha_ops.base_eigvals,
            dtype=torch.float32,
            device=device,
        )
        projected_y_base = torch.matmul(
            base_projected_t.unsqueeze(0),
            y_base_fit_batch,
        )

    out = np.empty((n_noise_samples, n_eval_sets), dtype=np.float32)
    sample_offsets = torch.arange(
        n_noise_samples,
        dtype=torch.long,
        device=device,
    )[:, None] * n_eval_sets
    for start in range(0, n_eval_sets, eval_set_batch_size):
        stop = min(start + eval_set_batch_size, n_eval_sets)
        pred_by_key = []
        for eval_idx in range(start, stop):
            key = eval_keys[eval_idx]
            eval_fit = eval_y_fit_batch[:, eval_idx]
            if isinstance(alpha_ops, EvalAugmentedLooRidgeOps):
                if projected_y_base is None or eigvals is None:
                    raise RuntimeError("Missing feature-space projected targets")
                base_pred_by_alpha = _feature_base_predictions_by_alpha_gpu(
                    alpha_ops,
                    alpha_values,
                    key,
                    projected_y_base,
                    eigvals,
                    device,
                )
            else:
                base_pred_by_alpha = _kernel_base_predictions_by_alpha_gpu(
                    alpha_ops,
                    alpha_values,
                    key,
                    y_base_fit_batch,
                    device,
                )
            s_inv_by_alpha = [
                torch.as_tensor(
                    alpha_ops.eval_s_inv[float(alpha)][key],
                    dtype=torch.float32,
                    device=device,
                )
                for alpha in alpha_values
            ]
            pred_by_key.append(
                _nested_eval_predictions_from_alpha_stack_gpu(
                    alpha_values=alpha_values,
                    eval_fit=eval_fit,
                    base_pred_by_alpha=base_pred_by_alpha,
                    s_inv_by_alpha=s_inv_by_alpha,
                )
            )

        pred_all = torch.cat(pred_by_key, dim=1)
        eval_offsets = torch.arange(start, stop, dtype=torch.long, device=device)[None, :]
        reference_idx = (sample_offsets + eval_offsets).reshape(-1)
        reference_chunk = teacher_reference_batch[reference_idx]
        chunk_scores = _score_prediction_batch_gpu(
            pred_all,
            n_eval_sets=stop - start,
            n_eval_images=n_eval_images,
            teacher_reference_batch=reference_chunk,
            metric=metric,
            corr_type=corr_type,
            device=device,
        )
        out[:, start:stop] = chunk_scores
        del pred_all, pred_by_key
    return out


def _rows_from_score_tensor(
    *,
    scores_by_sample_eval: np.ndarray,
    rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    model_set: str,
    track: dict[str, Any],
    model_names: list[str],
    equivalence_labels: list[int],
    teacher_idx: int,
    teacher: str,
    eval_keys: tuple[str, ...],
    eval_meta: dict[str, tuple[str, int]],
    off_equiv: np.ndarray,
    teacher_equiv_label: int,
    n_equiv_classes: int,
    noise_mult: float,
    noise_ceiling: float,
    refit_repeat_idx: int,
    refit_train_n: int,
    refit_val_n: int,
    eval_noise_mode: str,
    fit_noise_calibration: str,
    rdm_calibration_comparison: str,
    eval_refit_mode: str,
    response_noise_std: float,
    achieved_fit_rdm_reliability: float,
    target_dim: int,
    metric: str,
    corr_type: str,
    batch_noise_samples: bool,
    rdm_device: torch.device,
    gpu_alpha_batch: bool,
    gpu_predict_batch: bool,
    gpu_eval_noise_batch: bool,
) -> None:
    for noise_sample_idx in range(scores_by_sample_eval.shape[0]):
        for eval_idx, key in enumerate(eval_keys):
            scores = np.nan_to_num(
                scores_by_sample_eval[noise_sample_idx, eval_idx],
                nan=-np.inf,
            )
            recovered_idx = int(np.argmax(scores))
            subset_type, subset_idx = eval_meta[key]
            competitor_scores = scores[off_equiv]
            margins = scores[teacher_idx] - competitor_scores
            teacher_margin = float(np.min(margins)) if len(margins) else float("nan")
            recovered_equiv_label = int(equivalence_labels[recovered_idx])
            exact_recovered_correct = bool(recovered_idx == teacher_idx)
            recovered_correct = bool(recovered_equiv_label == teacher_equiv_label)
            row = {
                "model_set": model_set,
                "track": track["name"],
                "track_type": track.get("type", "identity"),
                "metric": metric,
                "corr_type": corr_type,
                "subset_type": subset_type,
                "subset_idx": subset_idx,
                "teacher_model": teacher,
                "recovered_model": model_names[recovered_idx],
                "recovered_correct": recovered_correct,
                "exact_recovered_correct": exact_recovered_correct,
                "teacher_equivalence_label": teacher_equiv_label,
                "recovered_equivalence_label": recovered_equiv_label,
                "best_test_score": float(scores[recovered_idx]),
                "teacher_self_test_score": float(scores[teacher_idx]),
                "teacher_margin": teacher_margin,
                "noise_mult": noise_mult,
                "noise_ceiling": noise_ceiling,
                "relative_snr": np.inf if noise_mult <= 0 else 1.0 / noise_mult,
                "noise_sample_idx": int(noise_sample_idx),
                "refit_repeat_idx": int(refit_repeat_idx),
                "refit_pool_size": int(refit_train_n + refit_val_n),
                "refit_train_n": int(refit_train_n),
                "refit_val_n": int(refit_val_n),
                "eval_noise_mode": eval_noise_mode,
                "fit_noise_calibration": fit_noise_calibration,
                "rdm_calibration_comparison": rdm_calibration_comparison,
                "eval_refit_mode": eval_refit_mode,
                "alpha_selection": (
                    NESTED_ALPHA_SELECTION
                    if eval_refit_mode == "eval_augmented_nested_loo"
                    else VALIDATION_ALPHA_SELECTION
                ),
                "response_noise_std": float(response_noise_std),
                "achieved_fit_rdm_reliability": float(achieved_fit_rdm_reliability),
                "n_equivalence_classes": int(n_equiv_classes),
                "target_dim": int(target_dim),
                "batch_noise_samples": bool(batch_noise_samples),
                "rdm_device": str(rdm_device),
                "gpu_alpha_batch": bool(gpu_alpha_batch),
                "gpu_predict_batch": bool(gpu_predict_batch),
                "gpu_eval_noise_batch": bool(gpu_eval_noise_batch),
            }
            rows.append(row)
            teacher_rows.append(row)


def _run_batched_response_noise_level(
    *,
    model_set: str,
    track: dict[str, Any],
    refit_repeat_idx: int,
    eval_y_clean: dict[str, np.ndarray],
    y_train_clean: np.ndarray,
    y_val_clean: np.ndarray,
    y_base_fit_clean: np.ndarray | None,
    eval_refit_mode: str,
    eval_keys: tuple[str, ...],
    eval_meta: dict[str, tuple[str, int]],
    candidate_ops: dict[
        str,
        IndependentRidgeOps
        | EvalAugmentedLooRidgeOps
        | EvalAugmentedNestedLooKernelOps
        | dict[float, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    ],
    model_names: list[str],
    equivalence_labels: list[int],
    teacher_idx: int,
    teacher: str,
    off_equiv: np.ndarray,
    teacher_equiv_label: int,
    n_equiv_classes: int,
    noise_mult: float,
    noise_ceiling: float,
    n_noise_samples: int,
    refit_train_n: int,
    refit_val_n: int,
    response_noise_std: float,
    achieved_fit_rdm_reliability: float,
    metric: str,
    corr_type: str,
    fit_noise_calibration: str,
    rdm_calibration_comparison: str,
    teacher_rng: np.random.Generator,
    rdm_device: torch.device,
    gpu_alpha_batch: bool,
    gpu_predict_batch: bool,
    gpu_eval_noise_batch: bool,
    seed: int,
    rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
) -> None:
    if not gpu_alpha_batch or not gpu_predict_batch:
        raise ValueError("Batched production path currently requires GPU alpha and prediction")
    if rdm_device.type != "cuda":
        raise ValueError("Batched production path requires a CUDA RDM device")

    eval_y_clean_all = _pack_eval_targets(eval_y_clean, eval_keys)
    n_eval_sets, n_eval_images, target_dim = eval_y_clean_all.shape
    train_noise = teacher_rng.standard_normal(
        (n_noise_samples, *y_train_clean.shape),
        dtype=np.float32,
    )
    train_noise *= np.float32(response_noise_std)
    val_noise = teacher_rng.standard_normal(
        (n_noise_samples, *y_val_clean.shape),
        dtype=np.float32,
    )
    val_noise *= np.float32(response_noise_std)
    y_train_batch = y_train_clean[None, :, :] + train_noise
    y_val_batch = y_val_clean[None, :, :] + val_noise
    y_train_t = torch.as_tensor(y_train_batch, dtype=torch.float32, device=rdm_device)
    y_val_t = torch.as_tensor(y_val_batch, dtype=torch.float32, device=rdm_device)

    torch_gen = torch.Generator(device=rdm_device)
    torch_gen.manual_seed(
        seed + stable_seed(track["name"], teacher, noise_mult, "torch_eval_noise")
    )
    eval_clean_t = torch.as_tensor(
        eval_y_clean_all,
        dtype=torch.float32,
        device=rdm_device,
    )
    if eval_refit_mode in EVAL_AUGMENTED_MODES:
        if y_base_fit_clean is None:
            raise RuntimeError("Missing base fit targets for eval-augmented refit")
        if (
            y_base_fit_clean.shape[0]
            == y_train_clean.shape[0] + y_val_clean.shape[0]
        ):
            y_base_fit_t = torch.cat([y_train_t, y_val_t], dim=1)
        else:
            base_noise = teacher_rng.standard_normal(
                (n_noise_samples, *y_base_fit_clean.shape),
                dtype=np.float32,
            )
            base_noise *= np.float32(response_noise_std)
            y_base_fit_t = torch.as_tensor(
                y_base_fit_clean[None, :, :] + base_noise,
                dtype=torch.float32,
                device=rdm_device,
            )
        if gpu_eval_noise_batch:
            eval_fit_noise = torch.randn(
                (n_noise_samples, *eval_clean_t.shape),
                generator=torch_gen,
                dtype=torch.float32,
                device=rdm_device,
            )
            eval_fit_noise *= float(response_noise_std)
            eval_y_fit_t = eval_clean_t.unsqueeze(0) + eval_fit_noise
        else:
            eval_fit_noise_np = teacher_rng.standard_normal(
                (n_noise_samples, *eval_y_clean_all.shape),
                dtype=np.float32,
            )
            eval_fit_noise_np *= np.float32(response_noise_std)
            eval_y_fit_t = torch.as_tensor(
                eval_y_clean_all[None, :, :, :] + eval_fit_noise_np,
                dtype=torch.float32,
                device=rdm_device,
            )
    else:
        y_base_fit_t = None
        eval_y_fit_t = None

    if gpu_eval_noise_batch:
        eval_score_noise = torch.randn(
            (n_noise_samples, *eval_clean_t.shape),
            generator=torch_gen,
            dtype=torch.float32,
            device=rdm_device,
        )
        eval_score_noise *= float(response_noise_std)
        y_eval_noisy_t = eval_clean_t.unsqueeze(0) + eval_score_noise
        teacher_rdm = get_rdm_vector(
            y_eval_noisy_t.reshape(
                n_noise_samples * n_eval_sets,
                n_eval_images,
                target_dim,
            ),
            metric=metric,
        )
    else:
        eval_score_noise_np = teacher_rng.standard_normal(
            (n_noise_samples, *eval_y_clean_all.shape),
            dtype=np.float32,
        )
        eval_score_noise_np *= np.float32(response_noise_std)
        y_eval_noisy = eval_y_clean_all[None, :, :, :] + eval_score_noise_np
        teacher_rdm = get_rdm_vector(
            torch.as_tensor(
                y_eval_noisy.reshape(
                    n_noise_samples * n_eval_sets,
                    n_eval_images,
                    target_dim,
                ),
                dtype=torch.float32,
                device=rdm_device,
            ),
            metric=metric,
        )
    teacher_reference = prepare_correlation_reference_batch(teacher_rdm, corr_type)
    scores_by_sample_eval = np.full(
        (n_noise_samples, n_eval_sets, len(model_names)),
        np.nan,
        dtype=np.float32,
    )
    for candidate_idx, candidate in enumerate(model_names):
        ops = candidate_ops[candidate]
        pred_all: torch.Tensor | None = None
        candidate_scores: np.ndarray | None = None
        if eval_refit_mode == "independent":
            if not isinstance(ops, IndependentRidgeOps):
                raise TypeError("independent batched path requires IndependentRidgeOps")
            if tuple(ops.eval_projected) != eval_keys:
                raise ValueError("Candidate eval projection order does not match teacher targets")
            n_eval_images_candidate = _require_equal_eval_lengths(ops.eval_lengths)
            if n_eval_images_candidate != n_eval_images:
                raise ValueError("Candidate eval-set size does not match teacher targets")
            coeff_selected = _select_independent_coefficients_gpu(
                ops,
                y_train_t,
                y_val_t,
                rdm_device,
            )
            pred_all = _predict_independent_gpu(ops, coeff_selected, rdm_device)
        elif eval_refit_mode == "eval_augmented_loo":
            if isinstance(ops, IndependentRidgeOps):
                raise TypeError("eval_augmented_loo batched path requires LOO ops")
            alpha_values, best_alpha_idx = _select_eval_augmented_alphas_gpu(
                ops,
                y_train_t,
                y_val_t,
                rdm_device,
            )
            if y_base_fit_t is None or eval_y_fit_t is None:
                raise RuntimeError("Missing augmented eval fit targets")
            if isinstance(ops, EvalAugmentedLooRidgeOps):
                candidate_scores = _score_eval_augmented_loo_feature_gpu(
                    ops,
                    alpha_values,
                    best_alpha_idx,
                    eval_keys,
                    y_base_fit_t,
                    eval_y_fit_t,
                    teacher_reference,
                    metric=metric,
                    corr_type=corr_type,
                    device=rdm_device,
                )
            else:
                pred_all = _predict_eval_augmented_loo_gpu(
                    ops,
                    alpha_values,
                    best_alpha_idx,
                    eval_keys,
                    y_base_fit_t,
                    eval_y_fit_t,
                    rdm_device,
                )
        elif eval_refit_mode == "eval_augmented_nested_loo":
            if not isinstance(
                ops,
                (EvalAugmentedLooRidgeOps, EvalAugmentedNestedLooKernelOps),
            ):
                raise TypeError(
                    "eval_augmented_nested_loo batched path requires nested LOO ops"
                )
            if y_base_fit_t is None or eval_y_fit_t is None:
                raise RuntimeError("Missing augmented eval fit targets")
            candidate_scores = _score_eval_augmented_nested_loo_gpu(
                ops,
                eval_keys,
                y_base_fit_t,
                eval_y_fit_t,
                teacher_reference,
                metric=metric,
                corr_type=corr_type,
                device=rdm_device,
            )
        else:
            raise ValueError(f"Unsupported eval_refit_mode: {eval_refit_mode}")
        if candidate_scores is None:
            if pred_all is None:
                raise RuntimeError("Missing candidate predictions for scoring")
            candidate_scores = _score_prediction_batch_gpu(
                pred_all,
                n_eval_sets=n_eval_sets,
                n_eval_images=n_eval_images,
                teacher_reference_batch=teacher_reference,
                metric=metric,
                corr_type=corr_type,
                device=rdm_device,
            )
        scores_by_sample_eval[:, :, candidate_idx] = candidate_scores
        del pred_all

    _rows_from_score_tensor(
        scores_by_sample_eval=scores_by_sample_eval,
        rows=rows,
        teacher_rows=teacher_rows,
        model_set=model_set,
        track=track,
        model_names=model_names,
        equivalence_labels=equivalence_labels,
        teacher_idx=teacher_idx,
        teacher=teacher,
        eval_keys=eval_keys,
        eval_meta=eval_meta,
        off_equiv=off_equiv,
        teacher_equiv_label=teacher_equiv_label,
        n_equiv_classes=n_equiv_classes,
        noise_mult=noise_mult,
        noise_ceiling=noise_ceiling,
        refit_repeat_idx=refit_repeat_idx,
        refit_train_n=refit_train_n,
        refit_val_n=refit_val_n,
        eval_noise_mode="response",
        fit_noise_calibration=fit_noise_calibration,
        rdm_calibration_comparison=rdm_calibration_comparison,
        eval_refit_mode=eval_refit_mode,
        response_noise_std=response_noise_std,
        achieved_fit_rdm_reliability=achieved_fit_rdm_reliability,
        target_dim=target_dim,
        metric=metric,
        corr_type=corr_type,
        batch_noise_samples=True,
        rdm_device=rdm_device,
        gpu_alpha_batch=gpu_alpha_batch,
        gpu_predict_batch=gpu_predict_batch,
        gpu_eval_noise_batch=gpu_eval_noise_batch,
    )


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str | None) -> list[float]:
    return [float(item) for item in parse_csv_list(value)]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def parse_index_list(value: str | None, n_items: int) -> set[int] | None:
    if not value:
        return None
    out: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo_s, hi_s = item.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ValueError(f"Bad index range: {item}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(item))
    bad = sorted(idx for idx in out if idx < 0 or idx >= n_items)
    if bad:
        raise ValueError(f"Teacher indices out of range 0..{n_items - 1}: {bad}")
    return out


def build_eval_raw_and_meta(
    *,
    selected_raw: dict[str, np.ndarray],
    random_raw_union: dict[str, np.ndarray],
    random_subset_positions: list[np.ndarray],
    model_names: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, tuple[str, int]]]:
    eval_raw: dict[str, dict[str, np.ndarray]] = {"selected|0": selected_raw}
    eval_meta: dict[str, tuple[str, int]] = {"selected|0": ("selected", 0)}
    for subset_idx, pos in enumerate(random_subset_positions):
        key = f"random|{subset_idx}"
        eval_raw[key] = {model: random_raw_union[model][pos] for model in model_names}
        eval_meta[key] = ("random", subset_idx)
    return eval_raw, eval_meta


def load_encoding_params_for_models(
    encoding_root: Path,
    model_names: list[str],
    encoding_name: str,
    *,
    roi_subset: str | None,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    params_by_encoding = load_encoding_params_by_encoding(
        encoding_root=encoding_root,
        model_list_csv=paths.model_list_csv(),
        encoding_names=[encoding_name],
        device=device,
        roi_subset=roi_subset,
    )
    if encoding_name not in params_by_encoding:
        raise FileNotFoundError(
            f"No encoding parameters loaded for {encoding_name} under {encoding_root}"
        )
    params = params_by_encoding[encoding_name]
    missing = [model for model in model_names if model not in params]
    if missing:
        raise FileNotFoundError(
            f"Missing encoding parameters for {encoding_name}: {missing}"
        )
    return {model: params[model] for model in model_names}


def build_candidate_ops(
    *,
    random_raw_union: dict[str, np.ndarray],
    eval_raw: dict[str, dict[str, np.ndarray]],
    refit_positions: np.ndarray,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    base_fit_pos: np.ndarray | None,
    model_names: list[str],
    alphas: list[float],
    eval_refit_mode: str,
) -> dict[
    str,
    IndependentRidgeOps
    | EvalAugmentedLooRidgeOps
    | EvalAugmentedNestedLooKernelOps
    | dict[float, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
]:
    candidate_ops = {}
    for candidate_idx, candidate in enumerate(model_names):
        print(
            f"    ridge ops {candidate_idx + 1}/{len(model_names)}: {candidate}",
            flush=True,
        )
        x = random_raw_union[candidate]
        eval_x = {key: raw_by_model[candidate] for key, raw_by_model in eval_raw.items()}
        if eval_refit_mode == "independent":
            standardized = standardize_from_train(
                x[train_pos],
                x[val_pos],
                *eval_x.values(),
                scale_by_sqrt_features=True,
            )
            eval_x_std = dict(zip(eval_x.keys(), standardized[2:]))
            candidate_ops[candidate] = build_independent_ridge_ops(
                standardized[0],
                standardized[1],
                eval_x_std,
                alphas,
            )
            print(
                f"      backend={candidate_ops[candidate].backend}",
                flush=True,
            )
        elif eval_refit_mode == "eval_augmented_loo":
            if base_fit_pos is None:
                raise ValueError("base_fit_pos is required for eval_augmented_loo")
            standardized = standardize_from_train(
                x[train_pos],
                x[val_pos],
                x[base_fit_pos],
                *eval_x.values(),
                scale_by_sqrt_features=True,
            )
            x_train = standardized[0]
            x_val = standardized[1]
            x_base = standardized[2]
            eval_x_std = dict(zip(eval_x.keys(), standardized[3:]))
            candidate_ops[candidate] = build_eval_augmented_loo_ops(
                x_train,
                x_val,
                x_base,
                eval_x_std,
                alphas,
            )
            backend = (
                candidate_ops[candidate].backend
                if isinstance(candidate_ops[candidate], EvalAugmentedLooRidgeOps)
                else "kernel"
            )
            print(f"      eval_augmented_loo_backend={backend}", flush=True)
        elif eval_refit_mode == "eval_augmented_nested_loo":
            if base_fit_pos is None:
                raise ValueError("base_fit_pos is required for eval_augmented_nested_loo")
            standardized = standardize_from_train(
                x[train_pos],
                x[val_pos],
                x[base_fit_pos],
                *eval_x.values(),
                scale_by_sqrt_features=True,
            )
            x_train = standardized[0]
            x_val = standardized[1]
            x_base = standardized[2]
            eval_x_std = dict(zip(eval_x.keys(), standardized[3:]))
            candidate_ops[candidate] = build_eval_augmented_nested_loo_ops(
                x_train,
                x_val,
                x_base,
                eval_x_std,
                alphas,
            )
            print(
                f"      eval_augmented_nested_loo_backend={candidate_ops[candidate].backend}",
                flush=True,
            )
        else:
            raise ValueError(f"Unsupported eval_refit_mode: {eval_refit_mode}")
    return candidate_ops


def detect_equivalent_models(
    raw_features: dict[str, np.ndarray],
    model_names: list[str],
    *,
    max_rows: int = 512,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> list[int]:
    labels = list(range(len(model_names)))
    for i, mi in enumerate(model_names):
        xi = raw_features[mi][:max_rows]
        for j in range(i):
            mj = model_names[j]
            xj = raw_features[mj][:max_rows]
            if xi.shape != xj.shape:
                continue
            if np.allclose(xi, xj, atol=atol, rtol=rtol):
                labels[i] = labels[j]
                break
    return labels


def load_cached_teacher_rows(
    path: Path,
    *,
    teacher: str,
    track_name: str,
    expected_rows: int,
    required_values: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"    ignoring unreadable teacher cache {path}: {exc}", flush=True)
        return None
    if len(df) != expected_rows:
        print(
            f"    ignoring incomplete teacher cache {path.name}: "
            f"{len(df)}/{expected_rows} rows",
            flush=True,
        )
        return None
    if "teacher_model" not in df or "track" not in df:
        return None
    if set(df["teacher_model"].astype(str)) != {teacher}:
        return None
    if set(df["track"].astype(str)) != {track_name}:
        return None
    if required_values:
        for column, expected in required_values.items():
            if column not in df:
                return None
            values = set(df[column].astype(str))
            if values != {str(expected)}:
                return None
    return df.to_dict("records")


def write_teacher_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def _teacher_cache_path(
    teacher_cache_dir: Path | None,
    track_name: str,
    teacher_idx: int,
    teacher: str,
) -> Path | None:
    if teacher_cache_dir is None:
        return None
    return (
        teacher_cache_dir
        / f"{safe_name(track_name)}__{teacher_idx:03d}__{safe_name(teacher)}.csv"
    )


def summarize_rows(rows: list[dict[str, Any]], n_models: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "refit_repeat_idx" not in df:
        df["refit_repeat_idx"] = 0
    if "rdm_calibration_comparison" not in df:
        df["rdm_calibration_comparison"] = "clean_to_noisy"
    out = []
    keys = [
        "model_set",
        "track",
        "track_type",
        "noise_mult",
        "noise_ceiling",
        "rdm_calibration_comparison",
        "subset_type",
        "subset_idx",
        "refit_repeat_idx",
    ]
    for key, group in df.groupby(keys, sort=False, dropna=False):
        correct = group["recovered_correct"].astype(float).to_numpy()
        margins = group["teacher_margin"].astype(float).to_numpy()
        out.append(
            {
                "model_set": key[0],
                "recovery_orientation": "teacher_student_independent_refit_rdm_recovery",
                "track": key[1],
                "track_type": key[2],
                "metric": group["metric"].iloc[0],
                "corr_type": group["corr_type"].iloc[0],
                "noise_mult": float(key[3]),
                "relative_snr": np.inf if float(key[3]) <= 0 else 1.0 / float(key[3]),
                "noise_ceiling": float(key[4]),
                "rdm_calibration_comparison": key[5],
                "subset_type": key[6],
                "subset_idx": int(key[7]),
                "refit_repeat_idx": int(key[8]),
                "recovery_accuracy": float(correct.mean()),
                "error_prob": float(1.0 - correct.mean()),
                "mean_margin": float(np.mean(margins)),
                "n_units": int(len(correct)),
                "n_models": int(n_models),
                "n_equivalence_classes": int(group.get("n_equivalence_classes", pd.Series([n_models])).iloc[0]),
                "n_noise_samples": int(group["noise_sample_idx"].nunique()),
                "refit_pool_size": int(group["refit_pool_size"].iloc[0]),
                "refit_train_n": int(group["refit_train_n"].iloc[0]),
                "refit_val_n": int(group["refit_val_n"].iloc[0]),
                "eval_noise_mode": group["eval_noise_mode"].iloc[0],
                "fit_noise_calibration": group["fit_noise_calibration"].iloc[0],
                "eval_refit_mode": group.get("eval_refit_mode", pd.Series(["independent"])).iloc[0],
                "n_refit_repeats": 1,
            }
        )
    summary = pd.DataFrame(out)
    agg_keys = [
        "model_set",
        "recovery_orientation",
        "track",
        "track_type",
        "metric",
        "corr_type",
        "noise_mult",
        "relative_snr",
        "noise_ceiling",
        "subset_type",
        "rdm_calibration_comparison",
    ]
    rows2 = []
    for key, group in summary.groupby(agg_keys, sort=False, dropna=False):
        weights = group["n_units"].to_numpy(float)
        acc = np.average(group["recovery_accuracy"], weights=weights)
        margin = np.average(group["mean_margin"], weights=weights)
        acc_vals = group["recovery_accuracy"].astype(float).to_numpy()
        margin_vals = group["mean_margin"].astype(float).to_numpy()
        if len(acc_vals) > 1:
            acc_sd = float(np.std(acc_vals, ddof=1))
            acc_sem = float(acc_sd / math.sqrt(len(acc_vals)))
        else:
            acc_sd = float("nan")
            n_units = float(group["n_units"].sum())
            acc_sem = float(math.sqrt(max(acc * (1.0 - acc), 0.0) / max(n_units, 1.0)))
        if len(margin_vals) > 1:
            margin_sd = float(np.std(margin_vals, ddof=1))
            margin_sem = float(margin_sd / math.sqrt(len(margin_vals)))
        else:
            margin_sd = float("nan")
            margin_sem = float("nan")
        rows2.append(
            dict(zip(agg_keys, key))
            | {
                "recovery_accuracy": float(acc),
                "recovery_accuracy_sd": acc_sd,
                "recovery_accuracy_sem": acc_sem,
                "error_prob": float(1.0 - acc),
                "mean_margin": float(margin),
                "mean_margin_sd": margin_sd,
                "mean_margin_sem": margin_sem,
                "n_units": int(group["n_units"].sum()),
                "n_subsets": int(group["subset_idx"].nunique()),
                "n_refit_repeats": int(group["refit_repeat_idx"].nunique()),
                "n_models": int(n_models),
                "n_equivalence_classes": int(group.get("n_equivalence_classes", pd.Series([n_models])).iloc[0]),
                "n_noise_samples": int(group["n_noise_samples"].max()),
                "refit_pool_size": int(group["refit_pool_size"].iloc[0]),
                "refit_train_n": int(group["refit_train_n"].iloc[0]),
                "refit_val_n": int(group["refit_val_n"].iloc[0]),
                "eval_noise_mode": group["eval_noise_mode"].iloc[0],
                "fit_noise_calibration": group["fit_noise_calibration"].iloc[0],
                "rdm_calibration_comparison": key[10],
                "eval_refit_mode": group.get("eval_refit_mode", pd.Series(["independent"])).iloc[0],
            }
        )
    return pd.DataFrame(rows2), summary, df


def load_existing_detail(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "teacher_student_recoveries.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read existing detail CSV for resume: {exc}", flush=True)
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    return df


def completed_tracks_from_detail(
    detail: pd.DataFrame,
    *,
    tracks: list[dict[str, Any]],
    n_models: int,
    n_noise_levels: int,
    n_noise_samples: int,
    n_eval_sets: int,
    n_refit_repeats: int = 1,
) -> set[str]:
    if detail.empty or "track" not in detail:
        return set()
    expected = n_models * n_noise_levels * n_noise_samples * n_eval_sets * n_refit_repeats
    completed: set[str] = set()
    for track in tracks:
        name = track["name"]
        track_rows = detail[detail["track"].astype(str) == name]
        if len(track_rows) >= expected:
            completed.add(name)
    return completed


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, Any]],
    n_models: int,
    *,
    write_detail: bool = True,
) -> None:
    agg, subset, detail = summarize_rows(rows, n_models)
    agg.to_csv(out_dir / "discriminability.csv", index=False)
    subset.to_csv(out_dir / "subset_discriminability.csv", index=False)
    if write_detail:
        detail.to_csv(out_dir / "teacher_student_recoveries.csv", index=False)


def merge_complete_cache_tracks(
    *,
    out_dir: Path,
    cache_dir: Path,
    tracks: list[dict[str, Any]],
    model_names: list[str],
    noise_mults: np.ndarray,
    n_noise_samples: int,
    n_eval_sets: int,
    refit_repeat_indices: list[int],
    write_detail: bool = False,
    required_cache_values: dict[str, Any] | None = None,
) -> set[str]:
    existing = load_existing_detail(out_dir)
    expected_track_rows = (
        len(model_names)
        * len(noise_mults)
        * n_noise_samples
        * n_eval_sets
        * len(refit_repeat_indices)
    )
    expected_teacher_rows = len(noise_mults) * n_noise_samples * n_eval_sets
    rows: list[dict[str, Any]] = []
    complete_tracks: set[str] = set()
    existing_complete = completed_tracks_from_detail(
        existing,
        tracks=tracks,
        n_models=len(model_names),
        n_noise_levels=len(noise_mults),
        n_noise_samples=n_noise_samples,
        n_eval_sets=n_eval_sets,
        n_refit_repeats=len(refit_repeat_indices),
    )

    model_cache_dir = cache_dir
    for track in tracks:
        track_name = track["name"]
        track_cache_rows: list[dict[str, Any]] = []
        for refit_repeat_idx in refit_repeat_indices:
            repeat_cache_dir = model_cache_dir / f"refit_repeat_{refit_repeat_idx:03d}"
            if refit_repeat_idx == 0 and not repeat_cache_dir.exists():
                repeat_cache_dir = model_cache_dir
            for teacher_idx, teacher in enumerate(model_names):
                path = (
                    repeat_cache_dir
                    / f"{safe_name(track_name)}__{teacher_idx:03d}__{safe_name(teacher)}.csv"
                )
                cached = load_cached_teacher_rows(
                    path,
                    teacher=teacher,
                    track_name=track_name,
                    expected_rows=expected_teacher_rows,
                    required_values=required_cache_values,
                )
                if cached is None:
                    track_cache_rows = []
                    break
                for row in cached:
                    row.setdefault("refit_repeat_idx", int(refit_repeat_idx))
                track_cache_rows.extend(cached)
            if not track_cache_rows:
                break
        if len(track_cache_rows) == expected_track_rows:
            rows.extend(track_cache_rows)
            complete_tracks.add(track_name)
        elif track_name in existing_complete:
            rows.extend(existing[existing["track"].astype(str) == track_name].to_dict("records"))
            complete_tracks.add(track_name)

    if rows:
        write_outputs(out_dir, rows, len(model_names), write_detail=write_detail)
    return complete_tracks


def _run_single_teacher_rdm_recovery(
    *,
    model_set: str,
    track: dict[str, Any],
    refit_repeat_idx: int,
    eval_target: dict[str, dict[str, np.ndarray]],
    random_target_union: dict[str, np.ndarray],
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    base_fit_pos: np.ndarray | None,
    eval_meta: dict[str, tuple[str, int]],
    candidate_ops: dict[
        str,
        IndependentRidgeOps
        | EvalAugmentedLooRidgeOps
        | EvalAugmentedNestedLooKernelOps
        | dict[float, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    ],
    model_names: list[str],
    equivalence_labels: list[int],
    teacher_idx: int,
    noise_mults: np.ndarray,
    n_noise_samples: int,
    refit_train_n: int,
    refit_val_n: int,
    base_noise_ceiling: float,
    metric: str,
    corr_type: str,
    eval_noise_mode: str,
    fit_noise_calibration: str,
    rdm_calibration_comparison: str,
    eval_refit_mode: str,
    calibration_images: int,
    calibration_noise_samples: int,
    calibration_max_iter: int,
    target_dim: int | None,
    teacher_cache_dir: Path | None,
    expected_teacher_rows: int,
    seed: int,
    batch_noise_samples: bool = False,
    rdm_device: torch.device | None = None,
    gpu_alpha_batch: bool = False,
    gpu_predict_batch: bool = False,
    gpu_eval_noise_batch: bool = False,
) -> list[dict[str, Any]]:
    rdm_device = rdm_device or torch.device("cpu")
    teacher = model_names[teacher_idx]
    cache_path = _teacher_cache_path(
        teacher_cache_dir,
        track["name"],
        teacher_idx,
        teacher,
    )
    if cache_path is not None:
        required_cache_values = {
            "eval_refit_mode": eval_refit_mode,
            "fit_noise_calibration": fit_noise_calibration,
            "rdm_calibration_comparison": rdm_calibration_comparison,
        }
        if batch_noise_samples:
            required_cache_values.update(
                {
                    "batch_noise_samples": True,
                    "rdm_device": str(rdm_device),
                    "gpu_alpha_batch": bool(gpu_alpha_batch),
                    "gpu_predict_batch": bool(gpu_predict_batch),
                    "gpu_eval_noise_batch": bool(gpu_eval_noise_batch),
                }
            )
        cached_rows = load_cached_teacher_rows(
            cache_path,
            teacher=teacher,
            track_name=track["name"],
            expected_rows=expected_teacher_rows,
            required_values=required_cache_values,
        )
        if cached_rows is not None:
            print(
                f"    teacher {teacher_idx + 1}/{len(model_names)}: {teacher} "
                f"(cached)",
                flush=True,
            )
            return cached_rows

    rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    print(
        f"    teacher {teacher_idx + 1}/{len(model_names)}: {teacher}",
        flush=True,
    )
    teacher_rng = np.random.default_rng(
        seed + stable_seed(track["name"], teacher, "teacher_noise")
    )
    clean_y = random_target_union[teacher]
    eval_y = {key: target_by_model[teacher] for key, target_by_model in eval_target.items()}
    target_cols = None
    if target_dim is not None and 0 < target_dim < clean_y.shape[1]:
        target_rng = np.random.default_rng(
            seed + stable_seed(track["name"], teacher, "target_cols")
        )
        target_cols = np.sort(target_rng.choice(clean_y.shape[1], size=target_dim, replace=False))
    if target_cols is not None:
        clean_y = clean_y[:, target_cols]
        eval_y = {key: y[:, target_cols] for key, y in eval_y.items()}
    if eval_refit_mode in EVAL_AUGMENTED_MODES:
        if base_fit_pos is None:
            raise ValueError("base_fit_pos is required for eval-augmented refit")
        standardized_y = standardize_from_train(
            clean_y[train_pos],
            clean_y[val_pos],
            clean_y[base_fit_pos],
            *eval_y.values(),
        )
    elif eval_refit_mode == "independent":
        standardized_y = standardize_from_train(
            clean_y[train_pos],
            clean_y[val_pos],
            *eval_y.values(),
        )
    else:
        raise ValueError(f"Unsupported eval_refit_mode: {eval_refit_mode}")
    y_train_clean = standardized_y[0]
    y_val_clean = standardized_y[1]
    if eval_refit_mode in EVAL_AUGMENTED_MODES:
        y_base_fit_clean = standardized_y[2]
        eval_y_clean = dict(zip(eval_y.keys(), standardized_y[3:]))
    else:
        y_base_fit_clean = None
        eval_y_clean = dict(zip(eval_y.keys(), standardized_y[2:]))

    clean_eval_rdms = (
        {
            key: get_rdm_vector_np(y_clean, metric)
            for key, y_clean in eval_y_clean.items()
        }
        if eval_noise_mode == "rdm"
        else {}
    )
    teacher_equiv_label = int(equivalence_labels[teacher_idx])
    off_equiv = np.asarray(
        [label != teacher_equiv_label for label in equivalence_labels],
        dtype=bool,
    )
    n_equiv_classes = len(set(equivalence_labels))
    if calibration_images > 0 and calibration_images < y_train_clean.shape[0]:
        calib_rng = np.random.default_rng(
            seed + stable_seed(track["name"], teacher, "calibration_subset")
        )
        calib_idx = np.sort(
            calib_rng.choice(y_train_clean.shape[0], size=calibration_images, replace=False)
        )
        y_calib_clean = y_train_clean[calib_idx]
    else:
        y_calib_clean = y_train_clean

    for noise_mult in noise_mults:
        noise_mult = float(noise_mult)
        if fit_noise_calibration == "rdm_empirical":
            noise_ceiling = multiplier_to_rdm_reliability(
                noise_mult,
                base_noise_ceiling,
                rdm_calibration_comparison,
            )
        else:
            noise_ceiling = multiplier_to_noise_ceiling(
                noise_mult,
                base_noise_ceiling,
            )
        achieved_fit_rdm_reliability = np.nan
        if fit_noise_calibration == "rdm_empirical":
            cal_rng = np.random.default_rng(
                seed
                + stable_seed(
                    track["name"],
                    teacher,
                    noise_mult,
                    "rdm_empirical",
                    rdm_calibration_comparison,
                )
            )
            (
                response_noise_std,
                achieved_fit_rdm_reliability,
            ) = calibrate_response_noise_for_rdm_reliability(
                y_calib_clean,
                target_reliability=noise_ceiling,
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
        if batch_noise_samples and eval_noise_mode == "response":
            _run_batched_response_noise_level(
                model_set=model_set,
                track=track,
                refit_repeat_idx=refit_repeat_idx,
                eval_y_clean=eval_y_clean,
                y_train_clean=y_train_clean,
                y_val_clean=y_val_clean,
                y_base_fit_clean=y_base_fit_clean,
                eval_refit_mode=eval_refit_mode,
                eval_keys=tuple(eval_y_clean),
                eval_meta=eval_meta,
                candidate_ops=candidate_ops,
                model_names=model_names,
                equivalence_labels=equivalence_labels,
                teacher_idx=teacher_idx,
                teacher=teacher,
                off_equiv=off_equiv,
                teacher_equiv_label=teacher_equiv_label,
                n_equiv_classes=n_equiv_classes,
                noise_mult=noise_mult,
                noise_ceiling=noise_ceiling,
                n_noise_samples=n_noise_samples,
                refit_train_n=refit_train_n,
                refit_val_n=refit_val_n,
                response_noise_std=response_noise_std,
                achieved_fit_rdm_reliability=achieved_fit_rdm_reliability,
                metric=metric,
                corr_type=corr_type,
                fit_noise_calibration=fit_noise_calibration,
                rdm_calibration_comparison=rdm_calibration_comparison,
                teacher_rng=teacher_rng,
                rdm_device=rdm_device,
                gpu_alpha_batch=gpu_alpha_batch,
                gpu_predict_batch=gpu_predict_batch,
                gpu_eval_noise_batch=gpu_eval_noise_batch,
                seed=seed,
                rows=rows,
                teacher_rows=teacher_rows,
            )
            continue
        for noise_sample_idx in range(n_noise_samples):
            y_train = y_train_clean + teacher_rng.normal(
                0.0,
                response_noise_std,
                y_train_clean.shape,
            ).astype(np.float32)
            y_val = y_val_clean + teacher_rng.normal(
                0.0,
                response_noise_std,
                y_val_clean.shape,
            ).astype(np.float32)
            if eval_refit_mode in EVAL_AUGMENTED_MODES:
                if y_base_fit_clean is None:
                    raise RuntimeError("Missing base fit targets for eval-augmented refit")
                if (
                    y_base_fit_clean.shape[0]
                    == y_train_clean.shape[0] + y_val_clean.shape[0]
                ):
                    y_base_fit = np.concatenate([y_train, y_val], axis=0)
                else:
                    y_base_fit = y_base_fit_clean + teacher_rng.normal(
                        0.0,
                        response_noise_std,
                        y_base_fit_clean.shape,
                    ).astype(np.float32)
                eval_y_fit = {
                    key: y_clean
                    + teacher_rng.normal(
                        0.0,
                        response_noise_std,
                        y_clean.shape,
                    ).astype(np.float32)
                    for key, y_clean in eval_y_clean.items()
                }
            else:
                y_base_fit = None
                eval_y_fit = {}

            noisy_teacher_rdms: dict[str, np.ndarray] = {}
            if eval_noise_mode == "rdm":
                for key, clean_rdm in clean_eval_rdms.items():
                    std = rdm_noise_std_from_clean(
                        clean_rdm,
                        base_noise_ceiling,
                        noise_mult,
                    )
                    noisy_teacher_rdms[key] = clean_rdm + teacher_rng.normal(
                        0.0,
                        std,
                        clean_rdm.shape,
                    ).astype(np.float32)
            elif eval_noise_mode == "response":
                for key, y_clean in eval_y_clean.items():
                    y_eval_noisy = eval_y_clean[key] + teacher_rng.normal(
                        0.0,
                        response_noise_std,
                        y_clean.shape,
                    ).astype(np.float32)
                    noisy_teacher_rdms[key] = get_rdm_vector_np(y_eval_noisy, metric)
            else:
                raise ValueError(eval_noise_mode)

            scores_by_eval = {
                key: np.full(len(model_names), np.nan, dtype=np.float32)
                for key in eval_meta
            }
            for candidate_idx, candidate in enumerate(model_names):
                ops = candidate_ops[candidate]
                if eval_refit_mode == "eval_augmented_nested_loo":
                    if not isinstance(
                        ops,
                        (EvalAugmentedLooRidgeOps, EvalAugmentedNestedLooKernelOps),
                    ):
                        raise TypeError(
                            "eval_augmented_nested_loo scalar path requires nested LOO ops"
                        )
                    if y_base_fit is None:
                        raise RuntimeError("Missing augmented eval fit targets")
                    for key in ops.eval_keys:
                        pred = predict_eval_augmented_nested_loo(
                            alpha_ops=ops,
                            eval_key=key,
                            y_base_fit=y_base_fit,
                            eval_y_fit=eval_y_fit[key],
                        )
                        pred_rdm = get_rdm_vector_np(pred, metric)
                        scores_by_eval[key][candidate_idx] = calculate_correlation_value(
                            pred_rdm,
                            noisy_teacher_rdms[key],
                            corr_type,
                        )
                    continue

                (
                    alpha_values,
                    best_alpha_idx,
                    coefficient_cache,
                ) = select_targetwise_alpha_indices(
                    ops,
                    y_train,
                    y_val,
                )
                for key in eval_keys_for_ops(ops, alpha_values):
                    pred = predict_eval_with_targetwise_alphas(
                        alpha_ops=ops,
                        alpha_values=alpha_values,
                        best_alpha_idx=best_alpha_idx,
                        eval_key=key,
                        y_train=y_train,
                        eval_refit_mode=eval_refit_mode,
                        y_base_fit=y_base_fit,
                        eval_y_fit=eval_y_fit.get(key) if eval_y_fit else None,
                        coefficient_cache=coefficient_cache,
                    )
                    pred_rdm = get_rdm_vector_np(pred, metric)
                    scores_by_eval[key][candidate_idx] = calculate_correlation_value(
                        pred_rdm,
                        noisy_teacher_rdms[key],
                        corr_type,
                    )

            for key, scores in scores_by_eval.items():
                scores = np.nan_to_num(scores, nan=-np.inf)
                recovered_idx = int(np.argmax(scores))
                subset_type, subset_idx = eval_meta[key]
                competitor_scores = scores[off_equiv]
                margins = scores[teacher_idx] - competitor_scores
                teacher_margin = float(np.min(margins)) if len(margins) else float("nan")
                recovered_equiv_label = int(equivalence_labels[recovered_idx])
                exact_recovered_correct = bool(recovered_idx == teacher_idx)
                recovered_correct = bool(recovered_equiv_label == teacher_equiv_label)
                row = {
                    "model_set": model_set,
                    "track": track["name"],
                    "track_type": track.get("type", "identity"),
                    "metric": metric,
                    "corr_type": corr_type,
                    "subset_type": subset_type,
                    "subset_idx": subset_idx,
                    "teacher_model": teacher,
                    "recovered_model": model_names[recovered_idx],
                    "recovered_correct": recovered_correct,
                    "exact_recovered_correct": exact_recovered_correct,
                    "teacher_equivalence_label": teacher_equiv_label,
                    "recovered_equivalence_label": recovered_equiv_label,
                    "best_test_score": float(scores[recovered_idx]),
                    "teacher_self_test_score": float(scores[teacher_idx]),
                    "teacher_margin": teacher_margin,
                    "noise_mult": noise_mult,
                    "noise_ceiling": noise_ceiling,
                    "relative_snr": np.inf if noise_mult <= 0 else 1.0 / noise_mult,
                    "noise_sample_idx": noise_sample_idx,
                    "refit_repeat_idx": int(refit_repeat_idx),
                    "refit_pool_size": int(refit_train_n + refit_val_n),
                    "refit_train_n": int(refit_train_n),
                    "refit_val_n": int(refit_val_n),
                    "eval_noise_mode": eval_noise_mode,
                    "fit_noise_calibration": fit_noise_calibration,
                    "rdm_calibration_comparison": rdm_calibration_comparison,
                    "eval_refit_mode": eval_refit_mode,
                    "alpha_selection": (
                        NESTED_ALPHA_SELECTION
                        if eval_refit_mode == "eval_augmented_nested_loo"
                        else VALIDATION_ALPHA_SELECTION
                    ),
                    "response_noise_std": float(response_noise_std),
                    "achieved_fit_rdm_reliability": float(achieved_fit_rdm_reliability),
                    "n_equivalence_classes": int(n_equiv_classes),
                    "target_dim": int(clean_y.shape[1]),
                    "batch_noise_samples": False,
                    "rdm_device": str(rdm_device),
                    "gpu_alpha_batch": False,
                    "gpu_predict_batch": False,
                    "gpu_eval_noise_batch": False,
                }
                rows.append(row)
                teacher_rows.append(row)
    if cache_path is not None:
        write_teacher_cache(cache_path, teacher_rows)
    return rows


def run_track_rdm_recovery(
    *,
    model_set: str,
    track: dict[str, Any],
    refit_repeat_idx: int,
    selected_target: dict[str, np.ndarray],
    random_target_union: dict[str, np.ndarray],
    random_subset_positions: list[np.ndarray],
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    base_fit_pos: np.ndarray | None,
    eval_meta: dict[str, tuple[str, int]],
    candidate_ops: dict[
        str,
        IndependentRidgeOps
        | EvalAugmentedLooRidgeOps
        | EvalAugmentedNestedLooKernelOps
        | dict[float, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    ],
    model_names: list[str],
    equivalence_labels: list[int],
    alphas: list[float],
    noise_mults: np.ndarray,
    n_noise_samples: int,
    refit_train_n: int,
    refit_val_n: int,
    base_noise_ceiling: float,
    metric: str,
    corr_type: str,
    eval_noise_mode: str,
    fit_noise_calibration: str,
    rdm_calibration_comparison: str,
    eval_refit_mode: str,
    calibration_images: int,
    calibration_noise_samples: int,
    calibration_max_iter: int,
    target_dim: int | None,
    teacher_cache_dir: Path | None,
    teacher_indices: set[int] | None,
    seed: int,
    batch_noise_samples: bool = False,
    rdm_device: torch.device | None = None,
    gpu_alpha_batch: bool = False,
    gpu_predict_batch: bool = False,
    gpu_eval_noise_batch: bool = False,
    teacher_workers: int = 1,
) -> list[dict[str, Any]]:
    del alphas
    rdm_device = rdm_device or torch.device("cpu")
    if teacher_workers < 1:
        raise ValueError("teacher_workers must be >= 1")
    if teacher_workers > 1 and (
        batch_noise_samples
        or rdm_device.type != "cpu"
        or gpu_alpha_batch
        or gpu_predict_batch
        or gpu_eval_noise_batch
    ):
        raise ValueError(
            "teacher_workers > 1 is currently supported only for CPU scalar scoring. "
            "Use teacher_workers=1 for CUDA/batched evaluation."
        )

    eval_target: dict[str, dict[str, np.ndarray]] = {"selected|0": selected_target}
    for subset_idx, pos in enumerate(random_subset_positions):
        key = f"random|{subset_idx}"
        eval_target[key] = {model: random_target_union[model][pos] for model in model_names}

    selected_teacher_indices = [
        teacher_idx
        for teacher_idx in range(len(model_names))
        if teacher_indices is None or teacher_idx in teacher_indices
    ]
    expected_teacher_rows = len(noise_mults) * n_noise_samples * len(eval_meta)
    common_kwargs = dict(
        model_set=model_set,
        track=track,
        refit_repeat_idx=refit_repeat_idx,
        eval_target=eval_target,
        random_target_union=random_target_union,
        train_pos=train_pos,
        val_pos=val_pos,
        base_fit_pos=base_fit_pos,
        eval_meta=eval_meta,
        candidate_ops=candidate_ops,
        model_names=model_names,
        equivalence_labels=equivalence_labels,
        noise_mults=noise_mults,
        n_noise_samples=n_noise_samples,
        refit_train_n=refit_train_n,
        refit_val_n=refit_val_n,
        base_noise_ceiling=base_noise_ceiling,
        metric=metric,
        corr_type=corr_type,
        eval_noise_mode=eval_noise_mode,
        fit_noise_calibration=fit_noise_calibration,
        rdm_calibration_comparison=rdm_calibration_comparison,
        eval_refit_mode=eval_refit_mode,
        calibration_images=calibration_images,
        calibration_noise_samples=calibration_noise_samples,
        calibration_max_iter=calibration_max_iter,
        target_dim=target_dim,
        teacher_cache_dir=teacher_cache_dir,
        expected_teacher_rows=expected_teacher_rows,
        seed=seed,
        batch_noise_samples=batch_noise_samples,
        rdm_device=rdm_device,
        gpu_alpha_batch=gpu_alpha_batch,
        gpu_predict_batch=gpu_predict_batch,
        gpu_eval_noise_batch=gpu_eval_noise_batch,
    )

    rows: list[dict[str, Any]] = []
    if teacher_workers == 1 or len(selected_teacher_indices) <= 1:
        for teacher_idx in selected_teacher_indices:
            rows.extend(
                _run_single_teacher_rdm_recovery(
                    teacher_idx=teacher_idx,
                    **common_kwargs,
                )
            )
        return rows

    max_workers = min(teacher_workers, len(selected_teacher_indices))
    print(
        f"    parallel teacher workers: {max_workers} "
        f"for {len(selected_teacher_indices)} teachers",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            teacher_idx: executor.submit(
                _run_single_teacher_rdm_recovery,
                teacher_idx=teacher_idx,
                **common_kwargs,
            )
            for teacher_idx in selected_teacher_indices
        }
        for teacher_idx in selected_teacher_indices:
            rows.extend(futures[teacher_idx].result())
    return rows
