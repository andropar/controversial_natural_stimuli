#!/usr/bin/env python3
"""Compute full-refit target-adaptation scores across the canonical weight grid.

This is the full-refit counterpart to
``02_score_target_adaptation_fixed_alpha.py``.  For each requested
subject/model/target-set/weight block, it recomputes weighted feature and
response normalization, reselects per-voxel ridge alphas on that weighted
training objective, and then scores CSTIM LOSO plus held-out Vicco.  It can also
compute the matching Vicco LOSO blocks so the output mirrors the canonical
target-adaptation table.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import _paths  # noqa: F401
from _paths import RESULTS_DIR

import numpy as np
import pandas as pd
import scipy.linalg

from cstims.encoding.ridge_gcv_fast import RidgeCVFast
from cstims import constants, paths
from cstims.target_adaptation import (
    BASELINE_SET,
    CSTIM_SETS,
    Selection,
    atomic_write_csv,
    layer_sweep_eval_design,
    parse_weights,
    rsa_spearman,
    rsa_spearman_bootstrap_mean,
)
from srp_utils import FEATURE_PROTOCOL, SRP_TARGET_DIM


DEFAULT_WEIGHTS = "0,0.25,0.5,1,2,4,8,16,32,47"
DEFAULT_N_VICCO_BOOT = 1000
DEFAULT_OUTPUT_STEM = "default"
REFIT_RESULTS_DIR = RESULTS_DIR / "03_refit_alpha"
FEATURE_EPS = 1e-6
RESPONSE_EPS = 1e-6
FIT_SCOPE = "deepvision-plus-target-full-refit"
ALPHA_RULE = "per_voxel_weighted_deepvision_plus_target_ridgecvfast_loo"
ZSCORE_REFERENCE = "weighted_deepvision_unique_plus_target"


def load_main_scorer_module():
    path = Path(__file__).with_name("02_score_target_adaptation_fixed_alpha.py")
    spec = importlib.util.spec_from_file_location("_target_adaptation_main_scorer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = load_main_scorer_module()
ALPHA_GRID = SCORER.ALPHA_GRID
CANONICAL_SCORE_CSV = SCORER.SCORE_CSV


def output_paths(output_stem: str) -> tuple[Path, Path, Path]:
    stem = output_stem.strip()
    known_dirs = {
        "default": REFIT_RESULTS_DIR,
        "target_adaptation_full_refit_all_weights": REFIT_RESULTS_DIR,
        "target_adaptation_full_refit_w4700": REFIT_RESULTS_DIR / "w4700",
        "target_adaptation_full_refit_all_weights_plus4700": REFIT_RESULTS_DIR / "plus4700",
    }
    if stem in known_dirs:
        out_dir = known_dirs[stem]
        return (
            out_dir / "scores.csv",
            out_dir / "summary.csv",
            out_dir / "metadata.json",
        )
    if stem.startswith("target_adaptation_full_refit_all_weights_by_model_"):
        out_dir = REFIT_RESULTS_DIR / "by_model" / "all_weights"
    elif stem.startswith("target_adaptation_full_refit_w4700_by_model_"):
        out_dir = REFIT_RESULTS_DIR / "by_model" / "w4700"
    else:
        out_dir = REFIT_RESULTS_DIR
    return (
        out_dir / f"{stem}_scores.csv",
        out_dir / f"{stem}_summary.csv",
        out_dir / f"{stem}_metadata.json",
    )


def weighted_zscore(
    values: np.ndarray,
    sample_weight: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float32)
    w = np.asarray(sample_weight, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got {x.shape}")
    if w.shape[0] != x.shape[0]:
        raise ValueError(f"Weight length {w.shape[0]} does not match rows {x.shape[0]}")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Sample weights must have positive total weight")
    mean = ((x.astype(np.float64) * w[:, None]).sum(axis=0) / total).astype(np.float32)
    centered = x.astype(np.float64) - mean.astype(np.float64)
    var = ((centered * centered) * w[:, None]).sum(axis=0) / total
    scale = np.sqrt(np.maximum(var, eps * eps)).astype(np.float32)
    return ((x - mean[None, :]) / scale[None, :]).astype(np.float32), mean, scale


def fit_weighted_full_refit_alphas(
    *,
    X_train_z: np.ndarray,
    Y_train_z: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, float]:
    t0 = time.time()
    ridge = RidgeCVFast(
        alphas=ALPHA_GRID,
        scoring="pearson_r",
        alpha_per_target=True,
        fit_intercept=True,
        gcv_mode=None,
        store_cv_values=False,
    )
    ridge.fit(
        np.asarray(X_train_z, dtype=np.float64),
        np.asarray(Y_train_z, dtype=np.float64),
        sample_weight=np.asarray(sample_weight, dtype=np.float64),
    )
    return ridge.alpha_.astype(np.float32), time.time() - t0


def weighted_dual_decomposition(
    X_train_fit_z: np.ndarray,
    Y_train_z: np.ndarray,
    sample_weight: np.ndarray,
) -> dict:
    X = np.asarray(X_train_fit_z, dtype=np.float64)
    Y = np.asarray(Y_train_z, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64)
    sqrt_w = np.sqrt(w)
    weight_total = float(w.sum())

    x_bar = (X * w[:, None]).sum(axis=0) / weight_total
    y_bar = (Y * w[:, None]).sum(axis=0) / weight_total
    Xc = X - x_bar[None, :]
    Yc = Y - y_bar[None, :]
    Xw = Xc * sqrt_w[:, None]
    Yw = Yc * sqrt_w[:, None]

    kernel = Xw @ Xw.T
    kernel = (kernel + kernel.T) * 0.5
    eigvals, eigvecs = scipy.linalg.eigh(kernel, overwrite_a=True, check_finite=False)
    eigvals = np.maximum(eigvals.astype(np.float64), 0.0)
    eigvecs = eigvecs.astype(np.float64, copy=False)
    return {
        "X": X,
        "Y": Y,
        "w": w,
        "sqrt_w": sqrt_w,
        "weight_total": weight_total,
        "x_bar": x_bar,
        "y_bar": y_bar,
        "Xc": Xc,
        "Xw": Xw,
        "Yq": eigvecs.T @ Yw,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
    }


def predict_heldout_from_weighted_refit(
    *,
    decomp: dict,
    X_eval_z: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    X_eval = np.asarray(X_eval_z, dtype=np.float64) - decomp["x_bar"][None, :]
    eval_u = (X_eval @ decomp["Xw"].T) @ decomp["eigvecs"]
    pred = np.empty((X_eval_z.shape[0], decomp["Yq"].shape[1]), dtype=np.float32)
    for alpha in np.unique(alphas):
        vox_idx = np.where(np.isclose(alphas, alpha))[0]
        if len(vox_idx) == 0:
            continue
        inv = 1.0 / (decomp["eigvals"] + float(alpha))
        coef_q = decomp["Yq"][:, vox_idx] * inv[:, None]
        pred[:, vox_idx] = (
            decomp["y_bar"][vox_idx][None, :] + eval_u @ coef_q
        ).astype(np.float32)
    return pred


def weighted_refit_predictions(
    *,
    X_train_fit_z: np.ndarray,
    Y_train_z: np.ndarray,
    sample_weight: np.ndarray,
    target_indices: np.ndarray | None,
    X_target_eval_z: np.ndarray,
    X_vicco_eval_z: np.ndarray | None,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, float]:
    """Predict target LOSO and optional held-out Vicco.

    When ``target_indices`` is ``None`` the target samples were not included in
    the weighted fit, which is the correct full-refit limit for
    ``target_weight=0``.  In that case target predictions are held-out
    predictions and leverage is exactly zero.
    """
    t0 = time.time()
    decomp = weighted_dual_decomposition(X_train_fit_z, Y_train_z, sample_weight)
    if target_indices is None:
        target_pred = predict_heldout_from_weighted_refit(
            decomp=decomp,
            X_eval_z=X_target_eval_z,
            alphas=alphas,
        )
        vicco_pred = (
            predict_heldout_from_weighted_refit(
                decomp=decomp,
                X_eval_z=X_vicco_eval_z,
                alphas=alphas,
            )
            if X_vicco_eval_z is not None
            else None
        )
        target_hat = np.zeros_like(target_pred, dtype=np.float32)
        return target_pred, vicco_pred, target_hat, time.time() - t0

    X = decomp["X"]
    Y = decomp["Y"]
    w = decomp["w"]
    sqrt_w = decomp["sqrt_w"]
    eigvals = decomp["eigvals"]
    eigvecs = decomp["eigvecs"]
    Yq = decomp["Yq"]
    y_bar = decomp["y_bar"]
    x_bar = decomp["x_bar"]
    Xw = decomp["Xw"]

    target_indices = np.asarray(target_indices, dtype=int)
    if np.any(w[target_indices] <= 0):
        raise ValueError("target_indices require strictly positive sample weights")

    target_fit = decomp["Xc"][target_indices]
    target_eval = np.asarray(X_target_eval_z, dtype=np.float64) - x_bar[None, :]
    target_weight = w[target_indices]
    target_eval_u = (target_eval @ Xw.T) @ eigvecs
    target_fit_u = (eigvecs[target_indices] * eigvals[None, :]) / sqrt_w[
        target_indices, None
    ]
    target_fit_cross_u = eigvecs.T @ (Xw @ target_fit.T)
    target_eval_dot_fit = np.sum(target_eval * target_fit, axis=1)

    n_target = len(target_indices)
    n_voxels = Y.shape[1]
    target_loso = np.empty((n_target, n_voxels), dtype=np.float32)
    target_hat = np.empty((n_target, n_voxels), dtype=np.float32)

    if X_vicco_eval_z is not None:
        vicco_eval = np.asarray(X_vicco_eval_z, dtype=np.float64) - x_bar[None, :]
        vicco_eval_u = (vicco_eval @ Xw.T) @ eigvecs
        vicco_pred = np.empty((X_vicco_eval_z.shape[0], n_voxels), dtype=np.float32)
    else:
        vicco_eval_u = None
        vicco_pred = None

    for alpha in np.unique(alphas):
        vox_idx = np.where(np.isclose(alphas, alpha))[0]
        if len(vox_idx) == 0:
            continue
        inv = 1.0 / (eigvals + float(alpha))
        coef_q = Yq[:, vox_idx] * inv[:, None]

        target_fit_pred = y_bar[vox_idx][None, :] + target_fit_u @ coef_q
        target_eval_pred = y_bar[vox_idx][None, :] + target_eval_u @ coef_q
        residual = Y[target_indices[:, None], vox_idx] - target_fit_pred
        hat_feature = (eigvecs[target_indices] * eigvecs[target_indices]) @ (
            eigvals * inv
        )
        hat = np.clip(target_weight / decomp["weight_total"] + hat_feature, 0.0, 0.999999)
        dual_cross = np.sum(
            target_eval_u * inv[None, :] * target_fit_cross_u.T,
            axis=1,
        )
        cross = target_weight / decomp["weight_total"] + target_weight * (
            (target_eval_dot_fit - dual_cross) / float(alpha)
        )
        target_loso[:, vox_idx] = (
            target_eval_pred - cross[:, None] * residual / (1.0 - hat[:, None])
        ).astype(np.float32)
        target_hat[:, vox_idx] = hat[:, None].astype(np.float32)
        if vicco_pred is not None and vicco_eval_u is not None:
            vicco_pred[:, vox_idx] = (
                y_bar[vox_idx][None, :] + vicco_eval_u @ coef_q
            ).astype(np.float32)

    return target_loso, vicco_pred, target_hat, time.time() - t0


def select_rows(
    *,
    model_set: str,
    subject: str,
    models: list[str] | None,
    max_selections: int | None,
) -> pd.DataFrame:
    selections = SCORER.load_best_shared_selections()
    if model_set != "all":
        selections = selections[selections["model"].isin(set(constants.MODEL_SETS[model_set]))]
    if subject != "all":
        selections = selections[selections["subject"].eq(subject)]
    if models:
        selections = selections[selections["model"].isin(models)]
    selections = selections.sort_values(["subject", "model"]).reset_index(drop=True)
    if max_selections is not None:
        selections = selections.head(max_selections).copy()
    if selections.empty:
        raise RuntimeError("No selections after filtering")
    return selections


def target_sets_for_selection(model: str, *, model_set: str, score_all_model_sets: bool) -> list[str]:
    if model_set != "all":
        return [model_set] if model in set(constants.MODEL_SETS[model_set]) else []
    return SCORER.target_sets_for_model(
        model,
        score_membership_only=not score_all_model_sets,
    )


def canonical_lookup(canonical: pd.DataFrame) -> dict[tuple[str, str, str, str, str, float], pd.Series]:
    out = {}
    for row in canonical.itertuples(index=False):
        out[
            (
                str(row.subject),
                str(row.model),
                str(row.model_set),
                str(row.adaptation_target),
                str(row.eval_target),
                float(row.target_weight),
            )
        ] = row
    return out


def canonical_values(
    canonical_rows: dict[tuple[str, str, str, str, str, float], pd.Series],
    *,
    sel: Selection,
    model_set: str,
    adaptation_target: str,
    eval_target: str,
    weight: float,
) -> tuple[float, float, float, float]:
    row = canonical_rows.get(
        (sel.subject, sel.model, model_set, adaptation_target, eval_target, float(weight))
    )
    if row is None:
        return np.nan, np.nan, np.nan, np.nan
    canonical_score = float(row.mrsa_loso)
    canonical_sem = float(row.mrsa_loso_sem) if np.isfinite(row.mrsa_loso_sem) else np.nan
    original_score = float(row.original_best_shared_mrsa)
    original_sem = (
        float(row.original_best_shared_mrsa_sem)
        if np.isfinite(row.original_best_shared_mrsa_sem)
        else np.nan
    )
    return canonical_score, canonical_sem, original_score, original_sem


def add_score_row(
    out: list[dict],
    *,
    sel: Selection,
    model_set: str,
    adaptation_target: str,
    eval_target: str,
    stimulus_type: str,
    weight: float,
    score: float,
    score_sem: float,
    canonical_score: float,
    canonical_sem: float,
    original_score: float,
    original_sem: float,
    n_deepvision_train: int,
    n_target_train: int,
    n_stimuli_scored: int,
    n_score_bootstrap: int,
    score_sample_size: int,
    alphas: np.ndarray,
    hat_diag: np.ndarray | None,
    feature_reference: str,
    response_reference: str,
    training_target_scope: str,
    runtime_alpha: float,
    runtime_prediction: float,
    runtime_total: float,
    fit_sample_weight_note: str,
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
            "target_weight": float(weight),
            "cstim_weight": float(weight),
            "mrsa_loso": float(score),
            "mrsa_loso_sem": float(score_sem) if np.isfinite(score_sem) else np.nan,
            "canonical_fixed_dv_stats_mrsa": canonical_score,
            "canonical_fixed_dv_stats_mrsa_sem": canonical_sem,
            "delta_vs_canonical_fixed_dv_stats": float(score) - canonical_score
            if np.isfinite(score) and np.isfinite(canonical_score)
            else np.nan,
            "original_best_shared_mrsa": original_score,
            "original_best_shared_mrsa_sem": original_sem,
            "delta_vs_original": float(score) - original_score
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
            "fit_scope": FIT_SCOPE,
            "fit_sample_weight_note": fit_sample_weight_note,
            "alpha_rule": ALPHA_RULE,
            "alpha_median": float(np.median(alphas)),
            "alpha_mean": float(np.mean(alphas)),
            "alpha_std": float(np.std(alphas)),
            "n_alpha_unique": int(len(np.unique(alphas))),
            "hat_diag_mean": float(np.mean(hat_diag)) if hat_diag is not None else np.nan,
            "hat_diag_max": float(np.max(hat_diag)) if hat_diag is not None else np.nan,
            "target_zscore_reference": response_reference,
            "feature_zscore_reference": feature_reference,
            "prediction_protocol": "layer_sweep_stream_predict_v1",
            "training_target_scope": training_target_scope,
            "runtime_seconds_alpha": float(runtime_alpha),
            "runtime_seconds_prediction": float(runtime_prediction),
            "runtime_seconds_subject_model": float(runtime_total),
        }
    )


class SelectionContext:
    """Per-subject/model cache for arrays and reusable DeepVision-only fits."""

    def __init__(self, sel: Selection):
        self.sel = sel
        self.key = (sel.subject, sel.model, sel.layer)
        self.cstim_data = SCORER.load_cstim_subject_data(sel.subject)
        self.X_dv_raw = SCORER.load_feature(
            SCORER.dv_feature_path(sel.subject, sel.model), sel.layer
        )
        self.Y_dv_raw, _hlvis = SCORER.load_deepvision_responses(sel.subject)
        if self.X_dv_raw.shape[0] != self.Y_dv_raw.shape[0]:
            raise ValueError(
                f"DeepVision feature/response length mismatch for {sel.subject}/{sel.model}: "
                f"{self.X_dv_raw.shape[0]} vs {self.Y_dv_raw.shape[0]}"
            )
        self.X_dv_raw = self.X_dv_raw.astype(np.float32, copy=False)
        self.Y_dv_raw = self.Y_dv_raw.astype(np.float32, copy=False)
        self._blocks: dict[str, dict] = {}
        self._vicco_bootstrap: dict[int, tuple[list[np.ndarray], list[np.ndarray], int]] = {}
        self._zero_state: dict | None = None
        self._zero_shared_runtime_pending = False

    def stimulus_block(self, model_set: str) -> dict:
        if model_set not in self._blocks:
            X_all = SCORER.load_feature(SCORER.feature_path(self.sel.model, model_set), self.sel.layer)
            file_idx = self.cstim_data["group_file_idx"][model_set]
            brain_idx = self.cstim_data["group_brain_idx"][model_set]
            self._blocks[model_set] = {
                "X_raw": X_all[file_idx].astype(np.float32),
                "Y_raw": self.cstim_data["betas_hlvis"][:, brain_idx].T.astype(np.float32),
                "file_idx": file_idx,
                "brain_idx": brain_idx,
            }
        return self._blocks[model_set]

    def vicco_bootstrap(
        self, n_vicco_boot: int
    ) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        if n_vicco_boot not in self._vicco_bootstrap:
            vicco = self.stimulus_block(BASELINE_SET)
            self._vicco_bootstrap[n_vicco_boot] = SCORER.load_or_compute_vicco_bootstrap(
                subject=self.sel.subject,
                betas_hlvis=self.cstim_data["betas_hlvis"],
                vicco_brain_idx=vicco["brain_idx"],
                n_bootstrap=n_vicco_boot,
            )
        return self._vicco_bootstrap[n_vicco_boot]

    def zero_refit_state_for_job(self) -> tuple[dict, float, float]:
        """Return the DeepVision-only fit and runtime to charge to this job."""

        if self._zero_state is None:
            sample_weight = np.ones(self.X_dv_raw.shape[0], dtype=np.float64)
            X_train_z, feature_mean, feature_scale = weighted_zscore(
                self.X_dv_raw, sample_weight, eps=FEATURE_EPS
            )
            Y_train_z, _response_mean, _response_scale = weighted_zscore(
                self.Y_dv_raw, sample_weight, eps=RESPONSE_EPS
            )
            alphas, alpha_runtime = fit_weighted_full_refit_alphas(
                X_train_z=X_train_z,
                Y_train_z=Y_train_z,
                sample_weight=sample_weight,
            )
            t_decomp = time.time()
            decomp = weighted_dual_decomposition(X_train_z, Y_train_z, sample_weight)
            decomp_runtime = time.time() - t_decomp
            self._zero_state = {
                "X_train_z": X_train_z,
                "Y_train_z": Y_train_z,
                "sample_weight": sample_weight,
                "feature_mean": feature_mean,
                "feature_scale": feature_scale,
                "alphas": alphas,
                "decomp": decomp,
                "n_dv": int(self.X_dv_raw.shape[0]),
                "weight_note": "deepvision=1,target=0",
                "alpha_runtime": alpha_runtime,
                "decomp_runtime": decomp_runtime,
            }
            self._zero_shared_runtime_pending = True

        if self._zero_shared_runtime_pending:
            self._zero_shared_runtime_pending = False
            return (
                self._zero_state,
                float(self._zero_state["alpha_runtime"]),
                float(self._zero_state["decomp_runtime"]),
            )
        return self._zero_state, 0.0, 0.0


def eval_design_from_raw(
    X_raw: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    X_z = ((X_raw - feature_mean[None, :]) / feature_scale[None, :]).astype(np.float32)
    return layer_sweep_eval_design(X_z, feature_mean, feature_scale)


def prepare_weighted_training(
    *,
    X_dv_raw: np.ndarray,
    Y_dv_raw: np.ndarray,
    X_target_raw: np.ndarray,
    Y_target_raw: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int, str]:
    if X_dv_raw.shape[0] != Y_dv_raw.shape[0]:
        raise ValueError(
            f"DeepVision feature/response length mismatch: {X_dv_raw.shape[0]} vs "
            f"{Y_dv_raw.shape[0]}"
        )
    if weight <= 0:
        sample_weight = np.ones(X_dv_raw.shape[0], dtype=np.float64)
        return (
            X_dv_raw.astype(np.float32, copy=False),
            Y_dv_raw.astype(np.float32, copy=False),
            sample_weight,
            None,
            int(X_dv_raw.shape[0]),
            "deepvision=1,target=0",
        )
    sample_weight = np.concatenate(
        [
            np.ones(X_dv_raw.shape[0], dtype=np.float64),
            np.full(X_target_raw.shape[0], float(weight), dtype=np.float64),
        ]
    )
    target_indices = np.arange(X_dv_raw.shape[0], X_dv_raw.shape[0] + X_target_raw.shape[0])
    return (
        np.vstack([X_dv_raw, X_target_raw]).astype(np.float32),
        np.vstack([Y_dv_raw, Y_target_raw]).astype(np.float32),
        sample_weight,
        target_indices.astype(int),
        int(X_dv_raw.shape[0]),
        f"deepvision=1,target={float(weight):g}",
    )


def score_cstim_job(
    sel: Selection,
    *,
    model_set: str,
    weight: float,
    n_vicco_boot: int,
    canonical_rows: dict[tuple[str, str, str, str, str, float], pd.Series],
    ctx: SelectionContext | None = None,
) -> list[dict]:
    t0 = time.time()
    ctx = ctx or SelectionContext(sel)
    target = ctx.stimulus_block(model_set)
    vicco = ctx.stimulus_block(BASELINE_SET)
    X_target_raw = target["X_raw"]
    Y_target_raw = target["Y_raw"]
    X_vicco_raw = vicco["X_raw"]
    Y_vicco_raw = vicco["Y_raw"]

    if weight <= 0:
        state, alpha_runtime, shared_pred_runtime = ctx.zero_refit_state_for_job()
        feature_mean = state["feature_mean"]
        feature_scale = state["feature_scale"]
        X_target_eval_z = eval_design_from_raw(X_target_raw, feature_mean, feature_scale)
        X_vicco_eval_z = eval_design_from_raw(X_vicco_raw, feature_mean, feature_scale)
        t_pred = time.time()
        target_pred = predict_heldout_from_weighted_refit(
            decomp=state["decomp"],
            X_eval_z=X_target_eval_z,
            alphas=state["alphas"],
        )
        vicco_pred = predict_heldout_from_weighted_refit(
            decomp=state["decomp"],
            X_eval_z=X_vicco_eval_z,
            alphas=state["alphas"],
        )
        pred_runtime = shared_pred_runtime + time.time() - t_pred
        target_hat = np.zeros_like(target_pred, dtype=np.float32)
        alphas = state["alphas"]
        n_dv = state["n_dv"]
        weight_note = state["weight_note"]
    else:
        X_train_raw, Y_train_raw, sample_weight, target_indices, n_dv, weight_note = (
            prepare_weighted_training(
                X_dv_raw=ctx.X_dv_raw,
                Y_dv_raw=ctx.Y_dv_raw,
                X_target_raw=X_target_raw,
                Y_target_raw=Y_target_raw,
                weight=weight,
            )
        )
        X_train_z, feature_mean, feature_scale = weighted_zscore(
            X_train_raw, sample_weight, eps=FEATURE_EPS
        )
        Y_train_z, _response_mean, _response_scale = weighted_zscore(
            Y_train_raw, sample_weight, eps=RESPONSE_EPS
        )
        X_target_eval_z = eval_design_from_raw(X_target_raw, feature_mean, feature_scale)
        X_vicco_eval_z = eval_design_from_raw(X_vicco_raw, feature_mean, feature_scale)

        alphas, alpha_runtime = fit_weighted_full_refit_alphas(
            X_train_z=X_train_z,
            Y_train_z=Y_train_z,
            sample_weight=sample_weight,
        )
        target_pred, vicco_pred, target_hat, pred_runtime = weighted_refit_predictions(
            X_train_fit_z=X_train_z,
            Y_train_z=Y_train_z,
            sample_weight=sample_weight,
            target_indices=target_indices,
            X_target_eval_z=X_target_eval_z,
            X_vicco_eval_z=X_vicco_eval_z,
            alphas=alphas,
        )
        if vicco_pred is None:
            raise RuntimeError("Expected held-out Vicco predictions for CSTIM job")

    cstim_score = rsa_spearman(target_pred, Y_target_raw)
    vicco_boot, vicco_brain_ranks, vicco_sample_size = ctx.vicco_bootstrap(n_vicco_boot)
    if vicco_boot:
        vicco_score, vicco_sem = rsa_spearman_bootstrap_mean(
            vicco_pred,
            boot=vicco_boot,
            brain_ranks=vicco_brain_ranks,
        )
        vicco_n_boot = len(vicco_boot)
        vicco_n_scored = vicco_sample_size
    else:
        vicco_score = rsa_spearman(vicco_pred, Y_vicco_raw)
        vicco_sem = np.nan
        vicco_n_boot = 1
        vicco_n_scored = Y_vicco_raw.shape[0]

    rows = []
    elapsed = time.time() - t0
    for eval_target, stimulus_type, score, sem, n_scored, n_boot, hat, scope in [
        (
            "cstim_loso",
            "controversial",
            cstim_score,
            np.nan,
            X_target_raw.shape[0],
            1,
            target_hat,
            f"full_refit_deepvision_unique_plus_{model_set}_target_loso",
        ),
        (
            "vicco_heldout",
            "baseline",
            vicco_score,
            vicco_sem,
            vicco_n_scored,
            vicco_n_boot,
            None,
            f"full_refit_deepvision_unique_plus_{model_set}_target_vicco_held_out",
        ),
    ]:
        canonical_score, canonical_sem, original_score, original_sem = canonical_values(
            canonical_rows,
            sel=sel,
            model_set=model_set,
            adaptation_target=model_set,
            eval_target=eval_target,
            weight=weight,
        )
        add_score_row(
            rows,
            sel=sel,
            model_set=model_set,
            adaptation_target=model_set,
            eval_target=eval_target,
            stimulus_type=stimulus_type,
            weight=weight,
            score=score,
            score_sem=sem,
            canonical_score=canonical_score,
            canonical_sem=canonical_sem,
            original_score=original_score,
            original_sem=original_sem,
            n_deepvision_train=n_dv,
            n_target_train=X_target_raw.shape[0],
            n_stimuli_scored=n_scored,
            n_score_bootstrap=n_boot,
            score_sample_size=n_scored,
            alphas=alphas,
            hat_diag=hat,
            feature_reference=ZSCORE_REFERENCE,
            response_reference=ZSCORE_REFERENCE,
            training_target_scope=scope,
            runtime_alpha=alpha_runtime,
            runtime_prediction=pred_runtime,
            runtime_total=elapsed,
            fit_sample_weight_note=weight_note,
        )
    return rows


def score_vicco_loso_job(
    sel: Selection,
    *,
    weight: float,
    n_vicco_boot: int,
    canonical_rows: dict[tuple[str, str, str, str, str, float], pd.Series],
    ctx: SelectionContext | None = None,
) -> list[dict]:
    t0 = time.time()
    ctx = ctx or SelectionContext(sel)
    vicco = ctx.stimulus_block(BASELINE_SET)
    X_vicco_raw = vicco["X_raw"]
    Y_vicco_raw = vicco["Y_raw"]

    if weight <= 0:
        state, alpha_runtime, shared_pred_runtime = ctx.zero_refit_state_for_job()
        X_vicco_eval_z = eval_design_from_raw(
            X_vicco_raw,
            state["feature_mean"],
            state["feature_scale"],
        )
        t_pred = time.time()
        vicco_loso_pred = predict_heldout_from_weighted_refit(
            decomp=state["decomp"],
            X_eval_z=X_vicco_eval_z,
            alphas=state["alphas"],
        )
        pred_runtime = shared_pred_runtime + time.time() - t_pred
        vicco_hat = np.zeros_like(vicco_loso_pred, dtype=np.float32)
        alphas = state["alphas"]
        n_dv = state["n_dv"]
        weight_note = state["weight_note"]
    else:
        X_train_raw, Y_train_raw, sample_weight, target_indices, n_dv, weight_note = (
            prepare_weighted_training(
                X_dv_raw=ctx.X_dv_raw,
                Y_dv_raw=ctx.Y_dv_raw,
                X_target_raw=X_vicco_raw,
                Y_target_raw=Y_vicco_raw,
                weight=weight,
            )
        )
        X_train_z, feature_mean, feature_scale = weighted_zscore(
            X_train_raw, sample_weight, eps=FEATURE_EPS
        )
        Y_train_z, _response_mean, _response_scale = weighted_zscore(
            Y_train_raw, sample_weight, eps=RESPONSE_EPS
        )
        X_vicco_eval_z = eval_design_from_raw(X_vicco_raw, feature_mean, feature_scale)

        alphas, alpha_runtime = fit_weighted_full_refit_alphas(
            X_train_z=X_train_z,
            Y_train_z=Y_train_z,
            sample_weight=sample_weight,
        )
        vicco_loso_pred, _unused, vicco_hat, pred_runtime = weighted_refit_predictions(
            X_train_fit_z=X_train_z,
            Y_train_z=Y_train_z,
            sample_weight=sample_weight,
            target_indices=target_indices,
            X_target_eval_z=X_vicco_eval_z,
            X_vicco_eval_z=None,
            alphas=alphas,
        )

    vicco_boot, vicco_brain_ranks, vicco_sample_size = ctx.vicco_bootstrap(n_vicco_boot)
    if vicco_boot:
        vicco_score, vicco_sem = rsa_spearman_bootstrap_mean(
            vicco_loso_pred,
            boot=vicco_boot,
            brain_ranks=vicco_brain_ranks,
        )
        vicco_n_boot = len(vicco_boot)
        vicco_n_scored = vicco_sample_size
    else:
        vicco_score = rsa_spearman(vicco_loso_pred, Y_vicco_raw)
        vicco_sem = np.nan
        vicco_n_boot = 1
        vicco_n_scored = Y_vicco_raw.shape[0]

    canonical_score, canonical_sem, original_score, original_sem = canonical_values(
        canonical_rows,
        sel=sel,
        model_set=BASELINE_SET,
        adaptation_target=BASELINE_SET,
        eval_target="vicco_loso",
        weight=weight,
    )
    rows: list[dict] = []
    add_score_row(
        rows,
        sel=sel,
        model_set=BASELINE_SET,
        adaptation_target=BASELINE_SET,
        eval_target="vicco_loso",
        stimulus_type="baseline",
        weight=weight,
        score=vicco_score,
        score_sem=vicco_sem,
        canonical_score=canonical_score,
        canonical_sem=canonical_sem,
        original_score=original_score,
        original_sem=original_sem,
        n_deepvision_train=n_dv,
        n_target_train=X_vicco_raw.shape[0],
        n_stimuli_scored=vicco_n_scored,
        n_score_bootstrap=vicco_n_boot,
        score_sample_size=vicco_n_scored,
        alphas=alphas,
        hat_diag=vicco_hat,
        feature_reference=ZSCORE_REFERENCE,
        response_reference=ZSCORE_REFERENCE,
        training_target_scope="full_refit_deepvision_unique_plus_vicco_target_loso",
        runtime_alpha=alpha_runtime,
        runtime_prediction=pred_runtime,
        runtime_total=time.time() - t0,
        fit_sample_weight_note=weight_note,
    )
    return rows


def make_jobs(
    selections: pd.DataFrame,
    *,
    weights: list[float],
    model_set: str,
    score_all_model_sets: bool,
    include_vicco_loso: bool,
) -> list[dict]:
    jobs = []
    for row in selections.itertuples(index=False):
        target_sets = target_sets_for_selection(
            row.model,
            model_set=model_set,
            score_all_model_sets=score_all_model_sets,
        )
        for weight in weights:
            for target_set in target_sets:
                jobs.append(
                    {
                        "kind": "cstim",
                        "subject": row.subject,
                        "model": row.model,
                        "display_name": row.display_name,
                        "layer": row.layer,
                        "model_set": target_set,
                        "weight": float(weight),
                    }
                )
            if include_vicco_loso:
                jobs.append(
                    {
                        "kind": "vicco",
                        "subject": row.subject,
                        "model": row.model,
                        "display_name": row.display_name,
                        "layer": row.layer,
                        "model_set": BASELINE_SET,
                        "weight": float(weight),
                    }
                )
    return jobs


def sort_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores
    return scores.sort_values(
        [
            "subject",
            "model",
            "target_weight",
            "model_set",
            "adaptation_target",
            "eval_target",
        ],
        kind="stable",
    ).reset_index(drop=True)


def completed_rows_for_job(
    existing: pd.DataFrame,
    *,
    job: dict,
    n_vicco_boot: int,
) -> pd.DataFrame:
    if existing.empty:
        return pd.DataFrame(columns=existing.columns)
    required = {
        "subject",
        "model",
        "model_set",
        "adaptation_target",
        "target_weight",
        "eval_target",
        "feature_dim_analysis",
        "feature_protocol",
        "fit_scope",
        "n_score_bootstrap",
    }
    if required.difference(existing.columns):
        return pd.DataFrame(columns=existing.columns)
    block = existing[
        existing["subject"].eq(job["subject"])
        & existing["model"].eq(job["model"])
        & existing["model_set"].eq(job["model_set"])
        & np.isclose(existing["target_weight"].astype(float), float(job["weight"]))
        & existing["feature_dim_analysis"].eq(SRP_TARGET_DIM)
        & existing["feature_protocol"].eq(FEATURE_PROTOCOL)
        & existing["fit_scope"].eq(FIT_SCOPE)
    ].copy()
    if job["kind"] == "cstim":
        block = block[block["adaptation_target"].eq(job["model_set"])].copy()
        targets = {"cstim_loso", "vicco_heldout"}
        if not targets.issubset(set(block["eval_target"])):
            return pd.DataFrame(columns=existing.columns)
        rows = []
        for target in ["cstim_loso", "vicco_heldout"]:
            row = block[block["eval_target"].eq(target)].tail(1)
            if target == "vicco_heldout" and not row["n_score_bootstrap"].eq(n_vicco_boot).all():
                return pd.DataFrame(columns=existing.columns)
            rows.append(row)
        return pd.concat(rows, ignore_index=True)
    block = block[
        block["adaptation_target"].eq(BASELINE_SET) & block["eval_target"].eq("vicco_loso")
    ].copy()
    if block.empty or not block["n_score_bootstrap"].eq(n_vicco_boot).any():
        return pd.DataFrame(columns=existing.columns)
    return block.tail(1).copy()


def write_summary(scores: pd.DataFrame, summary_csv: Path) -> None:
    rows = []
    if scores.empty:
        atomic_write_csv(pd.DataFrame(), summary_csv)
        return
    group_cols = ["model_set", "adaptation_target", "eval_target", "target_weight"]
    for keys, block in scores.groupby(group_cols):
        vals = block["mrsa_loso"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        canonical_delta = block["delta_vs_canonical_fixed_dv_stats"].to_numpy(dtype=float)
        canonical_delta = canonical_delta[np.isfinite(canonical_delta)]
        original_delta = block["delta_vs_original"].to_numpy(dtype=float)
        original_delta = original_delta[np.isfinite(original_delta)]
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "mean_mrsa": float(vals.mean()) if len(vals) else np.nan,
                "sem_mrsa": float(vals.std(ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1
                else np.nan,
                "mean_delta_vs_canonical_fixed_dv_stats": float(canonical_delta.mean())
                if len(canonical_delta)
                else np.nan,
                "mean_delta_vs_original": float(original_delta.mean())
                if len(original_delta)
                else np.nan,
                "n": int(len(block)),
                "n_models": int(block["model"].nunique()),
                "n_subjects": int(block["subject"].nunique()),
                "mean_runtime_seconds_subject_model": float(
                    block["runtime_seconds_subject_model"].mean()
                ),
            }
        )
    atomic_write_csv(pd.DataFrame(rows), summary_csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--model-set",
        default="all",
        choices=["all", *CSTIM_SETS],
        help="Target set to compute, or all membership target sets.",
    )
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-selections", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--score-all-model-sets", action="store_true")
    parser.add_argument("--skip-vicco-loso", action="store_true")
    parser.add_argument("--n-vicco-boot", type=int, default=DEFAULT_N_VICCO_BOOT)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    score_csv, summary_csv, meta_json = output_paths(args.output_stem)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    if score_csv.exists() and not args.overwrite and not args.resume:
        print(f"[cached] {score_csv} exists; use --overwrite or --resume", flush=True)
        return

    weights = parse_weights(args.weights)
    canonical = pd.read_csv(CANONICAL_SCORE_CSV)
    canonical_rows = canonical_lookup(canonical)
    selections = select_rows(
        model_set=args.model_set,
        subject=args.subject,
        models=args.models,
        max_selections=args.max_selections,
    )
    jobs = make_jobs(
        selections,
        weights=weights,
        model_set=args.model_set,
        score_all_model_sets=args.score_all_model_sets,
        include_vicco_loso=not args.skip_vicco_loso,
    )
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    if not jobs:
        raise RuntimeError("No jobs after filtering")

    rows: list[dict] = []
    completed_job_ids: set[int] = set()
    if args.resume and score_csv.exists():
        existing = pd.read_csv(score_csv)
        for job_idx, job in enumerate(jobs):
            complete = completed_rows_for_job(
                existing,
                job=job,
                n_vicco_boot=args.n_vicco_boot,
            )
            if complete.empty:
                continue
            rows.extend(complete.to_dict("records"))
            completed_job_ids.add(job_idx)
        print(
            f"[resume] keeping {len(rows)} rows from {len(completed_job_ids)} complete jobs",
            flush=True,
        )

    print(
        "Full-refit all-weights target adaptation: "
        f"selections={len(selections)} jobs={len(jobs)} "
        f"weights={','.join(f'{w:g}' for w in weights)} "
        f"model_set={args.model_set} n_vicco_boot={args.n_vicco_boot}",
        flush=True,
    )
    current_ctx: SelectionContext | None = None
    current_ctx_key: tuple[str, str, str] | None = None
    for job_idx, job in enumerate(jobs, start=1):
        if job_idx - 1 in completed_job_ids:
            continue
        sel = Selection(
            subject=job["subject"],
            model=job["model"],
            display_name=job["display_name"],
            layer=job["layer"],
        )
        print(
            f"[{job_idx:04d}/{len(jobs):04d}] {job['kind']} "
            f"{sel.subject} {sel.model} layer={sel.layer} "
            f"target={job['model_set']} weight={job['weight']:g}",
            flush=True,
        )
        ctx_key = (sel.subject, sel.model, sel.layer)
        if ctx_key != current_ctx_key:
            current_ctx = SelectionContext(sel)
            current_ctx_key = ctx_key
        if job["kind"] == "cstim":
            new_rows = score_cstim_job(
                sel,
                model_set=job["model_set"],
                weight=float(job["weight"]),
                n_vicco_boot=args.n_vicco_boot,
                canonical_rows=canonical_rows,
                ctx=current_ctx,
            )
        else:
            new_rows = score_vicco_loso_job(
                sel,
                weight=float(job["weight"]),
                n_vicco_boot=args.n_vicco_boot,
                canonical_rows=canonical_rows,
                ctx=current_ctx,
            )
        rows.extend(new_rows)
        new_scores = pd.DataFrame(new_rows)
        status = "; ".join(
            f"{row.eval_target}={row.mrsa_loso:.4f} "
            f"dcanon={row.delta_vs_canonical_fixed_dv_stats:+.4f}"
            for row in new_scores.itertuples(index=False)
        )
        runtime = float(new_scores["runtime_seconds_subject_model"].max())
        print(f"  {status}; runtime={runtime / 60.0:.1f} min", flush=True)
        scores = sort_scores(pd.DataFrame(rows))
        atomic_write_csv(scores, score_csv)
        write_summary(scores, summary_csv)
        print(f"  wrote checkpoint rows={len(scores)} -> {score_csv}", flush=True)

    scores = sort_scores(pd.DataFrame(rows))
    atomic_write_csv(scores, score_csv)
    write_summary(scores, summary_csv)
    meta = {
        "weights": [float(w) for w in weights],
        "model_set": args.model_set,
        "score_all_model_sets": bool(args.score_all_model_sets),
        "include_vicco_loso": not args.skip_vicco_loso,
        "n_selections": int(len(selections)),
        "n_jobs_requested": int(len(jobs)),
        "n_jobs_resumed": int(len(completed_job_ids)),
        "n_rows": int(len(scores)),
        "n_vicco_boot": int(args.n_vicco_boot),
        "feature_dim_analysis": int(SRP_TARGET_DIM),
        "feature_protocol": FEATURE_PROTOCOL,
        "alpha_grid": [float(x) for x in ALPHA_GRID],
        "alpha_rule": ALPHA_RULE,
        "preprocessing": (
            "weighted feature/response zscore over DeepVision unique plus the "
            "target set; target_weight=0 uses a per-process cached DeepVision-only "
            "limit for each subject/model/layer"
        ),
        "fit_scope": FIT_SCOPE,
        "prediction_protocol": "layer_sweep_stream_predict_v1",
        "canonical_score_csv": str(CANONICAL_SCORE_CSV),
        "score_csv": str(score_csv),
        "summary_csv": str(summary_csv),
        "python": os.sys.executable,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_json.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(scores)} rows -> {score_csv}", flush=True)
    print(f"Wrote summary -> {summary_csv}", flush=True)
    print(f"Wrote metadata -> {meta_json}", flush=True)


if __name__ == "__main__":
    main()
