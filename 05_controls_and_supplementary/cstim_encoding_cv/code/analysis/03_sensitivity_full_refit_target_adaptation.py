#!/usr/bin/env python3
"""Full-refit sensitivity check for weighted CSTIM target adaptation.

This script is intentionally separate from the canonical target-adaptation
scorer.  It asks whether the conclusions change if one CSTIM target set is
included in the full ridge fit with weighted preprocessing and weighted alpha
selection, while still using the selected SRP5920 layer/features.
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
from cstims.paper import config
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


DEFAULT_MODEL_SET = "sota"
DEFAULT_WEIGHT = "auto"
DEFAULT_N_VICCO_BOOT = 1000
DEFAULT_FIT_SCOPE = "deepvision-plus-cstim"
FEATURE_EPS = 1e-6
RESPONSE_EPS = 1e-6

FIT_SCOPE_CONFIG = {
    "deepvision-plus-cstim": {
        "stem": "target_adaptation_full_refit_sensitivity",
        "label": "DeepVision+CSTIM full refit",
        "alpha_rule": "per_voxel_weighted_deepvision_plus_cstim_ridgecvfast_loo",
        "zscore_reference": "weighted_deepvision_unique_plus_cstim_target",
        "training_target_scope": "full_refit_deepvision_unique_plus_{model_set}",
        "metadata_alpha_rule": (
            "RidgeCVFast analytical LOO with sample_weight on weighted "
            "DeepVision+CSTIM training set"
        ),
        "metadata_preprocessing": (
            "weighted feature/response zscore over DeepVision unique plus target CSTIM set"
        ),
    },
    "cstim-only": {
        "stem": "target_adaptation_cstim_only_sensitivity",
        "label": "CSTIM-only fit",
        "alpha_rule": "per_voxel_cstim_only_ridgecvfast_loo",
        "zscore_reference": "cstim_target_only",
        "training_target_scope": "cstim_only_{model_set}",
        "metadata_alpha_rule": "RidgeCVFast analytical LOO on CSTIM target samples only",
        "metadata_preprocessing": "feature/response zscore over CSTIM target samples only",
    },
}


def load_main_scorer_module():
    path = Path(__file__).with_name("02_score_target_adaptation_srp5920_per_voxel_alpha.py")
    spec = importlib.util.spec_from_file_location("_target_adaptation_main_scorer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = load_main_scorer_module()
ALPHA_GRID = SCORER.ALPHA_GRID
CANONICAL_SCORE_CSV = SCORER.SCORE_CSV


def output_paths(fit_scope: str) -> tuple[Path, Path, Path]:
    stem = FIT_SCOPE_CONFIG[fit_scope]["stem"]
    return (
        RESULTS_DIR / f"{stem}_scores.csv",
        RESULTS_DIR / f"{stem}_summary.csv",
        RESULTS_DIR / f"{stem}_metadata.json",
    )


def weighted_zscore(
    values: np.ndarray,
    sample_weight: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns using weighted moments over rows."""
    x = np.asarray(values, dtype=np.float32)
    w = np.asarray(sample_weight, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")
    if w.shape[0] != x.shape[0]:
        raise ValueError(f"Weight length {w.shape[0]} does not match rows {x.shape[0]}")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Sample weights must have positive total weight")
    mean = ((x.astype(np.float64) * w[:, None]).sum(axis=0) / total).astype(np.float32)
    centered = x.astype(np.float64) - mean.astype(np.float64)
    var = ((centered * centered) * w[:, None]).sum(axis=0) / total
    scale = np.sqrt(np.maximum(var, eps * eps)).astype(np.float32)
    z = ((x - mean[None, :]) / scale[None, :]).astype(np.float32)
    return z, mean, scale


def choose_auto_weight(canonical: pd.DataFrame, model_set: str) -> float:
    block = canonical[
        canonical["model_set"].eq(model_set) & canonical["eval_target"].eq("cstim_loso")
    ].copy()
    if block.empty:
        raise RuntimeError(f"No canonical CSTIM rows found for model_set={model_set!r}")
    summary = (
        block.groupby("target_weight", as_index=False)["delta_vs_original"]
        .mean()
        .sort_values("delta_vs_original", ascending=False)
    )
    return float(summary.iloc[0]["target_weight"])


def select_rows(
    *,
    model_set: str,
    subject: str,
    models: list[str] | None,
    max_selections: int | None,
) -> pd.DataFrame:
    selections = SCORER.load_best_shared_selections()
    model_members = set(config.MODEL_SETS[model_set])
    selections = selections[selections["model"].isin(model_members)].copy()
    if subject != "all":
        selections = selections[selections["subject"].eq(subject)].copy()
    if models:
        selections = selections[selections["model"].isin(models)].copy()
    selections = selections.sort_values(["subject", "model"]).reset_index(drop=True)
    if max_selections is not None:
        selections = selections.head(max_selections).copy()
    if selections.empty:
        raise RuntimeError("No selections after filtering")
    return selections


def fit_weighted_full_refit_alphas(
    *,
    X_train_z: np.ndarray,
    Y_train_z: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    elapsed = np.array(time.time() - t0, dtype=np.float32)
    return ridge.alpha_.astype(np.float32), elapsed


def weighted_refit_predictions(
    *,
    X_train_fit_z: np.ndarray,
    Y_train_z: np.ndarray,
    sample_weight: np.ndarray,
    target_indices: np.ndarray,
    X_target_eval_z: np.ndarray,
    X_vicco_eval_z: np.ndarray,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Predict target LOSO and held-out Vicco for fixed weighted-refit alphas.

    The fit design is the weighted-standardized training feature matrix.  The
    target and Vicco evaluation designs can differ from the fit design so the
    script can preserve the layer-sweep stream prediction convention.
    """
    t0 = time.time()
    X = np.asarray(X_train_fit_z, dtype=np.float64)
    Y = np.asarray(Y_train_z, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64)
    target_indices = np.asarray(target_indices, dtype=int)
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
    Yq = eigvecs.T @ Yw

    target_fit = Xc[target_indices]
    target_eval = np.asarray(X_target_eval_z, dtype=np.float64) - x_bar[None, :]
    vicco_eval = np.asarray(X_vicco_eval_z, dtype=np.float64) - x_bar[None, :]
    target_weight = w[target_indices]

    target_eval_u = (target_eval @ Xw.T) @ eigvecs
    vicco_eval_u = (vicco_eval @ Xw.T) @ eigvecs
    target_fit_u = (eigvecs[target_indices] * eigvals[None, :]) / sqrt_w[
        target_indices, None
    ]

    target_fit_cross_u = eigvecs.T @ (Xw @ target_fit.T)
    target_eval_dot_fit = np.sum(target_eval * target_fit, axis=1)

    n_target = len(target_indices)
    n_vicco = X_vicco_eval_z.shape[0]
    n_voxels = Y.shape[1]
    target_loso = np.empty((n_target, n_voxels), dtype=np.float32)
    vicco_pred = np.empty((n_vicco, n_voxels), dtype=np.float32)
    target_hat = np.empty((n_target, n_voxels), dtype=np.float32)

    for alpha in np.unique(alphas):
        vox_idx = np.where(np.isclose(alphas, alpha))[0]
        if len(vox_idx) == 0:
            continue
        inv = 1.0 / (eigvals + float(alpha))
        coef_q = Yq[:, vox_idx] * inv[:, None]

        target_fit_pred = y_bar[vox_idx][None, :] + target_fit_u @ coef_q
        target_eval_pred = y_bar[vox_idx][None, :] + target_eval_u @ coef_q
        residual = Y[target_indices[:, None], vox_idx] - target_fit_pred

        hat_feature = (
            eigvecs[target_indices] * eigvecs[target_indices]
        ) @ (eigvals * inv)
        hat = np.clip(target_weight / weight_total + hat_feature, 0.0, 0.999999)

        dual_cross = np.sum(
            target_eval_u * inv[None, :] * target_fit_cross_u.T,
            axis=1,
        )
        cross = target_weight / weight_total + target_weight * (
            (target_eval_dot_fit - dual_cross) / float(alpha)
        )

        target_loso[:, vox_idx] = (
            target_eval_pred - cross[:, None] * residual / (1.0 - hat[:, None])
        ).astype(np.float32)
        vicco_pred[:, vox_idx] = (
            y_bar[vox_idx][None, :] + vicco_eval_u @ coef_q
        ).astype(np.float32)
        target_hat[:, vox_idx] = hat[:, None].astype(np.float32)

    return target_loso, vicco_pred, target_hat, time.time() - t0


def canonical_lookup(canonical: pd.DataFrame) -> dict[tuple[str, str, str, float], pd.Series]:
    out = {}
    for row in canonical.itertuples(index=False):
        out[(row.subject, row.model, row.eval_target, float(row.target_weight))] = row
    return out


def score_one_selection(
    sel: Selection,
    *,
    model_set: str,
    weight: float,
    fit_scope: str,
    n_vicco_boot: int,
    canonical_rows: dict[tuple[str, str, str, float], pd.Series],
) -> list[dict]:
    t0 = time.time()
    scope_cfg = FIT_SCOPE_CONFIG[fit_scope]
    cstim_data = SCORER.load_cstim_subject_data(sel.subject)

    X_target_all = SCORER.load_feature(SCORER.feature_path(sel.model, model_set), sel.layer)
    target_file_idx = cstim_data["group_file_idx"][model_set]
    target_brain_idx = cstim_data["group_brain_idx"][model_set]
    X_target_raw = X_target_all[target_file_idx].astype(np.float32)
    Y_target_raw = cstim_data["betas_hlvis"][:, target_brain_idx].T.astype(np.float32)

    X_vicco_all = SCORER.load_feature(SCORER.feature_path(sel.model, BASELINE_SET), sel.layer)
    vicco_file_idx = cstim_data["group_file_idx"][BASELINE_SET]
    vicco_brain_idx = cstim_data["group_brain_idx"][BASELINE_SET]
    X_vicco_raw = X_vicco_all[vicco_file_idx].astype(np.float32)
    Y_vicco_raw = cstim_data["betas_hlvis"][:, vicco_brain_idx].T.astype(np.float32)

    if fit_scope == "deepvision-plus-cstim":
        X_dv_raw = SCORER.load_feature(
            SCORER.dv_feature_path(sel.subject, sel.model), sel.layer
        )
        Y_dv_raw, _hlvis = SCORER.load_deepvision_responses(sel.subject)
        X_train_raw = np.vstack([X_dv_raw, X_target_raw]).astype(np.float32)
        Y_train_raw = np.vstack([Y_dv_raw, Y_target_raw]).astype(np.float32)
        sample_weight = np.concatenate(
            [
                np.ones(X_dv_raw.shape[0], dtype=np.float64),
                np.full(X_target_raw.shape[0], float(weight), dtype=np.float64),
            ]
        )
        target_indices = np.arange(X_dv_raw.shape[0], X_train_raw.shape[0], dtype=int)
        n_deepvision_train = int(X_dv_raw.shape[0])
        fit_sample_weight_note = f"deepvision=1,cstim={float(weight):g}"
    elif fit_scope == "cstim-only":
        X_train_raw = X_target_raw.astype(np.float32, copy=False)
        Y_train_raw = Y_target_raw.astype(np.float32, copy=False)
        sample_weight = np.ones(X_target_raw.shape[0], dtype=np.float64)
        target_indices = np.arange(X_target_raw.shape[0], dtype=int)
        n_deepvision_train = 0
        fit_sample_weight_note = "cstim=1"
    else:
        raise ValueError(f"Unknown fit_scope={fit_scope!r}")

    X_train_z, feature_mean, feature_scale = weighted_zscore(
        X_train_raw, sample_weight, eps=FEATURE_EPS
    )
    Y_train_z, response_mean, response_scale = weighted_zscore(
        Y_train_raw, sample_weight, eps=RESPONSE_EPS
    )
    X_target_z = ((X_target_raw - feature_mean[None, :]) / feature_scale[None, :]).astype(
        np.float32
    )
    X_vicco_z = ((X_vicco_raw - feature_mean[None, :]) / feature_scale[None, :]).astype(
        np.float32
    )
    X_target_eval_z = layer_sweep_eval_design(X_target_z, feature_mean, feature_scale)
    X_vicco_eval_z = layer_sweep_eval_design(X_vicco_z, feature_mean, feature_scale)

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

    cstim_score = rsa_spearman(target_pred, Y_target_raw)
    vicco_boot, vicco_brain_ranks, vicco_sample_size = SCORER.load_or_compute_vicco_bootstrap(
        subject=sel.subject,
        betas_hlvis=cstim_data["betas_hlvis"],
        vicco_brain_idx=vicco_brain_idx,
        n_bootstrap=n_vicco_boot,
    )
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
    for eval_target, score, sem, n_scored, n_boot in [
        ("cstim_loso", cstim_score, np.nan, X_target_raw.shape[0], 1),
        ("vicco_heldout", vicco_score, vicco_sem, vicco_n_scored, vicco_n_boot),
    ]:
        canonical = canonical_rows.get((sel.subject, sel.model, eval_target, float(weight)))
        canonical_score = float(canonical.mrsa_loso) if canonical is not None else np.nan
        original_score = (
            float(canonical.original_best_shared_mrsa) if canonical is not None else np.nan
        )
        rows.append(
            {
                "subject": sel.subject,
                "model": sel.model,
                "display_name": sel.display_name,
                "selected_layer": sel.layer,
                "model_set": model_set,
                "target_weight": float(weight),
                "eval_target": eval_target,
                "mrsa_loso": float(score),
                "mrsa_loso_sem": float(sem) if np.isfinite(sem) else np.nan,
                "canonical_fixed_dv_stats_mrsa": canonical_score,
                "delta_vs_canonical_fixed_dv_stats": float(score) - canonical_score
                if np.isfinite(canonical_score)
                else np.nan,
                "original_best_shared_mrsa": original_score,
                "delta_vs_original": float(score) - original_score
                if np.isfinite(original_score)
                else np.nan,
                "n_deepvision_train": n_deepvision_train,
                "n_target_train": int(X_target_raw.shape[0]),
                "n_stimuli_scored": int(n_scored),
                "n_score_bootstrap": int(n_boot),
                "score_sample_size": int(n_scored),
                "feature_dim_analysis": int(SRP_TARGET_DIM),
                "feature_protocol": FEATURE_PROTOCOL,
                "fit_scope": fit_scope,
                "fit_scope_label": scope_cfg["label"],
                "fit_sample_weight_note": fit_sample_weight_note,
                "alpha_rule": scope_cfg["alpha_rule"],
                "alpha_median": float(np.median(alphas)),
                "alpha_mean": float(np.mean(alphas)),
                "alpha_std": float(np.std(alphas)),
                "n_alpha_unique": int(len(np.unique(alphas))),
                "hat_diag_mean": float(np.mean(target_hat)) if eval_target == "cstim_loso" else np.nan,
                "hat_diag_max": float(np.max(target_hat)) if eval_target == "cstim_loso" else np.nan,
                "target_zscore_reference": scope_cfg["zscore_reference"],
                "feature_zscore_reference": scope_cfg["zscore_reference"],
                "prediction_protocol": "layer_sweep_stream_predict_v1",
                "training_target_scope": scope_cfg["training_target_scope"].format(
                    model_set=model_set
                ),
                "runtime_seconds_alpha": float(alpha_runtime),
                "runtime_seconds_prediction": float(pred_runtime),
                "runtime_seconds_subject_model": float(time.time() - t0),
            }
        )
    return rows


def write_summary(scores: pd.DataFrame, summary_csv: Path) -> None:
    rows = []
    group_cols = ["model_set", "target_weight", "eval_target"]
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


def completed_selection_keys(
    existing: pd.DataFrame,
    *,
    model_set: str,
    weight: float,
    n_vicco_boot: int,
) -> set[tuple[str, str]]:
    if existing.empty:
        return set()
    required = {
        "subject",
        "model",
        "model_set",
        "target_weight",
        "eval_target",
        "feature_dim_analysis",
        "feature_protocol",
        "n_score_bootstrap",
    }
    if required.difference(existing.columns):
        return set()
    block = existing[
        existing["model_set"].eq(model_set)
        & np.isclose(existing["target_weight"].astype(float), weight)
        & existing["feature_dim_analysis"].eq(SRP_TARGET_DIM)
        & existing["feature_protocol"].eq(FEATURE_PROTOCOL)
    ].copy()
    completed: set[tuple[str, str]] = set()
    for (subject, model), rows in block.groupby(["subject", "model"]):
        eval_targets = set(rows["eval_target"])
        if not {"cstim_loso", "vicco_heldout"}.issubset(eval_targets):
            continue
        vicco = rows[rows["eval_target"].eq("vicco_heldout")]
        if not vicco["n_score_bootstrap"].eq(int(n_vicco_boot)).any():
            continue
        completed.add((str(subject), str(model)))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-set", default=DEFAULT_MODEL_SET, choices=CSTIM_SETS)
    parser.add_argument(
        "--fit-scope",
        default=DEFAULT_FIT_SCOPE,
        choices=sorted(FIT_SCOPE_CONFIG),
        help="Training data used by the sensitivity fit.",
    )
    parser.add_argument(
        "--weight",
        default=DEFAULT_WEIGHT,
        help="Target weight to test, or 'auto' for the best mean canonical CSTIM delta.",
    )
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-selections", type=int, default=None)
    parser.add_argument("--n-vicco-boot", type=int, default=DEFAULT_N_VICCO_BOOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    score_csv, summary_csv, meta_json = output_paths(args.fit_scope)
    scope_cfg = FIT_SCOPE_CONFIG[args.fit_scope]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if score_csv.exists() and not args.overwrite and not args.resume:
        print(f"[cached] {score_csv} exists; use --overwrite to recompute", flush=True)
        return

    canonical = pd.read_csv(CANONICAL_SCORE_CSV)
    if args.weight == "auto":
        weight = choose_auto_weight(canonical, args.model_set)
    else:
        parsed = parse_weights(args.weight)
        if len(parsed) != 1:
            raise ValueError("--weight expects one value for this sensitivity script")
        weight = float(parsed[0])

    selections = select_rows(
        model_set=args.model_set,
        subject=args.subject,
        models=args.models,
        max_selections=args.max_selections,
    )
    canonical = canonical[
        canonical["model_set"].eq(args.model_set)
        & np.isclose(canonical["target_weight"].astype(float), weight)
        & canonical["eval_target"].isin(["cstim_loso", "vicco_heldout"])
    ].copy()
    canonical_rows = canonical_lookup(canonical)

    rows = []
    completed: set[tuple[str, str]] = set()
    if args.resume and score_csv.exists():
        existing = pd.read_csv(score_csv)
        completed = completed_selection_keys(
            existing,
            model_set=args.model_set,
            weight=weight,
            n_vicco_boot=args.n_vicco_boot,
        )
        if completed:
            keep_mask = (
                existing["model_set"].eq(args.model_set)
                & np.isclose(existing["target_weight"].astype(float), weight)
                & existing.apply(
                    lambda row: (str(row["subject"]), str(row["model"])) in completed,
                    axis=1,
                )
            )
            rows = existing[keep_mask].to_dict("records")
        print(
            f"[resume] keeping {len(rows)} rows from {len(completed)} complete selections",
            flush=True,
        )

    print(
        f"{scope_cfg['label']} sensitivity: model_set={args.model_set} "
        f"canonical_comparison_weight={weight:g} selections={len(selections)} "
        f"n_vicco_boot={args.n_vicco_boot}",
        flush=True,
    )

    for idx, row in enumerate(selections.itertuples(index=False), start=1):
        sel = Selection(
            subject=row.subject,
            model=row.model,
            display_name=row.display_name,
            layer=row.layer,
        )
        if (sel.subject, sel.model) in completed:
            continue
        print(
            f"[{idx:03d}/{len(selections):03d}] {sel.subject} {sel.model} "
            f"layer={sel.layer}",
            flush=True,
        )
        new_rows = score_one_selection(
            sel,
            model_set=args.model_set,
            weight=weight,
            fit_scope=args.fit_scope,
            n_vicco_boot=args.n_vicco_boot,
            canonical_rows=canonical_rows,
        )
        rows.extend(new_rows)
        new_scores = pd.DataFrame(new_rows)
        cstim = new_scores[new_scores["eval_target"].eq("cstim_loso")].iloc[0]
        vicco = new_scores[new_scores["eval_target"].eq("vicco_heldout")].iloc[0]
        print(
            "  "
            f"cstim={cstim.mrsa_loso:.4f} "
            f"delta_canonical={cstim.delta_vs_canonical_fixed_dv_stats:+.4f}; "
            f"baseline={vicco.mrsa_loso:.4f} "
            f"delta_canonical={vicco.delta_vs_canonical_fixed_dv_stats:+.4f}; "
            f"runtime={cstim.runtime_seconds_subject_model / 60.0:.1f} min",
            flush=True,
        )
        scores = pd.DataFrame(rows)
        atomic_write_csv(scores, score_csv)
        write_summary(scores, summary_csv)
        print(f"  wrote checkpoint rows={len(rows)} -> {score_csv}", flush=True)

    scores = pd.DataFrame(rows)
    atomic_write_csv(scores, score_csv)
    write_summary(scores, summary_csv)
    meta = {
        "model_set": args.model_set,
        "fit_scope": args.fit_scope,
        "fit_scope_label": scope_cfg["label"],
        "target_weight": float(weight),
        "weight_selection": args.weight,
        "n_selections": int(len(selections)),
        "n_vicco_boot": int(args.n_vicco_boot),
        "feature_dim_analysis": int(SRP_TARGET_DIM),
        "feature_protocol": FEATURE_PROTOCOL,
        "alpha_grid": [float(x) for x in ALPHA_GRID],
        "alpha_rule": scope_cfg["metadata_alpha_rule"],
        "preprocessing": scope_cfg["metadata_preprocessing"],
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
