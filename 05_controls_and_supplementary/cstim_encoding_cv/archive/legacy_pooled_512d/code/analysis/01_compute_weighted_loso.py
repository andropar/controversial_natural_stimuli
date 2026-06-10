#!/usr/bin/env python3
"""Compute weighted CSTIM-included encoding predictions with analytic LOSO.

The dense layer is fixed to the layer selected on DeepVision shared images in
the layer-sweep follow-up. For each available subject/model selected layer,
the script refits a ridge encoding model on DeepVision unique plus all five
CSTIM sets, then uses the ridge hat-matrix identity to get leave-one-CSTIM-out
predictions without fitting 500 separate models.

The default run is feature-cache-only: it uses exact selected-layer features
from a local targeted cache first, then falls back to the source layer-sweep
cache. Old dense-sweep alpha files are used when present; otherwise a global
median alpha from the available dense-sweep fits is used and recorded per row.
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

import config
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
DV_CACHE_ROOT = SHARE_ROOT / "01_brain_model_alignment" / "cache_or_heavy" / "brain_data"
BRAIN_CACHE_ROOT = SHARE_ROOT / "01_brain_model_alignment" / "cache_or_heavy" / "brain_data_cache" / "data"
SCORE_CSV = RESULTS_DIR / "cstim_loso_weighted_scores.csv"
AUDIT_CSV = RESULTS_DIR / "cached_selection_audit.csv"
RUN_META_JSON = RESULTS_DIR / "run_metadata.json"


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
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("No cstim weights provided")
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
        ["subject", "model", "eval_target", "eval_model_set", "mrsa_mean", "selected_layer"]
    ].rename(
        columns={
            "eval_model_set": "model_set",
            "mrsa_mean": "original_best_shared_mrsa",
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
                feature_reasons.append(f"missing_cstim_feature:{cset}")
        if not npz_has_key_any(deepvision_feature_paths(row.subject, row.model), row.layer):
            feature_reasons.append("missing_deepvision_unique_feature")
        alpha_ready = encoding_path(row.subject, row.model, row.layer).exists()
        reasons = list(feature_reasons)
        if not alpha_ready:
            reasons.append("missing_encoding_alpha_cache")
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

    rng = np.random.default_rng(stable_seed(model, layer, "analysis_projection"))
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


def fit_weighted_loso_predictions(
    X_dv: np.ndarray,
    Y_dv: np.ndarray,
    X_cstim: np.ndarray,
    Y_cstim: np.ndarray,
    *,
    alpha: float,
    cstim_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit weighted ridge and return analytic leave-one-CSTIM-out predictions."""
    n_dv = X_dv.shape[0]
    n_cstim = X_cstim.shape[0]
    X_all = np.vstack([X_dv, X_cstim]).astype(np.float64, copy=False)
    Y_all = np.vstack([Y_dv, Y_cstim]).astype(np.float64, copy=False)

    design = np.empty((X_all.shape[0], X_all.shape[1] + 1), dtype=np.float64)
    design[:, :-1] = X_all
    design[:, -1] = 1.0

    weights = np.ones(X_all.shape[0], dtype=np.float64)
    weights[n_dv:] = float(cstim_weight)
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
        Xc = design[n_dv:]
        yhat = Xc @ coef
        if cstim_weight > 0:
            inv_x = scipy.linalg.cho_solve(c_factor, Xc.T, check_finite=False)
            h = cstim_weight * np.einsum("ij,ji->i", Xc, inv_x, optimize=True)
        else:
            h = np.zeros(n_cstim, dtype=np.float64)
    except scipy.linalg.LinAlgError:
        coef = np.linalg.pinv(gram, rcond=1e-6) @ rhs
        Xc = design[n_dv:]
        yhat = Xc @ coef
        if cstim_weight > 0:
            inv_x = np.linalg.pinv(gram, rcond=1e-6) @ Xc.T
            h = cstim_weight * np.einsum("ij,ji->i", Xc, inv_x, optimize=True)
        else:
            h = np.zeros(n_cstim, dtype=np.float64)

    denom = np.maximum(1.0 - h, 1e-4)
    loo = (yhat - h[:, None] * Y_cstim) / denom[:, None]
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
                "median_dense_sweep_alpha",
            )
    return float(fallback_alpha), float("nan"), float("nan"), "global_median_dense_sweep_alpha"


def should_keep_model_for_set(model: str, model_set: str) -> bool:
    return model in set(config.MODEL_SETS[model_set])


def compute_one_selection(
    sel: Selection,
    *,
    cstim_weights: list[float],
    analysis_dim: int,
    fallback_alpha: float,
    original_ref: pd.DataFrame,
    score_membership_only: bool,
) -> list[dict]:
    t0 = time.time()
    X_dv = load_feature_from_paths(deepvision_feature_paths(sel.subject, sel.model), sel.layer)
    cstim_data = load_cstim_subject_data(sel.subject)

    cstim_feature_blocks = []
    cstim_response_blocks = []
    cstim_feature_dims = {}
    slices = {}
    start = 0
    for group in CSTIM_SETS:
        X_group_all = load_feature_from_paths(cstim_feature_paths(sel.model, group), sel.layer)
        cstim_feature_dims[group] = int(X_group_all.shape[1])
        file_idx = cstim_data["group_file_idx"][group]
        brain_idx = cstim_data["group_brain_idx"][group]
        X_group = X_group_all[file_idx]
        Y_group = cstim_data["betas_hlvis"][:, brain_idx].T.astype(np.float32)
        cstim_feature_blocks.append(X_group)
        cstim_response_blocks.append(Y_group)
        slices[group] = slice(start, start + len(file_idx))
        start += len(file_idx)

    X_cstim = np.vstack(cstim_feature_blocks).astype(np.float32)
    Y_cstim_raw = np.vstack(cstim_response_blocks).astype(np.float32)
    X_vicco_all = load_feature_from_paths(cstim_feature_paths(sel.model, BASELINE_SET), sel.layer)
    vicco_feature_dim = int(X_vicco_all.shape[1])
    vicco_file_idx = cstim_data["group_file_idx"][BASELINE_SET]
    vicco_brain_idx = cstim_data["group_brain_idx"][BASELINE_SET]
    X_vicco = X_vicco_all[vicco_file_idx].astype(np.float32)
    Y_vicco_raw = cstim_data["betas_hlvis"][:, vicco_brain_idx].T.astype(np.float32)
    Y_dv_raw, _hlvis = load_deepvision_responses(sel.subject)
    if X_dv.shape[0] != Y_dv_raw.shape[0]:
        raise ValueError(
            f"DeepVision feature/response length mismatch for {sel.subject}/{sel.model}: "
            f"{X_dv.shape[0]} vs {Y_dv_raw.shape[0]}"
        )

    [X_dv, X_cstim, X_vicco], final_dim, projected = project_if_needed(
        [X_dv, X_cstim, X_vicco], model=sel.model, layer=sel.layer, target_dim=analysis_dim
    )
    X_dv_z, [X_cstim_z, X_vicco_z], _x_mean, _x_scale = standardize_from_reference(
        X_dv, X_cstim, X_vicco
    )
    Y_dv_z, [Y_cstim_z], _y_mean, _y_scale = standardize_from_reference(
        Y_dv_raw, Y_cstim_raw
    )
    alpha_median, alpha_mean, alpha_std, alpha_rule = load_alpha(
        sel.subject,
        sel.model,
        sel.layer,
        fallback_alpha=fallback_alpha,
    )

    ref_rows = original_ref[
        original_ref["subject"].eq(sel.subject)
        & original_ref["model"].eq(sel.model)
        & original_ref["layer"].eq(sel.layer)
    ]
    ref_map = {
        (row.eval_target, row.model_set): float(row.original_best_shared_mrsa)
        for row in ref_rows.itertuples(index=False)
    }

    out = []
    for weight in cstim_weights:
        pred_loso, hat, coef = fit_weighted_loso_predictions(
            X_dv_z,
            Y_dv_z,
            X_cstim_z,
            Y_cstim_z,
            alpha=alpha_median,
            cstim_weight=weight,
        )
        pred_vicco = predict_with_coef(X_vicco_z, coef)
        vicco_score = rsa_spearman(pred_vicco, Y_vicco_raw)
        vicco_ref = ref_map.get((BASELINE_SET, BASELINE_SET), np.nan)
        for group in CSTIM_SETS:
            if score_membership_only and not should_keep_model_for_set(sel.model, group):
                continue
            sl = slices[group]
            score = rsa_spearman(pred_loso[sl], Y_cstim_raw[sl])
            original_score = ref_map.get(("cstim", group), np.nan)
            out.append(
                {
                    "subject": sel.subject,
                    "model": sel.model,
                    "display_name": sel.display_name,
                    "selected_layer": sel.layer,
                    "model_set": group,
                    "eval_target": "cstim_loso",
                    "stimulus_type": "controversial",
                    "cstim_weight": float(weight),
                    "mrsa_loso": score,
                    "original_best_shared_mrsa": original_score,
                    "delta_vs_original": score - original_score,
                    "n_deepvision_train": int(X_dv_z.shape[0]),
                    "n_cstim_total": int(X_cstim_z.shape[0]),
                    "n_stimuli_scored": int(sl.stop - sl.start),
                    "feature_dim_original": cstim_feature_dims[group],
                    "feature_dim_analysis": int(final_dim),
                    "feature_projected": bool(projected),
                    "alpha_rule": alpha_rule,
                    "alpha_median": alpha_median,
                    "alpha_mean": alpha_mean,
                    "alpha_std": alpha_std,
                    "hat_diag_mean": float(np.mean(hat[sl])),
                    "hat_diag_max": float(np.max(hat[sl])),
                    "target_zscore_reference": "deepvision_unique",
                    "feature_zscore_reference": "deepvision_unique",
                    "training_cstim_scope": "all_cstim_sets_except_heldout_via_analytic_loso",
                    "runtime_seconds_subject_model_weightless": np.nan,
                }
            )
            out.append(
                {
                    "subject": sel.subject,
                    "model": sel.model,
                    "display_name": sel.display_name,
                    "selected_layer": sel.layer,
                    "model_set": group,
                    "eval_target": BASELINE_SET,
                    "stimulus_type": "baseline",
                    "cstim_weight": float(weight),
                    "mrsa_loso": vicco_score,
                    "original_best_shared_mrsa": vicco_ref,
                    "delta_vs_original": vicco_score - vicco_ref,
                    "n_deepvision_train": int(X_dv_z.shape[0]),
                    "n_cstim_total": int(X_cstim_z.shape[0]),
                    "n_stimuli_scored": int(X_vicco_z.shape[0]),
                    "feature_dim_original": vicco_feature_dim,
                    "feature_dim_analysis": int(final_dim),
                    "feature_projected": bool(projected),
                    "alpha_rule": alpha_rule,
                    "alpha_median": alpha_median,
                    "alpha_mean": alpha_mean,
                    "alpha_std": alpha_std,
                    "hat_diag_mean": np.nan,
                    "hat_diag_max": np.nan,
                    "target_zscore_reference": "deepvision_unique",
                    "feature_zscore_reference": "deepvision_unique",
                    "training_cstim_scope": "all_cstim_sets_weighted_vicco_held_out",
                    "runtime_seconds_subject_model_weightless": np.nan,
                }
            )
    elapsed = time.time() - t0
    for row in out:
        row["runtime_seconds_subject_model_weightless"] = elapsed
    return out


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--analysis-dim", type=int, default=1024)
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--score-all-model-sets", action="store_true",
                        help="Score every cached model on every cstim set. Default only "
                             "keeps rows where the model belongs to the plotted model set.")
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
        print(
            f"[{idx:03d}/{total:03d}] {sel.subject} {sel.model} layer={sel.layer}",
            flush=True,
        )
        rows.extend(
            compute_one_selection(
                sel,
                cstim_weights=weights,
                analysis_dim=args.analysis_dim,
                fallback_alpha=fallback_alpha,
                original_ref=original_ref,
                score_membership_only=not args.score_all_model_sets,
            )
        )
        if idx % 5 == 0:
            atomic_write_csv(pd.DataFrame(rows), SCORE_CSV)
            print(f"  wrote checkpoint rows={len(rows)} -> {SCORE_CSV}", flush=True)

    scores = pd.DataFrame(rows)
    atomic_write_csv(scores, SCORE_CSV)
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
        "fallback_alpha_rule": "median of available per-selection dense-sweep median alphas",
        "n_cache_ready_selections": int(len(ready)),
        "n_total_selections_considered": int(len(audit)),
        "score_membership_only": not args.score_all_model_sets,
        "cstim_eval_target": "cstim_loso",
        "baseline_eval_target": BASELINE_SET,
        "baseline_prediction_scheme": "same_cstim_weighted_refit_vicco_held_out",
        "python": os.sys.executable,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    RUN_META_JSON.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(scores)} rows -> {SCORE_CSV}", flush=True)
    print(f"Wrote audit -> {AUDIT_CSV}", flush=True)


if __name__ == "__main__":
    main()
