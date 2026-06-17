#!/usr/bin/env python3
"""Compute fitted teacher/student model-recovery curves.

This mirrors the noisy-by-clean recovery analysis, but replaces direct
RDM-to-RDM matching with a teacher/student refit:

1. Pick one model as the teacher.
2. Add calibrated Gaussian noise to the teacher target on a train/val/test split.
3. Refit every candidate model from its raw features to that noisy teacher target
   with ridge regression.
4. Score held-out prediction and recover the model with the best score.

The target is raw model features for the raw track and predicted brain responses
for encoding tracks. Candidate features are always raw model features.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[4]
ANALYSIS_DIR = (
    ROOT
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "code"
    / "analysis"
)
SRC_DIR = ROOT / "src"
for path in (SRC_DIR, ANALYSIS_DIR):
    sys.path.insert(0, str(path))

from cstims.paper import config as paper_config  # noqa: E402
import utils as eval_utils  # noqa: E402


MODEL_SET_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
DEFAULT_RESULTS = (
    ROOT
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "final_stimuli_recovery"
    / "teacher_student"
    / "results"
)
DEFAULT_RANDOM_FEATURE_DIR = ROOT / "shared" / "cache_or_heavy" / "natural_pool_subset_10k"
SELECTION_ROOT = ROOT / "00_stimulus_selection" / "results" / "selected_stimuli"
ENV_CONFIG_ROOT = ROOT / "00_stimulus_selection" / "resources" / "configs" / "paths"
MODEL_LIST_CSV = ROOT / "00_stimulus_selection" / "resources" / "model_list.csv"
ORIENTATION = "teacher_student_fitted"


def load_disc_module():
    path = ANALYSIS_DIR / "02_compute_discriminability.py"
    spec = importlib.util.spec_from_file_location("selection_eval_discriminability", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


disc = load_disc_module()


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


def load_repo_env_paths(env: str) -> dict[str, Any]:
    config_path = ENV_CONFIG_ROOT / f"{env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Environment config not found: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f) or {}

    paths = dict(config.get("paths", {}))
    local_model_csv = MODEL_LIST_CSV
    if local_model_csv.exists():
        paths["model_list_csv"] = str(local_model_csv)
    paths["output_base"] = str(SELECTION_ROOT)
    return paths


def apply_env_paths(payload: dict, env: str | None) -> dict:
    if not env:
        return payload

    env_paths = load_repo_env_paths(env)
    payload = dict(payload)
    config = dict(payload.get("config", {}))
    old_paths = dict(config.get("paths", {}))
    eval_utils._warn_path_divergence(old_paths, env_paths, env)
    config["paths"] = {**old_paths, **env_paths}
    payload["config"] = config
    print(f"Using paths from env={env}: {ENV_CONFIG_ROOT / f'{env}.yaml'}")
    return payload


def _filter_model_dict(data: Any, keep_models: list[str]) -> Any:
    if not isinstance(data, dict):
        return data
    return {model: data[model] for model in keep_models if model in data}


def filter_payload_to_models(payload: dict, keep_models: list[str]) -> dict:
    payload = dict(payload)
    payload["model_names"] = list(keep_models)

    for key in [
        "selected_features_raw",
        "greedy_features_raw",
        "best_raw_combined_features_raw",
        "selected_features",
    ]:
        if key in payload:
            payload[key] = _filter_model_dict(payload[key], keep_models)

    for key in [
        "selected_features_by_view",
        "selected_features_by_encoding",
        "greedy_features_by_encoding",
        "best_raw_combined_features_by_encoding",
    ]:
        if isinstance(payload.get(key), dict):
            payload[key] = {
                track: _filter_model_dict(features, keep_models)
                for track, features in payload[key].items()
            }

    if isinstance(payload.get("var_noise_by_model"), dict):
        payload["var_noise_by_model"] = {
            track: _filter_model_dict(noise_by_model, keep_models)
            for track, noise_by_model in payload["var_noise_by_model"].items()
        }

    return payload


def _load_npz_feature_array(path: Path, n_images: int | None = None) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        if "features" in z.files:
            arr = z["features"]
        else:
            candidates = [
                key
                for key in z.files
                if not key.startswith("_") and getattr(z[key], "ndim", 0) >= 2
            ]
            if not candidates:
                raise ValueError(f"No feature array found in {path}")
            arr = z[candidates[0]]
        arr = np.asarray(arr, dtype=np.float32)
    if n_images is not None:
        arr = arr[: min(int(n_images), arr.shape[0])]
    return arr


def available_random_models(random_feature_dir: Path, model_names: list[str]) -> list[str]:
    return [model for model in model_names if (random_feature_dir / f"{model}.npz").exists()]


def load_random_raw_features(
    payload: dict,
    model_names: list[str],
    n_random: int,
    random_feature_dir: Path | None,
) -> dict[str, np.ndarray]:
    if random_feature_dir is None:
        return {
            model: np.asarray(arr, dtype=np.float32)
            for model, arr in eval_utils._load_random_identity(
                payload, model_names, n_random, "raw"
            ).items()
        }

    out = {}
    for model in model_names:
        path = random_feature_dir / f"{model}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing random feature cache for {model}: {path}")
        out[model] = _load_npz_feature_array(path, n_random)
    print(
        f"  [DEBUG] Loaded local random raw features: {len(out)} models, "
        f"{next(iter(out.values())).shape[0]} samples"
    )
    return out


def selected_raw_from_payload(
    payload: dict,
    model_names: list[str],
    selection_variant: str,
) -> dict[str, np.ndarray]:
    selected = eval_utils._load_selected_identity(
        payload,
        "raw",
        torch.device("cpu"),
        selection_variant,
    )
    out = {}
    for model in model_names:
        arr = selected[model]
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        out[model] = np.asarray(arr, dtype=np.float32)
    return out


def sanitize_layer_name(layer: str | int) -> str:
    return (
        str(layer)
        .strip()
        .replace(".", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def load_model_layers(model_list_csv: Path) -> dict[str, str]:
    table = pd.read_csv(model_list_csv)
    return dict(zip(table["model"], table["layer"]))


def load_encoding_params(
    encoding_root: Path,
    model_list_csv: Path,
    model_names: list[str],
    target_track: str,
    *,
    roi_subset: str | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    layers = load_model_layers(model_list_csv)
    params = {}
    roi_mask = None
    for model in model_names:
        if model not in layers:
            raise KeyError(f"Model not in {model_list_csv}: {model}")
        layer_safe = sanitize_layer_name(layers[model])
        path = encoding_root / f"{target_track}_{model}.layer{layer_safe}" / "encoding_model.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing encoding model: {path}")
        with np.load(path) as z:
            weights = np.asarray(z["weights"], dtype=np.float32)
            bias = np.asarray(z["intercept"], dtype=np.float32)
            if roi_subset:
                roi_key = f"roi_{roi_subset}"
                if roi_key in z:
                    if roi_mask is None:
                        roi_mask = np.asarray(z[roi_key], dtype=bool)
                    weights = weights[:, roi_mask]
                    bias = bias[roi_mask]
        params[model] = (weights, bias)
    return params


def encode_raw_features(
    raw_features: dict[str, np.ndarray],
    encoding_params: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    encoded = {}
    use_cuda = device.type == "cuda"
    for model, features in raw_features.items():
        weights_np, bias_np = encoding_params[model]
        chunks = []
        with torch.no_grad():
            weights = torch.as_tensor(weights_np, device=device, dtype=torch.float32)
            bias = torch.as_tensor(bias_np, device=device, dtype=torch.float32)
            for start in range(0, features.shape[0], batch_size):
                stop = min(start + batch_size, features.shape[0])
                feats = torch.as_tensor(
                    features[start:stop], device=device, dtype=torch.float32
                )
                chunks.append((feats @ weights + bias).detach().cpu().numpy())
                del feats
            del weights, bias
        if use_cuda:
            torch.cuda.empty_cache()
        encoded[model] = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    return encoded


def standardize_train_apply(
    train: np.ndarray,
    *others: np.ndarray,
    scale_by_sqrt_features: bool = False,
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    out = [(train - mean) / scale]
    out.extend((arr - mean) / scale for arr in others)
    if scale_by_sqrt_features:
        denom = math.sqrt(train.shape[1])
        out = [arr / denom for arr in out]
    return tuple(np.asarray(arr, dtype=np.float32) for arr in out)


def flat_corr_batch(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = pred.reshape(pred.shape[0], -1).astype(np.float64, copy=False)
    y = target.reshape(target.shape[0], -1).astype(np.float64, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    out = np.full(pred.shape[0], np.nan, dtype=np.float64)
    valid = denom > 0
    out[valid] = np.sum(x[valid] * y[valid], axis=1) / denom[valid]
    return out


def multiplier_to_noise_ceiling(noise_mult: float, nc_base: float) -> float:
    if noise_mult <= 0:
        return 1.0
    if nc_base <= 0 or nc_base >= 1:
        return nc_base
    term = noise_mult * noise_mult * (1.0 / (nc_base * nc_base) - 1.0)
    return float(1.0 / math.sqrt(1.0 + term))


def noise_std_from_multiplier(noise_mult: float, nc_base: float) -> float:
    if noise_mult <= 0 or nc_base <= 0 or nc_base >= 1:
        return 0.0
    return float(noise_mult * math.sqrt(1.0 / (nc_base * nc_base) - 1.0))


def ridge_prediction_operators(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    alphas: list[float],
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    kernel = (x_train @ x_train.T).astype(np.float64)
    k_val = (x_val @ x_train.T).astype(np.float64)
    k_test = (x_test @ x_train.T).astype(np.float64)
    eye = np.eye(kernel.shape[0], dtype=np.float64)
    operators = {}
    for alpha in alphas:
        inv = np.linalg.inv(kernel + float(alpha) * eye)
        operators[float(alpha)] = (
            np.asarray(k_val @ inv, dtype=np.float32),
            np.asarray(k_test @ inv, dtype=np.float32),
        )
    return operators


def split_indices(
    n_items: int,
    rng: np.random.Generator,
    train_n: int,
    val_n: int,
    test_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_n + val_n + test_n > n_items:
        raise ValueError(
            f"Split sizes ({train_n}+{val_n}+{test_n}) exceed n_items={n_items}"
        )
    perm = rng.permutation(n_items)
    train = perm[:train_n]
    val = perm[train_n : train_n + val_n]
    test = perm[train_n + val_n : train_n + val_n + test_n]
    return train, val, test


def _init_noise_stats() -> dict[str, Any]:
    return {
        "correct_sum": 0.0,
        "n_units": 0,
        "pairwise_dominance_sum": 0.0,
        "pairwise_margin_sum": 0.0,
        "n_pairwise": 0,
    }


def _score_teacher_batch(
    *,
    y_train_clean: np.ndarray,
    y_val_clean: np.ndarray,
    y_test_clean: np.ndarray,
    ridge_ops: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]],
    model_names: list[str],
    alphas: list[float],
    noise_std: float,
    n_noise_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_std > 0:
        y_train = y_train_clean[None, :, :] + rng.normal(
            0.0, noise_std, (n_noise_samples, *y_train_clean.shape)
        ).astype(np.float32)
        y_val = y_val_clean[None, :, :] + rng.normal(
            0.0, noise_std, (n_noise_samples, *y_val_clean.shape)
        ).astype(np.float32)
        y_test = y_test_clean[None, :, :] + rng.normal(
            0.0, noise_std, (n_noise_samples, *y_test_clean.shape)
        ).astype(np.float32)
    else:
        y_train = np.broadcast_to(
            y_train_clean[None, :, :], (n_noise_samples, *y_train_clean.shape)
        )
        y_val = np.broadcast_to(
            y_val_clean[None, :, :], (n_noise_samples, *y_val_clean.shape)
        )
        y_test = np.broadcast_to(
            y_test_clean[None, :, :], (n_noise_samples, *y_test_clean.shape)
        )

    scores = np.empty((n_noise_samples, len(model_names)), dtype=np.float32)
    for candidate_idx, candidate in enumerate(model_names):
        best_val = np.full(n_noise_samples, -np.inf, dtype=np.float64)
        best_test = np.full(n_noise_samples, np.nan, dtype=np.float64)
        for alpha in alphas:
            val_op, test_op = ridge_ops[candidate][float(alpha)]
            pred_val = np.einsum("vt,std->svd", val_op, y_train, optimize=True)
            val_scores = flat_corr_batch(pred_val, y_val)
            pred_test = np.einsum("ut,std->sud", test_op, y_train, optimize=True)
            test_scores = flat_corr_batch(pred_test, y_test)
            val_scores = np.nan_to_num(val_scores, nan=-np.inf)
            improve = val_scores > best_val
            best_val[improve] = val_scores[improve]
            best_test[improve] = test_scores[improve]
        scores[:, candidate_idx] = best_test.astype(np.float32)
    return scores


def run_subset_recovery_metrics(
    *,
    model_set: str,
    subset_type: str,
    subset_idx: int,
    track: dict,
    raw_by_model: dict[str, np.ndarray],
    target_by_model: dict[str, np.ndarray],
    model_names: list[str],
    n_splits: int,
    n_noise_samples: int,
    train_n: int,
    val_n: int,
    test_n: int,
    alphas: list[float],
    base_noise_ceiling: float,
    noise_mults: np.ndarray,
    seed: int,
    write_details: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    n_items = next(iter(raw_by_model.values())).shape[0]
    n_models = len(model_names)

    stats = {float(noise_mult): _init_noise_stats() for noise_mult in noise_mults}
    confusion = {
        float(noise_mult): np.zeros((n_models, n_models), dtype=np.int64)
        for noise_mult in noise_mults
    }
    detail_rows: list[dict[str, Any]] = []

    for split_idx in range(n_splits):
        train_idx, val_idx, test_idx = split_indices(n_items, rng, train_n, val_n, test_n)

        ridge_ops: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
        for candidate in model_names:
            x = raw_by_model[candidate]
            x_train, x_val, x_test = standardize_train_apply(
                x[train_idx],
                x[val_idx],
                x[test_idx],
                scale_by_sqrt_features=True,
            )
            ridge_ops[candidate] = ridge_prediction_operators(
                x_train, x_val, x_test, alphas
            )

        for teacher_idx, teacher in enumerate(model_names):
            clean_y = target_by_model[teacher]
            y_train_clean, y_val_clean, y_test_clean = standardize_train_apply(
                clean_y[train_idx],
                clean_y[val_idx],
                clean_y[test_idx],
            )
            offdiag = np.ones(n_models, dtype=bool)
            offdiag[teacher_idx] = False

            for noise_mult in noise_mults:
                noise_mult = float(noise_mult)
                noise_std = noise_std_from_multiplier(noise_mult, base_noise_ceiling)
                scores = _score_teacher_batch(
                    y_train_clean=y_train_clean,
                    y_val_clean=y_val_clean,
                    y_test_clean=y_test_clean,
                    ridge_ops=ridge_ops,
                    model_names=model_names,
                    alphas=alphas,
                    noise_std=noise_std,
                    n_noise_samples=n_noise_samples,
                    rng=rng,
                )
                scores = np.nan_to_num(scores, nan=-np.inf)
                recovered_idx = np.argmax(scores, axis=1)
                correct = recovered_idx == teacher_idx
                teacher_scores = scores[:, teacher_idx]
                margins = teacher_scores[:, None] - scores[:, offdiag]
                dominance = (
                    (margins > 0).astype(np.float64)
                    + 0.5 * np.isclose(margins, 0.0).astype(np.float64)
                )

                current = stats[noise_mult]
                current["correct_sum"] += float(correct.sum())
                current["n_units"] += int(correct.size)
                current["pairwise_dominance_sum"] += float(dominance.sum())
                current["pairwise_margin_sum"] += float(margins.sum())
                current["n_pairwise"] += int(margins.size)

                np.add.at(
                    confusion[noise_mult],
                    (np.full(correct.size, teacher_idx, dtype=np.int64), recovered_idx),
                    1,
                )

                if write_details:
                    for sample_idx, (rec_idx, is_correct, best_score, self_score) in enumerate(
                        zip(
                            recovered_idx,
                            correct,
                            scores[np.arange(scores.shape[0]), recovered_idx],
                            teacher_scores,
                        )
                    ):
                        detail_rows.append(
                            {
                                "model_set": model_set,
                                "subset_type": subset_type,
                                "subset_idx": subset_idx,
                                "track": track["name"],
                                "track_type": track.get("type", "identity"),
                                "split_idx": split_idx,
                                "noise_sample_idx": sample_idx,
                                "teacher_model": teacher,
                                "recovered_model": model_names[int(rec_idx)],
                                "recovered_correct": bool(is_correct),
                                "best_test_score": float(best_score),
                                "teacher_self_test_score": float(self_score),
                                "noise_mult": noise_mult,
                            }
                        )

    metric_rows = []
    pairwise_rows = []
    for noise_mult in noise_mults:
        noise_mult = float(noise_mult)
        current = stats[noise_mult]
        n_units = max(int(current["n_units"]), 1)
        n_pairwise = max(int(current["n_pairwise"]), 1)
        recovery_accuracy = current["correct_sum"] / n_units
        error_prob = 1.0 - recovery_accuracy
        pairwise_dominance = current["pairwise_dominance_sum"] / n_pairwise
        mean_margin = current["pairwise_margin_sum"] / n_pairwise
        row = {
            "model_set": model_set,
            "recovery_orientation": ORIENTATION,
            "track": track["name"],
            "track_type": track.get("type", "identity"),
            "metric": "fitted_prediction",
            "corr_type": "pearson_flat",
            "noise_mult": noise_mult,
            "noise_ceiling": multiplier_to_noise_ceiling(noise_mult, base_noise_ceiling),
            "subset_type": subset_type,
            "subset_idx": subset_idx,
            "recovery_accuracy": recovery_accuracy,
            "error_prob": error_prob,
            "error_prob_std": math.sqrt(error_prob * (1.0 - error_prob) / n_units),
            "error_prob_mc_std": math.sqrt(error_prob * (1.0 - error_prob) / n_units),
            "error_prob_mc_ci_lo": max(0.0, error_prob - 1.96 * math.sqrt(error_prob * (1.0 - error_prob) / n_units)),
            "error_prob_mc_ci_hi": min(1.0, error_prob + 1.96 * math.sqrt(error_prob * (1.0 - error_prob) / n_units)),
            "pairwise_dominance": pairwise_dominance,
            "pairwise_error_prob": 1.0 - pairwise_dominance,
            "mean_margin": mean_margin,
            "n_units": n_units,
            "n_pairwise": n_pairwise,
            "n_models": n_models,
            "n_splits": n_splits,
            "n_noise_samples": n_noise_samples,
            "base_noise_ceiling": base_noise_ceiling,
        }
        metric_rows.append(row)
        pairwise_rows.append(
            {
                **row,
                "pairwise_dominance_subset_std": np.nan,
                "pairwise_dominance_mc_std": math.sqrt(
                    pairwise_dominance * (1.0 - pairwise_dominance) / n_pairwise
                ),
                "pairwise_dominance_mc_ci_lo": np.nan,
                "pairwise_dominance_mc_ci_hi": np.nan,
                "mean_margin_subset_std": np.nan,
                "mean_margin_mc_std": np.nan,
                "mean_margin_mc_ci_lo": np.nan,
                "mean_margin_mc_ci_hi": np.nan,
                "random_feature_source": "",
            }
        )

    confusion_rows = []
    for noise_mult, matrix in confusion.items():
        for teacher_idx, teacher in enumerate(model_names):
            total = int(matrix[teacher_idx].sum())
            for recovered_idx, recovered in enumerate(model_names):
                count = int(matrix[teacher_idx, recovered_idx])
                confusion_rows.append(
                    {
                        "model_set": model_set,
                        "recovery_orientation": ORIENTATION,
                        "subset_type": subset_type,
                        "subset_idx": subset_idx,
                        "track": track["name"],
                        "track_type": track.get("type", "identity"),
                        "noise_mult": float(noise_mult),
                        "noise_ceiling": multiplier_to_noise_ceiling(
                            float(noise_mult), base_noise_ceiling
                        ),
                        "teacher_model": teacher,
                        "recovered_model": recovered,
                        "count": count,
                        "proportion": count / total if total else np.nan,
                    }
                )

    return metric_rows, pairwise_rows, confusion_rows + detail_rows


def aggregate_curve_rows(
    subset_rows: pd.DataFrame,
    *,
    random_feature_source: str,
) -> pd.DataFrame:
    rows = []
    keys = ["track", "track_type", "metric", "corr_type", "noise_mult", "noise_ceiling", "subset_type"]
    for group_key, group in subset_rows.groupby(keys, sort=False):
        track, track_type, metric, corr_type, noise_mult, noise_ceiling, subset_type = group_key
        if subset_type == "selected":
            row = group.iloc[0].to_dict()
            row["error_prob_std"] = np.nan
        else:
            row = {
                "model_set": group["model_set"].iloc[0],
                "recovery_orientation": ORIENTATION,
                "track": track,
                "track_type": track_type,
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": float(noise_mult),
                "noise_ceiling": float(noise_ceiling),
                "subset_type": subset_type,
                "subset_idx": -1,
                "recovery_accuracy": float(group["recovery_accuracy"].mean()),
                "error_prob": float(group["error_prob"].mean()),
                "error_prob_std": float(group["error_prob"].std(ddof=1))
                if len(group) > 1
                else np.nan,
                "error_prob_mc_std": float(group["error_prob_mc_std"].mean()),
                "error_prob_mc_ci_lo": float(group["error_prob_mc_ci_lo"].mean()),
                "error_prob_mc_ci_hi": float(group["error_prob_mc_ci_hi"].mean()),
                "pairwise_dominance": float(group["pairwise_dominance"].mean()),
                "pairwise_error_prob": float(group["pairwise_error_prob"].mean()),
                "mean_margin": float(group["mean_margin"].mean()),
                "n_units": int(group["n_units"].sum()),
                "n_pairwise": int(group["n_pairwise"].sum()),
                "n_models": int(group["n_models"].iloc[0]),
                "n_splits": int(group["n_splits"].iloc[0]),
                "n_noise_samples": int(group["n_noise_samples"].iloc[0]),
                "base_noise_ceiling": float(group["base_noise_ceiling"].iloc[0]),
            }
        row["random_feature_source"] = random_feature_source
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_pairwise_rows(
    subset_rows: pd.DataFrame,
    *,
    random_feature_source: str,
) -> pd.DataFrame:
    rows = []
    keys = ["track", "track_type", "metric", "corr_type", "noise_mult", "noise_ceiling", "subset_type"]
    for group_key, group in subset_rows.groupby(keys, sort=False):
        track, track_type, metric, corr_type, noise_mult, noise_ceiling, subset_type = group_key
        first = group.iloc[0]
        if subset_type == "selected":
            pairwise_subset_std = np.nan
            margin_subset_std = np.nan
        else:
            pairwise_subset_std = float(group["pairwise_dominance"].std(ddof=1)) if len(group) > 1 else np.nan
            margin_subset_std = float(group["mean_margin"].std(ddof=1)) if len(group) > 1 else np.nan
        rows.append(
            {
                "track": track,
                "track_type": track_type,
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": float(noise_mult),
                "noise_ceiling": float(noise_ceiling),
                "subset_type": subset_type,
                "pairwise_dominance": float(group["pairwise_dominance"].mean()),
                "pairwise_dominance_subset_std": pairwise_subset_std,
                "pairwise_dominance_mc_std": float(group["pairwise_dominance_mc_std"].mean()),
                "pairwise_dominance_mc_ci_lo": np.nan,
                "pairwise_dominance_mc_ci_hi": np.nan,
                "pairwise_error_prob": float(group["pairwise_error_prob"].mean()),
                "mean_margin": float(group["mean_margin"].mean()),
                "mean_margin_subset_std": margin_subset_std,
                "mean_margin_mc_std": np.nan,
                "mean_margin_mc_ci_lo": np.nan,
                "mean_margin_mc_ci_hi": np.nan,
                "model_set": first["model_set"],
                "recovery_orientation": ORIENTATION,
                "random_feature_source": random_feature_source,
                "n_models": int(first["n_models"]),
            }
        )
    return pd.DataFrame(rows)


def auc_rows_from_curves(
    curves: pd.DataFrame,
    pairwise_curves: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    auc_rows = []
    pairwise_auc_rows = []
    for track, track_df in curves.groupby("track", sort=False):
        selected = track_df[track_df["subset_type"] == "selected"].sort_values("noise_mult")
        random = track_df[track_df["subset_type"] == "random"].sort_values("noise_mult")
        if selected.empty or random.empty:
            continue

        selected_auc = disc.compute_auc(
            selected["noise_mult"].to_numpy(float),
            selected["error_prob"].to_numpy(float),
        )
        random_auc = disc.compute_auc(
            random["noise_mult"].to_numpy(float),
            random["error_prob"].to_numpy(float),
        )
        auc_rows.append(
            {
                "track": track,
                "model_set": selected["model_set"].iloc[0],
                "recovery_orientation": ORIENTATION,
                "random_feature_source": selected["random_feature_source"].iloc[0],
                "n_models": int(selected["n_models"].iloc[0]),
                "selected_auc": float(selected_auc),
                "selected_auc_mc_std": float(selected["error_prob_mc_std"].mean()),
                "selected_auc_mc_ci_lo": np.nan,
                "selected_auc_mc_ci_hi": np.nan,
                "random_auc_mean": float(random_auc),
                "random_auc_subset_std": float(random["error_prob_std"].mean()),
                "random_auc_subset_ci_lo": np.nan,
                "random_auc_subset_ci_hi": np.nan,
                "random_auc_mc_std": float(random["error_prob_mc_std"].mean()),
                "p_value_empirical": np.nan,
                "z_score": np.nan,
            }
        )

        pairwise_track = pairwise_curves[pairwise_curves["track"] == track]
        pairwise_selected = pairwise_track[
            pairwise_track["subset_type"] == "selected"
        ].sort_values("noise_mult")
        pairwise_random = pairwise_track[
            pairwise_track["subset_type"] == "random"
        ].sort_values("noise_mult")
        selected_dom_auc = disc.compute_auc(
            pairwise_selected["noise_mult"].to_numpy(float),
            pairwise_selected["pairwise_dominance"].to_numpy(float),
        )
        random_dom_auc = disc.compute_auc(
            pairwise_random["noise_mult"].to_numpy(float),
            pairwise_random["pairwise_dominance"].to_numpy(float),
        )
        selected_margin_auc = disc.compute_auc(
            pairwise_selected["noise_mult"].to_numpy(float),
            pairwise_selected["mean_margin"].to_numpy(float),
        )
        random_margin_auc = disc.compute_auc(
            pairwise_random["noise_mult"].to_numpy(float),
            pairwise_random["mean_margin"].to_numpy(float),
        )
        pairwise_auc_rows.append(
            {
                "track": track,
                "model_set": selected["model_set"].iloc[0],
                "recovery_orientation": ORIENTATION,
                "random_feature_source": selected["random_feature_source"].iloc[0],
                "n_models": int(selected["n_models"].iloc[0]),
                "selected_pairwise_dominance_auc": float(selected_dom_auc),
                "selected_pairwise_dominance_auc_mc_std": float(
                    pairwise_selected["pairwise_dominance_mc_std"].mean()
                ),
                "selected_pairwise_dominance_auc_mc_ci_lo": np.nan,
                "selected_pairwise_dominance_auc_mc_ci_hi": np.nan,
                "random_pairwise_dominance_auc_mean": float(random_dom_auc),
                "random_pairwise_dominance_auc_subset_std": float(
                    pairwise_random["pairwise_dominance_subset_std"].mean()
                ),
                "random_pairwise_dominance_auc_mc_std": float(
                    pairwise_random["pairwise_dominance_mc_std"].mean()
                ),
                "selected_mean_margin_auc": float(selected_margin_auc),
                "selected_mean_margin_auc_mc_std": np.nan,
                "selected_mean_margin_auc_mc_ci_lo": np.nan,
                "selected_mean_margin_auc_mc_ci_hi": np.nan,
                "random_mean_margin_auc_mean": float(random_margin_auc),
                "random_mean_margin_auc_subset_std": float(
                    pairwise_random["mean_margin_subset_std"].mean()
                ),
                "random_mean_margin_auc_mc_std": np.nan,
                "pairwise_dominance_auc_z_score": np.nan,
                "mean_margin_auc_z_score": np.nan,
                "pairwise_dominance_p_value_empirical": np.nan,
                "mean_margin_p_value_empirical": np.nan,
            }
        )

    return pd.DataFrame(auc_rows), pd.DataFrame(pairwise_auc_rows)


def run_model_set(
    model_set: str,
    args: argparse.Namespace,
    encoding_root_map: dict[str, Path] | None,
) -> None:
    result_dir = args.selection_root / model_set
    payload = apply_env_paths(eval_utils.load_selection_payload(result_dir), args.env)
    original_models = list(payload["model_names"])

    if args.random_feature_dir is None:
        available_models = original_models
        missing: list[str] = []
        random_feature_source = (
            f"candidate_pool:env-{args.env}" if args.env else "candidate_pool:payload_paths"
        )
    else:
        available_models = available_random_models(args.random_feature_dir, original_models)
        missing = sorted(set(original_models) - set(available_models))
        if missing and args.strict_random_models:
            raise FileNotFoundError(
                f"{model_set}: random feature cache is missing {len(missing)} models: {missing}"
            )
        if missing:
            print(
                f"[{model_set}] WARNING: dropping {len(missing)} models missing from "
                f"{args.random_feature_dir}: {missing}"
            )
        random_feature_source = f"local_random_pool:{args.random_feature_dir}"

    if len(available_models) < 2:
        raise RuntimeError(f"{model_set}: need at least two models after filtering")

    payload = filter_payload_to_models(payload, available_models)
    config_payload = payload.get("config", {})
    target_nc = float(args.noise_ceiling or config_payload.get("noise_ceiling_target", 0.46))
    roi_subset = args.roi_subset or config_payload.get("encoding_roi_subset", "hlvis")
    tracks = [
        track
        for track in eval_utils.get_all_tracks_for_evaluation(payload)
        if track["name"] in args.tracks
    ]
    if not tracks:
        raise ValueError(f"{model_set}: no requested tracks found: {args.tracks}")

    out_dir = args.output_root / f"{model_set}_teacher_student_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "model": original_models,
            "included": [model in available_models for model in original_models],
            "reason": [
                "included" if model in available_models else "missing_random_pool_feature"
                for model in original_models
            ],
            "random_feature_source": random_feature_source,
        }
    ).to_csv(out_dir / "model_roster.csv", index=False)

    metadata = {
        "model_set": model_set,
        "selection_root": str(args.selection_root),
        "output_dir": str(out_dir),
        "recovery_orientation": ORIENTATION,
        "random_feature_source": random_feature_source,
        "tracks": [track["name"] for track in tracks],
        "n_models": len(available_models),
        "n_random_images": args.n_random_images,
        "n_random_subsets": args.n_random_subsets,
        "n_splits": args.n_splits,
        "n_noise_samples": args.n_noise_samples,
        "split_sizes": {"train": args.train_n, "val": args.val_n, "test": args.test_n},
        "alphas": args.alphas,
        "noise_multipliers": args.noise_mults.tolist(),
        "base_noise_ceiling": target_nc,
        "roi_subset": roi_subset,
        "which_selection": args.which_selection,
        "seed": args.seed,
        "note": (
            "Teacher target is raw features for raw track and encoded/predicted "
            "brain responses for encoding tracks; candidates are refit from raw "
            "features with ridge regression."
        ),
    }
    with (out_dir / "teacher_student_recovery_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[{model_set}] loading selected raw features")
    selected_raw = selected_raw_from_payload(payload, available_models, args.which_selection)
    n_selected = next(iter(selected_raw.values())).shape[0]
    print(f"[{model_set}] loading random raw features")
    random_raw = load_random_raw_features(
        payload,
        available_models,
        args.n_random_images,
        args.random_feature_dir,
    )
    max_available = min(arr.shape[0] for arr in random_raw.values())
    if n_selected > max_available:
        raise ValueError(
            f"{model_set}: selected n={n_selected} exceeds random pool n={max_available}"
        )

    rng = np.random.default_rng(args.seed + stable_seed(model_set, "random_subsets"))
    random_subset_indices = [
        rng.choice(max_available, size=n_selected, replace=False)
        for _ in range(args.n_random_subsets)
    ]
    random_union = (
        np.unique(np.concatenate(random_subset_indices))
        if random_subset_indices
        else np.asarray([], dtype=np.int64)
    )
    random_union_raw = {model: arr[random_union] for model, arr in random_raw.items()}

    all_subset_rows: list[dict[str, Any]] = []
    all_pairwise_subset_rows: list[dict[str, Any]] = []
    all_confusion_rows: list[dict[str, Any]] = []
    all_detail_rows: list[dict[str, Any]] = []

    encoding_params_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    selected_encoded_cache: dict[str, dict[str, np.ndarray]] = {}
    random_encoded_union_cache: dict[str, dict[str, np.ndarray]] = {}
    encoding_device = torch.device(args.encoding_device)

    for track_idx, track in enumerate(tracks):
        track_name = track["name"]
        track_type = track.get("type", "identity")
        print(f"\n[{model_set}] track {track_idx + 1}/{len(tracks)}: {track_name}")

        if track_type == "identity":
            selected_target = selected_raw
            random_union_target = random_union_raw
        elif track_type == "encoding":
            enc_name = track.get("encoding_name") or track_name
            if enc_name not in encoding_params_cache:
                enc_root = (encoding_root_map or {}).get(
                    enc_name,
                    Path(config_payload.get("paths", {}).get("encoding_root", "")),
                )
                if not enc_root:
                    raise ValueError(f"{model_set}/{enc_name}: missing encoding root")
                print(f"  Loading encoding params from {enc_root}")
                encoding_params_cache[enc_name] = load_encoding_params(
                    enc_root,
                    MODEL_LIST_CSV,
                    available_models,
                    enc_name,
                    roi_subset=roi_subset,
                )
            if enc_name not in selected_encoded_cache:
                print("  Encoding selected raw features")
                selected_encoded_cache[enc_name] = encode_raw_features(
                    selected_raw,
                    encoding_params_cache[enc_name],
                    device=encoding_device,
                    batch_size=args.encoding_batch_size,
                )
            if enc_name not in random_encoded_union_cache:
                print(
                    f"  Encoding random union ({len(random_union)} images from "
                    f"{args.n_random_subsets} subsets)"
                )
                random_encoded_union_cache[enc_name] = encode_raw_features(
                    random_union_raw,
                    encoding_params_cache[enc_name],
                    device=encoding_device,
                    batch_size=args.encoding_batch_size,
                )
            selected_target = selected_encoded_cache[enc_name]
            random_union_target = random_encoded_union_cache[enc_name]
        else:
            raise ValueError(f"Unsupported track type: {track_type}")

        print("  Running selected teacher/student recovery")
        subset_rows, pairwise_rows, confusion_or_details = run_subset_recovery_metrics(
            model_set=model_set,
            subset_type="selected",
            subset_idx=0,
            track=track,
            raw_by_model=selected_raw,
            target_by_model=selected_target,
            model_names=available_models,
            n_splits=args.n_splits,
            n_noise_samples=args.n_noise_samples,
            train_n=args.train_n,
            val_n=args.val_n,
            test_n=args.test_n,
            alphas=args.alphas,
            base_noise_ceiling=target_nc,
            noise_mults=args.noise_mults,
            seed=args.seed + stable_seed(model_set, track_name, "selected"),
            write_details=args.write_details,
        )
        all_subset_rows.extend(subset_rows)
        all_pairwise_subset_rows.extend(pairwise_rows)
        if args.write_details:
            all_detail_rows.extend(
                row for row in confusion_or_details if "recovered_correct" in row
            )
            all_confusion_rows.extend(
                row for row in confusion_or_details if "recovered_correct" not in row
            )
        else:
            all_confusion_rows.extend(confusion_or_details)

        union_lookup = {int(idx): pos for pos, idx in enumerate(random_union)}
        for subset_idx, sample_idx in enumerate(
            tqdm(random_subset_indices, desc=f"{model_set}/{track_name} random", leave=False)
        ):
            union_pos = np.asarray([union_lookup[int(idx)] for idx in sample_idx], dtype=np.int64)
            random_raw_subset = {
                model: random_raw[model][sample_idx] for model in available_models
            }
            if track_type == "identity":
                random_target_subset = random_raw_subset
            else:
                random_target_subset = {
                    model: random_union_target[model][union_pos]
                    for model in available_models
                }

            subset_rows, pairwise_rows, confusion_or_details = run_subset_recovery_metrics(
                model_set=model_set,
                subset_type="random",
                subset_idx=subset_idx,
                track=track,
                raw_by_model=random_raw_subset,
                target_by_model=random_target_subset,
                model_names=available_models,
                n_splits=args.n_splits,
                n_noise_samples=args.n_noise_samples,
                train_n=args.train_n,
                val_n=args.val_n,
                test_n=args.test_n,
                alphas=args.alphas,
                base_noise_ceiling=target_nc,
                noise_mults=args.noise_mults,
                seed=args.seed + stable_seed(model_set, track_name, "random", subset_idx),
                write_details=args.write_details,
            )
            all_subset_rows.extend(subset_rows)
            all_pairwise_subset_rows.extend(pairwise_rows)
            if args.write_details:
                all_detail_rows.extend(
                    row for row in confusion_or_details if "recovered_correct" in row
                )
                all_confusion_rows.extend(
                    row for row in confusion_or_details if "recovered_correct" not in row
                )
            else:
                all_confusion_rows.extend(confusion_or_details)

        subset_df = pd.DataFrame(all_subset_rows)
        subset_df.to_csv(out_dir / "teacher_student_subset_curves.csv", index=False)
        pairwise_subset_df = pd.DataFrame(all_pairwise_subset_rows)
        pairwise_subset_df.to_csv(out_dir / "teacher_student_pairwise_subset_curves.csv", index=False)
        pd.DataFrame(all_confusion_rows).to_csv(
            out_dir / "teacher_student_confusion_matrix.csv", index=False
        )
        if args.write_details:
            pd.DataFrame(all_detail_rows).to_csv(
                out_dir / "teacher_student_recoveries.csv", index=False
            )

        gc.collect()
        if encoding_device.type == "cuda":
            torch.cuda.empty_cache()

    subset_df = pd.DataFrame(all_subset_rows)
    subset_df.to_csv(out_dir / "teacher_student_subset_curves.csv", index=False)
    curves = aggregate_curve_rows(
        subset_df,
        random_feature_source=random_feature_source,
    )
    curves.to_csv(out_dir / "discriminability.csv", index=False)
    pairwise_curves = aggregate_pairwise_rows(
        pd.DataFrame(all_pairwise_subset_rows),
        random_feature_source=random_feature_source,
    )
    pairwise_curves.to_csv(out_dir / "pairwise_margin.csv", index=False)
    auc_df, pairwise_auc_df = auc_rows_from_curves(curves, pairwise_curves)
    auc_df.to_csv(out_dir / "auc_significance.csv", index=False)
    pairwise_auc_df.to_csv(out_dir / "pairwise_auc.csv", index=False)
    pd.DataFrame(all_confusion_rows).to_csv(
        out_dir / "teacher_student_confusion_matrix.csv", index=False
    )
    if args.write_details:
        pd.DataFrame(all_detail_rows).to_csv(
            out_dir / "teacher_student_recoveries.csv", index=False
        )
    print(f"[{model_set}] saved {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-sets", default=",".join(MODEL_SET_ORDER))
    parser.add_argument("--tracks", default="raw," + ",".join(ENCODING_TRACKS))
    parser.add_argument("--selection-root", type=Path, default=SELECTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--env",
        choices=eval_utils.VALID_ENVS,
        default=None,
        help="Override payload paths with the repo env config.",
    )
    parser.add_argument(
        "--random-feature-dir",
        type=Path,
        default=None,
        help=(
            "Optional local .npz random-feature cache. Omit for the candidate-pool "
            "baseline from payload/env paths."
        ),
    )
    parser.add_argument("--strict-random-models", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random-subsets", type=int, default=20)
    parser.add_argument("--n-random-images", type=int, default=10000)
    parser.add_argument("--n-splits", type=int, default=8)
    parser.add_argument("--n-noise-samples", type=int, default=20)
    parser.add_argument("--train-n", type=int, default=60)
    parser.add_argument("--val-n", type=int, default=20)
    parser.add_argument("--test-n", type=int, default=20)
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--noise-ceiling", type=float, default=None)
    parser.add_argument(
        "--noise-mults",
        default=None,
        help="Comma-separated noise multipliers. Default: cstims recovery grid.",
    )
    parser.add_argument(
        "--which-selection",
        choices=["final", "greedy", "best_raw_combined"],
        default="final",
    )
    parser.add_argument("--unique-encodings", action="store_true", default=True)
    parser.add_argument("--shared-encodings", action="store_false", dest="unique_encodings")
    parser.add_argument("--encoding-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoding-batch-size", type=int, default=256)
    parser.add_argument("--roi-subset", default=None)
    parser.add_argument("--write-details", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_sets = [item.strip() for item in args.model_sets.split(",") if item.strip()]
    args.tracks = [item.strip() for item in args.tracks.split(",") if item.strip()]
    args.selection_root = args.selection_root.resolve()
    args.output_root = args.output_root.resolve()
    args.alphas = parse_float_list(args.alphas)
    if args.noise_mults:
        args.noise_mults = np.asarray(parse_float_list(args.noise_mults), dtype=np.float64)
    else:
        args.noise_mults = np.asarray(disc.get_default_noise_level_multipliers(), dtype=np.float64)
    if args.random_feature_dir is not None:
        args.random_feature_dir = args.random_feature_dir.resolve()
        if not args.random_feature_dir.exists():
            raise FileNotFoundError(f"Random feature directory not found: {args.random_feature_dir}")

    if args.encoding_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested for encoding but unavailable; using CPU")
        args.encoding_device = "cpu"
    if args.encoding_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    encoding_root_map = None
    if args.unique_encodings:
        encoding_root_map = {
            key: Path(value) for key, value in paper_config.UNIQUE_ENCODING_DIRS.items()
        }
        print(f"Using unique encoding roots: {list(encoding_root_map)}")

    if args.random_feature_dir is None:
        print("Using candidate-pool random baseline from payload/env paths")
    else:
        print(f"Using local random-feature cache: {args.random_feature_dir}")

    for model_set in tqdm(args.model_sets, desc="Model sets"):
        run_model_set(model_set, args, encoding_root_map)


if __name__ == "__main__":
    main()
