#!/usr/bin/env python3
"""Standalone profiler for teacher/student recovery hot operations.

This file intentionally duplicates the core numerical operations instead of
importing them from ``cstims``.  It is meant to answer one question cleanly:
where does time go inside the repeated ridge-predict -> RDM -> Spearman loop?
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

EPSILON = 1e-9
TRIU_INDEX_CACHE: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def find_share_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for path in (start, *start.parents):
        if (
            (path / "pyproject.toml").exists()
            and (path / "00_stimulus_selection").exists()
            and (path / "01_brain_model_alignment").exists()
        ):
            return path
    return Path.cwd().resolve()


ROOT = find_share_root()


@dataclass
class TimingRecord:
    label: str
    seconds: float
    calls: int = 1
    detail: str = ""


class Profiler:
    def __init__(self, *, synchronize_cuda: bool = False) -> None:
        self.records: list[TimingRecord] = []
        self.synchronize_cuda = bool(synchronize_cuda)

    def _sync(self) -> None:
        if self.synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def timed(self, label: str, *, detail: str = "", calls: int = 1):
        self._sync()
        start = perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.records.append(
                TimingRecord(
                    label=label,
                    seconds=perf_counter() - start,
                    calls=calls,
                    detail=detail,
                )
            )

    def add(self, label: str, seconds: float, *, detail: str = "", calls: int = 1) -> None:
        self.records.append(TimingRecord(label, seconds, calls=calls, detail=detail))

    def detail_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.__dict__ for record in self.records])

    def summary_frame(self) -> pd.DataFrame:
        grouped: dict[str, list[TimingRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.label].append(record)
        rows = []
        for label, records in grouped.items():
            seconds = np.asarray([r.seconds for r in records], dtype=np.float64)
            calls = int(sum(r.calls for r in records))
            rows.append(
                {
                    "function": label,
                    "records": int(len(records)),
                    "calls": calls,
                    "completion_time_s": float(seconds.sum()),
                    "mean_record_s": float(seconds.mean()),
                    "mean_call_s": float(seconds.sum() / max(calls, 1)),
                    "max_record_s": float(seconds.max()),
                }
            )
        return pd.DataFrame(rows).sort_values("completion_time_s", ascending=False)


def print_markdown_table(df: pd.DataFrame, *, max_rows: int) -> None:
    df = df.head(max_rows)
    print(
        "| function | records | calls | completion time (s) | "
        "mean call (ms) | max record (ms) |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for row in df.itertuples(index=False):
        print(
            f"| `{row.function}` | {row.records} | {row.calls} | "
            f"{row.completion_time_s:.3f} | {row.mean_call_s * 1000:.3f} | "
            f"{row.max_record_s * 1000:.3f} |"
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
        raise ValueError(f"indices out of range 0..{n_items - 1}: {bad}")
    return out


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def short_path_hash(path: Path) -> str:
    return hashlib.blake2b(str(path.resolve()).encode("utf-8"), digest_size=4).hexdigest()


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("wb") as f:
        np.save(f, array)
    tmp.replace(path)


def cached_triu_indices(
    n_images: int,
    device: torch.device,
    profiler: Profiler,
    prefix: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_key = str(device)
    if device.type == "cuda":
        device_key = f"cuda:{torch.cuda.current_device()}"
    key = (int(n_images), device_key)
    cached = TRIU_INDEX_CACHE.get(key)
    if cached is None:
        with profiler.timed(f"{prefix}.triu_indices_create"):
            idx = torch.triu_indices(n_images, n_images, offset=1, device=device)
        cached = (idx[0], idx[1])
        TRIU_INDEX_CACHE[key] = cached
    else:
        with profiler.timed(f"{prefix}.triu_indices_cache_hit"):
            pass
    return cached


def npy_feature_cache_path(path: Path) -> Path:
    return path.parent / "_npy_cache" / f"{path.stem}.npy"


def feature_array_from_npz(data: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
    if "features" in data.files:
        return data["features"]
    candidates = [
        key
        for key in data.files
        if not key.startswith("_") and getattr(data[key], "ndim", 0) >= 2
    ]
    if not candidates:
        raise ValueError(f"No feature array found in {path}")
    return data[candidates[0]]


def load_npz_feature_array(path: Path, n_images: int | None = None) -> np.ndarray:
    npy_path = npy_feature_cache_path(path)
    if npy_path.exists():
        arr = np.load(npy_path, mmap_mode="r")
        return arr[: min(int(n_images), arr.shape[0])] if n_images is not None else arr
    with np.load(path, allow_pickle=True) as data:
        arr = np.asarray(feature_array_from_npz(data, path), dtype=np.float32)
    return arr[: min(int(n_images), arr.shape[0])] if n_images is not None else arr


def load_payload(payload_root: Path, model_set: str) -> dict[str, Any]:
    path = payload_root / model_set / "selected_stimuli_data.pkl"
    with path.open("rb") as f:
        payload = pickle.load(f)
    if "model_names" not in payload:
        raise ValueError(f"Payload has no model_names: {path}")
    return payload


def available_random_models(random_feature_dir: Path, model_names: list[str]) -> list[str]:
    return [
        model
        for model in model_names
        if (random_feature_dir / f"{model}.npz").exists()
    ]


def load_selected_raw_features(
    payload: dict[str, Any],
    model_names: list[str],
) -> dict[str, np.ndarray]:
    features = payload.get("selected_features_raw")
    if not features:
        by_view = payload.get("selected_features_by_view") or {}
        features = by_view.get("raw")
    if not features:
        raise ValueError("Payload does not contain selected raw features")
    out = {}
    for model in model_names:
        arr = features[model]
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        out[model] = np.asarray(arr, dtype=np.float32)
    return out


def load_random_features(
    random_feature_dir: Path,
    model_names: list[str],
    n_random: int,
) -> dict[str, np.ndarray]:
    return {
        model: load_npz_feature_array(random_feature_dir / f"{model}.npz", n_random)
        for model in model_names
    }


def sanitize_layer_name(layer: str | int) -> str:
    layer_str = str(layer).strip()
    return (
        layer_str.replace(".", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def load_model_layers(model_list_csv: Path) -> dict[str, str]:
    with model_list_csv.open("r") as f:
        reader = csv.DictReader(f)
        return {row["model"]: row["layer"] for row in reader}


def load_encoding_params_profiled(
    *,
    encoding_root: Path,
    model_list_csv: Path,
    encoding_name: str,
    model_names: list[str],
    device: torch.device,
    roi_subset: str | None,
    cache_dir: Path | None,
    use_cache: bool,
    build_cache: bool,
    profiler: Profiler,
) -> dict[str, dict[str, torch.Tensor]]:
    with profiler.timed("encoding.load_model_layers_csv"):
        layers_by_model = load_model_layers(model_list_csv)
    params: dict[str, dict[str, torch.Tensor]] = {}
    roi_mask: np.ndarray | None = None
    cache_subdir = None
    if cache_dir is not None:
        roi_name = roi_subset or "all"
        cache_subdir = (
            cache_dir
            / f"{safe_name(encoding_name)}_{safe_name(roi_name)}_{short_path_hash(encoding_root)}"
        )
    for model in model_names:
        layer = layers_by_model[model]
        model_dir = (
            encoding_root
            / f"{encoding_name}_{model}.layer{sanitize_layer_name(layer)}"
        )
        enc_npz = model_dir / "encoding_model.npz"
        if not enc_npz.exists():
            raise FileNotFoundError(f"Missing encoding file: {enc_npz}")
        weights_cache = None
        bias_cache = None
        if cache_subdir is not None:
            stem = safe_name(model)
            weights_cache = cache_subdir / f"{stem}_weights.float32.npy"
            bias_cache = cache_subdir / f"{stem}_bias.float32.npy"
        if (
            use_cache
            and weights_cache is not None
            and bias_cache is not None
            and weights_cache.exists()
            and bias_cache.exists()
        ):
            with profiler.timed("encoding_cache.load_weights_npy", detail=model):
                weights = np.load(weights_cache, mmap_mode="c")
            with profiler.timed("encoding_cache.load_bias_npy", detail=model):
                intercept = np.load(bias_cache, mmap_mode="c")
            with profiler.timed("encoding_cache.weights_to_device", detail=model):
                weights_t = torch.from_numpy(np.asarray(weights)).to(
                    device=device,
                    dtype=torch.float32,
                )
            with profiler.timed("encoding_cache.bias_to_device", detail=model):
                bias_t = torch.from_numpy(np.asarray(intercept)).to(
                    device=device,
                    dtype=torch.float32,
                )
            params[model] = {"W": weights_t, "bias": bias_t}
            continue
        with profiler.timed("encoding.load_npz", detail=model):
            with np.load(enc_npz) as z:
                with profiler.timed("encoding.read_weights_intercept", detail=model):
                    weights = z["weights"]
                    intercept = z["intercept"]
                if roi_subset is not None and roi_mask is None:
                    roi_key = f"roi_{roi_subset}"
                    if roi_key in z:
                        with profiler.timed("encoding.read_roi_mask", detail=roi_key):
                            roi_mask = z[roi_key].astype(bool)
                if roi_mask is not None:
                    with profiler.timed("encoding.apply_roi_mask", detail=model):
                        weights = weights[:, roi_mask]
                        intercept = intercept[roi_mask]
                weights = np.ascontiguousarray(weights, dtype=np.float32)
                intercept = np.ascontiguousarray(intercept, dtype=np.float32)
                if build_cache and weights_cache is not None and bias_cache is not None:
                    with profiler.timed("encoding_cache.save_weights_npy", detail=model):
                        atomic_save_npy(weights_cache, weights)
                    with profiler.timed("encoding_cache.save_bias_npy", detail=model):
                        atomic_save_npy(bias_cache, intercept)
                with profiler.timed("encoding.weights_to_device", detail=model):
                    weights_t = torch.from_numpy(weights).to(
                        device=device,
                        dtype=torch.float32,
                    )
                with profiler.timed("encoding.bias_to_device", detail=model):
                    bias_t = torch.from_numpy(intercept).to(
                        device=device,
                        dtype=torch.float32,
                    )
        params[model] = {"W": weights_t, "bias": bias_t}
    return params


def encode_raw_feature_arrays_profiled(
    *,
    raw_features: dict[str, np.ndarray],
    encoding_params: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    batch_size: int,
    profiler: Profiler,
    prefix: str,
    empty_cache: bool = False,
) -> dict[str, np.ndarray]:
    model_names = list(raw_features)
    n_samples = next(iter(raw_features.values())).shape[0]
    encoded_batches: dict[str, list[np.ndarray]] = {model: [] for model in model_names}
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_n = end_idx - start_idx
            batch_torch: dict[str, torch.Tensor] = {}
            for model in model_names:
                with profiler.timed(
                    f"{prefix}.batch_to_device",
                    detail=f"{model}:{start_idx}:{end_idx}",
                ):
                    batch_torch[model] = torch.as_tensor(
                        raw_features[model][start_idx:end_idx],
                        device=device,
                        dtype=torch.float32,
                    )
            for model in model_names:
                W = encoding_params[model]["W"]
                bias = encoding_params[model]["bias"]
                feats = batch_torch[model]
                with profiler.timed(
                    f"{prefix}.matmul_plus_bias",
                    detail=f"{model}:batch_n={batch_n}",
                ):
                    encoded = feats @ W + bias
                with profiler.timed(
                    f"{prefix}.to_numpy",
                    detail=f"{model}:batch_n={batch_n}",
                ):
                    encoded_batches[model].append(encoded.detach().cpu().numpy())
            del batch_torch
            if empty_cache and device.type == "cuda":
                with profiler.timed(f"{prefix}.cuda_empty_cache"):
                    torch.cuda.empty_cache()
    out = {}
    for model, chunks in encoded_batches.items():
        with profiler.timed(f"{prefix}.concat", detail=model):
            out[model] = np.concatenate(chunks, axis=0).astype(
                np.float32,
                copy=False,
            )
    return out


def standardize_from_train_profiled(
    train: np.ndarray,
    *others: np.ndarray,
    scale_by_sqrt_features: bool,
    profiler: Profiler,
    prefix: str,
) -> tuple[np.ndarray, ...]:
    with profiler.timed(f"{prefix}.cast_train_and_others"):
        train = np.asarray(train, dtype=np.float32)
        others = tuple(np.asarray(arr, dtype=np.float32) for arr in others)
    with profiler.timed(f"{prefix}.mean"):
        mean = train.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
    with profiler.timed(f"{prefix}.std"):
        scale = train.std(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
        scale[scale < 1e-6] = 1.0
    if scale_by_sqrt_features:
        with profiler.timed(f"{prefix}.fold_sqrt_feature_scale"):
            scale *= np.float32(math.sqrt(train.shape[1]))
    with profiler.timed(f"{prefix}.apply_train"):
        out = [(train - mean) / scale]
    with profiler.timed(f"{prefix}.apply_others", calls=len(others)):
        out.extend((arr - mean) / scale for arr in others)
    with profiler.timed(f"{prefix}.as_float32", calls=len(out)):
        return tuple(np.asarray(arr, dtype=np.float32) for arr in out)


@dataclass
class RidgeOps:
    alphas: tuple[float, ...]
    eigvals: np.ndarray
    train_projected: np.ndarray
    val_projected: np.ndarray
    eval_projected: dict[str, np.ndarray]
    eval_keys: tuple[str, ...]
    eval_lengths: tuple[int, ...]
    eval_projected_all: np.ndarray


def build_kernel_ridge_ops_profiled(
    x_train: np.ndarray,
    x_val: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    alphas: list[float],
    profiler: Profiler,
) -> RidgeOps:
    with profiler.timed("ridge.cast_train32"):
        x_train32 = np.asarray(x_train, dtype=np.float32)
    with profiler.timed("ridge.kernel_train_matmul"):
        k_train = (x_train32 @ x_train32.T).astype(np.float64, copy=False)
    with profiler.timed("ridge.eigh"):
        eigvals, eigvecs = np.linalg.eigh(k_train)
        eigvals = np.maximum(eigvals, 0.0).astype(np.float32, copy=False)
    with profiler.timed("ridge.train_projected_cast"):
        eigvecs32 = np.asarray(eigvecs, dtype=np.float32)
        train_projected = eigvecs32
    with profiler.timed("ridge.feature_projector_matmul"):
        projector = x_train32.T @ eigvecs32
    with profiler.timed("ridge.val_projected_matmul"):
        val_projected = np.asarray(
            np.asarray(x_val, dtype=np.float32) @ projector,
            dtype=np.float32,
        )
    eval_projected: dict[str, np.ndarray] = {}
    for key, x_eval in eval_sets.items():
        with profiler.timed("ridge.eval_projected_matmul"):
            eval_projected[key] = np.asarray(
                np.asarray(x_eval, dtype=np.float32) @ projector,
                dtype=np.float32,
            )
    with profiler.timed("ridge.pack_eval_projected_all"):
        eval_keys = tuple(eval_projected)
        eval_lengths = tuple(int(eval_projected[key].shape[0]) for key in eval_keys)
        eval_projected_all = np.concatenate(
            [eval_projected[key] for key in eval_keys],
            axis=0,
        ).astype(np.float32, copy=False)
    return RidgeOps(
        alphas=tuple(float(alpha) for alpha in alphas),
        eigvals=np.asarray(eigvals, dtype=np.float32),
        train_projected=train_projected,
        val_projected=val_projected,
        eval_projected=eval_projected,
        eval_keys=eval_keys,
        eval_lengths=eval_lengths,
        eval_projected_all=eval_projected_all,
    )


def pearson_columns_profiled(
    x: np.ndarray,
    y: np.ndarray,
    profiler: Profiler,
) -> np.ndarray:
    with profiler.timed("alpha_pearson.cast"):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
    with profiler.timed("alpha_pearson.center"):
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)
    with profiler.timed("alpha_pearson.denom"):
        denom = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    with profiler.timed("alpha_pearson.divide"):
        out = np.full(x.shape[1], np.nan, dtype=np.float64)
        ok = denom > 0
        out[ok] = np.sum(x[:, ok] * y[:, ok], axis=0) / denom[ok]
        return out


def select_targetwise_alpha_profiled(
    ops: RidgeOps,
    y_train: np.ndarray,
    y_val: np.ndarray,
    profiler: Profiler,
) -> tuple[list[float], np.ndarray, dict[float, np.ndarray]]:
    alpha_values = list(ops.alphas)
    with profiler.timed("alpha.project_targets"):
        projected_y = ops.train_projected.T @ np.asarray(y_train, dtype=np.float32)
    coeffs_by_alpha = []
    coefficient_cache: dict[float, np.ndarray] = {}
    for alpha in alpha_values:
        with profiler.timed("alpha.coefficients_from_projected"):
            denom = ops.eigvals.astype(np.float32, copy=False) + np.float32(alpha)
            coeff = np.asarray(projected_y / denom[:, None], dtype=np.float32)
            coefficient_cache[float(alpha)] = coeff
            coeffs_by_alpha.append(coeff)
    with profiler.timed("alpha.concatenate_coefficients"):
        coeff_all = np.concatenate(coeffs_by_alpha, axis=1)
    with profiler.timed("alpha.validation_prediction_matmul"):
        pred_val_all = np.asarray(ops.val_projected @ coeff_all, dtype=np.float32)
    scores = np.empty((len(alpha_values), y_train.shape[1]), dtype=np.float64)
    n_targets = y_train.shape[1]
    for alpha_idx, _alpha in enumerate(alpha_values):
        start = alpha_idx * n_targets
        stop = start + n_targets
        scores[alpha_idx] = pearson_columns_profiled(
            pred_val_all[:, start:stop],
            y_val,
            profiler,
        )
    with profiler.timed("alpha.nan_to_num_and_argmax"):
        scores = np.nan_to_num(scores, nan=-np.inf)
        best_alpha_idx = np.argmax(scores, axis=0).astype(np.int32)
    return alpha_values, best_alpha_idx, coefficient_cache


def pack_selected_coefficients_profiled(
    alpha_values: list[float],
    best_alpha_idx: np.ndarray,
    coefficient_cache: dict[float, np.ndarray],
    profiler: Profiler,
) -> np.ndarray:
    with profiler.timed("pack_coefficients.allocate"):
        first = next(iter(coefficient_cache.values()))
        coeff = np.empty(first.shape, dtype=np.float32)
    for alpha_idx, alpha in enumerate(alpha_values):
        with profiler.timed("pack_coefficients.find_target_columns"):
            cols = np.flatnonzero(best_alpha_idx == alpha_idx)
        if cols.size == 0:
            continue
        with profiler.timed("pack_coefficients.copy_columns"):
            coeff[:, cols] = coefficient_cache[float(alpha)][:, cols]
    return coeff


def predict_all_eval_profiled(
    ops: RidgeOps,
    coeff_selected: np.ndarray,
    profiler: Profiler,
) -> np.ndarray:
    with profiler.timed(
        "predict.all_eval_matmul",
        detail=(
            f"n_eval_total={ops.eval_projected_all.shape[0]},"
            f"n_train={ops.eval_projected_all.shape[1]},"
            f"n_targets={coeff_selected.shape[1]}"
        ),
    ):
        return np.asarray(
            ops.eval_projected_all @ np.asarray(coeff_selected, dtype=np.float32),
            dtype=np.float32,
        )


def pearson_alpha_scores_batch_profiled(
    pred_val_all: np.ndarray,
    y_val_batch: np.ndarray,
    profiler: Profiler,
) -> np.ndarray:
    """Pearson scores for [alpha, sample, val_image, target] predictions."""
    with profiler.timed("alpha_batch.pearson_center"):
        x = np.asarray(pred_val_all, dtype=np.float32)
        y = np.asarray(y_val_batch, dtype=np.float32)[None, :, :, :]
        x = x - x.mean(axis=2, keepdims=True)
        y = y - y.mean(axis=2, keepdims=True)
    with profiler.timed("alpha_batch.pearson_num_denom"):
        numerator = np.sum(x * y, axis=2)
        denom = np.sqrt(np.sum(x * x, axis=2) * np.sum(y * y, axis=2))
    with profiler.timed("alpha_batch.pearson_divide"):
        out = np.full(numerator.shape, np.nan, dtype=np.float32)
        ok = denom > 0
        out[ok] = numerator[ok] / denom[ok]
        return out


def select_targetwise_alpha_batch_profiled(
    ops: RidgeOps,
    y_train_batch: np.ndarray,
    y_val_batch: np.ndarray,
    profiler: Profiler,
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Choose target-wise alphas for all noise samples in one batched pass."""
    alpha_values = list(ops.alphas)
    with profiler.timed("alpha_batch.project_targets"):
        projected_y = np.matmul(
            ops.train_projected.T[None, :, :],
            np.asarray(y_train_batch, dtype=np.float32),
        )
    coeffs_by_alpha = []
    for alpha in alpha_values:
        with profiler.timed("alpha_batch.coefficients_from_projected"):
            denom = ops.eigvals.astype(np.float32, copy=False) + np.float32(alpha)
            coeffs_by_alpha.append(
                np.asarray(projected_y / denom[None, :, None], dtype=np.float32)
            )
    with profiler.timed("alpha_batch.stack_coefficients"):
        coeff_stack = np.stack(coeffs_by_alpha, axis=0)
    with profiler.timed("alpha_batch.validation_prediction_matmul"):
        pred_val_all = np.einsum(
            "vr,asrt->asvt",
            ops.val_projected,
            coeff_stack,
            optimize=True,
        )
    scores = pearson_alpha_scores_batch_profiled(pred_val_all, y_val_batch, profiler)
    with profiler.timed("alpha_batch.nan_to_num_and_argmax"):
        scores = np.nan_to_num(scores, nan=-np.inf)
        best_alpha_idx = np.argmax(scores, axis=0).astype(np.int32)
    return alpha_values, best_alpha_idx, coeff_stack


def pack_selected_coefficients_batch_profiled(
    best_alpha_idx: np.ndarray,
    coeff_stack: np.ndarray,
    profiler: Profiler,
) -> np.ndarray:
    """Pack selected coefficients for [sample, rank, target]."""
    n_alphas, n_samples, n_rank, n_targets = coeff_stack.shape
    with profiler.timed("pack_coefficients_batch.allocate"):
        coeff = np.empty((n_samples, n_rank, n_targets), dtype=np.float32)
    for alpha_idx in range(n_alphas):
        with profiler.timed("pack_coefficients_batch.copy_alpha_columns"):
            for sample_idx in range(n_samples):
                cols = np.flatnonzero(best_alpha_idx[sample_idx] == alpha_idx)
                if cols.size:
                    coeff[sample_idx, :, cols] = coeff_stack[
                        alpha_idx,
                        sample_idx,
                        :,
                        cols,
                    ]
    return coeff


def select_and_pack_targetwise_alpha_batch_torch_profiled(
    ops: RidgeOps,
    y_train_batch: torch.Tensor,
    y_val_batch: torch.Tensor,
    device: torch.device,
    profiler: Profiler,
) -> torch.Tensor:
    """Choose target-wise alphas and pack coefficients fully on the GPU."""
    with profiler.timed("alpha_batch_gpu.ops_to_tensor"):
        train_projected_t = torch.as_tensor(
            ops.train_projected.T,
            dtype=torch.float32,
            device=device,
        )
        val_projected_t = torch.as_tensor(
            ops.val_projected,
            dtype=torch.float32,
            device=device,
        )
        eigvals_t = torch.as_tensor(ops.eigvals, dtype=torch.float32, device=device)
        alphas_t = torch.as_tensor(ops.alphas, dtype=torch.float32, device=device)
    with profiler.timed("alpha_batch_gpu.project_targets"):
        projected_y = torch.matmul(train_projected_t.unsqueeze(0), y_train_batch)
    with profiler.timed("alpha_batch_gpu.coefficients_from_projected"):
        denom = eigvals_t[None, None, :, None] + alphas_t[:, None, None, None]
        coeff_stack = projected_y.unsqueeze(0) / denom
    with profiler.timed("alpha_batch_gpu.validation_prediction_matmul"):
        pred_val_all = torch.matmul(
            val_projected_t.unsqueeze(0).unsqueeze(0),
            coeff_stack,
        )
    with profiler.timed("alpha_batch_gpu.pearson_center"):
        x = pred_val_all - pred_val_all.mean(dim=2, keepdim=True)
        y = y_val_batch.unsqueeze(0)
        y = y - y.mean(dim=2, keepdim=True)
    with profiler.timed("alpha_batch_gpu.pearson_num_denom"):
        numerator = torch.sum(x * y, dim=2)
        denom = torch.sqrt(torch.sum(x * x, dim=2) * torch.sum(y * y, dim=2))
    with profiler.timed("alpha_batch_gpu.pearson_divide"):
        scores = numerator / (denom + EPSILON)
        scores = torch.nan_to_num(scores, nan=-float("inf"))
    with profiler.timed("alpha_batch_gpu.argmax"):
        best_alpha_idx = torch.argmax(scores, dim=0)
    with profiler.timed("pack_coefficients_batch_gpu.gather"):
        gather_idx = best_alpha_idx.unsqueeze(0).unsqueeze(2).expand(
            1,
            -1,
            coeff_stack.shape[2],
            -1,
        )
        return torch.gather(coeff_stack, dim=0, index=gather_idx).squeeze(0)


def predict_all_eval_noise_batch_profiled(
    ops: RidgeOps,
    coeff_selected: np.ndarray,
    profiler: Profiler,
) -> np.ndarray:
    with profiler.timed(
        "predict_noise_batch.all_eval_matmul",
        detail=(
            f"n_samples={coeff_selected.shape[0]},"
            f"n_eval_total={ops.eval_projected_all.shape[0]},"
            f"n_train={ops.eval_projected_all.shape[1]},"
            f"n_targets={coeff_selected.shape[2]}"
        ),
    ):
        return np.asarray(
            np.matmul(
                ops.eval_projected_all[None, :, :],
                np.asarray(coeff_selected, dtype=np.float32),
            ),
            dtype=np.float32,
        )


def predict_all_eval_noise_batch_torch_profiled(
    ops: RidgeOps,
    coeff_selected: np.ndarray | torch.Tensor,
    device: torch.device,
    profiler: Profiler,
) -> torch.Tensor:
    with profiler.timed("predict_noise_batch_gpu.eval_projected_to_tensor"):
        eval_projected = torch.as_tensor(
            ops.eval_projected_all,
            dtype=torch.float32,
            device=device,
        )
    with profiler.timed("predict_noise_batch_gpu.coefficients_to_tensor"):
        coeff = torch.as_tensor(coeff_selected, dtype=torch.float32, device=device)
    with profiler.timed(
        "predict_noise_batch_gpu.bmm",
        detail=(
            f"n_samples={coeff.shape[0]},"
            f"n_eval_total={eval_projected.shape[0]},"
            f"n_train={eval_projected.shape[1]},"
            f"n_targets={coeff.shape[2]}"
        ),
    ):
        return torch.bmm(
            eval_projected.unsqueeze(0).expand(coeff.shape[0], -1, -1),
            coeff,
        )


def run_batched_noise_samples_profiled(
    *,
    model_names: list[str],
    candidate_ops: dict[str, RidgeOps],
    teacher: str,
    teacher_idx: int,
    y_train_clean: np.ndarray,
    y_val_clean: np.ndarray,
    eval_y_clean_all: np.ndarray,
    eval_keys: tuple[str, ...],
    noise_mult: float,
    response_noise_std: float,
    achieved: float,
    n_noise_samples: int,
    metric: str,
    corr_type: str,
    rdm_device: torch.device,
    gpu_alpha_batch: bool,
    gpu_predict_batch: bool,
    gpu_eval_noise_batch: bool,
    teacher_rng: np.random.Generator,
    profiler: Profiler,
) -> list[dict[str, Any]]:
    with profiler.timed("noise_batch.train_standard_normal"):
        train_noise = teacher_rng.standard_normal(
            (n_noise_samples, *y_train_clean.shape),
            dtype=np.float32,
        )
        train_noise *= np.float32(response_noise_std)
    with profiler.timed("noise_batch.val_standard_normal"):
        val_noise = teacher_rng.standard_normal(
            (n_noise_samples, *y_val_clean.shape),
            dtype=np.float32,
        )
        val_noise *= np.float32(response_noise_std)
    with profiler.timed("noise_batch.add_train_val"):
        y_train_batch = y_train_clean[None, :, :] + train_noise
        y_val_batch = y_val_clean[None, :, :] + val_noise
    if gpu_alpha_batch:
        with profiler.timed("alpha_batch_gpu.y_train_to_tensor"):
            y_train_batch_t = torch.as_tensor(
                y_train_batch,
                dtype=torch.float32,
                device=rdm_device,
            )
        with profiler.timed("alpha_batch_gpu.y_val_to_tensor"):
            y_val_batch_t = torch.as_tensor(
                y_val_batch,
                dtype=torch.float32,
                device=rdm_device,
            )
    else:
        y_train_batch_t = None
        y_val_batch_t = None

    if gpu_eval_noise_batch:
        with profiler.timed("noise_batch_gpu.eval_clean_to_tensor"):
            eval_clean_t = torch.as_tensor(
                eval_y_clean_all,
                dtype=torch.float32,
                device=rdm_device,
            )
        with profiler.timed("noise_batch_gpu.eval_randn", calls=n_noise_samples):
            eval_noise_t = torch.randn(
                (n_noise_samples, *eval_clean_t.shape),
                dtype=torch.float32,
                device=rdm_device,
            )
            eval_noise_t *= float(response_noise_std)
        with profiler.timed("noise_batch_gpu.eval_add", calls=n_noise_samples):
            y_eval_noisy_t = eval_clean_t.unsqueeze(0) + eval_noise_t
        noisy_teacher_rdm_batch = get_rdm_batch_tensor_profiled(
            y_eval_noisy_t.reshape(
                n_noise_samples * eval_y_clean_all.shape[0],
                eval_y_clean_all.shape[1],
                eval_y_clean_all.shape[2],
            ),
            metric,
            rdm_device,
            profiler,
            "noisy_eval_rdm_noise_batch",
        )
        del eval_clean_t, eval_noise_t, y_eval_noisy_t
    else:
        with profiler.timed("noise_batch.eval_standard_normal", calls=n_noise_samples):
            eval_noise = teacher_rng.standard_normal(
                (n_noise_samples, *eval_y_clean_all.shape),
                dtype=np.float32,
            )
            eval_noise *= np.float32(response_noise_std)
        with profiler.timed("noise_batch.eval_add", calls=n_noise_samples):
            y_eval_noisy = eval_y_clean_all[None, :, :, :] + eval_noise
        noisy_teacher_rdm_batch = get_rdm_batch_tensor_profiled(
            y_eval_noisy.reshape(
                n_noise_samples * eval_y_clean_all.shape[0],
                eval_y_clean_all.shape[1],
                eval_y_clean_all.shape[2],
            ),
            metric,
            rdm_device,
            profiler,
            "noisy_eval_rdm_noise_batch",
        )
    teacher_reference_batch = prepare_reference_vectors_profiled(
        noisy_teacher_rdm_batch,
        corr_type,
        profiler,
    )

    n_eval_sets = len(eval_keys)
    scores_by_sample_eval = np.full(
        (n_noise_samples, n_eval_sets, len(model_names)),
        np.nan,
        dtype=np.float32,
    )
    for candidate_idx, candidate in enumerate(model_names):
        ops = candidate_ops[candidate]
        if gpu_alpha_batch:
            if y_train_batch_t is None or y_val_batch_t is None:
                raise RuntimeError("GPU alpha tensors were not initialized")
            coeff_selected = select_and_pack_targetwise_alpha_batch_torch_profiled(
                ops,
                y_train_batch_t,
                y_val_batch_t,
                rdm_device,
                profiler,
            )
        else:
            _alpha_values, best_alpha_idx, coeff_stack = (
                select_targetwise_alpha_batch_profiled(
                    ops,
                    y_train_batch,
                    y_val_batch,
                    profiler,
                )
            )
            coeff_selected = pack_selected_coefficients_batch_profiled(
                best_alpha_idx,
                coeff_stack,
                profiler,
            )
        if gpu_predict_batch:
            pred_all_t = predict_all_eval_noise_batch_torch_profiled(
                ops,
                coeff_selected,
                rdm_device,
                profiler,
            )
            pred_rdm_batch = get_rdm_batch_tensor_profiled(
                pred_all_t.reshape(
                    n_noise_samples * n_eval_sets,
                    ops.eval_lengths[0],
                    pred_all_t.shape[2],
                ),
                metric,
                rdm_device,
                profiler,
                "prediction_rdm_noise_batch",
            )
            del pred_all_t
        else:
            pred_all = predict_all_eval_noise_batch_profiled(
                ops,
                coeff_selected,
                profiler,
            )
            pred_rdm_batch = get_rdm_batch_tensor_profiled(
                pred_all.reshape(
                    n_noise_samples * n_eval_sets,
                    ops.eval_lengths[0],
                    pred_all.shape[2],
                ),
                metric,
                rdm_device,
                profiler,
                "prediction_rdm_noise_batch",
            )
        candidate_scores = score_prediction_batch_profiled(
            pred_rdm_batch,
            teacher_reference_batch,
            corr_type,
            profiler,
        ).reshape(n_noise_samples, n_eval_sets)
        scores_by_sample_eval[:, :, candidate_idx] = candidate_scores

    rows = []
    with profiler.timed("recover.argmax_noise_batch", calls=n_noise_samples * n_eval_sets):
        for noise_sample_idx in range(n_noise_samples):
            for eval_idx, key in enumerate(eval_keys):
                scores = np.nan_to_num(
                    scores_by_sample_eval[noise_sample_idx, eval_idx],
                    nan=-np.inf,
                )
                recovered_idx = int(np.argmax(scores))
                rows.append(
                    {
                        "teacher": teacher,
                        "noise_mult": noise_mult,
                        "noise_sample_idx": noise_sample_idx,
                        "eval_key": key,
                        "recovered": model_names[recovered_idx],
                        "teacher_score": float(scores[teacher_idx]),
                        "best_score": float(scores[recovered_idx]),
                        "achieved_fit_rdm_reliability": float(achieved),
                    }
                )
    return rows


def get_rdm_vector_torch_profiled(
    activations: np.ndarray,
    metric: str,
    device: torch.device,
    profiler: Profiler,
    prefix: str,
) -> np.ndarray:
    with profiler.timed(f"{prefix}.to_tensor"):
        tensor = torch.as_tensor(activations, dtype=torch.float32, device=device)
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D activations, got {tuple(tensor.shape)}")
    if tensor.shape[0] < 2:
        return np.asarray([], dtype=np.float32)

    if metric == "cosine":
        with profiler.timed(f"{prefix}.norm"):
            norm = torch.norm(tensor.float(), p=2, dim=1, keepdim=True)
        with profiler.timed(f"{prefix}.normalize"):
            tensor_norm = tensor.float() / (norm + 1e-9)
        with profiler.timed(f"{prefix}.similarity_matmul"):
            similarity_matrix = torch.matmul(tensor_norm, tensor_norm.t())
        with profiler.timed(f"{prefix}.clamp_and_distance"):
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
    elif metric == "correlation":
        with profiler.timed(f"{prefix}.row_mean_std"):
            mean_val = torch.mean(tensor.float(), dim=1, keepdim=True)
            std_dev = torch.std(tensor.float(), dim=1, keepdim=True)
        with profiler.timed(f"{prefix}.normalize"):
            tensor_norm = (tensor.float() - mean_val) / (std_dev + 1e-9)
        with profiler.timed(f"{prefix}.similarity_matmul"):
            similarity_matrix = (
                torch.matmul(tensor_norm, tensor_norm.t()) / tensor.shape[1]
            )
        with profiler.timed(f"{prefix}.clamp_and_distance"):
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
    else:
        with profiler.timed(f"{prefix}.torch_pdist"):
            rdm = torch.pdist(tensor.float(), p=2)
        with profiler.timed(f"{prefix}.to_numpy"):
            return rdm.detach().cpu().numpy()

    indices = cached_triu_indices(rdm_matrix.shape[0], tensor.device, profiler, prefix)
    with profiler.timed(f"{prefix}.extract_triangle"):
        rdm = rdm_matrix[indices[0], indices[1]]
    with profiler.timed(f"{prefix}.to_numpy"):
        return rdm.detach().cpu().numpy()


def get_rdm_batch_tensor_profiled(
    activations: np.ndarray | torch.Tensor,
    metric: str,
    device: torch.device,
    profiler: Profiler,
    prefix: str,
) -> torch.Tensor:
    with profiler.timed(f"{prefix}.to_tensor"):
        tensor = torch.as_tensor(activations, dtype=torch.float32, device=device)
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D activations, got {tuple(tensor.shape)}")
    batch_n, n_images, n_features = tensor.shape
    if n_images < 2:
        return torch.zeros((batch_n, 0), device=device, dtype=torch.float32)

    if metric == "cosine":
        with profiler.timed(f"{prefix}.norm"):
            norm = torch.norm(tensor.float(), p=2, dim=2, keepdim=True)
        with profiler.timed(f"{prefix}.normalize"):
            tensor_norm = tensor.float() / (norm + 1e-9)
        with profiler.timed(f"{prefix}.similarity_bmm"):
            similarity_matrix = torch.bmm(tensor_norm, tensor_norm.transpose(1, 2))
        with profiler.timed(f"{prefix}.clamp_and_distance"):
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
    elif metric == "correlation":
        with profiler.timed(f"{prefix}.row_mean_std"):
            mean_val = torch.mean(tensor.float(), dim=2, keepdim=True)
            std_dev = torch.std(tensor.float(), dim=2, keepdim=True)
        with profiler.timed(f"{prefix}.normalize"):
            tensor_norm = (tensor.float() - mean_val) / (std_dev + 1e-9)
        with profiler.timed(f"{prefix}.similarity_bmm"):
            similarity_matrix = (
                torch.bmm(tensor_norm, tensor_norm.transpose(1, 2)) / n_features
            )
        with profiler.timed(f"{prefix}.clamp_and_distance"):
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
    else:
        with profiler.timed(f"{prefix}.torch_cdist"):
            rdm_matrix = torch.cdist(tensor.float(), tensor.float(), p=2)

    indices = cached_triu_indices(n_images, device, profiler, prefix)
    with profiler.timed(f"{prefix}.extract_triangle"):
        return rdm_matrix[:, indices[0], indices[1]]


def rank_standardize_batch_profiled(
    values: torch.Tensor,
    profiler: Profiler,
    prefix: str,
) -> torch.Tensor:
    with profiler.timed(f"{prefix}.argsort_first"):
        values = values.float()
        order = torch.argsort(values, dim=-1)
    with profiler.timed(f"{prefix}.scatter_ranks"):
        ranks = torch.empty_like(values, dtype=torch.float32)
        base = torch.arange(
            values.shape[-1],
            device=values.device,
            dtype=torch.float32,
        ).expand_as(values)
        ranks.scatter_(dim=-1, index=order, src=base)
    with profiler.timed(f"{prefix}.standardize"):
        return (ranks - ranks.mean(dim=-1, keepdim=True)) / (
            ranks.std(dim=-1, unbiased=False, keepdim=True) + EPSILON
        )


def vector_standardize_batch_profiled(
    values: torch.Tensor,
    profiler: Profiler,
    prefix: str,
) -> torch.Tensor:
    with profiler.timed(f"{prefix}.standardize"):
        values = values.float()
        return (values - values.mean(dim=-1, keepdim=True)) / (
            values.std(dim=-1, unbiased=False, keepdim=True) + EPSILON
        )


def prepare_reference_vectors_profiled(
    rdm_batch: torch.Tensor,
    corr_type: str,
    profiler: Profiler,
) -> torch.Tensor:
    if corr_type == "spearman":
        return rank_standardize_batch_profiled(
            rdm_batch,
            profiler,
            "teacher_rank",
        )
    return vector_standardize_batch_profiled(
        rdm_batch,
        profiler,
        "teacher_pearson_ref",
    )


def score_prediction_batch_profiled(
    pred_rdm_batch: torch.Tensor,
    teacher_reference_batch: torch.Tensor,
    corr_type: str,
    profiler: Profiler,
) -> np.ndarray:
    if corr_type == "spearman":
        pred_reference = rank_standardize_batch_profiled(
            pred_rdm_batch,
            profiler,
            "pred_rank",
        )
    else:
        pred_reference = vector_standardize_batch_profiled(
            pred_rdm_batch,
            profiler,
            "pred_pearson_ref",
        )
    with profiler.timed("score_batch.mean_dot"):
        scores = torch.mean(pred_reference * teacher_reference_batch, dim=-1)
    with profiler.timed("score_batch.to_numpy"):
        return scores.detach().cpu().numpy().astype(np.float32, copy=False)


def reshape_concatenated_eval_matrix(
    matrix: np.ndarray,
    lengths: tuple[int, ...],
) -> np.ndarray:
    unique_lengths = set(lengths)
    if len(unique_lengths) != 1:
        raise ValueError(
            "Batched profile requires equal eval-set sizes; got "
            f"{sorted(unique_lengths)}"
        )
    n_eval_sets = len(lengths)
    n_images = lengths[0]
    return matrix.reshape(n_eval_sets, n_images, matrix.shape[1])


def calculate_correlation_torch_profiled(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    corr_type: str,
    device: torch.device,
    profiler: Profiler,
) -> float:
    with profiler.timed("score.to_tensor"):
        a = torch.as_tensor(vec_a, dtype=torch.float32, device=device).reshape(-1)
        b = torch.as_tensor(vec_b, dtype=torch.float32, device=device).reshape(-1)
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    with profiler.timed("score.variance"):
        var_a, var_b = torch.var(a.float()), torch.var(b.float())
    if var_a < EPSILON and var_b < EPSILON:
        return 1.0
    if var_a < EPSILON or var_b < EPSILON:
        return 0.0
    if corr_type == "spearman":
        with profiler.timed("score.argsort_a_first"):
            a_float = a.float()
            order_a = torch.argsort(a_float)
        with profiler.timed("score.scatter_rank_a"):
            rank_a = torch.empty_like(a_float, dtype=torch.float32)
            rank_a.scatter_(
                dim=0,
                index=order_a,
                src=torch.arange(a.numel(), device=a.device, dtype=torch.float32),
            )
        with profiler.timed("score.argsort_b_first"):
            b_float = b.float()
            order_b = torch.argsort(b_float)
        with profiler.timed("score.scatter_rank_b"):
            rank_b = torch.empty_like(b_float, dtype=torch.float32)
            rank_b.scatter_(
                dim=0,
                index=order_b,
                src=torch.arange(b.numel(), device=b.device, dtype=torch.float32),
            )
        with profiler.timed("score.rank_standardize"):
            rank_a_std = (rank_a - rank_a.mean()) / (
                rank_a.std(unbiased=False) + EPSILON
            )
            rank_b_std = (rank_b - rank_b.mean()) / (
                rank_b.std(unbiased=False) + EPSILON
            )
        with profiler.timed("score.mean_dot"):
            r = torch.mean(rank_a_std * rank_b_std)
    else:
        with profiler.timed("score.pearson_corrcoef"):
            r = torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1]
    with profiler.timed("score.to_float"):
        return float(torch.nan_to_num(r, nan=0.0).detach().cpu().item())


def multiplier_to_noise_ceiling(k: float, nc_base: float) -> float:
    if k <= 0:
        return 1.0
    if nc_base <= 0 or nc_base >= 1:
        return nc_base
    term = k * k * (1.0 / (nc_base * nc_base) - 1.0)
    return float(1.0 / np.sqrt(1.0 + term))


def response_noise_std_from_multiplier(noise_mult: float, nc_base: float) -> float:
    if noise_mult <= 0 or nc_base <= 0 or nc_base >= 1:
        return 0.0
    return float(noise_mult * math.sqrt(1.0 / (nc_base * nc_base) - 1.0))


def empirical_response_noise_rdm_reliability_profiled(
    y_clean: np.ndarray,
    clean_rdm: np.ndarray,
    noise_std: float,
    *,
    metric: str,
    corr_type: str,
    rng: np.random.Generator,
    n_samples: int,
    device: torch.device,
    profiler: Profiler,
) -> float:
    if noise_std <= 0:
        return 1.0
    vals = []
    for _ in range(n_samples):
        with profiler.timed("calibration.noise_rng"):
            noise = rng.normal(0.0, noise_std, y_clean.shape).astype(np.float32)
        with profiler.timed("calibration.add_noise"):
            y_noisy = y_clean + noise
        noisy_rdm = get_rdm_vector_torch_profiled(
            y_noisy,
            metric,
            device,
            profiler,
            "calibration_rdm",
        )
        vals.append(
            calculate_correlation_torch_profiled(
                noisy_rdm,
                clean_rdm,
                corr_type,
                device,
                profiler,
            )
        )
    with profiler.timed("calibration.nanmean"):
        return float(np.nanmean(vals))


def calibrate_response_noise_for_rdm_reliability_profiled(
    y_clean: np.ndarray,
    *,
    target_reliability: float,
    metric: str,
    corr_type: str,
    rng: np.random.Generator,
    n_samples: int,
    max_iter: int,
    device: torch.device,
    profiler: Profiler,
) -> tuple[float, float]:
    if target_reliability <= 0:
        target_reliability = 1e-6
    if target_reliability >= 1:
        return 0.0, 1.0
    clean_rdm = get_rdm_vector_torch_profiled(
        y_clean,
        metric,
        device,
        profiler,
        "calibration_clean_rdm",
    )
    lo = 0.0
    hi = max(math.sqrt(1.0 / target_reliability - 1.0), 1e-3)
    for _ in range(10):
        rel = empirical_response_noise_rdm_reliability_profiled(
            y_clean,
            clean_rdm,
            hi,
            metric=metric,
            corr_type=corr_type,
            rng=rng,
            n_samples=n_samples,
            device=device,
            profiler=profiler,
        )
        if np.isfinite(rel) and rel <= target_reliability:
            break
        hi *= 2.0
    best_std = hi
    best_rel = empirical_response_noise_rdm_reliability_profiled(
        y_clean,
        clean_rdm,
        hi,
        metric=metric,
        corr_type=corr_type,
        rng=rng,
        n_samples=n_samples,
        device=device,
        profiler=profiler,
    )
    best_err = abs(best_rel - target_reliability) if np.isfinite(best_rel) else np.inf
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rel = empirical_response_noise_rdm_reliability_profiled(
            y_clean,
            clean_rdm,
            mid,
            metric=metric,
            corr_type=corr_type,
            rng=rng,
            n_samples=n_samples,
            device=device,
            profiler=profiler,
        )
        err = abs(rel - target_reliability) if np.isfinite(rel) else np.inf
        if err < best_err:
            best_std = mid
            best_rel = rel
            best_err = err
        if not np.isfinite(rel) or rel < target_reliability:
            hi = mid
        else:
            lo = mid
    return float(best_std), float(best_rel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-set", default="all_models")
    parser.add_argument(
        "--payload-root",
        type=Path,
        default=ROOT / "00_stimulus_selection" / "results" / "selected_stimuli",
    )
    parser.add_argument(
        "--random-feature-dir",
        type=Path,
        default=ROOT
        / "shared"
        / "cache_or_heavy"
        / "natural_pool_subset_100k_seed42",
    )
    parser.add_argument(
        "--encoding-root",
        type=Path,
        default=ROOT
        / "01_brain_model_alignment"
        / "results"
        / "encoding_models"
        / "shared_subject_encoding_models"
        / "encoding_20251222_141301",
    )
    parser.add_argument(
        "--model-list-csv",
        type=Path,
        default=ROOT / "00_stimulus_selection" / "resources" / "model_list.csv",
    )
    parser.add_argument("--profile-encoding", action="store_true")
    parser.add_argument(
        "--score-encoding-targets",
        action="store_true",
        help="Use the profiled encoded targets as teacher targets instead of raw features.",
    )
    parser.add_argument("--encoding-name", default="sub-01")
    parser.add_argument("--encoding-roi-subset", default="hlvis")
    parser.add_argument("--encoding-device", default="cuda")
    parser.add_argument("--encoding-batch-size", type=int, default=1024)
    parser.add_argument(
        "--encoding-cache-dir",
        type=Path,
        default=ROOT / "shared" / "cache_or_heavy" / "encoding_param_npy_cache",
    )
    parser.add_argument("--use-encoding-cache", action="store_true")
    parser.add_argument("--build-encoding-cache", action="store_true")
    parser.add_argument("--encoding-empty-cache", action="store_true")
    parser.add_argument("--n-random-images", type=int, default=100000)
    parser.add_argument("--refit-pool-size", type=int, default=100)
    parser.add_argument("--refit-val-size", type=int, default=20)
    parser.add_argument("--max-refit-pool-size", type=int, default=10000)
    parser.add_argument("--n-random-subsets", type=int, default=100)
    parser.add_argument("--n-noise-samples", type=int, default=1)
    parser.add_argument(
        "--batch-noise-samples",
        action="store_true",
        help="Batch all noise samples for each noise level in the hot scoring loop.",
    )
    parser.add_argument(
        "--gpu-predict-batch",
        action="store_true",
        help="In batched-noise mode, do all-eval prediction matmuls on the RDM CUDA device.",
    )
    parser.add_argument(
        "--gpu-alpha-batch",
        action="store_true",
        help="In batched-noise mode, select target-wise alphas and pack coefficients on the RDM CUDA device.",
    )
    parser.add_argument(
        "--gpu-eval-noise-batch",
        action="store_true",
        help="In batched-noise mode, generate noisy eval responses on the RDM CUDA device.",
    )
    parser.add_argument("--noise-mults", default="1")
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--teacher-indices", default="0")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--corr-type", choices=["spearman", "pearson"], default="spearman")
    parser.add_argument("--base-noise-ceiling", type=float, default=None)
    parser.add_argument("--calibration-images", type=int, default=100)
    parser.add_argument("--calibration-noise-samples", type=int, default=2)
    parser.add_argument("--calibration-max-iter", type=int, default=8)
    parser.add_argument("--rdm-device", default="cpu")
    parser.add_argument(
        "--synchronize-cuda-timings",
        action="store_true",
        help="Synchronize CUDA before/after every timed block for accurate GPU attribution.",
    )
    parser.add_argument("--target-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "profile_reports",
    )
    parser.add_argument("--summary-rows", type=int, default=80)
    args = parser.parse_args()

    if args.rdm_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--rdm-device cuda requested but CUDA is unavailable")
    if args.gpu_predict_batch and args.rdm_device != "cuda":
        raise ValueError("--gpu-predict-batch requires --rdm-device cuda")
    if args.gpu_alpha_batch and args.rdm_device != "cuda":
        raise ValueError("--gpu-alpha-batch requires --rdm-device cuda")
    if args.gpu_alpha_batch and not args.gpu_predict_batch:
        raise ValueError("--gpu-alpha-batch requires --gpu-predict-batch")
    if args.gpu_eval_noise_batch and args.rdm_device != "cuda":
        raise ValueError("--gpu-eval-noise-batch requires --rdm-device cuda")
    if args.profile_encoding and args.encoding_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--encoding-device cuda requested but CUDA is unavailable")
    if args.score_encoding_targets and not args.profile_encoding:
        raise ValueError("--score-encoding-targets requires --profile-encoding")
    rdm_device = torch.device(args.rdm_device)
    encoding_device = torch.device(
        args.encoding_device if args.profile_encoding else "cpu"
    )
    profiler = Profiler(synchronize_cuda=args.synchronize_cuda_timings)
    wall_start = perf_counter()

    with profiler.timed("load.payload"):
        payload = load_payload(args.payload_root, args.model_set)
        payload_model_names = list(payload["model_names"])
        model_names = available_random_models(args.random_feature_dir, payload_model_names)
        if not model_names:
            raise ValueError("No requested models have random feature files")
        metric = args.metric or payload.get("config", {}).get("metric", "cosine")
        base_noise_ceiling = float(
            args.base_noise_ceiling
            or payload.get("config", {}).get("noise_ceiling_target", 0.46)
        )

    teacher_indices = parse_index_list(args.teacher_indices, len(model_names))
    teacher_idx_list = sorted(teacher_indices or set(range(len(model_names))))
    alphas = [float(x) for x in parse_csv_list(args.alphas)]
    noise_mults = np.asarray(parse_float_list(args.noise_mults), dtype=np.float64)

    with profiler.timed("load.selected_raw"):
        selected_raw = load_selected_raw_features(payload, model_names)
    with profiler.timed("load.random_features"):
        random_raw = load_random_features(
            args.random_feature_dir,
            model_names,
            args.n_random_images,
        )

    n_available = min(arr.shape[0] for arr in random_raw.values())
    n_selected = next(iter(selected_raw.values())).shape[0]
    if args.max_refit_pool_size + args.n_random_subsets * n_selected > n_available:
        raise ValueError("Not enough random images for disjoint refit/eval pools")

    with profiler.timed("setup.sample_indices"):
        rng = np.random.default_rng(
            args.seed + stable_seed(args.model_set, "standalone_hot_ops")
        )
        natural_pool_order = rng.permutation(n_available)
        refit_indices = natural_pool_order[: args.refit_pool_size]
        random_eval_pool = natural_pool_order[
            args.max_refit_pool_size : args.max_refit_pool_size
            + args.n_random_subsets * n_selected
        ]
        random_subset_indices = [
            random_eval_pool[
                subset_idx * n_selected : (subset_idx + 1) * n_selected
            ]
            for subset_idx in range(args.n_random_subsets)
        ]
        union_indices = np.unique(np.concatenate([refit_indices, *random_subset_indices]))
        union_lookup = {int(idx): pos for pos, idx in enumerate(union_indices)}
        refit_positions = np.asarray(
            [union_lookup[int(idx)] for idx in refit_indices],
            dtype=np.int64,
        )
        random_subset_positions = [
            np.asarray([union_lookup[int(idx)] for idx in subset], dtype=np.int64)
            for subset in random_subset_indices
        ]
        split_rng = np.random.default_rng(
            args.seed + stable_seed(args.model_set, "standalone_hot_ops_split")
        )
        refit_perm = split_rng.permutation(len(refit_positions))
        train_pos = refit_positions[refit_perm[: args.refit_pool_size - args.refit_val_size]]
        val_pos = refit_positions[
            refit_perm[
                args.refit_pool_size - args.refit_val_size : args.refit_pool_size
            ]
        ]

    random_raw_union = {}
    for model, arr in random_raw.items():
        with profiler.timed(
            "setup.slice_random_union_model",
            detail=f"{model}:shape={arr.shape}",
        ):
            random_raw_union[model] = arr[union_indices]

    eval_raw: dict[str, dict[str, np.ndarray]] = {}
    with profiler.timed("setup.build_eval_raw_selected", calls=len(model_names)):
        eval_raw["selected|0"] = selected_raw
    for subset_idx, pos in enumerate(random_subset_positions):
        with profiler.timed(
            "setup.build_eval_raw_random_subset",
            calls=len(model_names),
            detail=f"subset={subset_idx}",
        ):
            eval_raw[f"random|{subset_idx}"] = {
                model: random_raw_union[model][pos] for model in model_names
            }

    selected_target_source = selected_raw
    random_target_union_source = random_raw_union
    if args.profile_encoding:
        roi_subset = args.encoding_roi_subset or None
        encoding_params = load_encoding_params_profiled(
            encoding_root=args.encoding_root,
            model_list_csv=args.model_list_csv,
            encoding_name=args.encoding_name,
            model_names=model_names,
            device=encoding_device,
            roi_subset=roi_subset,
            cache_dir=args.encoding_cache_dir,
            use_cache=args.use_encoding_cache,
            build_cache=args.build_encoding_cache,
            profiler=profiler,
        )
        _selected_encoded = encode_raw_feature_arrays_profiled(
            raw_features=selected_raw,
            encoding_params=encoding_params,
            device=encoding_device,
            batch_size=args.encoding_batch_size,
            profiler=profiler,
            prefix="encoding_selected",
            empty_cache=args.encoding_empty_cache,
        )
        random_union_encoded = encode_raw_feature_arrays_profiled(
            raw_features=random_raw_union,
            encoding_params=encoding_params,
            device=encoding_device,
            batch_size=args.encoding_batch_size,
            profiler=profiler,
            prefix="encoding_random_union",
            empty_cache=args.encoding_empty_cache,
        )
        if args.score_encoding_targets:
            selected_target_source = _selected_encoded
            random_target_union_source = random_union_encoded

    candidate_ops: dict[str, RidgeOps] = {}
    for candidate in model_names:
        x = random_raw_union[candidate]
        eval_x = {key: raw_by_model[candidate] for key, raw_by_model in eval_raw.items()}
        standardized = standardize_from_train_profiled(
            x[train_pos],
            x[val_pos],
            *eval_x.values(),
            scale_by_sqrt_features=True,
            profiler=profiler,
            prefix="candidate_standardize",
        )
        eval_x_std = dict(zip(eval_x.keys(), standardized[2:]))
        candidate_ops[candidate] = build_kernel_ridge_ops_profiled(
            standardized[0],
            standardized[1],
            eval_x_std,
            alphas,
            profiler,
        )

    eval_target: dict[str, dict[str, np.ndarray]] = {}
    with profiler.timed("setup.build_eval_targets_selected", calls=len(model_names)):
        eval_target["selected|0"] = selected_target_source
    for subset_idx, pos in enumerate(random_subset_positions):
        with profiler.timed(
            "setup.build_eval_targets_random_subset",
            calls=len(model_names),
            detail=f"subset={subset_idx}",
        ):
            eval_target[f"random|{subset_idx}"] = {
                model: random_target_union_source[model][pos] for model in model_names
            }

    rows = []
    for teacher_idx in teacher_idx_list:
        teacher = model_names[teacher_idx]
        teacher_rng = np.random.default_rng(
            args.seed + stable_seed(args.model_set, teacher, "standalone_teacher")
        )
        clean_y = random_target_union_source[teacher]
        eval_y = {key: target_by_model[teacher] for key, target_by_model in eval_target.items()}
        if args.target_dim is not None and 0 < args.target_dim < clean_y.shape[1]:
            target_rng = np.random.default_rng(
                args.seed + stable_seed(args.model_set, teacher, "target_cols")
            )
            target_cols = np.sort(
                target_rng.choice(clean_y.shape[1], size=args.target_dim, replace=False)
            )
            clean_y = clean_y[:, target_cols]
            eval_y = {key: y[:, target_cols] for key, y in eval_y.items()}

        standardized_y = standardize_from_train_profiled(
            clean_y[train_pos],
            clean_y[val_pos],
            *eval_y.values(),
            scale_by_sqrt_features=False,
            profiler=profiler,
            prefix="teacher_standardize",
        )
        y_train_clean = standardized_y[0]
        y_val_clean = standardized_y[1]
        eval_y_clean = dict(zip(eval_y.keys(), standardized_y[2:]))

        eval_keys = tuple(eval_y_clean)
        reference_ops = candidate_ops[model_names[0]]
        if eval_keys != reference_ops.eval_keys:
            raise ValueError("Eval target keys do not match candidate eval projection keys")
        with profiler.timed("teacher.pack_eval_clean_targets", calls=len(eval_keys)):
            eval_y_clean_all = np.stack([eval_y_clean[key] for key in eval_keys], axis=0)

        if args.calibration_images > 0 and args.calibration_images < y_train_clean.shape[0]:
            calib_rng = np.random.default_rng(
                args.seed + stable_seed(args.model_set, teacher, "calibration_subset")
            )
            calib_idx = np.sort(
                calib_rng.choice(
                    y_train_clean.shape[0],
                    size=args.calibration_images,
                    replace=False,
                )
            )
            y_calib_clean = y_train_clean[calib_idx]
        else:
            y_calib_clean = y_train_clean

        for noise_mult in noise_mults:
            noise_mult = float(noise_mult)
            noise_ceiling = multiplier_to_noise_ceiling(noise_mult, base_noise_ceiling)
            response_noise_std, achieved = (
                calibrate_response_noise_for_rdm_reliability_profiled(
                    y_calib_clean,
                    target_reliability=noise_ceiling,
                    metric=metric,
                    corr_type=args.corr_type,
                    rng=np.random.default_rng(
                        args.seed
                        + stable_seed(
                            args.model_set,
                            teacher,
                            noise_mult,
                            "standalone_calibration",
                        )
                    ),
                    n_samples=args.calibration_noise_samples,
                    max_iter=args.calibration_max_iter,
                    device=rdm_device,
                    profiler=profiler,
                )
            )
            if args.batch_noise_samples:
                rows.extend(
                    run_batched_noise_samples_profiled(
                        model_names=model_names,
                        candidate_ops=candidate_ops,
                        teacher=teacher,
                        teacher_idx=teacher_idx,
                        y_train_clean=y_train_clean,
                        y_val_clean=y_val_clean,
                        eval_y_clean_all=eval_y_clean_all,
                        eval_keys=eval_keys,
                        noise_mult=noise_mult,
                        response_noise_std=response_noise_std,
                        achieved=achieved,
                        n_noise_samples=args.n_noise_samples,
                        metric=metric,
                        corr_type=args.corr_type,
                        rdm_device=rdm_device,
                        gpu_alpha_batch=args.gpu_alpha_batch,
                        gpu_predict_batch=args.gpu_predict_batch,
                        gpu_eval_noise_batch=args.gpu_eval_noise_batch,
                        teacher_rng=teacher_rng,
                        profiler=profiler,
                    )
                )
                continue
            for noise_sample_idx in range(args.n_noise_samples):
                with profiler.timed("noise.train_rng"):
                    train_noise = teacher_rng.normal(
                        0.0,
                        response_noise_std,
                        y_train_clean.shape,
                    ).astype(np.float32)
                with profiler.timed("noise.val_rng"):
                    val_noise = teacher_rng.normal(
                        0.0,
                        response_noise_std,
                        y_val_clean.shape,
                    ).astype(np.float32)
                with profiler.timed("noise.add_train_val"):
                    y_train = y_train_clean + train_noise
                    y_val = y_val_clean + val_noise

                with profiler.timed("noise.eval_rng_batched", calls=len(eval_keys)):
                    eval_noise = teacher_rng.normal(
                        0.0,
                        response_noise_std,
                        eval_y_clean_all.shape,
                    ).astype(np.float32)
                with profiler.timed("noise.eval_add_batched", calls=len(eval_keys)):
                    y_eval_noisy_all = eval_y_clean_all + eval_noise
                noisy_teacher_rdm_batch = get_rdm_batch_tensor_profiled(
                    y_eval_noisy_all,
                    metric,
                    rdm_device,
                    profiler,
                    "noisy_eval_rdm_batch",
                )
                teacher_reference_batch = prepare_reference_vectors_profiled(
                    noisy_teacher_rdm_batch,
                    args.corr_type,
                    profiler,
                )

                scores_by_eval = {
                    key: np.full(len(model_names), np.nan, dtype=np.float32)
                    for key in eval_keys
                }
                for candidate_idx, candidate in enumerate(model_names):
                    ops = candidate_ops[candidate]
                    alpha_values, best_alpha_idx, coefficient_cache = (
                        select_targetwise_alpha_profiled(
                            ops,
                            y_train,
                            y_val,
                            profiler,
                        )
                    )
                    coeff_selected = pack_selected_coefficients_profiled(
                        alpha_values,
                        best_alpha_idx,
                        coefficient_cache,
                        profiler,
                    )
                    pred_all = predict_all_eval_profiled(
                        ops,
                        coeff_selected,
                        profiler,
                    )
                    pred_batch = reshape_concatenated_eval_matrix(
                        pred_all,
                        ops.eval_lengths,
                    )
                    pred_rdm_batch = get_rdm_batch_tensor_profiled(
                        pred_batch,
                        metric,
                        rdm_device,
                        profiler,
                        "prediction_rdm_batch",
                    )
                    candidate_scores = score_prediction_batch_profiled(
                        pred_rdm_batch,
                        teacher_reference_batch,
                        args.corr_type,
                        profiler,
                    )
                    for key, score in zip(ops.eval_keys, candidate_scores):
                        scores_by_eval[key][candidate_idx] = score

                with profiler.timed("recover.argmax_all_eval_sets"):
                    for key, scores in scores_by_eval.items():
                        scores = np.nan_to_num(scores, nan=-np.inf)
                        recovered_idx = int(np.argmax(scores))
                        rows.append(
                            {
                                "teacher": teacher,
                                "noise_mult": noise_mult,
                                "noise_sample_idx": noise_sample_idx,
                                "eval_key": key,
                                "recovered": model_names[recovered_idx],
                                "teacher_score": float(scores[teacher_idx]),
                                "best_score": float(scores[recovered_idx]),
                                "achieved_fit_rdm_reliability": float(achieved),
                            }
                        )

    wall_s = perf_counter() - wall_start
    summary = profiler.summary_frame()
    detail = profiler.detail_frame()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    stem = f"standalone_hot_ops_{safe_name(args.model_set)}_{stamp}_pid{os.getpid()}"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    detail_path = args.output_dir / f"{stem}_detail.csv"
    rows_path = args.output_dir / f"{stem}_recoveries.csv"
    metadata_path = args.output_dir / f"{stem}_metadata.json"
    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    with metadata_path.open("w") as f:
        json.dump(
            {
                "wall_time_s": wall_s,
                "model_set": args.model_set,
                "model_names": model_names,
                "teacher_indices": teacher_idx_list,
                "n_selected": int(n_selected),
                "n_random_subsets": int(args.n_random_subsets),
                "n_eval_sets": int(1 + args.n_random_subsets),
                "n_noise_samples": int(args.n_noise_samples),
                "batch_noise_samples": bool(args.batch_noise_samples),
                "gpu_alpha_batch": bool(args.gpu_alpha_batch),
                "gpu_predict_batch": bool(args.gpu_predict_batch),
                "gpu_eval_noise_batch": bool(args.gpu_eval_noise_batch),
                "noise_mults": noise_mults.tolist(),
                "refit_pool_size": int(args.refit_pool_size),
                "refit_train_n": int(args.refit_pool_size - args.refit_val_size),
                "refit_val_n": int(args.refit_val_size),
                "metric": metric,
                "corr_type": args.corr_type,
                "rdm_device": str(rdm_device),
                "synchronize_cuda_timings": bool(args.synchronize_cuda_timings),
                "profile_encoding": bool(args.profile_encoding),
                "score_encoding_targets": bool(args.score_encoding_targets),
                "encoding_name": args.encoding_name if args.profile_encoding else None,
                "encoding_root": str(args.encoding_root) if args.profile_encoding else None,
                "encoding_roi_subset": (
                    args.encoding_roi_subset if args.profile_encoding else None
                ),
                "encoding_device": (
                    str(encoding_device) if args.profile_encoding else None
                ),
                "encoding_batch_size": (
                    int(args.encoding_batch_size) if args.profile_encoding else None
                ),
                "encoding_cache_dir": (
                    str(args.encoding_cache_dir) if args.profile_encoding else None
                ),
                "use_encoding_cache": (
                    bool(args.use_encoding_cache) if args.profile_encoding else None
                ),
                "build_encoding_cache": (
                    bool(args.build_encoding_cache) if args.profile_encoding else None
                ),
                "encoding_empty_cache": (
                    bool(args.encoding_empty_cache) if args.profile_encoding else None
                ),
                "base_noise_ceiling": float(base_noise_ceiling),
                "summary_csv": str(summary_path),
                "detail_csv": str(detail_path),
                "recoveries_csv": str(rows_path),
            },
            f,
            indent=2,
        )

    print(f"Wall time: {wall_s:.3f}s")
    print(f"Summary CSV: {summary_path}")
    print(f"Detail CSV: {detail_path}")
    print(f"Recoveries CSV: {rows_path}")
    print(f"Metadata: {metadata_path}")
    print()
    print_markdown_table(summary, max_rows=args.summary_rows)


if __name__ == "__main__":
    main()
