#!/usr/bin/env python3
"""Feature-only selection-method sweep with corrected recovery evaluation.

This script is intentionally separate from the production stimulus selector. It
keeps the feature-level ingredients fixed, skips image filtering/refinement, and
compares several greedy selection objectives on the same candidate pool.

After selection it writes minimal selection payloads and runs the isolated
noisy-by-clean recovery evaluator on each method.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import trange

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
SRC_DIR = ROOT / "src"
for path in (SRC_DIR,):
    sys.path.insert(0, str(path))

from cstims.data_loader import (  # noqa: E402
    build_selected_image_records,
    load_natural_features_with_metadata,
    max_images_for_ram,
)
from cstims.encoding.linear import (  # noqa: E402
    EncodingParamsByEncoding,
    encode_batch_for_all_encodings,
    load_encoding_params_by_encoding,
)
from cstims.noise_estimation import rdm_noise_by_model  # noqa: E402
from cstims.rdm_cuda import get_rdm_vector  # noqa: E402
from cstims.selection.primitives import (  # noqa: E402
    compute_pairwise_distances,
)

ENV_CONFIG_ROOT = ROOT / "00_stimulus_selection" / "resources" / "configs" / "paths"
MODEL_SET_CONFIG_ROOT = (
    ROOT / "00_stimulus_selection" / "resources" / "configs" / "model_set"
)
MODEL_LIST_CSV = ROOT / "00_stimulus_selection" / "resources" / "model_list.csv"
RECOVERY_SCRIPT = (
    ROOT
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "feature_method_sweep_recovery"
    / "noisy_by_clean"
    / "01_compute_recovery.py"
)
RECOVERY_RESULTS_ROOT = (
    ROOT
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "feature_method_sweep_recovery"
    / "noisy_by_clean"
    / "results"
)
UNIQUE_ENCODING_ROOT = (
    ROOT
    / "01_brain_model_alignment"
    / "results"
    / "encoding_models"
    / "subject_unique_encoding_models"
    / "runs"
)
UNIQUE_ENCODING_DIRS = {
    "sub-01": UNIQUE_ENCODING_ROOT / "20260317_170621",
    "sub-03": UNIQUE_ENCODING_ROOT / "20260319_152751",
    "sub-05": UNIQUE_ENCODING_ROOT / "20260317_170621",
    "sub-06": UNIQUE_ENCODING_ROOT / "20260319_152752",
    "sub-07": UNIQUE_ENCODING_ROOT / "20260317_170621",
}

ENCODING_TRACKS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")


@dataclass(frozen=True)
class TrackSpec:
    name: str
    type: str
    encoding_name: str | None = None


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    label: str
    tracks: tuple[TrackSpec, ...]
    track_agg_method: str
    track_norm_method: str
    within: str
    across: str
    weights: dict[str, float] | None = None
    raw_weight: float | None = None
    objective_noise_ceiling_target: float | None = None
    summary_weights: dict[str, float] = field(default_factory=dict)
    description: str = ""


@dataclass
class TrackRuntime:
    spec: TrackSpec
    selected_features: dict[str, torch.Tensor]
    rdm_by_model: dict[str, torch.Tensor]
    noise_vars: torch.Tensor
    var_noise_by_model: dict[str, float]
    rdm_len: int
    rdm_sum: torch.Tensor
    rdm_sumsq: torch.Tensor
    rdm_dot: torch.Tensor


@dataclass
class MethodRuntime:
    spec: MethodSpec
    current_indices: list[int]
    pool_mask: np.ndarray
    tracks: dict[str, TrackRuntime]
    scores_combined: list[float] = field(default_factory=list)
    scores_per_track_history: dict[str, list[float]] = field(default_factory=dict)
    trace_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PoolRun:
    pool_size: int
    output_root: Path
    payload_root: Path
    eval_root: Path
    comparison_root: Path
    run_config: dict[str, Any]
    runtimes: dict[str, MethodRuntime] = field(default_factory=dict)


def all_track_specs() -> tuple[TrackSpec, ...]:
    return (
        *(TrackSpec(name=name, type="encoding", encoding_name=name) for name in ENCODING_TRACKS),
        TrackSpec(name="raw", type="identity"),
    )


def weighted_raw_encoding_weights(raw_weight: float = 0.5) -> dict[str, float]:
    per_encoding = (1.0 - raw_weight) / len(ENCODING_TRACKS)
    weights = {"raw": raw_weight}
    weights.update({name: per_encoding for name in ENCODING_TRACKS})
    return weights


def default_methods() -> list[MethodSpec]:
    raw = (TrackSpec(name="raw", type="identity"),)
    sub01 = (TrackSpec(name="sub-01", type="encoding", encoding_name="sub-01"),)
    all_tracks = all_track_specs()
    w05 = weighted_raw_encoding_weights(0.5)
    return [
        MethodSpec(
            method_id="raw_only_mean_min",
            label="Raw only, mean/min",
            tracks=raw,
            track_agg_method="identity",
            track_norm_method="none",
            within="mean",
            across="min",
            summary_weights={"raw": 1.0},
            description="Raw feature track only; old model-level mean alternative, min source-model objective.",
        ),
        MethodSpec(
            method_id="raw_only_mean_min_no_attenuation",
            label="Raw only, mean/min, no attenuation",
            tracks=raw,
            track_agg_method="identity",
            track_norm_method="none",
            within="mean",
            across="min",
            objective_noise_ceiling_target=1.0,
            summary_weights={"raw": 1.0},
            description=(
                "Raw feature track only with zero objective noise, i.e. no "
                "analytical attenuation. Recovery evaluation still uses the "
                "run-level target noise."
            ),
        ),
        MethodSpec(
            method_id="sub01_only_mean_min",
            label="sub-01 only, mean/min",
            tracks=sub01,
            track_agg_method="identity",
            track_norm_method="none",
            within="mean",
            across="min",
            summary_weights={"sub-01": 1.0},
            description="Single sub-01 encoding track only.",
        ),
        MethodSpec(
            method_id="sub01_only_mean_min_no_attenuation",
            label="sub-01 only, mean/min, no attenuation",
            tracks=sub01,
            track_agg_method="identity",
            track_norm_method="none",
            within="mean",
            across="min",
            objective_noise_ceiling_target=1.0,
            summary_weights={"sub-01": 1.0},
            description=(
                "Single sub-01 encoding track with zero objective noise, i.e. "
                "no analytical attenuation. Recovery evaluation still uses the "
                "run-level target noise."
            ),
        ),
        MethodSpec(
            method_id="paper_effective_identity_sub01_mean_min",
            label="Paper effective identity sub-01",
            tracks=all_tracks,
            track_agg_method="identity",
            track_norm_method="zscore",
            within="mean",
            across="min",
            raw_weight=0.5,
            summary_weights={"sub-01": 1.0},
            description=(
                "Configured like the frozen paper run: raw plus five subject encodings, "
                "but identity track aggregation makes sub-01 the effective selection score."
            ),
        ),
        MethodSpec(
            method_id="raw_enc_w05_mean_min",
            label="Raw+enc w0.5, mean/min",
            tracks=all_tracks,
            track_agg_method="weighted_mean",
            track_norm_method="zscore",
            within="mean",
            across="min",
            weights=w05,
            raw_weight=0.5,
            summary_weights=w05,
            description="Corrected intended weighted raw-plus-all-encodings method with old model-level objective.",
        ),
        MethodSpec(
            method_id="raw_enc_w05_mean_min_no_attenuation",
            label="Raw+enc w0.5, mean/min, no attenuation",
            tracks=all_tracks,
            track_agg_method="weighted_mean",
            track_norm_method="zscore",
            within="mean",
            across="min",
            weights=w05,
            raw_weight=0.5,
            objective_noise_ceiling_target=1.0,
            summary_weights=w05,
            description=(
                "Corrected weighted raw-plus-all-encodings method with zero "
                "objective noise, i.e. no analytical attenuation. Recovery "
                "evaluation still uses the run-level target noise."
            ),
        ),
        MethodSpec(
            method_id="raw_enc_w05_max_mean",
            label="Raw+enc w0.5, max/mean",
            tracks=all_tracks,
            track_agg_method="weighted_mean",
            track_norm_method="zscore",
            within="max",
            across="mean",
            weights=w05,
            raw_weight=0.5,
            summary_weights=w05,
            description="Hardest-alternative margin, averaged over source models.",
        ),
        MethodSpec(
            method_id="raw_enc_w05_max_min",
            label="Raw+enc w0.5, max/min",
            tracks=all_tracks,
            track_agg_method="weighted_mean",
            track_norm_method="zscore",
            within="max",
            across="min",
            weights=w05,
            raw_weight=0.5,
            summary_weights=w05,
            description="Hardest-alternative margin, worst source model.",
        ),
        MethodSpec(
            method_id="paper_effective_identity_sub01_mean_min_no_attenuation",
            label="Paper effective identity sub-01, no attenuation",
            tracks=all_tracks,
            track_agg_method="identity",
            track_norm_method="zscore",
            within="mean",
            across="min",
            raw_weight=0.5,
            objective_noise_ceiling_target=1.0,
            summary_weights={"sub-01": 1.0},
            description=(
                "Same effective selector as the frozen paper run (identity aggregation, "
                "therefore sub-01 only; mean alternative, min source model), but with "
                "zero objective noise, i.e. no analytical attenuation. Recovery "
                "evaluation still uses the run-level target noise."
            ),
        ),
    ]


def parse_method_filter(value: str | None, methods: list[MethodSpec]) -> list[MethodSpec]:
    if not value:
        return methods
    keep = {item.strip() for item in value.split(",") if item.strip()}
    unknown = sorted(keep - {method.method_id for method in methods})
    if unknown:
        raise ValueError(f"Unknown method id(s): {unknown}")
    return [method for method in methods if method.method_id in keep]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_env_paths(env: str) -> dict[str, Any]:
    config = load_yaml(ENV_CONFIG_ROOT / f"{env}.yaml")
    paths = dict(config.get("paths", {}))
    if not Path(paths.get("model_list_csv", "")).exists() and MODEL_LIST_CSV.exists():
        paths["model_list_csv"] = str(MODEL_LIST_CSV)
    return paths


def load_model_set(model_set: str) -> tuple[str, list[str]]:
    config = load_yaml(MODEL_SET_CONFIG_ROOT / f"{model_set}.yaml")
    return config.get("model_set_name", model_set), list(config["model_names"])


def load_encoding_params_for_sweep(
    *,
    paths: dict[str, Any],
    model_list_csv: Path,
    encoding_names: list[str],
    device: torch.device,
    roi_subset: str | None,
    shared_encodings: bool,
) -> EncodingParamsByEncoding:
    if not encoding_names:
        return {}

    if shared_encodings:
        return load_encoding_params_by_encoding(
            encoding_root=Path(paths["encoding_root"]),
            model_list_csv=model_list_csv,
            encoding_names=encoding_names,
            device=device,
            roi_subset=roi_subset,
        )

    params: EncodingParamsByEncoding = {}
    for encoding_name in encoding_names:
        encoding_root = UNIQUE_ENCODING_DIRS.get(encoding_name)
        if encoding_root is None:
            raise ValueError(
                f"No unique encoding root configured for encoding '{encoding_name}'. "
                "Use --shared-encodings to load all encodings from paths.encoding_root."
            )
        if not encoding_root.exists():
            raise FileNotFoundError(
                f"Unique encoding root for '{encoding_name}' does not exist: {encoding_root}"
            )
        loaded = load_encoding_params_by_encoding(
            encoding_root=encoding_root,
            model_list_csv=model_list_csv,
            encoding_names=[encoding_name],
            device=device,
            roi_subset=roi_subset,
        )
        params.update(loaded)
    return params


def load_layer_names(model_list_csv: Path, model_names: list[str]) -> list[str]:
    import csv

    with model_list_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    layer_by_model = {row["model"]: row["layer"] for row in rows}
    return [layer_by_model[name] for name in model_names]


def _npz_feature_key(data: np.lib.npyio.NpzFile, path: Path) -> str:
    if "features" in data.files:
        return "features"
    candidates = [
        key
        for key in data.files
        if not key.startswith("_") and getattr(data[key], "ndim", 0) >= 2
    ]
    if not candidates:
        raise ValueError(f"No 2D feature array found in {path}")
    return candidates[0]


def load_npz_pool_features(
    pool_feature_dir: Path,
    model_names: list[str],
    max_images: int | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    pool_feature_dir = pool_feature_dir.resolve()
    if not pool_feature_dir.exists():
        raise FileNotFoundError(f"Pool feature directory not found: {pool_feature_dir}")

    manifest_path = pool_feature_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            manifest = json.load(f)

    features: dict[str, np.ndarray] = {}
    n_rows_by_model: dict[str, int] = {}
    for model_name in model_names:
        path = pool_feature_dir / f"{model_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing pool features for {model_name}: {path}")
        with np.load(path, allow_pickle=True) as data:
            key = _npz_feature_key(data, path)
            array = np.asarray(data[key])
        if array.ndim != 2:
            raise ValueError(f"{path} feature array must be 2D, got shape {array.shape}")
        n_rows_by_model[model_name] = int(array.shape[0])
        features[model_name] = array

    if len(set(n_rows_by_model.values())) != 1:
        raise ValueError(f"Pool feature row counts differ by model: {n_rows_by_model}")

    n_available = next(iter(n_rows_by_model.values()))
    n_take = n_available if max_images is None else int(max_images)
    if n_take > n_available:
        raise ValueError(
            f"Requested max_images={n_take}, but npz pool only has {n_available} rows"
        )
    if n_take <= 0:
        raise ValueError(f"Invalid npz pool image count: {n_take}")
    if n_take < n_available:
        features = {model: array[:n_take] for model, array in features.items()}

    source_indices_path = pool_feature_dir / "_sampled_indices.npy"
    source_indices = None
    if source_indices_path.exists():
        loaded = np.load(source_indices_path)
        if loaded.shape[0] < n_take:
            raise ValueError(
                f"{source_indices_path} has {loaded.shape[0]} rows, expected at least {n_take}"
            )
        source_indices = loaded[:n_take].astype(np.int64, copy=False)

    draw_order_path = pool_feature_dir / "_sampled_indices_draw_order.npy"
    records: list[dict[str, Any]] = []
    for pool_row in range(n_take):
        record = {
            "global_index": int(pool_row),
            "pool_row": int(pool_row),
            "pool_feature_dir": str(pool_feature_dir),
        }
        if source_indices is not None:
            record["source_global_index"] = int(source_indices[pool_row])
        records.append(record)

    pool_info = {
        "pool_feature_dir": str(pool_feature_dir),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "source_indices_path": str(source_indices_path) if source_indices_path.exists() else None,
        "draw_order_indices_path": str(draw_order_path) if draw_order_path.exists() else None,
        "n_available": int(n_available),
        "n_loaded": int(n_take),
        "features_key": manifest.get("features_key", "features"),
    }
    return features, records, pool_info


def build_selection_image_records(
    global_indices: list[int],
    raw_shard_slices: Any,
    pool_records_by_index: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if pool_records_by_index is None:
        return build_selected_image_records(global_indices, raw_shard_slices)

    records = []
    for idx in global_indices:
        if idx < 0 or idx >= len(pool_records_by_index):
            raise IndexError(f"Selected pool row {idx} out of range for npz pool")
        records.append(dict(pool_records_by_index[idx]))
    return records


def normalize_scores(scores: torch.Tensor, method: str) -> torch.Tensor:
    if method == "none":
        return scores
    if method == "zscore":
        return (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-6)
    if method == "minmax":
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    raise ValueError(f"Unknown track normalization method: {method}")


def aggregate_track_scores(
    scores_per_track: dict[str, torch.Tensor],
    method: MethodSpec,
) -> torch.Tensor:
    track_order = [track.name for track in method.tracks]
    normalized = {
        name: normalize_scores(scores_per_track[name], method.track_norm_method)
        for name in track_order
    }
    if method.track_agg_method == "identity":
        return normalized[track_order[0]]

    stacked = torch.stack([normalized[name] for name in track_order], dim=0)
    if method.track_agg_method == "weighted_mean":
        weights = method.weights or {name: 1.0 for name in track_order}
        weight_vec = torch.tensor(
            [weights.get(name, 0.0) for name in track_order],
            dtype=stacked.dtype,
            device=stacked.device,
        )
        if float(weight_vec.sum()) <= 0:
            raise ValueError(f"{method.method_id}: all track weights are zero")
        weight_vec = weight_vec / weight_vec.sum()
        return (weight_vec[:, None] * stacked).sum(dim=0)
    if method.track_agg_method == "mean":
        return stacked.mean(dim=0)
    if method.track_agg_method == "min":
        return stacked.min(dim=0).values
    raise ValueError(f"Unsupported track aggregation: {method.track_agg_method}")


def compute_model_utilities_custom(
    correlations: torch.Tensor,
    within: str,
) -> torch.Tensor:
    """Return per-source-model utility from expected noisy-by-clean correlations."""
    _, n_models, _ = correlations.shape
    r_self = torch.diagonal(correlations, dim1=1, dim2=2)
    diag = torch.eye(n_models, dtype=torch.bool, device=correlations.device).unsqueeze(0)

    if within == "mean":
        r_sum = correlations.sum(dim=2)
        r_other = (r_sum - r_self) / (n_models - 1)
    elif within == "max":
        masked = torch.where(diag, torch.full_like(correlations, -float("inf")), correlations)
        r_other = masked.max(dim=2).values
    elif within == "min":
        masked = torch.where(diag, torch.full_like(correlations, float("inf")), correlations)
        r_other = masked.min(dim=2).values
    else:
        raise ValueError(f"Unsupported within-model aggregation: {within}")

    return r_self - r_other


def aggregate_across_models_custom(utilities_per_model: torch.Tensor, across: str) -> torch.Tensor:
    if across == "mean":
        return utilities_per_model.mean(dim=1)
    if across == "min":
        return utilities_per_model.min(dim=1).values
    raise ValueError(f"Unsupported across-model aggregation: {across}")


def compute_rdm_stats(
    rdm_by_model: dict[str, torch.Tensor],
    model_names: list[str],
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    rdms = torch.stack([rdm_by_model[model] for model in model_names], dim=0)
    return (
        int(rdms.shape[1]),
        rdms.sum(dim=1),
        (rdms * rdms).sum(dim=1),
        rdms @ rdms.t(),
    )


@torch.no_grad()
def compute_track_scores(
    candidate_features: dict[str, torch.Tensor],
    runtime: TrackRuntime,
    metric: str,
    corr_type: str,
    within: str,
    across: str,
) -> torch.Tensor:
    if corr_type != "correlation":
        raise ValueError("This feature sweep currently supports analytical Pearson correlation only.")

    model_names = list(runtime.selected_features)
    batch_size = candidate_features[model_names[0]].shape[0]

    new_dissims = {
        model: compute_pairwise_distances(
            candidate_features[model],
            runtime.selected_features[model],
            metric=metric,
        )
        for model in model_names
    }
    new_dissims_tensor = torch.stack([new_dissims[model] for model in model_names], dim=1)

    total_len = runtime.rdm_len + int(new_dissims_tensor.shape[2])
    new_sum = new_dissims_tensor.sum(dim=2)
    new_sumsq = (new_dissims_tensor * new_dissims_tensor).sum(dim=2)
    new_dot = torch.bmm(new_dissims_tensor, new_dissims_tensor.transpose(1, 2))

    total_sum = runtime.rdm_sum.unsqueeze(0) + new_sum
    total_sumsq = runtime.rdm_sumsq.unsqueeze(0) + new_sumsq
    total_dot = runtime.rdm_dot.unsqueeze(0) + new_dot

    mean = total_sum / total_len
    rdm_vars = (total_sumsq / total_len - mean.square()).clamp_min(0.0)
    cov = total_dot / total_len - mean.unsqueeze(2) * mean.unsqueeze(1)
    std = torch.sqrt(rdm_vars) + 1e-8
    correlations = cov / (std.unsqueeze(2) * std.unsqueeze(1))
    correlations = torch.nan_to_num(correlations, nan=0.0)

    attenuation = torch.sqrt(rdm_vars / (rdm_vars + runtime.noise_vars.unsqueeze(0) + 1e-8))
    expected_correlations = correlations * attenuation.unsqueeze(2)

    utilities_per_model = compute_model_utilities_custom(expected_correlations, within)
    return aggregate_across_models_custom(utilities_per_model, across)


def get_track_candidate_features(
    track: TrackSpec,
    raw_batch: dict[str, torch.Tensor],
    encoded_batch: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if track.type == "identity":
        return raw_batch
    if track.type == "encoding" and track.encoding_name:
        return encoded_batch[track.encoding_name]
    raise ValueError(f"Unsupported track spec: {track}")


def get_track_selected_features(
    track: TrackSpec,
    raw_selected: dict[str, torch.Tensor],
    encoded_selected: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if track.type == "identity":
        return {model: tensor.clone() for model, tensor in raw_selected.items()}
    if track.type == "encoding" and track.encoding_name:
        return {
            model: tensor.clone()
            for model, tensor in encoded_selected[track.encoding_name].items()
        }
    raise ValueError(f"Unsupported track spec: {track}")


def build_runtime(
    method: MethodSpec,
    selected_indices: list[int],
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    encoding_params: EncodingParamsByEncoding,
    var_noise_by_track: dict[str, dict[str, float]],
    metric: str,
    device: torch.device,
    pool_size: int,
) -> MethodRuntime:
    raw_selected = {
        model: torch.from_numpy(raw_features_np[model][selected_indices]).to(
            device=device, dtype=torch.float32
        )
        for model in model_names
    }
    required_encodings = sorted(
        {track.encoding_name for track in method.tracks if track.type == "encoding" and track.encoding_name}
    )
    encoded_selected = (
        encode_batch_for_all_encodings(
            raw_selected,
            {name: encoding_params[name] for name in required_encodings},
        )
        if required_encodings
        else {}
    )

    tracks: dict[str, TrackRuntime] = {}
    for track in method.tracks:
        selected = get_track_selected_features(track, raw_selected, encoded_selected)
        rdm_by_model = {model: get_rdm_vector(selected[model], metric) for model in model_names}
        rdm_len, rdm_sum, rdm_sumsq, rdm_dot = compute_rdm_stats(rdm_by_model, model_names)
        noise = var_noise_by_track[track.name]
        if method.objective_noise_ceiling_target is None:
            objective_noise = noise
        elif method.objective_noise_ceiling_target >= 1.0:
            objective_noise = {model: 0.0 for model in model_names}
        else:
            raise ValueError(
                "Method-specific objective_noise_ceiling_target currently only "
                "supports values >= 1.0 for the no-attenuation objective."
            )
        tracks[track.name] = TrackRuntime(
            spec=track,
            selected_features=selected,
            rdm_by_model=rdm_by_model,
            noise_vars=torch.tensor(
                [objective_noise[model] for model in model_names],
                device=device,
                dtype=torch.float32,
            ),
            var_noise_by_model=dict(noise),
            rdm_len=rdm_len,
            rdm_sum=rdm_sum,
            rdm_sumsq=rdm_sumsq,
            rdm_dot=rdm_dot,
        )

    pool_mask = np.ones(pool_size, dtype=bool)
    pool_mask[np.asarray(selected_indices, dtype=np.int64)] = False
    return MethodRuntime(
        spec=method,
        current_indices=list(map(int, selected_indices)),
        pool_mask=pool_mask,
        tracks=tracks,
        scores_per_track_history={track.name: [] for track in method.tracks},
    )


def append_new_stimulus(
    runtime: MethodRuntime,
    new_idx: int,
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    encoding_params: EncodingParamsByEncoding,
    metric: str,
    device: torch.device,
) -> None:
    raw_new = {
        model: torch.from_numpy(raw_features_np[model][new_idx : new_idx + 1]).to(
            device=device, dtype=torch.float32
        )
        for model in model_names
    }
    required_encodings = sorted(
        {track.encoding_name for track in runtime.spec.tracks if track.type == "encoding" and track.encoding_name}
    )
    encoded_new = (
        encode_batch_for_all_encodings(
            raw_new,
            {name: encoding_params[name] for name in required_encodings},
        )
        if required_encodings
        else {}
    )

    for track in runtime.spec.tracks:
        track_runtime = runtime.tracks[track.name]
        new_features = get_track_selected_features(track, raw_new, encoded_new)
        for model in model_names:
            existing = track_runtime.selected_features[model]
            new_feat = new_features[model]
            new_dists = compute_pairwise_distances(new_feat, existing, metric=metric).squeeze()
            if new_dists.ndim == 0:
                new_dists = new_dists.unsqueeze(0)
            track_runtime.selected_features[model] = torch.cat([existing, new_feat], dim=0)
            track_runtime.rdm_by_model[model] = torch.cat(
                [track_runtime.rdm_by_model[model], new_dists]
            )
        (
            track_runtime.rdm_len,
            track_runtime.rdm_sum,
            track_runtime.rdm_sumsq,
            track_runtime.rdm_dot,
        ) = compute_rdm_stats(track_runtime.rdm_by_model, model_names)

    runtime.current_indices.append(int(new_idx))
    runtime.pool_mask[int(new_idx)] = False


def save_runtime_progress(
    runtime: MethodRuntime,
    payload_root: Path,
    raw_features_np: dict[str, np.ndarray],
    raw_shard_slices: Any,
    model_names: list[str],
    run_config: dict[str, Any],
    pool_records_by_index: list[dict[str, Any]] | None = None,
) -> None:
    method_dir = payload_root / runtime.spec.method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(runtime.current_indices, dtype=np.int64)
    np.save(method_dir / "selected_indices.npy", indices)
    pd.DataFrame(runtime.trace_rows).to_csv(method_dir / "selection_trace.csv", index=False)

    image_records = build_selection_image_records(
        indices.tolist(),
        raw_shard_slices,
        pool_records_by_index,
    )
    pd.DataFrame(image_records).to_csv(method_dir / "selected_image_records.csv", index=False)
    selected_source_indices = None
    if pool_records_by_index is not None and all(
        "source_global_index" in record for record in image_records
    ):
        selected_source_indices = np.asarray(
            [record["source_global_index"] for record in image_records],
            dtype=np.int64,
        )
        np.save(method_dir / "selected_source_global_indices.npy", selected_source_indices)

    selected_raw_np = {
        model: np.asarray(raw_features_np[model][indices], dtype=np.float32)
        for model in model_names
    }
    selected_features_by_encoding = {
        track_name: {
            model: runtime.tracks[track_name].selected_features[model].detach().cpu().numpy()
            for model in model_names
        }
        for track_name, track_runtime in runtime.tracks.items()
        if track_runtime.spec.type == "encoding"
    }

    track_definitions = [
        {
            "name": track.name,
            "type": track.type,
            **({"encoding_name": track.encoding_name} if track.encoding_name else {}),
        }
        for track in runtime.spec.tracks
    ]
    var_noise_payload = {
        track_name: track_runtime.var_noise_by_model
        for track_name, track_runtime in runtime.tracks.items()
    }
    objective_var_noise_payload = {
        track_name: {
            model: float(track_runtime.noise_vars[idx].detach().cpu().item())
            for idx, model in enumerate(model_names)
        }
        for track_name, track_runtime in runtime.tracks.items()
    }
    track_aggregation = {
        "norm_method": runtime.spec.track_norm_method,
        "agg_method": runtime.spec.track_agg_method,
        "raw_weight": runtime.spec.raw_weight,
        "weights": runtime.spec.weights,
    }
    config = {
        **run_config,
        "method_name": runtime.spec.method_id,
        "feature_method_sweep": True,
        "feature_method_spec": asdict(runtime.spec),
        "track_aggregation": track_aggregation,
    }

    payload = {
        "multi_view": len(track_definitions) > 1,
        "encoding_multi": False,
        "selected_global_indices": indices,
        "greedy_indices": indices,
        "best_raw_combined_indices": indices,
        "best_raw_combined_score": runtime.scores_combined[-1] if runtime.scores_combined else np.nan,
        "best_raw_combined_pass": -1,
        "model_names": model_names,
        "selected_image_records": image_records,
        "greedy_image_records": image_records,
        "best_raw_combined_image_records": image_records,
        "var_noise_by_model": var_noise_payload,
        "selection_objective_var_noise_by_model": objective_var_noise_payload,
        "scores": runtime.scores_combined,
        "selected_features_raw": selected_raw_np,
        "greedy_features_raw": selected_raw_np,
        "best_raw_combined_features_raw": selected_raw_np,
        "selected_features_by_encoding": selected_features_by_encoding or None,
        "greedy_features_by_encoding": selected_features_by_encoding or None,
        "best_raw_combined_features_by_encoding": selected_features_by_encoding or None,
        "selected_features_by_view": {
            "raw": selected_raw_np,
            **selected_features_by_encoding,
        },
        "scores_per_view_history": runtime.scores_per_track_history,
        "scores_per_rep_history": None,
        "refinement_history": [],
        "filter_records": [],
        "track_definitions": track_definitions,
        "track_aggregation": track_aggregation,
        "config": config,
    }
    if pool_records_by_index is not None:
        payload["selected_pool_indices"] = indices
        payload["greedy_pool_indices"] = indices
        payload["best_raw_combined_pool_indices"] = indices
    if selected_source_indices is not None:
        payload["selected_source_global_indices"] = selected_source_indices
        payload["greedy_source_global_indices"] = selected_source_indices
        payload["best_raw_combined_source_global_indices"] = selected_source_indices
    with (method_dir / "selected_stimuli_data.pkl").open("wb") as f:
        pickle.dump(payload, f)
    with (method_dir / "method_config.json").open("w") as f:
        json.dump(config, f, indent=2, default=str)


def save_manifest(methods: list[MethodSpec], payload_root: Path) -> None:
    rows = []
    for method in methods:
        rows.append(
            {
                "method_id": method.method_id,
                "label": method.label,
                "tracks": ",".join(track.name for track in method.tracks),
                "track_agg_method": method.track_agg_method,
                "track_norm_method": method.track_norm_method,
                "within": method.within,
                "across": method.across,
                "objective_noise_ceiling_target": method.objective_noise_ceiling_target,
                "raw_weight": method.raw_weight,
                "weights_json": json.dumps(method.weights or {}, sort_keys=True),
                "summary_weights_json": json.dumps(method.summary_weights, sort_keys=True),
                "description": method.description,
            }
        )
    pd.DataFrame(rows).to_csv(payload_root / "method_manifest.csv", index=False)


def calibrate_noise_by_track(
    raw_features_np: dict[str, np.ndarray],
    model_names: list[str],
    track_specs: list[TrackSpec],
    encoding_params: EncodingParamsByEncoding,
    metric: str,
    corr_type: str,
    target_nc: float,
    seed: int,
    device: torch.device,
    calib_n_examples: int,
    n_repeats: int,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    if any(track.type == "identity" for track in track_specs):
        print("Calibrating RDM-space noise for raw track")
        result["raw"] = rdm_noise_by_model(
            raw_features_np,
            model_names,
            device,
            metric=metric,
            target_nc=target_nc,
            calib_n_examples=calib_n_examples,
            n_repeats=n_repeats,
            seed=seed,
            corr_type=corr_type,
        )

    for track in track_specs:
        if track.type != "encoding" or not track.encoding_name or track.name in result:
            continue
        print(f"Calibrating RDM-space noise for encoding track {track.name}")
        n_calib = min(calib_n_examples, next(iter(raw_features_np.values())).shape[0])
        raw_calib = {
            model: torch.from_numpy(raw_features_np[model][:n_calib]).to(
                device=device, dtype=torch.float32
            )
            for model in model_names
        }
        encoded = encode_batch_for_all_encodings(
            raw_calib,
            {track.encoding_name: encoding_params[track.encoding_name]},
        )[track.encoding_name]
        encoded_np = {model: encoded[model].detach().cpu().numpy() for model in model_names}
        result[track.name] = rdm_noise_by_model(
            encoded_np,
            model_names,
            device,
            metric=metric,
            target_nc=target_nc,
            calib_n_examples=n_calib,
            n_repeats=n_repeats,
            seed=seed,
            corr_type=corr_type,
        )
        del raw_calib, encoded, encoded_np
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def load_existing_indices(method_dir: Path) -> list[int] | None:
    path = method_dir / "selected_indices.npy"
    if not path.exists():
        return None
    return [int(x) for x in np.load(path).tolist()]


def load_existing_trace(method_dir: Path) -> list[dict[str, Any]]:
    path = method_dir / "selection_trace.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{sec:04.1f}s"


def parse_count_token(token: str) -> int:
    value = token.strip().lower().replace("_", "")
    if not value:
        raise ValueError("Empty count token")
    multiplier = 1
    if value.endswith("k"):
        multiplier = 1_000
        value = value[:-1]
    elif value.endswith("m"):
        multiplier = 1_000_000
        value = value[:-1]
    count = int(float(value) * multiplier)
    if count <= 0:
        raise ValueError(f"Count must be positive, got {token!r}")
    return count


def parse_pool_sizes(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    sizes = sorted({parse_count_token(part) for part in value.split(",") if part.strip()})
    if not sizes:
        raise ValueError("--pool-sizes did not contain any valid sizes")
    return sizes


def pool_size_dir_name(pool_size: int) -> str:
    return f"pool_{pool_size:09d}"


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "cuda" in message and "out of memory" in message
    )


def write_selection_progress(output_root: Path, event: str, **payload: Any) -> None:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    latest_path = output_root / "selection_progress_latest.json"
    tmp_path = output_root / "selection_progress_latest.json.tmp"
    with tmp_path.open("w") as f:
        json.dump(record, f, indent=2, default=str)
    tmp_path.replace(latest_path)
    with (output_root / "selection_progress.jsonl").open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def run_selection(args: argparse.Namespace, methods: list[MethodSpec]) -> tuple[Path, Path, Path]:
    paths = load_env_paths(args.env)
    model_set_name, model_names = load_model_set(args.model_set)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    requested_pool_sizes = parse_pool_sizes(args.pool_sizes)
    multi_pool = requested_pool_sizes is not None

    model_list_csv = Path(paths["model_list_csv"])
    layer_names = load_layer_names(model_list_csv, model_names)

    pool_records_by_index: list[dict[str, Any]] | None = None
    pool_info: dict[str, Any] | None = None

    max_requested_images = max(requested_pool_sizes) if requested_pool_sizes else args.max_images
    if args.pool_feature_dir is not None:
        print(f"Loading npz feature pool from {args.pool_feature_dir.resolve()}")
        raw_features_np, pool_records_by_index, pool_info = load_npz_pool_features(
            pool_feature_dir=args.pool_feature_dir,
            model_names=model_names,
            max_images=max_requested_images,
        )
        raw_shard_slices = []
        max_images = int(next(iter(raw_features_np.values())).shape[0])
    else:
        if max_requested_images is not None:
            max_images = int(max_requested_images)
        else:
            max_images = max_images_for_ram(
                subset_root=Path(paths["subset_root"]),
                model_names=model_names,
                max_ram_bytes=int(args.max_ram_gb * 1024**3),
                model_csv=model_list_csv,
            )
    if max_images <= args.init_size:
        raise ValueError(f"Candidate pool too small: max_images={max_images}")

    pool_sizes = requested_pool_sizes or [max_images]
    if pool_sizes[-1] > max_images:
        raise ValueError(
            f"Largest requested pool size {pool_sizes[-1]} exceeds loaded max_images={max_images}"
        )
    too_small = [size for size in pool_sizes if size <= args.init_size]
    if too_small:
        raise ValueError(
            f"Pool sizes must be larger than init_size={args.init_size}: {too_small}"
        )

    if args.pool_feature_dir is None:
        print(f"Loading {max_images} images for model_set={model_set_name}")
        raw_features_np, raw_shard_slices = load_natural_features_with_metadata(
            subset_root=Path(paths["subset_root"]),
            preprocessed_dir=Path(paths["preprocessed_dirs"]["raw"]),
            model_names=model_names,
            layer_names=layer_names,
            max_images=max_images,
            model_csv=model_list_csv,
        )

    all_needed_tracks = {
        track.name: track
        for method in methods
        for track in method.tracks
    }
    required_encodings = sorted(
        {
            track.encoding_name
            for track in all_needed_tracks.values()
            if track.type == "encoding" and track.encoding_name
        }
    )
    encoding_params = load_encoding_params_for_sweep(
        paths=paths,
        model_list_csv=model_list_csv,
        encoding_names=required_encodings,
        device=device,
        roi_subset=args.encoding_roi_subset,
        shared_encodings=args.shared_encodings,
    )
    missing_encodings = sorted(set(required_encodings) - set(encoding_params))
    if missing_encodings:
        raise RuntimeError(f"Missing encoding params for: {missing_encodings}")

    var_noise_by_track = calibrate_noise_by_track(
        raw_features_np=raw_features_np,
        model_names=model_names,
        track_specs=list(all_needed_tracks.values()),
        encoding_params=encoding_params,
        metric=args.metric,
        corr_type=args.corr_type,
        target_nc=args.noise_ceiling_target,
        seed=args.seed,
        device=device,
        calib_n_examples=args.noise_calib_examples,
        n_repeats=args.noise_calib_repeats,
    )

    rng = np.random.default_rng(args.seed)
    initial_pool_size = min(pool_sizes)
    initial_indices = rng.choice(
        initial_pool_size,
        size=args.init_size,
        replace=False,
    ).astype(np.int64)
    base_run_config = {
        "model_set_name": model_set_name,
        "model_names": model_names,
        "paths": paths,
        "target_size": args.target_size,
        "init_size": args.init_size,
        "seed": args.seed,
        "metric": args.metric,
        "corr_type": args.corr_type,
        "use_analytical": True,
        "aggregation_within": "method_specific",
        "aggregation_across": "method_specific",
        "noise_ceiling_target": args.noise_ceiling_target,
        "noise_in_feature_space": False,
        "image_filter": {"enabled": False},
        "refinement": {"max_passes": 0, "min_replacements": 0},
        "max_ram_gb": args.max_ram_gb,
        "max_loaded_images": max_images,
        "candidate_pool": (
            str(args.pool_feature_dir.resolve())
            if args.pool_feature_dir is not None
            else "np.arange(max_loaded_images)"
        ),
        "pool_feature_dir": (
            str(args.pool_feature_dir.resolve())
            if args.pool_feature_dir is not None
            else None
        ),
        "pool_info": pool_info,
        "recovery_random_feature_dir": (
            str((args.random_feature_dir or args.pool_feature_dir).resolve())
            if (args.random_feature_dir or args.pool_feature_dir) is not None
            else None
        ),
        "recovery_n_random_images": args.n_random_images,
        "encoding_roi_subset": args.encoding_roi_subset,
        "adaptive_batch_size": args.adaptive_batch_size,
        "initial_batch_size": args.batch_size,
        "max_batch_size": args.max_batch_size,
        "min_batch_size": args.min_batch_size,
        "pool_sizes": pool_sizes,
        "multi_pool_size_sweep": multi_pool,
    }

    with (output_root / "run_config.json").open("w") as f:
        json.dump(base_run_config, f, indent=2, default=str)
    np.save(output_root / "pool_indices.npy", np.arange(max_images, dtype=np.int64))
    if pool_records_by_index is not None and all(
        "source_global_index" in record for record in pool_records_by_index
    ):
        source_indices = np.asarray(
            [record["source_global_index"] for record in pool_records_by_index],
            dtype=np.int64,
        )
        np.save(output_root / "pool_source_global_indices.npy", source_indices)
    else:
        source_indices = None

    pool_runs: dict[int, PoolRun] = {}
    for pool_size in pool_sizes:
        pool_output_root = (
            output_root / pool_size_dir_name(pool_size) if multi_pool else output_root
        )
        payload_root = pool_output_root / "payloads"
        eval_root = (
            RECOVERY_RESULTS_ROOT / output_root.name / pool_size_dir_name(pool_size) / "eval"
            if multi_pool
            else RECOVERY_RESULTS_ROOT / output_root.name / "eval"
        )
        comparison_root = (
            RECOVERY_RESULTS_ROOT
            / output_root.name
            / pool_size_dir_name(pool_size)
            / "comparison"
            if multi_pool
            else RECOVERY_RESULTS_ROOT / output_root.name / "comparison"
        )
        payload_root.mkdir(parents=True, exist_ok=True)
        eval_root.mkdir(parents=True, exist_ok=True)
        comparison_root.mkdir(parents=True, exist_ok=True)

        run_config = {
            **base_run_config,
            "max_images": pool_size,
            "candidate_pool_size": pool_size,
            "candidate_pool": (
                str(args.pool_feature_dir.resolve())
                if args.pool_feature_dir is not None
                else f"np.arange({pool_size})"
            ),
            "pool_size_sweep_output_root": str(output_root) if multi_pool else None,
        }
        with (pool_output_root / "run_config.json").open("w") as f:
            json.dump(run_config, f, indent=2, default=str)
        np.save(pool_output_root / "pool_indices.npy", np.arange(pool_size, dtype=np.int64))
        if source_indices is not None:
            np.save(pool_output_root / "pool_source_global_indices.npy", source_indices[:pool_size])
        save_manifest(methods, payload_root)
        pool_runs[pool_size] = PoolRun(
            pool_size=pool_size,
            output_root=pool_output_root,
            payload_root=payload_root,
            eval_root=eval_root,
            comparison_root=comparison_root,
            run_config=run_config,
        )

    pd.DataFrame(
        [
            {
                "pool_size": pool_run.pool_size,
                "pool_dir": pool_size_dir_name(pool_run.pool_size),
                "output_root": str(pool_run.output_root),
                "payload_root": str(pool_run.payload_root),
                "eval_root": str(pool_run.eval_root),
                "comparison_root": str(pool_run.comparison_root),
            }
            for pool_run in pool_runs.values()
        ]
    ).to_csv(output_root / "pool_size_manifest.csv", index=False)

    write_selection_progress(
        output_root,
        event="selection_started",
        model_set_name=model_set_name,
        max_images=max_images,
        pool_sizes=pool_sizes,
        target_size=args.target_size,
        init_size=args.init_size,
        methods=[method.method_id for method in methods],
    )

    for pool_run in pool_runs.values():
        for method in methods:
            method_dir = pool_run.payload_root / method.method_id
            existing = load_existing_indices(method_dir) if args.resume else None
            selected = existing if existing is not None else initial_indices.tolist()
            if max(selected) >= pool_run.pool_size:
                raise ValueError(
                    f"{method_dir} contains selected index outside pool_size={pool_run.pool_size}"
                )
            runtime = build_runtime(
                method=method,
                selected_indices=selected,
                raw_features_np=raw_features_np,
                model_names=model_names,
                encoding_params=encoding_params,
                var_noise_by_track=var_noise_by_track,
                metric=args.metric,
                device=device,
                pool_size=pool_run.pool_size,
            )
            runtime.trace_rows = load_existing_trace(method_dir) if args.resume else []
            if runtime.trace_rows:
                runtime.scores_combined = [
                    float(row["score_combined"]) for row in runtime.trace_rows
                ]
                for row in runtime.trace_rows:
                    for track in method.tracks:
                        key = f"score_{track.name}"
                        if key in row and pd.notna(row[key]):
                            runtime.scores_per_track_history[track.name].append(float(row[key]))
            pool_run.runtimes[method.method_id] = runtime
            save_runtime_progress(
                runtime,
                pool_run.payload_root,
                raw_features_np,
                raw_shard_slices,
                model_names,
                pool_run.run_config,
                pool_records_by_index=pool_records_by_index,
            )

    current_batch_size = int(args.batch_size)
    min_batch_size = max(1, int(args.min_batch_size))
    max_batch_size = int(args.max_batch_size) if int(args.max_batch_size) > 0 else current_batch_size
    max_batch_size = max(max_batch_size, current_batch_size)
    successful_batches_since_resize = 0

    for greedy_step in trange(args.init_size, args.target_size, desc="Feature-only greedy"):
        active = [
            (pool_run, method_id, runtime)
            for pool_run in pool_runs.values()
            for method_id, runtime in pool_run.runtimes.items()
            if len(runtime.current_indices) < args.target_size
        ]
        if not active:
            break

        greedy_iter = greedy_step - args.init_size + 1
        greedy_total = args.target_size - args.init_size
        iter_start_time = time.monotonic()
        pool_remaining = {
            f"{pool_run.pool_size}:{method_id}": int(runtime.pool_mask.sum())
            for pool_run, method_id, runtime in active
        }
        n_batches_estimate = (max_images + current_batch_size - 1) // current_batch_size
        print(
            f"[selection] greedy {greedy_iter}/{greedy_total}: "
            f"active_methods={len(active)}, batches~={n_batches_estimate}, "
            f"batch_size={current_batch_size}, "
            f"pool_remaining={pool_remaining}",
            flush=True,
        )
        write_selection_progress(
            output_root,
            event="greedy_iteration_start",
            greedy_iter=greedy_iter,
            greedy_total=greedy_total,
            active_methods=[
                {"pool_size": pool_run.pool_size, "method_id": method_id}
                for pool_run, method_id, _runtime in active
            ],
            n_batches_estimate=n_batches_estimate,
            batch_size=current_batch_size,
            pool_remaining=pool_remaining,
        )

        buffers: dict[tuple[int, str], dict[str, Any]] = {}
        for pool_run, method_id, runtime in active:
            n_candidates = int(runtime.pool_mask.sum())
            buffers[(pool_run.pool_size, method_id)] = {
                "candidate_indices": np.empty(n_candidates, dtype=np.int64),
                "scores_per_track": {
                    track.name: torch.empty(n_candidates, dtype=torch.float32)
                    for track in runtime.spec.tracks
                },
                "write_pos": 0,
            }

        start = 0
        batch_idx = 0
        while start < max_images:
            end = min(start + current_batch_size, max_images)
            raw_batch_all = None
            encoded_batch_all = None
            write_pos_snapshot = {
                key: int(buf["write_pos"]) for key, buf in buffers.items()
            }
            try:
                raw_batch_all = {
                    model: torch.from_numpy(raw_features_np[model][start:end]).to(
                        device=device, dtype=torch.float32
                    )
                    for model in model_names
                }
                encoded_batch_all = (
                    encode_batch_for_all_encodings(raw_batch_all, encoding_params)
                    if required_encodings
                    else {}
                )

                for pool_run, method_id, runtime in active:
                    if start >= pool_run.pool_size:
                        continue
                    candidate_stop = min(end, pool_run.pool_size)
                    batch_indices = np.arange(start, candidate_stop, dtype=np.int64)
                    valid_mask = runtime.pool_mask[batch_indices]
                    if not valid_mask.any():
                        continue
                    valid_candidate_indices = batch_indices[valid_mask]
                    valid_positions_np = (valid_candidate_indices - start).astype(np.int64)
                    valid_positions = torch.from_numpy(valid_positions_np).to(device=device)
                    buf = buffers[(pool_run.pool_size, method_id)]
                    write_pos = int(buf["write_pos"])
                    n_valid = len(valid_positions_np)
                    buf["candidate_indices"][write_pos : write_pos + n_valid] = (
                        valid_candidate_indices
                    )

                    raw_batch = {
                        model: tensor.index_select(0, valid_positions)
                        for model, tensor in raw_batch_all.items()
                    }
                    encoded_batch = {
                        enc: {
                            model: tensor.index_select(0, valid_positions)
                            for model, tensor in encoded.items()
                        }
                        for enc, encoded in encoded_batch_all.items()
                    }

                    for track in runtime.spec.tracks:
                        cand = get_track_candidate_features(track, raw_batch, encoded_batch)
                        scores = compute_track_scores(
                            candidate_features=cand,
                            runtime=runtime.tracks[track.name],
                            metric=args.metric,
                            corr_type=args.corr_type,
                            within=runtime.spec.within,
                            across=runtime.spec.across,
                        )
                        buf["scores_per_track"][track.name][write_pos : write_pos + n_valid] = (
                            scores.detach().cpu().to(torch.float32)
                        )
                    buf["write_pos"] = write_pos + n_valid

                del raw_batch_all, encoded_batch_all
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as exc:
                raw_batch_all = None
                encoded_batch_all = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if (
                    args.adaptive_batch_size
                    and is_cuda_oom(exc)
                    and current_batch_size > min_batch_size
                ):
                    for key, write_pos in write_pos_snapshot.items():
                        buffers[key]["write_pos"] = write_pos
                    new_batch_size = max(min_batch_size, current_batch_size // 2)
                    print(
                        f"[selection] CUDA OOM at batch_size={current_batch_size}; "
                        f"retrying start={start} with batch_size={new_batch_size}",
                        flush=True,
                    )
                    write_selection_progress(
                        output_root,
                        event="batch_size_backoff",
                        greedy_iter=greedy_iter,
                        greedy_total=greedy_total,
                        start=start,
                        old_batch_size=current_batch_size,
                        new_batch_size=new_batch_size,
                    )
                    current_batch_size = new_batch_size
                    successful_batches_since_resize = 0
                    continue
                raise

            batch_idx += 1
            successful_batches_since_resize += 1

            if (
                args.progress_every_batches > 0
                and (
                    batch_idx == 1
                    or end == max_images
                    or batch_idx % args.progress_every_batches == 0
                )
            ):
                elapsed = time.monotonic() - iter_start_time
                n_batches_estimate = (
                    (max_images - start + current_batch_size - 1) // current_batch_size
                )
                print(
                    f"[selection] greedy {greedy_iter}/{greedy_total}: "
                    f"batch {batch_idx}/~{batch_idx + max(0, n_batches_estimate - 1)} "
                    f"({end}/{max_images} images scanned), "
                    f"batch_size={current_batch_size}, elapsed={format_seconds(elapsed)}",
                    flush=True,
                )
                write_selection_progress(
                    output_root,
                    event="candidate_batch",
                    greedy_iter=greedy_iter,
                    greedy_total=greedy_total,
                    batch_idx=batch_idx,
                    n_batches_estimate=batch_idx + max(0, n_batches_estimate - 1),
                    batch_size=current_batch_size,
                    images_scanned=end,
                    max_images=max_images,
                    elapsed_seconds=elapsed,
                    elapsed=format_seconds(elapsed),
                )

            start = end
            if (
                args.adaptive_batch_size
                and current_batch_size < max_batch_size
                and successful_batches_since_resize >= 2
            ):
                new_batch_size = min(max_batch_size, current_batch_size * 2)
                if new_batch_size > current_batch_size:
                    print(
                        f"[selection] increasing batch_size "
                        f"{current_batch_size}->{new_batch_size}",
                        flush=True,
                    )
                    write_selection_progress(
                        output_root,
                        event="batch_size_growth",
                        greedy_iter=greedy_iter,
                        greedy_total=greedy_total,
                        old_batch_size=current_batch_size,
                        new_batch_size=new_batch_size,
                    )
                    current_batch_size = new_batch_size
                    successful_batches_since_resize = 0

        selected_summaries = []
        selected_records = []
        for pool_run, method_id, runtime in active:
            buf = buffers[(pool_run.pool_size, method_id)]
            n_written = int(buf["write_pos"])
            if n_written == 0:
                raise RuntimeError(
                    f"No candidates left for pool_size={pool_run.pool_size}, method={method_id}"
                )
            candidate_indices = buf["candidate_indices"][:n_written]
            scores_per_track = {
                name: scores[:n_written]
                for name, scores in buf["scores_per_track"].items()
            }
            combined = aggregate_track_scores(scores_per_track, runtime.spec)
            best_pos = int(torch.argmax(combined).item())
            best_idx = int(candidate_indices[best_pos])
            best_score = float(combined[best_pos].item())
            best_track_scores = {
                name: float(scores[best_pos].item())
                for name, scores in scores_per_track.items()
            }

            row = {
                "iteration": len(runtime.current_indices) - args.init_size + 1,
                "n_selected": len(runtime.current_indices) + 1,
                "selected_index": best_idx,
                "score_combined": best_score,
                "method_id": runtime.spec.method_id,
                "within": runtime.spec.within,
                "across": runtime.spec.across,
                "pool_size": pool_run.pool_size,
            }
            for track_name, score in best_track_scores.items():
                row[f"score_{track_name}"] = score
                runtime.scores_per_track_history[track_name].append(score)
            runtime.trace_rows.append(row)
            runtime.scores_combined.append(best_score)
            selected_summaries.append(
                f"{pool_size_dir_name(pool_run.pool_size)}/{method_id}:"
                f"idx={best_idx},score={best_score:.4f}"
            )
            selected_records.append(
                {
                    "pool_size": pool_run.pool_size,
                    "method_id": runtime.spec.method_id,
                    "selected_index": best_idx,
                    "score": best_score,
                }
            )

            append_new_stimulus(
                runtime=runtime,
                new_idx=best_idx,
                raw_features_np=raw_features_np,
                model_names=model_names,
                encoding_params=encoding_params,
                metric=args.metric,
                device=device,
            )
            save_runtime_progress(
                runtime,
                pool_run.payload_root,
                raw_features_np,
                raw_shard_slices,
                model_names,
                pool_run.run_config,
                pool_records_by_index=pool_records_by_index,
            )

        elapsed = time.monotonic() - iter_start_time
        print(
            f"[selection] greedy {greedy_iter}/{greedy_total}: selected "
            + "; ".join(selected_summaries)
            + f" | iteration_elapsed={format_seconds(elapsed)}",
            flush=True,
        )
        write_selection_progress(
            output_root,
            event="greedy_iteration_selected",
            greedy_iter=greedy_iter,
            greedy_total=greedy_total,
            selected=selected_records,
            elapsed_seconds=elapsed,
            elapsed=format_seconds(elapsed),
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del raw_features_np, encoding_params
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    final_pool = pool_runs[max(pool_runs)]
    return final_pool.payload_root, final_pool.eval_root, final_pool.comparison_root


def run_evaluation(args: argparse.Namespace, methods: list[MethodSpec], payload_root: Path, eval_root: Path) -> None:
    if args.skip_eval:
        print("Skipping recovery evaluation")
        return

    random_feature_dir = args.random_feature_dir or args.pool_feature_dir
    for method in methods:
        cmd = [
            sys.executable,
            str(RECOVERY_SCRIPT),
            "--model-sets",
            method.method_id,
            "--selection-root",
            str(payload_root),
            "--output-root",
            str(eval_root),
            "--env",
            args.env,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--n-random-subsets",
            str(args.n_random_subsets),
            "--n-noise-samples",
            str(args.n_noise_samples),
            "--n-bootstrap",
            str(args.n_bootstrap),
            "--which-selection",
            "final",
            "--tracks",
            "raw," + ",".join(ENCODING_TRACKS),
        ]
        if random_feature_dir is not None:
            cmd.extend(
                [
                    "--random-feature-dir",
                    str(random_feature_dir.resolve()),
                    "--strict-random-models",
                ]
            )
            if args.n_random_images is not None:
                cmd.extend(["--n-random-images", str(args.n_random_images)])
        if args.shared_encodings:
            cmd.append("--shared-encodings")
        print("\nRunning recovery evaluation:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT))


def weighted_value(values: dict[str, float], weights: dict[str, float]) -> float:
    available = {k: v for k, v in values.items() if k in weights and pd.notna(v)}
    if not available:
        return float("nan")
    total = sum(weights[k] for k in available)
    if total == 0:
        return float("nan")
    return sum(values[k] * weights[k] for k in available) / total


def compare_results(methods: list[MethodSpec], eval_root: Path, comparison_root: Path) -> None:
    recovery_rows = []
    pairwise_rows = []
    spec_by_id = {method.method_id: method for method in methods}

    for method in methods:
        out_dir = eval_root / f"{method.method_id}_noisy_by_clean_boot"
        auc_path = out_dir / "auc_significance.csv"
        pairwise_path = out_dir / "pairwise_auc.csv"
        if auc_path.exists():
            df = pd.read_csv(auc_path)
            df.insert(0, "method_id", method.method_id)
            df.insert(1, "method_label", method.label)
            recovery_rows.append(df)
        if pairwise_path.exists():
            df = pd.read_csv(pairwise_path)
            df.insert(0, "method_id", method.method_id)
            df.insert(1, "method_label", method.label)
            pairwise_rows.append(df)

    comparison_root.mkdir(parents=True, exist_ok=True)
    recovery = pd.concat(recovery_rows, ignore_index=True) if recovery_rows else pd.DataFrame()
    pairwise = pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()
    recovery.to_csv(comparison_root / "recovery_auc_by_method.csv", index=False)
    pairwise.to_csv(comparison_root / "pairwise_auc_by_method.csv", index=False)

    summary_rows = []
    for method_id, grp in recovery.groupby("method_id") if not recovery.empty else []:
        spec = spec_by_id[method_id]
        selected_auc = dict(zip(grp["track"], grp["selected_auc"]))
        random_auc = dict(zip(grp["track"], grp["random_auc_mean"]))
        encoding_tracks = [track for track in ENCODING_TRACKS if track in selected_auc]
        row = {
            "method_id": method_id,
            "method_label": spec.label,
            "within": spec.within,
            "across": spec.across,
            "track_agg_method": spec.track_agg_method,
            "selected_auc_effective_weighted": weighted_value(selected_auc, spec.summary_weights),
            "random_auc_effective_weighted": weighted_value(random_auc, spec.summary_weights),
            "selected_auc_mean_tracks": float(np.mean(list(selected_auc.values()))),
            "selected_auc_worst_track": float(np.max(list(selected_auc.values()))),
            "selected_auc_raw": selected_auc.get("raw", np.nan),
            "selected_auc_sub01": selected_auc.get("sub-01", np.nan),
            "selected_auc_mean_encoding_tracks": (
                float(np.mean([selected_auc[track] for track in encoding_tracks]))
                if encoding_tracks
                else np.nan
            ),
        }
        if pd.notna(row["selected_auc_effective_weighted"]) and pd.notna(
            row["random_auc_effective_weighted"]
        ):
            row["auc_delta_effective_weighted_random_minus_selected"] = (
                row["random_auc_effective_weighted"] - row["selected_auc_effective_weighted"]
            )
        else:
            row["auc_delta_effective_weighted_random_minus_selected"] = np.nan

        if not pairwise.empty:
            pgrp = pairwise[pairwise["method_id"] == method_id]
            dom = dict(zip(pgrp["track"], pgrp["selected_pairwise_dominance_auc"]))
            margin = dict(zip(pgrp["track"], pgrp["selected_mean_margin_auc"]))
            row["pairwise_dominance_auc_effective_weighted"] = weighted_value(
                dom, spec.summary_weights
            )
            row["mean_margin_auc_effective_weighted"] = weighted_value(
                margin, spec.summary_weights
            )
            row["pairwise_dominance_auc_mean_tracks"] = (
                float(np.mean(list(dom.values()))) if dom else np.nan
            )
            row["mean_margin_auc_mean_tracks"] = (
                float(np.mean(list(margin.values()))) if margin else np.nan
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("selected_auc_effective_weighted", ascending=True)
    summary.to_csv(comparison_root / "method_summary.csv", index=False)

    with (comparison_root / "method_summary.md").open("w") as f:
        f.write("# Feature Method Sweep Summary\n\n")
        f.write("Strict top-1 recovery AUC is lower-is-better. Pairwise dominance/margin AUCs are higher-is-better.\n\n")
        if summary.empty:
            f.write("No completed evaluation outputs found.\n")
        else:
            f.write("```text\n")
            f.write(summary.to_string(index=False))
            f.write("\n```\n")
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="raven", choices=["raven", "iris"])
    parser.add_argument("--model-set", default="sota")
    parser.add_argument("--methods", default=None, help="Comma-separated subset of method ids.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--pool-feature-dir",
        type=Path,
        default=None,
        help="Optional candidate-pool directory containing one <model>.npz file per model.",
    )
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--init-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--corr-type", default="correlation")
    parser.add_argument("--noise-ceiling-target", type=float, default=0.46)
    parser.add_argument("--max-ram-gb", type=float, default=50.0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--pool-sizes",
        default=None,
        help=(
            "Comma-separated nested candidate-pool sizes to evaluate in one shared "
            "selection pass, e.g. '1k,10k,50k,100k,250k,500k,1M,5M,10M'. "
            "When set, --max-images defaults to the largest requested pool."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2500)
    parser.add_argument("--adaptive-batch-size", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--min-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoding-roi-subset", default="hlvis")
    parser.add_argument("--noise-calib-examples", type=int, default=1000)
    parser.add_argument("--noise-calib-repeats", type=int, default=100)
    parser.add_argument("--n-random-subsets", type=int, default=50)
    parser.add_argument(
        "--random-feature-dir",
        type=Path,
        default=None,
        help="Optional .npz feature cache for recovery random baselines. Defaults to --pool-feature-dir when provided.",
    )
    parser.add_argument(
        "--n-random-images",
        type=int,
        default=None,
        help="Number of random baseline images to load for recovery when using a random feature dir.",
    )
    parser.add_argument("--n-noise-samples", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=10,
        help="Print one progress line every N candidate batches during each greedy step. Set 0 to disable.",
    )
    parser.add_argument("--shared-encodings", action="store_true")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = (
            SCRIPT.parents[1] / "results" / f"{args.model_set}_feature_sweep_{stamp}"
        )
    return args


def main() -> None:
    args = parse_args()
    methods = parse_method_filter(args.methods, default_methods())
    if args.skip_selection:
        payload_root = args.output_root.resolve() / "payloads"
        eval_root = RECOVERY_RESULTS_ROOT / args.output_root.resolve().name / "eval"
        comparison_root = RECOVERY_RESULTS_ROOT / args.output_root.resolve().name / "comparison"
    else:
        payload_root, eval_root, comparison_root = run_selection(args, methods)
    if args.pool_sizes:
        if not args.skip_eval:
            print(
                "\nMulti-pool selection writes one standard result directory per pool size. "
                "Skipping the legacy noisy-by-clean evaluator for this mode; run the "
                "teacher-student recovery queue over the pool_* directories instead."
            )
        else:
            print("\nDone. Multi-pool selection payloads are under:")
            print(args.output_root.resolve())
        print(
            "Pool manifest:",
            (args.output_root.resolve() / "pool_size_manifest.csv").resolve(),
        )
        return
    run_evaluation(args, methods, payload_root, eval_root)
    compare_results(methods, eval_root, comparison_root)
    print(f"\nDone. Selection payloads: {args.output_root.resolve()}")
    print(f"Recovery summary: {(comparison_root / 'method_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
