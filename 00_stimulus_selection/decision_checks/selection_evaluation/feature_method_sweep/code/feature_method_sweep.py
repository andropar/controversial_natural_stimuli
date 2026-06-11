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
ROOT = SCRIPT.parents[5]
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
    compute_correlation_matrix,
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
    / "decision_checks"
    / "selection_evaluation"
    / "noisy_by_clean_recovery"
    / "code"
    / "compute_noisy_by_clean_recovery.py"
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
    summary_weights: dict[str, float] = field(default_factory=dict)
    description: str = ""


@dataclass
class TrackRuntime:
    spec: TrackSpec
    selected_features: dict[str, torch.Tensor]
    rdm_by_model: dict[str, torch.Tensor]
    noise_vars: torch.Tensor
    var_noise_by_model: dict[str, float]


@dataclass
class MethodRuntime:
    spec: MethodSpec
    current_indices: list[int]
    pool_mask: np.ndarray
    tracks: dict[str, TrackRuntime]
    scores_combined: list[float] = field(default_factory=list)
    scores_per_track_history: dict[str, list[float]] = field(default_factory=dict)
    trace_rows: list[dict[str, Any]] = field(default_factory=list)


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

    current_rdms = torch.stack([runtime.rdm_by_model[model] for model in model_names], dim=0)
    current_rdms = current_rdms.unsqueeze(0).expand(batch_size, -1, -1)
    augmented_rdms = torch.cat([current_rdms, new_dissims_tensor], dim=2)

    rdm_vars = augmented_rdms.var(dim=2, unbiased=False)
    attenuation = torch.sqrt(rdm_vars / (rdm_vars + runtime.noise_vars.unsqueeze(0) + 1e-8))
    correlations = compute_correlation_matrix(augmented_rdms, augmented_rdms, corr_type="correlation")
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
        noise = var_noise_by_track[track.name]
        tracks[track.name] = TrackRuntime(
            spec=track,
            selected_features=selected,
            rdm_by_model=rdm_by_model,
            noise_vars=torch.tensor(
                [noise[model] for model in model_names],
                device=device,
                dtype=torch.float32,
            ),
            var_noise_by_model=dict(noise),
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

    runtime.current_indices.append(int(new_idx))
    runtime.pool_mask[int(new_idx)] = False


def save_runtime_progress(
    runtime: MethodRuntime,
    payload_root: Path,
    raw_features_np: dict[str, np.ndarray],
    raw_shard_slices: Any,
    model_names: list[str],
    run_config: dict[str, Any],
) -> None:
    method_dir = payload_root / runtime.spec.method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(runtime.current_indices, dtype=np.int64)
    np.save(method_dir / "selected_indices.npy", indices)
    pd.DataFrame(runtime.trace_rows).to_csv(method_dir / "selection_trace.csv", index=False)

    image_records = build_selected_image_records(indices.tolist(), raw_shard_slices)
    pd.DataFrame(image_records).to_csv(method_dir / "selected_image_records.csv", index=False)

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
    payload_root = output_root / "payloads"
    eval_root = output_root / "eval"
    comparison_root = output_root / "comparison"
    payload_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)

    model_list_csv = Path(paths["model_list_csv"])
    layer_names = load_layer_names(model_list_csv, model_names)

    if args.max_images is not None:
        max_images = int(args.max_images)
    else:
        max_images = max_images_for_ram(
            subset_root=Path(paths["subset_root"]),
            model_names=model_names,
            max_ram_bytes=int(args.max_ram_gb * 1024**3),
            model_csv=model_list_csv,
        )
    if max_images <= args.init_size:
        raise ValueError(f"Candidate pool too small: max_images={max_images}")

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
    initial_indices = rng.choice(max_images, size=args.init_size, replace=False).astype(np.int64)
    run_config = {
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
        "max_images": max_images,
        "candidate_pool": "np.arange(max_images)",
        "candidate_pool_size": max_images,
        "encoding_roi_subset": args.encoding_roi_subset,
    }
    with (output_root / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2, default=str)
    np.save(output_root / "pool_indices.npy", np.arange(max_images, dtype=np.int64))
    save_manifest(methods, payload_root)
    write_selection_progress(
        output_root,
        event="selection_started",
        model_set_name=model_set_name,
        max_images=max_images,
        target_size=args.target_size,
        init_size=args.init_size,
        methods=[method.method_id for method in methods],
    )

    runtimes: dict[str, MethodRuntime] = {}
    for method in methods:
        method_dir = payload_root / method.method_id
        existing = load_existing_indices(method_dir) if args.resume else None
        selected = existing if existing is not None else initial_indices.tolist()
        runtime = build_runtime(
            method=method,
            selected_indices=selected,
            raw_features_np=raw_features_np,
            model_names=model_names,
            encoding_params=encoding_params,
            var_noise_by_track=var_noise_by_track,
            metric=args.metric,
            device=device,
            pool_size=max_images,
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
        runtimes[method.method_id] = runtime
        save_runtime_progress(runtime, payload_root, raw_features_np, raw_shard_slices, model_names, run_config)

    n_batches_total = (max_images + args.batch_size - 1) // args.batch_size
    for greedy_step in trange(args.init_size, args.target_size, desc="Feature-only greedy"):
        active = [
            runtime
            for runtime in runtimes.values()
            if len(runtime.current_indices) < args.target_size
        ]
        if not active:
            break

        greedy_iter = greedy_step - args.init_size + 1
        greedy_total = args.target_size - args.init_size
        iter_start_time = time.monotonic()
        pool_sizes = {
            runtime.spec.method_id: int(runtime.pool_mask.sum()) for runtime in active
        }
        print(
            f"[selection] greedy {greedy_iter}/{greedy_total}: "
            f"active_methods={len(active)}, batches={n_batches_total}, "
            f"pool_remaining={pool_sizes}",
            flush=True,
        )
        write_selection_progress(
            output_root,
            event="greedy_iteration_start",
            greedy_iter=greedy_iter,
            greedy_total=greedy_total,
            active_methods=[runtime.spec.method_id for runtime in active],
            n_batches=n_batches_total,
            pool_remaining=pool_sizes,
        )

        buffers: dict[str, dict[str, Any]] = {}
        for runtime in active:
            n_candidates = int(runtime.pool_mask.sum())
            buffers[runtime.spec.method_id] = {
                "candidate_indices": np.empty(n_candidates, dtype=np.int64),
                "scores_per_track": {
                    track.name: torch.empty(n_candidates, dtype=torch.float32)
                    for track in runtime.spec.tracks
                },
                "write_pos": 0,
            }

        for batch_idx, start in enumerate(range(0, max_images, args.batch_size), start=1):
            end = min(start + args.batch_size, max_images)
            batch_indices = np.arange(start, end, dtype=np.int64)
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

            for runtime in active:
                valid_mask = runtime.pool_mask[batch_indices]
                if not valid_mask.any():
                    continue
                valid_positions_np = np.flatnonzero(valid_mask).astype(np.int64)
                valid_positions = torch.from_numpy(valid_positions_np).to(device=device)
                buf = buffers[runtime.spec.method_id]
                write_pos = int(buf["write_pos"])
                n_valid = len(valid_positions_np)
                buf["candidate_indices"][write_pos : write_pos + n_valid] = batch_indices[
                    valid_positions_np
                ]

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

            if (
                args.progress_every_batches > 0
                and (
                    batch_idx == 1
                    or batch_idx == n_batches_total
                    or batch_idx % args.progress_every_batches == 0
                )
            ):
                elapsed = time.monotonic() - iter_start_time
                print(
                    f"[selection] greedy {greedy_iter}/{greedy_total}: "
                    f"batch {batch_idx}/{n_batches_total} "
                    f"({end}/{max_images} images scanned), elapsed={format_seconds(elapsed)}",
                    flush=True,
                )
                write_selection_progress(
                    output_root,
                    event="candidate_batch",
                    greedy_iter=greedy_iter,
                    greedy_total=greedy_total,
                    batch_idx=batch_idx,
                    n_batches=n_batches_total,
                    images_scanned=end,
                    max_images=max_images,
                    elapsed_seconds=elapsed,
                    elapsed=format_seconds(elapsed),
                )

        selected_summaries = []
        selected_records = []
        for runtime in active:
            buf = buffers[runtime.spec.method_id]
            n_written = int(buf["write_pos"])
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
            }
            for track_name, score in best_track_scores.items():
                row[f"score_{track_name}"] = score
                runtime.scores_per_track_history[track_name].append(score)
            runtime.trace_rows.append(row)
            runtime.scores_combined.append(best_score)
            selected_summaries.append(
                f"{runtime.spec.method_id}:idx={best_idx},score={best_score:.4f}"
            )
            selected_records.append(
                {
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
                payload_root,
                raw_features_np,
                raw_shard_slices,
                model_names,
                run_config,
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

    return payload_root, eval_root, comparison_root


def run_evaluation(args: argparse.Namespace, methods: list[MethodSpec], payload_root: Path, eval_root: Path) -> None:
    if args.skip_eval:
        print("Skipping recovery evaluation")
        return

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
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--init-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--corr-type", default="correlation")
    parser.add_argument("--noise-ceiling-target", type=float, default=0.46)
    parser.add_argument("--max-ram-gb", type=float, default=50.0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoding-roi-subset", default="hlvis")
    parser.add_argument("--noise-calib-examples", type=int, default=1000)
    parser.add_argument("--noise-calib-repeats", type=int, default=100)
    parser.add_argument("--n-random-subsets", type=int, default=50)
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
        eval_root = args.output_root.resolve() / "eval"
        comparison_root = args.output_root.resolve() / "comparison"
    else:
        payload_root, eval_root, comparison_root = run_selection(args, methods)
    run_evaluation(args, methods, payload_root, eval_root)
    compare_results(methods, eval_root, comparison_root)
    print(f"\nDone. Results: {args.output_root.resolve()}")
    print(f"Summary: {(comparison_root / 'method_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
