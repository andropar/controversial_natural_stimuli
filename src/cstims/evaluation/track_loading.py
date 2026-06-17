"""Track discovery and feature loading for selection evaluation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from cstims.data_loader import load_natural_features_with_metadata
from cstims.encoding.linear import (
    encode_batch_for_all_encodings,
    EncodingParamsByModel,
    load_encoding_params_by_encoding,
)
from cstims.evaluation.data_loader import load_selected_features
from cstims.paths import model_list_csv as default_model_list_csv


SelectionVariant = Literal["final", "greedy", "best_raw_combined"]
VALID_SELECTION_VARIANTS: tuple[SelectionVariant, ...] = (
    "final",
    "greedy",
    "best_raw_combined",
)
RandomFeatureLoader = Callable[
    [dict, list[str], int, str],
    dict[str, np.ndarray],
]
LogMemoryFn = Callable[[str], None]


def get_all_tracks_for_evaluation(payload: dict) -> list[dict]:
    """Extract identity and encoding tracks that should be evaluated."""
    tracks: list[dict] = []
    seen_names: set[str] = set()

    for track in payload.get("track_definitions", []):
        if not isinstance(track, dict):
            continue
        track_type = track.get("type", "identity")
        track_name = track.get("name")
        if track_type in ("identity", "encoding") and track_name:
            tracks.append(
                {
                    "name": track_name,
                    "type": track_type,
                    "encoding_name": track.get("encoding_name"),
                }
            )
            seen_names.add(track_name)

    features_by_encoding = payload.get("selected_features_by_encoding") or {}
    for enc_name in features_by_encoding:
        if enc_name not in seen_names:
            tracks.append(
                {
                    "name": enc_name,
                    "type": "encoding",
                    "encoding_name": enc_name,
                }
            )
            seen_names.add(enc_name)

    return tracks


def get_variant_feature_keys(variant: SelectionVariant) -> tuple[str, str, str]:
    """Return payload keys for one selected-stimulus variant."""
    if variant == "final":
        return (
            "selected_features_raw",
            "selected_features_by_encoding",
            "selected_global_indices",
        )
    if variant == "greedy":
        return "greedy_features_raw", "greedy_features_by_encoding", "greedy_indices"
    if variant == "best_raw_combined":
        return (
            "best_raw_combined_features_raw",
            "best_raw_combined_features_by_encoding",
            "best_raw_combined_indices",
        )
    raise ValueError(
        f"Unknown selection variant: {variant}. Valid: {VALID_SELECTION_VARIANTS}"
    )


def _compute_weighted_mean(
    scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    if not scores:
        return float("-inf")
    if weights is None:
        return sum(scores.values()) / len(scores)

    total_weight = 0.0
    weighted_sum = 0.0
    for name, score in scores.items():
        weight = weights.get(name, 1.0)
        weighted_sum += weight * score
        total_weight += weight
    if total_weight == 0:
        return float("-inf")
    return weighted_sum / total_weight


def _reconstruct_greedy_indices_from_history(payload: dict) -> np.ndarray | None:
    refinement_history = payload.get("refinement_history")
    selected_indices = payload.get("selected_global_indices")
    if not refinement_history or selected_indices is None:
        return None

    selected_indices = np.asarray(selected_indices)
    n_selected = len(selected_indices)
    records_by_pass: dict[int, list[dict]] = {}
    for record in refinement_history:
        pass_num = record.get("pass", record.get("pass_num", 0))
        records_by_pass.setdefault(pass_num, []).append(record)

    if 0 not in records_by_pass:
        return None

    pass0_records = records_by_pass[0]
    if len(pass0_records) != n_selected:
        print(
            f"  [WARN] Pass 0 has {len(pass0_records)} records "
            f"but expected {n_selected}"
        )

    greedy_indices = np.zeros(n_selected, dtype=np.int64)
    for record in pass0_records:
        position = record.get("position", -1)
        old_idx = record.get("old_idx", -1)
        if 0 <= position < n_selected and old_idx >= 0:
            greedy_indices[position] = old_idx

    return greedy_indices


def reconstruct_best_raw_combined_from_history(
    payload: dict,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, float, int]:
    """Replay refinement history to recover best raw-combined indices."""
    greedy_indices = payload.get("greedy_indices")
    if greedy_indices is None:
        greedy_indices = _reconstruct_greedy_indices_from_history(payload)
        if greedy_indices is not None:
            print("  [INFO] Reconstructed greedy_indices from refinement history pass 0")
        else:
            raise ValueError(
                "Cannot reconstruct best_raw_combined: greedy_indices not found "
                "and could not be reconstructed from refinement_history"
            )
    greedy_indices = np.asarray(greedy_indices)

    refinement_history = payload.get("refinement_history")
    scores_per_view = payload.get("scores_per_view_history") or {}
    scores_per_rep = payload.get("scores_per_rep_history") or {}
    all_track_scores = {**scores_per_view, **scores_per_rep}
    if all_track_scores:
        final_greedy_scores = {
            name: scores[-1] for name, scores in all_track_scores.items() if scores
        }
        greedy_combined = _compute_weighted_mean(final_greedy_scores, weights)
    else:
        greedy_combined = float("-inf")

    if not refinement_history:
        return greedy_indices.copy(), greedy_combined, -1

    best_indices = greedy_indices.copy()
    best_score = greedy_combined
    best_pass = -1
    current_indices = greedy_indices.copy()

    records_by_pass: dict[int, list[dict]] = {}
    for record in refinement_history:
        pass_num = record.get("pass", record.get("pass_num", 0))
        records_by_pass.setdefault(pass_num, []).append(record)

    for pass_num in sorted(records_by_pass):
        pass_records = sorted(
            records_by_pass[pass_num],
            key=lambda record: record.get("position", 0),
        )
        for record in pass_records:
            if record.get("replaced", False):
                position = record.get("position", 0)
                new_idx = record.get("new_idx", -1)
                if 0 <= position < len(current_indices) and new_idx >= 0:
                    current_indices[position] = new_idx

            scores_per_track = record.get("scores_per_track")
            if not scores_per_track:
                continue
            record_score = _compute_weighted_mean(scores_per_track, weights)
            if record_score > best_score:
                best_score = record_score
                best_indices = current_indices.copy()
                best_pass = pass_num

    return best_indices, best_score, best_pass


def get_selected_indices(
    payload: dict,
    variant: SelectionVariant = "final",
) -> np.ndarray:
    """Return selected global indices for one selected-stimulus variant."""
    if variant == "best_raw_combined":
        print("  [INFO] Reconstructing best_raw_combined from refinement history...")
        track_agg = payload.get("track_aggregation", {})
        weights = track_agg.get("weights")
        indices, score, pass_num = reconstruct_best_raw_combined_from_history(
            payload,
            weights,
        )
        print(
            f"  [INFO] Reconstructed best_raw_combined: score={score:.4f}, "
            f"pass={pass_num + 1 if pass_num >= 0 else 'greedy'}"
        )
        return indices

    _, _, indices_key = get_variant_feature_keys(variant)
    indices = payload.get(indices_key)
    if indices is None:
        raise ValueError(
            f"Indices not found for variant '{variant}' (key: {indices_key})"
        )
    return np.asarray(indices)


def _to_feature_tensors(
    features: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: (
            torch.tensor(value, device=device, dtype=torch.float32)
            if not isinstance(value, torch.Tensor)
            else value.to(device=device, dtype=torch.float32)
        )
        for key, value in features.items()
    }


def _log_memory(log_memory_fn: LogMemoryFn | None, label: str) -> None:
    if log_memory_fn is not None:
        log_memory_fn(label)


def load_features_for_track(
    payload: dict,
    track: dict,
    device: torch.device,
    encoding_params_cache: dict[str, Any] | None = None,
    n_random: int = 10000,
    selection_variant: SelectionVariant = "final",
    encoding_root_map: dict[str, Path] | None = None,
    random_feature_loader: RandomFeatureLoader | None = None,
    log_memory_fn: LogMemoryFn | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Load selected and random features for one evaluation track."""
    track_type = track.get("type", "identity")
    track_name = track["name"]
    encoding_name = track.get("encoding_name")
    model_names = list(payload["model_names"])

    if track_type == "identity":
        selected = _load_selected_identity(
            payload,
            track_name,
            device,
            selection_variant,
        )
        random = _load_random_identity(
            payload,
            model_names,
            n_random,
            track_name,
            random_feature_loader=random_feature_loader,
            log_memory_fn=log_memory_fn,
        )
    elif track_type == "encoding":
        selected, random = _load_encoded_features(
            payload=payload,
            encoding_name=encoding_name or track_name,
            model_names=model_names,
            device=device,
            encoding_params_cache=encoding_params_cache,
            n_random=n_random,
            selection_variant=selection_variant,
            encoding_root_map=encoding_root_map,
            random_feature_loader=random_feature_loader,
            log_memory_fn=log_memory_fn,
        )
    else:
        raise ValueError(f"Unsupported track type: {track_type}")

    return selected, random


def load_selected_raw_features(
    payload: dict,
    model_names: list[str] | None = None,
    selection_variant: SelectionVariant = "final",
) -> dict[str, np.ndarray]:
    """Load selected raw features from a payload as CPU NumPy arrays."""
    names = list(model_names or payload["model_names"])
    selected = _load_selected_identity(
        payload,
        "raw",
        torch.device("cpu"),
        selection_variant,
    )
    out = {}
    for model_name in names:
        arr = selected[model_name]
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        out[model_name] = np.asarray(arr, dtype=np.float32)
    return out


def encode_raw_feature_arrays(
    raw_features: dict[str, np.ndarray],
    encoding_name: str,
    encoding_params: EncodingParamsByModel,
    *,
    device: torch.device,
    batch_size: int = 1000,
) -> dict[str, np.ndarray]:
    """Apply one encoding model to raw NumPy feature arrays."""
    model_names = list(raw_features)
    n_samples = next(iter(raw_features.values())).shape[0]
    encoded_batches: dict[str, list[np.ndarray]] = {model: [] for model in model_names}
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_torch = {
                model: torch.as_tensor(
                    values[start_idx:end_idx],
                    device=device,
                    dtype=torch.float32,
                )
                for model, values in raw_features.items()
            }
            encoded_batch = encode_batch_for_all_encodings(
                batch_torch,
                {encoding_name: encoding_params},
            )[encoding_name]
            for model_name in model_names:
                encoded_batches[model_name].append(
                    encoded_batch[model_name].detach().cpu().numpy()
                )
            del batch_torch, encoded_batch
            if use_cuda:
                torch.cuda.empty_cache()
    return {
        model: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for model, chunks in encoded_batches.items()
    }


def _load_selected_identity(
    payload: dict,
    view_name: str,
    device: torch.device,
    selection_variant: SelectionVariant = "final",
) -> dict[str, torch.Tensor]:
    if selection_variant == "best_raw_combined":
        return _extract_features_for_variant(payload, selection_variant, device)

    raw_key, _, _ = get_variant_feature_keys(selection_variant)
    if selection_variant == "greedy":
        variant_features = payload.get(raw_key)
        if variant_features:
            return _to_feature_tensors(variant_features, device)

    features_by_view = payload.get("selected_features_by_view") or {}
    if view_name in features_by_view:
        return _to_feature_tensors(features_by_view[view_name], device)

    if view_name == "raw" and payload.get("selected_features_raw"):
        return _to_feature_tensors(payload["selected_features_raw"], device)

    if payload.get("selected_features"):
        return _to_feature_tensors(payload["selected_features"], device)

    return load_selected_features(payload, device, view_name=view_name)


def _extract_features_for_variant(
    payload: dict,
    selection_variant: SelectionVariant,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    print(f"  [INFO] Extracting features on-the-fly for variant '{selection_variant}'...")
    indices = get_selected_indices(payload, selection_variant)

    config = payload.get("config", {})
    paths = config.get("paths", {})
    model_names = payload["model_names"]
    preprocessed_dirs = paths.get("preprocessed_dirs", {})
    preproc_dir = preprocessed_dirs.get("raw")
    model_csv = Path(paths.get("model_list_csv", default_model_list_csv()))
    if not model_csv.exists():
        model_csv = default_model_list_csv()

    load_kwargs = {
        "subset_root": Path(paths["subset_root"]),
        "model_names": model_names,
        "model_csv": model_csv,
        "max_images": None,
    }
    if preproc_dir and Path(preproc_dir).exists():
        load_kwargs["preprocessed_dir"] = Path(preproc_dir)

    features, _ = load_natural_features_with_metadata(**load_kwargs)
    return {
        name: torch.tensor(features[name][indices], device=device, dtype=torch.float32)
        for name in model_names
    }


def _load_random_identity(
    payload: dict,
    model_names: list[str],
    n_random: int,
    view_name: str = "raw",
    *,
    random_feature_loader: RandomFeatureLoader | None = None,
    log_memory_fn: LogMemoryFn | None = None,
) -> dict[str, np.ndarray]:
    if random_feature_loader is not None:
        return random_feature_loader(payload, model_names, n_random, view_name)

    config = payload.get("config", {})
    paths = config.get("paths", {})
    preprocessed_dirs = paths.get("preprocessed_dirs", {})
    preproc_dir = preprocessed_dirs.get(view_name) or preprocessed_dirs.get("raw")
    if not preproc_dir:
        raise ValueError(f"No preprocessed directory found for view '{view_name}'")

    print(
        f"  [DEBUG] Loading random features via shard-based loading "
        f"(max_images={n_random})"
    )
    print(f"  [DEBUG] Requested n_random={n_random}, models={len(model_names)}")
    _log_memory(log_memory_fn, "before_load_random")

    model_csv = Path(paths.get("model_list_csv", default_model_list_csv()))
    if not model_csv.exists():
        model_csv = default_model_list_csv()

    features, _ = load_natural_features_with_metadata(
        subset_root=Path(paths["subset_root"]),
        model_names=model_names,
        model_csv=model_csv,
        max_images=n_random,
        preprocessed_dir=None,
    )
    if any(features[model].size == 0 for model in model_names):
        print(
            "  [DEBUG] Shard-based loading returned empty arrays; "
            "falling back to memmap fast path + subsampling."
        )
        features_full, _ = load_natural_features_with_metadata(
            subset_root=Path(paths["subset_root"]),
            model_names=model_names,
            model_csv=model_csv,
            preprocessed_dir=Path(preproc_dir),
        )
        n_total = next(iter(features_full.values())).shape[0]
        n_take = min(n_random, n_total)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, size=n_take, replace=False)
        idx.sort()
        features = {
            model: np.asarray(features_full[model][idx])
            for model in model_names
        }

    _log_memory(log_memory_fn, "after_load_random")
    total_bytes = 0
    for model_name, arr in features.items():
        total_bytes += arr.nbytes
        print(
            f"  [DEBUG] {model_name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"size={arr.nbytes / 1024 / 1024:.1f}MB"
        )
    print(f"  [DEBUG] Total feature size: {total_bytes / 1024 / 1024:.1f}MB in RAM")

    return features


def _load_encoded_features(
    payload: dict,
    encoding_name: str,
    model_names: list[str],
    device: torch.device,
    encoding_params_cache: dict[str, Any] | None,
    n_random: int,
    batch_size: int = 1000,
    selection_variant: SelectionVariant = "final",
    encoding_root_map: dict[str, Path] | None = None,
    random_feature_loader: RandomFeatureLoader | None = None,
    log_memory_fn: LogMemoryFn | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    if selection_variant == "best_raw_combined":
        raw_selected = _load_selected_identity(
            payload,
            "raw",
            device,
            selection_variant,
        )
        encoding_params = _ensure_encoding_params(
            payload,
            [encoding_name],
            device,
            encoding_params_cache,
            encoding_root_map=encoding_root_map,
        )
        encoded = encode_batch_for_all_encodings(
            raw_selected,
            {encoding_name: encoding_params[encoding_name]},
        )
        selected = encoded[encoding_name]
    else:
        use_precomputed = encoding_root_map is None
        _, enc_key, _ = get_variant_feature_keys(selection_variant)
        features_by_encoding = payload.get(enc_key) or {} if use_precomputed else {}
        if not features_by_encoding and use_precomputed:
            features_by_encoding = payload.get("selected_features_by_encoding") or {}

        if encoding_name in features_by_encoding:
            selected = _to_feature_tensors(features_by_encoding[encoding_name], device)
        else:
            raw_selected = _load_selected_identity(
                payload,
                "raw",
                device,
                selection_variant,
            )
            encoding_params = _ensure_encoding_params(
                payload,
                [encoding_name],
                device,
                encoding_params_cache,
                encoding_root_map=encoding_root_map,
            )
            encoded = encode_batch_for_all_encodings(
                raw_selected,
                {encoding_name: encoding_params[encoding_name]},
            )
            selected = encoded[encoding_name]

    print(f"  [DEBUG] Loading raw random features for encoding '{encoding_name}'...")
    raw_random = _load_random_identity(
        payload,
        model_names,
        n_random,
        "raw",
        random_feature_loader=random_feature_loader,
        log_memory_fn=log_memory_fn,
    )

    encoding_params = _ensure_encoding_params(
        payload,
        [encoding_name],
        device,
        encoding_params_cache,
        encoding_root_map=encoding_root_map,
    )

    n_samples = next(iter(raw_random.values())).shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    print(
        f"  [DEBUG] Encoding {n_samples} samples in {n_batches} "
        f"batches of {batch_size}"
    )
    _log_memory(log_memory_fn, "before_encoding_loop")

    encoded_batches: dict[str, list[np.ndarray]] = {model: [] for model in model_names}
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_torch = {
            model: torch.tensor(
                values[start_idx:end_idx],
                device=device,
                dtype=torch.float32,
            )
            for model, values in raw_random.items()
        }
        encoded_batch = encode_batch_for_all_encodings(
            batch_torch,
            {encoding_name: encoding_params[encoding_name]},
        )
        for model_name in model_names:
            encoded_batches[model_name].append(
                encoded_batch[encoding_name][model_name].cpu().numpy()
            )
        del batch_torch, encoded_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    random = {
        model: np.concatenate(chunks, axis=0)
        for model, chunks in encoded_batches.items()
    }
    return selected, random


def _ensure_encoding_params(
    payload: dict,
    encoding_names: list[str],
    device: torch.device,
    cache: dict[str, Any] | None,
    encoding_root_map: dict[str, Path] | None = None,
) -> dict[str, Any]:
    if cache is None:
        cache = {}

    missing = [name for name in encoding_names if name not in cache]
    if not missing:
        return cache

    config = payload.get("config", {})
    paths = config.get("paths", {})
    default_root = Path(
        paths.get(
            "encoding_root",
            "/home/jroth/rsa_based_selection/outputs/deepvision_encoding_models",
        )
    )

    model_csv = Path(paths.get("model_list_csv", default_model_list_csv()))
    if not model_csv.exists():
        model_csv = default_model_list_csv()

    for enc_name in missing:
        enc_root = (encoding_root_map or {}).get(enc_name, default_root)
        params = load_encoding_params_by_encoding(
            encoding_root=enc_root,
            model_list_csv=model_csv,
            encoding_names=[enc_name],
            device=device,
        )
        cache.update(params)

    return cache
