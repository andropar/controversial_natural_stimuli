"""Shared utilities for the evaluation pipeline."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

# Selection variant types
SelectionVariant = Literal["final", "greedy", "best_raw_combined"]
VALID_SELECTION_VARIANTS: tuple[SelectionVariant, ...] = (
    "final",
    "greedy",
    "best_raw_combined",
)

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Valid environment names for path overrides
VALID_ENVS = ("iris", "raven")


# -----------------------------------------------------------------------------
# Environment Path Loading
# -----------------------------------------------------------------------------


def _load_env_paths(env: str) -> dict:
    """Load paths from conf/paths/{env}.yaml.

    Args:
        env: Environment name ("iris" or "raven")

    Returns:
        Dictionary of paths from the config file
    """
    config_path = PROJECT_ROOT / "conf" / "paths" / f"{env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Environment config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("paths", {})


def _warn_path_divergence(old_paths: dict, new_paths: dict, env: str) -> None:
    """Warn if path basenames differ between payload and env config.

    This helps catch potential misconfigurations where the env paths
    point to different data than what the selection was run with.

    Args:
        old_paths: Paths from the payload
        new_paths: Paths from the env config
        env: Environment name for the warning message
    """
    keys_to_check = ["subset_root", "model_list_csv", "encoding_root"]
    for key in keys_to_check:
        old = old_paths.get(key)
        new = new_paths.get(key)
        if old and new:
            old_name = Path(old).name
            new_name = Path(new).name
            # Check basename and parent dir name
            old_parent = Path(old).parent.name
            new_parent = Path(new).parent.name
            if old_name != new_name:
                warnings.warn(
                    f"Path '{key}' basename differs: payload='{old_name}', "
                    f"env={env}='{new_name}'. This may cause issues."
                )
            elif old_parent != new_parent:
                # Just log, don't warn - parent dirs often differ between machines
                print(f"  Note: '{key}' parent dir differs (payload='{old_parent}', env={env}='{new_parent}')")


# -----------------------------------------------------------------------------
# Best Raw Combined Reconstruction
# -----------------------------------------------------------------------------


def _reconstruct_greedy_indices_from_history(
    payload: dict,
) -> np.ndarray | None:
    """Reconstruct greedy indices from refinement history for older payloads.

    For payloads without greedy_indices, we can reconstruct them from
    pass 0 of refinement_history - each record's old_idx at position
    gives us the original greedy index.

    Args:
        payload: Selection payload dictionary

    Returns:
        Reconstructed greedy indices, or None if reconstruction not possible
    """
    refinement_history = payload.get("refinement_history")
    if not refinement_history:
        return None

    # Get selected_global_indices as the final state
    selected_indices = payload.get("selected_global_indices")
    if selected_indices is None:
        return None
    selected_indices = np.asarray(selected_indices)
    n_selected = len(selected_indices)

    # Group by pass
    records_by_pass: dict[int, list[dict]] = {}
    for record in refinement_history:
        pass_num = record.get("pass", record.get("pass_num", 0))
        if pass_num not in records_by_pass:
            records_by_pass[pass_num] = []
        records_by_pass[pass_num].append(record)

    if 0 not in records_by_pass:
        return None

    # Pass 0 should have one record per position
    pass0_records = records_by_pass[0]
    if len(pass0_records) != n_selected:
        print(
            f"  [WARN] Pass 0 has {len(pass0_records)} records but expected {n_selected}"
        )

    # Build greedy indices from pass 0 old_idx values
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
    """Reconstruct best_raw_combined indices from refinement history.

    For older payloads that don't have best_raw_combined_indices stored,
    this function replays the refinement history to find the state with
    the best raw combined score.

    Args:
        payload: Selection payload dictionary containing:
            - greedy_indices: Indices before refinement (or reconstructible from history)
            - refinement_history: List of refinement records with scores_per_track
        weights: Optional track weights for computing raw combined score.
            If None, uses equal weights (simple average).

    Returns:
        Tuple of (best_indices, best_score, best_pass) where:
            - best_indices: The indices at the best raw combined score
            - best_score: The best raw combined score achieved
            - best_pass: Which pass achieved the best (-1 = greedy, 0+ = refinement pass)

    Raises:
        ValueError: If required data is missing from payload
    """
    greedy_indices = payload.get("greedy_indices")
    if greedy_indices is None:
        # Try to reconstruct from refinement history (for older payloads)
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
    if not refinement_history:
        # No refinement - greedy is the best by default
        # Try to get greedy score from scores history
        scores_per_view = payload.get("scores_per_view_history") or {}
        scores_per_rep = payload.get("scores_per_rep_history") or {}
        all_track_scores = {**scores_per_view, **scores_per_rep}

        if all_track_scores:
            # Get the final greedy score
            final_scores = {
                name: scores[-1] for name, scores in all_track_scores.items() if scores
            }
            greedy_score = _compute_weighted_mean(final_scores, weights)
        else:
            greedy_score = float("-inf")

        return greedy_indices.copy(), greedy_score, -1

    # Compute greedy's raw combined score (end of greedy phase)
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

    # Track best seen
    best_indices = greedy_indices.copy()
    best_score = greedy_combined
    best_pass = -1  # -1 indicates greedy phase

    # Group refinement records by pass
    records_by_pass: dict[int, list[dict]] = {}
    for record in refinement_history:
        pass_num = record.get("pass", record.get("pass_num", 0))
        if pass_num not in records_by_pass:
            records_by_pass[pass_num] = []
        records_by_pass[pass_num].append(record)

    # Replay refinement history record by record, checking score after each replacement
    current_indices = greedy_indices.copy()

    for pass_num in sorted(records_by_pass.keys()):
        pass_records = records_by_pass[pass_num]

        # Sort by position to ensure correct replay order
        pass_records = sorted(pass_records, key=lambda r: r.get("position", 0))

        # Process each record in this pass
        for record in pass_records:
            if record.get("replaced", False):
                position = record.get("position", 0)
                new_idx = record.get("new_idx", -1)
                if 0 <= position < len(current_indices) and new_idx >= 0:
                    current_indices[position] = new_idx

            # Check score after this record (whether replaced or not)
            scores_per_track = record.get("scores_per_track")
            if scores_per_track:
                record_score = _compute_weighted_mean(scores_per_track, weights)

                if record_score > best_score:
                    best_score = record_score
                    best_indices = current_indices.copy()
                    best_pass = pass_num

    return best_indices, best_score, best_pass


def _compute_weighted_mean(
    scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted mean of scores.

    Args:
        scores: Dict mapping track name to score
        weights: Optional dict mapping track name to weight.
            If None, uses equal weights (simple average).

    Returns:
        Weighted mean score
    """
    if not scores:
        return float("-inf")

    if weights is None:
        # Simple average
        return sum(scores.values()) / len(scores)

    # Weighted average
    total_weight = 0.0
    weighted_sum = 0.0
    for name, score in scores.items():
        w = weights.get(name, 1.0)
        weighted_sum += w * score
        total_weight += w

    if total_weight == 0:
        return float("-inf")

    return weighted_sum / total_weight


def _get_memory_usage_mb() -> dict[str, float]:
    """Get current memory usage in MB."""
    result = {}
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        result["rss_mb"] = mem_info.rss / 1024 / 1024
        result["vms_mb"] = mem_info.vms / 1024 / 1024
        # System-wide
        vm = psutil.virtual_memory()
        result["system_used_mb"] = vm.used / 1024 / 1024
        result["system_available_mb"] = vm.available / 1024 / 1024
        result["system_percent"] = vm.percent
    except ImportError:
        result["error"] = "psutil not installed"
    return result


def _get_gpu_memory_mb() -> dict[str, float]:
    """Get GPU memory usage in MB."""
    result = {}
    if torch.cuda.is_available():
        result["allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        result["reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        result["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    return result


def log_memory(label: str) -> None:
    """Log current memory usage."""
    cpu_mem = _get_memory_usage_mb()
    gpu_mem = _get_gpu_memory_mb()

    parts = [f"[MEMORY {label}]"]
    if "rss_mb" in cpu_mem:
        parts.append(
            f"CPU: {cpu_mem['rss_mb']:.0f}MB (system: {cpu_mem['system_percent']:.0f}% used, {cpu_mem['system_available_mb']:.0f}MB free)"
        )
    if gpu_mem:
        parts.append(
            f"GPU: {gpu_mem['allocated_mb']:.0f}MB allocated, {gpu_mem['reserved_mb']:.0f}MB reserved"
        )

    print(" | ".join(parts))


from cstims.data_loader import load_natural_features_with_metadata
from cstims.encoding.linear import (
    encode_batch_for_all_encodings,
    load_encoding_params_by_encoding,
)
from cstims.evaluation.data_loader import load_selected_features
from cstims.evaluation.io import load_payload


def load_selection_payload(result_dir: Path) -> dict:
    """Load and validate selection payload from result directory.

    Args:
        result_dir: Path to selection result directory containing selected_stimuli_data.pkl

    Returns:
        Loaded payload dictionary
    """
    return load_payload(result_dir)


def get_variant_feature_keys(variant: SelectionVariant) -> tuple[str, str, str]:
    """Get payload keys for features based on selection variant.

    Args:
        variant: Which selection variant to use

    Returns:
        Tuple of (features_raw_key, features_by_encoding_key, indices_key)
    """
    if variant == "final":
        return (
            "selected_features_raw",
            "selected_features_by_encoding",
            "selected_global_indices",
        )
    elif variant == "greedy":
        return "greedy_features_raw", "greedy_features_by_encoding", "greedy_indices"
    elif variant == "best_raw_combined":
        return (
            "best_raw_combined_features_raw",
            "best_raw_combined_features_by_encoding",
            "best_raw_combined_indices",
        )
    else:
        raise ValueError(
            f"Unknown selection variant: {variant}. Valid: {VALID_SELECTION_VARIANTS}"
        )


def get_selected_indices(
    payload: dict, variant: SelectionVariant = "final"
) -> np.ndarray:
    """Get selected indices for a given selection variant.

    For 'best_raw_combined' variant, this function ALWAYS reconstructs from
    refinement history to ensure correctness (stored data may be incorrect
    due to a bug in some selection runs).

    Args:
        payload: Selection payload dictionary
        variant: Which selection variant to use

    Returns:
        Array of selected global indices
    """
    # For best_raw_combined, ALWAYS reconstruct from history
    # (stored best_raw_combined_indices may be incorrect due to selection bug)
    if variant == "best_raw_combined":
        print(
            "  [INFO] Reconstructing best_raw_combined from refinement history..."
        )
        # Get weights from track_aggregation config if available
        track_agg = payload.get("track_aggregation", {})
        weights = track_agg.get("weights")

        indices, score, pass_num = reconstruct_best_raw_combined_from_history(
            payload, weights
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


def get_output_dir(result_dir: Path, subdir: str = "eval_pipeline") -> Path:
    """Get or create output directory for evaluation results.

    Args:
        result_dir: Base result directory
        subdir: Subdirectory name for pipeline outputs

    Returns:
        Path to output directory (created if needed)
    """
    output_dir = result_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_all_tracks_for_evaluation(payload: dict) -> list[dict]:
    """Extract all tracks that should be evaluated.

    Discovers tracks from:
    1. track_definitions in payload (identity and encoding tracks)
    2. selected_features_by_encoding keys (for any additional encodings)

    Args:
        payload: Selection payload dictionary

    Returns:
        List of track dictionaries with keys: name, type, encoding_name (optional)
    """
    tracks = []
    seen_names = set()

    # From track_definitions
    track_defs = payload.get("track_definitions", [])
    for track in track_defs:
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

    # Also check selected_features_by_encoding for any extras
    features_by_encoding = payload.get("selected_features_by_encoding") or {}
    for enc_name in features_by_encoding.keys():
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


def get_track_noise_variances(
    payload: dict, track_name: str
) -> dict[str, float] | None:
    """Get noise variance by model for a specific track.

    Args:
        payload: Selection payload dictionary
        track_name: Name of the track

    Returns:
        Dict mapping model name to noise variance, or None if not found
    """
    var_noise = payload.get("var_noise_by_model", {})
    return var_noise.get(track_name)


def load_features_for_track(
    payload: dict,
    track: dict,
    device: torch.device,
    encoding_params_cache: dict[str, Any] | None = None,
    n_random: int = 10000,
    selection_variant: SelectionVariant = "final",
    encoding_root_map: dict[str, Path] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Load selected and random features for a track.

    Args:
        payload: Selection payload dictionary
        track: Track definition dict with name, type, encoding_name
        device: Torch device for tensors
        encoding_params_cache: Optional cache for encoding parameters
        n_random: Number of random samples to load
        selection_variant: Which selection to evaluate ("final", "greedy", "best_raw_combined")
        encoding_root_map: Optional per-encoding root mapping

    Returns:
        Tuple of (selected_features, random_features)
        - selected_features: Dict[model_name, Tensor] on device
        - random_features: Dict[model_name, np.ndarray]
    """
    track_type = track.get("type", "identity")
    track_name = track["name"]
    encoding_name = track.get("encoding_name")

    model_names = payload["model_names"]
    config = payload.get("config", {})
    paths = config.get("paths", {})

    if track_type == "identity":
        # Load directly from payload or memmaps
        selected = _load_selected_identity(
            payload, track_name, device, selection_variant
        )
        random = _load_random_identity(payload, model_names, n_random, track_name)

    elif track_type == "encoding":
        # Load raw features, then apply encoding transform
        enc_name = encoding_name or track_name
        selected, random = _load_encoded_features(
            payload=payload,
            encoding_name=enc_name,
            model_names=model_names,
            device=device,
            encoding_params_cache=encoding_params_cache,
            n_random=n_random,
            selection_variant=selection_variant,
            encoding_root_map=encoding_root_map,
        )
    else:
        raise ValueError(f"Unsupported track type: {track_type}")

    return selected, random


def _load_selected_identity(
    payload: dict,
    view_name: str,
    device: torch.device,
    selection_variant: SelectionVariant = "final",
) -> dict[str, torch.Tensor]:
    """Load selected features for an identity track.

    For best_raw_combined variant, this ALWAYS extracts features on-the-fly
    using reconstructed indices (stored features may be incorrect due to
    a bug in some selection runs).
    """
    # For best_raw_combined, ALWAYS extract on-the-fly using reconstructed indices
    # (stored best_raw_combined_features_raw may be incorrect due to selection bug)
    if selection_variant == "best_raw_combined":
        return _extract_features_for_variant(payload, selection_variant, device)

    raw_key, _, _ = get_variant_feature_keys(selection_variant)

    # For greedy variant, try the variant-specific raw features first
    if selection_variant == "greedy":
        variant_features = payload.get(raw_key)
        if variant_features:
            return {
                k: torch.tensor(v, device=device, dtype=torch.float32)
                if not isinstance(v, torch.Tensor)
                else v.to(device=device, dtype=torch.float32)
                for k, v in variant_features.items()
            }

    # Try selected_features_by_view first (for final variant or fallback)
    features_by_view = payload.get("selected_features_by_view") or {}
    if view_name in features_by_view:
        return {
            k: torch.tensor(v, device=device, dtype=torch.float32)
            if not isinstance(v, torch.Tensor)
            else v.to(device=device, dtype=torch.float32)
            for k, v in features_by_view[view_name].items()
        }

    # Try selected_features_raw for "raw" view
    if view_name == "raw" and payload.get("selected_features_raw"):
        return {
            k: torch.tensor(v, device=device, dtype=torch.float32)
            if not isinstance(v, torch.Tensor)
            else v.to(device=device, dtype=torch.float32)
            for k, v in payload["selected_features_raw"].items()
        }

    # Fallback to single selected_features
    if payload.get("selected_features"):
        return {
            k: torch.tensor(v, device=device, dtype=torch.float32)
            if not isinstance(v, torch.Tensor)
            else v.to(device=device, dtype=torch.float32)
            for k, v in payload["selected_features"].items()
        }

    # Last resort: use load_selected_features from evaluation module
    return load_selected_features(payload, device, view_name=view_name)


def _extract_features_for_variant(
    payload: dict,
    selection_variant: SelectionVariant,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Extract features on-the-fly for a selection variant.

    Used when the variant's features aren't cached in the payload
    (e.g., best_raw_combined for older runs).

    Args:
        payload: Selection payload dictionary
        selection_variant: Which selection variant
        device: Torch device

    Returns:
        Dict mapping model name to feature tensor
    """
    print(
        f"  [INFO] Extracting features on-the-fly for variant '{selection_variant}'..."
    )

    # Get the indices for this variant (will reconstruct if needed)
    indices = get_selected_indices(payload, selection_variant)

    # Load features from data source
    config = payload.get("config", {})
    paths = config.get("paths", {})
    model_names = payload["model_names"]

    # Try to load from memmap first (faster)
    preprocessed_dirs = paths.get("preprocessed_dirs", {})
    preproc_dir = preprocessed_dirs.get("raw")

    if preproc_dir and Path(preproc_dir).exists():
        # Load from memmap and index
        features, _ = load_natural_features_with_metadata(
            subset_root=Path(paths["subset_root"]),
            preprocessed_dir=Path(preproc_dir),
            model_names=model_names,
            max_images=None,  # Load all to get correct indexing
            model_csv=Path(paths["model_list_csv"]),
        )
        # Index into the loaded features
        result = {}
        for name in model_names:
            arr = features[name][indices]
            result[name] = torch.tensor(arr, device=device, dtype=torch.float32)
        return result
    else:
        # Fallback to shard-based loading (slower but always works)
        # This requires loading all features and then indexing
        features, _ = load_natural_features_with_metadata(
            subset_root=Path(paths["subset_root"]),
            model_names=model_names,
            max_images=None,
            model_csv=Path(paths["model_list_csv"]),
        )
        result = {}
        for name in model_names:
            arr = features[name][indices]
            result[name] = torch.tensor(arr, device=device, dtype=torch.float32)
        return result


def _load_random_identity(
    payload: dict,
    model_names: list[str],
    n_random: int,
    view_name: str = "raw",
) -> dict[str, np.ndarray]:
    """Load random baseline features for an identity track."""
    config = payload.get("config", {})
    paths = config.get("paths", {})
    preprocessed_dirs = paths.get("preprocessed_dirs", {})

    preproc_dir = preprocessed_dirs.get(view_name) or preprocessed_dirs.get("raw")
    if not preproc_dir:
        raise ValueError(f"No preprocessed directory found for view '{view_name}'")

    print(
        f"  [DEBUG] Loading random features via shard-based loading (max_images={n_random})"
    )
    print(f"  [DEBUG] Requested n_random={n_random}, models={len(model_names)}")
    log_memory("before_load_random")

    # Try shard-based loading (respects max_images) first; if any model comes
    # back empty (e.g., shards missing on this machine), fall back to the
    # memmap fast path and subsample afterwards.
    features, _ = load_natural_features_with_metadata(
        subset_root=Path(paths["subset_root"]),
        model_names=model_names,
        model_csv=Path(paths["model_list_csv"]),
        max_images=n_random,
        preprocessed_dir=None,  # Force shard-based loading to respect max_images
    )
    if any(features[m].size == 0 for m in model_names):
        print(
            "  [DEBUG] Shard-based loading returned empty arrays; "
            "falling back to memmap fast path + subsampling."
        )
        features_full, _ = load_natural_features_with_metadata(
            subset_root=Path(paths["subset_root"]),
            model_names=model_names,
            model_csv=Path(paths["model_list_csv"]),
            preprocessed_dir=Path(preproc_dir),
        )
        n_total = next(iter(features_full.values())).shape[0]
        n_take = min(n_random, n_total)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, size=n_take, replace=False)
        idx.sort()
        features = {m: np.asarray(features_full[m][idx]) for m in model_names}

    # Log what we got
    log_memory("after_load_random")
    total_bytes = 0
    for model_name, arr in features.items():
        is_memmap = isinstance(arr, np.memmap)
        arr_bytes = arr.nbytes
        total_bytes += arr_bytes
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
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Load features for an encoding track by projecting raw features.

    Args:
        payload: Selection payload dictionary
        encoding_name: Name of the encoding to apply
        model_names: List of model names
        device: Torch device for computation
        encoding_params_cache: Optional cache for encoding parameters
        n_random: Number of random samples to load
        batch_size: Batch size for encoding random features (to avoid OOM)
        selection_variant: Which selection to evaluate ("final", "greedy", "best_raw_combined")
        encoding_root_map: Optional per-encoding root mapping

    Returns:
        Tuple of (selected_features, random_features)
    """
    config = payload.get("config", {})
    paths = config.get("paths", {})

    # For best_raw_combined, ALWAYS compute from raw features using reconstructed indices
    # (stored best_raw_combined encoded features may be incorrect due to selection bug)
    if selection_variant == "best_raw_combined":
        raw_selected = _load_selected_identity(
            payload, "raw", device, selection_variant
        )
        encoding_params = _ensure_encoding_params(
            payload, [encoding_name], device, encoding_params_cache,
            encoding_root_map=encoding_root_map,
        )
        encoded = encode_batch_for_all_encodings(
            raw_selected, {encoding_name: encoding_params[encoding_name]}
        )
        selected = encoded[encoding_name]
    else:
        # When using a custom encoding_root_map, skip pre-computed features
        # (they were encoded with the original shared encodings)
        use_precomputed = encoding_root_map is None

        _, enc_key, _ = get_variant_feature_keys(selection_variant)

        # Check if we have pre-computed encoded features in payload for this variant
        features_by_encoding = payload.get(enc_key) or {} if use_precomputed else {}

        # Fall back to final selection features if variant-specific not available
        if not features_by_encoding and use_precomputed:
            features_by_encoding = payload.get("selected_features_by_encoding") or {}

        if encoding_name in features_by_encoding:
            selected = {
                k: torch.tensor(v, device=device, dtype=torch.float32)
                if not isinstance(v, torch.Tensor)
                else v.to(device=device, dtype=torch.float32)
                for k, v in features_by_encoding[encoding_name].items()
            }
        else:
            # Load raw selected features and encode
            raw_selected = _load_selected_identity(
                payload, "raw", device, selection_variant
            )
            encoding_params = _ensure_encoding_params(
                payload, [encoding_name], device, encoding_params_cache,
                encoding_root_map=encoding_root_map,
            )
            encoded = encode_batch_for_all_encodings(
                raw_selected, {encoding_name: encoding_params[encoding_name]}
            )
            selected = encoded[encoding_name]

    # Load raw random features (stays on CPU as numpy)
    print(f"  [DEBUG] Loading raw random features for encoding '{encoding_name}'...")
    raw_random = _load_random_identity(payload, model_names, n_random, "raw")

    # Ensure encoding params are loaded
    encoding_params = _ensure_encoding_params(
        payload, [encoding_name], device, encoding_params_cache,
        encoding_root_map=encoding_root_map,
    )

    # Encode random features in batches to avoid OOM
    n_samples = next(iter(raw_random.values())).shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    print(
        f"  [DEBUG] Encoding {n_samples} samples in {n_batches} batches of {batch_size}"
    )
    log_memory("before_encoding_loop")

    encoded_batches: dict[str, list[np.ndarray]] = {m: [] for m in model_names}

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)

        # Extract batch and move to GPU
        batch_torch = {
            k: torch.tensor(v[start_idx:end_idx], device=device, dtype=torch.float32)
            for k, v in raw_random.items()
        }

        # Encode batch
        encoded_batch = encode_batch_for_all_encodings(
            batch_torch, {encoding_name: encoding_params[encoding_name]}
        )

        # Move results to CPU and collect
        for model_name in model_names:
            encoded_batches[model_name].append(
                encoded_batch[encoding_name][model_name].cpu().numpy()
            )

        # Free GPU memory
        del batch_torch, encoded_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Concatenate all batches
    random = {k: np.concatenate(v, axis=0) for k, v in encoded_batches.items()}

    return selected, random


def _ensure_encoding_params(
    payload: dict,
    encoding_names: list[str],
    device: torch.device,
    cache: dict[str, Any] | None,
    encoding_root_map: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Load encoding parameters, using cache if available.

    Args:
        encoding_root_map: Optional per-encoding root mapping
            (e.g. {"sub-01": Path("..."), "sub-03": Path("...")}). If provided,
            each encoding is loaded from its own root. Otherwise falls back to
            the payload's encoding_root.
    """
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

    # Resolve model_list_csv — payload may have Raven cluster path
    model_list_csv = Path(paths["model_list_csv"])
    if not model_list_csv.exists():
        # Fall back to local path
        project_root = Path(__file__).resolve().parents[4]
        model_list_csv = project_root / "resources" / "model_list.csv"

    for enc_name in missing:
        enc_root = (encoding_root_map or {}).get(enc_name, default_root)
        params = load_encoding_params_by_encoding(
            encoding_root=enc_root,
            model_list_csv=model_list_csv,
            encoding_names=[enc_name],
            device=device,
        )
        cache.update(params)

    return cache


def add_standard_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add standard CLI arguments to parser.

    Args:
        parser: ArgumentParser to modify

    Returns:
        Modified parser with standard args added
    """
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Path to selection result directory containing selected_stimuli_data.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <result-dir>/eval_pipeline/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=VALID_ENVS,
        default=None,
        help="Override payload paths with paths from conf/paths/{env}.yaml (e.g., 'iris' or 'raven')",
    )
    return parser


def setup_from_args(args: argparse.Namespace) -> tuple[dict, Path, torch.device]:
    """Setup common resources from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (payload, output_dir, device)
    """
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load payload
    payload = load_selection_payload(args.result_dir)

    # Override paths if --env specified
    env = getattr(args, "env", None)
    if env:
        env_paths = _load_env_paths(env)
        old_paths = payload.get("config", {}).get("paths", {})
        _warn_path_divergence(old_paths, env_paths, env)

        # Ensure config exists and override paths
        if "config" not in payload:
            payload["config"] = {}
        payload["config"]["paths"] = env_paths
        print(f"Using paths from env={env}")

    # Setup output directory
    output_dir = args.output_dir or get_output_dir(args.result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    return payload, output_dir, device
