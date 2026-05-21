import csv
import os
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from cstims.data_loader import load_natural_features_with_metadata


def load_selected_features(
    payload: dict, device: torch.device, view_name: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Load selected features from payload or shards.

    Args:
        payload: The selection payload dictionary
        device: Target device for tensors
        view_name: For multi-view payloads, specify which view to load.
                   If None and multi-view, returns the first view.
                   For single-view payloads, this parameter is ignored.

    Returns:
        Dict mapping model names to feature tensors
    """
    multi_view = payload.get("multi_view", False)

    if multi_view:
        # Check for new format first
        if (
            "selected_features_by_view" in payload
            and payload["selected_features_by_view"]
        ):
            print("Using pre-saved selected_features_by_view from payload")
            features_by_view = payload["selected_features_by_view"]
        # Backward compatibility: check if old format stored multi-view features under "selected_features"
        elif (
            "selected_features" in payload
            and payload["selected_features"]
            and isinstance(payload["selected_features"], dict)
            and all(isinstance(v, dict) for v in payload["selected_features"].values())
        ):
            print("Using pre-saved selected_features from payload (old multi-view format)")
            features_by_view = payload["selected_features"]
        else:
            raise ValueError(
                "Multi-view payload detected but 'selected_features_by_view' not found. "
                "Cannot load features from shards for multi-view yet."
            )

        if view_name is None:
            view_name = list(features_by_view.keys())[0]
            print(f"No view_name specified, using first view: {view_name}")

        if view_name not in features_by_view:
            raise ValueError(
                f"View '{view_name}' not found in payload. Available views: {list(features_by_view.keys())}"
            )

        selected_features = features_by_view[view_name]
        return {
            k: torch.tensor(v, device=device, dtype=torch.float32)
            if not isinstance(v, torch.Tensor)
            else v.to(device=device, dtype=torch.float32)
            for k, v in selected_features.items()
        }

    if "selected_features" in payload and payload["selected_features"]:
        print("Using pre-saved selected_features from payload")
        return {
            k: torch.tensor(v, device=device, dtype=torch.float32)
            if not isinstance(v, torch.Tensor)
            else v.to(device=device, dtype=torch.float32)
            for k, v in payload["selected_features"].items()
        }

    print("Loading features from shards...")
    config = payload.get("config", {})
    paths = config.get("paths", {})
    model_names = payload["model_names"]
    selected_indices = [int(x) for x in payload["selected_global_indices"]]

    subset_root = Path(paths.get("subset_root", ""))
    
    # Resolve preprocessed_dir: check explicit 'preprocessed_dir' first, then 'preprocessed_dirs'
    preprocessed_dir = None
    if paths.get("preprocessed_dir"):
        preprocessed_dir = Path(paths["preprocessed_dir"])
    else:
        # Fallback to preprocessed_dirs (plural)
        preprocessed_dirs = paths.get("preprocessed_dirs", {})
        if preprocessed_dirs:
            target_view = view_name
            if target_view is None:
                # Try to find default view from config or take first available
                views = config.get("views", [])
                if views:
                    target_view = views[0]
                else:
                    # Take the first key from preprocessed_dirs
                    target_view = next(iter(preprocessed_dirs.keys()))
            
            if target_view and target_view in preprocessed_dirs:
                preprocessed_dir = Path(preprocessed_dirs[target_view])

    if preprocessed_dir and preprocessed_dir.exists():
        print(f"Loading from preprocessed dir: {preprocessed_dir}")
        all_features, _ = load_natural_features_with_metadata(
            subset_root=subset_root,
            model_names=model_names,
            model_csv=Path(paths.get("model_list_csv", "")),
            max_images=max(selected_indices) + 1,
            preprocessed_dir=preprocessed_dir,
        )
        return {
            name: torch.tensor(
                all_features[name][selected_indices], device=device, dtype=torch.float32
            )
            for name in model_names
        }

    # Fallback to loading from shards
    print("Falling back to loading from payload shard info...")
    shard_info = payload.get("subset_shards", [])
    if not shard_info:
        raise ValueError("No subset_shards in payload and no preprocessed_dir")

    def _load_features_from_shards_simple():
        subset_root = Path(paths.get("subset_root", ""))
        model_csv = Path(paths.get("model_list_csv", ""))
        feature_output_root = (
            Path(paths.get("feature_output_root", ""))
            if paths.get("feature_output_root")
            else None
        )

        shard_starts = [s["start_index"] for s in shard_info]
        indices_by_shard = defaultdict(list)
        for sel_idx in selected_indices:
            shard_idx = bisect_right(shard_starts, sel_idx) - 1
            if shard_idx < 0 or shard_idx >= len(shard_info):
                continue
            local_idx = sel_idx - shard_starts[shard_idx]
            indices_by_shard[shard_idx].append((sel_idx, local_idx))

        with model_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = {row.get("model"): row for row in reader if row.get("model")}

        layer_map = {}
        for name in model_names:
            if name not in rows:
                continue
            row = rows[name]
            layer = row.get("layer") or row.get("layer_uid")
            if layer:
                layer_tag = (
                    str(layer)
                    .replace("/", "_")
                    .replace(os.sep, "_")
                    .replace(".", "_")
                    .replace(" ", "_")
                )
                layer_map[name] = layer_tag

        features_dict = {name: {} for name in model_names}

        for shard_idx, indices_list in indices_by_shard.items():
            tar_path = Path(shard_info[shard_idx]["tar_path"])
            if feature_output_root:
                try:
                    rel = tar_path.relative_to(subset_root)
                except ValueError:
                    rel = Path(tar_path.name)
                base_dir = feature_output_root / rel.parent
            else:
                base_dir = tar_path.parent

            stem = tar_path.stem
            for model_name in model_names:
                layer_tag = layer_map.get(model_name, "")
                feature_path = base_dir / f"{stem}.{model_name}.{layer_tag}.npy"
                if not feature_path.exists():
                    alt_paths = sorted(base_dir.glob(f"{stem}.{model_name}.*.npy"))
                    if alt_paths:
                        feature_path = alt_paths[0]
                    else:
                        continue

                arr = np.load(feature_path)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)

                for global_idx, local_idx in indices_list:
                    features_dict[model_name][global_idx] = arr[local_idx]

        result = {}
        for model_name in model_names:
            ordered = [
                features_dict[model_name][idx]
                for idx in selected_indices
                if idx in features_dict[model_name]
            ]
            if ordered:
                result[model_name] = np.stack(ordered, axis=0)
            else:
                result[model_name] = np.empty((0, 0), dtype=np.float32)

        return result

    feats_np = _load_features_from_shards_simple()
    return {
        name: torch.tensor(feats_np[name], device=device, dtype=torch.float32)
        for name in model_names
        if name in feats_np
    }

