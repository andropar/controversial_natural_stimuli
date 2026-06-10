"""Conditional sparse random projection for large layer features."""

from __future__ import annotations

import re
import warnings
from typing import Dict, Tuple

import numpy as np
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.random_projection import SparseRandomProjection


SRP_TARGET_DIM = 5920
SRP_SEED = 0
FEATURE_PROTOCOL = "flatten_srp5920_v1"


def safe_layer_key(layer: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(layer)).strip("_")


def flattened_feature_dim(features: np.ndarray) -> int:
    if features.ndim == 1:
        return 1
    return int(np.prod(features.shape[1:]))


def _fit_srp(original_dim: int, *, target_dim: int, seed: int) -> SparseRandomProjection:
    srp = SparseRandomProjection(n_components=target_dim, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataDimensionalityWarning)
        # ``fit`` only needs the feature dimensionality, but sklearn still
        # validates values; zeros avoid uninitialized NaNs from ``np.empty``.
        srp.fit(np.zeros((1, original_dim), dtype=np.float32))
    return srp


def _srp_meta(original_dim: int, stored_dim: int, *, target_dim: int, seed: int):
    return {
        "srp_applied": True,
        "srp_target_dim": int(target_dim),
        "srp_seed": int(seed),
        "original_feature_dim": int(original_dim),
        "stored_feature_dim": int(stored_dim),
        "feature_protocol": FEATURE_PROTOCOL,
    }


class SRPProjectorCache:
    """Cache deterministic SRP matrices by original feature dimensionality."""

    def __init__(self, *, target_dim: int = SRP_TARGET_DIM, seed: int = SRP_SEED):
        self.target_dim = int(target_dim)
        self.seed = int(seed)
        self._projectors: Dict[int, SparseRandomProjection] = {}

    def transform(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, int | bool | str]]:
        if features.ndim != 2:
            features = features.reshape(features.shape[0], -1)
        features = features.astype(np.float32, copy=False)
        original_dim = int(features.shape[1])
        if original_dim not in self._projectors:
            self._projectors[original_dim] = _fit_srp(
                original_dim,
                target_dim=self.target_dim,
                seed=self.seed,
            )
        out = self._projectors[original_dim].transform(features).astype(np.float32, copy=False)
        return out, _srp_meta(
            original_dim,
            int(out.shape[1]),
            target_dim=self.target_dim,
            seed=self.seed,
        )


def maybe_apply_srp(
    features: np.ndarray,
    *,
    target_dim: int = SRP_TARGET_DIM,
    seed: int = SRP_SEED,
) -> Tuple[np.ndarray, Dict[str, int | bool | str]]:
    """Flatten features and project to target_dim with deterministic SRP.

    The projection is deterministic for a given input dimensionality and seed.
    We intentionally project even when ``original_dim < target_dim`` to match
    the veRSA procedure's constant 5920-dimensional layer space.
    """
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    cache = SRPProjectorCache(target_dim=target_dim, seed=seed)
    return cache.transform(features)


def metadata_arrays(layer: str, meta: Dict[str, int | bool | str]) -> Dict[str, np.ndarray]:
    safe = safe_layer_key(layer)
    return {
        f"_srp_applied__{safe}": np.array(bool(meta["srp_applied"]), dtype=bool),
        f"_srp_target_dim__{safe}": np.array(int(meta["srp_target_dim"]), dtype=np.int32),
        f"_srp_seed__{safe}": np.array(int(meta["srp_seed"]), dtype=np.int32),
        f"_original_feature_dim__{safe}": np.array(int(meta["original_feature_dim"]), dtype=np.int32),
        f"_stored_feature_dim__{safe}": np.array(int(meta["stored_feature_dim"]), dtype=np.int32),
        f"_feature_protocol__{safe}": np.array(str(meta.get("feature_protocol", FEATURE_PROTOCOL))),
    }


def read_layer_metadata(npz, layer: str) -> Dict[str, int | bool]:
    safe = safe_layer_key(layer)

    def get(name, default):
        key = f"_{name}__{safe}"
        if key not in npz.files:
            return default
        return np.asarray(npz[key]).item()

    return {
        "srp_applied": bool(get("srp_applied", False)),
        "srp_target_dim": int(get("srp_target_dim", SRP_TARGET_DIM)),
        "srp_seed": int(get("srp_seed", SRP_SEED)),
        "original_feature_dim": int(get("original_feature_dim", 0)),
        "stored_feature_dim": int(get("stored_feature_dim", 0)),
        "feature_protocol": str(get("feature_protocol", "")),
    }


def cached_layer_current(npz, layer: str, *, target_dim: int = SRP_TARGET_DIM) -> bool:
    if layer not in npz.files:
        return False
    dim = flattened_feature_dim(npz[layer])
    if dim != target_dim:
        return False
    meta = read_layer_metadata(npz, layer)
    if meta.get("feature_protocol") != FEATURE_PROTOCOL:
        return False
    if not meta.get("srp_applied", False):
        return False
    return True
