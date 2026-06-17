"""Random-feature loading helpers for evaluation analyses."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def npy_feature_cache_path(path: Path) -> Path:
    """Return the uncompressed mmap cache path for a compressed feature file."""
    return path.parent / "_npy_cache" / f"{path.stem}.npy"


def _feature_array_from_npz(data: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
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


def ensure_npy_feature_cache(
    random_feature_dir: Path,
    model_names: Sequence[str],
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Create uncompressed ``.npy`` feature caches for fast mmap loading."""
    cache_paths: list[Path] = []
    cache_dir = random_feature_dir / "_npy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for model_name in model_names:
        npz_path = random_feature_dir / f"{model_name}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing random feature cache for {model_name}: {npz_path}")
        npy_path = npy_feature_cache_path(npz_path)
        if npy_path.exists() and not overwrite:
            cache_paths.append(npy_path)
            continue
        tmp_path = npy_path.with_suffix(npy_path.suffix + ".tmp")
        with np.load(npz_path, allow_pickle=True) as data:
            arr = _feature_array_from_npz(data, npz_path)
            with tmp_path.open("wb") as f:
                np.save(f, np.asarray(arr))
        tmp_path.replace(npy_path)
        cache_paths.append(npy_path)
    return cache_paths


def load_npz_feature_array(path: Path, n_images: int | None = None) -> np.ndarray:
    """Load a 2D feature array from a local ``.npz`` feature cache file."""
    npy_path = npy_feature_cache_path(path)
    if npy_path.exists():
        arr = np.load(npy_path, mmap_mode="r")
        if n_images is not None:
            return arr[: min(int(n_images), arr.shape[0])]
        return arr

    with np.load(path, allow_pickle=True) as data:
        arr = _feature_array_from_npz(data, path)
        arr = np.asarray(arr, dtype=np.float32)
        if n_images is not None:
            arr = arr[: min(int(n_images), arr.shape[0])]
        return arr


def available_random_models(
    random_feature_dir: Path,
    model_names: Sequence[str],
) -> list[str]:
    """Return models with a matching local random-feature cache file."""
    return [
        model
        for model in model_names
        if (random_feature_dir / f"{model}.npz").exists()
    ]


def load_random_feature_cache(
    random_feature_dir: Path,
    model_names: Sequence[str],
    n_random: int,
    view_name: str = "raw",
) -> dict[str, np.ndarray]:
    """Load local cached random features for the requested models."""
    if view_name != "raw":
        print(f"  [INFO] random view '{view_name}' uses raw natural-pool features")

    features: dict[str, np.ndarray] = {}
    for model_name in model_names:
        path = random_feature_dir / f"{model_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing random feature cache for {model_name}: {path}")
        features[model_name] = load_npz_feature_array(path, n_random)

    print(
        f"  [DEBUG] Loaded local random pool: {len(features)} models, "
        f"{next(iter(features.values())).shape[0]} samples"
    )
    return features


def make_random_feature_cache_loader(random_feature_dir: Path):
    """Build a track-loader-compatible callable for local random features."""

    def _loader(
        payload: dict,
        model_names: list[str],
        n_random: int,
        view_name: str = "raw",
    ) -> dict[str, np.ndarray]:
        del payload
        return load_random_feature_cache(
            random_feature_dir=random_feature_dir,
            model_names=model_names,
            n_random=n_random,
            view_name=view_name,
        )

    return _loader
