"""Readers for frozen CSTIM analysis cache artifacts.

This module intentionally does not build caches.  It defines the read/validation
contract for analysis-ready artifacts produced elsewhere in the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cstims import constants, paths


_REQUIRED_STIM_INFO_COLUMNS = ("group", "stim_idx", "stim_key")


def _as_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def _feature_cache_dir(cache_dir: Path | str | None = None) -> Path:
    if cache_dir is None:
        return paths.cstim_feature_cache_dir()
    cache_dir = Path(cache_dir).expanduser().resolve()
    if (cache_dir / "cstim").is_dir():
        cache_dir = cache_dir / "cstim"
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"CSTIM feature cache directory not found: {cache_dir}")
    return cache_dir


def _brain_subject_dir(subject: str, cache_root: Path | str | None = None) -> Path:
    root = _as_path(cache_root)
    if root is None:
        return paths.get_subject_data_dir(subject)
    subject_dir = root / subject
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"CSTIM brain cache directory not found: {subject_dir}")
    return subject_dir


def cstim_brain_cache_exists(
    subject: str,
    *,
    cache_root: Path | str | None = None,
    require_repetitions: bool = False,
) -> bool:
    """Return whether the expected CSTIM brain-cache files exist."""
    try:
        subject_dir = _brain_subject_dir(subject, cache_root)
    except FileNotFoundError:
        return False
    names = [
        "cstim_betas_averaged.npz",
        "voxel_metadata.npz",
        "cstim_stimulus_info.csv",
    ]
    if require_repetitions:
        names.append("cstim_betas_by_rep.npz")
    return all((subject_dir / name).is_file() for name in names)


def load_cstim_stimulus_info(
    subject: str,
    *,
    cache_root: Path | str | None = None,
) -> pd.DataFrame:
    """Load and validate CSTIM stimulus metadata for one subject."""
    subject_dir = _brain_subject_dir(subject, cache_root)
    stim_path = _require_file(
        subject_dir / "cstim_stimulus_info.csv",
        f"CSTIM stimulus info cache for {subject}",
    )
    stim_info = pd.read_csv(stim_path)
    _validate_stim_info(stim_info, stim_path)
    return stim_info


def load_cstim_voxel_metadata(
    subject: str,
    *,
    cache_root: Path | str | None = None,
) -> dict[str, np.ndarray]:
    """Load CSTIM voxel metadata for one subject."""
    subject_dir = _brain_subject_dir(subject, cache_root)
    voxel_path = _require_file(
        subject_dir / "voxel_metadata.npz",
        f"CSTIM voxel metadata cache for {subject}",
    )
    with np.load(voxel_path, allow_pickle=True) as voxel_npz:
        return {key: np.asarray(voxel_npz[key]) for key in voxel_npz.files}


def load_cstim_group_info(
    subject: str,
    stimulus_group: str,
    *,
    cache_root: Path | str | None = None,
    sort_by_stim_idx: bool = False,
) -> pd.DataFrame:
    """Load stimulus metadata rows for one subject/group."""
    stim_info = load_cstim_stimulus_info(subject, cache_root=cache_root)
    return _stim_group_frame(
        stim_info,
        stimulus_group,
        sort_by_stim_idx=sort_by_stim_idx,
    ).reset_index(drop=True)


def load_cstim_feature_indices(
    subject: str,
    stimulus_group: str,
    *,
    cache_root: Path | str | None = None,
    sort_by_stim_idx: bool = False,
) -> np.ndarray:
    """Load feature-array indices for one subject/group without loading betas."""
    frame = load_cstim_group_info(
        subject,
        stimulus_group,
        cache_root=cache_root,
        sort_by_stim_idx=sort_by_stim_idx,
    )
    return _feature_indices_from_frame(frame, stimulus_group)


def _validate_stim_info(stim_info: pd.DataFrame, path: Path) -> None:
    missing = [col for col in _REQUIRED_STIM_INFO_COLUMNS if col not in stim_info.columns]
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")

    for group, expected in constants.EXPECTED_CSTIM_IMAGE_COUNTS.items():
        n_rows = int((stim_info["group"] == group).sum())
        if n_rows and n_rows != expected:
            raise RuntimeError(
                f"Unexpected CSTIM row count for {group!r} in {path}: "
                f"found {n_rows}, expected {expected}."
            )


def _stim_group_frame(
    stim_info: pd.DataFrame,
    stimulus_group: str,
    *,
    sort_by_stim_idx: bool,
) -> pd.DataFrame:
    frame = stim_info.loc[stim_info["group"] == stimulus_group].copy()
    if frame.empty:
        raise KeyError(f"Stimulus group {stimulus_group!r} is absent from CSTIM cache")
    if sort_by_stim_idx:
        frame = frame.sort_values("stim_idx", kind="stable")
    return frame


def _feature_indices_from_frame(frame: pd.DataFrame, stimulus_group: str) -> np.ndarray:
    idx = frame["stim_idx"].to_numpy(dtype=np.int64)
    if stimulus_group == "vicco":
        idx = idx - 1
    return idx


def load_cstim_feature_groups(
    model_name: str,
    *,
    cache_dir: Path | str | None = None,
    dtype: np.dtype | type | None = None,
    required_groups: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load all grouped CSTIM features for one model.

    The canonical grouped feature cache stores one ``.npz`` per model under
    ``FEATURE_CACHE_DIR/cstim``. Each array key is a stimulus group such as
    ``all_models`` or ``vicco``.
    """
    feature_dir = _feature_cache_dir(cache_dir)
    path = _require_file(feature_dir / f"{model_name}.npz", f"CSTIM feature cache for {model_name}")

    groups = list(required_groups) if required_groups is not None else None
    with np.load(path) as z:
        if groups is None:
            groups = [name for name in z.files if not name.startswith("_")]
        missing = [group for group in groups if group not in z.files]
        if missing:
            raise KeyError(f"{path} is missing feature groups: {missing}")
        out = {}
        for group in groups:
            arr = np.asarray(z[group])
            expected = constants.EXPECTED_CSTIM_IMAGE_COUNTS.get(group)
            if expected is not None and arr.shape[0] != expected:
                raise RuntimeError(
                    f"Unexpected feature row count for {model_name}/{group}: "
                    f"found {arr.shape[0]}, expected {expected} in {path}."
                )
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            out[group] = arr
    return out


def load_cstim_features(
    model_name: str,
    stimulus_group: str = "all_models",
    *,
    cache_dir: Path | str | None = None,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Load CSTIM features for one model and stimulus group."""
    return load_cstim_feature_groups(
        model_name,
        cache_dir=cache_dir,
        dtype=dtype,
        required_groups=(stimulus_group,),
    )[stimulus_group]


def load_consensus_features(
    model_name: str,
    stimulus_group: str = "all_models",
    *,
    cache_dir: Path | str | None = None,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Load the older integrated-explanation feature cache explicitly."""
    root = _as_path(cache_dir)
    if root is None:
        root = paths.consensus_data_dir() / "features"
    path = _require_file(
        root / stimulus_group / f"{model_name}.npz",
        f"consensus feature cache for {model_name}/{stimulus_group}",
    )
    with np.load(path) as z:
        if "features" not in z.files:
            raise KeyError(f"{path} is missing required key 'features'")
        arr = np.asarray(z["features"])
    expected = constants.EXPECTED_CSTIM_IMAGE_COUNTS.get(stimulus_group)
    if expected is not None and arr.shape[0] != expected:
        raise RuntimeError(
            f"Unexpected consensus feature row count for {model_name}/{stimulus_group}: "
            f"found {arr.shape[0]}, expected {expected} in {path}."
        )
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


@dataclass(frozen=True)
class CstimBrainCache:
    """Averaged CSTIM brain-response cache for one subject and ROI."""

    subject: str
    subject_dir: Path
    roi: str
    betas: np.ndarray
    stim_keys: np.ndarray
    voxel_metadata: dict[str, np.ndarray]
    stim_info: pd.DataFrame
    roi_mask: np.ndarray
    betas_roi: np.ndarray

    @property
    def available_groups(self) -> list[str]:
        return sorted(self.stim_info["group"].unique().tolist())

    @property
    def n_roi_voxels(self) -> int:
        return int(self.roi_mask.sum())

    def group_info(self, stimulus_group: str, *, sort_by_stim_idx: bool = False) -> pd.DataFrame:
        """Return stimulus metadata rows for one group."""
        return _stim_group_frame(
            self.stim_info,
            stimulus_group,
            sort_by_stim_idx=sort_by_stim_idx,
        ).reset_index(drop=True)

    def stim_keys_for_group(
        self,
        stimulus_group: str,
        *,
        sort_by_stim_idx: bool = False,
    ) -> np.ndarray:
        return self.group_info(stimulus_group, sort_by_stim_idx=sort_by_stim_idx)[
            "stim_key"
        ].to_numpy()

    def brain_indices(
        self,
        stimulus_group: str,
        *,
        sort_by_stim_idx: bool = False,
    ) -> np.ndarray:
        """Return indices into beta-cache stimulus columns."""
        keys = self.stim_keys_for_group(
            stimulus_group, sort_by_stim_idx=sort_by_stim_idx
        )
        key_to_idx = {str(key): idx for idx, key in enumerate(self.stim_keys)}
        try:
            return np.array([key_to_idx[str(key)] for key in keys], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(
                f"{self.subject}: stim_key {exc.args[0]!r} is missing from betas cache"
            ) from exc

    def feature_indices(
        self,
        stimulus_group: str,
        *,
        sort_by_stim_idx: bool = False,
    ) -> np.ndarray:
        """Return indices into CSTIM feature arrays.

        ``vicco`` image filenames are 1-based, while the feature arrays are
        zero-based. Controversial groups already use zero-based indices.
        """
        frame = self.group_info(stimulus_group, sort_by_stim_idx=sort_by_stim_idx)
        return _feature_indices_from_frame(frame, stimulus_group)

    def betas_for_group(
        self,
        stimulus_group: str,
        *,
        sort_by_stim_idx: bool = False,
    ) -> np.ndarray:
        """Return ROI betas as ``n_voxels x n_stimuli``."""
        idx = self.brain_indices(stimulus_group, sort_by_stim_idx=sort_by_stim_idx)
        return self.betas_roi[:, idx]

    def patterns(
        self,
        stimulus_group: str,
        *,
        sort_by_stim_idx: bool = False,
    ) -> np.ndarray:
        """Return ROI patterns as ``n_stimuli x n_voxels``."""
        return self.betas_for_group(
            stimulus_group, sort_by_stim_idx=sort_by_stim_idx
        ).T

    def group_brain_indices(self, *, sort_by_stim_idx: bool = False) -> dict[str, np.ndarray]:
        return {
            group: self.brain_indices(group, sort_by_stim_idx=sort_by_stim_idx)
            for group in self.available_groups
        }

    def group_feature_indices(self, *, sort_by_stim_idx: bool = False) -> dict[str, np.ndarray]:
        return {
            group: self.feature_indices(group, sort_by_stim_idx=sort_by_stim_idx)
            for group in self.available_groups
        }

    def as_legacy_group_dict(self, *, sort_by_stim_idx: bool = False) -> dict:
        """Return the common dict shape used by older analysis scripts."""
        group_indices = self.group_brain_indices(sort_by_stim_idx=sort_by_stim_idx)
        group_feature_idx = self.group_feature_indices(sort_by_stim_idx=sort_by_stim_idx)
        return {
            "betas_hlvis": self.betas_roi,
            "group_indices": group_indices,
            "group_stim_idx": group_feature_idx,
            "group_file_idx": group_feature_idx,
            "available_groups": self.available_groups,
            "hlvis_mask": self.roi_mask,
            "n_hlvis": self.n_roi_voxels,
            "n_roi_voxels": self.n_roi_voxels,
            "stim_info": self.stim_info.copy(),
        }


@dataclass(frozen=True)
class CstimRepetitionCache:
    """Per-repetition CSTIM brain-response cache for one subject and ROI."""

    subject: str
    subject_dir: Path
    roi: str
    betas_by_rep: dict[str, np.ndarray]
    voxel_metadata: dict[str, np.ndarray]
    stim_info: pd.DataFrame
    roi_mask: np.ndarray

    @property
    def available_groups(self) -> list[str]:
        return sorted(self.stim_info["group"].unique().tolist())

    @property
    def n_roi_voxels(self) -> int:
        return int(self.roi_mask.sum())

    def group_info(self, stimulus_group: str, *, sort_by_stim_idx: bool = False) -> pd.DataFrame:
        return _stim_group_frame(
            self.stim_info,
            stimulus_group,
            sort_by_stim_idx=sort_by_stim_idx,
        ).reset_index(drop=True)

    def stim_keys_for_group(
        self,
        stimulus_group: str,
        *,
        sort_by_stim_idx: bool = False,
    ) -> list[str]:
        return [
            str(key)
            for key in self.group_info(stimulus_group, sort_by_stim_idx=sort_by_stim_idx)[
                "stim_key"
            ].tolist()
        ]


def load_cstim_brain_cache(
    subject: str,
    *,
    roi: str = "hlvis",
    cache_root: Path | str | None = None,
    missing_ok: bool = False,
) -> CstimBrainCache | None:
    """Load averaged CSTIM brain responses for one subject."""
    try:
        subject_dir = _brain_subject_dir(subject, cache_root)
        betas_path = _require_file(
            subject_dir / "cstim_betas_averaged.npz",
            f"CSTIM averaged beta cache for {subject}",
        )
        voxel_path = subject_dir / "voxel_metadata.npz"
        stim_path = subject_dir / "cstim_stimulus_info.csv"
        _require_file(voxel_path, f"CSTIM voxel metadata cache for {subject}")
        _require_file(stim_path, f"CSTIM stimulus info cache for {subject}")
    except FileNotFoundError:
        if missing_ok:
            return None
        raise

    with np.load(betas_path, allow_pickle=True) as betas_npz:
        betas = np.asarray(betas_npz["betas"])
        stim_keys = np.asarray(betas_npz["stim_keys"])
    voxel_metadata = load_cstim_voxel_metadata(subject, cache_root=cache_root)
    stim_info = load_cstim_stimulus_info(subject, cache_root=cache_root)

    if roi in {"all", "full"}:
        mask_label = "all-voxel mask"
        roi_mask = np.ones(betas.shape[0], dtype=bool)
    else:
        mask_key = f"{roi}_mask"
        mask_label = mask_key
        if mask_key not in voxel_metadata:
            raise KeyError(f"{voxel_path} is missing ROI mask {mask_key!r}")
        roi_mask = np.asarray(voxel_metadata[mask_key], dtype=bool)
    if betas.shape[0] != roi_mask.shape[0]:
        raise RuntimeError(
            f"{subject}: beta row count ({betas.shape[0]}) does not match "
            f"{mask_label} length ({roi_mask.shape[0]})"
        )
    if betas.shape[1] != stim_keys.shape[0]:
        raise RuntimeError(
            f"{subject}: beta column count ({betas.shape[1]}) does not match "
            f"stim_keys length ({stim_keys.shape[0]})"
        )

    cache = CstimBrainCache(
        subject=subject,
        subject_dir=subject_dir,
        roi=roi,
        betas=betas,
        stim_keys=stim_keys,
        voxel_metadata=voxel_metadata,
        stim_info=stim_info,
        roi_mask=roi_mask,
        betas_roi=betas[roi_mask, :],
    )
    # Force index validation once at load time.
    cache.group_brain_indices()
    cache.group_feature_indices()
    return cache


def load_cstim_repetition_cache(
    subject: str,
    *,
    roi: str = "hlvis",
    cache_root: Path | str | None = None,
    missing_ok: bool = False,
) -> CstimRepetitionCache | None:
    """Load per-repetition CSTIM brain responses for one subject."""
    try:
        subject_dir = _brain_subject_dir(subject, cache_root)
        reps_path = _require_file(
            subject_dir / "cstim_betas_by_rep.npz",
            f"CSTIM repeated beta cache for {subject}",
        )
        voxel_path = subject_dir / "voxel_metadata.npz"
        stim_path = subject_dir / "cstim_stimulus_info.csv"
        _require_file(voxel_path, f"CSTIM voxel metadata cache for {subject}")
        _require_file(stim_path, f"CSTIM stimulus info cache for {subject}")
    except FileNotFoundError:
        if missing_ok:
            return None
        raise

    voxel_metadata = load_cstim_voxel_metadata(subject, cache_root=cache_root)
    if roi in {"all", "full"}:
        with np.load(reps_path, allow_pickle=True) as reps_npz:
            if not reps_npz.files:
                raise RuntimeError(f"{reps_path} contains no repeated beta arrays")
            first_key = reps_npz.files[0]
            n_voxels = np.asarray(reps_npz[first_key]).shape[0]
        mask_label = "all-voxel mask"
        roi_mask = np.ones(n_voxels, dtype=bool)
    else:
        mask_key = f"{roi}_mask"
        mask_label = mask_key
        if mask_key not in voxel_metadata:
            raise KeyError(f"{voxel_path} is missing ROI mask {mask_key!r}")
        roi_mask = np.asarray(voxel_metadata[mask_key], dtype=bool)

    with np.load(reps_path, allow_pickle=True) as reps_npz:
        betas_by_rep = {}
        for key in reps_npz.files:
            arr = np.asarray(reps_npz[key])
            if arr.shape[0] != roi_mask.shape[0]:
                raise RuntimeError(
                    f"{subject}/{key}: repeated beta row count ({arr.shape[0]}) "
                    f"does not match {mask_label} length ({roi_mask.shape[0]})"
                )
            betas_by_rep[str(key)] = arr[roi_mask]

    stim_info = load_cstim_stimulus_info(subject, cache_root=cache_root)
    missing_keys = [
        str(key)
        for key in stim_info["stim_key"].tolist()
        if str(key) not in betas_by_rep
    ]
    if missing_keys:
        raise KeyError(
            f"{subject}: repeated beta cache is missing {len(missing_keys)} stim_keys; "
            f"first missing: {missing_keys[:5]}"
        )

    return CstimRepetitionCache(
        subject=subject,
        subject_dir=subject_dir,
        roi=roi,
        betas_by_rep=betas_by_rep,
        voxel_metadata=voxel_metadata,
        stim_info=stim_info,
        roi_mask=roi_mask,
    )


__all__ = [
    "CstimBrainCache",
    "CstimRepetitionCache",
    "cstim_brain_cache_exists",
    "load_consensus_features",
    "load_cstim_brain_cache",
    "load_cstim_feature_groups",
    "load_cstim_feature_indices",
    "load_cstim_features",
    "load_cstim_group_info",
    "load_cstim_repetition_cache",
    "load_cstim_stimulus_info",
    "load_cstim_voxel_metadata",
]
