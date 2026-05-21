from __future__ import annotations

import os
from pathlib import Path


def find_share_root(start: Path | None = None) -> Path:
    """Find the cstims_share root from an installed or in-place package."""
    start = (start or Path(__file__)).resolve()
    for path in (start, *start.parents):
        if (
            (path / "pyproject.toml").exists()
            and (path / "00_stimulus_selection").exists()
            and (path / "01_brain_model_alignment").exists()
        ):
            return path
    return Path.cwd().resolve()


def resources_dir() -> Path:
    override = os.environ.get("CSTIMS_RESOURCES_DIR")
    if override:
        return Path(override).expanduser().resolve()

    root = find_share_root()
    for candidate in (
        root / "00_stimulus_selection/inputs/resources",
        root / "data/resources",
        root / "src/data/resources",
    ):
        if candidate.exists():
            return candidate
    return root / "00_stimulus_selection/inputs/resources"


def model_list_csv() -> Path:
    override = os.environ.get("CSTIMS_MODEL_LIST_CSV")
    if override:
        return Path(override).expanduser().resolve()
    return resources_dir() / "model_list.csv"


def deepvision_fmri_root() -> Path:
    override = os.environ.get("CSTIMS_DEEPVISION_FMRI_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return find_share_root() / "external_data/deepvision_fmri"
