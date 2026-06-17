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


def project_root() -> Path:
    override = os.environ.get("CSTIMS_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return find_share_root()


def paper_root() -> Path:
    override = os.environ.get("CSTIMS_PAPER_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return find_share_root()


def resources_dir() -> Path:
    override = os.environ.get("CSTIMS_RESOURCES_DIR")
    if override:
        return Path(override).expanduser().resolve()

    root = find_share_root()
    for candidate in (
        root / "00_stimulus_selection/resources",
    ):
        if candidate.exists():
            return candidate
    return root / "00_stimulus_selection/resources"


def model_list_csv() -> Path:
    override = os.environ.get("CSTIMS_MODEL_LIST_CSV")
    if override:
        return Path(override).expanduser().resolve()
    return resources_dir() / "model_list.csv"


def deepvision_fmri_root() -> Path:
    override = os.environ.get("CSTIMS_DEEPVISION_FMRI_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return external_data_root() / "deepvision_fmri"


def external_data_root() -> Path:
    override = os.environ.get("CSTIMS_EXTERNAL_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return find_share_root() / "external_data"


def cstim_hdf5_root() -> Path:
    override = os.environ.get("CSTIMS_CSTIM_HDF5_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return external_data_root() / "final_cstims_hdf5_files"


def shared_encoding_root() -> Path:
    return (
        find_share_root()
        / "01_brain_model_alignment"
        / "results"
        / "encoding_models"
        / "shared_subject_encoding_models"
        / "encoding_20251222_141301"
    )


def unique_encoding_root() -> Path:
    return (
        find_share_root()
        / "01_brain_model_alignment"
        / "results"
        / "encoding_models"
        / "subject_unique_encoding_models"
        / "runs"
    )


def unique_encoding_dirs() -> dict[str, Path]:
    root = unique_encoding_root()
    return {
        "sub-01": root / "20260317_170621",
        "sub-03": root / "20260319_152751",
        "sub-05": root / "20260317_170621",
        "sub-06": root / "20260319_152752",
        "sub-07": root / "20260317_170621",
    }


def get_encoding_root(subject: str | None = None) -> Path:
    dirs = unique_encoding_dirs()
    if subject in dirs:
        return dirs[subject]
    return shared_encoding_root()


def selected_stimuli_root() -> Path:
    return find_share_root() / "00_stimulus_selection" / "results" / "selected_stimuli"


def selected_stimuli_payload(model_set: str = "all_models") -> Path:
    return selected_stimuli_root() / model_set / "selected_stimuli_data.pkl"


def selection_evaluation_results_dir() -> Path:
    return find_share_root() / "00_stimulus_selection" / "selection_evaluation" / "results"


def brain_data_dir() -> Path:
    return (
        find_share_root()
        / "01_brain_model_alignment"
        / "cache_or_heavy"
        / "cstim_brain_response_cache"
        / "data"
    )


def rsa_data_dir() -> Path:
    return find_share_root() / "01_brain_model_alignment" / "results" / "rsa_scores"


def reliability_data_dir() -> Path:
    return find_share_root() / "02_alignment_reliability" / "results"


def stats_data_dir() -> Path:
    return find_share_root() / "03_alignment_inference" / "results"


def robustness_data_dir() -> Path:
    return find_share_root() / "04_alignment_robustness" / "results"


def simulation_data_dir() -> Path:
    return (
        find_share_root()
        / "05_controls_and_supplementary"
        / "simulation_validation"
        / "results"
    )


def counterfactual_baseline_data_dir() -> Path:
    return (
        find_share_root()
        / "05_controls_and_supplementary"
        / "counterfactual_baselines"
        / "results"
    )


def ood_data_dir() -> Path:
    return (
        find_share_root()
        / "05_controls_and_supplementary"
        / "low_level_and_ood"
        / "ood_controls"
        / "results"
    )


def consensus_data_dir() -> Path:
    return (
        find_share_root()
        / "05_controls_and_supplementary"
        / "integrated_explanation"
        / "results"
    )


def feature_cache_dir() -> Path:
    shared = (
        find_share_root()
        / "shared"
        / "cache_or_heavy"
        / "cstim_paper_feature_cache"
        / "feature_cache"
    )
    legacy = paper_root() / "cache" / "feature_cache"
    override = os.environ.get("CSTIMS_FEATURE_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return shared if shared.exists() else legacy


def cstim_feature_cache_dir() -> Path:
    return feature_cache_dir() / "cstim"


def deepvision_feature_cache_dir() -> Path:
    return feature_cache_dir() / "deepvision"


def vicco_feature_cache_dir() -> Path:
    return feature_cache_dir() / "vicco"


def voxel_cache_dir() -> Path:
    return (
        find_share_root()
        / "01_brain_model_alignment"
        / "cache_or_heavy"
        / "deepvision_benchmark_cache"
        / "voxel_sets"
    )


def get_brain_input_dir(subject: str) -> Path:
    return brain_data_dir() / subject


def get_subject_data_dir(subject: str) -> Path:
    return brain_data_dir() / subject
