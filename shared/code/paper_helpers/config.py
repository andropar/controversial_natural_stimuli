"""
Single source of truth for all paths, constants, and configuration.

All analysis and figure scripts import from this module instead of
hardcoding paths or using environment variable hacks.

Encoding models: always uses unique (per-subject) encodings.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict

# =============================================================================
# Root Paths
# =============================================================================

def _find_share_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / "00_stimulus_selection").exists()
            and (candidate / "01_brain_model_alignment").exists()
        ):
            return candidate
    return start.parents[3]


SHARE_ROOT = Path(
    os.environ.get("CSTIMS_SHARE_ROOT", _find_share_root(Path(__file__).resolve()))
)
PROJECT_ROOT = Path(
    os.environ.get("CSTIMS_PROJECT_ROOT", SHARE_ROOT)
)
PAPER_ROOT = Path(
    os.environ.get("CSTIMS_PAPER_ROOT", SHARE_ROOT)
)

# =============================================================================
# Machine-Dependent Paths
# =============================================================================

EXTERNAL_DATA_ROOT = Path(
    os.environ.get("CSTIMS_EXTERNAL_DATA_ROOT", SHARE_ROOT / "external_data")
)
DEEPVISION_ROOT = Path(
    os.environ.get("CSTIMS_DEEPVISION_FMRI_ROOT", EXTERNAL_DATA_ROOT / "deepvision_fmri")
)
CSTIM_HDF5_ROOT = Path(
    os.environ.get("CSTIMS_CSTIM_HDF5_ROOT", EXTERNAL_DATA_ROOT / "final_cstims_hdf5_files")
)
MODEL_LIST_CSV = Path(
    os.environ.get(
        "CSTIMS_MODEL_LIST_CSV",
        SHARE_ROOT / "00_stimulus_selection" / "resources" / "model_list.csv",
    )
)

# =============================================================================
# Encoding Models
# =============================================================================

# Shared encoding models (trained on DeepVision ~515 shared images)
SHARED_ENCODING_ROOT = (
    SHARE_ROOT
    / "01_brain_model_alignment"
    / "results"
    / "encoding_models"
    / "shared_subject_encoding_models"
    / "encoding_20251222_141301"
)

# Unique encoding models (trained on per-subject unique images)
UNIQUE_ENCODING_ROOT = (
    SHARE_ROOT
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

# Encoding mode is always "unique" (per-subject encoding models).
# Kept as a constant for any code that still reads it.
ENCODING_MODE = "unique"


def get_encoding_root(subject: str = None) -> Path:
    """Get encoding root for a subject (always unique per-subject encodings)."""
    if subject in UNIQUE_ENCODING_DIRS:
        return UNIQUE_ENCODING_DIRS[subject]
    return SHARED_ENCODING_ROOT  # fallback for unknown subjects

# =============================================================================
# Selection Payload + Eval Pipeline
# =============================================================================

# Original selection output (frozen - never rerun)
SELECTION_OUTPUT_ROOT = SHARE_ROOT / "00_stimulus_selection" / "results" / "selected_stimuli"
SELECTION_PAYLOAD = SELECTION_OUTPUT_ROOT / "all_models" / "selected_stimuli_data.pkl"

# Eval pipeline results
EVAL_DATA_DIR = (
    SHARE_ROOT / "00_stimulus_selection" / "decision_checks" / "selection_evaluation" / "results"
)


def get_eval_pipeline_dir(model_set: str) -> Path:
    """Get eval pipeline data directory for a model set."""
    return EVAL_DATA_DIR / model_set

# =============================================================================
# Result Directories
# =============================================================================

# Per-section result directories. Names ending in DATA_DIR are retained for
# compatibility with older scripts, but they point at the public `results/`
# layout.
BRAIN_DATA_DIR = SHARE_ROOT / "01_brain_model_alignment" / "cache_or_heavy" / "brain_data_cache" / "data"
RSA_DATA_DIR = SHARE_ROOT / "01_brain_model_alignment" / "results" / "rsa_scores"
RELIABILITY_DATA_DIR = SHARE_ROOT / "02_alignment_reliability" / "results"
STATS_DATA_DIR = SHARE_ROOT / "03_alignment_inference" / "results"
ROBUSTNESS_DATA_DIR = SHARE_ROOT / "04_alignment_robustness" / "results"
SIM_DATA_DIR = SHARE_ROOT / "05_controls_and_supplementary" / "simulation_validation" / "results"
UMC_DATA_DIR = SHARE_ROOT / "05_controls_and_supplementary" / "counterfactual_baselines" / "results"
OOD_DATA_DIR = (
    SHARE_ROOT / "05_controls_and_supplementary" / "low_level_and_ood" / "ood_controls" / "results"
)
CONSENSUS_DATA_DIR = SHARE_ROOT / "05_controls_and_supplementary" / "integrated_explanation" / "results"

# Feature cache (shared across sections)
_SHARED_FEATURE_CACHE_DIR = (
    SHARE_ROOT / "shared" / "cache_or_heavy" / "cstim_paper_feature_cache" / "feature_cache"
)
_LEGACY_FEATURE_CACHE_DIR = PAPER_ROOT / "cache" / "feature_cache"
FEATURE_CACHE_DIR = Path(
    os.environ.get(
        "CSTIMS_FEATURE_CACHE_DIR",
        _SHARED_FEATURE_CACHE_DIR if _SHARED_FEATURE_CACHE_DIR.exists() else _LEGACY_FEATURE_CACHE_DIR,
    )
)
CSTIM_FEATURE_CACHE = FEATURE_CACHE_DIR / "cstim"
DV_FEATURE_CACHE = FEATURE_CACHE_DIR / "deepvision"
VICCO_FEATURE_CACHE = FEATURE_CACHE_DIR / "vicco"

# Voxel cache (heavy local cache, not part of the public results payload)
VOXEL_CACHE_DIR = SHARE_ROOT / "01_brain_model_alignment" / "cache_or_heavy" / "brain_data" / "voxel_sets"


def get_brain_input_dir(subject: str) -> Path:
    """Get the per-subject directory for brain betas, voxel metadata, stimulus info."""
    return BRAIN_DATA_DIR / subject


def get_subject_data_dir(subject: str) -> Path:
    """Get the per-subject brain data directory."""
    return BRAIN_DATA_DIR / subject

# =============================================================================
# Subjects
# =============================================================================

# Subjects with at least one cstim session
SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]

# All possible cstim sessions
CSTIM_SESSION_CANDIDATES = ["ses-32", "ses-33", "ses-34"]

INPUT_SOURCE = "tedana"

# =============================================================================
# Model Sets
# =============================================================================

MODEL_SETS: Dict[str, list] = {
    "architecture": [
        "torchvision_vgg16_imagenet1k_v1",
        "torchvision_resnet50_imagenet1k_v1",
        "torchvision_convnext_base_imagenet1k_v1",
        "torchvision_vit_l_16_imagenet1k_v1",
        "cornet_s",
    ],
    "training_objective": [
        "vissl_resnet50_supervised",
        "vissl_resnet50_barlowtwins",
        "vissl_resnet50_mocov2",
        "vicreg_resnet50",
        "robustness_imagenet_l2_eps3",
    ],
    "sota": [
        "slip_vit_l_slip",
        "slip_vit_l_simclr",
        "timm_vit_large_patch14_clip_224_laion2b",
        "dinov2_vitl14",
        "openclip_vit_so400m_14_siglip_webli",
        "torchvision_convnext_base_imagenet1k_v1",
    ],
    "dataset": [
        "openclip_vit_l_14_quickgelu_metaclip_400m",
        "openclip_vit_l_14_quickgelu_metaclip_fullcc",
        "timm_vit_large_patch14_clip_224_dfn2b",
        "timm_vit_large_patch14_clip_quickgelu_224_openai",
        "openclip_vit_l_14_laion400m_e31",
    ],
}

# all_models = union of all sets
MODEL_SETS["all_models"] = sorted(set(
    model for models in MODEL_SETS.values() for model in models
))

# Exclude VICReg from mRSA analyses (extreme leverage, see Appendix D)
MODELS_EXCL_VICREG = [m for m in MODEL_SETS["all_models"] if m != "vicreg_resnet50"]

# Model display names for plotting
MODEL_DISPLAY_NAMES = {
    "torchvision_vgg16_imagenet1k_v1": "VGG-16",
    "torchvision_resnet50_imagenet1k_v1": "ResNet-50",
    "torchvision_convnext_base_imagenet1k_v1": "ConvNeXt-B",
    "torchvision_vit_l_16_imagenet1k_v1": "ViT-L/16",
    "cornet_s": "CORnet-S",
    "vissl_resnet50_supervised": "Supervised",
    "vissl_resnet50_barlowtwins": "BarlowTwins",
    "vissl_resnet50_mocov2": "MoCoV2",
    "vicreg_resnet50": "VICReg",
    "robustness_imagenet_l2_eps3": "Robust-L2",
    "slip_vit_l_slip": "SLIP",
    "slip_vit_l_simclr": "SimCLR-ViT",
    "timm_vit_large_patch14_clip_224_laion2b": "CLIP-L2B",
    "dinov2_vitl14": "DINOv2",
    "openclip_vit_so400m_14_siglip_webli": "SigLIP",
    "openclip_vit_l_14_quickgelu_metaclip_400m": "MetaCLIP-400M",
    "openclip_vit_l_14_quickgelu_metaclip_fullcc": "MetaCLIP-Full",
    "timm_vit_large_patch14_clip_224_dfn2b": "DFN-2B",
    "timm_vit_large_patch14_clip_quickgelu_224_openai": "CLIP-OpenAI",
    "openclip_vit_l_14_laion400m_e31": "CLIP-L400M",
    "torchvision_alexnet_imagenet1k_v1": "AlexNet",
}

# =============================================================================
# Ridge Regression
# =============================================================================

RIDGE_ALPHAS = np.logspace(-2, 6, 50)
