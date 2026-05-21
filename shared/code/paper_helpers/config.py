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
        SHARE_ROOT / "00_stimulus_selection" / "inputs" / "resources" / "model_list.csv",
    )
)

# =============================================================================
# Encoding Models
# =============================================================================

# Shared encoding models (trained on DeepVision ~515 shared images)
SHARED_ENCODING_ROOT = (
    PROJECT_ROOT / "experiments" / "encoding_fitting" / "results" / "encoding_20251222_141301"
)

# Unique encoding models (trained on per-subject unique images)
UNIQUE_ENCODING_DIRS = {
    "sub-01": PROJECT_ROOT / "outputs" / "deepvision_encoding_models" / "runs" / "20260317_170621",
    "sub-03": PROJECT_ROOT / "outputs" / "deepvision_encoding_models" / "runs" / "20260319_152751",
    "sub-05": PROJECT_ROOT / "outputs" / "deepvision_encoding_models" / "runs" / "20260317_170621",
    "sub-06": PROJECT_ROOT / "outputs" / "deepvision_encoding_models" / "runs" / "20260319_152752",
    "sub-07": PROJECT_ROOT / "outputs" / "deepvision_encoding_models" / "runs" / "20260317_170621",
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

# Original selection output (frozen — never rerun)
SELECTION_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_cstims_v2_full"
SELECTION_PAYLOAD = (
    SELECTION_OUTPUT_ROOT / "all_models" / "method-raw_plus_all_encodings"
    / "20251222_175721" / "selected_stimuli_data.pkl"
)

# Eval pipeline results (copied into paper directory)
EVAL_DATA_DIR = PAPER_ROOT / "00_selection_evaluation" / "data"


def get_eval_pipeline_dir(model_set: str) -> Path:
    """Get eval pipeline data directory for a model set."""
    return EVAL_DATA_DIR / model_set

# =============================================================================
# Data Directories
# =============================================================================

# Per-section data directories
BRAIN_DATA_DIR = PAPER_ROOT / "01_brain_data" / "data"
RSA_DATA_DIR = PAPER_ROOT / "02_rsa_scores" / "data"
STATS_DATA_DIR = PAPER_ROOT / "03_statistics" / "data"
SIM_DATA_DIR = PAPER_ROOT / "04_simulation" / "data"
UMC_DATA_DIR = PAPER_ROOT / "05_unique_contribution" / "data"
OOD_DATA_DIR = PAPER_ROOT / "06_ood" / "data"
CONSENSUS_DATA_DIR = PAPER_ROOT / "07_consensus" / "data"

# Feature cache (shared across sections)
FEATURE_CACHE_DIR = PAPER_ROOT / "cache" / "feature_cache"
CSTIM_FEATURE_CACHE = FEATURE_CACHE_DIR / "cstim"
DV_FEATURE_CACHE = FEATURE_CACHE_DIR / "deepvision"
VICCO_FEATURE_CACHE = FEATURE_CACHE_DIR / "vicco"

# Voxel cache (existing, at project root)
VOXEL_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "voxel_sets"


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
