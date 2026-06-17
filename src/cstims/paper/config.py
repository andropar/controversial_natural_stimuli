"""Compatibility configuration for paper scripts.

The source of truth for paths, model sets, and subject constants now lives in
``cstims.paths``, ``cstims.models``, and ``cstims.subjects``.  This module keeps
the historical uppercase names used by paper scripts.
"""

from __future__ import annotations

import numpy as np

from cstims import paths
from cstims.models import MODEL_DISPLAY_NAMES, MODEL_SETS, MODELS_EXCL_VICREG
from cstims.subjects import CSTIM_SESSION_CANDIDATES, INPUT_SOURCE, SUBJECTS


SHARE_ROOT = paths.find_share_root()
PROJECT_ROOT = paths.project_root()
PAPER_ROOT = paths.paper_root()

EXTERNAL_DATA_ROOT = paths.external_data_root()
DEEPVISION_ROOT = paths.deepvision_fmri_root()
CSTIM_HDF5_ROOT = paths.cstim_hdf5_root()
MODEL_LIST_CSV = paths.model_list_csv()

SHARED_ENCODING_ROOT = paths.shared_encoding_root()
UNIQUE_ENCODING_ROOT = paths.unique_encoding_root()
UNIQUE_ENCODING_DIRS = paths.unique_encoding_dirs()
ENCODING_MODE = "unique"

SELECTION_OUTPUT_ROOT = paths.selected_stimuli_root()
SELECTION_PAYLOAD = paths.selected_stimuli_payload("all_models")
EVAL_DATA_DIR = paths.selection_evaluation_results_dir()

BRAIN_DATA_DIR = paths.brain_data_dir()
RSA_DATA_DIR = paths.rsa_data_dir()
RELIABILITY_DATA_DIR = paths.reliability_data_dir()
STATS_DATA_DIR = paths.stats_data_dir()
ROBUSTNESS_DATA_DIR = paths.robustness_data_dir()
SIM_DATA_DIR = paths.simulation_data_dir()
UMC_DATA_DIR = paths.counterfactual_baseline_data_dir()
OOD_DATA_DIR = paths.ood_data_dir()
CONSENSUS_DATA_DIR = paths.consensus_data_dir()

FEATURE_CACHE_DIR = paths.feature_cache_dir()
CSTIM_FEATURE_CACHE = paths.cstim_feature_cache_dir()
DV_FEATURE_CACHE = paths.deepvision_feature_cache_dir()
VICCO_FEATURE_CACHE = paths.vicco_feature_cache_dir()
VOXEL_CACHE_DIR = paths.voxel_cache_dir()

RIDGE_ALPHAS = np.logspace(-2, 6, 50)


def get_encoding_root(subject: str | None = None):
    return paths.get_encoding_root(subject)


def get_eval_pipeline_dir(model_set: str):
    return EVAL_DATA_DIR / model_set


def get_brain_input_dir(subject: str):
    return paths.get_brain_input_dir(subject)


def get_subject_data_dir(subject: str):
    return paths.get_subject_data_dir(subject)

