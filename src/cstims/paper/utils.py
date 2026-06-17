"""
Shared utilities for the cstim paper pipeline.

All path references use ``cstims.paper.config`` for backwards-compatible names.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats

from cstims.paper import config
from cstims.paper.config import get_subject_data_dir  # re-export for callers

# =============================================================================
# Session & Subject Utilities
# =============================================================================

def detect_available_sessions(subject: str) -> List[str]:
    """Detect which cstim sessions exist on disk for a subject."""
    glms_root = (
        config.DEEPVISION_ROOT / "derivatives/functional/1sTR_1pt5mm/glmsingle"
        / config.INPUT_SOURCE / subject
    )
    sessions = []
    for ses in config.CSTIM_SESSION_CANDIDATES:
        h5_path = glms_root / ses / "TYPED_FITHRF_GLMDENOISE_RR.hdf5"
        if h5_path.exists():
            sessions.append(ses)
    return sessions


def parse_subject_arg(subject_arg: str) -> List[str]:
    """Parse --subject argument: 'all' returns SUBJECTS, otherwise validates and returns [subject]."""
    if subject_arg == "all":
        return config.SUBJECTS
    if subject_arg not in config.SUBJECTS:
        available = detect_available_sessions(subject_arg)
        if not available:
            raise ValueError(f"No cstim sessions found for {subject_arg}")
    return [subject_arg]


# =============================================================================
# Model to Encoding Folder Mapping
# =============================================================================

def load_model_layer_mapping() -> Dict[str, str]:
    """Load mapping from model name to layer name from model_list.csv."""
    df = pd.read_csv(config.MODEL_LIST_CSV)
    mapping = {}
    for _, row in df.iterrows():
        layer = row["layer"].replace(".", "_")
        mapping[row["model"]] = f"layer{layer}"
    return mapping


def get_encoding_folder(subject: str, model: str) -> Path:
    """Get path to encoding model folder for a subject/model pair."""
    layer_mapping = load_model_layer_mapping()
    layer = layer_mapping[model]
    folder_name = f"{subject}_{model}.{layer}"
    return config.get_encoding_root(subject) / folder_name


# =============================================================================
# Label Correction
# =============================================================================

# NOTE: Trial labels for dataset and architecture are swapped in the data!
# See DATA_INVENTORY.md: "dataset_i72 should be architecture_i72, and vice versa"

def correct_stimulus_label(label: str) -> str:
    """Correct the swapped dataset/architecture labels in trial_info."""
    if label.startswith("shared_4rep_LAION_controversial_dataset_"):
        return label.replace("_dataset_", "_architecture_")
    elif label.startswith("shared_4rep_LAION_controversial_architecture_"):
        return label.replace("_architecture_", "_dataset_")
    return label


def parse_stimulus_label(label: str) -> Tuple[str, Optional[int]]:
    """Parse a stimulus label into (group, index)."""
    if label == "blank":
        return "blank", None

    label = correct_stimulus_label(label)

    if "vicco" in label:
        idx = int(label.split("_")[-1].replace(".jpg", ""))
        return "vicco", idx
    elif "controversial" in label:
        label_clean = label.replace(".jpg", "")
        idx_part = label_clean.split("_i")[-1]
        idx = int(idx_part)
        after_controversial = label_clean.split("controversial_")[1]
        model_set = after_controversial.rsplit("_i", 1)[0]
        if model_set == "all":
            model_set = "all_models"
        elif model_set == "training":
            model_set = "training_objective"
        return model_set, idx
    else:
        raise ValueError(f"Unknown label format: {label}")


# =============================================================================
# RDM Utilities
# =============================================================================

def compute_rdm_correlation(features: np.ndarray) -> np.ndarray:
    """Compute RDM using correlation distance (1 - Pearson r)."""
    corr = np.corrcoef(features)
    rdm = 1 - corr
    np.fill_diagonal(rdm, 0)
    return rdm


def rdm_to_vector(rdm: np.ndarray) -> np.ndarray:
    """Extract upper triangular of RDM as vector."""
    n = rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    return rdm[idx]


def compute_rsa_score(rdm1: np.ndarray, rdm2: np.ndarray, method: str = "spearman") -> float:
    """Compute RSA score (correlation between RDM vectors)."""
    vec1 = rdm_to_vector(rdm1)
    vec2 = rdm_to_vector(rdm2)
    if method == "spearman":
        r, _ = stats.spearmanr(vec1, vec2)
    elif method == "pearson":
        r, _ = stats.pearsonr(vec1, vec2)
    else:
        raise ValueError(f"Unknown method: {method}")
    return r


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_summary_stats(scores: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a set of scores (one per model)."""
    scores = np.asarray(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    score_range = np.max(scores) - np.min(scores)
    norm_var = (std / np.abs(mean)) ** 2 if mean != 0 else np.nan
    return {"range": score_range, "norm_variance": norm_var, "mean": mean, "std": std}


# =============================================================================
# Bootstrap Utilities
# =============================================================================

def bootstrap_sample_indices(
    n_total: int, n_sample: int, n_bootstrap: int = 10, seed: int = 0
) -> List[np.ndarray]:
    """Generate bootstrap sample indices."""
    samples = []
    for i in range(n_bootstrap):
        rng = np.random.default_rng(seed + i)
        idx = rng.choice(n_total, size=n_sample, replace=False)
        samples.append(np.sort(idx))
    return samples


# =============================================================================
# Encoding Model Loading (ONE canonical definition)
# =============================================================================

def load_encoding_model(model_name: str, subject: str) -> dict:
    """Load pre-fitted encoding model weights.

    Uses config.get_encoding_root(subject) to resolve the per-subject
    encoding directory.
    """
    layer_mapping = load_model_layer_mapping()
    layer = layer_mapping[model_name]
    folder_name = f"{subject}_{model_name}.{layer}"
    folder = config.get_encoding_root(subject) / folder_name
    npz_path = folder / "encoding_model.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Encoding model not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return {
        "weights": data["weights"],
        "intercept": data["intercept"],
        "feature_mean": data["feature_mean"],
        "feature_scale": data["feature_scale"],
        "roi_hlvis": data["roi_hlvis"],
    }


def predict_voxel_responses(features: np.ndarray, encoding: dict) -> np.ndarray:
    """Predict voxel responses from features using encoding model."""
    features_scaled = features.copy().astype(np.float64)
    if encoding["feature_mean"] is not None and np.any(encoding["feature_mean"] != 0):
        features_scaled = features_scaled - encoding["feature_mean"]
    if encoding["feature_scale"] is not None and np.any(encoding["feature_scale"] != 1):
        features_scaled = features_scaled / (encoding["feature_scale"] + 1e-8)
    return features_scaled @ encoding["weights"] + encoding["intercept"]


# =============================================================================
# Feature & Brain Data Loading
# =============================================================================

def load_cached_features(model_name: str, model_set: str = "all_models") -> np.ndarray:
    """Load cached features for a model from the unified feature cache."""
    path = config.CSTIM_FEATURE_CACHE / f"{model_name}.npz"
    if path.exists():
        data = np.load(path)
        if model_set in data:
            return data[model_set]
        elif "features" in data:
            return data["features"]
    # Fallback: try consensus data cache location
    old_path = (
        config.CONSENSUS_DATA_DIR / "features" / model_set / f"{model_name}.npz"
    )
    if old_path.exists():
        return np.load(old_path)["features"]
    raise FileNotFoundError(
        f"Features not found for {model_name} (set={model_set}). "
        f"Checked: {path}, {old_path}"
    )


def load_subject_brain_data(
    subject: str,
    stimulus_group: str = "all_models",
    n_stimuli: int = None,
    bootstrap_idx: int = 0,
    bootstrap_seed: int = 0,
) -> Optional[dict]:
    """Load brain data and stimulus indices for a subject."""
    data_dir = config.get_subject_data_dir(subject)
    betas_data = np.load(data_dir / "cstim_betas_averaged.npz", allow_pickle=True)
    voxel_data = np.load(data_dir / "voxel_metadata.npz", allow_pickle=True)
    stim_info = pd.read_csv(data_dir / "cstim_stimulus_info.csv")

    hlvis_mask = voxel_data["hlvis_mask"]
    betas_hlvis = betas_data["betas"][hlvis_mask, :]
    stim_keys = betas_data["stim_keys"]
    stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}

    mask = stim_info["group"] == stimulus_group
    if mask.sum() == 0:
        return None

    keys = stim_info.loc[mask, "stim_key"].values
    brain_idx = np.array([stim_key_to_idx[k] for k in keys])
    file_idx = stim_info.loc[mask, "stim_idx"].values.astype(int)
    if stimulus_group == "vicco":
        file_idx = file_idx - 1

    stim_subset = stim_info.loc[mask].copy().reset_index(drop=True)
    if n_stimuli is not None and len(file_idx) > n_stimuli:
        rng = np.random.default_rng(bootstrap_seed + bootstrap_idx)
        subset = np.sort(rng.choice(len(file_idx), size=n_stimuli, replace=False))
        brain_idx = brain_idx[subset]
        file_idx = file_idx[subset]
        stim_subset = stim_subset.iloc[subset].reset_index(drop=True)

    return {
        "betas_hlvis": betas_hlvis,
        "brain_idx": brain_idx,
        "file_idx": file_idx,
        "stim_info": stim_subset,
        "hlvis_mask": hlvis_mask,
        "stimulus_group": stimulus_group,
    }


# =============================================================================
# Stimulus-Level Cross-Validation
# =============================================================================

def stimulus_cv_splits(n_stim: int, n_splits: int = 10, random_state: int = 42):
    """Generate train/test pair indices using stimulus-level cross-validation.

    Pairs sharing a stimulus with the held-out set are excluded entirely,
    avoiding information leakage through shared stimuli.
    """
    rng = np.random.default_rng(random_state)
    stim_indices = rng.permutation(n_stim)

    pair_stim = []
    for i in range(n_stim):
        for j in range(i + 1, n_stim):
            pair_stim.append((i, j))
    pair_stim = np.array(pair_stim)

    splits = []
    fold_size = n_stim // n_splits
    for k in range(n_splits):
        start = k * fold_size
        end = start + fold_size if k < n_splits - 1 else n_stim
        test_stim = set(stim_indices[start:end].tolist())

        train_mask = np.array([
            s[0] not in test_stim and s[1] not in test_stim for s in pair_stim
        ])
        test_mask = np.array([
            s[0] in test_stim and s[1] in test_stim for s in pair_stim
        ])
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return splits
