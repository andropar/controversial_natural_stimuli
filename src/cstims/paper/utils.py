"""
Shared utilities for the cstim paper pipeline.

Path references are resolved through ``cstims.paths``; stable rosters and
analysis constants live in ``cstims.constants``.
"""

import numpy as np

from cstims import paths


# =============================================================================
# Encoding Model Loading (ONE canonical definition)
# =============================================================================

def load_encoding_model(model_name: str, subject: str) -> dict:
    """Load pre-fitted encoding model weights.

    Uses paths.encoding_model_dir(subject, model_name) to resolve the
    per-subject encoding directory.
    """
    folder = paths.encoding_model_dir(subject, model_name)
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
