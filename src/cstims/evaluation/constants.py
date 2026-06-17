"""
Constants for evaluation module.
"""

import numpy as np

DEFAULT_METRICS = ["euclidean", "cosine", "correlation"]
DEFAULT_CORR_TYPES = ["spearman", "pearson"]
DEFAULT_N_RANDOM_SUBSETS = 100
DEFAULT_N_NOISE_SAMPLES = 100
DEFAULT_N_BOOTSTRAP = 500
DEFAULT_N_NOISE_LEVELS = 20
DEFAULT_NOISE_LEVEL_RANGE = (-1, 2)
DEFAULT_SEED = 42
MODEL_SET_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]

DEFAULT_DISCRIMINABILITY_METRICS = [
    "score_deltas",
    "d_primes",
    "parametric_pairwise_error_probs",
    "non_parametric_multiclass_error_prob",
]


def get_default_noise_level_multipliers() -> np.ndarray:
    """Get default noise level multipliers."""
    multipliers = np.logspace(*DEFAULT_NOISE_LEVEL_RANGE, DEFAULT_N_NOISE_LEVELS)
    # Ensure commonly useful reference points are always included
    special_levels = np.array([0.1, 1.0, 3.0, 5.0, 10.0])
    multipliers = np.sort(np.unique(np.concatenate([multipliers, special_levels])))
    return multipliers
