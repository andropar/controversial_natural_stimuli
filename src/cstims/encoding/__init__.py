"""Encoding model utilities for mapping features to brain responses."""

from .fitting import (
    compute_image_r,
    compute_versa,
    compute_voxel_r,
    create_encoding_model,
    fit_voxelwise_ridgecv,
    fit_voxelwise_ridgecv_fast,
    pearsonr_metric,
    refit_with_chosen_alphas,
    refit_with_chosen_alphas_fast,
)
from .linear import (
    EncodingParamsByEncoding,
    EncodingParamsByModel,
    encode_batch_for_all_encodings,
    load_encoding_params_by_encoding,
)
from .model import LinearEncodingModel

__all__ = [
    # Model class
    "LinearEncodingModel",
    # Fitting utilities
    "fit_voxelwise_ridgecv",
    "fit_voxelwise_ridgecv_fast",
    "refit_with_chosen_alphas",
    "create_encoding_model",
    # Metrics
    "pearsonr_metric",
    "compute_versa",
    "compute_voxel_r",
    "compute_image_r",
    # Legacy/batch inference
    "load_encoding_params_by_encoding",
    "encode_batch_for_all_encodings",
    "EncodingParamsByEncoding",
    "EncodingParamsByModel",
]
