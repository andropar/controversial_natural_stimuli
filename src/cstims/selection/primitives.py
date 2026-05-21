"""
Core mathematical primitives for stimulus selection.

This module contains foundational functions for:
- Computing pairwise distances between features
- Computing correlation matrices
- Aggregating utilities across and within models
"""
from __future__ import annotations

from typing import Dict

import torch

from ..timing import timed

# ==============================================================================
# Distance and Correlation Computation
# ==============================================================================

# Compile optimizations for performance-critical functions
_compute_pairwise_distances_cosine_corr = None
_compute_correlation_matrix_compiled = None


def _compute_pairwise_distances_impl(
    cand: torch.Tensor, curr: torch.Tensor, metric: str
) -> torch.Tensor:
    """Implementation of pairwise distances computation."""
    if metric == "euclidean":
        # GEMM-based Euclidean distance (leverages TF32 on Ampere+)
        # d2 = ||a||^2 + ||b||^2 - 2 a b^T
        a2 = torch.sum(cand * cand, dim=1, keepdim=True)  # [N,1]
        b2 = torch.sum(curr * curr, dim=1, keepdim=True).t()  # [1,M]
        prod = cand @ curr.t()  # [N,M]
        d2 = a2 + b2 - 2.0 * prod
        # Use clamp_min (non-inplace) for compatibility with torch.compile
        d2 = torch.clamp_min(d2, 0.0)
        return torch.sqrt(d2)
    elif metric == "cosine":
        cand_norm = cand / (torch.norm(cand, p=2, dim=1, keepdim=True) + 1e-9)
        curr_norm = curr / (torch.norm(curr, p=2, dim=1, keepdim=True) + 1e-9)
        sim = cand_norm @ curr_norm.t()
        sim = torch.clamp(
            sim, -1.0, 1.0
        )  # Use non-inplace version for compiled functions
        return 1.0 - sim
    elif metric == "correlation":
        cand_mean = cand.mean(dim=1, keepdim=True)
        cand_std = cand.std(dim=1, keepdim=True)
        curr_mean = curr.mean(dim=1, keepdim=True)
        curr_std = curr.std(dim=1, keepdim=True)
        cand_z = (cand - cand_mean) / (cand_std + 1e-9)
        curr_z = (curr - curr_mean) / (curr_std + 1e-9)
        sim = (cand_z @ curr_z.t()) / cand.shape[1]
        sim = torch.clamp(
            sim, -1.0, 1.0
        )  # Use non-inplace version for compiled functions
        return 1.0 - sim
    else:
        return torch.cdist(cand, curr, p=2)


@timed
def compute_pairwise_distances(
    candidates: torch.Tensor, current: torch.Tensor, metric: str
) -> torch.Tensor:
    """Compute pairwise distances with optional torch.compile optimization."""
    if current.numel() == 0 or current.shape[0] == 0:
        return torch.empty(
            (candidates.shape[0], 0), device=candidates.device, dtype=torch.float32
        )
    # Avoid unnecessary float conversion if already float32
    if candidates.dtype != torch.float32:
        cand = candidates.float()
    else:
        cand = candidates
    if current.dtype != torch.float32:
        curr = current.float()
    else:
        curr = current

    # Use compiled version for cosine/correlation (1.5-1.7x faster)
    # Euclidean doesn't benefit from compilation, use original
    if (
        metric in ("cosine", "correlation")
        and _compute_pairwise_distances_cosine_corr is not None
    ):
        return _compute_pairwise_distances_cosine_corr(cand, curr, metric)
    else:
        return _compute_pairwise_distances_impl(cand, curr, metric)


def _compute_correlation_matrix_impl(
    X: torch.Tensor,  # [B, M, n_pairs] - noisy RDMs
    Y: torch.Tensor,  # [B, M, n_pairs] - true RDMs
    corr_type: str,
) -> torch.Tensor:
    """
    Implementation of correlation matrix computation.
    Uses non-stable sort for Spearman (faster, safe if no ties).
    """
    B, M, n_pairs = X.shape

    # Convert to float (use inplace if already float32)
    if X.dtype != torch.float32:
        X_data = X.float()
    else:
        X_data = X
    if Y.dtype != torch.float32:
        Y_data = Y.float()
    else:
        Y_data = Y

    # For Spearman, rank the data first
    # Use non-stable sort for speed (safe if no ties in continuous data)
    if corr_type == "spearman":
        # Rank along the n_pairs dimension using argsort twice
        # Non-stable sort is faster and safe for continuous RDM values
        X_data = torch.argsort(
            torch.argsort(X_data, dim=2, stable=False), dim=2, stable=False
        ).float()
        Y_data = torch.argsort(
            torch.argsort(Y_data, dim=2, stable=False), dim=2, stable=False
        ).float()

    # Standardize each RDM vector (mean=0, std=1)
    # Fuse mean and subtraction
    X_mean = X_data.mean(dim=2, keepdim=True)  # [B, M, 1]
    Y_mean = Y_data.mean(dim=2, keepdim=True)  # [B, M, 1]

    X_centered = X_data - X_mean
    Y_centered = Y_data - Y_mean

    # Compute std and normalize in one pass
    X_std = X_centered.std(dim=2, keepdim=True, unbiased=False) + 1e-8  # [B, M, 1]
    Y_std = Y_centered.std(dim=2, keepdim=True, unbiased=False) + 1e-8  # [B, M, 1]

    X_normalized = X_centered / X_std  # [B, M, n_pairs]
    Y_normalized = Y_centered / Y_std  # [B, M, n_pairs]

    # Compute correlation using bmm (faster than einsum for this pattern)
    # X_normalized: [B, M, n_pairs], Y_normalized: [B, M, n_pairs]
    # Transpose Y: [B, n_pairs, M]
    # bmm(X, Y^T): [B, M, M]
    correlations = torch.bmm(X_normalized, Y_normalized.transpose(1, 2)) / n_pairs

    # Handle NaNs (from zero variance) - use inplace if possible
    correlations = torch.nan_to_num(correlations, nan=0.0)

    return correlations


@timed
def compute_correlation_matrix(
    X: torch.Tensor,  # [B, M, n_pairs] - noisy RDMs
    Y: torch.Tensor,  # [B, M, n_pairs] - true RDMs
    corr_type: str,
) -> torch.Tensor:
    """
    Compute pairwise correlation matrix between X and Y.

    For each batch b, computes correlation between X[b, i, :] and Y[b, j, :]
    for all pairs (i, j).

    Returns: [B, M, M] where [b, i, j] = corr(X[b, i, :], Y[b, j, :])
    """
    # Use compiled version if available (especially beneficial for Spearman: 2.07x faster)
    if _compute_correlation_matrix_compiled is not None:
        return _compute_correlation_matrix_compiled(X, Y, corr_type)
    else:
        return _compute_correlation_matrix_impl(X, Y, corr_type)


# ==============================================================================
# Aggregation Functions
# ==============================================================================

# Cache for diagonal masks (M is constant per run)
_diag_mask_cache: Dict[tuple[int, torch.device], torch.Tensor] = {}


@timed
def aggregate_across_models(
    utilities_per_model: torch.Tensor,  # [B, M]
    aggregation_across: str,
    beta: float = 5.0,
) -> torch.Tensor:
    """
    Aggregate utilities across models to get single score per candidate.

    Args:
        utilities_per_model: [B, M] utility for each model
        aggregation_across: 'mean', 'min', 'smooth_min'
        beta: temperature for smooth_min

    Returns: [B] aggregated utilities
    """
    B, M = utilities_per_model.shape

    if aggregation_across == "mean":
        # Simple average across models
        utilities = utilities_per_model.mean(dim=1)  # [B]

    elif aggregation_across == "min":
        # Worst-case across models (most conservative)
        utilities = utilities_per_model.min(dim=1)[0]  # [B]

    elif aggregation_across == "smooth_min":
        # Smooth minimum using LogSumExp
        # smooth_min = -1/beta * logsumexp(-beta * x)
        utilities = -torch.logsumexp(-beta * utilities_per_model, dim=1) / beta  # [B]

    else:
        raise ValueError(f"Unknown aggregation_across: {aggregation_across}")

    return utilities


@timed
def compute_model_utilities(
    correlations: torch.Tensor,  # [B, M, M]
    aggregation_within: str,
    beta: float = 5.0,
) -> torch.Tensor:
    """
    Compute utility for each model based on correlation matrix.

    For each model m: U(m) = r_mm - aggregate(r_mm' for m' != m)

    Args:
        correlations: [B, M, M] correlation matrix
        aggregation_within: 'mean', 'min', or 'smooth_min'
        beta: temperature parameter for smooth_min (higher = closer to true min)

    Returns: [B, M] utilities
    """
    B, M, _ = correlations.shape

    # Extract diagonal (self-correlations)
    r_self = torch.diagonal(correlations, dim1=1, dim2=2)  # [B, M]

    if aggregation_within == "mean":
        # Mean of off-diagonal correlations
        r_sum = correlations.sum(dim=2)  # [B, M]
        r_others_mean = (r_sum - r_self) / (M - 1)
        utilities = r_self - r_others_mean

    elif aggregation_within == "min":
        # Minimum of off-diagonal correlations
        # For each model m, we want min over m' != m of correlations[b, m, m']

        # Use cached diagonal mask
        cache_key = (M, correlations.device)
        if cache_key not in _diag_mask_cache:
            _diag_mask_cache[cache_key] = torch.eye(
                M, device=correlations.device, dtype=torch.bool
            )
        diag_mask_base = _diag_mask_cache[cache_key]
        diag_mask = diag_mask_base.unsqueeze(0).expand(B, -1, -1)  # [B, M, M]

        # Use where to avoid clone: set diagonal to infinity for min
        correlations_masked = torch.where(diag_mask, float("inf"), correlations)

        r_others_min = correlations_masked.min(dim=2)[0]  # [B, M]
        utilities = r_self - r_others_min

    elif aggregation_within == "smooth_min":
        # Use cached diagonal mask
        cache_key = (M, correlations.device)
        if cache_key not in _diag_mask_cache:
            _diag_mask_cache[cache_key] = torch.eye(
                M, device=correlations.device, dtype=torch.bool
            )
        diag_mask_base = _diag_mask_cache[cache_key]
        diag_mask = diag_mask_base.unsqueeze(0).expand(B, -1, -1)  # [B, M, M]

        # Use where to avoid clone: set diagonal to large value
        # This ensures exp(-beta * large_value) ≈ 0
        correlations_masked = torch.where(diag_mask, 1e9, correlations)

        # Compute smooth minimum: -1/beta * logsumexp(-beta * x)
        smooth_min = (
            -torch.logsumexp(-beta * correlations_masked, dim=2) / beta
        )  # [B, M]

        utilities = r_self - smooth_min
    else:
        raise ValueError(f"Unknown aggregation_within: {aggregation_within}")

    return utilities


# ==============================================================================
# Torch Compilation Setup
# ==============================================================================

# Set up compiled versions after all implementation functions are defined
if hasattr(torch, "compile"):
    # Compile compute_pairwise_distances for cosine/correlation metrics (1.5-1.7x faster)
    # Note: euclidean doesn't benefit, so we'll use original for that
    # Use 'default' mode to avoid CUDA graph issues with in-place operations
    _compute_pairwise_distances_cosine_corr = torch.compile(
        _compute_pairwise_distances_impl, mode="default"
    )

    # Compile compute_correlation_matrix (especially beneficial for Spearman: 2.07x faster)
    _compute_correlation_matrix_compiled = torch.compile(
        _compute_correlation_matrix_impl, mode="default"
    )
