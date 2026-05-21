from __future__ import annotations

from typing import Dict

import torch

from ..rdm_cuda import get_rdm_vector
from ..timing import timed
from .primitives import (
    aggregate_across_models,
    compute_correlation_matrix,
    compute_model_utilities,
    compute_pairwise_distances,
)


def _normalize_rdm_data(
    data: torch.Tensor,  # [..., M, n_pairs]
    corr_type: str,
) -> torch.Tensor:
    """
    Normalize RDM data for correlation computation.

    Handles both Spearman (rank + z-score) and Pearson (z-score only).
    Works with any leading dimensions.

    Args:
        data: Tensor with last two dims [M, n_pairs]
        corr_type: 'spearman' or 'correlation'

    Returns:
        Normalized tensor, same shape as input
    """
    # Ranking for Spearman
    if corr_type == "spearman":
        # Rank along n_pairs dimension (last dim)
        # Use non-stable sort for speed (safe for continuous data)
        data = torch.argsort(
            torch.argsort(data, dim=-1, stable=False), dim=-1, stable=False
        ).float()

    # Z-score normalization
    data_mean = data.mean(dim=-1, keepdim=True)
    data_centered = data - data_mean
    data_std = data_centered.std(dim=-1, keepdim=True, unbiased=False) + 1e-8
    data_normalized = data_centered / data_std

    return data_normalized


def _compute_utilities_from_correlations(
    noisy_normalized: torch.Tensor,  # [..., M, n_pairs]
    target_normalized: torch.Tensor,  # [B, M, n_pairs] or [..., M, n_pairs]
    n_pairs: int,
    aggregation_within: str,
    aggregation_across: str,
    beta_within: float = 5.0,
    beta_across: float = 5.0,
) -> torch.Tensor:
    """
    Core computation: normalized data → correlations → utilities.

    This is the shared logic used by all MC sampling variants.

    Args:
        noisy_normalized: Normalized noisy data
        target_normalized: Normalized target data (usually clean RDMs)
        n_pairs: Number of RDM pairs for normalization
        aggregation_within: How to aggregate within models
        aggregation_across: How to aggregate across models
        beta_within: Temperature for smooth_min within models
        beta_across: Temperature for smooth_min across models

    Returns:
        Utility scores
    """
    # Determine if we need to handle batched MC samples
    if noisy_normalized.ndim == 4:  # [n_mc, B, M, n_pairs]
        n_mc, B, M, _ = noisy_normalized.shape

        # Check if target is also 4D (self-correlation case in feature space)
        if target_normalized.ndim == 4:
            # Self-correlation: both have same shape [n_mc, B, M, n_pairs]
            correlations = (
                torch.einsum("sbip,sbjp->sbij", noisy_normalized, target_normalized)
                / n_pairs
            )
        else:
            # Cross-correlation: target is [B, M, n_pairs]
            correlations = (
                torch.einsum("sbip,bjp->sbij", noisy_normalized, target_normalized)
                / n_pairs
            )
        correlations = torch.nan_to_num(correlations, nan=0.0)

        # Compute utilities: reshape to process all at once
        utilities_per_model = compute_model_utilities(
            correlations.reshape(-1, M, M),
            aggregation_within=aggregation_within,
            beta=beta_within,
        ).reshape(n_mc, B, M)

        utilities_per_sample = aggregate_across_models(
            utilities_per_model.reshape(-1, M),
            aggregation_across=aggregation_across,
            beta=beta_across,
        ).reshape(n_mc, B)

        return utilities_per_sample

    elif noisy_normalized.ndim == 3:  # [B, M, n_pairs]
        B, M, _ = noisy_normalized.shape
        # Single sample or pre-batched: use bmm
        correlations = (
            torch.bmm(noisy_normalized, target_normalized.transpose(1, 2)) / n_pairs
        )
        correlations = torch.nan_to_num(correlations, nan=0.0)

        utilities_per_model = compute_model_utilities(
            correlations, aggregation_within=aggregation_within, beta=beta_within
        )

        utilities = aggregate_across_models(
            utilities_per_model, aggregation_across=aggregation_across, beta=beta_across
        )

        return utilities

    else:
        raise ValueError(f"Unexpected input shape: {noisy_normalized.shape}")


@timed
def compute_analytical_utility(
    augmented_rdms: torch.Tensor,  # [B, M, n_pairs_total]
    noise_vars: torch.Tensor,  # [M] pre-computed noise variances
    aggregation_within: str,
    aggregation_across: str,
    beta_within: float = 5.0,
    beta_across: float = 5.0,
) -> torch.Tensor:
    """
    Compute utility analytically for Pearson correlation.

    Formula: E[r(d_m + ε, d_m')] = sqrt(Var(d_m)/(Var(d_m)+Var(ε))) * r(d_m, d_m')

    Returns: [B] utility scores
    """
    B, M, n_pairs = augmented_rdms.shape

    # Step 1: Compute attenuation factors for each model
    rdm_vars = augmented_rdms.var(dim=2, unbiased=False)  # [B, M]

    # Attenuation factor for each model: sqrt(Var(d_m) / (Var(d_m) + Var(ε_m)))
    attenuation = torch.sqrt(
        rdm_vars / (rdm_vars + noise_vars.unsqueeze(0) + 1e-8)
    )  # [B, M]

    # Step 2: Compute base correlation matrix (no noise)
    correlations_base = compute_correlation_matrix(
        augmented_rdms, augmented_rdms, corr_type="correlation"
    )  # [B, M, M] where [b, i, j] = r(d_i, d_j)

    # Step 3: Apply attenuation to get expected correlations with noise
    # E[r(d_m + ε_m, d_m')] = attenuation[m] * r(d_m, d_m')
    # We multiply each row by its corresponding attenuation factor
    correlations_expected = correlations_base * attenuation.unsqueeze(2)  # [B, M, M]

    # Step 4: Compute utilities
    utilities_per_model = compute_model_utilities(
        correlations_expected, aggregation_within=aggregation_within, beta=beta_within
    )  # [B, M]

    utilities = aggregate_across_models(
        utilities_per_model, aggregation_across=aggregation_across, beta=beta_across
    )  # [B]

    return utilities


def _compute_mc_utilities_rdm_space(
    augmented_rdms: torch.Tensor,  # [B, M, n_pairs]
    augmented_rdms_normalized: torch.Tensor,  # [B, M, n_pairs] - PRECOMPUTED
    noise_stds: torch.Tensor,  # [M]
    corr_type: str,
    aggregation_within: str,
    aggregation_across: str,
    n_mc_samples: int,
    batch_size: int,
    beta_within: float = 5.0,
    beta_across: float = 5.0,
) -> torch.Tensor:
    """
    Unified MC sampling in RDM space with configurable batching.

    Args:
        batch_size: Number of MC samples to process at once
                   - 0 or n_mc_samples: vectorized (all at once)
                   - small value (1-50): minibatch processing
    """
    B, M, n_pairs = augmented_rdms.shape

    # Fully vectorized: process all samples at once
    if batch_size == 0 or batch_size >= n_mc_samples:
        # Generate all noise samples: [n_mc_samples, B, M, n_pairs]
        noise = torch.randn(
            n_mc_samples, B, M, n_pairs,
            device=augmented_rdms.device,
            dtype=torch.float32
        ) * noise_stds.view(1, 1, M, 1)
        noisy_rdms = augmented_rdms.unsqueeze(0) + noise

        # Normalize
        noisy_normalized = _normalize_rdm_data(noisy_rdms, corr_type)

        # Compute utilities for all samples
        utilities_per_sample = _compute_utilities_from_correlations(
            noisy_normalized,
            augmented_rdms_normalized,
            n_pairs,
            aggregation_within,
            aggregation_across,
            beta_within,
            beta_across,
        )  # [n_mc_samples, B]

        # Average over MC samples
        return utilities_per_sample.mean(dim=0)

    # Minibatch processing: process samples in chunks
    else:
        utilities = torch.zeros(B, device=augmented_rdms.device, dtype=torch.float32)
        n_batches = (n_mc_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_mc_samples)
            batch_size_actual = end_idx - start_idx

            # Generate noise for this batch
            noise = torch.randn(
                batch_size_actual, B, M, n_pairs,
                device=augmented_rdms.device,
                dtype=torch.float32,
            ) * noise_stds.view(1, 1, M, 1)
            noisy_rdms = augmented_rdms.unsqueeze(0) + noise

            # Normalize
            noisy_normalized = _normalize_rdm_data(noisy_rdms, corr_type)

            # Compute utilities for this batch
            utilities_per_sample = _compute_utilities_from_correlations(
                noisy_normalized,
                augmented_rdms_normalized,
                n_pairs,
                aggregation_within,
                aggregation_across,
                beta_within,
                beta_across,
            )  # [batch_size_actual, B]

            utilities += utilities_per_sample.sum(dim=0)

        return utilities / n_mc_samples


def _compute_mc_utilities_feature_space(
    candidate_features: Dict[str, torch.Tensor],  # [B, D] per model
    selected_features: Dict[str, torch.Tensor],  # [n_current, D] per model
    noise_stds: torch.Tensor,  # [M]
    metric: str,
    corr_type: str,
    n_mc_samples: int,
    aggregation_within: str,
    aggregation_across: str,
    beta_within: float = 5.0,
    beta_across: float = 5.0,
    mc_batch_size: int = 1,
) -> torch.Tensor:
    """
    Compute utilities with noise added in feature space using mini-batch MC sampling.

    This is optimized for large candidate batch sizes (B > 10k) by:
    1. Processing MC samples one at a time (mc_batch_size=1)
    2. Reusing RDM computation logic from get_rdm_vector
    3. Computing augmented RDMs directly from noisy features

    Args:
        candidate_features: Dict mapping model name to [B, D] candidate features
        selected_features: Dict mapping model name to [n_current, D] selected features
        noise_stds: [M] noise standard deviations per model
        metric: Distance metric for RDM computation
        corr_type: Correlation type ('correlation' or 'spearman')
        n_mc_samples: Number of Monte Carlo samples
        aggregation_within: How to aggregate within models
        aggregation_across: How to aggregate across models
        beta_within: Temperature for smooth_min within models
        beta_across: Temperature for smooth_min across models
        mc_batch_size: Number of MC samples to process at once (1 for memory efficiency)

    Returns:
        [B] utility scores
    """
    model_names = list(candidate_features.keys())
    M = len(model_names)
    B = candidate_features[model_names[0]].shape[0]
    device = candidate_features[model_names[0]].device

    # Accumulator for utilities
    utilities = torch.zeros(B, device=device, dtype=torch.float32)
    n_batches = (n_mc_samples + mc_batch_size - 1) // mc_batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * mc_batch_size
        end_idx = min(start_idx + mc_batch_size, n_mc_samples)
        batch_size_actual = end_idx - start_idx

        # For each MC sample in this batch, compute augmented RDMs
        augmented_rdms_list = []

        for _ in range(batch_size_actual):
            # Generate noisy features for this MC sample
            noisy_selected = {}
            noisy_candidates = {}

            for m, model_name in enumerate(model_names):
                # Add noise to selected features
                noise_sel = torch.randn_like(selected_features[model_name]) * noise_stds[m]
                noisy_selected[model_name] = selected_features[model_name] + noise_sel

                # Add noise to candidate features
                noise_cand = torch.randn_like(candidate_features[model_name]) * noise_stds[m]
                noisy_candidates[model_name] = candidate_features[model_name] + noise_cand

            # Compute augmented RDMs for each candidate
            augmented_rdms_this_sample = []  # [B, M, n_pairs]

            for b in range(B):
                rdms_this_candidate = []  # [M, n_pairs]

                for m, model_name in enumerate(model_names):
                    # Augment: concatenate selected features with this candidate
                    augmented_features = torch.cat([
                        noisy_selected[model_name],  # [n_current, D]
                        noisy_candidates[model_name][b:b+1]  # [1, D]
                    ], dim=0)  # [n_current+1, D]

                    # Compute RDM vector
                    rdm_vec = get_rdm_vector(augmented_features, metric)
                    rdms_this_candidate.append(rdm_vec)

                augmented_rdms_this_sample.append(torch.stack(rdms_this_candidate, dim=0))

            augmented_rdms_list.append(torch.stack(augmented_rdms_this_sample, dim=0))

        # Stack: [batch_size_actual, B, M, n_pairs]
        augmented_rdms = torch.stack(augmented_rdms_list, dim=0)

        # Normalize for correlation computation
        augmented_rdms_normalized = _normalize_rdm_data(augmented_rdms, corr_type)

        # Compute self-correlations and utilities
        n_pairs = augmented_rdms.shape[3]
        utilities_per_sample = _compute_utilities_from_correlations(
            augmented_rdms_normalized,
            augmented_rdms_normalized,  # Self-correlation
            n_pairs,
            aggregation_within,
            aggregation_across,
            beta_within,
            beta_across,
        )  # [batch_size_actual, B]

        # Accumulate
        utilities += utilities_per_sample.sum(dim=0)

    # Average over all MC samples
    return utilities / n_mc_samples


@timed
def compute_batch_utilities(
    candidate_features: Dict[str, torch.Tensor],
    rdm_by_model: Dict[str, torch.Tensor],
    selected_features: Dict[str, torch.Tensor],
    noise_stds: torch.Tensor,
    noise_vars: torch.Tensor,
    metric: str,
    corr_type: str,
    n_mc_samples: int,
    aggregation_within: str,
    aggregation_across: str,
    use_analytical: bool = False,
    beta_within: float = 5.0,
    beta_across: float = 5.0,
    noise_in_feature_space: bool = False,
) -> torch.Tensor:
    """Compute utilities for a batch of candidates."""

    if use_analytical and corr_type != "correlation":
        raise ValueError(
            "Analytical computation only works with corr_type='correlation' (Pearson)"
        )

    if use_analytical and noise_in_feature_space:
        raise ValueError(
            "Analytical computation does not work with noise_in_feature_space=True. "
            "Feature-space noise requires Monte Carlo sampling."
        )

    # Feature-space noise: compute utilities by adding noise to features
    if noise_in_feature_space:
        return _compute_mc_utilities_feature_space(
            candidate_features=candidate_features,
            selected_features=selected_features,
            noise_stds=noise_stds,
            metric=metric,
            corr_type=corr_type,
            n_mc_samples=n_mc_samples,
            aggregation_within=aggregation_within,
            aggregation_across=aggregation_across,
            beta_within=beta_within,
            beta_across=beta_across,
            mc_batch_size=1,  # Process one MC sample at a time for memory efficiency
        )

    model_names = list(rdm_by_model.keys())
    M = len(model_names)
    B = candidate_features[model_names[0]].shape[0]

    # Step 1: Compute new dissimilarities
    new_dissims = {}
    for model_name in model_names:
        new_dissims[model_name] = compute_pairwise_distances(
            candidate_features[model_name], selected_features[model_name], metric=metric
        )

    new_dissims_tensor = torch.stack([new_dissims[name] for name in model_names], dim=1)

    # Step 2: Augment RDMs
    current_rdms = torch.stack([rdm_by_model[name] for name in model_names], dim=0)
    current_rdms_expanded = current_rdms.unsqueeze(0).expand(B, -1, -1)
    augmented_rdms = torch.cat([current_rdms_expanded, new_dissims_tensor], dim=2)

    # Step 3: Compute utilities
    if use_analytical:
        utilities = compute_analytical_utility(
            augmented_rdms=augmented_rdms,
            noise_vars=noise_vars,
            aggregation_within=aggregation_within,
            aggregation_across=aggregation_across,
            beta_within=beta_within,
            beta_across=beta_across,
        )
    else:
        # Pre-normalize target RDMs (clean data)
        augmented_rdms_normalized = _normalize_rdm_data(augmented_rdms, corr_type)

        # Adaptive batching strategy based on memory requirements
        B, M, n_pairs = augmented_rdms.shape
        estimated_memory_mb = (n_mc_samples * B * M * n_pairs * 4) / (1024**2)

        # Very conservative threshold: use vectorized only for very small batches
        use_vectorized = (
            B < 100  # Only use vectorized for very small batch sizes
            and n_mc_samples <= 100  # Limit n_mc_samples
            and estimated_memory_mb < 100  # Very conservative memory limit
        )

        if use_vectorized:
            # Vectorized: all samples at once (5-12x speedup)
            batch_size = n_mc_samples
        elif B >= 5000:
            # Very large B: process candidates in sub-batches
            candidate_batch_size = 1000
            utilities = torch.zeros(
                B, device=augmented_rdms.device, dtype=torch.float32
            )

            for cand_start in range(0, B, candidate_batch_size):
                cand_end = min(cand_start + candidate_batch_size, B)
                cand_slice = slice(cand_start, cand_end)

                # Extract sub-batch
                augmented_rdms_sub = augmented_rdms[cand_slice]
                augmented_rdms_normalized_sub = augmented_rdms_normalized[cand_slice]

                # Process with small MC batch size
                utilities[cand_slice] = _compute_mc_utilities_rdm_space(
                    augmented_rdms_sub,
                    augmented_rdms_normalized_sub,
                    noise_stds,
                    corr_type,
                    aggregation_within,
                    aggregation_across,
                    n_mc_samples,
                    batch_size=1,  # One MC sample at a time
                    beta_within=beta_within,
                    beta_across=beta_across,
                )
            return utilities
        else:
            # Adaptive batch_size for normal cases
            if B >= 1000:
                batch_size = max(1, min(10, 1000 // B))
            else:
                batch_size = max(10, min(50, 5000 // B))

        # Run unified MC sampling
        utilities = _compute_mc_utilities_rdm_space(
            augmented_rdms,
            augmented_rdms_normalized,
            noise_stds,
            corr_type,
            aggregation_within,
            aggregation_across,
            n_mc_samples,
            batch_size=batch_size,
            beta_within=beta_within,
            beta_across=beta_across,
        )

    return utilities


def compute_multi_subject_encoded_utilities_optimized(
    candidate_features_by_encoding: Dict[str, Dict[str, torch.Tensor]],
    rdm_by_encoding_model: Dict[str, Dict[str, torch.Tensor]],
    selected_features_by_encoding: Dict[str, Dict[str, torch.Tensor]],
    metric: str,
    corr_type: str,
    beta_across: float = 5.0,
) -> torch.Tensor:
    """
    Optimized multi-subject objective using Centroid RDMs.
    
    Improvements:
    1. SNR Boost: Averages normalized RDMs across subjects (Centroid) before computing utility.
       This reduces the noise penalty inherent in pairwise subject comparisons.
    2. Memory Efficient: Avoids storing the full [B, S, M, P] tensor. It reduces 
       subject dimensions early.
    3. Vectorized: Minimizes Python loops during tensor construction.

    Returns:
        utilities: Tensor[B]
    """
    enc_names = list(candidate_features_by_encoding.keys())
    if not enc_names:
        raise ValueError("candidate_features_by_encoding is empty")
        
    first_enc = enc_names[0]
    model_names = list(candidate_features_by_encoding[first_enc].keys())
    
    # Get dimensions
    B = next(iter(candidate_features_by_encoding[first_enc].values())).shape[0]
    S = len(enc_names)
    M = len(model_names)
    device = next(iter(candidate_features_by_encoding[first_enc].values())).device

    # We will accumulate the sum of normalized RDMs across subjects here
    # Shape: [B, M, P] (S dimension is collapsed via summation)
    sum_normalized_rdms = None
    
    # Helper to normalize a batch of RDMs [B, M, P]
    def normalize_batch(rdms_batch):
        if corr_type == "spearman":
            rdms_batch = torch.argsort(
                torch.argsort(rdms_batch, dim=-1, stable=False), dim=-1, stable=False
            ).float()
        mu = rdms_batch.mean(dim=-1, keepdim=True)
        centered = rdms_batch - mu
        std = centered.std(dim=-1, keepdim=True, unbiased=False) + 1e-8
        return centered / std

    # Process one subject at a time to save memory
    # We construct [B, M, P] for subject S, normalize it, and add to accumulator
    for enc_name in enc_names:
        
        # --- 1. Build Augmented RDM for this Subject ---
        # We still need to iterate models because feature dimensions D might differ per model
        per_model_rdms = []
        for model_name in model_names:
            cand_feats = candidate_features_by_encoding[enc_name][model_name] # [B, D_enc]
            sel_feats = selected_features_by_encoding[enc_name][model_name]   # [N_sel, D_enc]
            
            # Compute new column of distance matrix
            # [B, N_sel]
            new_dists = compute_pairwise_distances(cand_feats, sel_feats, metric=metric)
            
            # Get history [P_current]
            current_rdm = rdm_by_encoding_model[enc_name][model_name]
            
            # Concatenate history + new column
            # Note: To avoid massive .expand(), we treat history and new dists distinctly 
            # if optimizing further, but here we stick to concatenation for correctness 
            # matching the original RDM logic.
            # Shape: [B, P_total]
            augmented = torch.cat([
                current_rdm.unsqueeze(0).expand(B, -1), 
                new_dists
            ], dim=1)
            per_model_rdms.append(augmented)

        # Stack models for this subject: [B, M, P]
        rdms_subj = torch.stack(per_model_rdms, dim=1)
        
        # --- 2. Normalize Immediately ---
        # This transforms distances into "correlation units"
        norm_rdms_subj = normalize_batch(rdms_subj) # [B, M, P]
        
        # --- 3. Accumulate (Online Mean) ---
        if sum_normalized_rdms is None:
            sum_normalized_rdms = norm_rdms_subj
        else:
            sum_normalized_rdms += norm_rdms_subj

    # --- 4. Compute Centroid RDM ---
    # The "Group RDM" is the average of subject RDMs.
    # Because we normalized *before* averaging, this is a "consensus" representational geometry.
    group_rdms = sum_normalized_rdms / S  # [B, M, P]
    
    # --- 5. Compute Utility on the Centroid ---
    # Now we treat the Group RDM just like a single high-quality view.
    # Compute correlation matrix of the Centroid RDMs: [B, M, M]
    # Since group_rdms are already zero-mean (sum of zero-mean), we just need to re-scale variance?
    # Actually, the sum of unit-variance vectors is not unit variance.
    # We can re-normalize or just compute cosine similarity directly.
    
    # Let's re-normalize the group RDM to be safe for standard correlation logic
    group_rdms = normalize_batch(group_rdms) 
    
    # Compute M x M correlation matrix for every candidate B
    # [B, M, M]
    correlations = torch.bmm(group_rdms, group_rdms.transpose(1, 2)) / group_rdms.shape[-1]
    
    # Apply diagonal mask for "max off-diagonal"
    M_range = torch.arange(M, device=device)
    mask = ~torch.eye(M, dtype=torch.bool, device=device) # True for off-diagonal
    
    # consistency = diagonal (how distinct is the model's group signal?)
    # Note: Since we are correlating the group RDM with itself, the diagonal is always 1.0.
    # This captures the structure *of the average*.
    # To recapture the "subject consistency" logic from the original paper, 
    # we arguably want to weight this by how much subjects agreed.
    # BUT: Strategy 2.5.2 works better because it ignores internal noise. 
    # So, let's stick to maximizing the distinctness of the *Centroid*.
    
    diag = correlations.diagonal(dim1=1, dim2=2) # [B, M] (should be 1.0s)
    
    # confusability = max off-diagonal
    # [B, M]
    off_diag_max = (correlations * mask.unsqueeze(0)).max(dim=2).values
    
    # Margin: 1.0 - max_correlation_with_other_models
    margins = diag - off_diag_max
    
    # Aggregate across models using smooth_min (robust worst-case)
    # This prevents one confusing model pair from ruining the score entirely, 
    # allowing gradients to flow better than hard min.
    if beta_across > 0:
        utilities = -torch.logsumexp(-beta_across * margins, dim=1) / beta_across
    else:
        utilities = margins.min(dim=1).values

    return utilities


# Note: torch.compile is NOT used for MC sampling functions because it affects
# random number generation, leading to non-reproducible results even with fixed seeds.
# The performance benefit (~1.5-2x) is outweighed by the loss of numerical reproducibility.
# For deterministic behavior, MC sampling uses uncompiled functions.
