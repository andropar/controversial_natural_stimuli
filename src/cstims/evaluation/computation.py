"""
Pure computation functions for evaluation (no I/O).
"""

from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from cstims.evaluation.model_discrimination import model_discriminability
from cstims.rdm_cuda import get_rdm_vector
from cstims.selection.primitives import compute_correlation_matrix


def compute_all_rdms(
    features: Dict[str, torch.Tensor], metrics: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Compute RDMs for all metrics.

    Args:
        features: Dictionary mapping model names to feature tensors
        metrics: List of RDM metrics to compute

    Returns:
        Dictionary mapping metric names to stacked RDM tensors [M, n_pairs]
    """
    return {
        metric: torch.stack(
            [get_rdm_vector(feat, metric=metric) for feat in features.values()]
        )
        for metric in metrics
    }


def compute_correlation_at_target_noise(
    rdms: torch.Tensor,
    noise_stds: torch.Tensor,
    corr_type: str,
    n_noise_samples: int = 100,
) -> torch.Tensor:
    """
    Compute correlation matrix at target noise level.

    Args:
        rdms: RDM tensor of shape [M, n_pairs]
        noise_stds: Noise standard deviations of shape [M, 1]
        corr_type: Correlation type ('spearman' or 'pearson')
        n_noise_samples: Number of noise samples to average over

    Returns:
        Correlation matrix of shape [M, M]
    """
    noised_rdms = (
        rdms + torch.randn((n_noise_samples, *rdms.shape)).to(rdms.device) * noise_stds
    )
    repeat_correlations = compute_correlation_matrix(
        rdms.repeat(n_noise_samples, 1, 1), noised_rdms, corr_type
    )
    return repeat_correlations.mean(dim=0).cpu()


def compute_clean_correlation_matrix(
    rdms: torch.Tensor,
    corr_type: str,
) -> torch.Tensor:
    """
    Compute clean (no-noise) correlation matrix between RDMs.

    Args:
        rdms: RDM tensor of shape [M, n_pairs]
        corr_type: Correlation type ('spearman' or 'pearson')

    Returns:
        Correlation matrix of shape [M, M]
    """
    rdms_batched = rdms.unsqueeze(0)  # [1, M, n_pairs]
    correlations = compute_correlation_matrix(rdms_batched, rdms_batched, corr_type)
    return correlations[0].cpu()


def compute_discriminability_by_noise_level(
    rdms: torch.Tensor,
    noise_stds: torch.Tensor,
    n_noise_samples: int,
    noise_level_multipliers: np.ndarray,
    corr_type: str,
) -> List[Dict[str, torch.Tensor]]:
    """
    Compute discriminability metrics across noise levels.

    Args:
        rdms: RDM tensor of shape [M, n_pairs]
        noise_stds: Noise standard deviations of shape [M, 1]
        n_noise_samples: Number of noise samples per level
        noise_level_multipliers: Array of noise level multipliers
        corr_type: Correlation type ('spearman' or 'pearson')

    Returns:
        List of discriminability dictionaries, one per noise level
    """
    # Use GPU if available (noise_stds is typically on desired device)
    device = noise_stds.device
    rdms = rdms.to(device)

    noised_correlations = {}
    for noise_level_multiplier in noise_level_multipliers:
        noised_rdms = (
            rdms
            + torch.randn((n_noise_samples, *rdms.shape), device=device)
            * noise_stds
            * noise_level_multiplier
        )
        repeat_correlations = compute_correlation_matrix(
            rdms.repeat(n_noise_samples, 1, 1), noised_rdms, corr_type
        )
        noised_correlations[noise_level_multiplier] = repeat_correlations.to("cpu")

    discriminability_by_noise_level = [
        model_discriminability(noised_correlations[multiplier])
        for multiplier in noise_level_multipliers
    ]

    return discriminability_by_noise_level


def compute_random_baseline_rdms(
    random_features: Dict[str, np.ndarray],
    model_names: List[str],
    metrics: List[str],
    n_selected_stimuli: int,
    n_random_subsets: int,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    """
    Compute RDMs for random baseline subsets (Vectorized).

    Args:
        random_features: Dictionary of random features by model name (numpy arrays)
        model_names: List of model names
        metrics: List of RDM metrics
        n_selected_stimuli: Number of selected stimuli (size of random subsets)
        n_random_subsets: Number of random subsets to generate
        device: Torch device

    Returns:
        List of RDM dictionaries, one per random subset.
        Each dict maps metric -> Tensor[M, n_pairs]
    """
    import gc
    try:
        import psutil
        def _log_mem(label):
            p = psutil.Process()
            rss = p.memory_info().rss / 1024 / 1024
            vm = psutil.virtual_memory()
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            print(f"    [RDM-MEM {label}] CPU: {rss:.0f}MB | System: {vm.percent:.0f}% used | GPU: {gpu_mb:.0f}MB")
    except ImportError:
        def _log_mem(label):
            pass

    max_available = min([len(random_features[name]) for name in model_names])
    print(f"    [DEBUG] Computing random RDMs: {n_random_subsets} subsets of {n_selected_stimuli} from {max_available} available")

    # Pre-sample all indices: [n_random_subsets, n_selected_stimuli]
    # Use numpy for efficient random sampling
    all_random_indices = np.zeros((n_random_subsets, n_selected_stimuli), dtype=int)
    for i in range(n_random_subsets):
        all_random_indices[i] = np.random.choice(
            max_available, size=n_selected_stimuli, replace=False
        )

    # Intermediate storage: results_intermediate[metric][model_name] = Tensor[S, P]
    # Store on CPU to save GPU memory
    results_intermediate = {m: {} for m in metrics}

    # Process each model in batch
    _log_mem("before_model_loop")
    for i, model_name in enumerate(tqdm(model_names, desc="Computing random RDMs (vectorized)")):
        # Gather features: [n_subsets, n_selected, D]
        # Use fancy indexing
        feats_np = random_features[model_name]
        is_memmap = isinstance(feats_np, np.memmap)
        feat_shape = feats_np.shape
        gathered_feats = feats_np[all_random_indices]

        if i == 0:
            print(f"    [DEBUG] First model '{model_name}': shape={feat_shape}, memmap={is_memmap}, "
                  f"gathered_shape={gathered_feats.shape}, gathered_size={gathered_feats.nbytes/1024/1024:.1f}MB")

        # Move to GPU once
        gathered_feats_torch = torch.tensor(
            gathered_feats, device=device, dtype=torch.float32
        )

        # Free the numpy intermediate immediately
        del gathered_feats

        for metric in metrics:
            # Compute RDMs in batch: [S, P]
            # get_rdm_vector now supports [B, N, D] inputs
            rdms_batch = get_rdm_vector(gathered_feats_torch, metric=metric)
            # Store on CPU to free GPU memory
            results_intermediate[metric][model_name] = rdms_batch.cpu()

        # Free GPU memory after processing this model
        del gathered_feats_torch
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Delete this model's features from random_features to free CPU RAM
        # This is safe because we've already extracted what we need
        del random_features[model_name]
        gc.collect()

        if i == 0 or (i + 1) % 5 == 0:
            _log_mem(f"after_model_{i+1}/{len(model_names)}")

    # Reconstruct the expected output structure: List[Dict[metric, Tensor[M, P]]]
    final_list = []
    for i in range(n_random_subsets):
        subset_dict = {}
        for metric in metrics:
            # Stack models for this metric and subset: [M, P]
            model_rdms = [
                results_intermediate[metric][name][i] for name in model_names
            ]
            # Keep on CPU - will be moved to GPU as needed during discriminability
            subset_dict[metric] = torch.stack(model_rdms)
        final_list.append(subset_dict)

    return final_list
