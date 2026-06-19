"""
Pure computation functions for evaluation (no I/O).
"""

from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from cstims.evaluation.model_discrimination import model_discriminability
from cstims.rdm import get_rdm_vector
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
    orientation: str = "clean_by_noisy",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Compute correlation matrix at target noise level.

    Args:
        rdms: RDM tensor of shape [M, n_pairs]
        noise_stds: Noise standard deviations of shape [M, 1]
        corr_type: Correlation type ('spearman' or 'pearson')
        n_noise_samples: Number of noise samples to average over
        orientation: Whether scores are ``corr(clean_i, noisy_j)`` or
            ``corr(noisy_i, clean_j)``. The default preserves the historical
            behavior.
        generator: Optional torch random generator for reproducible noise.

    Returns:
        Correlation matrix of shape [M, M]
    """
    correlations = compute_noised_correlation_matrices(
        rdms,
        noise_stds,
        corr_type,
        n_noise_samples=n_noise_samples,
        orientation=orientation,
        generator=generator,
    )
    return correlations.mean(dim=0).cpu()


def compute_noised_correlation_matrices(
    rdms: torch.Tensor,
    noise_stds: torch.Tensor,
    corr_type: str,
    *,
    n_noise_samples: int,
    noise_multiplier: float = 1.0,
    orientation: str = "clean_by_noisy",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample score matrices from clean RDMs plus Gaussian RDM noise."""
    noise_stds = noise_stds.to(rdms.device)
    noise_kwargs = {"device": rdms.device}
    if generator is not None:
        noise_kwargs["generator"] = generator
    noised_rdms = (
        rdms
        + torch.randn((n_noise_samples, *rdms.shape), **noise_kwargs)
        * noise_stds
        * float(noise_multiplier)
    )
    clean_rdms = rdms.unsqueeze(0).expand(n_noise_samples, -1, -1)
    if orientation == "clean_by_noisy":
        return compute_correlation_matrix(clean_rdms, noised_rdms, corr_type)
    elif orientation == "noisy_by_clean":
        return compute_correlation_matrix(noised_rdms, clean_rdms, corr_type)
    raise ValueError(f"Unsupported orientation: {orientation}")


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


def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute normalized AUC over log-scaled x values."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    x_log = np.log10(x_sorted + 1e-10)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    raw_auc = float(integrate(y_sorted, x_log))
    log_span = x_log[-1] - x_log[0]
    if log_span > 0:
        return raw_auc / log_span
    return raw_auc


def compute_discriminability_by_noise_level(
    rdms: torch.Tensor,
    noise_stds: torch.Tensor,
    n_noise_samples: int,
    noise_level_multipliers: np.ndarray,
    corr_type: str,
    *,
    orientation: str = "clean_by_noisy",
    n_bootstrap: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Compute model-discriminability metrics across RDM noise levels.

    Args:
        rdms: RDM tensor of shape [M, n_pairs]
        noise_stds: Noise standard deviations of shape [M, 1]
        n_noise_samples: Number of noise samples per level
        noise_level_multipliers: Array of noise level multipliers
        corr_type: Correlation type ('spearman' or 'pearson')
        orientation: Whether scores are ``corr(clean_i, noisy_j)`` or
            ``corr(noisy_i, clean_j)``.
        n_bootstrap: Optional number of bootstrap resamples over the
            Monte Carlo noise-draw dimension.
        seed: Optional seed for deterministic noise and bootstrap sampling.

    Returns:
        Plain dict with the sampled noise multipliers, per-level metric dicts,
        scalar metric curves, and optional bootstrap arrays.
    """
    device = noise_stds.device
    rdms = rdms.to(device)
    noise_stds = noise_stds.to(device)

    noise_gen = torch.Generator(device=device)
    boot_gen = torch.Generator(device="cpu")
    if seed is not None:
        noise_gen.manual_seed(int(seed))
        boot_gen.manual_seed(int(seed) + 1)
    else:
        noise_gen.seed()
        boot_gen.seed()

    noise_multipliers = np.asarray(noise_level_multipliers, dtype=np.float64)
    n_levels = len(noise_multipliers)
    do_bootstrap = n_bootstrap is not None and n_bootstrap > 0

    metrics_by_noise_level: list[dict] = []
    multiclass_error_probability = np.empty(n_levels, dtype=np.float64)
    pairwise_dominance = np.empty(n_levels, dtype=np.float64)
    pairwise_error_probability = np.empty(n_levels, dtype=np.float64)
    mean_pairwise_margin = np.empty(n_levels, dtype=np.float64)

    multiclass_error_probability_bootstrap = None
    pairwise_dominance_bootstrap = None
    pairwise_error_probability_bootstrap = None
    mean_pairwise_margin_bootstrap = None
    if do_bootstrap:
        n_bootstrap_int = int(n_bootstrap)
        multiclass_error_probability_bootstrap = np.empty(
            (n_levels, n_bootstrap_int),
            dtype=np.float64,
        )
        pairwise_dominance_bootstrap = np.empty(
            (n_levels, n_bootstrap_int),
            dtype=np.float64,
        )
        pairwise_error_probability_bootstrap = np.empty(
            (n_levels, n_bootstrap_int),
            dtype=np.float64,
        )
        mean_pairwise_margin_bootstrap = np.empty(
            (n_levels, n_bootstrap_int),
            dtype=np.float64,
        )

    for level_idx, noise_multiplier in enumerate(noise_multipliers):
        scores = compute_noised_correlation_matrices(
            rdms,
            noise_stds,
            corr_type,
            n_noise_samples=n_noise_samples,
            noise_multiplier=float(noise_multiplier),
            orientation=orientation,
            generator=noise_gen,
        )
        metrics = model_discriminability(
            scores,
            n_bootstrap=int(n_bootstrap) if do_bootstrap else None,
            generator=boot_gen if do_bootstrap else None,
        )
        metrics_by_noise_level.append(metrics)

        multiclass_error_probability[level_idx] = float(
            metrics["non_parametric_multiclass_error_prob"]
        )
        pairwise_dominance[level_idx] = float(metrics["pairwise_dominance"])
        pairwise_error_probability[level_idx] = float(metrics["pairwise_error_prob"])
        mean_pairwise_margin[level_idx] = float(metrics["mean_margin"])

        if do_bootstrap:
            multiclass_error_probability_bootstrap[level_idx] = metrics[
                "error_prob_boot"
            ]
            pairwise_dominance_bootstrap[level_idx] = metrics[
                "pairwise_dominance_boot"
            ]
            pairwise_error_probability_bootstrap[level_idx] = (
                1.0 - metrics["pairwise_dominance_boot"]
            )
            mean_pairwise_margin_bootstrap[level_idx] = metrics["mean_margin_boot"]

        del scores, metrics

    return {
        "noise_multipliers": noise_multipliers,
        "metrics": metrics_by_noise_level,
        "multiclass_error_probability": multiclass_error_probability,
        "multiclass_error_probability_bootstrap": (
            multiclass_error_probability_bootstrap
        ),
        "pairwise_dominance": pairwise_dominance,
        "pairwise_error_probability": pairwise_error_probability,
        "pairwise_error_probability_bootstrap": pairwise_error_probability_bootstrap,
        "mean_pairwise_margin": mean_pairwise_margin,
        "pairwise_dominance_bootstrap": pairwise_dominance_bootstrap,
        "mean_pairwise_margin_bootstrap": mean_pairwise_margin_bootstrap,
    }


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
