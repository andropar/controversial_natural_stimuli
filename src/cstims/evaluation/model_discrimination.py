"""Model-discriminability and feature-noise helpers."""

import numpy as np
import torch

from cstims.rdm import get_rdm_vector


def calibrate_feature_noise(
    features,
    target_self_correlation,
    rdm_metric="euclidean",
    n_samples=20,
    max_iterations=20,
    tolerance=0.01,
    device=None,
):
    """
    Line-search for isotropic Gaussian feature noise that matches a target
    clean-vs-noised RDM self-correlation.
    """
    if device is None:
        device = features.device

    features = features.to(device)
    clean_rdm = get_rdm_vector(features, metric=rdm_metric).detach()
    clean_rdm_centered = clean_rdm - clean_rdm.mean()

    sigma_low = 0.0
    sigma_high = torch.std(features).item() * 10.0
    best_sigma = None
    best_corr_error = float("inf")

    for _ in range(max_iterations):
        sigma = (sigma_low + sigma_high) / 2.0
        corrs = []
        for _ in range(n_samples):
            noise = torch.randn_like(features) * sigma
            noisy_feats = features + noise
            noisy_rdm = get_rdm_vector(noisy_feats, metric=rdm_metric).detach()
            noisy_rdm_centered = noisy_rdm - noisy_rdm.mean()
            corr = torch.nn.functional.cosine_similarity(
                clean_rdm_centered.unsqueeze(0), noisy_rdm_centered.unsqueeze(0)
            ).item()
            corrs.append(corr)
        mean_corr = float(np.mean(corrs))
        corr_error = abs(mean_corr - target_self_correlation)
        if corr_error < best_corr_error:
            best_corr_error = corr_error
            best_sigma = sigma
        if best_corr_error < tolerance:
            return best_sigma
        if mean_corr < target_self_correlation:
            sigma_high = sigma
        else:
            sigma_low = sigma

    return best_sigma


def calc_confusion_matrix(model_score: torch.Tensor) -> torch.Tensor:
    """Count which clean model has the maximum score for each noised model."""
    _, M, _ = model_score.shape
    detected_models = torch.argmax(model_score, dim=2)
    one_hot_detections = torch.nn.functional.one_hot(
        detected_models, num_classes=M
    ).float()
    return torch.einsum("sij->ij", one_hot_detections)


def _validate_model_score(model_score: torch.Tensor) -> tuple[int, int, torch.device]:
    """Validate model-score shape and return common dimensions."""
    if model_score.dim() != 3 or model_score.shape[1] != model_score.shape[2]:
        raise ValueError("Input must be a (n_simulations, M, M) tensor.")
    n_simulations, n_models, _ = model_score.shape
    return n_simulations, n_models, model_score.device


def _off_diagonal_mask(n_models: int, device: torch.device) -> torch.Tensor:
    """Mask the off-diagonal model-pair entries in an ``M x M`` matrix."""
    return ~torch.eye(n_models, dtype=torch.bool, device=device)


def _pairwise_margins(
    model_score: torch.Tensor,
    off_diagonal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute self-model margins against every candidate clean model."""
    diagonal_scores = model_score.diagonal(dim1=1, dim2=2)
    pairwise_margin = diagonal_scores.unsqueeze(2) - model_score
    pairwise_margin_offdiag = pairwise_margin[:, off_diagonal_mask]
    return pairwise_margin, pairwise_margin_offdiag


def _tie_adjusted_dominance(pairwise_margin: torch.Tensor) -> torch.Tensor:
    """Score wins as 1, losses as 0, and exact ties as 0.5."""
    return (
        (pairwise_margin > 0).float()
        + 0.5
        * torch.isclose(
            pairwise_margin,
            torch.zeros_like(pairwise_margin),
        ).float()
    )


def _parametric_pairwise_metrics(
    model_score: torch.Tensor,
    pairwise_margin: torch.Tensor,
    off_diagonal_mask: torch.Tensor,
) -> dict:
    """Compute mean-margin d-primes and Gaussian pairwise error estimates."""
    _, n_models, _ = model_score.shape
    device = model_score.device

    mu = model_score.mean(dim=0)
    mu_ii = torch.diagonal(mu).unsqueeze(1)
    score_deltas = mu_ii - mu

    std_of_differences = pairwise_margin.std(dim=0, unbiased=True)
    d_primes = score_deltas / (std_of_differences + 1e-10)
    d_primes.fill_diagonal_(float("nan"))

    normal_dist = torch.distributions.normal.Normal(0, 1)
    pairwise_error_probabilities = torch.full(
        (n_models, n_models), float("nan"), device=device
    )
    valid_d_primes = d_primes[off_diagonal_mask]
    pairwise_error_probabilities[off_diagonal_mask] = 1 - normal_dist.cdf(
        valid_d_primes / (2**0.5)
    )

    return {
        "score_deltas": score_deltas,
        "d_primes": d_primes,
        "parametric_pairwise_error_probs": pairwise_error_probabilities,
        "average_parametric_pairwise_error_probability": torch.nanmean(
            pairwise_error_probabilities
        ).item(),
    }


def _non_parametric_pairwise_metrics(
    model_score: torch.Tensor,
    pairwise_margin_offdiag: torch.Tensor,
) -> dict:
    """Compute empirical pairwise and multiclass recovery metrics."""
    diagonal_scores_sims = model_score.diagonal(dim1=1, dim2=2).unsqueeze(2)
    error_made = (diagonal_scores_sims < model_score) + 0.5 * torch.isclose(
        diagonal_scores_sims, model_score
    )
    pairwise_error_probabilities = error_made.float().mean(dim=0)
    pairwise_error_probabilities.fill_diagonal_(float("nan"))

    confusion_matrix = calc_confusion_matrix(model_score)
    total_recoveries = torch.sum(confusion_matrix)
    correct_recoveries = torch.trace(confusion_matrix)
    multiclass_error_prob = (total_recoveries - correct_recoveries) / total_recoveries

    pairwise_dominance = _tie_adjusted_dominance(pairwise_margin_offdiag)
    pairwise_dominance_mean = pairwise_dominance.mean()

    return {
        "non_parametric_pairwise_error_probs": pairwise_error_probabilities,
        "non_parametric_confusion_matrix": confusion_matrix,
        "non_parametric_multiclass_error_prob": multiclass_error_prob,
        "pairwise_dominance": float(pairwise_dominance_mean.item()),
        "pairwise_error_prob": float(1.0 - pairwise_dominance_mean.item()),
        "mean_margin": float(pairwise_margin_offdiag.mean().item()),
    }


def _bootstrap_discriminability_metrics(
    model_score: torch.Tensor,
    pairwise_margin_offdiag: torch.Tensor,
    n_bootstrap: int,
    generator: torch.Generator | None,
) -> dict:
    """Bootstrap empirical recovery and margin metrics over simulations."""
    n_simulations, n_models, _ = model_score.shape

    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.seed()

    model_score_cpu = model_score.detach().cpu()
    detected_cpu = torch.argmax(model_score_cpu, dim=2)
    true_idx_cpu = torch.arange(n_models).unsqueeze(0)
    correct_cpu = (detected_cpu == true_idx_cpu).float()

    pairwise_margin_cpu = pairwise_margin_offdiag.detach().cpu()
    pairwise_dominance_cpu = _tie_adjusted_dominance(pairwise_margin_cpu)

    idx = torch.randint(
        0,
        n_simulations,
        (int(n_bootstrap), n_simulations),
        generator=generator,
    )
    boot_acc = correct_cpu[idx].mean(dim=(1, 2))

    return {
        "error_prob_boot": (1.0 - boot_acc).numpy(),
        "pairwise_dominance_boot": pairwise_dominance_cpu[idx]
        .mean(dim=(1, 2))
        .numpy(),
        "mean_margin_boot": pairwise_margin_cpu[idx].mean(dim=(1, 2)).numpy(),
    }


def model_discriminability(
    model_score: torch.Tensor,
    *,
    n_bootstrap: int | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """
    Analyze a simulation tensor of shape ``(n_simulations, n_models, n_models)``.
    Diagonal entries are self-model scores; off-diagonal entries are confusions.
    """
    _, n_models, device = _validate_model_score(model_score)
    off_diagonal_mask = _off_diagonal_mask(n_models, device)
    pairwise_margin, pairwise_margin_offdiag = _pairwise_margins(
        model_score,
        off_diagonal_mask,
    )

    result = {}
    result.update(
        _parametric_pairwise_metrics(
            model_score,
            pairwise_margin,
            off_diagonal_mask,
        )
    )
    result.update(
        _non_parametric_pairwise_metrics(
            model_score,
            pairwise_margin_offdiag,
        )
    )

    if n_bootstrap is not None and n_bootstrap > 0:
        result.update(
            _bootstrap_discriminability_metrics(
                model_score,
                pairwise_margin_offdiag,
                int(n_bootstrap),
                generator,
            )
        )

    return result
