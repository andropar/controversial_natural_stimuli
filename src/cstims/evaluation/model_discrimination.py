"""Model-discriminability and feature-noise helpers."""

import numpy as np
import torch

from cstims.rdm_cuda import get_rdm_vector


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


def model_discriminability(model_score: torch.Tensor) -> dict:
    """
    Analyze a simulation tensor of shape ``(n_simulations, n_models, n_models)``.
    Diagonal entries are self-model scores; off-diagonal entries are confusions.
    """
    if model_score.dim() != 3 or model_score.shape[1] != model_score.shape[2]:
        raise ValueError("Input must be a (n_simulations, M, M) tensor.")

    _, M, _ = model_score.shape
    device = model_score.device

    mu = model_score.mean(dim=0)
    mu_ii = torch.diagonal(mu).unsqueeze(1)
    score_deltas = mu_ii - mu

    diagonal_scores = model_score.diagonal(dim1=1, dim2=2)
    differences_tensor = diagonal_scores.unsqueeze(2) - model_score
    std_of_differences = differences_tensor.std(dim=0, unbiased=True)
    d_primes = score_deltas / (std_of_differences + 1e-10)
    d_primes.fill_diagonal_(float("nan"))

    normal_dist = torch.distributions.normal.Normal(0, 1)
    parametric_pairwise_error_probabilities = torch.full(
        (M, M), float("nan"), device=device
    )
    off_diagonal_mask = ~torch.eye(M, dtype=torch.bool, device=device)
    valid_d_primes = d_primes[off_diagonal_mask]
    parametric_pairwise_error_probabilities[off_diagonal_mask] = 1 - normal_dist.cdf(
        valid_d_primes / (2**0.5)
    )
    average_parametric_pairwise_error_probability = torch.nanmean(
        parametric_pairwise_error_probabilities
    )

    diagonal_scores_sims = model_score.diagonal(dim1=1, dim2=2).unsqueeze(2)
    error_made = (diagonal_scores_sims < model_score) + 0.5 * torch.isclose(
        diagonal_scores_sims, model_score
    )
    non_parametric_pairwise_error_probabilities = error_made.float().mean(dim=0)
    non_parametric_pairwise_error_probabilities.fill_diagonal_(float("nan"))

    confusion_matrix = calc_confusion_matrix(model_score)
    total_recoveries = torch.sum(confusion_matrix)
    correct_recoveries = torch.trace(confusion_matrix)
    non_parametric_multiclass_error_prob = (
        total_recoveries - correct_recoveries
    ) / total_recoveries

    return {
        "score_deltas": score_deltas,
        "d_primes": d_primes,
        "parametric_pairwise_error_probs": parametric_pairwise_error_probabilities,
        "average_parametric_pairwise_error_probability": average_parametric_pairwise_error_probability.item(),
        "non_parametric_pairwise_error_probs": non_parametric_pairwise_error_probabilities,
        "non_parametric_confusion_matrix": confusion_matrix,
        "non_parametric_multiclass_error_prob": non_parametric_multiclass_error_prob,
    }
