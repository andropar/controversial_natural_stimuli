"""Noise calibration functions for evaluation."""

import math
from typing import Dict, List

import numpy as np
import torch

from cstims.evaluation.results import NoiseParameters
from cstims.noise_estimation import rdm_noise_by_model
from cstims.rdm import calculate_correlation_value, get_rdm_vector_np


def multiplier_to_noise_ceiling(k: float, nc_base: float) -> float:
    """Convert a noise multiplier to clean-vs-noisy RDM reliability."""
    if k <= 0:
        return 1.0
    if nc_base <= 0 or nc_base >= 1:
        return nc_base
    term = k * k * (1.0 / (nc_base * nc_base) - 1.0)
    return float(1.0 / np.sqrt(1.0 + term))


def multiplier_to_noisy_pair_reliability(k: float, nc_base: float) -> float:
    """Convert a noise multiplier to noisy-vs-noisy RDM reliability."""
    if k <= 0:
        return 1.0
    if nc_base <= 0 or nc_base >= 1:
        return nc_base
    term = k * k * (1.0 / nc_base - 1.0)
    return float(1.0 / (1.0 + term))


def multiplier_to_rdm_reliability(k: float, nc_base: float, comparison: str) -> float:
    """Convert a noise multiplier to the requested empirical RDM reliability target."""
    if comparison == "clean_to_noisy":
        return multiplier_to_noise_ceiling(k, nc_base)
    if comparison == "noisy_to_noisy":
        return multiplier_to_noisy_pair_reliability(k, nc_base)
    raise ValueError(f"Unsupported RDM calibration comparison: {comparison}")


def noise_std_from_multiplier(noise_mult: float, nc_base: float) -> float:
    """Convert a response-noise multiplier to an analytic response noise std."""
    if noise_mult <= 0 or nc_base <= 0 or nc_base >= 1:
        return 0.0
    return float(noise_mult * math.sqrt(1.0 / (nc_base * nc_base) - 1.0))


def response_noise_std_from_rdm_multiplier(noise_mult: float, nc_base: float) -> float:
    """Analytic response-noise std implied by an RDM reliability multiplier."""
    target_nc = multiplier_to_noise_ceiling(noise_mult, nc_base)
    if noise_mult <= 0 or target_nc <= 0 or target_nc >= 1:
        return 0.0
    return float(math.sqrt(1.0 / target_nc - 1.0))


def response_noise_std_from_mode(noise_mult: float, nc_base: float, mode: str) -> float:
    """Resolve the requested analytic response-noise calibration mode."""
    if mode == "response":
        return noise_std_from_multiplier(noise_mult, nc_base)
    if mode == "rdm_analytic":
        return response_noise_std_from_rdm_multiplier(noise_mult, nc_base)
    if mode == "rdm_empirical":
        raise RuntimeError("rdm_empirical calibration needs teacher responses")
    raise ValueError(f"Unsupported fit_noise_calibration: {mode}")


def rdm_noise_std_from_clean(
    clean_rdm: np.ndarray,
    base_noise_ceiling: float,
    noise_mult: float,
) -> float:
    """Analytic RDM-space noise std for a target base noisy-vs-clean reliability."""
    if noise_mult <= 0 or base_noise_ceiling <= 0 or base_noise_ceiling >= 1:
        return 0.0
    var = float(np.var(clean_rdm))
    if var <= 1e-12:
        return 0.0
    return float(
        noise_mult
        * math.sqrt(var * (1.0 / (base_noise_ceiling * base_noise_ceiling) - 1.0))
    )


def empirical_response_noise_rdm_reliability(
    y_clean: np.ndarray,
    clean_rdm: np.ndarray,
    noise_std: float,
    *,
    metric: str,
    corr_type: str,
    rng: np.random.Generator,
    n_samples: int,
    comparison: str = "clean_to_noisy",
) -> float:
    """Estimate RDM reliability after adding response-space noise."""
    if noise_std <= 0:
        return 1.0
    vals = []
    for _ in range(n_samples):
        y_noisy = y_clean + rng.normal(0.0, noise_std, y_clean.shape).astype(np.float32)
        noisy_rdm = get_rdm_vector_np(y_noisy, metric)
        if comparison == "clean_to_noisy":
            vals.append(calculate_correlation_value(noisy_rdm, clean_rdm, corr_type))
        elif comparison == "noisy_to_noisy":
            y_noisy_b = y_clean + rng.normal(0.0, noise_std, y_clean.shape).astype(np.float32)
            noisy_rdm_b = get_rdm_vector_np(y_noisy_b, metric)
            vals.append(calculate_correlation_value(noisy_rdm, noisy_rdm_b, corr_type))
        else:
            raise ValueError(f"Unsupported RDM calibration comparison: {comparison}")
    return float(np.nanmean(vals))


def calibrate_response_noise_for_rdm_reliability(
    y_clean: np.ndarray,
    *,
    target_reliability: float,
    metric: str,
    corr_type: str,
    rng: np.random.Generator,
    n_samples: int,
    max_iter: int,
    comparison: str = "clean_to_noisy",
) -> tuple[float, float]:
    """Find response noise whose empirical RDM reliability matches target."""
    if target_reliability <= 0:
        target_reliability = 1e-6
    if target_reliability >= 1:
        return 0.0, 1.0
    clean_rdm = get_rdm_vector_np(y_clean, metric)
    lo = 0.0
    hi = math.sqrt(1.0 / target_reliability - 1.0)
    hi = max(hi, 1e-3)
    for _ in range(10):
        rel = empirical_response_noise_rdm_reliability(
            y_clean,
            clean_rdm,
            hi,
            metric=metric,
            corr_type=corr_type,
            rng=rng,
            n_samples=n_samples,
            comparison=comparison,
        )
        if np.isfinite(rel) and rel <= target_reliability:
            break
        hi *= 2.0
    best_std = hi
    best_rel = empirical_response_noise_rdm_reliability(
        y_clean,
        clean_rdm,
        hi,
        metric=metric,
        corr_type=corr_type,
        rng=rng,
        n_samples=n_samples,
        comparison=comparison,
    )
    best_err = abs(best_rel - target_reliability) if np.isfinite(best_rel) else np.inf
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rel = empirical_response_noise_rdm_reliability(
            y_clean,
            clean_rdm,
            mid,
            metric=metric,
            corr_type=corr_type,
            rng=rng,
            n_samples=n_samples,
            comparison=comparison,
        )
        err = abs(rel - target_reliability) if np.isfinite(rel) else np.inf
        if err < best_err:
            best_std = mid
            best_rel = rel
            best_err = err
        if not np.isfinite(rel) or rel < target_reliability:
            hi = mid
        else:
            lo = mid
    return float(best_std), float(best_rel)


def calibrate_noise_parameters(
    features: Dict[str, np.ndarray],
    model_names: List[str],
    metrics: List[str],
    target_nc: float,
    device: torch.device,
    mode: str = "analytical",
    calib_n_examples: int = 100,
    n_repeats: int = 100,
    seed: int = 42,
) -> NoiseParameters:
    """
    Calibrate noise parameters for all metrics.

    Args:
        features: Dictionary mapping model names to numpy feature arrays
        model_names: List of model names
        metrics: List of RDM metrics
        target_nc: Target noise ceiling
        device: Torch device
        mode: Noise estimation mode ('analytical' or 'numeric')
        calib_n_examples: Number of examples for calibration
        n_repeats: Number of repeats for calibration
        seed: Random seed

    Returns:
        NoiseParameters object
    """
    noise_by_model_by_metric = {
        metric: rdm_noise_by_model(
            features_by_model_np=features,
            model_names=model_names,
            device=device,
            metric=metric,
            target_nc=target_nc,
            calib_n_examples=calib_n_examples,
            n_repeats=n_repeats,
            mode=mode,
            seed=seed,
        )
        for metric in metrics
    }

    noise_by_model_by_metric_torch = {
        metric: torch.stack(
            [
                torch.tensor(
                    noise_by_model_by_metric[metric][model],
                    device=device,
                    dtype=torch.float32,
                )
                for model in model_names
            ]
        )
        for metric in metrics
    }

    return NoiseParameters(
        noise_by_model_by_metric=noise_by_model_by_metric_torch,
        model_names=model_names,
        metrics=metrics,
    )
