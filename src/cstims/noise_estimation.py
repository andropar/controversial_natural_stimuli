from __future__ import annotations

import torch
import math
import numpy as np
from typing import Dict, Sequence
from tqdm import tqdm

from .rdm_cuda import get_rdm_vector
from .rdm_cuda import calculate_correlation

def estimate_noise_variance(
    features: torch.Tensor,
    metric: str,
    noise_ceiling_target: float,
    n_estimation: int = 1000,
    n_repeats: int = 50,
    device: torch.device = torch.device("cpu"),
    corr_type: str = "correlation",
    mc_repeats: int = 32,
    seed: int | None = None,
) -> dict[str, object]:
    """
    Estimate the fixed noise variance for a single feature set for a given noise ceiling target.

    This function implements a Monte Carlo approach to estimate noise variance parameters that
    achieve a target noise ceiling in RSA-based stimulus selection. The noise ceiling represents
    the theoretical maximum reliability achievable given the inherent variability in the data.

    Methodological approach:
    1. Randomly sample subsets of features (n_estimation samples) multiple times (n_repeats)
    2. Calculate RDM vectors for each subset using the specified metric
    3. Compute variance of RDM vectors across repeats to estimate typical RDM variance
    4. Use the noise ceiling target to derive required noise variance parameters
    5. Verify achieved correlation via Monte Carlo simulation

    The noise variance is calculated using the formula:
        var_noise = var_rdm * (1 / (noise_ceiling_target^2) - 1)

    The 0.5 correction factor accounts for the nonlinear propagation of feature noise to
    pairwise distances in RDMs. The base formula assumes a simple signal-to-noise relationship,
    but empirically we observe that noise compounds when computing pairwise distances.

    Args:
        features (torch.Tensor): Feature matrix (n_samples, n_features)
        metric (str): Distance metric for RDM calculation ('euclidean', 'cosine', 'correlation')
        noise_ceiling_target (float): Target noise ceiling (0.0 to 1.0). Values >= 1.0 set noise to 0.
        n_estimation (int): Number of samples to use for each variance estimation (default: 1000)
        n_repeats (int): Number of Monte Carlo repeats for robust variance estimation (default: 50)
        device (torch.device): Device for tensor operations (default: cpu)
        corr_type (str): Correlation type for verification ('correlation', 'spearman', etc.)
        mc_repeats (int): Number of Monte Carlo repeats for verification (default: 32)
        seed (int | None): Random seed for reproducibility

    Returns:
        dict[str, object]:
            {
                'var_noise': float (estimated variance),
                'final_corr': float (achieved correlation from verification)
            }

    Raises:
        ValueError: If n_estimation < 3 (minimum required for RDM variance calculation)

    Note:
        - The function automatically handles cases where RDM vectors are empty
        - For noise_ceiling_target >= 1.0, returns {'var_noise': 0.0, 'final_corr': 1.0}
        - Uses a minimum noise variance of 0.01 for numerical stability
    """
    num_samples = features.shape[0]
    estimation_n = min(n_estimation, num_samples)

    all_repeat_vars: list[float] = []
    for _ in range(n_repeats):
        estimation_indices = torch.from_numpy(
            np.random.choice(num_samples, size=estimation_n, replace=False)
        ).to(features.device)
        estimation_features = features[estimation_indices]
        rdm_vec = get_rdm_vector(estimation_features, metric)
        if rdm_vec.numel() > 0:
            var_val = torch.var(rdm_vec).item()
            all_repeat_vars.append(var_val)

    avg_typical_rdm_var = np.mean(all_repeat_vars) if all_repeat_vars else 0.1

    if noise_ceiling_target >= 1.0:
        var_noise_fixed = 0.0
    else:
        nc_target_clamped = min(noise_ceiling_target, 0.9999)
        var_noise_fixed = (
            0.01
            if avg_typical_rdm_var <= 1e-9
            else avg_typical_rdm_var * (1.0 / (nc_target_clamped**2) - 1.0)
        )
        var_noise_fixed = max(0.0, var_noise_fixed)

    if noise_ceiling_target >= 1.0:
        return {"var_noise": var_noise_fixed, "final_corr": 1.0}

    clean_rdm = get_rdm_vector(features, metric)
    if clean_rdm.numel() == 0:
        return {"var_noise": var_noise_fixed, "final_corr": float("nan")}

    achieved_corr = _estimate_single_achieved_correlation(
        features,
        clean_rdm,
        var_noise_fixed,
        metric,
        corr_type,
        mc_repeats,
        seed,
        eval_offset=0,
    )

    return {"var_noise": var_noise_fixed, "final_corr": achieved_corr}


def _generate_gaussian_noise(
    shape: tuple[int, ...],
    variance: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Generate Gaussian noise with a specified variance.

    Args:
        shape (tuple[int, ...]): Shape of the noise tensor.
        variance (float): Variance of the noise to generate.
        device (torch.device): The device on which the noise tensor will be created.
        dtype (torch.dtype): Data type of the returned tensor.
        generator (torch.Generator): An optional torch random generator for reproducibility.

    Returns:
        torch.Tensor: Gaussian noise tensor with given shape and variance.
    """
    if variance <= 0.0:
        return torch.zeros(shape, device=device, dtype=dtype)
    std = math.sqrt(variance)
    return torch.normal(
        mean=0.0,
        std=std,
        size=shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )


from cstims.selection.primitives import compute_correlation_matrix

def _estimate_single_achieved_correlation(
    clean_features: torch.Tensor,
    clean_rdm: torch.Tensor,
    noise_var: float,
    metric: str,
    corr_type: str,
    mc_repeats: int,
    seed: int | None,
    eval_offset: int,
) -> float:
    """
    Estimate the mean correlation between noisy and clean RDMs via Monte Carlo sampling.
    Vectorized over mc_repeats.

    Args:
        clean_features (torch.Tensor): Original (clean) feature matrix.
        clean_rdm (torch.Tensor): Reference RDM vector for clean features.
        noise_var (float): Variance of isotropic Gaussian noise to apply.
        metric (str): Dissimilarity metric ('euclidean', 'correlation', etc.).
        corr_type (str): Correlation type ('correlation', 'spearman', etc.).
        mc_repeats (int): Number of Monte Carlo repeats.
        seed (int | None): Optional random seed for reproducibility.
        eval_offset (int): Offset to the seed (to decorrelate multiple trials).

    Returns:
        float: Average achieved correlation between noised and clean RDMs.
    """
    if clean_rdm.numel() == 0:
        return float("nan")

    device = clean_features.device
    dtype = clean_features.dtype
    P = clean_rdm.shape[0]

    # Generate noise [mc_repeats, P]
    # Note: _generate_gaussian_noise uses torch.normal which supports size=(mc_repeats, P)
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        # We only need to set seed once if we generate all noise at once
        generator.manual_seed(seed + eval_offset)

    noise = _generate_gaussian_noise(
        (mc_repeats, P),
        noise_var,
        device,
        dtype,
        generator,
    )
    
    # noisy_rdm: [mc_repeats, P]
    # clean_rdm: [P] -> broadcast to [mc_repeats, P]
    noisy_rdm = clean_rdm.unsqueeze(0) + noise
    
    # Compute correlations
    # compute_correlation_matrix expects inputs [B, M, P]
    # Here B = mc_repeats, M = 1, P = n_pairs
    
    X = noisy_rdm.unsqueeze(1) # [mc_repeats, 1, P]
    Y = clean_rdm.unsqueeze(0).unsqueeze(1).expand(mc_repeats, 1, P) # [mc_repeats, 1, P]
    
    # Returns [mc_repeats, 1, 1]
    corrs_batched = compute_correlation_matrix(X, Y, corr_type)
    
    return float(torch.nan_to_num(corrs_batched.mean(), nan=0.0).item())


def estimate_noise_variance_numeric(
    features: torch.Tensor,
    metric: str,
    noise_ceiling_target: float,
    corr_type: str = "correlation",
    mc_repeats: int = 64,
    max_iters: int = 20,
    tol: float = 1e-3,
    noise_bounds: tuple[float, float] | None = None,
    device: torch.device | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """
    Numerically estimate the noise variance required to reach a target representational similarity analysis (RSA) reliability.

    This method injects isotropic Gaussian noise with variance ``var_noise`` directly into the
    feature matrices, recomputes RDMs, and averages the correlation between clean and noised RDMs
    across Monte Carlo repeats. A 1D bisection search is used to find the variance whose
    correlation matches the desired noise ceiling.

    Args:
        features (torch.Tensor): Feature matrix (num_samples, feature_dim).
        metric (str): Dissimilarity metric for RDM computation ('euclidean', etc.).
        noise_ceiling_target (float): Target achieved RDM correlation (e.g., 0.9).
        corr_type (str): Correlation type ('correlation', 'spearman', etc.).
        mc_repeats (int): Number of Monte Carlo repeats for estimation.
        max_iters (int): Maximum number of bisection iterations.
        tol (float): Absolute tolerance for stopping criterion.
        noise_bounds (tuple[float, float] | None): Initial lower and upper search bounds for variance.
        device (torch.device | None): Compute device.
        seed (int | None): Random seed for reproducibility.

    Returns:
        dict[str, object]: 
            {
                'var_noise': float (estimated variance),
                'final_corr': float (achieved correlation),
                'corr_history': list[tuple[float, float]] (search history (variance, achieved_corr)),
                'iterations': int (number of bisection iterations)
            }
    """
    if noise_ceiling_target >= 1.0:
        return {
            "var_noise": 0.0,
            "final_corr": 1.0,
            "corr_history": [(0.0, 1.0)],
            "iterations": 0,
        }

    if mc_repeats <= 0:
        raise ValueError("mc_repeats must be positive")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")

    device = device or features.device
    features = features.to(device)

    clean_rdm = get_rdm_vector(features, metric)

    if clean_rdm.numel() == 0:
        return {
            "var_noise": 0.0,
            "final_corr": float("nan"),
            "corr_history": [],
            "iterations": 0,
        }

    eval_counter = 0

    def _corr_for_variance(variance: float) -> float:
        """
        Estimate achieved correlation for the provided features at this noise variance.
        """
        nonlocal eval_counter
        corr_val = _estimate_single_achieved_correlation(
            features,
            clean_rdm,
            variance,
            metric,
            corr_type,
            mc_repeats,
            seed,
            eval_counter,
        )
        eval_counter += 1
        return corr_val

    lower, upper = noise_bounds if noise_bounds is not None else (0.0, 1.0)
    lower_corr = _corr_for_variance(lower)
    corr_history: list[tuple[float, float]] = [(lower, lower_corr)]

    target = float(noise_ceiling_target)
    if lower_corr < target:
        return {
            "var_noise_A": lower,
            "var_noise_B": lower,
            "final_corr": lower_corr,
            "corr_history": corr_history,
            "iterations": 0,
        }

    step_scale = 2.0
    upper_corr = _corr_for_variance(upper)
    corr_history.append((upper, upper_corr))
    while upper_corr > target and upper < 1e6:
        lower = upper
        lower_corr = upper_corr
        upper *= step_scale
        upper_corr = _corr_for_variance(upper)
        corr_history.append((upper, upper_corr))
        if math.isinf(upper):
            break

    iterations = 0
    final_corr = upper_corr
    while iterations < max_iters:
        mid = 0.5 * (lower + upper)
        mid_corr = _corr_for_variance(mid)
        corr_history.append((mid, mid_corr))
        if abs(mid_corr - target) <= tol:
            final_corr = mid_corr
            lower = upper = mid
            break
        if mid_corr > target:
            lower = mid
            lower_corr = mid_corr
            final_corr = mid_corr
        else:
            upper = mid
            final_corr = mid_corr
        iterations += 1
        if upper - lower <= tol:
            break

    variance_estimate = 0.5 * (lower + upper)
    final_corr = _corr_for_variance(variance_estimate)
    corr_history.append((variance_estimate, final_corr))
    return {
        "var_noise": variance_estimate,
        "final_corr": final_corr,
        "corr_history": corr_history,
        "iterations": iterations,
    }

def rdm_noise_by_model(
    features_by_model_np: Dict[str, np.ndarray],
    model_names: Sequence[str],
    device: torch.device,
    metric: str,
    target_nc: float,
    calib_n_examples: int = 1000,
    n_repeats: int = 100, 
    seed: int = 42,
    mode: str = "numeric",
    corr_type: str = "correlation",
) -> Dict[str, float]:
    """
    Estimate per-model noise variance so that RDM reliability matches a specified target. 

    Args:
        features_by_model_np (Dict[str, np.ndarray]): Mapping from model name to features array.
        model_names (Sequence[str]): Sequence of model names to estimate for.
        device (torch.device): Target computation device.
        metric (str): Dissimilarity metric for RDM computation.
        target_nc (float): Target reliability/noise ceiling (e.g., 0.9).
        mode (str): Either 'analytical' or 'numeric' estimation mode.
        calib_n_examples (int): Number of samples to use for calibration/estimation.
        n_repeats (int): Number of repetitions for stochastic methods.
        seed (int): Random seed for reproducibility.
        corr_type (str): Correlation type for RDM computation.
    Returns:
        Dict[str, float]: Mapping from model name to estimated noise variance.
    """
    rng = np.random.default_rng(seed)
    noise: Dict[str, float] = {}
    for name in tqdm(model_names, desc="Estimating per-model noise variances"):
        X = features_by_model_np[name]
        n = X.shape[0]
        take = min(n, max(8, int(calib_n_examples)))
        if take < n:
            idx = rng.choice(n, size=take, replace=False)
            X = X[idx]
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        if mode == "analytical":
            result = estimate_noise_variance(
                X_t,
                metric=metric,
                noise_ceiling_target=target_nc,
                n_estimation=take,
                n_repeats=max(1, int(n_repeats)),
                device=device,
                corr_type=corr_type,
                mc_repeats=max(1, int(n_repeats)),
                seed=seed,
            )
            var_a = float(result["var_noise"])
        elif mode == "numeric":
            numeric_result = estimate_noise_variance_numeric(
                X_t,
                metric=metric,
                noise_ceiling_target=target_nc,
                corr_type=corr_type,
                mc_repeats=max(1, int(n_repeats)),
                max_iters=20,
                tol=1e-3,
                noise_bounds=(0.0, 1.0),
                device=device,
                seed=seed,
            )
            var_a = float(numeric_result["var_noise"])
        else:
            raise ValueError(f"Invalid mode: {mode}")
        noise[name] = float(var_a)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return noise