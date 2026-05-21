"""
Noise calibration functions for evaluation.
"""

from typing import Dict, List

import numpy as np
import torch

from cstims.evaluation.results import NoiseParameters
from cstims.noise_estimation import rdm_noise_by_model


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
