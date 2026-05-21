"""
Data classes for evaluation results.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class NoiseParameters:
    """Container for noise parameters."""

    noise_by_model_by_metric: Dict[str, torch.Tensor]
    model_names: List[str]
    metrics: List[str]

    def get_noise_stds(self, metric: str) -> torch.Tensor:
        """Get noise standard deviations for a given metric."""
        return torch.sqrt(self.noise_by_model_by_metric[metric]).view(-1, 1)
