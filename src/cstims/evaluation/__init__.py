"""
Evaluation module for stimulus selection results.

This module provides functions for:
- Loading evaluation payloads (io.py)
- Computing evaluation metrics (computation.py)
- Visualizing evaluation results (plotting.py)
- Constants used by evaluation scripts (constants.py)
- Noise calibration (noise_calibration.py)
"""

from . import (
    computation,
    constants,
    io,
    noise_calibration,
    plotting,
    results,
)

__all__ = [
    "plotting",
    "computation",
    "io",
    "constants",
    "noise_calibration",
    "results",
]
