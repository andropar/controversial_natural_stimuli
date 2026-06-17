"""
Evaluation module for stimulus selection results.

This module provides functions for:
- Loading evaluation payloads (io.py)
- Computing evaluation metrics (computation.py)
- Visualizing evaluation results (plotting.py)
- Constants used by evaluation scripts (constants.py)
- Noise calibration (noise_calibration.py)
"""

import importlib

from . import (
    computation,
    constants,
    io,
    matrix_rows,
    memory,
    noise_calibration,
    payload,
    plotting,
    random_features,
    recovery,
    results,
    track_loading,
)

__all__ = [
    "plotting",
    "computation",
    "io",
    "matrix_rows",
    "memory",
    "constants",
    "noise_calibration",
    "payload",
    "random_features",
    "recovery",
    "results",
    "track_loading",
    "teacher_student",
]


def __getattr__(name: str):
    if name == "teacher_student":
        return importlib.import_module(f"{__name__}.teacher_student")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
