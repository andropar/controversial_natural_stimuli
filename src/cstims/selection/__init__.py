"""
Stimulus selection module for RSA-based experimental design.

This module provides functions for selecting controversial stimuli that maximize
discriminability between competing models under noisy observations.
"""

from .checkpoint import SelectionCheckpoint, load_checkpoint, save_checkpoint
from .image_filter import FilterRecord, ImageFilter, ImageFilterConfig, ValidationResult
from .selector import (
    ExtractedFeatures,
    RefinementRecord,
    SelectionResult,
    TrackAggregationConfig,
    TrackDefinition,
    TrackRuntimeState,
    evaluate_candidates_tracks,
    extract_features_for_indices,
    select_stimuli_multitrack,
)

__all__ = [
    "ExtractedFeatures",
    "FilterRecord",
    "ImageFilter",
    "ImageFilterConfig",
    "RefinementRecord",
    "SelectionCheckpoint",
    "SelectionResult",
    "TrackAggregationConfig",
    "TrackDefinition",
    "TrackRuntimeState",
    "ValidationResult",
    "evaluate_candidates_tracks",
    "extract_features_for_indices",
    "load_checkpoint",
    "save_checkpoint",
    "select_stimuli_multitrack",
]
