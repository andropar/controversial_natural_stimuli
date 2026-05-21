"""Feature extraction module for neural network features."""

from .universal_extractor import LayerResolution, UniversalFeatureExtractor, get_custom_model

__all__ = [
    "UniversalFeatureExtractor",
    "LayerResolution",
    "get_custom_model",
]
