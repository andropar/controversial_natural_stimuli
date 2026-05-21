"""
RSA-Based Controversial Stimuli Selection

A GPU-accelerated implementation for selecting controversial stimuli using RSA.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

def estimate_noise_variance(*args, **kwargs):
    from .noise_estimation import estimate_noise_variance as _estimate_noise_variance

    return _estimate_noise_variance(*args, **kwargs)


def estimate_noise_variance_numeric(*args, **kwargs):
    from .noise_estimation import (
        estimate_noise_variance_numeric as _estimate_noise_variance_numeric,
    )

    return _estimate_noise_variance_numeric(*args, **kwargs)


def get_rdm_vector(*args, **kwargs):
    from .rdm_cuda import get_rdm_vector as _get_rdm_vector

    return _get_rdm_vector(*args, **kwargs)

__all__ = [
    "__version__",
    "estimate_noise_variance",
    "estimate_noise_variance_numeric",
    "get_rdm_vector",
]
