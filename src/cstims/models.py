"""Compatibility re-exports for model constants.

New code should import these names from :mod:`cstims.constants`.
"""

from __future__ import annotations

from cstims.constants import (
    MODEL_DISPLAY_NAMES,
    MODEL_SET_ORDER,
    MODEL_SETS,
    MODELS_EXCL_VICREG,
)

__all__ = [
    "MODEL_DISPLAY_NAMES",
    "MODEL_SET_ORDER",
    "MODEL_SETS",
    "MODELS_EXCL_VICREG",
]

