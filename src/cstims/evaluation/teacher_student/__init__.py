"""Teacher/student model-recovery evaluation.

This subpackage contains the reusable implementation for the updated recovery
simulation used by the selection-evaluation scripts. Experiment directories
should call into this package instead of carrying separate copies of the method.
"""

from __future__ import annotations

import importlib


__all__ = ["independent_refit_rdm_recovery", "independent_refit", "recovery"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)
