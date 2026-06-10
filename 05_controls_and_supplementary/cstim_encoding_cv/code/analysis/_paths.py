"""Forward path bootstrap to the cstim_encoding_cv code root."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT_PATHS = Path(__file__).resolve().parents[1] / "_paths.py"
_SPEC = importlib.util.spec_from_file_location("_cstim_encoding_cv_paths", _ROOT_PATHS)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load {_ROOT_PATHS}")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

globals().update({k: getattr(_MOD, k) for k in dir(_MOD) if not k.startswith("__")})

