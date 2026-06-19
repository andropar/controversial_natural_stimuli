"""Path bootstrap for the CSTIM encoding-CV follow-up."""

from __future__ import annotations

import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
if HERE.parents[0].name in {"analysis", "figures"}:
    CODE_DIR = HERE.parents[1]
    ROOT = HERE.parents[2]
else:
    CODE_DIR = HERE.parents[0]
    ROOT = HERE.parents[1]

SHARE_ROOT = ROOT.parents[1]
LAYER_SWEEP_ROOT = SHARE_ROOT / "05_controls_and_supplementary" / "model_scope_followups" / "layer_sweep"
SOURCE_LAYER_SWEEP_ROOT = (
    Path("/data/home_roth/_stachelschwein/rsa_based_selection")
    / "experiments"
    / "cstim_paper"
    / "11_layer_sweep"
)

for path in (
    SHARE_ROOT / "src",
    SHARE_ROOT / "src",
    LAYER_SWEEP_ROOT / "code",
    LAYER_SWEEP_ROOT / "code" / "analysis",
):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PNG_DIR = FIGURES_DIR / "png"


def _cache_dir() -> Path:
    override = os.environ.get("CSTIM_ENCODING_CV_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    try:
        from cstims import paths as cstims_paths

        return cstims_paths.feature_cache_dir() / "cstim_encoding_cv"
    except Exception:
        return ROOT / "cache_or_heavy"


CACHE_DIR = _cache_dir()
