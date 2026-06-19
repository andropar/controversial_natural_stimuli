"""Bootstrap paths for the portable layer_sweep share copy.

The scripts in this directory were copied from the historical layer-sweep
analysis. Keep shared inputs and helper modules available from the source
repository, but resolve layer_sweep outputs relative to this share directory.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()

if _HERE.parents[1].name == "code":
    LAYER_SWEEP_ROOT = _HERE.parents[2]
    CODE_ROOT = _HERE.parents[1]
else:
    LAYER_SWEEP_ROOT = _HERE.parents[1]
    CODE_ROOT = LAYER_SWEEP_ROOT

_LOCAL_SHARE_ROOT = LAYER_SWEEP_ROOT.parents[2]

_BOOTSTRAP_PATHS = (
    CODE_ROOT,
    CODE_ROOT / "analysis",
    _LOCAL_SHARE_ROOT / "src",
    LAYER_SWEEP_ROOT,
)

for p in reversed(_BOOTSTRAP_PATHS):
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)

from cstims import paths as cstims_paths  # noqa: E402

SHARE_ROOT = cstims_paths.project_root()
SOURCE_PROJECT_ROOT = SHARE_ROOT
SOURCE_PAPER_ROOT = cstims_paths.paper_root()
SOURCE_LAYER_SWEEP_ROOT = LAYER_SWEEP_ROOT

_LEGACY_SOURCE_PAPER_ROOT = SHARE_ROOT / "experiments" / "cstim_paper"
_LEGACY_SOURCE_LAYER_SWEEP_ROOT = _LEGACY_SOURCE_PAPER_ROOT / "11_layer_sweep"

_PATHS = (
    *_BOOTSTRAP_PATHS,
    SHARE_ROOT,
    SHARE_ROOT / "src",
    *(
        p for p in (_LEGACY_SOURCE_PAPER_ROOT, _LEGACY_SOURCE_LAYER_SWEEP_ROOT)
        if p.exists()
    ),
)

for p in reversed(_PATHS):
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
