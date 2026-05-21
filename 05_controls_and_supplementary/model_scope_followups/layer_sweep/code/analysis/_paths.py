"""Bootstrap paths for the portable layer_sweep share copy.

The scripts in this directory are copied out of
experiments/cstim_paper/11_layer_sweep. Keep shared inputs and helper modules
coming from the source repository, but resolve layer_sweep outputs relative to
this share directory.
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

SOURCE_PROJECT_ROOT = Path("/data/home_roth/_stachelschwein/rsa_based_selection")
SOURCE_PAPER_ROOT = SOURCE_PROJECT_ROOT / "experiments" / "cstim_paper"
SOURCE_LAYER_SWEEP_ROOT = SOURCE_PAPER_ROOT / "11_layer_sweep"

_PATHS = (
    CODE_ROOT,
    CODE_ROOT / "analysis",
    LAYER_SWEEP_ROOT,
    SOURCE_PROJECT_ROOT,
    SOURCE_PAPER_ROOT,
    SOURCE_LAYER_SWEEP_ROOT,
)

# Insert in reverse so the portable share copy stays ahead of the source
# repository. The source repo is still available for shared helper modules.
for p in reversed(_PATHS):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
