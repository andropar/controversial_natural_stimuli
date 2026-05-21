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
    CODE_ROOT / "figures",
    CODE_ROOT / "analysis",
    LAYER_SWEEP_ROOT,
    SOURCE_PROJECT_ROOT,
    SOURCE_PAPER_ROOT,
    SOURCE_LAYER_SWEEP_ROOT,
)

# Insert in reverse so local share-copy modules override the source
# repository while source-only helpers remain importable.
for p in reversed(_PATHS):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
