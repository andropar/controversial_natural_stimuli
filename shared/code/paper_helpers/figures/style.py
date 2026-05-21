"""
Shared figure style for all paper figures.

Import and call apply_style() at the top of every generate_*.py script.
Use the FONT dict for explicit fontsize= arguments.
Use the W_* constants for figsize widths to stay within Nature Neuroscience
column limits (see FIGURE_CONVENTIONS.md).

All panels should be exported as PDF at their approximate final physical size
so they can be placed at ~100% in Affinity Designer with consistent fonts.
"""
import matplotlib

# ---------------------------------------------------------------------------
# Nature Neuroscience column widths — DESIGN AT 2× SCALE
# Source: https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
#
# Actual NN print widths:  single=89mm(3.5"), 1.5col=120mm(4.72"), double=183mm(7.2")
# We design at 2× so figures are comfortable to work with on screen.
# Place in Affinity Designer at 50% → renders at correct physical size.
# ---------------------------------------------------------------------------
W_SINGLE  = 7.00   # 2 × 89mm  — single column
W_1_5COL  = 9.44   # 2 × 120mm — 1.5 column
W_DOUBLE  = 14.40  # 2 × 183mm — full double column
MAX_HEIGHT = 19.44 # 2 × 247mm — maximum figure height

# ---------------------------------------------------------------------------
# Font sizes (points) — ALL 2× the target NN print size.
# After 50% placement in Affinity these render at: panel_label=8pt, body=6-7pt,
# tick=6pt, annotation/legend=6pt, small=5pt — all within NN requirements.
# NN body text range: 5–7 pt; panel labels: 8 pt bold; absolute minimum: 5 pt.
# ---------------------------------------------------------------------------
FONT = {
    "panel_label": 16,     # renders as 8 pt at 50% — NN panel label
    "title": 14,           # renders as 7 pt at 50%
    "axis_label": 14,      # renders as 7 pt at 50%
    "tick": 12,            # renders as 6 pt at 50%
    "legend": 12,          # renders as 6 pt at 50%
    "annotation": 12,      # renders as 6 pt at 50%
    "small": 10,           # renders as 5 pt at 50% — NN minimum
}

# ---------------------------------------------------------------------------
# Shared rcParams
# ---------------------------------------------------------------------------
RC_PARAMS = {
    # Font sizes (defaults for elements not set explicitly) — 2× NN target sizes
    "font.size": 14,
    "axes.titlesize": FONT["title"],
    "axes.labelsize": FONT["axis_label"],
    "xtick.labelsize": FONT["tick"],
    "ytick.labelsize": FONT["tick"],
    "legend.fontsize": FONT["legend"],

    # Font family — Nature Neuroscience requires Arial or Helvetica
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],

    # Spine / tick styling
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,

    # Export
    "pdf.fonttype": 42,       # TrueType — editable in Affinity/Illustrator
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 150,
}

DPI = 300


def apply_style():
    """Apply shared rcParams. Call once at module level."""
    matplotlib.rcParams.update(RC_PARAMS)
