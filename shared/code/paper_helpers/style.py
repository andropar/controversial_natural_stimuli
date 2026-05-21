"""
Shared figure style for all paper figures.

Import and call apply_style() at the top of every generate_*.py script.
Use the FONT dict for explicit fontsize= arguments.

All panels should be exported as PDF at their approximate final physical size
so they can be placed at ~100% in Affinity Designer with consistent fonts.
"""
import matplotlib

# ---------------------------------------------------------------------------
# Font sizes (points) — use these for explicit fontsize= arguments
# ---------------------------------------------------------------------------
FONT = {
    "panel_label": 10,     # a, b, c, d
    "title": 8,            # subplot/panel titles
    "axis_label": 7,       # xlabel, ylabel
    "tick": 6,             # tick labels
    "legend": 6,           # legend text
    "annotation": 6,       # in-plot annotations, stat text
    "small": 5.5,          # tiny labels only when space-constrained
}

# ---------------------------------------------------------------------------
# Shared rcParams
# ---------------------------------------------------------------------------
RC_PARAMS = {
    # Font sizes (defaults for elements not set explicitly)
    "font.size": 7,
    "axes.titlesize": FONT["title"],
    "axes.labelsize": FONT["axis_label"],
    "xtick.labelsize": FONT["tick"],
    "ytick.labelsize": FONT["tick"],
    "legend.fontsize": FONT["legend"],

    # Font family
    "font.family": "sans-serif",

    # Spine / tick styling
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,

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
