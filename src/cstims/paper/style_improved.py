"""
Improved shared style for all paper figures.

Self-contained improved style with:
- Okabe-Ito-based palette (colorblind-safe)
- A single consistent model-set palette + ordering used everywhere
- A consistent encoding for mixed vs fixed RSA
- A consistent encoding for controversial vs baseline stimuli
- Panel-label helper that places "a", "b", ... in upper-left
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Figure widths follow the original cstim-paper figure convention:
# Nature Neuroscience print widths at 2x working scale.
W_SINGLE = 7.00
W_1_5COL = 9.44
W_DOUBLE = 14.40
MAX_HEIGHT = 19.44

FONT = {
    "panel_label": 16,
    "title": 14,
    "axis_label": 14,
    "tick": 12,
    "legend": 12,
    "annotation": 12,
    "small": 10,
}

RC_PARAMS = {
    "font.size": 14,
    "axes.titlesize": FONT["title"],
    "axes.labelsize": FONT["axis_label"],
    "xtick.labelsize": FONT["tick"],
    "ytick.labelsize": FONT["tick"],
    "legend.fontsize": FONT["legend"],
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 150,
}

DPI = 300

# ---------------------------------------------------------------------------
# Okabe-Ito (colorblind-safe) base palette
# ---------------------------------------------------------------------------
OKABE_ITO = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
}

# ---------------------------------------------------------------------------
# Canonical encodings — use these consistently across every figure
# ---------------------------------------------------------------------------

# Controversial vs. baseline (high-contrast, works in grayscale)
COLOR_CSTIM    = OKABE_ITO["vermillion"]    # #D55E00
COLOR_BASELINE = OKABE_ITO["blue"]          # #0072B2

# Train / population reference distribution (image-statistics figs)
COLOR_TRAIN = "#777777"

# RSA flavor — same hue family with different lightness/style.
# Mixed RSA = filled / solid; Fixed RSA = hatched / dashed.
RSA_STYLES = {
    "mixed": dict(filled=True,  linestyle="-",  hatch=None),
    "fixed": dict(filled=False, linestyle="--", hatch="///"),
}

# ---------------------------------------------------------------------------
# Model-set palette + ordering — Okabe-Ito qualitative
#
# Order chosen to match the paper text: All Models reads first as the union;
# the four controlled sets follow.
# ---------------------------------------------------------------------------
MODEL_SET_ORDER = [
    "all_models",
    "sota",
    "training_objective",
    "architecture",
    "dataset",
]

MODEL_SET_DISPLAY = {
    "all_models":         "All Models",
    "sota":               "State of the Art",
    "training_objective": "Training Objective",
    "architecture":       "Architecture",
    "dataset":            "Dataset",
}

MODEL_SET_DISPLAY_SHORT = {
    "all_models":         "All",
    "sota":               "SOTA",
    "training_objective": "Train. Obj.",
    "architecture":       "Architecture",
    "dataset":            "Dataset",
}

MODEL_SET_COLORS = {
    "all_models":         OKABE_ITO["reddish_purple"],   # purple — union/overall
    "sota":               OKABE_ITO["bluish_green"],     # green
    "training_objective": OKABE_ITO["blue"],             # blue
    "architecture":       OKABE_ITO["vermillion"],       # vermillion
    "dataset":            OKABE_ITO["orange"],           # orange
}


def model_set_color(name: str) -> str:
    return MODEL_SET_COLORS.get(name, "#666666")


# ---------------------------------------------------------------------------
# Apply rcParams and add minor overrides
# ---------------------------------------------------------------------------
def apply_style():
    matplotlib.rcParams.update(RC_PARAMS)
    # Make connecting / reference lines a bit lighter by default.
    matplotlib.rcParams["grid.color"] = "#DDDDDD"
    matplotlib.rcParams["grid.linewidth"] = 0.4
    matplotlib.rcParams["axes.axisbelow"] = True


# ---------------------------------------------------------------------------
# Panel-label helper
# ---------------------------------------------------------------------------
def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.04, **kwargs):
    """Place a bold panel label (a, b, c, ...) in the upper-left of `ax`.

    Coordinates are in ax-fraction space; tweak per-figure if a long y-label
    forces the label outside the figure.
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=FONT["panel_label"],
        fontweight="bold",
        ha="right", va="bottom",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Convenience: lighten / darken an Okabe-Ito hex colour without leaving palette
# ---------------------------------------------------------------------------
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return "#" + "".join(f"{int(round(c*255)):02X}" for c in rgb)

def shade(hex_color: str, amount: float) -> str:
    """amount > 0 lightens toward white, < 0 darkens toward black, range [-1, 1]."""
    r, g, b = _hex_to_rgb(hex_color)
    if amount >= 0:
        target = (1.0, 1.0, 1.0)
    else:
        target = (0.0, 0.0, 0.0)
        amount = -amount
    r = r + (target[0] - r) * amount
    g = g + (target[1] - g) * amount
    b = b + (target[2] - b) * amount
    return _rgb_to_hex((r, g, b))


__all__ = [
    "apply_style",
    "FONT", "DPI",
    "W_SINGLE", "W_1_5COL", "W_DOUBLE", "MAX_HEIGHT",
    "OKABE_ITO",
    "COLOR_CSTIM", "COLOR_BASELINE", "COLOR_TRAIN",
    "RSA_STYLES",
    "MODEL_SET_ORDER", "MODEL_SET_DISPLAY", "MODEL_SET_DISPLAY_SHORT",
    "MODEL_SET_COLORS", "model_set_color",
    "add_panel_label",
    "shade",
]
