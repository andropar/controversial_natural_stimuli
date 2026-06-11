#!/usr/bin/env python3
"""Plot the noisy-by-clean recovery curves and AUC bars."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[4]
SOURCE_PLOT = (
    ROOT
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "code"
    / "figures"
    / "plot_insilico_evaluation_unique_improved.py"
)
RESULTS = SCRIPT.parents[1] / "results"
FIGURES = SCRIPT.parents[1] / "figures"


def load_source_plot():
    spec = importlib.util.spec_from_file_location("insilico_plot", SOURCE_PLOT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SOURCE_PLOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    plot = load_source_plot()
    plot.config.EVAL_DATA_DIR = RESULTS
    plot.DATA_SUFFIX = "_noisy_by_clean_boot"
    plot.OUTPUT_TARGETS = [(FIGURES, FIGURES / "png")]
    plot.make_figure("auc", "noisy_by_clean_recovery")


if __name__ == "__main__":
    main()
