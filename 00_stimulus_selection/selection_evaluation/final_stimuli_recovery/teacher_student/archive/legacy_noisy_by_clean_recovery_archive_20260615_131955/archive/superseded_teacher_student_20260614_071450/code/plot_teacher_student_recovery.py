#!/usr/bin/env python3
"""Plot teacher/student fitted recovery curves and pairwise diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
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
PAIRWISE_PLOT = SCRIPT.parent / "plot_pairwise_margin.py"
RESULTS = SCRIPT.parents[1] / "results"
FIGURES = SCRIPT.parents[1] / "figures"
DATA_SUFFIX = "_teacher_student_recovery"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--figures-root", type=Path, default=FIGURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot = load_module("insilico_plot", SOURCE_PLOT)
    plot.config.EVAL_DATA_DIR = args.results_root
    plot.DATA_SUFFIX = DATA_SUFFIX
    plot.OUTPUT_TARGETS = [(args.figures_root, args.figures_root / "png")]
    plot.make_figure("auc", "teacher_student_recovery")

    pairwise = load_module("pairwise_plot", PAIRWISE_PLOT)
    pairwise.DATA_SUFFIX = DATA_SUFFIX
    pairwise.OUTPUT_PREFIX = "teacher_student_"
    pairwise.plot_track_group(args.results_root, args.figures_root, "raw", ["raw"])
    pairwise.plot_track_group(
        args.results_root,
        args.figures_root,
        "encoding",
        pairwise.ENCODING_TRACKS,
    )


if __name__ == "__main__":
    main()
