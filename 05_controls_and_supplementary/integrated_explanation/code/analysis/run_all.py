#!/usr/bin/env python3
"""Run the explanation-analysis pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(script: str) -> None:
    path = HERE / script
    print(f"\n== {script} ==")
    subprocess.run([sys.executable, str(path)], check=True, cwd=HERE.parent)


def main() -> None:
    run("01_matched_counterfactual_ladder.py")
    run("02_reliability_and_residual_readout.py")
    run("03_pair_level_variance_partition.py")
    run("04_make_explanation_summary_figure.py")


if __name__ == "__main__":
    main()
