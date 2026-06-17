#!/usr/bin/env python3
"""Descriptive join of best-layer rescue against set-level low-level shift.

For each cstim_set, we compute:
    mean_rescue (across models, subject-averaged) of delta_best - delta_late
    mean_low_level_shift (set-level Mahalanobis distance from vicco baseline)

The low-level distance per stimulus is in
    05_controls_and_supplementary/low_level_and_ood/ood_controls/results/low_level_robustness_per_image_distances.csv

Output:
    results/rescue_vs_low_level.csv
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd

from cstims.paper.config import PAPER_ROOT

DATA_DIR = LAYER_SWEEP_ROOT / "results"
RESCUE_CSV = DATA_DIR / "layer_rescue_summary.csv"
LOWLEVEL_CSV = (
    PAPER_ROOT
    / "05_controls_and_supplementary"
    / "low_level_and_ood"
    / "ood_controls"
    / "results"
    / "low_level_robustness_per_image_distances.csv"
)
OUT_CSV = DATA_DIR / "rescue_vs_low_level.csv"


def main():
    rescue = pd.read_csv(RESCUE_CSV)
    if not LOWLEVEL_CSV.exists():
        print(f"[skip] {LOWLEVEL_CSV} not found; cannot join low-level shift")
        return

    low = pd.read_csv(LOWLEVEL_CSV)
    # Set-level mean Mahalanobis distance.
    set_mahal = low.groupby("stim_set", as_index=False)["mahal_distance"].mean()
    set_mahal = set_mahal.rename(columns={"stim_set": "model_set",
                                          "mahal_distance": "mean_mahal"})

    # Rescue summary across (subjects, models) per cstim_set.
    set_rescue = (rescue.groupby("model_set", as_index=False)
                  .agg(rescue_mean=("rescue", "mean"),
                       rescue_sem=("rescue", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                       delta_late_mean=("delta_late", "mean"),
                       delta_best_mean=("delta_best", "mean"),
                       n=("rescue", "size")))

    out = set_rescue.merge(set_mahal, on="model_set", how="left")
    out = out.sort_values("model_set")
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} set-level rows -> {OUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
