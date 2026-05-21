#!/usr/bin/env python3
"""Aggregate fixed_rsa_layer_sweep.csv into layer-level deltas.

For each (subject, model, layer, cstim_set) we compute:
    delta_rsa = rsa_cstim - mean(rsa_vicco_bootstraps_for_same_layer)

Output:
    11_layer_sweep/data/layer_drop_summary.csv
        columns: subject, model, display_name, layer, layer_depth_rank,
                 model_set, rsa_cstim, rsa_vicco_mean, rsa_vicco_std,
                 delta_rsa, n_vicco_boot

A second wide summary (mean across subjects) is written as
    layer_drop_summary_subject_avg.csv
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
from pathlib import Path

import numpy as np
import pandas as pd

from config import PAPER_ROOT
from layers_config import MODEL_LAYERS, layer_depth_rank, layer_depth_frac

DATA_DIR = LAYER_SWEEP_ROOT / "data"
IN_CSV = DATA_DIR / "fixed_rsa_layer_sweep.csv"
OUT_CSV = DATA_DIR / "layer_drop_summary.csv"
OUT_AVG_CSV = DATA_DIR / "layer_drop_summary_subject_avg.csv"


def main():
    df = pd.read_csv(IN_CSV)

    # vicco rows: per (subject, model, layer)
    vicco = df[df["stimulus_type"] == "vicco"]
    vicco_agg = (
        vicco.groupby(["subject", "model", "display_name", "layer"], as_index=False)
        .agg(rsa_vicco_mean=("rsa", "mean"),
             rsa_vicco_std=("rsa", "std"),
             n_vicco_boot=("rsa", "size"))
    )

    # cstim rows: per (subject, model, layer, model_set)
    cstim = df[df["stimulus_type"] == "controversial"][[
        "subject", "model", "display_name", "layer", "model_set", "rsa"
    ]].rename(columns={"rsa": "rsa_cstim"})

    merged = cstim.merge(
        vicco_agg, on=["subject", "model", "display_name", "layer"], how="left"
    )
    merged["delta_rsa"] = merged["rsa_cstim"] - merged["rsa_vicco_mean"]
    merged["layer_depth_rank"] = [
        layer_depth_rank(m, l) for m, l in zip(merged["model"], merged["layer"])
    ]
    merged["layer_depth_frac"] = [
        layer_depth_frac(m, l) for m, l in zip(merged["model"], merged["layer"])
    ]
    merged = merged[[
        "subject", "model", "display_name", "layer", "layer_depth_rank",
        "layer_depth_frac", "model_set",
        "rsa_cstim", "rsa_vicco_mean", "rsa_vicco_std", "delta_rsa", "n_vicco_boot",
    ]].sort_values(["model", "model_set", "subject", "layer_depth_rank"])
    merged.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(merged)} rows -> {OUT_CSV}")

    # Subject-averaged summary (mean ± SEM across subjects).
    avg = (merged.groupby(
        ["model", "display_name", "layer", "layer_depth_rank", "layer_depth_frac",
         "model_set"], as_index=False)
        .agg(rsa_cstim_mean=("rsa_cstim", "mean"),
             rsa_cstim_sem=("rsa_cstim", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
             rsa_vicco_mean_mean=("rsa_vicco_mean", "mean"),
             rsa_vicco_mean_sem=("rsa_vicco_mean", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
             delta_rsa_mean=("delta_rsa", "mean"),
             delta_rsa_sem=("delta_rsa", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
             n_subjects=("delta_rsa", "size")))
    avg = avg.sort_values(["model", "model_set", "layer_depth_rank"])
    avg.to_csv(OUT_AVG_CSV, index=False)
    print(f"Wrote {len(avg)} subject-averaged rows -> {OUT_AVG_CSV}")


if __name__ == "__main__":
    main()
