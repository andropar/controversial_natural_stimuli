#!/usr/bin/env python3
"""Build best-layer scores for both fRSA and mRSA-transfer.

Two strategies (each emits its own csv):
    1. best_layer_per_metric: for each (subject, model, set, metric),
       L = argmax over layers of metric.
    2. best_layer_on_cstim_mrsa: pick L = argmax mRSA over layers per
       (subject, model, set), use that L for BOTH fRSA and mRSA reporting.
       This is the "honest single-pick" version.

Output csvs use the schema of crsa_scores.csv / wrsa_transfer_scores.csv
so the existing plotting code can be reused.

Outputs:
    11_layer_sweep/data/best_layer_crsa_scores.csv     (fRSA, best by cstim fRSA)
    11_layer_sweep/data/best_layer_wrsa_scores.csv     (mRSA, best by cstim mRSA)
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd

from config import PAPER_ROOT, MODEL_DISPLAY_NAMES

DATA_DIR = LAYER_SWEEP_ROOT / "data"
FRSA_IN = DATA_DIR / "fixed_rsa_layer_sweep.csv"
MRSA_IN = DATA_DIR / "wrsa_layer_sweep.csv"


def best_per_metric(in_csv, out_csv, metric_col):
    df = pd.read_csv(in_csv)
    out_rows = []
    for (subject, model, mset), sub in df[df["stimulus_type"] == "controversial"].groupby(
        ["subject", "model", "model_set"]
    ):
        best = sub.loc[sub["rsa"].idxmax()]
        L = best["layer"]
        out_rows.append({
            "subject": subject, "model_set": mset, "model": model,
            "display_name": MODEL_DISPLAY_NAMES.get(model, model),
            "stimulus_type": "controversial",
            "bootstrap_idx": 0, "n_stimuli": int(best["n_stimuli"]),
            metric_col: float(best["rsa"]), "layer": L,
        })
        vicco = df[(df["subject"] == subject) & (df["model"] == model)
                   & (df["stimulus_type"] == "vicco") & (df["layer"] == L)]
        for _, r in vicco.iterrows():
            out_rows.append({
                "subject": subject, "model_set": mset, "model": model,
                "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                "stimulus_type": "vicco",
                "bootstrap_idx": int(r["bootstrap_idx"]),
                "n_stimuli": int(r["n_stimuli"]),
                metric_col: float(r["rsa"]), "layer": L,
            })
    out = pd.DataFrame(out_rows).sort_values(
        ["subject", "model_set", "model", "stimulus_type", "bootstrap_idx"])
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} rows -> {out_csv}")


def main():
    if FRSA_IN.exists():
        best_per_metric(FRSA_IN, DATA_DIR / "best_layer_crsa_scores.csv", "crsa")
    if MRSA_IN.exists():
        best_per_metric(MRSA_IN, DATA_DIR / "best_layer_wrsa_scores.csv", "wrsa_transfer")


if __name__ == "__main__":
    main()
