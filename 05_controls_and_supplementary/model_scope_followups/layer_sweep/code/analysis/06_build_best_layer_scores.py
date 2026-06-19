#!/usr/bin/env python3
"""Build a best-layer scores CSV in the exact schema of crsa_scores.csv.

For each (subject, model, model_set):
    L = argmax over layers of cstim_rsa
    cstim row: rsa at layer L
    vicco rows: 50 bootstrap rsa values at layer L

This is an "optimistic per-(subject, model, set) layer choice" — what fRSA
alignment would look like if we picked the layer that best aligns with each
cstim set, rather than the single paper layer per model.

Output:
    results/best_layer_crsa_scores.csv
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd

from cstims import paths
from cstims.constants import MODEL_DISPLAY_NAMES
PAPER_ROOT = paths.paper_root()

DATA_DIR = LAYER_SWEEP_ROOT / "results"
IN_CSV = DATA_DIR / "fixed_rsa_layer_sweep.csv"
OUT_CSV = DATA_DIR / "best_layer_crsa_scores.csv"


def main():
    df = pd.read_csv(IN_CSV)

    out_rows = []
    for (subject, model, mset), sub in df[df["stimulus_type"] == "controversial"].groupby(
        ["subject", "model", "model_set"]
    ):
        # Find best layer for this (subject, model, cstim_set)
        best = sub.loc[sub["rsa"].idxmax()]
        best_layer = best["layer"]
        out_rows.append({
            "subject": subject,
            "model_set": mset,
            "model": model,
            "display_name": MODEL_DISPLAY_NAMES.get(model, model),
            "stimulus_type": "controversial",
            "bootstrap_idx": 0,
            "n_stimuli": int(best["n_stimuli"]),
            "crsa": float(best["rsa"]),
            "layer": best_layer,
        })

        # Vicco rows: same subject/model/best_layer, all bootstraps
        vicco = df[(df["subject"] == subject)
                   & (df["model"] == model)
                   & (df["stimulus_type"] == "vicco")
                   & (df["layer"] == best_layer)]
        for _, vrow in vicco.iterrows():
            out_rows.append({
                "subject": subject,
                "model_set": mset,
                "model": model,
                "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                "stimulus_type": "vicco",
                "bootstrap_idx": int(vrow["bootstrap_idx"]),
                "n_stimuli": int(vrow["n_stimuli"]),
                "crsa": float(vrow["rsa"]),
                "layer": best_layer,
            })

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["subject", "model_set", "model", "stimulus_type", "bootstrap_idx"])
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows -> {OUT_CSV}")
    print("Best-layer choice diversity (counts):")
    chosen = out[out["stimulus_type"] == "controversial"].groupby(["model", "layer"]).size().unstack(fill_value=0)
    print(chosen.to_string())


if __name__ == "__main__":
    main()
