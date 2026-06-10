#!/usr/bin/env python3
"""Compare late-layer alignment to best-layer alignment per (model, cstim_set).

For each (subject, model, cstim_set):
    delta_late      = (rsa_cstim_late - rsa_vicco_late_mean)
    delta_best_cstim = max over layers of (rsa_cstim - rsa_vicco_mean)
    rescue          = delta_best_cstim - delta_late

Also reports:
    best_layer_vicco  = argmax_layer rsa_vicco_mean
    best_layer_cstim  = argmax_layer rsa_cstim
    best_layer_delta  = argmax_layer (rsa_cstim - rsa_vicco_mean)

Outputs:
    results/layer_rescue_summary.csv               (per subject)
    results/layer_rescue_summary_subject_avg.csv   (mean +/- SEM)
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd

from config import PAPER_ROOT
from layers_config import MODEL_LAYERS, LATE_LAYER, layer_depth_rank

DATA_DIR = LAYER_SWEEP_ROOT / "results"
IN_CSV = DATA_DIR / "layer_drop_summary.csv"
OUT_CSV = DATA_DIR / "layer_rescue_summary.csv"
OUT_AVG_CSV = DATA_DIR / "layer_rescue_summary_subject_avg.csv"


def main():
    df = pd.read_csv(IN_CSV)

    rows = []
    for (subject, model, model_set), sub in df.groupby(["subject", "model", "model_set"]):
        late = LATE_LAYER[model]
        late_row = sub[sub["layer"] == late]
        if late_row.empty:
            continue
        late_row = late_row.iloc[0]

        # Best-by metrics
        best_v = sub.loc[sub["rsa_vicco_mean"].idxmax()]
        best_c = sub.loc[sub["rsa_cstim"].idxmax()]
        best_d = sub.loc[sub["delta_rsa"].idxmax()]

        rescue = best_d["delta_rsa"] - late_row["delta_rsa"]
        rows.append({
            "subject": subject,
            "model": model,
            "display_name": late_row["display_name"],
            "model_set": model_set,
            "late_layer": late,
            "delta_late": late_row["delta_rsa"],
            "rsa_cstim_late": late_row["rsa_cstim"],
            "rsa_vicco_late": late_row["rsa_vicco_mean"],
            "best_layer_vicco": best_v["layer"],
            "best_layer_vicco_depth": best_v["layer_depth_rank"],
            "rsa_vicco_best": best_v["rsa_vicco_mean"],
            "best_layer_cstim": best_c["layer"],
            "best_layer_cstim_depth": best_c["layer_depth_rank"],
            "rsa_cstim_best": best_c["rsa_cstim"],
            "best_layer_delta": best_d["layer"],
            "best_layer_delta_depth": best_d["layer_depth_rank"],
            "delta_best": best_d["delta_rsa"],
            "rescue": rescue,
        })
    out = pd.DataFrame(rows).sort_values(["model", "model_set", "subject"])
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows -> {OUT_CSV}")

    avg = (out.groupby(["model", "display_name", "model_set"], as_index=False)
           .agg(delta_late_mean=("delta_late", "mean"),
                delta_late_sem=("delta_late", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                delta_best_mean=("delta_best", "mean"),
                delta_best_sem=("delta_best", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                rescue_mean=("rescue", "mean"),
                rescue_sem=("rescue", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                n_subjects=("rescue", "size")))
    avg = avg.sort_values(["model", "model_set"])
    avg.to_csv(OUT_AVG_CSV, index=False)
    print(f"Wrote {len(avg)} subject-averaged rows -> {OUT_AVG_CSV}")


if __name__ == "__main__":
    main()
