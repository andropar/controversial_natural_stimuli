#!/usr/bin/env python3
"""mRSA-transfer layer rescue summary (analogue to 03/04 for fRSA).

Outputs:
    11_layer_sweep/data/mrsa_layer_drop_summary.csv
    11_layer_sweep/data/mrsa_layer_rescue_summary.csv
    11_layer_sweep/data/mrsa_layer_drop_summary_subject_avg.csv
    11_layer_sweep/data/mrsa_layer_rescue_summary_subject_avg.csv
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd

from config import PAPER_ROOT
from layers_config import MAIN_LAYER, MODEL_LAYERS, layer_depth_rank, layer_depth_frac

DATA_DIR = LAYER_SWEEP_ROOT / "data"
IN_CSV = DATA_DIR / "wrsa_layer_sweep.csv"


def main():
    df = pd.read_csv(IN_CSV)

    vicco = df[df["stimulus_type"] == "vicco"]
    vicco_agg = (
        vicco.groupby(["subject", "model", "display_name", "layer"], as_index=False)
        .agg(rsa_vicco_mean=("rsa", "mean"),
             rsa_vicco_std=("rsa", "std"),
             n_vicco_boot=("rsa", "size"))
    )

    cstim = df[df["stimulus_type"] == "controversial"][[
        "subject", "model", "display_name", "layer", "model_set", "rsa"
    ]].rename(columns={"rsa": "rsa_cstim"})

    merged = cstim.merge(vicco_agg, on=["subject", "model", "display_name", "layer"], how="left")
    merged["delta_rsa"] = merged["rsa_cstim"] - merged["rsa_vicco_mean"]
    merged["layer_depth_rank"] = [
        layer_depth_rank(m, l) for m, l in zip(merged["model"], merged["layer"])
    ]
    merged["layer_depth_frac"] = [
        layer_depth_frac(m, l) for m, l in zip(merged["model"], merged["layer"])
    ]
    merged.to_csv(DATA_DIR / "mrsa_layer_drop_summary.csv", index=False)
    print(f"wrote {len(merged)} rows -> mrsa_layer_drop_summary.csv")

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
    avg.to_csv(DATA_DIR / "mrsa_layer_drop_summary_subject_avg.csv", index=False)

    # Rescue per (subject, model, set)
    rows = []
    for (subj, model, mset), sub in merged.groupby(["subject", "model", "model_set"]):
        L_paper = MAIN_LAYER[model]
        late_row = sub[sub["layer"] == L_paper]
        if late_row.empty:
            continue
        late_row = late_row.iloc[0]
        best_d = sub.loc[sub["delta_rsa"].idxmax()]
        rows.append({
            "subject": subj, "model": model,
            "display_name": late_row["display_name"],
            "model_set": mset,
            "paper_layer": L_paper,
            "delta_paper": late_row["delta_rsa"],
            "rsa_cstim_paper": late_row["rsa_cstim"],
            "rsa_vicco_paper": late_row["rsa_vicco_mean"],
            "best_layer_delta": best_d["layer"],
            "best_layer_delta_depth": best_d["layer_depth_rank"],
            "delta_best": best_d["delta_rsa"],
            "rescue": best_d["delta_rsa"] - late_row["delta_rsa"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "mrsa_layer_rescue_summary.csv", index=False)
    print(f"wrote {len(out)} rows -> mrsa_layer_rescue_summary.csv")

    avg2 = (out.groupby(["model", "display_name", "model_set"], as_index=False)
            .agg(delta_paper_mean=("delta_paper", "mean"),
                 delta_paper_sem=("delta_paper", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                 delta_best_mean=("delta_best", "mean"),
                 delta_best_sem=("delta_best", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                 rescue_mean=("rescue", "mean"),
                 rescue_sem=("rescue", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                 n_subjects=("rescue", "size")))
    avg2.to_csv(DATA_DIR / "mrsa_layer_rescue_summary_subject_avg.csv", index=False)
    print(f"wrote {len(avg2)} rows -> mrsa_layer_rescue_summary_subject_avg.csv")

    # Set-level summary
    setlevel = (out.groupby("model_set", as_index=False)
                .agg(rescue_mean=("rescue", "mean"),
                     rescue_sem=("rescue", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                     delta_paper_mean=("delta_paper", "mean"),
                     delta_best_mean=("delta_best", "mean"),
                     n=("rescue", "size")))
    print("\nmRSA-transfer set-level summary:")
    print(setlevel.to_string(index=False))


if __name__ == "__main__":
    main()
