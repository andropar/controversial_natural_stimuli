#!/usr/bin/env python3
"""Combined rescue summary: mRSA-transfer (top) and fRSA (bottom) per
(model, cstim_set). Subject-averaged, error bars = SEM across subjects.
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PAPER_ROOT, MODEL_DISPLAY_NAMES
from style import apply_style, FONT
from layers_config import MODEL_LAYERS

apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "results"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"
CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
MODEL_ORDER = list(MODEL_LAYERS.keys())


def main():
    mrsa = pd.read_csv(DATA_DIR / "mrsa_layer_rescue_summary_subject_avg.csv")
    frsa = pd.read_csv(DATA_DIR / "layer_rescue_summary_subject_avg.csv")

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.0), sharex=True)
    n_models = len(MODEL_ORDER)
    n_sets = len(CSTIM_SETS)
    width = 0.13
    x = np.arange(n_models)
    cmap = plt.get_cmap("tab10")
    set_colors = {s: cmap(i / max(1, n_sets - 1)) for i, s in enumerate(CSTIM_SETS)}

    for row, (label, df) in enumerate([("Mixed RSA (mRSA-transfer)", mrsa),
                                        ("Fixed RSA (fRSA)", frsa)]):
        ax = axes[row]
        for k, mset in enumerate(CSTIM_SETS):
            offsets = (k - (n_sets - 1) / 2) * width
            ys, ye = [], []
            for m in MODEL_ORDER:
                row_d = df[(df["model"] == m) & (df["model_set"] == mset)]
                ys.append(float(row_d["rescue_mean"].iloc[0]) if len(row_d) else np.nan)
                ye.append(float(row_d["rescue_sem"].iloc[0]) if len(row_d) else np.nan)
            ax.bar(x + offsets, ys, width=width, yerr=ye, color=set_colors[mset],
                   edgecolor="black", linewidth=0.4, capsize=1.5,
                   label=mset if row == 0 else None)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(f"{label}\nrescue", fontsize=FONT["axis_label"])
        if row == 0:
            ax.legend(loc="upper right", ncol=5, fontsize=FONT["legend"], frameon=False)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER],
                            rotation=45, ha="right", fontsize=FONT["tick"])
    fig.suptitle("Layer-rescue per model × stimulus set\n"
                 "Top: mixed RSA (paper's primary metric). Bottom: fixed RSA.",
                 fontsize=FONT["title"])
    fig.tight_layout()
    out_pdf = FIG_DIR / "rescue_summary_combined.pdf"
    out_png = FIG_DIR / "rescue_summary_combined.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


if __name__ == "__main__":
    main()
