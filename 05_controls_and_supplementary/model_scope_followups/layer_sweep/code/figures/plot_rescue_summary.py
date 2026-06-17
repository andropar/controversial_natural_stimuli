#!/usr/bin/env python3
"""Bar plot of rescue (delta_best - delta_late) per (model, cstim_set).

Subject-averaged, error bars = SEM across subjects.
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cstims.paper.config import PAPER_ROOT
from cstims.paper.style_improved import apply_style, FONT
from layers_config import MODEL_LAYERS

apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "results"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"
CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
MODEL_ORDER = list(MODEL_LAYERS.keys())


def main():
    df = pd.read_csv(DATA_DIR / "layer_rescue_summary_subject_avg.csv")

    fig, ax = plt.subplots(figsize=(13.0, 4.0))

    n_models = len(MODEL_ORDER)
    n_sets = len(CSTIM_SETS)
    width = 0.13
    x = np.arange(n_models)
    cmap = plt.get_cmap("tab10")
    set_colors = {s: cmap(i / max(1, n_sets - 1)) for i, s in enumerate(CSTIM_SETS)}

    for k, mset in enumerate(CSTIM_SETS):
        offsets = (k - (n_sets - 1) / 2) * width
        ys, ye = [], []
        for m in MODEL_ORDER:
            row = df[(df["model"] == m) & (df["model_set"] == mset)]
            ys.append(float(row["rescue_mean"].iloc[0]) if len(row) else np.nan)
            ye.append(float(row["rescue_sem"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + offsets, ys, width=width, yerr=ye, color=set_colors[mset],
               edgecolor="black", linewidth=0.4, capsize=1.5, label=mset)

    ax.axhline(0, color="black", lw=0.6)
    from cstims.paper.config import MODEL_DISPLAY_NAMES
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m.replace("_imagenet1k_v1", "").replace("torchvision_", ""))
                        for m in MODEL_ORDER], rotation=45, ha="right", fontsize=FONT["tick"])
    ax.set_ylabel("Rescue: Δ_best − Δ_paper-layer", fontsize=FONT["axis_label"])
    ax.set_title("Layer-rescue per model × stimulus set\n"
                 "Positive = some other layer reduces the cstim alignment drop "
                 "vs the paper's late-layer choice",
                 fontsize=FONT["title"])
    ax.legend(loc="upper right", ncol=2, fontsize=FONT["legend"], frameon=False)
    fig.tight_layout()
    out_pdf = FIG_DIR / "rescue_summary.pdf"
    out_png = FIG_DIR / "rescue_summary.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


if __name__ == "__main__":
    main()
