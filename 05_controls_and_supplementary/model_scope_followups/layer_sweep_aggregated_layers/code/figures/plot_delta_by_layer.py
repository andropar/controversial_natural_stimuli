#!/usr/bin/env python3
"""delta_rsa = rsa_cstim - rsa_vicco_mean per (model, layer, set).

One row per model, one column per cstim_set, x = layer depth, y = delta_rsa.
Horizontal zero line.
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PAPER_ROOT
from style import apply_style, FONT
from layers_config import MAIN_LAYER, MODEL_LAYERS

apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "results"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"
CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
MODEL_ORDER = list(MODEL_LAYERS.keys())

CSTIM_COLORS = {
    "all_models": "#d62728",
    "architecture": "#1f77b4",
    "dataset": "#2ca02c",
    "sota": "#9467bd",
    "training_objective": "#ff7f0e",
}


def main():
    df = pd.read_csv(DATA_DIR / "layer_drop_summary_subject_avg.csv")

    from config import MODEL_DISPLAY_NAMES
    n = len(MODEL_ORDER)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows),
                              sharey=False)
    axes = axes.flatten()

    for i, model in enumerate(MODEL_ORDER):
        ax = axes[i]
        layer_names = [n for n, _ in MODEL_LAYERS[model]]
        main_idx = layer_names.index(MAIN_LAYER[model])

        for mset in CSTIM_SETS:
            sub = df[(df["model"] == model) & (df["model_set"] == mset)].sort_values("layer_depth_rank")
            if sub.empty:
                continue
            x = sub["layer_depth_rank"].values
            y = sub["delta_rsa_mean"].values
            e = sub["delta_rsa_sem"].values
            ax.errorbar(x, y, yerr=e, fmt="-o", lw=1, ms=3, capsize=2,
                        color=CSTIM_COLORS[mset], label=mset, alpha=0.85)

        ax.axhline(0, color="black", lw=0.6, ls="-")
        ax.axvline(main_idx, color="k", lw=0.6, ls="--", alpha=0.7)
        ax.set_title(MODEL_DISPLAY_NAMES.get(model, model),
                     fontsize=FONT["title"])
        ax.set_xticks(range(len(layer_names)))
        # Show layer names but tighter on long ones
        short = [ln[:18] + "…" if len(ln) > 18 else ln for ln in layer_names]
        ax.set_xticklabels(short, rotation=70, fontsize=FONT["small"])
        if i % n_cols == 0:
            ax.set_ylabel("Δ RSA  (cstim − vicco)", fontsize=FONT["axis_label"])
    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(CSTIM_SETS),
               fontsize=FONT["legend"], frameon=False, bbox_to_anchor=(0.5, 1.005))

    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    out_pdf = FIG_DIR / "delta_by_layer.pdf"
    out_png = FIG_DIR / "delta_by_layer.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


if __name__ == "__main__":
    main()
