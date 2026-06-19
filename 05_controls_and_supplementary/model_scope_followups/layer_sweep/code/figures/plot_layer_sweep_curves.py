#!/usr/bin/env python3
"""Layer-sweep curves: RSA vs layer depth, per (model x stimulus_set).

x = layer_depth_rank (early -> late)
y = mean RSA across subjects (with SEM band)
two lines per panel: vicco baseline (mean of rsa_vicco_mean across subjects)
                     cstim (rsa_cstim across subjects)

One row per model, one column per cstim_set. Vicco curve is the same in every
column for a given model.

Saves PDF + PNG to figures/.
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cstims import paths
PAPER_ROOT = paths.paper_root()
from cstims.paper.style_improved import apply_style, FONT
from layers_config import MAIN_LAYER, MODEL_LAYERS

apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "results"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"

CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
MODEL_ORDER = list(MODEL_LAYERS.keys())

# Family groupings for readability (each family becomes its own PDF).
FAMILY_GROUPS = {
    "imagenet_supervised": [
        "torchvision_resnet50_imagenet1k_v1",
        "torchvision_vgg16_imagenet1k_v1",
        "torchvision_convnext_base_imagenet1k_v1",
        "torchvision_vit_l_16_imagenet1k_v1",
        "cornet_s",
    ],
    "ssl_resnets": [
        "vissl_resnet50_supervised",
        "vissl_resnet50_barlowtwins",
        "vissl_resnet50_mocov2",
        "vicreg_resnet50",
        "robustness_imagenet_l2_eps3",
    ],
    "transformers_sota": [
        "dinov2_vitl14",
        "slip_vit_l_slip",
        "slip_vit_l_simclr",
        "timm_vit_large_patch14_clip_224_laion2b",
        "openclip_vit_so400m_14_siglip_webli",
    ],
    "transformers_dataset": [
        "openclip_vit_l_14_quickgelu_metaclip_400m",
        "openclip_vit_l_14_quickgelu_metaclip_fullcc",
        "timm_vit_large_patch14_clip_224_dfn2b",
        "timm_vit_large_patch14_clip_quickgelu_224_openai",
        "openclip_vit_l_14_laion400m_e31",
    ],
}


def _render_for_models(df, models, out_pdf, out_png):
    n_rows = len(models)
    n_cols = len(CSTIM_SETS)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.4 * n_cols, 1.9 * n_rows),
                              sharex=False, sharey="row")
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i, model in enumerate(models):
        n_layers = len(MODEL_LAYERS[model])
        main_idx = [n for n, _ in MODEL_LAYERS[model]].index(MAIN_LAYER[model])

        for j, mset in enumerate(CSTIM_SETS):
            ax = axes[i, j]
            sub = df[(df["model"] == model) & (df["model_set"] == mset)].sort_values("layer_depth_rank")
            if sub.empty:
                ax.set_visible(False)
                continue
            x = sub["layer_depth_rank"].values
            y_v = sub["rsa_vicco_mean_mean"].values
            y_v_s = sub["rsa_vicco_mean_sem"].values
            y_c = sub["rsa_cstim_mean"].values
            y_c_s = sub["rsa_cstim_sem"].values

            ax.plot(x, y_v, "-o", color="tab:gray", lw=1, ms=3, label="vicco")
            ax.fill_between(x, y_v - y_v_s, y_v + y_v_s, color="tab:gray", alpha=0.2, lw=0)
            ax.plot(x, y_c, "-o", color="tab:red", lw=1, ms=3, label=mset)
            ax.fill_between(x, y_c - y_c_s, y_c + y_c_s, color="tab:red", alpha=0.2, lw=0)

            ax.axvline(main_idx, color="k", lw=0.6, ls="--", alpha=0.7)
            ax.axhline(0, color="black", lw=0.4, alpha=0.5)

            if i == 0:
                ax.set_title(mset, fontsize=FONT["title"])
            if j == 0:
                from cstims.constants import MODEL_DISPLAY_NAMES
                disp = MODEL_DISPLAY_NAMES.get(model, model)
                ax.set_ylabel(f"{disp}\nRSA", fontsize=FONT["axis_label"])
            # Show layer-name x-ticks on every row (since layers differ per model)
            ax.set_xticks(range(n_layers))
            short = [ln[:18] + "…" if len(ln) > 18 else ln for ln in [n for n, _ in MODEL_LAYERS[model]]]
            ax.set_xticklabels(short, rotation=70, fontsize=FONT["small"])
            if i == n_rows - 1:
                ax.set_xlabel("layer depth", fontsize=FONT["axis_label"])

    handles = [
        plt.Line2D([], [], color="tab:gray", marker="o", ms=3, lw=1, label="vicco baseline"),
        plt.Line2D([], [], color="tab:red", marker="o", ms=3, lw=1, label="controversial"),
        plt.Line2D([], [], color="k", lw=0.6, ls="--", label="paper layer"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               fontsize=FONT["legend"], frameon=False,
               bbox_to_anchor=(0.5, 1.005))

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


def main():
    df = pd.read_csv(DATA_DIR / "layer_drop_summary_subject_avg.csv")

    # Per-family figures
    for family, models in FAMILY_GROUPS.items():
        models = [m for m in models if m in MODEL_LAYERS]
        if not models:
            continue
        out_pdf = FIG_DIR / f"layer_sweep_curves_{family}.pdf"
        out_png = FIG_DIR / f"layer_sweep_curves_{family}.png"
        _render_for_models(df, models, out_pdf, out_png)

    # Single all-models figure (kept for completeness)
    _render_for_models(
        df, MODEL_ORDER,
        FIG_DIR / "layer_sweep_curves.pdf",
        FIG_DIR / "layer_sweep_curves.png",
    )


if __name__ == "__main__":
    main()
