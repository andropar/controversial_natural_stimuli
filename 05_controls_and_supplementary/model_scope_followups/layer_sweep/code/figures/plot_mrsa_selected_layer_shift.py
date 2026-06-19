#!/usr/bin/env python3
"""Plot how selected layers shift under shared vs cstim selection."""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cstims.constants import MODEL_DISPLAY_NAMES
from cstims.paper.style_improved import apply_style, DPI, FONT
from layers_config import get_layer_set


apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "results"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"
PNG_DIR = FIG_DIR / "png"

TRANSFER_CSV = DATA_DIR / "mrsa_dense_layer_selection_transfer.csv"
SUMMARY_CSV = DATA_DIR / "mrsa_selected_layer_shift_summary.csv"

CSTIM_SETS = ["all_models", "sota", "training_objective", "architecture", "dataset"]
CSTIM_LABELS = {
    "all_models": "All",
    "sota": "SOTA",
    "training_objective": "Objective",
    "architecture": "Architecture",
    "dataset": "Dataset",
}
COLORS = {
    "paper_layer": "#6E6E6E",
    "best_on_shared": "#2A9D8F",
    "best_on_cstim": "#E76F51",
}


def sem(vals: pd.Series) -> float:
    vals = vals.astype(float).dropna()
    if len(vals) <= 1:
        return 0.0
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def model_label(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def load_unique_selections() -> pd.DataFrame:
    cols = [
        "subject", "model", "display_name", "selection_rule",
        "selection_model_set", "selected_layer", "selected_layer_index",
        "selected_layer_frac",
    ]
    df = pd.read_csv(TRANSFER_CSV, usecols=cols).drop_duplicates()
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in df.groupby("model"):
        display = model_label(model)
        paper = sub[sub["selection_rule"].eq("paper_layer")]
        shared = sub[
            sub["selection_rule"].eq("best_on_shared")
            & sub["selection_model_set"].eq("deepvision_shared")
        ]
        for rule, block, model_set in [
            ("paper_layer", paper, "paper_layer"),
            ("best_on_shared", shared, "deepvision_shared"),
        ]:
            rows.append({
                "model": model,
                "display_name": display,
                "selection_rule": rule,
                "selection_model_set": model_set,
                "layer_frac_mean": float(block["selected_layer_frac"].mean()),
                "layer_frac_sem": sem(block["selected_layer_frac"]),
                "n_subjects": block["subject"].nunique(),
            })

        for cset in CSTIM_SETS:
            cstim = sub[
                sub["selection_rule"].eq("best_on_cstim")
                & sub["selection_model_set"].eq(cset)
            ]
            rows.append({
                "model": model,
                "display_name": display,
                "selection_rule": "best_on_cstim",
                "selection_model_set": cset,
                "layer_frac_mean": float(cstim["selected_layer_frac"].mean()),
                "layer_frac_sem": sem(cstim["selected_layer_frac"]),
                "n_subjects": cstim["subject"].nunique(),
            })
    return pd.DataFrame(rows)


def plot_shift(summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    layer_specs = get_layer_set("dense")
    model_order = list(layer_specs.keys())
    labels = [model_label(m) for m in model_order]
    y = np.arange(len(model_order))

    fig, axes = plt.subplots(1, len(CSTIM_SETS), figsize=(11.8, 6.8), sharey=True)
    if len(CSTIM_SETS) == 1:
        axes = [axes]

    for ax, cset in zip(axes, CSTIM_SETS):
        paper = (
            summary[
                summary["selection_rule"].eq("paper_layer")
                & summary["selection_model_set"].eq("paper_layer")
            ]
            .set_index("model")
        )
        shared = (
            summary[
                summary["selection_rule"].eq("best_on_shared")
                & summary["selection_model_set"].eq("deepvision_shared")
            ]
            .set_index("model")
        )
        cstim = (
            summary[
                summary["selection_rule"].eq("best_on_cstim")
                & summary["selection_model_set"].eq(cset)
            ]
            .set_index("model")
        )

        for i, model in enumerate(model_order):
            p = paper.loc[model, "layer_frac_mean"]
            s = shared.loc[model, "layer_frac_mean"]
            c = cstim.loc[model, "layer_frac_mean"]
            ax.plot([p, s], [i - 0.08, i - 0.08], color=COLORS["best_on_shared"],
                    alpha=0.35, linewidth=0.8)
            ax.plot([p, c], [i + 0.08, i + 0.08], color=COLORS["best_on_cstim"],
                    alpha=0.35, linewidth=0.8)

        ax.scatter(
            [paper.loc[m, "layer_frac_mean"] for m in model_order],
            y,
            marker="|",
            s=70,
            linewidths=1.2,
            color=COLORS["paper_layer"],
            label="Paper layer",
            zorder=4,
        )
        ax.errorbar(
            [shared.loc[m, "layer_frac_mean"] for m in model_order],
            y - 0.08,
            xerr=[shared.loc[m, "layer_frac_sem"] for m in model_order],
            fmt="o",
            markersize=3.3,
            color=COLORS["best_on_shared"],
            ecolor=COLORS["best_on_shared"],
            markeredgecolor="black",
            markeredgewidth=0.3,
            capsize=1.5,
            linewidth=0.7,
            label="Best on shared",
            zorder=5,
        )
        ax.errorbar(
            [cstim.loc[m, "layer_frac_mean"] for m in model_order],
            y + 0.08,
            xerr=[cstim.loc[m, "layer_frac_sem"] for m in model_order],
            fmt="o",
            markersize=3.3,
            color=COLORS["best_on_cstim"],
            ecolor=COLORS["best_on_cstim"],
            markeredgecolor="black",
            markeredgewidth=0.3,
            capsize=1.5,
            linewidth=0.7,
            label="Best on cstim",
            zorder=5,
        )

        ax.set_title(CSTIM_LABELS[cset], fontsize=FONT["title"])
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel("layer depth", fontsize=FONT["axis_label"])
        ax.grid(axis="x", color="0.9", linewidth=0.5)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=FONT["tick"])
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper left", frameon=False, fontsize=FONT["legend"])
    fig.suptitle("Selected dense-layer depth by rule", fontsize=FONT["title"], y=1.01)
    fig.tight_layout()

    out_pdf = FIG_DIR / "mrsa_selected_layer_shift.pdf"
    out_png = PNG_DIR / "mrsa_selected_layer_shift.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=DPI)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def main() -> None:
    selections = load_unique_selections()
    summary = summarize(selections)
    plot_shift(summary)


if __name__ == "__main__":
    main()
