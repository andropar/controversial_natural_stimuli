#!/usr/bin/env python3
"""Visualize dense-layer mRSA transfer for layer-selection rules.

This figure compares layer choices against the paper layer:
    - best_on_shared: layer selected on DeepVision shared stimuli
    - best_on_cstim: layer selected on the same cstim set being evaluated

For shared/Vicco evaluation, best_on_cstim is averaged across the five
cstim-set-specific oracle selections for each subject/model before summary.
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import MODEL_DISPLAY_NAMES
from style import apply_style, DPI, FONT


apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "results"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"
PNG_DIR = FIG_DIR / "png"

TRANSFER_CSV = DATA_DIR / "mrsa_dense_layer_selection_transfer.csv"
SUMMARY_CSV = DATA_DIR / "mrsa_selection_transfer_delta_summary.csv"

CSTIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective"]
CONDITIONS = [
    ("shared", "deepvision_shared", "Shared"),
    ("vicco", "vicco", "Vicco"),
    ("cstim", "all_models", "Cstim\nall"),
    ("cstim", "architecture", "Cstim\narch."),
    ("cstim", "dataset", "Cstim\ndataset"),
    ("cstim", "sota", "Cstim\nSOTA"),
    ("cstim", "training_objective", "Cstim\nobjective"),
]
RULE_ORDER = ["best_on_shared", "best_on_cstim"]
RULE_LABELS = {
    "best_on_shared": "Best on shared",
    "best_on_cstim": "Best on cstim",
}
RULE_COLORS = {
    "best_on_shared": "#2A9D8F",
    "best_on_cstim": "#E76F51",
}


def sem(values: pd.Series) -> float:
    vals = values.astype(float).dropna()
    if len(vals) <= 1:
        return np.nan
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def condition_label(eval_target: str, eval_model_set: str) -> str:
    for target, model_set, label in CONDITIONS:
        if target == eval_target and model_set == eval_model_set:
            return label
    return f"{eval_target}\n{eval_model_set}"


def build_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject", "model", "eval_target", "eval_model_set"]
    paper = (
        df[df["selection_rule"].eq("paper_layer")]
        [keys + ["mrsa_mean"]]
        .rename(columns={"mrsa_mean": "paper_mrsa"})
    )

    best_shared = df[df["selection_rule"].eq("best_on_shared")].copy()
    best_shared = best_shared[keys + ["selection_rule", "mrsa_mean"]]

    cstim = df[df["selection_rule"].eq("best_on_cstim")].copy()
    cstim_match = cstim[
        cstim["eval_target"].eq("cstim")
        & cstim["selection_model_set"].eq(cstim["eval_model_set"])
    ][keys + ["selection_rule", "mrsa_mean"]]

    cstim_id = cstim[cstim["eval_target"].isin(["shared", "vicco"])]
    cstim_id = (
        cstim_id.groupby(keys, as_index=False)["mrsa_mean"]
        .mean()
        .assign(selection_rule="best_on_cstim")
    )

    selected = pd.concat([best_shared, cstim_match, cstim_id], ignore_index=True)
    out = selected.merge(paper, on=keys, how="left", validate="many_to_one")
    if out["paper_mrsa"].isna().any():
        missing = out[out["paper_mrsa"].isna()][keys].drop_duplicates()
        raise RuntimeError(f"Missing paper-layer baseline rows:\n{missing}")

    out["delta_mrsa"] = out["mrsa_mean"] - out["paper_mrsa"]
    out["condition"] = [
        condition_label(t, s) for t, s in zip(out["eval_target"], out["eval_model_set"])
    ]
    out["condition_order"] = [
        CONDITIONS.index((t, s, condition_label(t, s)))
        for t, s in zip(out["eval_target"], out["eval_model_set"])
    ]
    out["rule_label"] = out["selection_rule"].map(RULE_LABELS)
    out["display_name"] = out["model"].map(MODEL_DISPLAY_NAMES).fillna(out["model"])
    return out


def summarize_for_plot(delta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_means = (
        delta.groupby(
            [
                "model", "display_name", "selection_rule", "rule_label",
                "condition", "condition_order", "eval_target", "eval_model_set",
            ],
            as_index=False,
        )["delta_mrsa"]
        .mean()
    )
    summary = (
        model_means.groupby(
            ["selection_rule", "rule_label", "condition", "condition_order", "eval_target", "eval_model_set"],
            as_index=False,
        )
        .agg(
            delta_mean=("delta_mrsa", "mean"),
            delta_sem=("delta_mrsa", sem),
            n_models=("model", "nunique"),
        )
        .sort_values(["condition_order", "selection_rule"])
    )
    return model_means, summary


def plot_delta(model_means: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    y = np.arange(len(CONDITIONS))
    offsets = {"best_on_shared": -0.13, "best_on_cstim": 0.13}

    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for rule in RULE_ORDER:
        sub = summary[summary["selection_rule"].eq(rule)].set_index("condition_order")
        xs = np.array([sub.loc[i, "delta_mean"] for i in range(len(CONDITIONS))])
        es = np.array([sub.loc[i, "delta_sem"] for i in range(len(CONDITIONS))])
        ypos = y + offsets[rule]
        ax.errorbar(
            xs,
            ypos,
            xerr=es,
            fmt="o",
            markersize=4.0,
            capsize=2.2,
            elinewidth=0.8,
            markerfacecolor=RULE_COLORS[rule],
            markeredgecolor="black",
            markeredgewidth=0.4,
            ecolor=RULE_COLORS[rule],
            label=RULE_LABELS[rule],
            zorder=3,
        )

    ax.axvline(0, color="black", lw=0.65)
    ax.axhline(1.5, color="0.7", lw=0.6, ls="--")
    ax.text(
        0.99,
        0.86,
        "In-distribution",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=FONT["annotation"],
        color="0.3",
    )
    ax.text(
        0.99,
        0.58,
        "Controversial stimuli",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=FONT["annotation"],
        color="0.3",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([label.replace("\n", " ") for _, _, label in CONDITIONS],
                       fontsize=FONT["tick"])
    ax.invert_yaxis()
    xmax = float((summary["delta_mean"] + summary["delta_sem"].fillna(0)).max())
    ax.set_xlim(-0.006, max(0.12, xmax + 0.015))
    ax.set_xlabel("mRSA gain over paper layer", fontsize=FONT["axis_label"])
    ax.set_title("Dense-layer selection transfer", fontsize=FONT["title"])
    ax.legend(loc="lower right", frameon=False, fontsize=FONT["legend"])
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    out_pdf = FIG_DIR / "mrsa_selection_transfer_delta.pdf"
    out_png = PNG_DIR / "mrsa_selection_transfer_delta.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=DPI)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def main() -> None:
    df = pd.read_csv(TRANSFER_CSV)
    delta = build_delta_table(df)
    model_means, summary = summarize_for_plot(delta)
    plot_delta(model_means, summary)
    print(
        summary.pivot(index="condition", columns="selection_rule", values="delta_mean")
        .loc[[label for _, _, label in CONDITIONS]]
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
