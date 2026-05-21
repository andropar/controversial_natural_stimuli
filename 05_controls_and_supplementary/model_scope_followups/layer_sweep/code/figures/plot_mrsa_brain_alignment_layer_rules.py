#!/usr/bin/env python3
"""Check whether layer selection changes the main mRSA alignment conclusions.

This is a compact mRSA-only counterpart to brain_alignment_improved.pdf. It
summarizes two quantities from the original figure:
    1. cstim - Vicco alignment drop
    2. cstim / Vicco between-model spread ratio

Both are computed for paper_layer, best_on_shared, and best_on_cstim.
"""

import itertools

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import MODEL_SETS as CONFIG_MODEL_SETS
from style import apply_style, DPI, FONT


apply_style()

DATA_DIR = LAYER_SWEEP_ROOT / "data"
FIG_DIR = LAYER_SWEEP_ROOT / "figures"
PNG_DIR = FIG_DIR / "png"

TRANSFER_CSV = DATA_DIR / "mrsa_dense_layer_selection_transfer.csv"
SUMMARY_CSV = DATA_DIR / "mrsa_brain_alignment_layer_rule_summary.csv"

SET_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
MODEL_SET_LABELS = {
    "all_models": "All",
    "sota": "SOTA",
    "training_objective": "Objective",
    "architecture": "Architecture",
    "dataset": "Dataset",
}
RULES = ["paper_layer", "best_on_shared", "best_on_cstim"]
RULE_LABELS = {
    "paper_layer": "Paper layer",
    "best_on_shared": "Best on shared",
    "best_on_cstim": "Best on cstim",
}
RULE_COLORS = {
    "paper_layer": "#6E6E6E",
    "best_on_shared": "#2A9D8F",
    "best_on_cstim": "#E76F51",
}


def sem(vals: pd.Series) -> float:
    vals = vals.astype(float).dropna()
    if len(vals) <= 1:
        return np.nan
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def median_pairwise_diff(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan
    return float(np.median([abs(a - b) for a, b in itertools.combinations(vals, 2)]))


def select_rows(df: pd.DataFrame, model_set: str, rule: str, eval_target: str) -> pd.DataFrame:
    if eval_target == "cstim":
        eval_model_set = model_set
    elif eval_target == "vicco":
        eval_model_set = "vicco"
    else:
        raise ValueError(eval_target)

    out = df[
        df["selection_rule"].eq(rule)
        & df["eval_target"].eq(eval_target)
        & df["eval_model_set"].eq(eval_model_set)
    ].copy()

    if rule == "paper_layer":
        out = out[out["selection_model_set"].eq("paper_layer")]
    elif rule == "best_on_shared":
        out = out[out["selection_model_set"].eq("deepvision_shared")]
    elif rule == "best_on_cstim":
        out = out[out["selection_model_set"].eq(model_set)]
    else:
        raise ValueError(rule)
    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["subject", "model", "display_name"]
    for model_set in SET_ORDER:
        for rule in RULES:
            cstim = select_rows(df, model_set, rule, "cstim")
            vicco = select_rows(df, model_set, rule, "vicco")
            pair = (
                cstim[keys + ["mrsa_mean"]]
                .rename(columns={"mrsa_mean": "cstim_mrsa"})
                .merge(
                    vicco[keys + ["mrsa_mean"]].rename(columns={"mrsa_mean": "vicco_mrsa"}),
                    on=keys,
                    how="inner",
                    validate="one_to_one",
                )
            )
            pair["drop"] = pair["cstim_mrsa"] - pair["vicco_mrsa"]
            allowed_models = set(CONFIG_MODEL_SETS[model_set])
            pair = pair[pair["model"].isin(allowed_models)]

            model_means = (
                pair.groupby(["model", "display_name"], as_index=False)
                .agg(
                    cstim_mrsa=("cstim_mrsa", "mean"),
                    vicco_mrsa=("vicco_mrsa", "mean"),
                    drop=("drop", "mean"),
                )
            )
            cstim_spread = median_pairwise_diff(model_means["cstim_mrsa"].to_numpy())
            vicco_spread = median_pairwise_diff(model_means["vicco_mrsa"].to_numpy())
            rows.append({
                "model_set": model_set,
                "selection_rule": rule,
                "selection_label": RULE_LABELS[rule],
                "n_models": model_means["model"].nunique(),
                "mean_cstim_mrsa": float(model_means["cstim_mrsa"].mean()),
                "mean_vicco_mrsa": float(model_means["vicco_mrsa"].mean()),
                "mean_drop": float(model_means["drop"].mean()),
                "sem_drop": sem(model_means["drop"]),
                "cstim_spread": cstim_spread,
                "vicco_spread": vicco_spread,
                "spread_ratio": cstim_spread / vicco_spread if vicco_spread > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    x = np.arange(len(SET_ORDER))
    width = 0.23
    offsets = {
        "paper_layer": -width,
        "best_on_shared": 0.0,
        "best_on_cstim": width,
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))

    ax = axes[0]
    for rule in RULES:
        sub = summary[summary["selection_rule"].eq(rule)].set_index("model_set")
        vals = [sub.loc[s, "mean_drop"] for s in SET_ORDER]
        errs = [sub.loc[s, "sem_drop"] for s in SET_ORDER]
        ax.bar(
            x + offsets[rule],
            vals,
            width=width,
            yerr=errs,
            capsize=2,
            color=RULE_COLORS[rule],
            edgecolor="black",
            linewidth=0.4,
            label=RULE_LABELS[rule],
        )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SET_LABELS[s] for s in SET_ORDER],
                       rotation=25, ha="right", fontsize=FONT["tick"])
    ax.set_ylabel("cstim - Vicco mRSA", fontsize=FONT["axis_label"])
    ax.set_title("Alignment drop", fontsize=FONT["title"])

    ax = axes[1]
    for rule in RULES:
        sub = summary[summary["selection_rule"].eq(rule)].set_index("model_set")
        vals = [sub.loc[s, "spread_ratio"] for s in SET_ORDER]
        ax.bar(
            x + offsets[rule],
            vals,
            width=width,
            color=RULE_COLORS[rule],
            edgecolor="black",
            linewidth=0.4,
            label=RULE_LABELS[rule],
        )
    ax.axhline(1, color="black", linewidth=0.6, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SET_LABELS[s] for s in SET_ORDER],
                       rotation=25, ha="right", fontsize=FONT["tick"])
    ax.set_ylabel("model spread ratio\ncstim / Vicco", fontsize=FONT["axis_label"])
    ax.set_title("Between-model spread", fontsize=FONT["title"])
    ax.legend(loc="upper right", frameon=False, fontsize=FONT["legend"])

    fig.suptitle("Does dense-layer selection change the mRSA brain-alignment conclusions?",
                 fontsize=FONT["title"], y=1.02)
    fig.tight_layout()

    out_pdf = FIG_DIR / "mrsa_brain_alignment_layer_rules.pdf"
    out_png = PNG_DIR / "mrsa_brain_alignment_layer_rules.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=DPI)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def main() -> None:
    df = pd.read_csv(TRANSFER_CSV)
    summary = build_summary(df)
    plot_summary(summary)
    cols = ["model_set", "selection_rule", "mean_drop", "spread_ratio"]
    print(summary[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
