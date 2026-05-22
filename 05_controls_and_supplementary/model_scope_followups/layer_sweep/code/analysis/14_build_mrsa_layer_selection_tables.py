#!/usr/bin/env python3
"""Build dense mRSA layer-score and selection-transfer tables.

Inputs:
    data/wrsa_dense_layer_sweep.csv
    data/wrsa_dense_shared_layer_sweep.csv

Outputs:
    data/mrsa_dense_all_eval_layer_scores.csv
        Per (subject, model, layer, eval target, bootstrap) score table.

    data/mrsa_dense_layer_selection_transfer.csv
        Compact table for paper layer, best-on-shared, and best-on-cstim
        selections, evaluated on shared, cstim sets, and Vicco.
"""

import _paths  # noqa: F401
from _paths import LAYER_SWEEP_ROOT
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import MODEL_DISPLAY_NAMES, PAPER_ROOT
from layers_config import MAIN_LAYER, STIMULUS_SETS, get_layer_set


DATA_DIR = LAYER_SWEEP_ROOT / "results"
CSTIM_SETS = [s for s in STIMULUS_SETS if s != "vicco"]
SHARED_SET = "deepvision_shared"


def layer_index_maps(layer_specs):
    idx = {}
    frac = {}
    for model, specs in layer_specs.items():
        names = [name for name, _ in specs]
        denom = max(len(names) - 1, 1)
        idx[model] = {name: i for i, name in enumerate(names)}
        frac[model] = {name: i / denom for i, name in enumerate(names)}
    return idx, frac


def normalize_layer_scores(wrsa_csv: Path, shared_csv: Path) -> pd.DataFrame:
    wrsa = pd.read_csv(wrsa_csv)
    shared = pd.read_csv(shared_csv)

    cstim = wrsa[wrsa["stimulus_type"].eq("controversial")].copy()
    cstim["eval_target"] = "cstim"

    vicco = wrsa[wrsa["stimulus_type"].eq("vicco")].copy()
    vicco["eval_target"] = "vicco"

    shared = shared.copy()
    shared["eval_target"] = "shared"

    out = pd.concat([cstim, vicco, shared], ignore_index=True)
    out = out.rename(columns={"rsa": "mrsa"})
    cols = [
        "subject", "model", "display_name", "layer",
        "eval_target", "model_set", "stimulus_type",
        "bootstrap_idx", "n_stimuli", "mrsa",
    ]
    return out[cols].sort_values(
        ["subject", "model", "layer", "eval_target", "model_set", "bootstrap_idx"],
        kind="stable",
    )


def choose_best_layer(df, group_cols):
    """Choose max layer after averaging over bootstrap rows when present."""
    means = (
        df.groupby([*group_cols, "layer"], as_index=False)["mrsa"]
        .mean()
    )
    idx = means.groupby(group_cols, sort=False)["mrsa"].idxmax()
    return means.loc[idx].rename(columns={"mrsa": "selection_mrsa"})


def make_selections(scores: pd.DataFrame, layer_specs):
    models = list(layer_specs.keys())
    rows = []

    for subject in sorted(scores["subject"].unique()):
        for model in models:
            if model not in MAIN_LAYER:
                continue
            rows.append({
                "subject": subject,
                "model": model,
                "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                "selection_rule": "paper_layer",
                "selection_model_set": "paper_layer",
                "selected_layer": MAIN_LAYER[model],
                "selection_mrsa": np.nan,
            })

    shared = scores[scores["eval_target"].eq("shared")]
    best_shared = choose_best_layer(shared, ["subject", "model"])
    for _, r in best_shared.iterrows():
        rows.append({
            "subject": r["subject"],
            "model": r["model"],
            "display_name": MODEL_DISPLAY_NAMES.get(r["model"], r["model"]),
            "selection_rule": "best_on_shared",
            "selection_model_set": SHARED_SET,
            "selected_layer": r["layer"],
            "selection_mrsa": float(r["selection_mrsa"]),
        })

    cstim = scores[scores["eval_target"].eq("cstim")]
    best_cstim = choose_best_layer(cstim, ["subject", "model", "model_set"])
    for _, r in best_cstim.iterrows():
        rows.append({
            "subject": r["subject"],
            "model": r["model"],
            "display_name": MODEL_DISPLAY_NAMES.get(r["model"], r["model"]),
            "selection_rule": "best_on_cstim",
            "selection_model_set": r["model_set"],
            "selected_layer": r["layer"],
            "selection_mrsa": float(r["selection_mrsa"]),
        })

    return pd.DataFrame(rows)


def summarize_eval(group: pd.DataFrame) -> pd.Series:
    vals = group["mrsa"].astype(float)
    n = len(vals)
    sem = vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return pd.Series({
        "n_bootstraps": n,
        "n_stimuli": int(group["n_stimuli"].iloc[0]),
        "mrsa_mean": float(vals.mean()),
        "mrsa_sem": float(sem) if np.isfinite(sem) else np.nan,
    })


def build_transfer_table(scores: pd.DataFrame, selections: pd.DataFrame, layer_specs) -> pd.DataFrame:
    idx_map, frac_map = layer_index_maps(layer_specs)
    eval_summary = (
        scores.groupby(
            ["subject", "model", "layer", "eval_target", "model_set"],
            as_index=False,
        )
        .apply(summarize_eval, include_groups=False)
        .reset_index(drop=True)
    )

    rows = []
    for _, sel in selections.iterrows():
        sub = eval_summary[
            eval_summary["subject"].eq(sel["subject"])
            & eval_summary["model"].eq(sel["model"])
            & eval_summary["layer"].eq(sel["selected_layer"])
        ]
        for _, ev in sub.iterrows():
            rows.append({
                "subject": sel["subject"],
                "model": sel["model"],
                "display_name": sel["display_name"],
                "selection_rule": sel["selection_rule"],
                "selection_model_set": sel["selection_model_set"],
                "selected_layer": sel["selected_layer"],
                "selected_layer_index": idx_map.get(sel["model"], {}).get(sel["selected_layer"], np.nan),
                "selected_layer_frac": frac_map.get(sel["model"], {}).get(sel["selected_layer"], np.nan),
                "selection_mrsa": sel["selection_mrsa"],
                "eval_target": ev["eval_target"],
                "eval_model_set": ev["model_set"],
                "n_bootstraps": int(ev["n_bootstraps"]),
                "n_stimuli": int(ev["n_stimuli"]),
                "mrsa_mean": float(ev["mrsa_mean"]),
                "mrsa_sem": ev["mrsa_sem"],
            })

    out = pd.DataFrame(rows)
    return out.sort_values(
        [
            "subject", "model", "selection_rule", "selection_model_set",
            "eval_target", "eval_model_set",
        ],
        kind="stable",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-set", choices=["configured", "dense"], default="dense")
    parser.add_argument("--wrsa-csv", default=str(DATA_DIR / "wrsa_dense_layer_sweep.csv"))
    parser.add_argument("--shared-csv", default=str(DATA_DIR / "wrsa_dense_shared_layer_sweep.csv"))
    parser.add_argument("--layer-scores-csv",
                        default=str(DATA_DIR / "mrsa_dense_all_eval_layer_scores.csv"))
    parser.add_argument("--selection-transfer-csv",
                        default=str(DATA_DIR / "mrsa_dense_layer_selection_transfer.csv"))
    args = parser.parse_args()

    layer_specs = get_layer_set(args.layer_set)
    scores = normalize_layer_scores(Path(args.wrsa_csv), Path(args.shared_csv))
    Path(args.layer_scores_csv).parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.layer_scores_csv, index=False)
    print(f"Wrote {len(scores)} rows -> {args.layer_scores_csv}")

    selections = make_selections(scores, layer_specs)
    transfer = build_transfer_table(scores, selections, layer_specs)
    transfer.to_csv(args.selection_transfer_csv, index=False)
    print(f"Wrote {len(transfer)} rows -> {args.selection_transfer_csv}")
    print()
    print("Selection counts:")
    print(selections.groupby(["selection_rule", "selection_model_set"]).size().to_string())


if __name__ == "__main__":
    main()
