#!/usr/bin/env python3
"""Summarize best shared-selected layer from currently available layer-sweep parts.

This is a bookkeeping table for "best layer we have so far" based on the
ongoing dense layer-sweep stream outputs. It does not recompute scores on the
full-data cstim cache; it records the currently available layer-sweep values so
the rerun directory has a stable snapshot of the selection state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
RERUN_ROOT = THIS.parents[1]
SHARE_ROOT = THIS.parents[3]
LAYER_SWEEP = (
    SHARE_ROOT
    / "05_controls_and_supplementary"
    / "model_scope_followups"
    / "layer_sweep"
)
PART_ROOT = LAYER_SWEEP / "results" / "stream_parts"
DEFAULT_OUT = RERUN_ROOT / "results" / "best_shared_layer_sofar_from_layer_sweep.csv"


def read_parts(part_dir: Path) -> pd.DataFrame:
    parts = sorted(part_dir.glob("*.csv"))
    if not parts:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-root", type=Path, default=PART_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    shared = read_parts(args.part_root / "wrsa_dense_shared_layer_sweep")
    cstim = read_parts(args.part_root / "wrsa_dense_layer_sweep")
    if shared.empty:
        raise SystemExit("No shared layer-sweep stream parts found.")

    shared_mean = (
        shared.groupby(["subject", "model", "display_name", "layer"], as_index=False)["rsa"]
        .mean()
        .rename(columns={"rsa": "shared_mrsa_mean"})
    )
    idx = shared_mean.groupby(["subject", "model"], sort=False)["shared_mrsa_mean"].idxmax()
    best = shared_mean.loc[idx].rename(columns={"layer": "selected_layer"})
    best["selection_rule"] = "best_shared_layer_sofar"

    rows = []
    if not cstim.empty:
        eval_mean = (
            cstim.groupby(
                ["subject", "model", "display_name", "layer", "model_set", "stimulus_type"],
                as_index=False,
            )["rsa"]
            .mean()
            .rename(columns={"rsa": "eval_mrsa_mean"})
        )
        for _, sel in best.iterrows():
            sub = eval_mean[
                eval_mean["subject"].eq(sel["subject"])
                & eval_mean["model"].eq(sel["model"])
                & eval_mean["layer"].eq(sel["selected_layer"])
            ]
            for _, ev in sub.iterrows():
                rows.append({
                    "subject": sel["subject"],
                    "model": sel["model"],
                    "display_name": sel["display_name"],
                    "selection_rule": sel["selection_rule"],
                    "selected_layer": sel["selected_layer"],
                    "selection_shared_mrsa_mean": sel["shared_mrsa_mean"],
                    "eval_source": "existing_dense_layer_sweep_stream_parts",
                    "eval_model_set": ev["model_set"],
                    "eval_stimulus_type": ev["stimulus_type"],
                    "eval_mrsa_mean": ev["eval_mrsa_mean"],
                })

    out = pd.DataFrame(rows)
    if out.empty:
        out = best.copy()
        out["eval_source"] = "existing_dense_layer_sweep_stream_parts"
        out["eval_model_set"] = np.nan
        out["eval_stimulus_type"] = np.nan
        out["eval_mrsa_mean"] = np.nan

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows -> {args.out}")
    print("Available subjects:", ", ".join(sorted(out["subject"].dropna().unique())))
    print("Available models:", out["model"].nunique())


if __name__ == "__main__":
    main()
