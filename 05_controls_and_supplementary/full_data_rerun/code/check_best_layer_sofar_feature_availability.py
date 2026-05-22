#!/usr/bin/env python3
"""Check whether selected-layer-so-far features are available for full-data refits."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
RERUN_ROOT = THIS.parents[1]
SHARE_ROOT = THIS.parents[3]
LAYER_SWEEP_ROOT = SHARE_ROOT / "05_controls_and_supplementary" / "model_scope_followups" / "layer_sweep"
DEFAULT_SELECTIONS = RERUN_ROOT / "results" / "best_shared_layer_sofar_from_layer_sweep.csv"
DEFAULT_OUT = RERUN_ROOT / "results" / "best_layer_sofar_feature_availability.csv"
DV_FEAT_CACHE = LAYER_SWEEP_ROOT / "cache_or_heavy" / "dv_features"
CSTIM_FEAT_CACHE = LAYER_SWEEP_ROOT / "cache_or_heavy" / "features"
SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
STIM_SETS = ["all_models", "architecture", "dataset", "sota", "training_objective", "vicco"]


def metadata_suffix(layer: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(layer)).strip("_")


def npz_has_key(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            return key in z.files
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sel = pd.read_csv(args.selections)
    pairs = (
        sel[["model", "display_name", "selected_layer", "selection_shared_mrsa_mean"]]
        .drop_duplicates()
        .sort_values(["model", "selected_layer"])
    )
    rows = []
    for _, row in pairs.iterrows():
        model = row["model"]
        layer = row["selected_layer"]
        for subject in SUBJECTS:
            dv_path = DV_FEAT_CACHE / subject / f"{model}.npz"
            rows.append({
                "subject": subject,
                "model": model,
                "display_name": row["display_name"],
                "selected_layer": layer,
                "selection_shared_mrsa_mean": row["selection_shared_mrsa_mean"],
                "feature_role": "deepvision_unique",
                "stimulus_set": "deepvision_unique",
                "path": str(dv_path),
                "available": npz_has_key(dv_path, layer),
            })
        for stim_set in STIM_SETS:
            feat_path = CSTIM_FEAT_CACHE / model / f"{stim_set}.npz"
            rows.append({
                "subject": "",
                "model": model,
                "display_name": row["display_name"],
                "selected_layer": layer,
                "selection_shared_mrsa_mean": row["selection_shared_mrsa_mean"],
                "feature_role": "cstim_eval",
                "stimulus_set": stim_set,
                "path": str(feat_path),
                "available": npz_has_key(feat_path, layer),
            })
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    by_role = out.groupby("feature_role")["available"].agg(["sum", "size"]).reset_index()
    print(by_role.to_string(index=False))
    missing = out[~out["available"]]
    if len(missing):
        print(f"Missing {len(missing)} selected-layer feature entries -> {args.out}")
    else:
        print(f"All selected-layer feature entries available -> {args.out}")


if __name__ == "__main__":
    main()
