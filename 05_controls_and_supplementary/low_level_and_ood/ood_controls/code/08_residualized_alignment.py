#!/usr/bin/env python3
"""Residualized alignment-vs-OOD control on model-level summaries.

This uses the existing model-level table `ood_vs_alignment.csv`. The unit of
analysis is model within model set, not RDM entry. It asks whether the
controversial-minus-baseline alignment delta is still negative after projecting
out measured PPCA feature/prediction OOD deltas and model-set fixed effects.

Output:
  05_controls_and_supplementary/low_level_and_ood/ood_controls/results/ood_residualization_results.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


STAGE = Path(__file__).resolve().parents[1]
DATA = STAGE / "results"
OUT = DATA / "ood_residualization_results.csv"


def design_matrix(df: pd.DataFrame, include_set_effects: bool = True) -> tuple[np.ndarray, list[str]]:
    cols = ["delta_ood_feature", "delta_ood_pred"]
    x = df[cols].to_numpy(dtype=float)
    names = cols.copy()
    if include_set_effects:
        dummies = pd.get_dummies(df["group"], prefix="set", drop_first=True, dtype=float)
        x = np.column_stack([x, dummies.to_numpy(dtype=float)])
        names.extend(dummies.columns.tolist())
    return x, names


def fit(df: pd.DataFrame, include_set_effects: bool) -> dict:
    x, names = design_matrix(df, include_set_effects=include_set_effects)
    y = df["delta_alignment"].to_numpy(dtype=float)
    model = LinearRegression(fit_intercept=True).fit(x, y)
    yhat = model.predict(x)
    resid = y - yhat
    out = {
        "model_spec": "delta_alignment ~ delta_ood_feature + delta_ood_pred"
        + (" + model_set_fixed_effects" if include_set_effects else ""),
        "n_rows": len(df),
        "r2": float(model.score(x, y)),
        "intercept": float(model.intercept_),
        "mean_observed_delta": float(y.mean()),
        "mean_residual": float(resid.mean()),
    }
    for name, coef in zip(names, model.coef_):
        out[f"coef_{name}"] = float(coef)
    return out


def bootstrap_intercept(df: pd.DataFrame, include_set_effects: bool, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        vals.append(fit(df.iloc[idx].reset_index(drop=True), include_set_effects)["intercept"])
    return tuple(np.percentile(vals, [2.5, 97.5]))


def main() -> None:
    df = pd.read_csv(DATA / "ood_vs_alignment.csv").dropna()
    rows = []
    for include_set_effects in [False, True]:
        row = fit(df, include_set_effects)
        lo, hi = bootstrap_intercept(df, include_set_effects)
        row["intercept_ci95_lo_boot_rows"] = float(lo)
        row["intercept_ci95_hi_boot_rows"] = float(hi)
        row["unit_of_analysis"] = "model_within_model_set"
        row["interpretation"] = (
            "negative intercept/mean delta indicates an alignment decrease not captured by measured PPCA OOD axes"
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
