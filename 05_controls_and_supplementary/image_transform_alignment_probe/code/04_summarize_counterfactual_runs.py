#!/usr/bin/env python3
"""Summarize counterfactual prototype runs into one comparison table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ANALYSIS_DIR / "results"


def add_pixel_rows(rows: list[dict]) -> None:
    for path in sorted(RESULTS_DIR.glob("resnet50_pair_0022_0046_sub-06*summary.csv")):
        name = path.name
        if "lowfreq" in name:
            continue
        frame = pd.read_csv(path)
        row = frame.iloc[0].to_dict()
        run_label = row.get("run_label", "")
        if pd.isna(run_label) or run_label == "":
            run_label = "pixel_default"
        rows.append(
            {
                "method": "pixel",
                "run_label": run_label,
                "target_mode": "raw",
                "target_z": row["brain_target_z"],
                "original_z": row["live_original_z"],
                "best_or_final_z": row["final_z"],
                "abs_error": row["final_alignment_abs_error"],
                "original_abs_error": row["original_alignment_abs_error"],
                "pixel_rmse": None,
                "notes": "unconstrained pixel optimization",
                "summary_path": str(path),
            }
        )


def add_parametric_row(rows: list[dict]) -> None:
    path = RESULTS_DIR / "resnet50_pair_0022_0046_sub-06_parametric_grid.csv"
    frame = pd.read_csv(path)
    best = frame.sort_values(["abs_error_quantile_target", "pixel_rmse"]).iloc[0]
    original = frame[frame["family"] == "original"].iloc[0]
    rows.append(
        {
            "method": "parametric_grid",
            "run_label": f"{best['family']}:{best['parameter']}={best['value']}",
            "target_mode": "quantile",
            "target_z": best["quantile_target_model_z"],
            "original_z": original["z_model_distance"],
            "best_or_final_z": best["z_model_distance"],
            "abs_error": best["abs_error_quantile_target"],
            "original_abs_error": original["abs_error_quantile_target"],
            "pixel_rmse": best["pixel_rmse"],
            "notes": "best deterministic edit in grid",
            "summary_path": str(path),
        }
    )


def add_lowfreq_rows(rows: list[dict]) -> None:
    for path in sorted(RESULTS_DIR.glob("resnet50_pair_0022_0046_sub-06_lowfreq*_summary.csv")):
        frame = pd.read_csv(path)
        row = frame.iloc[0].to_dict()
        rows.append(
            {
                "method": "lowfreq_residual",
                "run_label": row["run_label"],
                "target_mode": row["target_mode"],
                "target_z": row["target_z"],
                "original_z": row["live_original_z"],
                "best_or_final_z": row["best_z"],
                "abs_error": row["best_abs_error"],
                "original_abs_error": row["original_abs_error"],
                "pixel_rmse": row["final_pixel_rmse"],
                "notes": f"{row.get('residual_mode', 'rgb')} residual, size={row['residual_size']}",
                "summary_path": str(path),
            }
        )


def main() -> None:
    rows: list[dict] = []
    add_pixel_rows(rows)
    add_parametric_row(rows)
    add_lowfreq_rows(rows)
    out = pd.DataFrame(rows).sort_values(["target_mode", "abs_error", "method"])
    out_path = RESULTS_DIR / "counterfactual_method_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
