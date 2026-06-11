#!/usr/bin/env python3
"""Summarise available selection-track composition controls.

This script does not rerun stimulus selection. It converts the existing
selection-composition leaderboard into paper-facing CSVs and records the
feasibility status of a true leave-one-subject-out selection audit.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[2]
SHARE_ROOT = STAGE.parents[1]
SRC = SHARE_ROOT / "scripts" / "cursor" / "outputs" / "final_aggregate_plot" / "leaderboard_summary.csv"
OUT = STAGE / "results"
OUT.mkdir(parents=True, exist_ok=True)


def parse_method(name: str) -> dict:
    if name == "Random":
        return {
            "composition": "random",
            "selection_distance": "none",
            "track_family": "random",
            "track_subject": "",
        }
    match = re.match(r"^(?P<composition>.+) \((?P<distance>[^)]+)\)$", name)
    if match is None:
        raise ValueError(f"Cannot parse method label: {name}")
    composition = match.group("composition")
    if composition == "raw_plus_all_encodings":
        family, subject = "raw_plus_all_subjects", "all"
    elif composition == "raw_plus_group":
        family, subject = "raw_plus_group_average", "group"
    elif composition == "raw_only":
        family, subject = "raw_only", ""
    elif composition == "group_only":
        family, subject = "group_only", "group"
    elif composition.startswith("raw_plus_sub-"):
        family = "raw_plus_single_subject"
        subject = composition.replace("raw_plus_", "").replace("_hlvis", "")
    else:
        family, subject = "other", ""
    return {
        "composition": composition,
        "selection_distance": match.group("distance"),
        "track_family": family,
        "track_subject": subject,
    }


def build_long_table(df: pd.DataFrame) -> pd.DataFrame:
    parsed = pd.DataFrame([parse_method(v) for v in df["Full Method"]])
    out = pd.concat([df.reset_index(drop=True), parsed], axis=1)
    out = out.rename(columns={"metric": "evaluation_distance", "score_mean": "error_auc_mean", "score_se": "error_auc_se"})
    out["accuracy_auc_mean"] = 1.0 - out["error_auc_mean"]
    out["accuracy_auc_se"] = out["error_auc_se"]
    keep = [
        "evaluation_distance",
        "selection_distance",
        "composition",
        "track_family",
        "track_subject",
        "accuracy_auc_mean",
        "accuracy_auc_se",
        "error_auc_mean",
        "error_auc_se",
        "score_raw",
        "score_encoded_mean",
    ]
    return out[keep].sort_values(["evaluation_distance", "selection_distance", "track_family", "track_subject"])


def build_contribution_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (evaluation_distance, selection_distance), grp in long.groupby(["evaluation_distance", "selection_distance"]):
        all_row = grp[grp["track_family"] == "raw_plus_all_subjects"]
        if all_row.empty:
            continue
        all_acc = float(all_row["accuracy_auc_mean"].iloc[0])
        for family, label in [
            ("raw_only", "raw only"),
            ("group_only", "group encoding only"),
            ("raw_plus_group_average", "raw + group-average encoding"),
        ]:
            base = grp[grp["track_family"] == family]
            if base.empty:
                continue
            base_acc = float(base["accuracy_auc_mean"].iloc[0])
            rows.append(
                {
                    "evaluation_distance": evaluation_distance,
                    "selection_distance": selection_distance,
                    "comparison_baseline": label,
                    "accuracy_auc_all_subjects": all_acc,
                    "accuracy_auc_baseline": base_acc,
                    "delta_accuracy_auc": all_acc - base_acc,
                    "n_baseline_rows": len(base),
                }
            )
        singles = grp[grp["track_family"] == "raw_plus_single_subject"]
        if not singles.empty:
            for label, value in [
                ("mean raw + single-subject encoding", singles["accuracy_auc_mean"].mean()),
                ("best raw + single-subject encoding", singles["accuracy_auc_mean"].max()),
            ]:
                rows.append(
                    {
                        "evaluation_distance": evaluation_distance,
                        "selection_distance": selection_distance,
                        "comparison_baseline": label,
                        "accuracy_auc_all_subjects": all_acc,
                        "accuracy_auc_baseline": float(value),
                        "delta_accuracy_auc": all_acc - float(value),
                        "n_baseline_rows": int(len(singles)),
                    }
                )
    return pd.DataFrame(rows).sort_values(["evaluation_distance", "selection_distance", "comparison_baseline"])


def write_feasibility_note(long: pd.DataFrame) -> None:
    has_loo = long["composition"].str.contains("leave_one|leave-one|loo", case=False, regex=True).any()
    text = [
        "# Leave-One-Subject-Out Selection Audit Feasibility",
        "",
        "Existing leaderboard outputs cover raw-only, group-only, raw plus group-average encoding, raw plus each single-subject encoding, and raw plus all subject encodings.",
        "",
    ]
    if has_loo:
        text.append("A leave-one-subject-out composition appears in the leaderboard and can be summarised from `selection_track_composition_summary.csv`.")
    else:
        text.extend(
            [
                "No true leave-one-subject-out optimized selection output is present in `scripts/cursor/outputs/final_aggregate_plot/leaderboard_summary.csv` or in the final selection output directories inspected for this revision.",
                "",
                "A true LOO audit would require rerunning the stimulus-selection optimizer five times per model set with one subject-specific encoding track omitted. It cannot be inferred from the available raw-plus-single-subject selections because those selections were independently optimized and do not represent the same objective with one track removed.",
            ]
        )
    (OUT / "selection_leave_one_subject_out_feasibility.md").write_text("\n".join(text) + "\n")


def main() -> None:
    df = pd.read_csv(SRC)
    long = build_long_table(df)
    contribution = build_contribution_table(long)
    long.to_csv(OUT / "selection_track_composition_summary.csv", index=False)
    contribution.to_csv(OUT / "selection_track_contribution_summary.csv", index=False)
    write_feasibility_note(long)
    print(f"wrote selection-track audit outputs in {OUT}")
    print(contribution.replace({np.nan: ""}).to_string(index=False))


if __name__ == "__main__":
    main()
