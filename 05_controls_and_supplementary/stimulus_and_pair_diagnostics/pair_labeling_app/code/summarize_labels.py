#!/usr/bin/env python3
"""Summarize pair-label CSV completion and label counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = APP_ROOT / "results" / "pair_labels.csv"
DEFAULT_OUT = APP_ROOT / "results" / "pair_label_counts.csv"
DEFAULT_QUEUE = APP_ROOT / "results" / "pair_queue_anchor_balanced_300.csv"

LABEL_FIELDS = [
    "semantic_similarity",
    "visual_surface_similarity",
    "shape_layout_similarity",
    "scene_context_similarity",
    "dominant_relation",
    "confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--all-labels", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.labels)
    if not args.all_labels and args.queue.exists():
        queue = pd.read_csv(args.queue, usecols=["img_i", "img_j", "queue_group", "anchor_img"])
        df = queue.merge(df, on=["img_i", "img_j"], how="left")
    complete = df[LABEL_FIELDS].notna().all(axis=1) & (df[LABEL_FIELDS] != "").all(axis=1)
    rows = [
        {"field": "_all", "value": "rows", "n": int(len(df))},
        {"field": "_all", "value": "complete_rows", "n": int(complete.sum())},
    ]
    for field in LABEL_FIELDS:
        counts = df[field].fillna("").replace("", "missing").value_counts(dropna=False)
        rows.extend(
            {"field": field, "value": str(value), "n": int(n)}
            for value, n in counts.items()
        )
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
