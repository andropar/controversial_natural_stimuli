#!/usr/bin/env python3
"""Build a 300-pair anchor-balanced diagnostic labeling queue."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
SHARE_ROOT = APP_ROOT.parents[2]
DEFAULT_PAIR_SUMMARY = (
    SHARE_ROOT
    / "05_controls_and_supplementary"
    / "stimulus_and_pair_diagnostics"
    / "pair_level_brain_placement"
    / "results"
    / "pair_level_brain_placement_summary.csv"
)
DEFAULT_OUT = APP_ROOT / "results" / "pair_queue_anchor_balanced_300.csv"


def pair_key(i: int, j: int) -> tuple[int, int]:
    return (int(min(i, j)), int(max(i, j)))


def pairs_for_anchor(df: pd.DataFrame, anchor: int, selected: set[tuple[int, int]]) -> pd.DataFrame:
    mask = (df["img_i"] == anchor) | (df["img_j"] == anchor)
    out = df[mask].copy()
    out["_key"] = [pair_key(i, j) for i, j in zip(out["img_i"], out["img_j"])]
    out = out[~out["_key"].isin(selected)].copy()
    return out


def add_row(
    rows: list[dict[str, object]],
    selected: set[tuple[int, int]],
    row: pd.Series,
    *,
    queue_group: str,
    anchor: int,
    rank_within_anchor: int,
) -> None:
    key = pair_key(int(row["img_i"]), int(row["img_j"]))
    selected.add(key)
    rows.append(
        {
            "queue_index": len(rows),
            "queue_group": queue_group,
            "anchor_img": int(anchor),
            "rank_within_anchor": int(rank_within_anchor),
            "img_i": key[0],
            "img_j": key[1],
            "mismatch_score": float(row["mismatch_score"]),
            "mean_brain_z_position": float(row["mean_brain_z_position"]),
            "mean_brain_z": float(row["mean_brain_z"]),
            "sd_brain_z": float(row["sd_brain_z"]),
            "mean_brain_percentile": float(row["mean_brain_percentile"]),
            "model_spread_z": float(row["model_spread_z"]),
            "n_subjects": int(row["n_subjects"]),
        }
    )


def build_queue(df: pd.DataFrame, *, n_images: int, seed: int) -> pd.DataFrame:
    required = {
        "img_i",
        "img_j",
        "mean_brain_z",
        "sd_brain_z",
        "mean_brain_percentile",
        "mean_brain_z_position",
        "model_spread_z",
        "n_subjects",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Pair summary missing required columns: {sorted(missing)}")

    df = df.copy()
    df["img_i"] = df["img_i"].astype(int)
    df["img_j"] = df["img_j"].astype(int)
    df["mismatch_score"] = df["mean_brain_z_position"].abs()
    df = df[np.isfinite(df["mismatch_score"])].copy()
    df = df[df["n_subjects"] > 0].copy()

    anchors = list(range(n_images))
    rng = np.random.default_rng(seed)
    selected: set[tuple[int, int]] = set()
    rows: list[dict[str, object]] = []

    for anchor in anchors:
        candidates = pairs_for_anchor(df, anchor, selected)
        if candidates.empty:
            raise RuntimeError(f"No available worst-pair candidate for anchor {anchor}")
        candidates = candidates.sort_values(
            ["mismatch_score", "model_spread_z"],
            ascending=[False, False],
        )
        add_row(
            rows,
            selected,
            candidates.iloc[0],
            queue_group="worst_mismatch_per_anchor",
            anchor=anchor,
            rank_within_anchor=1,
        )

    worst_by_anchor = {
        int(row["anchor_img"]): row
        for row in rows
        if row["queue_group"] == "worst_mismatch_per_anchor"
    }
    spread_scale = float(df["model_spread_z"].quantile(0.75) - df["model_spread_z"].quantile(0.25))
    if spread_scale <= 0 or not np.isfinite(spread_scale):
        spread_scale = float(df["model_spread_z"].std())
    spread_scale = spread_scale if spread_scale > 0 else 1.0

    for anchor in anchors:
        worst = worst_by_anchor[anchor]
        candidates = pairs_for_anchor(df, anchor, selected)
        if candidates.empty:
            raise RuntimeError(f"No available control-pair candidate for anchor {anchor}")
        candidates = candidates.copy()
        candidates["spread_delta_scaled"] = (
            candidates["model_spread_z"] - float(worst["model_spread_z"])
        ).abs() / spread_scale
        # Prefer low model-brain mismatch while keeping model-spread close enough
        # that controls are not trivial low-disagreement pairs.
        candidates["control_score"] = candidates["mismatch_score"] + 0.35 * candidates["spread_delta_scaled"]
        candidates = candidates.sort_values(
            ["control_score", "mismatch_score", "spread_delta_scaled"],
            ascending=[True, True, True],
        )
        add_row(
            rows,
            selected,
            candidates.iloc[0],
            queue_group="spread_matched_low_mismatch_control",
            anchor=anchor,
            rank_within_anchor=1,
        )

    for anchor in anchors:
        candidates = pairs_for_anchor(df, anchor, selected)
        if candidates.empty:
            raise RuntimeError(f"No available random-pair candidate for anchor {anchor}")
        chosen_pos = int(rng.integers(0, len(candidates)))
        add_row(
            rows,
            selected,
            candidates.iloc[chosen_pos],
            queue_group="random_anchor_balanced",
            anchor=anchor,
            rank_within_anchor=1,
        )

    out = pd.DataFrame(rows)
    if len(out) != 3 * n_images:
        raise RuntimeError(f"Expected {3 * n_images} rows, got {len(out)}")
    if out[["img_i", "img_j"]].duplicated().any():
        raise RuntimeError("Queue contains duplicate image pairs")
    group_order = {
        "worst_mismatch_per_anchor": 0,
        "spread_matched_low_mismatch_control": 1,
        "random_anchor_balanced": 2,
    }
    out["_group_order"] = out["queue_group"].map(group_order).astype(int)
    out = out.sort_values(["anchor_img", "_group_order"]).reset_index(drop=True)
    out["queue_index"] = np.arange(len(out), dtype=int)
    out = out.drop(columns=["_group_order"])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-summary", type=Path, default=DEFAULT_PAIR_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260609)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.pair_summary)
    queue = build_queue(df, n_images=args.n_images, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.out, index=False)

    summary = (
        queue.groupby("queue_group")
        .agg(
            n_pairs=("queue_group", "size"),
            mean_mismatch=("mismatch_score", "mean"),
            median_mismatch=("mismatch_score", "median"),
            mean_model_spread=("model_spread_z", "mean"),
            median_model_spread=("model_spread_z", "median"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
