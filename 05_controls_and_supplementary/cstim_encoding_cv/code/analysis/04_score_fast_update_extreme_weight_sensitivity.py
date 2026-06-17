#!/usr/bin/env python3
"""Run CSTIM-dominant fast weighted-update sensitivity scores.

This uses the canonical fast weighted-update scorer with fixed DeepVision
normalization and fixed DeepVision-selected alphas. It writes separate outputs
so the canonical weight grid stays unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import _paths  # noqa: F401
from _paths import RESULTS_DIR

import numpy as np
import pandas as pd

from cstims.target_adaptation import Selection, atomic_write_csv, parse_weights
from srp_utils import FEATURE_PROTOCOL, SRP_TARGET_DIM


DEFAULT_WEIGHTS = "4700"
DEFAULT_N_VICCO_BOOT = 1000
DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 42

SCORE_CSV = RESULTS_DIR / "target_adaptation_fast_extreme_weight_scores.csv"
SUMMARY_CSV = RESULTS_DIR / "target_adaptation_fast_extreme_weight_summary.csv"
META_JSON = RESULTS_DIR / "target_adaptation_fast_extreme_weight_metadata.json"


def apply_output_tag(tag: str | None) -> None:
    global SCORE_CSV, SUMMARY_CSV, META_JSON
    if not tag:
        return
    safe = str(tag).replace("/", "_").replace(" ", "_")
    stem = f"target_adaptation_fast_extreme_weight_{safe}"
    SCORE_CSV = RESULTS_DIR / f"{stem}_scores.csv"
    SUMMARY_CSV = RESULTS_DIR / f"{stem}_summary.csv"
    META_JSON = RESULTS_DIR / f"{stem}_metadata.json"


def load_main_scorer_module():
    path = Path(__file__).with_name("02_score_target_adaptation_srp5920_per_voxel_alpha.py")
    spec = importlib.util.spec_from_file_location("_target_adaptation_main_scorer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = load_main_scorer_module()


def expected_rows_for_selection(model: str, *, n_weights: int) -> int:
    return SCORER.expected_rows_for_selection(
        model,
        n_weights=n_weights,
        score_membership_only=True,
    )


def completed_selection_keys(
    existing: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    weights: list[float],
) -> set[tuple[str, str]]:
    if existing.empty:
        return set()
    required = {
        "subject",
        "model",
        "target_weight",
        "feature_dim_analysis",
        "feature_protocol",
    }
    if required.difference(existing.columns):
        return set()
    completed = set()
    for row in selections.itertuples(index=False):
        block = existing[
            existing["subject"].eq(row.subject) & existing["model"].eq(row.model)
        ]
        expected = expected_rows_for_selection(row.model, n_weights=len(weights))
        ok = (
            len(block) == expected
            and block["feature_dim_analysis"].eq(SRP_TARGET_DIM).all()
            and block["feature_protocol"].eq(FEATURE_PROTOCOL).all()
        )
        if not ok:
            continue
        have_weights = sorted(float(w) for w in block["target_weight"].unique())
        if len(have_weights) == len(weights) and all(
            any(np.isclose(weight, have) for have in have_weights) for weight in weights
        ):
            completed.add((str(row.subject), str(row.model)))
    return completed


def write_summary(scores: pd.DataFrame) -> None:
    rows = []
    group_cols = ["model_set", "adaptation_target", "eval_target", "target_weight"]
    for keys, block in scores.groupby(group_cols):
        vals = block["mrsa_loso"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        deltas = block["delta_vs_original"].to_numpy(dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "mean_mrsa": float(vals.mean()) if len(vals) else np.nan,
                "sem_mrsa": float(vals.std(ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1
                else np.nan,
                "mean_delta_vs_original": float(deltas.mean()) if len(deltas) else np.nan,
                "n": int(len(block)),
                "n_models": int(block["model"].nunique()),
                "n_subjects": int(block["subject"].nunique()),
            }
        )
    atomic_write_csv(pd.DataFrame(rows), SUMMARY_CSV)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--subject", default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-selections", type=int, default=None)
    parser.add_argument("--n-vicco-boot", type=int, default=DEFAULT_N_VICCO_BOOT)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional suffix for sharded outputs, e.g. sub-01.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    apply_output_tag(args.output_tag)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    weights = parse_weights(args.weights)

    if SCORE_CSV.exists() and not args.overwrite and not args.resume:
        print(f"[cached] {SCORE_CSV} exists; use --overwrite to recompute", flush=True)
        return

    selections = SCORER.load_best_shared_selections()
    if args.subject != "all":
        selections = selections[selections["subject"].eq(args.subject)].copy()
    if args.models:
        selections = selections[selections["model"].isin(args.models)].copy()
    selections = selections.sort_values(["subject", "model"]).reset_index(drop=True)
    if args.max_selections:
        selections = selections.head(args.max_selections).copy()
    if selections.empty:
        raise RuntimeError("No selected subject/model/layer rows after filtering")

    original_ref = SCORER.load_original_reference()
    rows = []
    completed: set[tuple[str, str]] = set()
    if args.resume and SCORE_CSV.exists():
        existing = pd.read_csv(SCORE_CSV)
        completed = completed_selection_keys(existing, selections, weights=weights)
        if completed:
            rows = existing[
                existing.apply(
                    lambda row: (str(row["subject"]), str(row["model"])) in completed,
                    axis=1,
                )
            ].to_dict("records")
        print(
            f"[resume] keeping {len(rows)} rows from {len(completed)} complete selections",
            flush=True,
        )

    print(
        "CSTIM-dominant fast weighted update: "
        f"weights={','.join(f'{w:g}' for w in weights)} "
        f"selections={len(selections)} n_vicco_boot={args.n_vicco_boot}",
        flush=True,
    )

    total = len(selections)
    for idx, row in enumerate(selections.itertuples(index=False), start=1):
        sel = Selection(
            subject=row.subject,
            model=row.model,
            display_name=row.display_name,
            layer=row.layer,
        )
        if (sel.subject, sel.model) in completed:
            continue
        print(
            f"[{idx:03d}/{total:03d}] {sel.subject} {sel.model} layer={sel.layer}",
            flush=True,
        )
        new_rows = SCORER.compute_one_selection(
            sel,
            target_weights=weights,
            original_ref=original_ref,
            score_membership_only=True,
            n_folds=args.n_folds,
            seed=args.seed,
            overwrite_alpha=False,
            n_vicco_boot=args.n_vicco_boot,
        )
        for item in new_rows:
            item["sensitivity_scope"] = "fast_weighted_update_cstim_dominant"
            item["sensitivity_note"] = (
                "Fixed DeepVision preprocessing and alphas; target weight makes "
                "CSTIM dominate the weighted ridge objective."
            )
        rows.extend(new_rows)
        scores = pd.DataFrame(rows)
        atomic_write_csv(scores, SCORE_CSV)
        write_summary(scores)
        cstim = scores[
            scores["eval_target"].eq("cstim_loso")
            & scores["model_set"].eq("sota")
            & np.isclose(scores["target_weight"].astype(float), weights[-1])
        ]
        if not cstim.empty:
            print(
                f"  checkpoint rows={len(scores)}; "
                f"SOTA CSTIM mean delta={cstim['delta_vs_original'].mean():+.4f}",
                flush=True,
            )
        else:
            print(f"  checkpoint rows={len(scores)}", flush=True)

    scores = pd.DataFrame(rows)
    atomic_write_csv(scores, SCORE_CSV)
    write_summary(scores)
    meta = {
        "weights": weights,
        "n_selections": int(len(selections)),
        "n_vicco_boot": int(args.n_vicco_boot),
        "feature_dim_analysis": int(SRP_TARGET_DIM),
        "feature_protocol": FEATURE_PROTOCOL,
        "source_scorer": str(Path(SCORER.__file__).resolve()),
        "scheme": "fast weighted update with fixed DeepVision preprocessing and alphas",
        "interpretation": (
            "Weights much larger than 47 make CSTIM dominate total weighted loss; "
            "DeepVision still sets preprocessing, alphas, and contributes a small "
            "residual training term."
        ),
        "python": os.sys.executable,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    META_JSON.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(scores)} rows -> {SCORE_CSV}", flush=True)
    print(f"Wrote summary -> {SUMMARY_CSV}", flush=True)
    print(f"Wrote metadata -> {META_JSON}", flush=True)


if __name__ == "__main__":
    main()
