#!/usr/bin/env python3
"""Summarize model-RDM agreement for selected and random stimulus sets.

The selection-evaluation pipeline already writes cross-model RDM correlation
matrices for the raw feature track and each subject-specific encoding track:

    results/<model_set>_unique_boot/correlation_matrices.csv

This script converts those matrices into a compact Hosseini-style diagnostic:
the mean off-diagonal correlation between model RDMs within a stimulus set.

Outputs:
    results/model_rdm_agreement_pairs.csv
    results/model_rdm_agreement_track_summary.csv
    results/model_rdm_agreement_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT = Path(__file__).resolve()
SHARE_ROOT = SCRIPT.parents[4]
HELPERS = SHARE_ROOT / "src"
sys.path.insert(0, str(HELPERS))

from cstims import constants, paths
from cstims.paper.style_improved import MODEL_SET_ORDER


DATA_SUFFIX = "_unique_boot"
MATRIX_TYPES = {
    "random_clean": "random",
    "selected_clean": "controversial",
}
TRACK_TO_REPRESENTATION = {
    "raw": "fixed_raw",
    "sub-01": "mixed_encoding",
    "sub-03": "mixed_encoding",
    "sub-05": "mixed_encoding",
    "sub-06": "mixed_encoding",
    "sub-07": "mixed_encoding",
}
TRACK_ORDER = ["raw", "sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]


def _matrix_path(model_set: str) -> Path:
    return paths.selection_evaluation_results_dir() / f"{model_set}{DATA_SUFFIX}" / "correlation_matrices.csv"


def _offdiag_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one undirected off-diagonal entry per model pair."""
    out = df[df["model_i"] != df["model_j"]].copy()
    lo = np.minimum(out["model_i"].astype(str), out["model_j"].astype(str))
    hi = np.maximum(out["model_i"].astype(str), out["model_j"].astype(str))
    out["model_a"] = lo
    out["model_b"] = hi
    out = out.drop_duplicates(["track", "matrix_type", "model_a", "model_b"])
    return out


def build_pair_table(model_sets: list[str]) -> pd.DataFrame:
    rows = []
    for model_set in model_sets:
        path = _matrix_path(model_set)
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        df = pd.read_csv(path)
        df = df[df["matrix_type"].isin(MATRIX_TYPES)].copy()
        df = df[df["track"].isin(TRACK_TO_REPRESENTATION)].copy()
        df = _offdiag_pairs(df)
        n_models = int(len(set(df["model_i"]) | set(df["model_j"])))
        for row in df.itertuples(index=False):
            rows.append(
                {
                    "model_set": model_set,
                    "representation": TRACK_TO_REPRESENTATION[row.track],
                    "track": row.track,
                    "subject": row.track if row.track.startswith("sub-") else "",
                    "subset_type": MATRIX_TYPES[row.matrix_type],
                    "matrix_type": row.matrix_type,
                    "model_a": row.model_a,
                    "model_b": row.model_b,
                    "correlation": float(row.correlation),
                    "n_models_in_matrix": n_models,
                    "source_csv": str(path),
                }
            )
    return pd.DataFrame(rows)


def summarize_tracks(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    grouped = pairs.groupby(
        ["model_set", "representation", "track", "subject", "subset_type"],
        dropna=False,
    )
    rows = []
    for keys, grp in grouped:
        values = grp["correlation"].astype(float).to_numpy()
        rows.append(
            {
                "model_set": keys[0],
                "representation": keys[1],
                "track": keys[2],
                "subject": keys[3],
                "subset_type": keys[4],
                "n_model_pairs": int(len(values)),
                "n_models": int(grp["n_models_in_matrix"].iloc[0]),
                "mean_pairwise_r": float(np.mean(values)),
                "median_pairwise_r": float(np.median(values)),
                "sd_pairwise_r": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "sem_pairwise_r": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "p10_pairwise_r": float(np.quantile(values, 0.10)),
                "p90_pairwise_r": float(np.quantile(values, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def _paired_test(deltas: np.ndarray) -> dict[str, float]:
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < 2:
        return {
            "paired_t": np.nan,
            "paired_t_p": np.nan,
            "wilcoxon_p": np.nan,
            "n_positive_delta": int(np.sum(deltas > 0)),
            "n_negative_delta": int(np.sum(deltas < 0)),
        }
    t_stat, t_p = stats.ttest_1samp(deltas, 0.0)
    try:
        _, w_p = stats.wilcoxon(deltas)
    except ValueError:
        w_p = np.nan
    return {
        "paired_t": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "n_positive_delta": int(np.sum(deltas > 0)),
        "n_negative_delta": int(np.sum(deltas < 0)),
    }


def summarize_representations(track_summary: pd.DataFrame) -> pd.DataFrame:
    if track_summary.empty:
        return pd.DataFrame()
    rows = []
    for (model_set, representation), grp in track_summary.groupby(["model_set", "representation"]):
        wide = grp.pivot_table(
            index=["track", "subject", "n_models", "n_model_pairs"],
            columns="subset_type",
            values=["mean_pairwise_r", "median_pairwise_r"],
            aggfunc="first",
        )
        wide.columns = [f"{metric}_{subset}" for metric, subset in wide.columns]
        wide = wide.reset_index()
        if "mean_pairwise_r_random" not in wide or "mean_pairwise_r_controversial" not in wide:
            continue

        wide["delta_mean_r"] = wide["mean_pairwise_r_controversial"] - wide["mean_pairwise_r_random"]
        wide["delta_median_r"] = wide["median_pairwise_r_controversial"] - wide["median_pairwise_r_random"]
        wide["pct_delta_mean_r"] = 100.0 * wide["delta_mean_r"] / wide["mean_pairwise_r_random"].replace(0, np.nan)

        if representation == "mixed_encoding":
            unit_label = "subject_track"
        else:
            unit_label = "raw_track_descriptive"

        deltas = wide["delta_mean_r"].to_numpy(dtype=float)
        test = _paired_test(deltas)
        rows.append(
            {
                "model_set": model_set,
                "representation": representation,
                "inference_unit": unit_label,
                "n_units": int(len(wide)),
                "n_models": int(wide["n_models"].iloc[0]),
                "n_model_pairs": int(wide["n_model_pairs"].iloc[0]),
                "random_mean_pairwise_r": float(wide["mean_pairwise_r_random"].mean()),
                "controversial_mean_pairwise_r": float(wide["mean_pairwise_r_controversial"].mean()),
                "delta_mean_pairwise_r": float(wide["delta_mean_r"].mean()),
                "delta_mean_pairwise_r_sem": float(wide["delta_mean_r"].std(ddof=1) / np.sqrt(len(wide))) if len(wide) > 1 else 0.0,
                "pct_delta_mean_pairwise_r": float(wide["pct_delta_mean_r"].mean()),
                "random_median_pairwise_r": float(wide["median_pairwise_r_random"].mean()),
                "controversial_median_pairwise_r": float(wide["median_pairwise_r_controversial"].mean()),
                "delta_median_pairwise_r": float(wide["delta_median_r"].mean()),
                **test,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = paths.selection_evaluation_results_dir()
    pairs = build_pair_table(MODEL_SET_ORDER)
    track_summary = summarize_tracks(pairs)
    summary = summarize_representations(track_summary)

    pairs.to_csv(out_dir / "model_rdm_agreement_pairs.csv", index=False)
    track_summary.to_csv(out_dir / "model_rdm_agreement_track_summary.csv", index=False)
    summary.to_csv(out_dir / "model_rdm_agreement_summary.csv", index=False)

    print(f"Saved {out_dir / 'model_rdm_agreement_pairs.csv'}")
    print(f"Saved {out_dir / 'model_rdm_agreement_track_summary.csv'}")
    print(f"Saved {out_dir / 'model_rdm_agreement_summary.csv'}")
    if not summary.empty:
        cols = [
            "model_set",
            "representation",
            "random_mean_pairwise_r",
            "controversial_mean_pairwise_r",
            "delta_mean_pairwise_r",
            "pct_delta_mean_pairwise_r",
        ]
        print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
