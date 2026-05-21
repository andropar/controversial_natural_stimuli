#!/usr/bin/env python3
"""Compute selected-vs-random values of the selection objective.

This is an optimization diagnostic, complementary to the model-recovery
discriminability analysis. It evaluates already-computed cross-model RDM
correlation matrices with the same margin used during selection:

    u_i = C_ii - mean_{j != i} C_ij
    U   = min_i u_i

For noised matrices produced by ``compute_correlation_at_target_noise``, the
noise is applied to the second matrix argument, so model-specific margins are
computed column-wise. Clean matrices are symmetric, so row/column orientation is
irrelevant there.

Outputs:
    data/selection_objective_by_model.csv
    data/selection_objective_summary.csv
    data/selection_objective_combined.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

PAPER_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PAPER_ROOT / "figures"))

import config
from style_improved import MODEL_SET_ORDER


TRACK_ORDER = ["raw", "sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
ENCODING_TRACKS = TRACK_ORDER[1:]
RANDOM_TRACK_ORDER = ["raw", "sub-01"]
N_BOOTSTRAP = 10000


def _stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def _bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        value = float(values[0]) if len(values) else np.nan
        return value, value
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    boot = values[idx].min(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _combined_bootstrap_ci(
    margins_by_track: dict[str, np.ndarray],
    encoding_tracks: list[str],
    seed: int,
) -> dict[str, tuple[float, float]]:
    n_models = len(margins_by_track["raw"])
    if n_models < 2:
        raw = float(np.min(margins_by_track["raw"]))
        enc = float(np.mean([np.min(margins_by_track[t]) for t in encoding_tracks]))
        combined = 0.5 * raw + 0.5 * enc
        return {
            "raw": (raw, raw),
            "encoding_mean": (enc, enc),
            "combined_raw_plus_encoding": (combined, combined),
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_models, size=(N_BOOTSTRAP, n_models))
    raw_boot = margins_by_track["raw"][idx].min(axis=1)
    enc_track_boot = [margins_by_track[t][idx].min(axis=1) for t in encoding_tracks]
    enc_boot = np.mean(np.stack(enc_track_boot, axis=0), axis=0)
    combined_boot = 0.5 * raw_boot + 0.5 * enc_boot
    return {
        "raw": (
            float(np.quantile(raw_boot, 0.025)),
            float(np.quantile(raw_boot, 0.975)),
        ),
        "encoding_mean": (
            float(np.quantile(enc_boot, 0.025)),
            float(np.quantile(enc_boot, 0.975)),
        ),
        "combined_raw_plus_encoding": (
            float(np.quantile(combined_boot, 0.025)),
            float(np.quantile(combined_boot, 0.975)),
        ),
    }


def _matrix_path(model_set: str, matrix_source: str) -> Path:
    if matrix_source == "selected":
        return config.EVAL_DATA_DIR / model_set / "correlation_matrices.csv"
    if matrix_source == "random":
        return (
            config.SELECTION_OUTPUT_ROOT
            / model_set
            / "method-raw_plus_all_encodings"
            / "20251222_175721"
            / "eval_pipeline"
            / "correlation_matrices_with_random_noised_pool.csv"
        )
    raise ValueError(f"Unknown matrix_source: {matrix_source}")


def _load_matrix(
    model_set: str,
    matrix_source: str,
    track: str,
    matrix_type: str,
) -> tuple[list[str], np.ndarray] | None:
    path = _matrix_path(model_set, matrix_source)
    if not path.exists():
        print(f"  [WARN] Missing {matrix_source} matrix file: {path}")
        return None

    filtered = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["track"] == track and row["matrix_type"] == matrix_type:
                filtered.append(row)

    if not filtered:
        print(f"  [WARN] Missing matrix rows: {model_set}/{track}/{matrix_type}")
        return None

    models = list(dict.fromkeys(row["model_i"] for row in filtered))
    model_to_pos = {model: i for i, model in enumerate(models)}
    matrix = np.full((len(models), len(models)), np.nan, dtype=float)
    for row in filtered:
        i = model_to_pos[row["model_i"]]
        j = model_to_pos[row["model_j"]]
        matrix[i, j] = float(row["correlation"])
    if np.isnan(matrix).any():
        raise ValueError(f"NaNs in matrix for {model_set}/{track}/{matrix_type}")
    return models, matrix


def _margins_from_matrix(matrix: np.ndarray, noised: bool) -> list[dict[str, float]]:
    """Return per-model selection margins from a cross-model correlation matrix."""
    diag = np.diag(matrix)
    n_models = matrix.shape[0]
    rows = []
    for i in range(n_models):
        if noised:
            # compute_correlation_at_target_noise returns corr(clean_i, noised_j);
            # selection margins for noised model j therefore live in column j.
            off = np.delete(matrix[:, i], i)
        else:
            off = np.delete(matrix[i, :], i)
        mean_other = float(np.mean(off))
        self_corr = float(diag[i])
        rows.append(
            {
                "model_position": i,
                "self_corr": self_corr,
                "mean_other_corr": mean_other,
                "margin": self_corr - mean_other,
            }
        )
    return rows


def compute_objective_rows(model_sets: list[str]) -> tuple[list[dict], list[dict]]:
    by_model_rows = []
    summary_rows = []

    requests = [
        ("selected", "selected_clean", "clean", "selected"),
        ("selected", "selected_noised", "noised", "selected"),
        ("random", "random_clean", "clean", "random"),
        ("random", "random_noised", "noised", "random"),
    ]

    for model_set in model_sets:
        print(f"Processing {model_set}")
        for track in TRACK_ORDER:
            for matrix_source, matrix_type, noise_condition, subset_type in requests:
                if matrix_source == "random" and track not in RANDOM_TRACK_ORDER:
                    continue
                loaded = _load_matrix(model_set, matrix_source, track, matrix_type)
                if loaded is None:
                    continue
                models, matrix = loaded
                margin_rows = _margins_from_matrix(
                    matrix,
                    noised=noise_condition == "noised",
                )
                for row in margin_rows:
                    row.update(
                        {
                            "model_set": model_set,
                            "track": track,
                            "subset_type": subset_type,
                            "noise_condition": noise_condition,
                            "matrix_type": matrix_type,
                            "matrix_source": str(_matrix_path(model_set, matrix_source)),
                            "model": models[int(row["model_position"])],
                        }
                    )
                    by_model_rows.append(row)

                margins = np.array([row["margin"] for row in margin_rows], dtype=float)
                self_corrs = np.array([row["self_corr"] for row in margin_rows], dtype=float)
                mean_other_corrs = np.array(
                    [row["mean_other_corr"] for row in margin_rows], dtype=float
                )
                ci_low, ci_high = _bootstrap_ci(
                    margins,
                    _stable_seed(model_set, track, subset_type, noise_condition),
                )
                summary_rows.append(
                    {
                        "model_set": model_set,
                        "track": track,
                        "subset_type": subset_type,
                        "noise_condition": noise_condition,
                        "matrix_type": matrix_type,
                        "n_models": len(models),
                        "objective_min": float(np.min(margins)),
                        "objective_ci95_low": ci_low,
                        "objective_ci95_high": ci_high,
                        "objective_ci_method": "model_bootstrap_min",
                        "margin_mean": float(np.mean(margins)),
                        "margin_median": float(np.median(margins)),
                        "margin_sd": float(np.std(margins, ddof=1)) if len(margins) > 1 else 0.0,
                        "self_corr_mean": float(np.mean(self_corrs)),
                        "mean_other_corr_mean": float(np.mean(mean_other_corrs)),
                    }
                )

    return by_model_rows, summary_rows


def compute_combined_summary(summary: list[dict], by_model: list[dict]) -> list[dict]:
    rows = []
    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in summary:
        key = (row["model_set"], row["subset_type"], row["noise_condition"])
        grouped.setdefault(key, {})[row["track"]] = float(row["objective_min"])

    margin_grouped: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in by_model:
        key = (
            row["model_set"],
            row["subset_type"],
            row["noise_condition"],
            row["track"],
        )
        margin_grouped.setdefault(key, {})[row["model"]] = float(row["margin"])

    condition_keys = sorted(
        {
            (row["model_set"], row["noise_condition"])
            for row in summary
        }
    )
    for model_set, noise_condition in condition_keys:
        selected_vals = grouped.get((model_set, "selected", noise_condition), {})
        random_vals = grouped.get((model_set, "random", noise_condition), {})
        if "raw" not in selected_vals or "raw" not in random_vals:
            continue

        common_encoding_tracks = [
            track
            for track in ENCODING_TRACKS
            if track in selected_vals and track in random_vals
        ]
        if not common_encoding_tracks:
            continue

        encoding_tracks_used = ";".join(common_encoding_tracks)
        for subset_type, vals in [
            ("selected", selected_vals),
            ("random", random_vals),
        ]:
            model_sets = [
                set(
                    margin_grouped.get(
                        (model_set, subset_type, noise_condition, track), {}
                    )
                )
                for track in ["raw", *common_encoding_tracks]
            ]
            common_models = sorted(set.intersection(*model_sets)) if model_sets else []
            if not common_models:
                continue
            margins_by_track = {
                track: np.array(
                    [
                        margin_grouped[
                            (model_set, subset_type, noise_condition, track)
                        ][model]
                        for model in common_models
                    ],
                    dtype=float,
                )
                for track in ["raw", *common_encoding_tracks]
            }
            ci = _combined_bootstrap_ci(
                margins_by_track,
                common_encoding_tracks,
                _stable_seed(model_set, subset_type, noise_condition, "combined"),
            )

            raw = float(vals["raw"])
            enc_mean = float(np.mean([vals[t] for t in common_encoding_tracks]))
            combined = 0.5 * raw + 0.5 * enc_mean
            base = {
                "model_set": model_set,
                "subset_type": subset_type,
                "noise_condition": noise_condition,
                "raw_objective_min": raw,
                "encoding_objective_min_mean": enc_mean,
                "n_encoding_tracks": len(common_encoding_tracks),
                "encoding_tracks_used": encoding_tracks_used,
                "n_bootstrap_models": len(common_models),
                "objective_ci_method": "model_bootstrap_min",
            }
            rows.append(
                {
                    **base,
                    "track": "combined_raw_plus_encoding",
                    "objective_min": combined,
                    "objective_ci95_low": ci["combined_raw_plus_encoding"][0],
                    "objective_ci95_high": ci["combined_raw_plus_encoding"][1],
                }
            )
            rows.append(
                {
                    **base,
                    "track": "encoding_mean",
                    "objective_min": enc_mean,
                    "objective_ci95_low": ci["encoding_mean"][0],
                    "objective_ci95_high": ci["encoding_mean"][1],
                }
            )
            rows.append(
                {
                    **base,
                    "track": "raw",
                    "objective_min": raw,
                    "objective_ci95_low": ci["raw"][0],
                    "objective_ci95_high": ci["raw"][1],
                }
            )

    delta_lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    subset_grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["model_set"], row["noise_condition"], row["track"])
        subset_grouped.setdefault(key, {})[row["subset_type"]] = row["objective_min"]

    for key, vals in subset_grouped.items():
        if {"selected", "random"} <= set(vals):
            selected = float(vals["selected"])
            random = float(vals["random"])
            delta_lookup[key] = {
                "selected_minus_random": selected - random,
                "selected_over_random": selected / random if random != 0 else np.nan,
            }

    for row in rows:
        key = (row["model_set"], row["noise_condition"], row["track"])
        row.update(delta_lookup.get(key, {}))
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-sets",
        default=",".join(MODEL_SET_ORDER),
        help="Comma-separated model sets to process",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.EVAL_DATA_DIR,
        help="Output directory for CSV files",
    )
    args = parser.parse_args()

    model_sets = [m.strip() for m in args.model_sets.split(",") if m.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_model, summary = compute_objective_rows(model_sets)
    combined = compute_combined_summary(summary, by_model)

    by_model_path = args.out_dir / "selection_objective_by_model.csv"
    summary_path = args.out_dir / "selection_objective_summary.csv"
    combined_path = args.out_dir / "selection_objective_combined.csv"

    _write_csv(by_model_path, by_model)
    _write_csv(summary_path, summary)
    _write_csv(combined_path, combined)

    print(f"Saved {len(by_model)} rows -> {by_model_path}")
    print(f"Saved {len(summary)} rows -> {summary_path}")
    print(f"Saved {len(combined)} rows -> {combined_path}")


if __name__ == "__main__":
    main()
