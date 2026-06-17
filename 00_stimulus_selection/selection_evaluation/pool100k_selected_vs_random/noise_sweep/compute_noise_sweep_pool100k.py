#!/usr/bin/env python3
"""Noise sweep for the standalone 100k-pool selected-vs-random diagnostics.

This reuses the exact random subset indices from the target-noise analysis and
computes noisy-by-clean model recovery and attenuated model-model correlations
across multiple noise multipliers.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SCRIPT = Path(__file__).resolve()
SWEEP_DIR = SCRIPT.parent
ANALYSIS_DIR = SWEEP_DIR.parent
RESULTS_DIR = ANALYSIS_DIR / "results"
SHARE_ROOT = ANALYSIS_DIR.parents[2]

sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SHARE_ROOT / "src"))
sys.path.insert(0, str(SHARE_ROOT / "src"))

from cstims.paper import config  # noqa: E402
from compute_selected_vs_random_pool100k import (  # noqa: E402
    DEFAULT_MODEL_SETS,
    DEFAULT_POOL_DIR,
    DEFAULT_TRACKS,
    clean_matrix,
    compute_rdms,
    load_encoding_params,
    load_noise_stds,
    load_pool_features,
    load_selected_raw_features,
    matrix_component_rows,
    noisy_by_clean_matrix,
    parse_csv_list,
    recovery_from_mean_matrix,
    summarize_matrix,
    write_csv,
)


DEFAULT_MULTIPLIERS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


@dataclass(frozen=True)
class SweepConfig:
    pool_dir: str
    subset_indices: str
    output_dir: str
    model_sets: list[str]
    tracks: list[str]
    noise_multipliers: list[float]
    subset_size: int
    n_subsets: int
    n_noise_draws: int
    noise_batch: int
    subset_chunk: int
    metric: str
    corr_type: str
    seed: int
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument(
        "--subset-indices",
        type=Path,
        default=RESULTS_DIR / "random_subset_indices.npy",
    )
    parser.add_argument("--output-dir", type=Path, default=SWEEP_DIR / "results")
    parser.add_argument("--model-sets", default=",".join(DEFAULT_MODEL_SETS))
    parser.add_argument("--tracks", default=",".join(DEFAULT_TRACKS))
    parser.add_argument(
        "--noise-multipliers",
        default=",".join(str(v) for v in DEFAULT_MULTIPLIERS),
    )
    parser.add_argument("--n-noise-draws", type=int, default=100)
    parser.add_argument("--noise-batch", type=int, default=10)
    parser.add_argument("--subset-chunk", type=int, default=10)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--corr-type", default="pearson")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def snr_from_multiplier(multiplier: float) -> float:
    return float("inf") if multiplier == 0 else 1.0 / float(multiplier)


def add_matrix_rows(rows: list[dict], matrix: np.ndarray, models: list[str], base: dict) -> None:
    for i, row_model in enumerate(models):
        for j, col_model in enumerate(models):
            rows.append(
                {
                    **base,
                    "row_model": row_model,
                    "col_model": col_model,
                    "correlation": float(matrix[i, j]),
                }
            )


def add_summary_row(
    rows: list[dict],
    matrix: np.ndarray,
    base: dict,
    recovery_accuracy: float,
    recovery_accuracy_sd: float | None = None,
) -> None:
    rows.append(
        {
            **base,
            **summarize_matrix(matrix),
            "recovery_accuracy": recovery_accuracy,
            "recovery_accuracy_sd": recovery_accuracy_sd,
            "recovery_accuracy_from_mean_matrix": recovery_from_mean_matrix(matrix),
        }
    )


def noisy_or_clean_at_multiplier(
    rdms: torch.Tensor,
    clean: torch.Tensor,
    noise_stds: torch.Tensor,
    corr_type: str,
    multiplier: float,
    n_noise_draws: int,
    noise_batch: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if multiplier == 0:
        if clean.ndim == 2:
            matrix = clean.unsqueeze(0)
        else:
            matrix = clean
        n_items = matrix.shape[0]
        n_models = matrix.shape[1]
        pred = matrix.argmax(dim=2)
        truth = torch.arange(n_models, device=matrix.device).reshape(1, n_models)
        model_acc = (pred == truth).float()
        overall_acc = model_acc.mean(dim=1)
        return matrix, model_acc, overall_acc

    return noisy_by_clean_matrix(
        rdms=rdms,
        noise_stds=noise_stds * float(multiplier),
        corr_type=corr_type,
        n_noise_draws=n_noise_draws,
        noise_batch=noise_batch,
        generator=generator,
    )


def chunk_pool_features(
    pool_features: dict[str, np.ndarray],
    models: list[str],
    subset_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {model: pool_features[model][subset_indices] for model in models}


def main() -> None:
    args = parse_args()
    model_sets = parse_csv_list(args.model_sets)
    tracks = parse_csv_list(args.tracks)
    multipliers = [float(v) for v in parse_csv_list(args.noise_multipliers)]
    subsets = np.load(args.subset_indices)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_models = sorted({model for model_set in model_sets for model in config.MODEL_SETS[model_set]})
    pool_features = load_pool_features(args.pool_dir, all_models)
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(args.seed)
    encoding_cache: dict[str, dict] = {}

    matrix_rows: list[dict] = []
    component_rows: list[dict] = []
    recovery_rows: list[dict] = []
    summary_rows: list[dict] = []
    subset_summary_rows: list[dict] = []

    for model_set in tqdm(model_sets, desc="Model sets", unit="set"):
        models = [model for model in config.MODEL_SETS[model_set] if model in pool_features]
        selected_raw = load_selected_raw_features(model_set, models)

        for track in tqdm(tracks, desc=f"{model_set} tracks", leave=False, unit="track"):
            encoding_params = load_encoding_params(track, device, encoding_cache)
            base_noise_stds = load_noise_stds(model_set, track, models, device)

            selected_rdms = compute_rdms(
                selected_raw, models, track, args.metric, device, encoding_params
            )
            selected_clean = clean_matrix(selected_rdms, args.corr_type)

            for multiplier in tqdm(
                multipliers,
                desc=f"{model_set}/{track} selected noise",
                leave=False,
                unit="level",
            ):
                matrix, model_acc, overall_acc = noisy_or_clean_at_multiplier(
                    selected_rdms,
                    selected_clean,
                    base_noise_stds,
                    args.corr_type,
                    multiplier,
                    args.n_noise_draws,
                    args.noise_batch,
                    torch_gen,
                )
                matrix_np = matrix[0].detach().cpu().numpy()
                model_acc_np = model_acc[0].detach().cpu().numpy()
                recovery = float(overall_acc[0].detach().cpu())
                base = {
                    "model_set": model_set,
                    "subset_type": "selected",
                    "track": track,
                    "noise_multiplier": multiplier,
                    "relative_snr": snr_from_multiplier(multiplier),
                    "n_models": len(models),
                    "n_subsets": 1,
                    "n_noise_draws": 0 if multiplier == 0 else args.n_noise_draws,
                }
                add_matrix_rows(matrix_rows, matrix_np, models, base)
                component_rows.extend(matrix_component_rows(matrix_np, models, base))
                add_summary_row(summary_rows, matrix_np, base, recovery)
                for model, accuracy in zip(models, model_acc_np):
                    recovery_rows.append({**base, "model": model, "recovery_accuracy": float(accuracy)})

                del matrix, model_acc, overall_acc

            del selected_rdms, selected_clean
            if device.type == "cuda":
                torch.cuda.empty_cache()

            matrix_sums = {
                multiplier: np.zeros((len(models), len(models)), dtype=np.float64)
                for multiplier in multipliers
            }
            model_acc_sums = {multiplier: np.zeros(len(models), dtype=np.float64) for multiplier in multipliers}
            recovery_values = {multiplier: [] for multiplier in multipliers}

            for start in tqdm(
                range(0, len(subsets), args.subset_chunk),
                desc=f"{model_set}/{track} random chunks",
                leave=False,
                unit="chunk",
            ):
                stop = min(start + args.subset_chunk, len(subsets))
                subset_chunk = subsets[start:stop]
                chunk_raw = chunk_pool_features(pool_features, models, subset_chunk)
                random_rdms = compute_rdms(
                    chunk_raw, models, track, args.metric, device, encoding_params
                )
                random_clean = clean_matrix(random_rdms, args.corr_type)

                for multiplier in multipliers:
                    matrix, model_acc, overall_acc = noisy_or_clean_at_multiplier(
                        random_rdms,
                        random_clean,
                        base_noise_stds,
                        args.corr_type,
                        multiplier,
                        args.n_noise_draws,
                        args.noise_batch,
                        torch_gen,
                    )
                    matrix_np = matrix.detach().cpu().numpy()
                    model_acc_np = model_acc.detach().cpu().numpy()
                    acc_np = overall_acc.detach().cpu().numpy()

                    matrix_sums[multiplier] += matrix_np.sum(axis=0)
                    model_acc_sums[multiplier] += model_acc_np.sum(axis=0)
                    recovery_values[multiplier].extend(float(v) for v in acc_np)

                    for local_idx, subset_idx in enumerate(range(start, stop)):
                        info = summarize_matrix(matrix_np[local_idx])
                        subset_summary_rows.append(
                            {
                                "model_set": model_set,
                                "subset_type": "random",
                                "track": track,
                                "noise_multiplier": multiplier,
                                "relative_snr": snr_from_multiplier(multiplier),
                                "subset_index": subset_idx,
                                "n_models": len(models),
                                "subset_size": subset_chunk.shape[1],
                                "margin_min": info["margin_min"],
                                "margin_mean": info["margin_mean"],
                                "self_corr_mean": info["self_corr_mean"],
                                "mean_other_corr": info["mean_other_corr"],
                                "recovery_accuracy": float(acc_np[local_idx]),
                            }
                        )

                    del matrix, model_acc, overall_acc, matrix_np, model_acc_np, acc_np

                del chunk_raw, random_rdms, random_clean
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            for multiplier in multipliers:
                matrix_np = matrix_sums[multiplier] / float(len(subsets))
                model_acc = model_acc_sums[multiplier] / float(len(subsets))
                rec_values = np.asarray(recovery_values[multiplier], dtype=float)
                base = {
                    "model_set": model_set,
                    "subset_type": "random",
                    "track": track,
                    "noise_multiplier": multiplier,
                    "relative_snr": snr_from_multiplier(multiplier),
                    "n_models": len(models),
                    "n_subsets": len(subsets),
                    "n_noise_draws": 0 if multiplier == 0 else args.n_noise_draws,
                }
                add_matrix_rows(matrix_rows, matrix_np, models, base)
                component_rows.extend(matrix_component_rows(matrix_np, models, base))
                add_summary_row(
                    summary_rows,
                    matrix_np,
                    base,
                    float(rec_values.mean()),
                    float(rec_values.std(ddof=1)) if len(rec_values) > 1 else 0.0,
                )
                for model, accuracy in zip(models, model_acc):
                    recovery_rows.append({**base, "model": model, "recovery_accuracy": float(accuracy)})

    write_csv(args.output_dir / "sweep_mean_correlation_matrices.csv", matrix_rows)
    write_csv(args.output_dir / "sweep_model_components.csv", component_rows)
    write_csv(args.output_dir / "sweep_model_recovery.csv", recovery_rows)
    write_csv(args.output_dir / "sweep_track_summary.csv", summary_rows)
    write_csv(args.output_dir / "sweep_random_subset_summary.csv", subset_summary_rows)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT.relative_to(SHARE_ROOT)),
        "run_config": asdict(
            SweepConfig(
                pool_dir=str(args.pool_dir),
                subset_indices=str(args.subset_indices),
                output_dir=str(args.output_dir),
                model_sets=model_sets,
                tracks=tracks,
                noise_multipliers=multipliers,
                subset_size=int(subsets.shape[1]),
                n_subsets=int(subsets.shape[0]),
                n_noise_draws=args.n_noise_draws,
                noise_batch=args.noise_batch,
                subset_chunk=args.subset_chunk,
                metric=args.metric,
                corr_type=args.corr_type,
                seed=args.seed,
                device=str(device),
            )
        ),
        "notes": (
            "Rows are noisy model RDMs and columns are clean model RDMs. "
            "noise_multiplier=0 uses the clean matrix without Monte Carlo noise."
        ),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    print(f"Wrote sweep results to {args.output_dir}")


if __name__ == "__main__":
    main()
