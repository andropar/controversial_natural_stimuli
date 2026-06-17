#!/usr/bin/env python3
"""Selected-vs-random objective diagnostics using the 100k natural-image pool.

This is a standalone add-on analysis. It does not import or overwrite the
existing selection-evaluation outputs. For each model set and track it compares
the 100 selected cSTIMs against random 100-image subsets sampled from the 100k
feature cache.

Outputs:
    results/mean_correlation_matrices.csv
    results/model_components.csv
    results/model_recovery.csv
    results/track_summary.csv
    results/random_subset_summary.csv
    results/random_subset_indices.npy
    results/metadata.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


SCRIPT = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT.parent
SELECTION_EVAL_ROOT = ANALYSIS_DIR.parent
STIMULUS_ROOT = SELECTION_EVAL_ROOT.parent
SHARE_ROOT = STIMULUS_ROOT.parent

sys.path.insert(0, str(SHARE_ROOT / "src"))
sys.path.insert(0, str(SHARE_ROOT / "src"))

from cstims.paper import config  # noqa: E402
from cstims.encoding.linear import load_encoding_params_by_encoding  # noqa: E402
from cstims.rdm_cuda import get_rdm_vector  # noqa: E402
from cstims.selection.primitives import compute_correlation_matrix  # noqa: E402


DEFAULT_POOL_DIR = SHARE_ROOT / "shared" / "cache_or_heavy" / "natural_pool_subset_100k_seed42"
DEFAULT_MODEL_SETS = ("all_models", "sota", "training_objective", "architecture", "dataset")
DEFAULT_TRACKS = ("raw", "sub-01", "sub-03", "sub-05", "sub-06", "sub-07")


@dataclass(frozen=True)
class RunConfig:
    pool_dir: str
    output_dir: str
    model_sets: list[str]
    tracks: list[str]
    n_subsets: int
    subset_size: int
    n_noise_draws: int
    noise_batch: int
    subset_chunk: int
    metric: str
    corr_type: str
    seed: int
    device: str


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR / "results")
    parser.add_argument("--model-sets", default=",".join(DEFAULT_MODEL_SETS))
    parser.add_argument("--tracks", default=",".join(DEFAULT_TRACKS))
    parser.add_argument("--n-subsets", type=int, default=500)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--n-noise-draws", type=int, default=100)
    parser.add_argument("--noise-batch", type=int, default=10)
    parser.add_argument("--subset-chunk", type=int, default=10)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument(
        "--corr-type",
        default="pearson",
        help="Use 'spearman' for rank correlations; any other value is Pearson.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selected_payload_path(model_set: str) -> Path:
    return config.SELECTION_OUTPUT_ROOT / model_set / "selected_stimuli_data.pkl"


def load_selected_raw_features(model_set: str, models: list[str]) -> dict[str, np.ndarray]:
    path = selected_payload_path(model_set)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    selected = payload.get("selected_features_raw") or payload.get("selected_features")
    if not selected:
        raise ValueError(f"No selected raw features found in {path}")
    missing = [model for model in models if model not in selected]
    if missing:
        raise ValueError(f"Selected payload missing models for {model_set}: {missing}")
    return {model: np.asarray(selected[model], dtype=np.float32) for model in models}


def load_pool_features(pool_dir: Path, models: list[str]) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for model in tqdm(models, desc="Loading 100k pool features", unit="model"):
        path = pool_dir / f"{model}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing pool features for {model}: {path}")
        with np.load(path, allow_pickle=True) as z:
            features[model] = z["features"]
    return features


def make_random_subsets(
    n_total: int,
    n_subsets: int,
    subset_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    subsets = np.empty((n_subsets, subset_size), dtype=np.int64)
    for idx in range(n_subsets):
        subsets[idx] = np.sort(rng.choice(n_total, size=subset_size, replace=False))
    return subsets


def noise_calibration_path(model_set: str) -> Path:
    current = config.EVAL_DATA_DIR / f"{model_set}_unique_boot" / "noise_calibration.csv"
    if current.exists():
        return current
    fallback = config.EVAL_DATA_DIR / model_set / "noise_calibration.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No noise calibration found for {model_set}")


def load_noise_stds(
    model_set: str,
    track: str,
    models: list[str],
    device: torch.device,
) -> torch.Tensor:
    df = pd.read_csv(noise_calibration_path(model_set))
    sub = df[df["track"] == track]
    if sub.empty:
        raise ValueError(f"No noise calibration rows for {model_set}/{track}")
    lookup = dict(zip(sub["model"], sub["noise_std"]))
    missing = [model for model in models if model not in lookup]
    if missing:
        raise ValueError(f"Noise calibration missing models for {model_set}/{track}: {missing}")
    values = [float(lookup[model]) for model in models]
    return torch.tensor(values, dtype=torch.float32, device=device).reshape(1, len(models), 1)


def load_encoding_params(
    track: str,
    device: torch.device,
    cache: dict[str, dict],
) -> dict | None:
    if track == "raw":
        return None
    if track not in cache:
        params = load_encoding_params_by_encoding(
            encoding_root=config.UNIQUE_ENCODING_DIRS[track],
            model_list_csv=config.MODEL_LIST_CSV,
            encoding_names=[track],
            device=device,
            roi_subset="hlvis",
        )
        if track not in params:
            raise RuntimeError(f"Failed to load encoding parameters for {track}")
        cache[track] = params[track]
    return cache[track]


def tensor_from_numpy(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array, dtype=np.float32), device=device)


def project_if_needed(
    raw: np.ndarray,
    model: str,
    track: str,
    device: torch.device,
    encoding_params: dict | None,
) -> torch.Tensor:
    feats = tensor_from_numpy(raw, device)
    if track == "raw":
        return feats
    if encoding_params is None or model not in encoding_params:
        raise KeyError(f"Missing encoding parameters for {track}/{model}")
    original_shape = feats.shape
    flat = feats.reshape(-1, original_shape[-1])
    params = encoding_params[model]
    encoded = flat @ params["W"] + params["bias"]
    return encoded.reshape(*original_shape[:-1], encoded.shape[-1])


def compute_rdms(
    raw_features: dict[str, np.ndarray],
    models: list[str],
    track: str,
    metric: str,
    device: torch.device,
    encoding_params: dict | None,
) -> torch.Tensor:
    rdm_by_model = []
    for model in models:
        feats = project_if_needed(raw_features[model], model, track, device, encoding_params)
        rdm_by_model.append(get_rdm_vector(feats, metric=metric))
        del feats
    if rdm_by_model[0].ndim == 1:
        return torch.stack(rdm_by_model, dim=0)
    return torch.stack(rdm_by_model, dim=1)


def clean_matrix(rdms: torch.Tensor, corr_type: str) -> torch.Tensor:
    if rdms.ndim == 2:
        return compute_correlation_matrix(rdms.unsqueeze(0), rdms.unsqueeze(0), corr_type)[0]
    return compute_correlation_matrix(rdms, rdms, corr_type)


def noisy_by_clean_matrix(
    rdms: torch.Tensor,
    noise_stds: torch.Tensor,
    corr_type: str,
    n_noise_draws: int,
    noise_batch: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean noisy-by-clean matrices plus recovery accuracy.

    Rows are noisy model RDMs; columns are clean model RDMs. Recovery accuracy is
    computed on each individual noise draw before averaging matrices.
    """
    if rdms.ndim == 2:
        base = rdms.unsqueeze(0)
    else:
        base = rdms
    n_items, n_models, n_pairs = base.shape
    matrix_sum = torch.zeros((n_items, n_models, n_models), dtype=torch.float32, device=base.device)
    correct_by_model = torch.zeros((n_items, n_models), dtype=torch.float32, device=base.device)
    truth = torch.arange(n_models, device=base.device).reshape(1, 1, n_models)

    draws_done = 0
    while draws_done < n_noise_draws:
        draws = min(noise_batch, n_noise_draws - draws_done)
        noise = torch.randn(
            (draws, n_items, n_models, n_pairs),
            device=base.device,
            generator=generator,
        )
        noised = base.unsqueeze(0) + noise * noise_stds.reshape(1, 1, n_models, 1)
        noised = noised.reshape(draws * n_items, n_models, n_pairs)
        clean = (
            base.unsqueeze(0)
            .expand(draws, n_items, n_models, n_pairs)
            .reshape(draws * n_items, n_models, n_pairs)
        )
        corr = compute_correlation_matrix(noised, clean, corr_type)
        corr_view = corr.reshape(draws, n_items, n_models, n_models)
        matrix_sum += corr_view.sum(dim=0)
        pred = corr_view.argmax(dim=3)
        correct_by_model += (pred == truth).sum(dim=0).float()
        draws_done += draws

        del noise, noised, clean, corr, corr_view, pred

    mean_matrix = matrix_sum / float(n_noise_draws)
    model_accuracy = correct_by_model / float(n_noise_draws)
    overall_accuracy = model_accuracy.mean(dim=1)
    return mean_matrix, model_accuracy, overall_accuracy


def matrix_component_rows(
    matrix: np.ndarray,
    models: list[str],
    base: dict,
) -> list[dict]:
    rows = []
    for idx, model in enumerate(models):
        off = np.delete(matrix[idx, :], idx)
        self_corr = float(matrix[idx, idx])
        mean_other = float(np.mean(off))
        rows.append(
            {
                **base,
                "model": model,
                "self_corr": self_corr,
                "mean_other_corr": mean_other,
                "margin": self_corr - mean_other,
            }
        )
    return rows


def summarize_matrix(matrix: np.ndarray) -> dict[str, float]:
    diag = np.diag(matrix)
    off_mask = ~np.eye(matrix.shape[0], dtype=bool)
    rows = matrix_component_rows(matrix, [str(i) for i in range(matrix.shape[0])], {})
    margins = np.array([row["margin"] for row in rows], dtype=float)
    return {
        "self_corr_mean": float(np.mean(diag)),
        "self_corr_min": float(np.min(diag)),
        "mean_other_corr": float(np.mean(matrix[off_mask])),
        "margin_mean": float(np.mean(margins)),
        "margin_min": float(np.min(margins)),
    }


def recovery_from_mean_matrix(matrix: np.ndarray) -> float:
    pred = np.argmax(matrix, axis=1)
    return float(np.mean(pred == np.arange(matrix.shape[0])))


def add_matrix_rows(
    rows: list[dict],
    matrix: np.ndarray,
    models: list[str],
    base: dict,
) -> None:
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
    recovery_accuracy: float | None,
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
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = RunConfig(
        pool_dir=str(args.pool_dir),
        output_dir=str(args.output_dir),
        model_sets=model_sets,
        tracks=tracks,
        n_subsets=args.n_subsets,
        subset_size=args.subset_size,
        n_noise_draws=args.n_noise_draws,
        noise_batch=args.noise_batch,
        subset_chunk=args.subset_chunk,
        metric=args.metric,
        corr_type=args.corr_type,
        seed=args.seed,
        device=str(device),
    )

    all_models = sorted({model for model_set in model_sets for model in config.MODEL_SETS[model_set]})
    pool_features = load_pool_features(args.pool_dir, all_models)
    n_total = min(features.shape[0] for features in pool_features.values())
    subsets = make_random_subsets(n_total, args.n_subsets, args.subset_size, args.seed)
    np.save(args.output_dir / "random_subset_indices.npy", subsets)

    matrix_rows: list[dict] = []
    component_rows: list[dict] = []
    recovery_rows: list[dict] = []
    summary_rows: list[dict] = []
    subset_summary_rows: list[dict] = []
    encoding_cache: dict[str, dict] = {}
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(args.seed)

    for model_set in tqdm(model_sets, desc="Model sets", unit="set"):
        models = [model for model in config.MODEL_SETS[model_set] if model in pool_features]
        selected_raw = load_selected_raw_features(model_set, models)

        for track in tqdm(tracks, desc=f"{model_set} tracks", leave=False, unit="track"):
            encoding_params = load_encoding_params(track, device, encoding_cache)
            noise_stds = load_noise_stds(model_set, track, models, device)

            # Selected cSTIMs.
            selected_rdms = compute_rdms(
                selected_raw, models, track, args.metric, device, encoding_params
            )
            selected_clean = clean_matrix(selected_rdms, args.corr_type).detach().cpu().numpy()
            selected_noisy, selected_model_acc, selected_acc = noisy_by_clean_matrix(
                selected_rdms,
                noise_stds,
                args.corr_type,
                args.n_noise_draws,
                args.noise_batch,
                torch_gen,
            )
            selected_noisy_np = selected_noisy[0].detach().cpu().numpy()
            selected_model_acc_np = selected_model_acc[0].detach().cpu().numpy()
            selected_acc_value = float(selected_acc[0].detach().cpu())

            for matrix_type, matrix, recovery in [
                ("clean", selected_clean, recovery_from_mean_matrix(selected_clean)),
                ("noisy_by_clean", selected_noisy_np, selected_acc_value),
            ]:
                base = {
                    "model_set": model_set,
                    "subset_type": "selected",
                    "track": track,
                    "matrix_type": matrix_type,
                    "n_models": len(models),
                    "n_subsets": 1,
                    "n_noise_draws": args.n_noise_draws if matrix_type == "noisy_by_clean" else 0,
                }
                add_matrix_rows(matrix_rows, matrix, models, base)
                component_rows.extend(matrix_component_rows(matrix, models, base))
                add_summary_row(summary_rows, matrix, base, recovery)

            for model, accuracy in zip(models, selected_model_acc_np):
                recovery_rows.append(
                    {
                        "model_set": model_set,
                        "subset_type": "selected",
                        "track": track,
                        "n_models": len(models),
                        "n_subsets": 1,
                        "n_noise_draws": args.n_noise_draws,
                        "model": model,
                        "recovery_accuracy": float(accuracy),
                    }
                )

            del selected_rdms, selected_noisy, selected_model_acc, selected_acc
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Random 100-image subsets from the 100k pool.
            matrix_sums = {
                "clean": np.zeros((len(models), len(models)), dtype=np.float64),
                "noisy_by_clean": np.zeros((len(models), len(models)), dtype=np.float64),
            }
            model_acc_sum = np.zeros(len(models), dtype=np.float64)
            recovery_values: list[float] = []

            for start in tqdm(
                range(0, args.n_subsets, args.subset_chunk),
                desc=f"{model_set}/{track} random chunks",
                leave=False,
                unit="chunk",
            ):
                stop = min(start + args.subset_chunk, args.n_subsets)
                subset_chunk = subsets[start:stop]
                chunk_raw = chunk_pool_features(pool_features, models, subset_chunk)
                random_rdms = compute_rdms(
                    chunk_raw, models, track, args.metric, device, encoding_params
                )
                random_clean = clean_matrix(random_rdms, args.corr_type)
                random_noisy, random_model_acc, random_acc = noisy_by_clean_matrix(
                    random_rdms,
                    noise_stds,
                    args.corr_type,
                    args.n_noise_draws,
                    args.noise_batch,
                    torch_gen,
                )
                clean_np = random_clean.detach().cpu().numpy()
                noisy_np = random_noisy.detach().cpu().numpy()
                model_acc_np = random_model_acc.detach().cpu().numpy()
                acc_np = random_acc.detach().cpu().numpy()

                matrix_sums["clean"] += clean_np.sum(axis=0)
                matrix_sums["noisy_by_clean"] += noisy_np.sum(axis=0)
                model_acc_sum += model_acc_np.sum(axis=0)
                recovery_values.extend(float(v) for v in acc_np)

                for local_idx, subset_idx in enumerate(range(start, stop)):
                    clean_summary = summarize_matrix(clean_np[local_idx])
                    noisy_summary = summarize_matrix(noisy_np[local_idx])
                    subset_summary_rows.append(
                        {
                            "model_set": model_set,
                            "subset_type": "random",
                            "track": track,
                            "subset_index": subset_idx,
                            "n_models": len(models),
                            "subset_size": args.subset_size,
                            "clean_margin_min": clean_summary["margin_min"],
                            "clean_margin_mean": clean_summary["margin_mean"],
                            "clean_self_corr_mean": clean_summary["self_corr_mean"],
                            "clean_mean_other_corr": clean_summary["mean_other_corr"],
                            "noisy_margin_min": noisy_summary["margin_min"],
                            "noisy_margin_mean": noisy_summary["margin_mean"],
                            "noisy_self_corr_mean": noisy_summary["self_corr_mean"],
                            "noisy_mean_other_corr": noisy_summary["mean_other_corr"],
                            "noisy_by_clean_recovery_accuracy": float(acc_np[local_idx]),
                        }
                    )

                del chunk_raw, random_rdms, random_clean, random_noisy
                del random_model_acc, random_acc, clean_np, noisy_np, model_acc_np, acc_np
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            mean_matrices = {
                key: value / float(args.n_subsets) for key, value in matrix_sums.items()
            }
            mean_model_acc = model_acc_sum / float(args.n_subsets)
            recovery_sd = float(np.std(recovery_values, ddof=1)) if len(recovery_values) > 1 else 0.0

            for matrix_type, matrix in mean_matrices.items():
                recovery = (
                    float(np.mean(recovery_values))
                    if matrix_type == "noisy_by_clean"
                    else recovery_from_mean_matrix(matrix)
                )
                base = {
                    "model_set": model_set,
                    "subset_type": "random",
                    "track": track,
                    "matrix_type": matrix_type,
                    "n_models": len(models),
                    "n_subsets": args.n_subsets,
                    "n_noise_draws": args.n_noise_draws if matrix_type == "noisy_by_clean" else 0,
                }
                add_matrix_rows(matrix_rows, matrix, models, base)
                component_rows.extend(matrix_component_rows(matrix, models, base))
                add_summary_row(
                    summary_rows,
                    matrix,
                    base,
                    recovery,
                    recovery_sd if matrix_type == "noisy_by_clean" else None,
                )

            for model, accuracy in zip(models, mean_model_acc):
                recovery_rows.append(
                    {
                        "model_set": model_set,
                        "subset_type": "random",
                        "track": track,
                        "n_models": len(models),
                        "n_subsets": args.n_subsets,
                        "n_noise_draws": args.n_noise_draws,
                        "model": model,
                        "recovery_accuracy": float(accuracy),
                    }
                )

    write_csv(args.output_dir / "mean_correlation_matrices.csv", matrix_rows)
    write_csv(args.output_dir / "model_components.csv", component_rows)
    write_csv(args.output_dir / "model_recovery.csv", recovery_rows)
    write_csv(args.output_dir / "track_summary.csv", summary_rows)
    write_csv(args.output_dir / "random_subset_summary.csv", subset_summary_rows)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT.relative_to(SHARE_ROOT)),
        "run_config": asdict(run_config),
        "n_pool_images": int(n_total),
        "pool_feature_models": all_models,
        "notes": (
            "Noisy-by-clean matrices have noisy model RDMs in rows and clean "
            "model RDMs in columns. Recovery accuracy is computed per noise draw "
            "before averaging the noisy-by-clean matrices."
        ),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
