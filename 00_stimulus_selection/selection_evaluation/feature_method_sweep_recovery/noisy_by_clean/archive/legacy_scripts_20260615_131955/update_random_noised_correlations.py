#!/usr/bin/env python3
"""Add random noise-attenuated correlation matrices to existing eval outputs."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[4]
RECOVERY_SCRIPT = (
    ROOT
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "noisy_by_clean_recovery"
    / "code"
    / "compute_noisy_by_clean_recovery.py"
)
DEFAULT_RUN = SCRIPT.parents[1] / "results" / "sota_20260611_112941"
DEFAULT_RANDOM_FEATURE_DIR = ROOT / "shared" / "cache_or_heavy" / "natural_pool_subset_100k_seed42"
DEFAULT_METHODS = [
    "raw_only_mean_min",
    "sub01_only_mean_min",
    "paper_effective_identity_sub01_mean_min",
    "raw_enc_w05_mean_min",
    "raw_enc_w05_max_mean",
    "raw_enc_w05_max_min",
    "paper_effective_identity_sub01_mean_min_no_attenuation",
]
DEFAULT_TRACKS = ["raw", "sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]


def load_recovery_module():
    spec = importlib.util.spec_from_file_location("noisy_by_clean_recovery", RECOVERY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {RECOVERY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rec = load_recovery_module()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def matrix_rows(
    *,
    track_name: str,
    model_names: list[str],
    matrix: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    arr = matrix.cpu().numpy()
    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            rows.append(
                {
                    "track": track_name,
                    "matrix_type": "random_noised",
                    "model_i": model_i,
                    "model_j": model_j,
                    "correlation": float(arr[i, j]),
                    "recovery_orientation": rec.ORIENTATION,
                }
            )
    return rows


def load_filtered_payload(model_set: str, args: argparse.Namespace) -> tuple[dict, list[str]]:
    result_dir = args.selection_root / model_set
    payload = rec.apply_env_paths(rec.eval_utils.load_selection_payload(result_dir), args.env)
    original_models = list(payload["model_names"])
    if args.random_feature_dir is None:
        return payload, original_models

    available_models = rec.available_random_models(args.random_feature_dir, original_models)
    missing = sorted(set(original_models) - set(available_models))
    if missing and args.strict_random_models:
        raise FileNotFoundError(
            f"{model_set}: random feature cache is missing {len(missing)} models: {missing}"
        )
    if len(available_models) < 2:
        raise RuntimeError(f"{model_set}: need at least 2 models after random-cache filtering")
    return rec.filter_payload_to_models(payload, available_models), available_models


def calibrate_noise_stds(
    *,
    payload: dict,
    track_name: str,
    random_features: dict[str, np.ndarray],
    model_names: list[str],
    metric: str,
    target_nc: float,
    device: torch.device,
) -> torch.Tensor:
    precomputed_noise = rec.disc.get_track_noise_variances(payload, track_name)
    if precomputed_noise is not None:
        return torch.stack(
            [
                torch.tensor(precomputed_noise[model], device=device, dtype=torch.float32).sqrt()
                for model in model_names
            ]
        ).unsqueeze(1)

    noise_params = rec.disc.calibrate_noise_parameters(
        features=random_features,
        model_names=model_names,
        metrics=[metric],
        target_nc=target_nc,
        device=device,
        mode="analytical",
    )
    return noise_params.get_noise_stds(metric).to(device)


def compute_track_random_noised_rows(
    *,
    payload: dict,
    track: dict,
    args: argparse.Namespace,
    device: torch.device,
    encoding_params_cache: dict[str, Any],
    encoding_root_map: dict[str, Path] | None,
) -> list[dict[str, Any]]:
    track_name = track["name"]
    model_names = list(payload["model_names"])
    config = payload.get("config", {})
    metric = args.metric or config.get("metric", "cosine")
    corr_type = args.corr_type or config.get("corr_type", "spearman")
    target_nc = config.get("noise_ceiling_target", 0.46)

    print(f"\n  [{track_name}] loading features")
    selected_features, random_features = rec.eval_utils.load_features_for_track(
        payload=payload,
        track=track,
        device=device,
        encoding_params_cache=encoding_params_cache,
        n_random=args.n_random_images,
        selection_variant=args.which_selection,
        encoding_root_map=encoding_root_map,
    )
    n_selected = next(iter(selected_features.values())).shape[0]
    del selected_features
    gc.collect()

    print(f"  [{track_name}] calibrating noise")
    noise_stds = calibrate_noise_stds(
        payload=payload,
        track_name=track_name,
        random_features=random_features,
        model_names=model_names,
        metric=metric,
        target_nc=target_nc,
        device=device,
    )

    print(f"  [{track_name}] computing random RDMs")
    random_rdms = rec.disc.compute_random_baseline_rdms(
        random_features=random_features,
        model_names=model_names,
        metrics=[metric],
        n_selected_stimuli=n_selected,
        n_random_subsets=args.n_random_subsets,
        device=device,
    )
    del random_features
    gc.collect()

    print(f"  [{track_name}] computing random noised correlation matrix")
    matrices = []
    for random_rdm in tqdm(random_rdms, desc=f"{track_name} random_noised", leave=False):
        rdms = random_rdm[metric].to(device)
        corr = rec.disc.compute_correlation_at_target_noise(
            rdms,
            noise_stds,
            corr_type,
            args.n_noise_samples,
        )
        matrices.append(corr.cpu())
        del rdms, corr
        if device.type == "cuda":
            torch.cuda.empty_cache()
    random_noised = torch.stack(matrices).mean(dim=0)
    del random_rdms, matrices
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return matrix_rows(track_name=track_name, model_names=model_names, matrix=random_noised)


def update_model_set(model_set: str, args: argparse.Namespace, device: torch.device, encoding_root_map: dict[str, Path] | None) -> None:
    print(f"\n[{model_set}]")
    payload, _available_models = load_filtered_payload(model_set, args)
    tracks = [
        track
        for track in rec.eval_utils.get_all_tracks_for_evaluation(payload)
        if track["name"] in args.tracks
    ]
    encoding_params_cache: dict[str, Any] = {}
    rows = []
    for track in tracks:
        rows.extend(
            compute_track_random_noised_rows(
                payload=payload,
                track=track,
                args=args,
                device=device,
                encoding_params_cache=encoding_params_cache,
                encoding_root_map=encoding_root_map,
            )
        )

    out_dir = args.output_root / f"{model_set}_noisy_by_clean_boot"
    corr_path = out_dir / "correlation_matrices.csv"
    if not corr_path.exists():
        raise FileNotFoundError(f"Existing correlation CSV not found: {corr_path}")
    existing = pd.read_csv(corr_path)
    updated = pd.concat(
        [
            existing[existing["matrix_type"] != "random_noised"],
            pd.DataFrame(rows),
        ],
        ignore_index=True,
        sort=False,
    )
    updated = updated.sort_values(["track", "matrix_type", "model_i", "model_j"])
    updated.to_csv(corr_path, index=False)
    print(f"[{model_set}] updated {corr_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--tracks", default=",".join(DEFAULT_TRACKS))
    parser.add_argument("--env", choices=rec.eval_utils.VALID_ENVS, default="raven")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random-subsets", type=int, default=50)
    parser.add_argument("--n-random-images", type=int, default=10000)
    parser.add_argument("--n-noise-samples", type=int, default=100)
    parser.add_argument("--which-selection", default="final", choices=["final", "greedy", "best_raw_combined"])
    parser.add_argument("--random-feature-dir", type=Path, default=DEFAULT_RANDOM_FEATURE_DIR)
    parser.add_argument("--strict-random-models", action="store_true", default=True)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--corr-type", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.selection_root = args.run_dir / "cross_eval_full_tracks" / "payloads"
    args.output_root = args.run_dir / "cross_eval_full_tracks" / "eval"
    args.methods = parse_csv_list(args.methods)
    args.tracks = parse_csv_list(args.tracks)
    if args.random_feature_dir is not None:
        args.random_feature_dir = args.random_feature_dir.resolve()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    rec.install_orientation_patch()
    if args.random_feature_dir is not None:
        rec.install_random_pool_loader(args.random_feature_dir)
        print(f"Using random-feature cache: {args.random_feature_dir}")

    encoding_root_map = {
        key: Path(value) for key, value in rec.paper_config.UNIQUE_ENCODING_DIRS.items()
    }
    print(f"Using unique encoding roots: {list(encoding_root_map)}")

    for method in tqdm(args.methods, desc="Model sets"):
        update_model_set(method, args, device, encoding_root_map)


if __name__ == "__main__":
    main()
