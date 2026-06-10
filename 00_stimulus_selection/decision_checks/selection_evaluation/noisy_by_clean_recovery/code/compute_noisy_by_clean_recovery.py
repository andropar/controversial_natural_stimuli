#!/usr/bin/env python3
"""Compute recovery curves with noisy-by-clean model classification.

This is intentionally isolated from the primary selection-evaluation results.
It wraps the original discriminability code, replacing only the model-recovery
score orientation:

    old: corr(clean_i, noisy_j), classify along noisy columns
    new: corr(noisy_i, clean_j), classify along clean columns

The rest of the original AUC/bootstrap machinery is reused.
"""

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

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[5]
ANALYSIS_DIR = (
    ROOT
    / "00_stimulus_selection"
    / "decision_checks"
    / "selection_evaluation"
    / "code"
    / "analysis"
)
HELPERS_DIR = ROOT / "shared" / "code" / "paper_helpers"
SRC_DIR = ROOT / "src"
for path in (SRC_DIR, HELPERS_DIR, ANALYSIS_DIR):
    sys.path.insert(0, str(path))

import config as paper_config  # noqa: E402
import utils as eval_utils  # noqa: E402


MODEL_SET_ORDER = ["all_models", "sota", "training_objective", "architecture", "dataset"]
ENCODING_TRACKS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
DEFAULT_RESULTS = SCRIPT.parents[1] / "results"
DEFAULT_RANDOM_FEATURE_DIR = ROOT / "shared" / "cache_or_heavy" / "natural_pool_subset_10k"
SELECTION_ROOT = ROOT / "00_stimulus_selection" / "results" / "selected_stimuli"
ORIENTATION = "noisy_by_clean"


def load_disc_module():
    path = ANALYSIS_DIR / "02_compute_discriminability.py"
    spec = importlib.util.spec_from_file_location("selection_eval_discriminability", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


disc = load_disc_module()


def _multiclass_error_prob_from_scores(model_score: torch.Tensor) -> float:
    detected = torch.argmax(model_score, dim=2)
    n_models = model_score.shape[1]
    true_idx = torch.arange(n_models, device=model_score.device).unsqueeze(0)
    correct = (detected == true_idx).float().mean(dim=0)
    return float(1.0 - correct.mean().item())


def _normalize_for_correlation(data: torch.Tensor, corr_type: str) -> torch.Tensor:
    """Match cstims.selection.primitives correlation normalization."""
    if data.dtype != torch.float32:
        data = data.float()
    if corr_type == "spearman":
        data = torch.argsort(
            torch.argsort(data, dim=2, stable=False), dim=2, stable=False
        ).float()
    centered = data - data.mean(dim=2, keepdim=True)
    std = centered.std(dim=2, keepdim=True, unbiased=False) + 1e-8
    return centered / std


def _correlate_normalized(X_norm: torch.Tensor, Y_norm: torch.Tensor) -> torch.Tensor:
    n_pairs = X_norm.shape[2]
    correlations = torch.bmm(X_norm, Y_norm.transpose(1, 2)) / n_pairs
    return torch.nan_to_num(correlations, nan=0.0)


def _multiclass_error_dict(model_score: torch.Tensor) -> dict:
    return {
        "non_parametric_multiclass_error_prob": _multiclass_error_prob_from_scores(
            model_score
        )
    }


def _bootstrap_error_probs_from_scores(
    scores: torch.Tensor,
    n_bootstrap: int,
    generator: torch.Generator,
) -> np.ndarray:
    scores_cpu = scores.cpu()
    n_draws, n_models, _ = scores_cpu.shape
    detected = torch.argmax(scores_cpu, dim=2)
    true_idx = torch.arange(n_models).unsqueeze(0)
    correct = (detected == true_idx).float()
    idx = torch.randint(0, n_draws, (n_bootstrap, n_draws), generator=generator)
    boot_acc = correct[idx].mean(dim=(1, 2))
    return (1.0 - boot_acc).numpy()


def compute_discriminability_by_noise_level_noisy_by_clean(
    rdms: torch.Tensor,
    noise_stds: torch.Tensor,
    n_noise_samples: int,
    noise_level_multipliers: np.ndarray,
    corr_type: str,
    n_bootstrap: int | None = None,
    seed: int | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Original bootstrap discriminability, with scores = corr(noisy, clean)."""
    if n_bootstrap is None:
        n_bootstrap = int(disc.N_BOOTSTRAP_DEFAULT)

    device = noise_stds.device
    rdms = rdms.to(device)

    noise_gen = torch.Generator(device=device)
    boot_gen = torch.Generator(device="cpu")
    if seed is not None:
        noise_gen.manual_seed(int(seed))
        boot_gen.manual_seed(int(seed) + 1)
    else:
        noise_gen.seed()
        boot_gen.seed()

    n_levels = len(noise_level_multipliers)
    bootstrap_error_probs = np.empty((n_levels, n_bootstrap), dtype=np.float64)
    discriminability_by_noise_level: list[dict] = []

    clean_norm = _normalize_for_correlation(rdms.unsqueeze(0), corr_type)
    clean_norm = clean_norm.expand(n_noise_samples, -1, -1)
    for level_idx, noise_level_multiplier in enumerate(noise_level_multipliers):
        noised_rdms = (
            rdms
            + torch.randn(
                (n_noise_samples, *rdms.shape),
                device=device,
                generator=noise_gen,
            )
            * noise_stds
            * float(noise_level_multiplier)
        )
        noised_norm = _normalize_for_correlation(noised_rdms, corr_type)
        scores = _correlate_normalized(noised_norm, clean_norm)

        discriminability_by_noise_level.append(_multiclass_error_dict(scores))
        bootstrap_error_probs[level_idx] = _bootstrap_error_probs_from_scores(
            scores, n_bootstrap, boot_gen
        )

        del noised_rdms, noised_norm, scores

    del clean_norm
    return discriminability_by_noise_level, bootstrap_error_probs


def compute_correlation_at_target_noise_noisy_by_clean(
    rdms: torch.Tensor,
    noise_stds: torch.Tensor,
    corr_type: str,
    n_noise_samples: int = 100,
) -> torch.Tensor:
    """Average corr(noisy RDM, clean RDM) at a target noise level."""
    noised_rdms = (
        rdms
        + torch.randn((n_noise_samples, *rdms.shape), device=rdms.device)
        * noise_stds
    )
    clean_norm = _normalize_for_correlation(rdms.unsqueeze(0), corr_type).expand(
        n_noise_samples, -1, -1
    )
    noised_norm = _normalize_for_correlation(noised_rdms, corr_type)
    repeat_correlations = _correlate_normalized(noised_norm, clean_norm)
    return repeat_correlations.mean(dim=0).cpu()


def install_orientation_patch() -> None:
    disc.compute_discriminability_by_noise_level_with_bootstrap = (
        compute_discriminability_by_noise_level_noisy_by_clean
    )
    disc.compute_correlation_at_target_noise = compute_correlation_at_target_noise_noisy_by_clean


def _load_npz_feature_array(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        if "features" in z.files:
            arr = z["features"]
        else:
            candidates = [
                key
                for key in z.files
                if not key.startswith("_") and getattr(z[key], "ndim", 0) >= 2
            ]
            if not candidates:
                raise ValueError(f"No feature array found in {path}")
            arr = z[candidates[0]]
    return np.asarray(arr, dtype=np.float32)


def install_random_pool_loader(random_feature_dir: Path) -> None:
    def _load_random_identity_from_pool(
        payload: dict,
        model_names: list[str],
        n_random: int,
        view_name: str = "raw",
    ) -> dict[str, np.ndarray]:
        if view_name != "raw":
            print(f"  [INFO] random view '{view_name}' uses raw natural-pool features")
        features = {}
        for model_name in model_names:
            path = random_feature_dir / f"{model_name}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing random feature cache for {model_name}: {path}")
            arr = _load_npz_feature_array(path)
            features[model_name] = arr[: min(n_random, arr.shape[0])]
        print(
            f"  [DEBUG] Loaded local random pool: {len(features)} models, "
            f"{next(iter(features.values())).shape[0]} samples"
        )
        return features

    eval_utils._load_random_identity = _load_random_identity_from_pool


def available_random_models(random_feature_dir: Path, model_names: list[str]) -> list[str]:
    return [model for model in model_names if (random_feature_dir / f"{model}.npz").exists()]


def _filter_model_dict(data: Any, keep_models: list[str]) -> Any:
    if not isinstance(data, dict):
        return data
    return {model: data[model] for model in keep_models if model in data}


def filter_payload_to_models(payload: dict, keep_models: list[str]) -> dict:
    payload = dict(payload)
    payload["model_names"] = list(keep_models)

    for key in [
        "selected_features_raw",
        "greedy_features_raw",
        "best_raw_combined_features_raw",
        "selected_features",
    ]:
        if key in payload:
            payload[key] = _filter_model_dict(payload[key], keep_models)

    for key in [
        "selected_features_by_view",
        "selected_features_by_encoding",
        "greedy_features_by_encoding",
        "best_raw_combined_features_by_encoding",
    ]:
        if isinstance(payload.get(key), dict):
            payload[key] = {
                track: _filter_model_dict(features, keep_models)
                for track, features in payload[key].items()
            }

    if isinstance(payload.get("var_noise_by_model"), dict):
        payload["var_noise_by_model"] = {
            track: _filter_model_dict(noise_by_model, keep_models)
            for track, noise_by_model in payload["var_noise_by_model"].items()
        }

    return payload


def append_corr_rows(
    rows: list[dict],
    track_name: str,
    correlation_info: dict,
) -> None:
    model_names = correlation_info["model_names"]
    for matrix_type in ["selected_clean", "selected_noised", "random_clean"]:
        matrix = correlation_info[matrix_type]
        for i, model_i in enumerate(model_names):
            for j, model_j in enumerate(model_names):
                rows.append(
                    {
                        "track": track_name,
                        "matrix_type": matrix_type,
                        "model_i": model_i,
                        "model_j": model_j,
                        "correlation": matrix[i][j],
                        "recovery_orientation": ORIENTATION,
                    }
                )


def run_model_set(
    model_set: str,
    args: argparse.Namespace,
    device: torch.device,
    encoding_root_map: dict[str, Path] | None,
) -> None:
    result_dir = args.selection_root / model_set
    payload = eval_utils.load_selection_payload(result_dir)
    original_models = list(payload["model_names"])
    available_models = available_random_models(args.random_feature_dir, original_models)
    missing = sorted(set(original_models) - set(available_models))
    if missing and args.strict_random_models:
        raise FileNotFoundError(
            f"{model_set}: random feature cache is missing {len(missing)} models: {missing}"
        )
    if missing:
        print(
            f"[{model_set}] WARNING: dropping {len(missing)} models missing from "
            f"{args.random_feature_dir}: {missing}"
        )
    if len(available_models) < 2:
        raise RuntimeError(f"{model_set}: need at least 2 models after random-cache filtering")

    payload = filter_payload_to_models(payload, available_models)
    config_payload = payload.get("config", {})
    metric = args.metric or config_payload.get("metric", "cosine")
    corr_type = args.corr_type or config_payload.get("corr_type", "spearman")
    tracks = [
        track
        for track in eval_utils.get_all_tracks_for_evaluation(payload)
        if track["name"] in args.tracks
    ]
    noise_level_multipliers = disc.get_default_noise_level_multipliers()
    encoding_params_cache: dict[str, Any] = {}

    out_dir = args.output_root / f"{model_set}_noisy_by_clean_boot"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "model": original_models,
            "included": [model in available_models for model in original_models],
            "reason": [
                "included" if model in available_models else "missing_random_pool_feature"
                for model in original_models
            ],
        }
    ).to_csv(out_dir / "model_roster.csv", index=False)

    all_discrim_rows = []
    all_auc_rows = []
    all_noise_rows = []
    all_corr_rows = []

    for track_idx, track in enumerate(tracks):
        track_name = track["name"]
        print(f"\n[{model_set}] track {track_idx + 1}/{len(tracks)}: {track_name}")
        try:
            discrim_df, correlation_info, noise_info, auc_info = (
                disc.compute_discriminability_for_track(
                    payload=payload,
                    track=track,
                    device=device,
                    n_random_subsets=args.n_random_subsets,
                    n_noise_samples=args.n_noise_samples,
                    noise_level_multipliers=noise_level_multipliers,
                    metric=metric,
                    corr_type=corr_type,
                    encoding_params_cache=encoding_params_cache,
                    selection_variant=args.which_selection,
                    encoding_root_map=encoding_root_map,
                )
            )
        except Exception:
            print(f"[{model_set}/{track_name}] ERROR")
            raise

        discrim_df["model_set"] = model_set
        discrim_df["recovery_orientation"] = ORIENTATION
        discrim_df["random_feature_source"] = str(args.random_feature_dir)
        discrim_df["n_models"] = len(available_models)
        all_discrim_rows.append(discrim_df)

        all_auc_rows.append(
            {
                "track": track_name,
                "model_set": model_set,
                "recovery_orientation": ORIENTATION,
                "random_feature_source": str(args.random_feature_dir),
                "n_models": len(available_models),
                **{
                    key: value
                    for key, value in auc_info.items()
                    if key != "random_auc_per_subset"
                },
            }
        )
        for model_name, noise_std in noise_info.items():
            all_noise_rows.append(
                {
                    "track": track_name,
                    "model": model_name,
                    "noise_std": noise_std,
                    "recovery_orientation": ORIENTATION,
                }
            )
        append_corr_rows(all_corr_rows, track_name, correlation_info)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.concat(all_discrim_rows, ignore_index=True).to_csv(
        out_dir / "discriminability.csv", index=False
    )
    pd.DataFrame(all_auc_rows).to_csv(out_dir / "auc_significance.csv", index=False)
    pd.DataFrame(all_noise_rows).to_csv(out_dir / "noise_calibration.csv", index=False)
    pd.DataFrame(all_corr_rows).to_csv(out_dir / "correlation_matrices.csv", index=False)
    print(f"[{model_set}] saved {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-sets", default=",".join(MODEL_SET_ORDER))
    parser.add_argument("--tracks", default="raw," + ",".join(ENCODING_TRACKS))
    parser.add_argument("--selection-root", type=Path, default=SELECTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--random-feature-dir", type=Path, default=DEFAULT_RANDOM_FEATURE_DIR)
    parser.add_argument("--strict-random-models", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random-subsets", type=int, default=50)
    parser.add_argument("--n-noise-samples", type=int, default=disc.DEFAULT_N_NOISE_SAMPLES)
    parser.add_argument("--n-bootstrap", type=int, default=disc.N_BOOTSTRAP_DEFAULT)
    parser.add_argument(
        "--which-selection",
        choices=["final", "greedy", "best_raw_combined"],
        default="final",
    )
    parser.add_argument("--unique-encodings", action="store_true", default=True)
    parser.add_argument("--shared-encodings", action="store_false", dest="unique_encodings")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--corr-type", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_sets = [item.strip() for item in args.model_sets.split(",") if item.strip()]
    args.tracks = [item.strip() for item in args.tracks.split(",") if item.strip()]
    args.selection_root = args.selection_root.resolve()
    args.output_root = args.output_root.resolve()
    args.random_feature_dir = args.random_feature_dir.resolve()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    disc.N_BOOTSTRAP_DEFAULT = int(args.n_bootstrap)

    if not args.random_feature_dir.exists():
        raise FileNotFoundError(f"Random feature directory not found: {args.random_feature_dir}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    install_orientation_patch()
    install_random_pool_loader(args.random_feature_dir)

    encoding_root_map = None
    if args.unique_encodings:
        encoding_root_map = {key: Path(value) for key, value in paper_config.UNIQUE_ENCODING_DIRS.items()}
        print(f"Using unique encoding roots: {list(encoding_root_map)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for model_set in tqdm(args.model_sets, desc="Model sets"):
        run_model_set(model_set, args, device, encoding_root_map)


if __name__ == "__main__":
    main()
