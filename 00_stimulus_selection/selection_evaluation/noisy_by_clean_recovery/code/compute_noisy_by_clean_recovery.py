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
import yaml
from tqdm import tqdm

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[4]
ANALYSIS_DIR = (
    ROOT
    / "00_stimulus_selection"
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
ENV_CONFIG_ROOT = ROOT / "00_stimulus_selection" / "resources" / "configs" / "paths"
PAIRWISE_METRIC_CALLS: list[dict[str, np.ndarray]] = []


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


def _score_metrics_from_scores(
    scores: torch.Tensor,
    n_bootstrap: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    scores_cpu = scores.cpu()
    n_draws, n_models, _ = scores_cpu.shape
    detected = torch.argmax(scores_cpu, dim=2)
    true_idx = torch.arange(n_models).unsqueeze(0)
    correct = (detected == true_idx).float()

    diag_scores = scores_cpu.diagonal(dim1=1, dim2=2)
    pairwise_margin = diag_scores.unsqueeze(2) - scores_cpu
    off_diag_mask = ~torch.eye(n_models, dtype=torch.bool)
    pairwise_margin = pairwise_margin[:, off_diag_mask]
    pairwise_dominance = (
        (pairwise_margin > 0).float()
        + 0.5 * torch.isclose(pairwise_margin, torch.zeros_like(pairwise_margin)).float()
    )

    point_error = float(1.0 - correct.mean().item())
    point_dominance = float(pairwise_dominance.mean().item())
    point_margin = float(pairwise_margin.mean().item())

    idx = torch.randint(0, n_draws, (n_bootstrap, n_draws), generator=generator)
    boot_acc = correct[idx].mean(dim=(1, 2))
    boot_dominance = pairwise_dominance[idx].mean(dim=(1, 2))
    boot_margin = pairwise_margin[idx].mean(dim=(1, 2))

    return {
        "error_prob": point_error,
        "pairwise_dominance": point_dominance,
        "pairwise_error_prob": 1.0 - point_dominance,
        "mean_margin": point_margin,
        "error_prob_boot": (1.0 - boot_acc).numpy(),
        "pairwise_dominance_boot": boot_dominance.numpy(),
        "mean_margin_boot": boot_margin.numpy(),
    }


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
    pairwise_dominance = np.empty(n_levels, dtype=np.float64)
    pairwise_error_prob = np.empty(n_levels, dtype=np.float64)
    mean_margin = np.empty(n_levels, dtype=np.float64)
    pairwise_dominance_boot = np.empty((n_levels, n_bootstrap), dtype=np.float64)
    mean_margin_boot = np.empty((n_levels, n_bootstrap), dtype=np.float64)
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
        metrics = _score_metrics_from_scores(scores, n_bootstrap, boot_gen)

        discriminability_by_noise_level.append(
            {
                "non_parametric_multiclass_error_prob": metrics["error_prob"],
                "non_parametric_pairwise_error_prob": metrics["pairwise_error_prob"],
                "pairwise_dominance": metrics["pairwise_dominance"],
                "mean_margin": metrics["mean_margin"],
            }
        )
        bootstrap_error_probs[level_idx] = metrics["error_prob_boot"]
        pairwise_dominance[level_idx] = metrics["pairwise_dominance"]
        pairwise_error_prob[level_idx] = metrics["pairwise_error_prob"]
        mean_margin[level_idx] = metrics["mean_margin"]
        pairwise_dominance_boot[level_idx] = metrics["pairwise_dominance_boot"]
        mean_margin_boot[level_idx] = metrics["mean_margin_boot"]

        del noised_rdms, noised_norm, scores, metrics

    del clean_norm
    PAIRWISE_METRIC_CALLS.append(
        {
            "noise_mult": np.asarray(noise_level_multipliers, dtype=float),
            "pairwise_dominance": pairwise_dominance,
            "pairwise_error_prob": pairwise_error_prob,
            "mean_margin": mean_margin,
            "pairwise_dominance_boot": pairwise_dominance_boot,
            "mean_margin_boot": mean_margin_boot,
        }
    )
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


def load_repo_env_paths(env: str) -> dict[str, Any]:
    config_path = ENV_CONFIG_ROOT / f"{env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Environment config not found: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f) or {}

    paths = dict(config.get("paths", {}))

    # Keep repo-internal metadata tied to the checked-out repository. The Raven
    # YAML still carries the old /u/rothj/cstims clone path in some fields.
    local_model_csv = ROOT / "00_stimulus_selection" / "resources" / "model_list.csv"
    if local_model_csv.exists():
        paths["model_list_csv"] = str(local_model_csv)
    paths["output_base"] = str(SELECTION_ROOT)

    return paths


def apply_env_paths(payload: dict, env: str | None) -> dict:
    if not env:
        return payload

    env_paths = load_repo_env_paths(env)
    payload = dict(payload)
    config = dict(payload.get("config", {}))
    old_paths = dict(config.get("paths", {}))
    eval_utils._warn_path_divergence(old_paths, env_paths, env)
    merged_paths = {**old_paths, **env_paths}
    config["paths"] = merged_paths
    payload["config"] = config
    print(f"Using paths from env={env}: {ENV_CONFIG_ROOT / f'{env}.yaml'}")
    return payload


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


def _ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _bootstrap_auc(
    values_boot: np.ndarray,
    noise_mult: np.ndarray,
) -> np.ndarray:
    sort_idx = np.argsort(noise_mult.astype(float))
    x_sorted = noise_mult[sort_idx]
    return np.asarray(
        [
            disc.compute_auc(x_sorted, values_boot[sort_idx, boot_idx])
            for boot_idx in range(values_boot.shape[1])
        ],
        dtype=np.float64,
    )


def append_pairwise_rows(
    rows: list[dict],
    auc_rows: list[dict],
    calls: list[dict[str, np.ndarray]],
    *,
    model_set: str,
    track: dict,
    metric: str,
    corr_type: str,
    target_nc: float,
    random_feature_source: str,
    n_models: int,
) -> None:
    if not calls:
        return

    selected = calls[0]
    random_calls = calls[1:]
    noise_mult = selected["noise_mult"]
    track_name = track["name"]
    track_type = track.get("type", "identity")

    selected_dom_auc = disc.compute_auc(noise_mult, selected["pairwise_dominance"])
    selected_margin_auc = disc.compute_auc(noise_mult, selected["mean_margin"])
    selected_dom_auc_boot = _bootstrap_auc(
        selected["pairwise_dominance_boot"], noise_mult
    )
    selected_margin_auc_boot = _bootstrap_auc(selected["mean_margin_boot"], noise_mult)

    if random_calls:
        random_dom = np.stack([call["pairwise_dominance"] for call in random_calls])
        random_margin = np.stack([call["mean_margin"] for call in random_calls])
        random_dom_boot = np.stack(
            [call["pairwise_dominance_boot"] for call in random_calls]
        )
        random_margin_boot = np.stack(
            [call["mean_margin_boot"] for call in random_calls]
        )
        random_dom_mean = random_dom.mean(axis=0)
        random_margin_mean = random_margin.mean(axis=0)
        random_dom_subset_std = random_dom.std(axis=0, ddof=1)
        random_margin_subset_std = random_margin.std(axis=0, ddof=1)
        random_dom_mc_std = random_dom_boot.mean(axis=0).std(axis=1, ddof=1)
        random_margin_mc_std = random_margin_boot.mean(axis=0).std(axis=1, ddof=1)
        random_dom_auc_per_subset = np.asarray(
            [disc.compute_auc(noise_mult, curve) for curve in random_dom],
            dtype=np.float64,
        )
        random_margin_auc_per_subset = np.asarray(
            [disc.compute_auc(noise_mult, curve) for curve in random_margin],
            dtype=np.float64,
        )
        random_dom_auc_boot = _bootstrap_auc(random_dom_boot.mean(axis=0), noise_mult)
        random_margin_auc_boot = _bootstrap_auc(
            random_margin_boot.mean(axis=0), noise_mult
        )
    else:
        random_dom_mean = np.full_like(noise_mult, np.nan, dtype=np.float64)
        random_margin_mean = np.full_like(noise_mult, np.nan, dtype=np.float64)
        random_dom_subset_std = np.full_like(noise_mult, np.nan, dtype=np.float64)
        random_margin_subset_std = np.full_like(noise_mult, np.nan, dtype=np.float64)
        random_dom_mc_std = np.full_like(noise_mult, np.nan, dtype=np.float64)
        random_margin_mc_std = np.full_like(noise_mult, np.nan, dtype=np.float64)
        random_dom_auc_per_subset = np.asarray([np.nan], dtype=np.float64)
        random_margin_auc_per_subset = np.asarray([np.nan], dtype=np.float64)
        random_dom_auc_boot = np.asarray([np.nan], dtype=np.float64)
        random_margin_auc_boot = np.asarray([np.nan], dtype=np.float64)

    for level_idx, multiplier in enumerate(noise_mult):
        noise_ceiling = disc.multiplier_to_noise_ceiling(float(multiplier), target_nc)
        dom_lo, dom_hi = _ci(selected["pairwise_dominance_boot"][level_idx])
        margin_lo, margin_hi = _ci(selected["mean_margin_boot"][level_idx])

        rows.append(
            {
                "track": track_name,
                "track_type": track_type,
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": multiplier,
                "noise_ceiling": noise_ceiling,
                "subset_type": "selected",
                "pairwise_dominance": selected["pairwise_dominance"][level_idx],
                "pairwise_dominance_subset_std": np.nan,
                "pairwise_dominance_mc_std": selected["pairwise_dominance_boot"][
                    level_idx
                ].std(ddof=1),
                "pairwise_dominance_mc_ci_lo": dom_lo,
                "pairwise_dominance_mc_ci_hi": dom_hi,
                "pairwise_error_prob": 1.0
                - selected["pairwise_dominance"][level_idx],
                "mean_margin": selected["mean_margin"][level_idx],
                "mean_margin_subset_std": np.nan,
                "mean_margin_mc_std": selected["mean_margin_boot"][level_idx].std(
                    ddof=1
                ),
                "mean_margin_mc_ci_lo": margin_lo,
                "mean_margin_mc_ci_hi": margin_hi,
                "model_set": model_set,
                "recovery_orientation": ORIENTATION,
                "random_feature_source": random_feature_source,
                "n_models": n_models,
            }
        )

        rows.append(
            {
                "track": track_name,
                "track_type": track_type,
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": multiplier,
                "noise_ceiling": noise_ceiling,
                "subset_type": "random",
                "pairwise_dominance": random_dom_mean[level_idx],
                "pairwise_dominance_subset_std": random_dom_subset_std[level_idx],
                "pairwise_dominance_mc_std": random_dom_mc_std[level_idx],
                "pairwise_dominance_mc_ci_lo": np.nan,
                "pairwise_dominance_mc_ci_hi": np.nan,
                "pairwise_error_prob": 1.0 - random_dom_mean[level_idx],
                "mean_margin": random_margin_mean[level_idx],
                "mean_margin_subset_std": random_margin_subset_std[level_idx],
                "mean_margin_mc_std": random_margin_mc_std[level_idx],
                "mean_margin_mc_ci_lo": np.nan,
                "mean_margin_mc_ci_hi": np.nan,
                "model_set": model_set,
                "recovery_orientation": ORIENTATION,
                "random_feature_source": random_feature_source,
                "n_models": n_models,
            }
        )

    dom_lo, dom_hi = _ci(selected_dom_auc_boot)
    margin_lo, margin_hi = _ci(selected_margin_auc_boot)
    auc_rows.append(
        {
            "track": track_name,
            "model_set": model_set,
            "recovery_orientation": ORIENTATION,
            "random_feature_source": random_feature_source,
            "n_models": n_models,
            "selected_pairwise_dominance_auc": selected_dom_auc,
            "selected_pairwise_dominance_auc_mc_std": selected_dom_auc_boot.std(
                ddof=1
            ),
            "selected_pairwise_dominance_auc_mc_ci_lo": dom_lo,
            "selected_pairwise_dominance_auc_mc_ci_hi": dom_hi,
            "random_pairwise_dominance_auc_mean": random_dom_auc_per_subset.mean(),
            "random_pairwise_dominance_auc_subset_std": random_dom_auc_per_subset.std(
                ddof=1
            ),
            "random_pairwise_dominance_auc_mc_std": random_dom_auc_boot.std(ddof=1),
            "selected_mean_margin_auc": selected_margin_auc,
            "selected_mean_margin_auc_mc_std": selected_margin_auc_boot.std(ddof=1),
            "selected_mean_margin_auc_mc_ci_lo": margin_lo,
            "selected_mean_margin_auc_mc_ci_hi": margin_hi,
            "random_mean_margin_auc_mean": random_margin_auc_per_subset.mean(),
            "random_mean_margin_auc_subset_std": random_margin_auc_per_subset.std(
                ddof=1
            ),
            "random_mean_margin_auc_mc_std": random_margin_auc_boot.std(ddof=1),
            "pairwise_dominance_auc_z_score": (
                (selected_dom_auc - random_dom_auc_per_subset.mean())
                / random_dom_auc_per_subset.std(ddof=1)
                if random_dom_auc_per_subset.std(ddof=1) > 0
                else float("nan")
            ),
            "mean_margin_auc_z_score": (
                (selected_margin_auc - random_margin_auc_per_subset.mean())
                / random_margin_auc_per_subset.std(ddof=1)
                if random_margin_auc_per_subset.std(ddof=1) > 0
                else float("nan")
            ),
            "pairwise_dominance_p_value_empirical": (
                (int(np.sum(random_dom_auc_per_subset >= selected_dom_auc)) + 1)
                / (len(random_dom_auc_per_subset) + 1)
            ),
            "mean_margin_p_value_empirical": (
                (int(np.sum(random_margin_auc_per_subset >= selected_margin_auc)) + 1)
                / (len(random_margin_auc_per_subset) + 1)
            ),
        }
    )


def run_model_set(
    model_set: str,
    args: argparse.Namespace,
    device: torch.device,
    encoding_root_map: dict[str, Path] | None,
) -> None:
    result_dir = args.selection_root / model_set
    payload = apply_env_paths(eval_utils.load_selection_payload(result_dir), args.env)
    original_models = list(payload["model_names"])

    if args.random_feature_dir is None:
        available_models = original_models
        missing: list[str] = []
        random_feature_source = f"candidate_pool:env-{args.env}" if args.env else "candidate_pool:payload_paths"
    else:
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
        random_feature_source = f"local_random_pool:{args.random_feature_dir}"

    config_payload = payload.get("config", {})
    metric = args.metric or config_payload.get("metric", "cosine")
    corr_type = args.corr_type or config_payload.get("corr_type", "spearman")
    target_nc = config_payload.get("noise_ceiling_target", 0.46)
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
            "random_feature_source": random_feature_source,
        }
    ).to_csv(out_dir / "model_roster.csv", index=False)

    all_discrim_rows = []
    all_auc_rows = []
    all_noise_rows = []
    all_corr_rows = []
    all_pairwise_rows = []
    all_pairwise_auc_rows = []

    for track_idx, track in enumerate(tracks):
        track_name = track["name"]
        print(f"\n[{model_set}] track {track_idx + 1}/{len(tracks)}: {track_name}")
        PAIRWISE_METRIC_CALLS.clear()
        try:
            discrim_df, correlation_info, noise_info, auc_info = (
                disc.compute_discriminability_for_track(
                    payload=payload,
                    track=track,
                    device=device,
                    n_random_subsets=args.n_random_subsets,
                    n_random_images=args.n_random_images,
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

        append_pairwise_rows(
            all_pairwise_rows,
            all_pairwise_auc_rows,
            list(PAIRWISE_METRIC_CALLS),
            model_set=model_set,
            track=track,
            metric=metric,
            corr_type=corr_type,
            target_nc=target_nc,
            random_feature_source=random_feature_source,
            n_models=len(available_models),
        )
        PAIRWISE_METRIC_CALLS.clear()

        discrim_df["model_set"] = model_set
        discrim_df["recovery_orientation"] = ORIENTATION
        discrim_df["random_feature_source"] = random_feature_source
        discrim_df["n_models"] = len(available_models)
        all_discrim_rows.append(discrim_df)

        all_auc_rows.append(
            {
                "track": track_name,
                "model_set": model_set,
                "recovery_orientation": ORIENTATION,
                "random_feature_source": random_feature_source,
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
    pd.DataFrame(all_pairwise_rows).to_csv(out_dir / "pairwise_margin.csv", index=False)
    pd.DataFrame(all_pairwise_auc_rows).to_csv(out_dir / "pairwise_auc.csv", index=False)
    print(f"[{model_set}] saved {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-sets", default=",".join(MODEL_SET_ORDER))
    parser.add_argument("--tracks", default="raw," + ",".join(ENCODING_TRACKS))
    parser.add_argument("--selection-root", type=Path, default=SELECTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--env",
        choices=eval_utils.VALID_ENVS,
        default=None,
        help=(
            "Override payload paths with the repo env config under "
            "00_stimulus_selection/resources/configs/paths/{env}.yaml. "
            "Use --env raven for the full Raven candidate-pool rerun."
        ),
    )
    parser.add_argument(
        "--random-feature-dir",
        type=Path,
        default=None,
        help=(
            "Optional debug-only local .npz random-feature cache. Omit this for the "
            "proper candidate-pool baseline loaded from the payload/env paths. "
            f"Local smoke-test cache, if present: {DEFAULT_RANDOM_FEATURE_DIR}"
        ),
    )
    parser.add_argument(
        "--strict-random-models",
        action="store_true",
        help="Fail if --random-feature-dir is provided and is missing any selected model.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random-subsets", type=int, default=50)
    parser.add_argument(
        "--n-random-images",
        type=int,
        default=10000,
        help="Number of random baseline images to load from candidate pool/cache before subset sampling.",
    )
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
    if args.random_feature_dir is not None:
        args.random_feature_dir = args.random_feature_dir.resolve()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    disc.N_BOOTSTRAP_DEFAULT = int(args.n_bootstrap)

    if args.random_feature_dir is not None and not args.random_feature_dir.exists():
        raise FileNotFoundError(f"Random feature directory not found: {args.random_feature_dir}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    install_orientation_patch()
    if args.random_feature_dir is None:
        print("Using candidate-pool random baseline from payload/env paths")
    else:
        install_random_pool_loader(args.random_feature_dir)
        print(f"Using debug local random-feature cache: {args.random_feature_dir}")

    encoding_root_map = None
    if args.unique_encodings:
        encoding_root_map = {key: Path(value) for key, value in paper_config.UNIQUE_ENCODING_DIRS.items()}
        print(f"Using unique encoding roots: {list(encoding_root_map)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for model_set in tqdm(args.model_sets, desc="Model sets"):
        run_model_set(model_set, args, device, encoding_root_map)


if __name__ == "__main__":
    main()
