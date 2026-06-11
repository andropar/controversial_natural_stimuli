#!/usr/bin/env python3
"""Compute discriminability metrics for stimulus selection results.

Outputs:
- discriminability.csv: Discriminability metrics across noise levels and tracks
- correlation_matrices.csv: Correlation matrices (selected and random) per track
- noise_calibration.csv: Noise calibration parameters per track/model
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Ensure repository modules are on path.
SHARE_ROOT = Path(__file__).resolve().parents[4]
STIMULUS_ROOT = SHARE_ROOT / "00_stimulus_selection"
EVAL_DIR = Path(__file__).resolve().parent
for path in (SHARE_ROOT / "src", SHARE_ROOT, STIMULUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
# Use local utils (with unique encoding support) over scripts/eval/utils
sys.path.insert(0, str(EVAL_DIR))

from cstims.evaluation.computation import (
    compute_all_rdms,
    compute_clean_correlation_matrix,
    compute_correlation_at_target_noise,
    compute_discriminability_by_noise_level,
    compute_random_baseline_rdms,
)
from cstims.evaluation.constants import (
    DEFAULT_N_NOISE_SAMPLES,
    get_default_noise_level_multipliers,
)
from cstims.evaluation.model_discrimination import model_discriminability
from cstims.evaluation.noise_calibration import calibrate_noise_parameters
from cstims.selection.primitives import compute_correlation_matrix


# Number of bootstrap resamples for MC-noise-draw uncertainty on error_prob / AUC.
N_BOOTSTRAP_DEFAULT = 500


def _multiclass_error_prob_from_scores(model_score: torch.Tensor) -> float:
    """Vectorized non-parametric multiclass error prob from [n_sim, M, M] tensor."""
    detected = torch.argmax(model_score, dim=2)  # [n_sim, M]
    M = model_score.shape[1]
    true_idx = torch.arange(M, device=model_score.device).unsqueeze(0)  # [1, M]
    correct = (detected == true_idx).float().mean(dim=0)  # [M]
    return float(1.0 - correct.mean().item())


def compute_discriminability_by_noise_level_with_bootstrap(
    rdms: torch.Tensor,
    noise_stds: torch.Tensor,
    n_noise_samples: int,
    noise_level_multipliers: np.ndarray,
    corr_type: str,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Like compute_discriminability_by_noise_level but also returns per-level
    bootstrap distributions over the Monte Carlo noise-draw dimension.

    Returns:
        discriminability_by_noise_level: list of dicts (as returned by
            model_discriminability), one per noise level (point estimate over
            all n_noise_samples draws).
        bootstrap_error_probs: array of shape [n_levels, n_bootstrap] with the
            non-parametric multiclass error prob computed on each bootstrap
            resample of the n_noise_samples draws.
    """
    device = noise_stds.device
    rdms = rdms.to(device)

    # Dedicated RNGs for bit-reproducibility regardless of caller-side state.
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

    for li, noise_level_multiplier in enumerate(noise_level_multipliers):
        noised_rdms = (
            rdms
            + torch.randn(
                (n_noise_samples, *rdms.shape),
                device=device,
                generator=noise_gen,
            )
            * noise_stds
            * noise_level_multiplier
        )
        # [n_noise_samples, M, M]
        scores = compute_correlation_matrix(
            rdms.repeat(n_noise_samples, 1, 1), noised_rdms, corr_type
        )

        # Point estimate (all draws)
        discriminability_by_noise_level.append(model_discriminability(scores))

        # Bootstrap over the n_noise_samples dimension
        scores_cpu = scores.cpu()
        N = scores_cpu.shape[0]
        # Resample indices [n_bootstrap, N]
        idx = torch.randint(0, N, (n_bootstrap, N), generator=boot_gen)
        for b in range(n_bootstrap):
            bootstrap_error_probs[li, b] = _multiclass_error_prob_from_scores(
                scores_cpu[idx[b]]
            )

        del noised_rdms, scores, scores_cpu

    return discriminability_by_noise_level, bootstrap_error_probs
from utils import (
    add_standard_args,
    get_all_tracks_for_evaluation,
    get_output_dir,
    get_track_noise_variances,
    load_features_for_track,
    log_memory,
    setup_from_args,
)


def multiplier_to_noise_ceiling(k: float, nc_base: float) -> float:
    """Convert noise multiplier to effective noise ceiling.

    The relationship is derived from: var_noise = var_rdm * (1/nc^2 - 1)
    At multiplier k, var_noise_eff = k^2 * var_noise_base

    Args:
        k: Noise multiplier
        nc_base: Base noise ceiling (at k=1)

    Returns:
        Effective noise ceiling at multiplier k
    """
    if k <= 0:
        return 1.0
    if nc_base <= 0 or nc_base >= 1:
        return nc_base

    # nc_eff = 1 / sqrt(1 + k^2 * (1/nc_base^2 - 1))
    term = k * k * (1.0 / (nc_base * nc_base) - 1.0)
    return 1.0 / np.sqrt(1.0 + term)


def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute normalized AUC using trapezoidal rule on log-scale x.

    The AUC is normalized by dividing by the span of log10(x) so that
    the result is between 0 and 1 (assuming y is between 0 and 1).

    Args:
        x: Noise multiplier values
        y: Error probability values

    Returns:
        Normalized AUC value (between 0 and 1)
    """
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Use log-scale x for integration
    x_log = np.log10(x_sorted + 1e-10)

    # Compute raw AUC
    raw_auc = float(np.trapz(y_sorted, x_log))

    # Normalize by the span of log10(x) to get value between 0 and 1
    log_span = x_log[-1] - x_log[0]
    if log_span > 0:
        normalized_auc = raw_auc / log_span
    else:
        normalized_auc = raw_auc

    return normalized_auc


def compute_discriminability_for_track(
    payload: dict,
    track: dict,
    device: torch.device,
    n_random_subsets: int,
    n_random_images: int,
    n_noise_samples: int,
    noise_level_multipliers: np.ndarray,
    metric: str,
    corr_type: str,
    encoding_params_cache: dict[str, Any],
    selection_variant: str = "final",
    encoding_root_map: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Compute discriminability metrics for a single track.

    Args:
        payload: Selection payload
        track: Track definition dict
        device: Torch device
        n_random_subsets: Number of random baseline subsets
        n_random_images: Number of random baseline images to load before subset sampling
        n_noise_samples: Number of noise samples per level
        noise_level_multipliers: Array of noise multipliers
        metric: RDM metric (euclidean, cosine, correlation)
        corr_type: Correlation type (spearman, pearson)
        encoding_params_cache: Cache for encoding parameters
        selection_variant: Which selection to evaluate ("final", "greedy", "best_raw_combined")
        encoding_root_map: Optional per-encoding root mapping for unique encodings

    Returns:
        Tuple of (discriminability_df, correlation_info, noise_info)
    """
    track_name = track["name"]
    track_type = track.get("type", "identity")
    model_names = payload["model_names"]

    print(f"  Loading features for track '{track_name}' (type={track_type}, variant={selection_variant})...")
    log_memory("start_track")

    # Load selected and random features
    selected_features, random_features = load_features_for_track(
        payload=payload,
        track=track,
        device=device,
        encoding_params_cache=encoding_params_cache,
        n_random=n_random_images,
        selection_variant=selection_variant,
        encoding_root_map=encoding_root_map,
    )

    n_selected = next(iter(selected_features.values())).shape[0]
    log_memory("after_load_features")
    print(f"  [DEBUG] Loaded {n_selected} selected stimuli, {len(model_names)} models")

    # Calibrate noise parameters
    print(f"  Calibrating noise for track '{track_name}'...")
    config = payload.get("config", {})
    target_nc = config.get("noise_ceiling_target", 0.46)

    # Check if we have pre-computed noise variances.
    # `get_track_noise_variances` returns variances (sigma^2); sqrt to get stds
    # for injection as `randn * std`. Cf. NoiseParameters.get_noise_stds in
    # cstims/evaluation/results.py, which is the canonical accessor.
    precomputed_noise = get_track_noise_variances(payload, track_name)
    if precomputed_noise is not None:
        # Use precomputed noise (stored as variance; convert to std)
        noise_stds = torch.stack(
            [
                torch.tensor(precomputed_noise[m], device=device, dtype=torch.float32).sqrt()
                for m in model_names
            ]
        ).unsqueeze(1)
        noise_info = {m: float(noise_stds[i].item()) for i, m in enumerate(model_names)}
    else:
        # Calibrate noise (also returns variance; use get_noise_stds helper)
        noise_params = calibrate_noise_parameters(
            features=random_features,
            model_names=model_names,
            metrics=[metric],
            target_nc=target_nc,
            device=device,
            mode="analytical",
        )
        noise_stds = noise_params.get_noise_stds(metric)
        noise_info = {m: float(noise_stds[i].item()) for i, m in enumerate(model_names)}

    # Compute RDMs for selected stimuli
    print(f"  Computing RDMs for selected stimuli...")
    selected_rdms = compute_all_rdms(selected_features, [metric])
    log_memory("after_selected_rdms")

    # Free selected features - no longer needed after RDMs are computed
    del selected_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log_memory("after_cleanup_selected")

    # Compute RDMs for random baselines
    print(f"  Computing RDMs for {n_random_subsets} random subsets...")
    random_rdms = compute_random_baseline_rdms(
        random_features=random_features,
        model_names=model_names,
        metrics=[metric],
        n_selected_stimuli=n_selected,
        n_random_subsets=n_random_subsets,
        device=device,
    )

    # Free random features - no longer needed after RDMs are computed
    del random_features
    gc.collect()
    log_memory("after_random_rdms")

    # Compute clean correlation matrices
    print(f"  Computing correlation matrices...")
    selected_corr = compute_clean_correlation_matrix(selected_rdms[metric], corr_type)

    # Compute random correlation matrix (average across subsets)
    random_corr_list = []
    for random_rdm in random_rdms:
        corr = compute_clean_correlation_matrix(random_rdm[metric], corr_type)
        random_corr_list.append(corr.cpu())  # Keep on CPU to save GPU memory
    random_corr = torch.stack(random_corr_list).mean(dim=0)
    del random_corr_list
    gc.collect()

    # Compute correlation at target noise level
    selected_corr_noised = compute_correlation_at_target_noise(
        selected_rdms[metric], noise_stds, corr_type, n_noise_samples
    )

    # Compute discriminability across noise levels
    print(
        f"  Computing discriminability across {len(noise_level_multipliers)} noise levels..."
    )
    log_memory("before_discriminability")

    # Selected discriminability (with MC bootstrap over noise-draw dimension)
    print(
        f"  [DEBUG] Computing selected discriminability (RDM shape: {selected_rdms[metric].shape})..."
    )
    # Deterministic per-call seeds (different for selected vs each random subset)
    # so runs are bit-reproducible regardless of call order.
    track_seed = (hash(track_name) & 0xFFFFFFFF) ^ 0xC57A1EE
    selected_discrim, selected_boot = (
        compute_discriminability_by_noise_level_with_bootstrap(
            rdms=selected_rdms[metric],
            noise_stds=noise_stds,
            n_noise_samples=n_noise_samples,
            noise_level_multipliers=noise_level_multipliers,
            corr_type=corr_type,
            seed=track_seed,
        )
    )
    # selected_boot: [n_levels, n_bootstrap]
    log_memory("after_selected_discrim")

    # Random discriminability (compute for each subset, then aggregate)
    # Also collect MC bootstrap distributions per subset for AUC significance.
    print(
        f"  [DEBUG] Computing random discriminability for {len(random_rdms)} subsets..."
    )
    random_discrim_rows = []
    n_levels = len(noise_level_multipliers)
    # [n_random_subsets, n_levels, n_bootstrap]
    random_boot_all = np.empty(
        (len(random_rdms), n_levels, N_BOOTSTRAP_DEFAULT), dtype=np.float64
    )
    for subset_idx, random_rdm in enumerate(
        tqdm(random_rdms, desc="Random subsets", leave=False)
    ):
        subset_discrim, subset_boot = (
            compute_discriminability_by_noise_level_with_bootstrap(
                rdms=random_rdm[metric],
                noise_stds=noise_stds,
                n_noise_samples=n_noise_samples,
                noise_level_multipliers=noise_level_multipliers,
                corr_type=corr_type,
                seed=track_seed + 1 + subset_idx,
            )
        )
        random_boot_all[subset_idx] = subset_boot
        for level_idx, level_results in enumerate(subset_discrim):
            noise_mult = noise_level_multipliers[level_idx]
            error_prob = float(
                level_results.get("non_parametric_multiclass_error_prob", 0)
            )
            mc_std = float(subset_boot[level_idx].std(ddof=1))
            mc_ci_lo = float(np.quantile(subset_boot[level_idx], 0.025))
            mc_ci_hi = float(np.quantile(subset_boot[level_idx], 0.975))
            random_discrim_rows.append(
                {
                    "subset_idx": subset_idx,
                    "noise_mult": noise_mult,
                    "error_prob": error_prob,
                    "error_prob_mc_std": mc_std,
                    "error_prob_mc_ci_lo": mc_ci_lo,
                    "error_prob_mc_ci_hi": mc_ci_hi,
                }
            )
        # Periodic cleanup to prevent memory accumulation
        if subset_idx % 10 == 9:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # Free random RDMs after discriminability computation
    del random_rdms
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log_memory("after_random_discrim_cleanup")
    print(f"  [DEBUG] Finished discriminability computation for track '{track_name}'")

    # Build discriminability DataFrame
    discrim_rows = []

    # Selected rows
    for level_idx, level_results in enumerate(selected_discrim):
        noise_mult = noise_level_multipliers[level_idx]
        noise_ceiling = multiplier_to_noise_ceiling(noise_mult, target_nc)
        error_prob = float(level_results.get("non_parametric_multiclass_error_prob", 0))

        # Mean off-diagonal correlation
        corr_matrix = compute_correlation_at_target_noise(
            selected_rdms[metric],
            noise_stds * noise_mult,
            corr_type,
            n_noise_samples,
        )
        off_diag_mask = ~torch.eye(len(model_names), dtype=torch.bool)
        mean_offdiag = float(corr_matrix[off_diag_mask].mean())

        mc_std = float(selected_boot[level_idx].std(ddof=1))
        mc_ci_lo = float(np.quantile(selected_boot[level_idx], 0.025))
        mc_ci_hi = float(np.quantile(selected_boot[level_idx], 0.975))

        discrim_rows.append(
            {
                "track": track_name,
                "track_type": track_type,
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": noise_mult,
                "noise_ceiling": noise_ceiling,
                "subset_type": "selected",
                "error_prob": error_prob,
                "error_prob_std": np.nan,  # N/A for selected (single set)
                "error_prob_mc_std": mc_std,
                "error_prob_mc_ci_lo": mc_ci_lo,
                "error_prob_mc_ci_hi": mc_ci_hi,
                "mean_offdiag_corr": mean_offdiag,
            }
        )

    # Random rows (aggregate across subsets with std for error bars).
    # MC std/CI: average across subsets (within-subset noise-draw variance).
    random_df = pd.DataFrame(random_discrim_rows)
    for noise_mult, group in random_df.groupby("noise_mult"):
        mean_error = group["error_prob"].mean()
        std_error = group["error_prob"].std()
        noise_ceiling = multiplier_to_noise_ceiling(noise_mult, target_nc)
        discrim_rows.append(
            {
                "track": track_name,
                "track_type": track_type,
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": noise_mult,
                "noise_ceiling": noise_ceiling,
                "subset_type": "random",
                "error_prob": mean_error,
                "error_prob_std": std_error,
                "error_prob_mc_std": group["error_prob_mc_std"].mean(),
                "error_prob_mc_ci_lo": group["error_prob_mc_ci_lo"].mean(),
                "error_prob_mc_ci_hi": group["error_prob_mc_ci_hi"].mean(),
                "mean_offdiag_corr": np.nan,  # Not computed for random
            }
        )

    discrim_df = pd.DataFrame(discrim_rows)

    # Compute AUC for selected and random
    selected_data = discrim_df[discrim_df["subset_type"] == "selected"]
    random_data = discrim_df[discrim_df["subset_type"] == "random"]

    selected_auc = compute_auc(
        selected_data["noise_mult"].values,
        selected_data["error_prob"].values,
    )
    random_auc = compute_auc(
        random_data["noise_mult"].values,
        random_data["error_prob"].values,
    )

    # Add AUC column
    discrim_df["auc"] = discrim_df["subset_type"].apply(
        lambda x: selected_auc if x == "selected" else random_auc
    )

    # ---- AUC bootstrap distributions ----
    # Selected: bootstrap draws are independent across noise levels (fresh noise
    # per level), so we can combine draw b across levels to form one AUC sample.
    x_levels = np.array(noise_level_multipliers, dtype=float)
    sort_idx = np.argsort(x_levels)
    x_sorted = x_levels[sort_idx]

    def _auc_from_curve(y_per_level: np.ndarray) -> float:
        return compute_auc(x_sorted, y_per_level[sort_idx])

    # selected_boot: [n_levels, n_bootstrap]
    selected_auc_boot = np.array(
        [_auc_from_curve(selected_boot[:, b]) for b in range(selected_boot.shape[1])]
    )
    # Per-subset random AUC from point estimates (one per random subset)
    random_per_subset_auc = np.array(
        [
            _auc_from_curve(
                random_df[random_df["subset_idx"] == s]
                .sort_values("noise_mult")["error_prob"]
                .values
            )
            for s in sorted(random_df["subset_idx"].unique())
        ]
    )
    # Random AUC bootstrap: for each MC bootstrap b, average across subsets
    # (mirrors how the plotted random curve is the across-subset mean).
    # random_boot_all: [n_subsets, n_levels, n_bootstrap]
    n_boot = random_boot_all.shape[2]
    random_auc_boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        mean_curve = random_boot_all[:, :, b].mean(axis=0)  # [n_levels]
        random_auc_boot[b] = _auc_from_curve(mean_curve)

    auc_info = {
        "selected_auc": float(selected_auc),
        "selected_auc_mc_std": float(selected_auc_boot.std(ddof=1)),
        "selected_auc_mc_ci_lo": float(np.quantile(selected_auc_boot, 0.025)),
        "selected_auc_mc_ci_hi": float(np.quantile(selected_auc_boot, 0.975)),
        "random_auc_mean": float(random_per_subset_auc.mean()),
        "random_auc_subset_std": float(random_per_subset_auc.std(ddof=1)),
        "random_auc_subset_ci_lo": float(np.quantile(random_per_subset_auc, 0.025)),
        "random_auc_subset_ci_hi": float(np.quantile(random_per_subset_auc, 0.975)),
        "random_auc_mc_std": float(random_auc_boot.std(ddof=1)),
        "random_auc_per_subset": random_per_subset_auc.tolist(),
    }

    # Empirical one-sided p-value: selected AUC < random AUCs?
    # (Lower AUC = lower error prob across noise levels = better discrimination.)
    n_rand = len(random_per_subset_auc)
    n_le = int(np.sum(random_per_subset_auc <= selected_auc))
    auc_info["p_value_empirical"] = (n_le + 1) / (n_rand + 1)
    auc_info["z_score"] = (
        (selected_auc - random_per_subset_auc.mean())
        / random_per_subset_auc.std(ddof=1)
        if random_per_subset_auc.std(ddof=1) > 0
        else float("nan")
    )

    # Correlation info
    correlation_info = {
        "selected_clean": selected_corr.numpy().tolist(),
        "selected_noised": selected_corr_noised.numpy().tolist(),
        "random_clean": random_corr.numpy().tolist(),
        "model_names": model_names,
    }

    return discrim_df, correlation_info, noise_info, auc_info


def main():
    parser = argparse.ArgumentParser(
        description="Compute discriminability metrics for stimulus selection"
    )
    add_standard_args(parser)
    parser.add_argument(
        "--n-random-subsets",
        type=int,
        default=50,
        help="Number of random baseline subsets (default: 50)",
    )
    parser.add_argument(
        "--n-random-images",
        type=int,
        default=10000,
        help="Number of random baseline images to load before subset sampling (default: 10000)",
    )
    parser.add_argument(
        "--n-noise-samples",
        type=int,
        default=DEFAULT_N_NOISE_SAMPLES,
        help=f"Number of noise samples per level (default: {DEFAULT_N_NOISE_SAMPLES})",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="RDM metric (default: from payload config, or 'cosine')",
    )
    parser.add_argument(
        "--corr-type",
        type=str,
        default=None,
        help="Correlation type (default: from payload config, or 'spearman')",
    )
    parser.add_argument(
        "--which-selection",
        type=str,
        choices=["final", "greedy", "best_raw_combined"],
        default="final",
        help=(
            "Which selection variant to evaluate: "
            "'final' (after refinement), 'greedy' (before refinement), "
            "or 'best_raw_combined' (best raw combined score during refinement). "
            "Default: final"
        ),
    )
    parser.add_argument(
        "--unique-encodings",
        action="store_true",
        help="Use unique per-subject encoding models instead of shared encodings.",
    )
    args = parser.parse_args()

    # Setup
    payload, output_dir, device = setup_from_args(args)

    # Build encoding root map if using unique encodings
    encoding_root_map = None
    if args.unique_encodings:
        import sys as _sys
        _paper_root = Path(__file__).resolve().parents[2]  # cstim_paper/
        _sys.path.insert(0, str(_paper_root))
        from config import UNIQUE_ENCODING_DIRS
        encoding_root_map = {k: v for k, v in UNIQUE_ENCODING_DIRS.items()}
        print(f"Using UNIQUE per-subject encodings: {list(encoding_root_map.keys())}")

    config = payload.get("config", {})
    paths = config.get("paths", {})

    # Legacy Iris path fixup — only runs when /SSD/ exists (i.e., on Iris).
    # On Raven, use --env raven instead.
    if Path("/SSD").exists():
        _IRIS_OVERRIDES = {
            "model_list_csv": str(STIMULUS_ROOT / "resources" / "model_list.csv"),
            "subset_root": "/SSD/datasets/cstims_laion_natural_subset",
            "encoding_root": str(STIMULUS_ROOT / "experiments" / "encoding_fitting" / "results" / "encoding_20251222_141301"),
        }
        _IRIS_PREPROCESSED = {
            "raw": "/SSD/datasets/cstims_laion_natural_subset_memmaps",
        }
        for key, local in _IRIS_OVERRIDES.items():
            if key in paths and not Path(paths[key]).exists() and Path(local).exists():
                paths[key] = local
                print(f"  Overrode {key} -> {local}")
        if "preprocessed_dirs" in paths:
            for k, local in _IRIS_PREPROCESSED.items():
                if k in paths["preprocessed_dirs"] and not Path(paths["preprocessed_dirs"][k]).exists():
                    if Path(local).exists():
                        paths["preprocessed_dirs"][k] = local
                        print(f"  Overrode preprocessed_dirs.{k} -> {local}")

    metric = args.metric or config.get("metric", "cosine")
    corr_type = args.corr_type or config.get("corr_type", "spearman")

    # Get noise level multipliers
    noise_level_multipliers = get_default_noise_level_multipliers()

    # Discover all tracks
    tracks = get_all_tracks_for_evaluation(payload)
    print(f"Found {len(tracks)} tracks to evaluate: {[t['name'] for t in tracks]}")

    # Cache for encoding parameters
    encoding_params_cache: dict[str, Any] = {}

    # Collect results
    all_discrim_rows = []
    all_correlation_data = []
    all_noise_data = []
    all_auc_rows = []

    for track_idx, track in enumerate(tracks):
        track_name = track["name"]
        print(f"\n{'=' * 60}")
        print(f"Processing track {track_idx + 1}/{len(tracks)}: {track_name}")
        print(f"{'=' * 60}")
        log_memory(f"start_track_{track_idx}")

        try:
            discrim_df, correlation_info, noise_info, auc_info = (
                compute_discriminability_for_track(
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

            all_discrim_rows.append(discrim_df)

            # Store correlation matrices
            all_correlation_data.append(
                {
                    "track": track_name,
                    **correlation_info,
                }
            )

            # Store AUC significance info (per track)
            all_auc_rows.append({"track": track_name, **{k: v for k, v in auc_info.items() if k != "random_auc_per_subset"}})

            # Store noise calibration
            for model_name, noise_std in noise_info.items():
                all_noise_data.append(
                    {
                        "track": track_name,
                        "model": model_name,
                        "noise_std": noise_std,
                    }
                )

            log_memory(f"completed_track_{track_idx}")
            print(f"  [SUCCESS] Track '{track_name}' completed")

        except Exception as e:
            print(f"  ERROR processing track '{track_name}': {e}")
            import traceback

            traceback.print_exc()
            continue

    # Combine discriminability results
    if all_discrim_rows:
        discrim_df = pd.concat(all_discrim_rows, ignore_index=True)
        discrim_path = output_dir / "discriminability.csv"
        discrim_df.to_csv(discrim_path, index=False)
        print(f"\nSaved discriminability results to {discrim_path}")

        # Print summary
        print("\n=== Discriminability Summary (AUC) ===")
        summary = discrim_df.groupby(["track", "subset_type"])["auc"].first().unstack()
        print(summary.to_string())

    # Save correlation matrices as CSV (flattened format)
    if all_correlation_data:
        corr_rows = []
        for corr_info in all_correlation_data:
            track = corr_info["track"]
            model_names = corr_info["model_names"]
            for matrix_type in ["selected_clean", "selected_noised", "random_clean"]:
                matrix = corr_info[matrix_type]
                for i, mi in enumerate(model_names):
                    for j, mj in enumerate(model_names):
                        corr_rows.append(
                            {
                                "track": track,
                                "matrix_type": matrix_type,
                                "model_i": mi,
                                "model_j": mj,
                                "correlation": matrix[i][j],
                            }
                        )
        corr_df = pd.DataFrame(corr_rows)
        corr_path = output_dir / "correlation_matrices.csv"
        corr_df.to_csv(corr_path, index=False)
        print(f"Saved correlation matrices to {corr_path}")

    # Save AUC significance
    if all_auc_rows:
        auc_df = pd.DataFrame(all_auc_rows)
        auc_path = output_dir / "auc_significance.csv"
        auc_df.to_csv(auc_path, index=False)
        print(f"Saved AUC significance to {auc_path}")
        print("\n=== AUC Significance Summary ===")
        print(auc_df[["track", "selected_auc", "random_auc_mean",
                      "selected_auc_mc_ci_lo", "selected_auc_mc_ci_hi",
                      "random_auc_subset_ci_lo", "random_auc_subset_ci_hi",
                      "z_score", "p_value_empirical"]].to_string())

    # Save noise calibration
    if all_noise_data:
        noise_df = pd.DataFrame(all_noise_data)
        noise_path = output_dir / "noise_calibration.csv"
        noise_df.to_csv(noise_path, index=False)
        print(f"Saved noise calibration to {noise_path}")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
