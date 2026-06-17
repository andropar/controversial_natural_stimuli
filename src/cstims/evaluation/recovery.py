"""Track-level model recovery evaluation."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cstims.evaluation.computation import (
    compute_all_rdms,
    compute_auc,
    compute_clean_correlation_matrix,
    compute_correlation_at_target_noise,
    compute_discriminability_by_noise_level,
    compute_random_baseline_rdms,
)
from cstims.evaluation.constants import DEFAULT_N_BOOTSTRAP
from cstims.evaluation.noise_calibration import (
    calibrate_noise_parameters,
    multiplier_to_noise_ceiling,
)


FeatureLoader = Callable[..., tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]]
NoiseVarianceLoader = Callable[[dict, str], dict[str, float] | None]
LogMemoryFn = Callable[[str], None]


def get_track_noise_variances(payload: dict, track_name: str) -> dict[str, float] | None:
    """Get payload-stored RDM noise variances for one evaluation track."""
    var_noise = payload.get("var_noise_by_model", {})
    return var_noise.get(track_name)


def _stable_seed(seed: int | None, *parts: object) -> int | None:
    if seed is None:
        return None
    h = hashlib.blake2b(digest_size=4)
    h.update(str(int(seed)).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False)


def _log_memory(log_memory_fn: LogMemoryFn | None, label: str) -> None:
    if log_memory_fn is not None:
        log_memory_fn(label)


def _std(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1))


def _ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _bootstrap_auc(values_boot: np.ndarray, noise_mult: np.ndarray) -> np.ndarray:
    sort_idx = np.argsort(noise_mult.astype(float))
    x_sorted = noise_mult[sort_idx]
    return np.asarray(
        [
            compute_auc(x_sorted, values_boot[sort_idx, boot_idx])
            for boot_idx in range(values_boot.shape[1])
        ],
        dtype=np.float64,
    )


def _require_bootstrap(result: dict[str, Any], key: str) -> np.ndarray:
    values = result[key]
    if values is None:
        raise RuntimeError(f"Expected bootstrap result for {key}")
    return np.asarray(values, dtype=np.float64)


def _resolve_track_noise(
    *,
    payload: dict,
    track_name: str,
    model_names: list[str],
    random_features: dict[str, np.ndarray],
    metric: str,
    target_nc: float,
    device: torch.device,
    noise_variance_loader: NoiseVarianceLoader | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    loader = noise_variance_loader or get_track_noise_variances
    precomputed_noise = loader(payload, track_name)
    if precomputed_noise is not None:
        noise_stds = torch.stack(
            [
                torch.tensor(
                    precomputed_noise[model],
                    device=device,
                    dtype=torch.float32,
                ).sqrt()
                for model in model_names
            ]
        ).unsqueeze(1)
    else:
        noise_params = calibrate_noise_parameters(
            features=random_features,
            model_names=model_names,
            metrics=[metric],
            target_nc=target_nc,
            device=device,
            mode="analytical",
        )
        noise_stds = noise_params.get_noise_stds(metric)

    noise_stds = noise_stds.to(device)
    noise_info = {
        model: float(noise_stds[idx].item()) for idx, model in enumerate(model_names)
    }
    return noise_stds, noise_info


def _build_auc_info(
    *,
    noise_multipliers: np.ndarray,
    selected_errors: np.ndarray,
    selected_boot: np.ndarray,
    random_errors: np.ndarray,
    random_boot: np.ndarray,
) -> tuple[float, float, dict[str, Any]]:
    selected_auc = compute_auc(noise_multipliers, selected_errors)
    random_mean_curve = random_errors.mean(axis=0)
    random_auc = compute_auc(noise_multipliers, random_mean_curve)

    selected_auc_boot = _bootstrap_auc(selected_boot, noise_multipliers)
    random_auc_per_subset = np.asarray(
        [compute_auc(noise_multipliers, curve) for curve in random_errors],
        dtype=np.float64,
    )
    random_auc_boot = _bootstrap_auc(random_boot.mean(axis=0), noise_multipliers)

    auc_info = {
        "selected_auc": float(selected_auc),
        "selected_auc_mc_std": _std(selected_auc_boot),
        "selected_auc_mc_ci_lo": float(np.quantile(selected_auc_boot, 0.025)),
        "selected_auc_mc_ci_hi": float(np.quantile(selected_auc_boot, 0.975)),
        "random_auc_mean": float(random_auc_per_subset.mean()),
        "random_auc_subset_std": _std(random_auc_per_subset),
        "random_auc_subset_ci_lo": float(np.quantile(random_auc_per_subset, 0.025)),
        "random_auc_subset_ci_hi": float(np.quantile(random_auc_per_subset, 0.975)),
        "random_auc_mc_std": _std(random_auc_boot),
        "random_auc_per_subset": random_auc_per_subset.tolist(),
    }

    n_rand = len(random_auc_per_subset)
    n_le = int(np.sum(random_auc_per_subset <= selected_auc))
    auc_info["p_value_empirical"] = (n_le + 1) / (n_rand + 1)
    random_auc_subset_std = random_auc_per_subset.std(ddof=1) if n_rand > 1 else 0.0
    auc_info["z_score"] = (
        (selected_auc - random_auc_per_subset.mean()) / random_auc_subset_std
        if random_auc_subset_std > 0
        else float("nan")
    )
    return float(selected_auc), float(random_auc), auc_info


def _build_pairwise_outputs(
    *,
    track_name: str,
    track_type: str,
    metric: str,
    corr_type: str,
    target_nc: float,
    noise_multipliers: np.ndarray,
    selected_result: dict[str, Any],
    random_pairwise_dominance: np.ndarray,
    random_mean_margin: np.ndarray,
    random_pairwise_dominance_boot: np.ndarray,
    random_mean_margin_boot: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    selected_dom = np.asarray(selected_result["pairwise_dominance"], dtype=np.float64)
    selected_margin = np.asarray(
        selected_result["mean_pairwise_margin"],
        dtype=np.float64,
    )
    selected_dom_boot = _require_bootstrap(
        selected_result,
        "pairwise_dominance_bootstrap",
    )
    selected_margin_boot = _require_bootstrap(
        selected_result,
        "mean_pairwise_margin_bootstrap",
    )

    random_dom_mean = random_pairwise_dominance.mean(axis=0)
    random_margin_mean = random_mean_margin.mean(axis=0)
    random_dom_subset_std = (
        random_pairwise_dominance.std(axis=0, ddof=1)
        if random_pairwise_dominance.shape[0] > 1
        else np.full_like(noise_multipliers, np.nan, dtype=np.float64)
    )
    random_margin_subset_std = (
        random_mean_margin.std(axis=0, ddof=1)
        if random_mean_margin.shape[0] > 1
        else np.full_like(noise_multipliers, np.nan, dtype=np.float64)
    )
    random_dom_mc_std = random_pairwise_dominance_boot.mean(axis=0).std(axis=1, ddof=1)
    random_margin_mc_std = random_mean_margin_boot.mean(axis=0).std(axis=1, ddof=1)

    for level_idx, multiplier in enumerate(noise_multipliers):
        noise_ceiling = multiplier_to_noise_ceiling(float(multiplier), target_nc)
        dom_lo, dom_hi = _ci(selected_dom_boot[level_idx])
        margin_lo, margin_hi = _ci(selected_margin_boot[level_idx])
        rows.append(
            {
                "track": track_name,
                "track_type": track_type,
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": float(multiplier),
                "noise_ceiling": noise_ceiling,
                "subset_type": "selected",
                "pairwise_dominance": selected_dom[level_idx],
                "pairwise_dominance_subset_std": np.nan,
                "pairwise_dominance_mc_std": _std(selected_dom_boot[level_idx]),
                "pairwise_dominance_mc_ci_lo": dom_lo,
                "pairwise_dominance_mc_ci_hi": dom_hi,
                "pairwise_error_prob": 1.0 - selected_dom[level_idx],
                "mean_margin": selected_margin[level_idx],
                "mean_margin_subset_std": np.nan,
                "mean_margin_mc_std": _std(selected_margin_boot[level_idx]),
                "mean_margin_mc_ci_lo": margin_lo,
                "mean_margin_mc_ci_hi": margin_hi,
            }
        )
        rows.append(
            {
                "track": track_name,
                "track_type": track_type,
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": float(multiplier),
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
            }
        )

    selected_dom_auc = compute_auc(noise_multipliers, selected_dom)
    selected_margin_auc = compute_auc(noise_multipliers, selected_margin)
    selected_dom_auc_boot = _bootstrap_auc(selected_dom_boot, noise_multipliers)
    selected_margin_auc_boot = _bootstrap_auc(selected_margin_boot, noise_multipliers)
    random_dom_auc_per_subset = np.asarray(
        [compute_auc(noise_multipliers, curve) for curve in random_pairwise_dominance],
        dtype=np.float64,
    )
    random_margin_auc_per_subset = np.asarray(
        [compute_auc(noise_multipliers, curve) for curve in random_mean_margin],
        dtype=np.float64,
    )
    random_dom_auc_boot = _bootstrap_auc(
        random_pairwise_dominance_boot.mean(axis=0),
        noise_multipliers,
    )
    random_margin_auc_boot = _bootstrap_auc(
        random_mean_margin_boot.mean(axis=0),
        noise_multipliers,
    )

    dom_lo, dom_hi = _ci(selected_dom_auc_boot)
    margin_lo, margin_hi = _ci(selected_margin_auc_boot)
    pairwise_auc_info = {
        "selected_pairwise_dominance_auc": float(selected_dom_auc),
        "selected_pairwise_dominance_auc_mc_std": _std(selected_dom_auc_boot),
        "selected_pairwise_dominance_auc_mc_ci_lo": dom_lo,
        "selected_pairwise_dominance_auc_mc_ci_hi": dom_hi,
        "random_pairwise_dominance_auc_mean": float(
            random_dom_auc_per_subset.mean()
        ),
        "random_pairwise_dominance_auc_subset_std": _std(
            random_dom_auc_per_subset
        ),
        "random_pairwise_dominance_auc_mc_std": _std(random_dom_auc_boot),
        "selected_mean_margin_auc": float(selected_margin_auc),
        "selected_mean_margin_auc_mc_std": _std(selected_margin_auc_boot),
        "selected_mean_margin_auc_mc_ci_lo": margin_lo,
        "selected_mean_margin_auc_mc_ci_hi": margin_hi,
        "random_mean_margin_auc_mean": float(random_margin_auc_per_subset.mean()),
        "random_mean_margin_auc_subset_std": _std(random_margin_auc_per_subset),
        "random_mean_margin_auc_mc_std": _std(random_margin_auc_boot),
        "pairwise_dominance_auc_z_score": (
            (selected_dom_auc - random_dom_auc_per_subset.mean())
            / random_dom_auc_per_subset.std(ddof=1)
            if random_dom_auc_per_subset.size > 1
            and random_dom_auc_per_subset.std(ddof=1) > 0
            else float("nan")
        ),
        "mean_margin_auc_z_score": (
            (selected_margin_auc - random_margin_auc_per_subset.mean())
            / random_margin_auc_per_subset.std(ddof=1)
            if random_margin_auc_per_subset.size > 1
            and random_margin_auc_per_subset.std(ddof=1) > 0
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
    return pd.DataFrame(rows), pairwise_auc_info


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
    *,
    feature_loader: FeatureLoader,
    noise_variance_loader: NoiseVarianceLoader | None = None,
    log_memory_fn: LogMemoryFn | None = None,
    orientation: str = "clean_by_noisy",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict, dict[str, float], dict[str, Any], pd.DataFrame, dict[str, float]]:
    """Compute selected-vs-random model recovery for a single evaluation track."""
    track_name = track["name"]
    track_type = track.get("type", "identity")
    model_names = list(payload["model_names"])

    print(
        f"  Loading features for track '{track_name}' "
        f"(type={track_type}, variant={selection_variant})..."
    )
    _log_memory(log_memory_fn, "start_track")
    selected_features, random_features = feature_loader(
        payload=payload,
        track=track,
        device=device,
        encoding_params_cache=encoding_params_cache,
        n_random=n_random_images,
        selection_variant=selection_variant,
        encoding_root_map=encoding_root_map,
    )

    n_selected = next(iter(selected_features.values())).shape[0]
    _log_memory(log_memory_fn, "after_load_features")
    print(f"  [DEBUG] Loaded {n_selected} selected stimuli, {len(model_names)} models")

    config = payload.get("config", {})
    target_nc = float(config.get("noise_ceiling_target", 0.46))
    print(f"  Calibrating noise for track '{track_name}'...")
    noise_stds, noise_info = _resolve_track_noise(
        payload=payload,
        track_name=track_name,
        model_names=model_names,
        random_features=random_features,
        metric=metric,
        target_nc=target_nc,
        device=device,
        noise_variance_loader=noise_variance_loader,
    )

    print("  Computing RDMs for selected stimuli...")
    selected_rdms = compute_all_rdms(selected_features, [metric])
    _log_memory(log_memory_fn, "after_selected_rdms")

    del selected_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log_memory(log_memory_fn, "after_cleanup_selected")

    print(f"  Computing RDMs for {n_random_subsets} random subsets...")
    random_rdms = compute_random_baseline_rdms(
        random_features=random_features,
        model_names=model_names,
        metrics=[metric],
        n_selected_stimuli=n_selected,
        n_random_subsets=n_random_subsets,
        device=device,
    )
    del random_features
    gc.collect()
    _log_memory(log_memory_fn, "after_random_rdms")

    print("  Computing correlation matrices...")
    selected_corr = compute_clean_correlation_matrix(selected_rdms[metric], corr_type)

    random_corr_list = []
    random_corr_noised_list = []
    for random_rdm in random_rdms:
        random_metric_rdms = random_rdm[metric].to(device)
        corr = compute_clean_correlation_matrix(random_metric_rdms, corr_type)
        random_corr_list.append(corr.cpu())
        corr_noised = compute_correlation_at_target_noise(
            random_metric_rdms,
            noise_stds,
            corr_type,
            n_noise_samples,
            orientation=orientation,
        )
        random_corr_noised_list.append(corr_noised.cpu())
        del random_metric_rdms, corr, corr_noised
    random_corr = torch.stack(random_corr_list).mean(dim=0)
    random_corr_noised = torch.stack(random_corr_noised_list).mean(dim=0)
    del random_corr_list, random_corr_noised_list
    gc.collect()

    selected_corr_noised = compute_correlation_at_target_noise(
        selected_rdms[metric],
        noise_stds,
        corr_type,
        n_noise_samples,
        orientation=orientation,
    )

    noise_multipliers = np.asarray(noise_level_multipliers, dtype=np.float64)
    print(f"  Computing discriminability across {len(noise_multipliers)} noise levels...")
    _log_memory(log_memory_fn, "before_discriminability")

    selected_result = compute_discriminability_by_noise_level(
        rdms=selected_rdms[metric],
        noise_stds=noise_stds,
        n_noise_samples=n_noise_samples,
        noise_level_multipliers=noise_multipliers,
        corr_type=corr_type,
        orientation=orientation,
        n_bootstrap=n_bootstrap,
        seed=_stable_seed(seed, track_name, "selected"),
    )
    selected_boot = _require_bootstrap(
        selected_result,
        "multiclass_error_probability_bootstrap",
    )
    _log_memory(log_memory_fn, "after_selected_discrim")

    print(f"  [DEBUG] Computing random discriminability for {len(random_rdms)} subsets...")
    random_errors = []
    random_boot = []
    random_pairwise_dominance = []
    random_mean_margin = []
    random_pairwise_dominance_boot = []
    random_mean_margin_boot = []

    for subset_idx, random_rdm in enumerate(
        tqdm(random_rdms, desc="Random subsets", leave=False)
    ):
        subset_result = compute_discriminability_by_noise_level(
            rdms=random_rdm[metric],
            noise_stds=noise_stds,
            n_noise_samples=n_noise_samples,
            noise_level_multipliers=noise_multipliers,
            corr_type=corr_type,
            orientation=orientation,
            n_bootstrap=n_bootstrap,
            seed=_stable_seed(seed, track_name, "random", subset_idx),
        )
        random_errors.append(
            np.asarray(subset_result["multiclass_error_probability"], dtype=np.float64)
        )
        random_boot.append(
            _require_bootstrap(
                subset_result,
                "multiclass_error_probability_bootstrap",
            )
        )
        random_pairwise_dominance.append(
            np.asarray(subset_result["pairwise_dominance"], dtype=np.float64)
        )
        random_mean_margin.append(
            np.asarray(subset_result["mean_pairwise_margin"], dtype=np.float64)
        )
        random_pairwise_dominance_boot.append(
            _require_bootstrap(subset_result, "pairwise_dominance_bootstrap")
        )
        random_mean_margin_boot.append(
            _require_bootstrap(subset_result, "mean_pairwise_margin_bootstrap")
        )
        if subset_idx % 10 == 9:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    random_errors_arr = np.stack(random_errors)
    random_boot_arr = np.stack(random_boot)
    random_pairwise_dominance_arr = np.stack(random_pairwise_dominance)
    random_mean_margin_arr = np.stack(random_mean_margin)
    random_pairwise_dominance_boot_arr = np.stack(random_pairwise_dominance_boot)
    random_mean_margin_boot_arr = np.stack(random_mean_margin_boot)

    del random_rdms
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log_memory(log_memory_fn, "after_random_discrim_cleanup")
    print(f"  [DEBUG] Finished discriminability computation for track '{track_name}'")

    selected_errors = np.asarray(
        selected_result["multiclass_error_probability"],
        dtype=np.float64,
    )
    selected_auc, random_auc, auc_info = _build_auc_info(
        noise_multipliers=noise_multipliers,
        selected_errors=selected_errors,
        selected_boot=selected_boot,
        random_errors=random_errors_arr,
        random_boot=random_boot_arr,
    )

    discrim_rows: list[dict[str, Any]] = []
    off_diag_mask = ~torch.eye(len(model_names), dtype=torch.bool)
    for level_idx, multiplier in enumerate(noise_multipliers):
        noise_ceiling = multiplier_to_noise_ceiling(float(multiplier), target_nc)
        corr_matrix = compute_correlation_at_target_noise(
            selected_rdms[metric],
            noise_stds * float(multiplier),
            corr_type,
            n_noise_samples,
            orientation=orientation,
        )
        mean_offdiag = float(corr_matrix[off_diag_mask].mean())
        mc_lo, mc_hi = _ci(selected_boot[level_idx])
        discrim_rows.append(
            {
                "track": track_name,
                "track_type": track_type,
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": float(multiplier),
                "noise_ceiling": noise_ceiling,
                "subset_type": "selected",
                "error_prob": selected_errors[level_idx],
                "error_prob_std": np.nan,
                "error_prob_mc_std": _std(selected_boot[level_idx]),
                "error_prob_mc_ci_lo": mc_lo,
                "error_prob_mc_ci_hi": mc_hi,
                "mean_offdiag_corr": mean_offdiag,
            }
        )

        random_level_boot = random_boot_arr[:, level_idx, :]
        random_mc_std = np.asarray(
            [_std(curve) for curve in random_level_boot],
            dtype=np.float64,
        )
        random_mc_ci_lo = np.quantile(random_level_boot, 0.025, axis=1)
        random_mc_ci_hi = np.quantile(random_level_boot, 0.975, axis=1)
        discrim_rows.append(
            {
                "track": track_name,
                "track_type": track_type,
                "metric": metric,
                "corr_type": corr_type,
                "noise_mult": float(multiplier),
                "noise_ceiling": noise_ceiling,
                "subset_type": "random",
                "error_prob": float(random_errors_arr[:, level_idx].mean()),
                "error_prob_std": _std(random_errors_arr[:, level_idx]),
                "error_prob_mc_std": float(np.nanmean(random_mc_std)),
                "error_prob_mc_ci_lo": float(random_mc_ci_lo.mean()),
                "error_prob_mc_ci_hi": float(random_mc_ci_hi.mean()),
                "mean_offdiag_corr": np.nan,
            }
        )

    discrim_df = pd.DataFrame(discrim_rows)
    discrim_df["auc"] = discrim_df["subset_type"].apply(
        lambda subset_type: selected_auc if subset_type == "selected" else random_auc
    )

    pairwise_df, pairwise_auc_info = _build_pairwise_outputs(
        track_name=track_name,
        track_type=track_type,
        metric=metric,
        corr_type=corr_type,
        target_nc=target_nc,
        noise_multipliers=noise_multipliers,
        selected_result=selected_result,
        random_pairwise_dominance=random_pairwise_dominance_arr,
        random_mean_margin=random_mean_margin_arr,
        random_pairwise_dominance_boot=random_pairwise_dominance_boot_arr,
        random_mean_margin_boot=random_mean_margin_boot_arr,
    )

    correlation_info = {
        "selected_clean": selected_corr.numpy().tolist(),
        "selected_noised": selected_corr_noised.numpy().tolist(),
        "random_clean": random_corr.numpy().tolist(),
        "random_noised": random_corr_noised.numpy().tolist(),
        "model_names": model_names,
    }

    return (
        discrim_df,
        correlation_info,
        noise_info,
        auc_info,
        pairwise_df,
        pairwise_auc_info,
    )
