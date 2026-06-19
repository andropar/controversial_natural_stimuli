#!/usr/bin/env python3
"""Compute recovery robustness metrics for RDM- and feature-level noise.

This is a follow-up diagnostic for the feature-method sweep.  It keeps the
selected image sets and fitted encoding projections fixed, then compares two
synthetic noise models:

1. ``rdm``: add Gaussian noise directly to model RDM vectors.
2. ``feature``: add Gaussian noise to the evaluated feature space first
   (raw activations for raw; predicted voxel responses for encoded tracks),
   then recompute RDMs.

Both noise models are calibrated to target the run's clean/noisy RDM
self-correlation at ``noise_mult=1``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid


SCRIPT = Path(__file__).resolve()
ROOT = next(p for p in SCRIPT.parents if (p / "src" / "cstims").exists())
SWEEP_ROOT = ROOT / "00_stimulus_selection" / "feature_method_sweep"
RECOVERY_ROOT = (
    ROOT
    / "00_stimulus_selection"
    / "selection_evaluation"
    / "feature_method_sweep_recovery"
    / "noisy_by_clean"
)
RECOVERY_RESULTS_ROOT = RECOVERY_ROOT / "results"
SRC_DIR = ROOT / "src"
for path in (SRC_DIR,):
    sys.path.insert(0, str(path))

from cstims import constants, paths
from cstims.evaluation.constants import get_default_noise_level_multipliers  # noqa: E402
from cstims.evaluation.payload import apply_env_paths, filter_payload_to_models  # noqa: E402
from cstims.evaluation.random_features import (  # noqa: E402
    available_random_models,
    make_random_feature_cache_loader,
)
from cstims.evaluation.track_loading import (  # noqa: E402
    get_all_tracks_for_evaluation,
    load_features_for_track,
)
from cstims.rdm import get_rdm_vector  # noqa: E402


DEFAULT_RUN = SWEEP_ROOT / "results" / "sota_20260611_112941"
DEFAULT_FIG_DIR = RECOVERY_ROOT / "figures"
DEFAULT_METHODS = (
    "raw_only_mean_min",
    "sub01_only_mean_min",
    "raw_enc_w05_mean_min",
    "paper_effective_identity_sub01_mean_min_no_attenuation",
)
ENCODING_TRACKS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
TRACKS = ("raw", *ENCODING_TRACKS)
METHOD_LABELS = {
    "raw_only_mean_min": "Raw features only",
    "sub01_only_mean_min": "Sub-01 only (current)",
    "raw_enc_w05_mean_min": "Intended (Raw + enc, mean/min)",
    "paper_effective_identity_sub01_mean_min_no_attenuation": "Sub-01 only (no attenuation)",
}
METHOD_COLORS = {
    "raw_only_mean_min": "#4C78A8",
    "sub01_only_mean_min": "#F28E2B",
    "raw_enc_w05_mean_min": "#59A14F",
    "paper_effective_identity_sub01_mean_min_no_attenuation": "#9C755F",
}

def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def method_label(method_id: str) -> str:
    return METHOD_LABELS.get(method_id, method_id)


def method_color(method_id: str) -> str:
    return METHOD_COLORS.get(method_id, "#777777")


def normalize_rdm_vectors(data: torch.Tensor, corr_type: str) -> torch.Tensor:
    if data.dtype != torch.float32:
        data = data.float()
    if corr_type == "spearman":
        data = torch.argsort(torch.argsort(data, dim=-1, stable=False), dim=-1, stable=False).float()
    centered = data - data.mean(dim=-1, keepdim=True)
    std = centered.std(dim=-1, keepdim=True, unbiased=False) + 1e-8
    return centered / std


def correlate_noisy_to_clean(
    noisy_rdms: torch.Tensor,
    clean_rdms: torch.Tensor,
    corr_type: str,
) -> torch.Tensor:
    noisy_norm = normalize_rdm_vectors(noisy_rdms, corr_type)
    clean_norm = normalize_rdm_vectors(clean_rdms.unsqueeze(0), corr_type)
    clean_norm = clean_norm.expand(noisy_norm.shape[0], -1, -1)
    n_pairs = noisy_norm.shape[-1]
    scores = torch.bmm(noisy_norm, clean_norm.transpose(1, 2)) / n_pairs
    return torch.nan_to_num(scores, nan=0.0)


def rdm_self_corr(noisy_rdms: torch.Tensor, clean_rdm: torch.Tensor, corr_type: str) -> float:
    clean = clean_rdm.unsqueeze(0).expand(noisy_rdms.shape[0], -1)
    noisy_norm = normalize_rdm_vectors(noisy_rdms, corr_type)
    clean_norm = normalize_rdm_vectors(clean, corr_type)
    corr = (noisy_norm * clean_norm).mean(dim=1)
    return float(torch.nan_to_num(corr.mean(), nan=0.0).item())


def clean_rdms_from_features(
    features: dict[str, torch.Tensor],
    model_names: list[str],
    metric: str,
) -> torch.Tensor:
    return torch.stack([get_rdm_vector(features[model], metric=metric) for model in model_names])


def rdm_noise_stds(clean_rdms: torch.Tensor, target_self_corr: float) -> torch.Tensor:
    if target_self_corr >= 1.0:
        return torch.zeros((clean_rdms.shape[0], 1), device=clean_rdms.device)
    target = max(float(target_self_corr), 1e-6)
    var = clean_rdms.var(dim=1, unbiased=False).clamp_min(1e-12)
    var_noise = var * (1.0 / (target * target) - 1.0)
    return torch.sqrt(var_noise).unsqueeze(1)


def calibrate_feature_sigmas(
    features: dict[str, torch.Tensor],
    model_names: list[str],
    *,
    metric: str,
    corr_type: str,
    target_self_corr: float,
    n_repeats: int,
    max_iter: int,
    seed: int,
) -> torch.Tensor:
    sigmas: list[float] = []
    for model_idx, model in enumerate(model_names):
        feat = features[model].float()
        clean_rdm = get_rdm_vector(feat, metric=metric)
        if target_self_corr >= 1.0 or clean_rdm.numel() == 0:
            sigmas.append(0.0)
            continue

        gen = torch.Generator(device=feat.device)
        gen.manual_seed(int(seed) + model_idx * 1009)
        base_noise = torch.randn(
            (n_repeats, *feat.shape),
            device=feat.device,
            generator=gen,
            dtype=feat.dtype,
        )

        def achieved_corr(sigma: float) -> float:
            noisy = feat.unsqueeze(0) + base_noise * float(sigma)
            noisy_rdms = get_rdm_vector(noisy, metric=metric)
            return rdm_self_corr(noisy_rdms, clean_rdm, corr_type)

        high = max(float(feat.std(unbiased=False).item()), 1e-4)
        while achieved_corr(high) > target_self_corr and high < 1e4:
            high *= 2.0

        low = 0.0
        best = high
        best_err = abs(achieved_corr(high) - target_self_corr)
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            corr = achieved_corr(mid)
            err = abs(corr - target_self_corr)
            if err < best_err:
                best = mid
                best_err = err
            if corr < target_self_corr:
                high = mid
            else:
                low = mid
        sigmas.append(float(best))
    return torch.tensor(sigmas, device=next(iter(features.values())).device, dtype=torch.float32).unsqueeze(1)


def score_metrics(scores: torch.Tensor) -> dict[str, float]:
    n_draws, n_models, _ = scores.shape
    diag = scores.diagonal(dim1=1, dim2=2)
    eye = torch.eye(n_models, dtype=torch.bool, device=scores.device)
    other_scores = scores.masked_fill(eye.unsqueeze(0), -torch.inf)
    strongest_other = other_scores.max(dim=2).values
    hardest_margin = diag - strongest_other
    detected = scores.argmax(dim=2)
    true_idx = torch.arange(n_models, device=scores.device).unsqueeze(0)
    correct = detected == true_idx

    pairwise_margin = diag.unsqueeze(2) - scores
    offdiag = pairwise_margin[:, ~eye]
    pairwise_dom = (
        (offdiag > 0).float()
        + 0.5 * torch.isclose(offdiag, torch.zeros_like(offdiag)).float()
    )

    return {
        "accuracy": float(correct.float().mean().item()),
        "self_score": float(diag.mean().item()),
        "strongest_other_score": float(strongest_other.mean().item()),
        "mean_hardest_margin": float(hardest_margin.mean().item()),
        "p05_hardest_margin": float(torch.quantile(hardest_margin.flatten(), 0.05).item()),
        "mean_pairwise_margin": float(offdiag.mean().item()),
        "p05_pairwise_margin": float(torch.quantile(offdiag.flatten(), 0.05).item()),
        "pairwise_dominance": float(pairwise_dom.mean().item()),
    }


def compute_curve_rdm_noise(
    clean_rdms: torch.Tensor,
    noise_stds: torch.Tensor,
    *,
    noise_levels: np.ndarray,
    n_noise_samples: int,
    corr_type: str,
    seed: int,
) -> list[dict[str, float]]:
    gen = torch.Generator(device=clean_rdms.device)
    gen.manual_seed(int(seed))
    rows = []
    for noise_mult in noise_levels:
        noisy_rdms = clean_rdms.unsqueeze(0) + (
            torch.randn(
                (n_noise_samples, *clean_rdms.shape),
                device=clean_rdms.device,
                generator=gen,
            )
            * noise_stds.unsqueeze(0)
            * float(noise_mult)
        )
        scores = correlate_noisy_to_clean(noisy_rdms, clean_rdms, corr_type)
        rows.append({"noise_mult": float(noise_mult), **score_metrics(scores)})
        del noisy_rdms, scores
    return rows


def compute_curve_feature_noise(
    features: dict[str, torch.Tensor],
    model_names: list[str],
    clean_rdms: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    metric: str,
    corr_type: str,
    noise_levels: np.ndarray,
    n_noise_samples: int,
    seed: int,
) -> list[dict[str, float]]:
    gen = torch.Generator(device=clean_rdms.device)
    gen.manual_seed(int(seed))
    rows = []
    for noise_mult in noise_levels:
        noisy_rdms_by_model = []
        for model_idx, model in enumerate(model_names):
            feat = features[model].float()
            noisy = feat.unsqueeze(0) + (
                torch.randn(
                    (n_noise_samples, *feat.shape),
                    device=feat.device,
                    generator=gen,
                )
                * sigmas[model_idx]
                * float(noise_mult)
            )
            noisy_rdms_by_model.append(get_rdm_vector(noisy, metric=metric))
            del noisy
        noisy_rdms = torch.stack(noisy_rdms_by_model, dim=1)
        scores = correlate_noisy_to_clean(noisy_rdms, clean_rdms, corr_type)
        rows.append({"noise_mult": float(noise_mult), **score_metrics(scores)})
        del noisy_rdms, scores, noisy_rdms_by_model
    return rows


def feature_subset(
    random_features: dict[str, np.ndarray],
    model_names: list[str],
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        model: torch.tensor(random_features[model][indices], device=device, dtype=torch.float32)
        for model in model_names
    }


def random_indices(
    random_features: dict[str, np.ndarray],
    model_names: list[str],
    n_selected: int,
    n_random_subsets: int,
    seed: int,
) -> list[np.ndarray]:
    max_available = min(random_features[model].shape[0] for model in model_names)
    rng = np.random.default_rng(seed)
    return [
        rng.choice(max_available, size=n_selected, replace=False).astype(np.int64)
        for _ in range(n_random_subsets)
    ]


def multiplier_to_noise_ceiling(k: float, nc_base: float) -> float:
    if k <= 0:
        return 1.0
    if nc_base <= 0 or nc_base >= 1:
        return nc_base
    term = k * k * (1.0 / (nc_base * nc_base) - 1.0)
    return float(1.0 / np.sqrt(1.0 + term))


def compute_track_curves(
    *,
    method_id: str,
    payload: dict,
    track: dict[str, Any],
    device: torch.device,
    random_feature_dir: Path | None,
    n_random_subsets: int,
    n_random_images: int,
    n_noise_samples: int,
    noise_levels: np.ndarray,
    metric: str,
    corr_type: str,
    target_self_corr: float,
    feature_calib_repeats: int,
    feature_calib_iters: int,
    seed: int,
    encoding_params_cache: dict[str, Any],
    encoding_root_map: dict[str, Path] | None,
) -> pd.DataFrame:
    track_name = track["name"]
    random_feature_loader = (
        make_random_feature_cache_loader(random_feature_dir)
        if random_feature_dir is not None
        else None
    )
    selected_features, random_features = load_features_for_track(
        payload=payload,
        track=track,
        device=device,
        encoding_params_cache=encoding_params_cache,
        n_random=n_random_images,
        selection_variant="final",
        encoding_root_map=encoding_root_map,
        random_feature_loader=random_feature_loader,
    )
    model_names = list(payload["model_names"])
    n_selected = next(iter(selected_features.values())).shape[0]

    rows = []

    selected_rdms = clean_rdms_from_features(selected_features, model_names, metric)
    selected_rdm_stds = rdm_noise_stds(selected_rdms, target_self_corr)
    selected_feature_sigmas = calibrate_feature_sigmas(
        selected_features,
        model_names,
        metric=metric,
        corr_type=corr_type,
        target_self_corr=target_self_corr,
        n_repeats=feature_calib_repeats,
        max_iter=feature_calib_iters,
        seed=seed + 11,
    )

    for noise_model, curve_rows in [
        (
            "rdm",
            compute_curve_rdm_noise(
                selected_rdms,
                selected_rdm_stds,
                noise_levels=noise_levels,
                n_noise_samples=n_noise_samples,
                corr_type=corr_type,
                seed=seed + 101,
            ),
        ),
        (
            "feature",
            compute_curve_feature_noise(
                selected_features,
                model_names,
                selected_rdms,
                selected_feature_sigmas,
                metric=metric,
                corr_type=corr_type,
                noise_levels=noise_levels,
                n_noise_samples=n_noise_samples,
                seed=seed + 201,
            ),
        ),
    ]:
        for row in curve_rows:
            rows.append(
                {
                    "method_id": method_id,
                    "track": track_name,
                    "track_type": track.get("type", "identity"),
                    "subset_type": "selected",
                    "random_subset": -1,
                    "noise_model": noise_model,
                    "noise_ceiling": multiplier_to_noise_ceiling(row["noise_mult"], target_self_corr),
                    "target_self_corr": target_self_corr,
                    **row,
                }
            )

    idx_rows = random_indices(
        random_features,
        model_names,
        n_selected,
        n_random_subsets,
        seed=seed + 301,
    )
    if idx_rows:
        reference_random = feature_subset(random_features, model_names, idx_rows[0], device)
        reference_random_rdms = clean_rdms_from_features(reference_random, model_names, metric)
        random_rdm_stds = rdm_noise_stds(reference_random_rdms, target_self_corr)
        random_feature_sigmas = calibrate_feature_sigmas(
            reference_random,
            model_names,
            metric=metric,
            corr_type=corr_type,
            target_self_corr=target_self_corr,
            n_repeats=feature_calib_repeats,
            max_iter=feature_calib_iters,
            seed=seed + 401,
        )
        del reference_random, reference_random_rdms
    else:
        random_rdm_stds = None
        random_feature_sigmas = None

    for subset_idx, indices in enumerate(tqdm(idx_rows, desc=f"{method_id}/{track_name} random", leave=False)):
        feats = feature_subset(random_features, model_names, indices, device)
        clean_rdms = clean_rdms_from_features(feats, model_names, metric)
        for noise_model, curve_rows in [
            (
                "rdm",
                compute_curve_rdm_noise(
                    clean_rdms,
                    random_rdm_stds,
                    noise_levels=noise_levels,
                    n_noise_samples=n_noise_samples,
                    corr_type=corr_type,
                    seed=seed + 501 + subset_idx,
                ),
            ),
            (
                "feature",
                compute_curve_feature_noise(
                    feats,
                    model_names,
                    clean_rdms,
                    random_feature_sigmas,
                    metric=metric,
                    corr_type=corr_type,
                    noise_levels=noise_levels,
                    n_noise_samples=n_noise_samples,
                    seed=seed + 601 + subset_idx,
                ),
            ),
        ]:
            for row in curve_rows:
                rows.append(
                    {
                        "method_id": method_id,
                        "track": track_name,
                        "track_type": track.get("type", "identity"),
                        "subset_type": "random",
                        "random_subset": subset_idx,
                        "noise_model": noise_model,
                        "noise_ceiling": multiplier_to_noise_ceiling(row["noise_mult"], target_self_corr),
                        "target_self_corr": target_self_corr,
                        **row,
                    }
                )
        del feats, clean_rdms
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def aggregate_curves(curves: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "method_id",
        "track",
        "track_type",
        "subset_type",
        "noise_model",
        "noise_mult",
        "noise_ceiling",
        "target_self_corr",
    ]
    metrics = [
        "accuracy",
        "self_score",
        "strongest_other_score",
        "mean_hardest_margin",
        "p05_hardest_margin",
        "mean_pairwise_margin",
        "p05_pairwise_margin",
        "pairwise_dominance",
    ]
    agg = curves.groupby(keys, as_index=False)[metrics].agg(["mean", "std", "sem"])
    agg.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns.to_flat_index()
    ]
    return agg


def compute_auc_log_x(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return float("nan")
    order = np.argsort(x)
    x_log = np.log10(x[order])
    y_sorted = y[order]
    span = x_log[-1] - x_log[0]
    auc = float(np.trapz(y_sorted, x_log))
    return auc / span if span > 0 else auc


def threshold_drop(noise_mult: np.ndarray, accuracy: np.ndarray, threshold: float) -> tuple[float, float]:
    x = np.asarray(noise_mult, dtype=float)
    y = np.asarray(accuracy, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return float("nan"), float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    below = np.where(y <= threshold)[0]
    if below.size == 0:
        return float("nan"), float("nan")
    idx = int(below[0])
    if idx == 0:
        drop = float(x[0])
    else:
        x0, x1 = np.log10(x[idx - 1]), np.log10(x[idx])
        y0, y1 = y[idx - 1], y[idx]
        if abs(y1 - y0) < 1e-12:
            drop = float(x[idx])
        else:
            frac = (threshold - y0) / (y1 - y0)
            drop = float(10 ** (x0 + frac * (x1 - x0)))
    return drop, float(1.0 / drop) if drop > 0 else float("nan")


def summarize_curves(curves_agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["method_id", "track", "track_type", "subset_type", "noise_model"]
    for keys, grp in curves_agg.groupby(group_cols, sort=False):
        method_id, track, track_type, subset_type, noise_model = keys
        grp = grp.sort_values("noise_mult")
        emp = grp.iloc[(grp["noise_mult"] - 1.0).abs().to_numpy().argmin()]
        stress = grp[grp["noise_mult"] >= 1.0]
        drop90, snr90 = threshold_drop(grp["noise_mult"], grp["accuracy_mean"], 0.90)
        drop75, snr75 = threshold_drop(grp["noise_mult"], grp["accuracy_mean"], 0.75)
        rows.append(
            {
                "method_id": method_id,
                "method_label": method_label(method_id),
                "track": track,
                "track_type": track_type,
                "eval_space": "Raw" if track == "raw" else "Encoded",
                "subset_type": subset_type,
                "noise_model": noise_model,
                "empirical_noise_mult": float(emp["noise_mult"]),
                "empirical_accuracy": float(emp["accuracy_mean"]),
                "empirical_pairwise_dominance": float(emp["pairwise_dominance_mean"]),
                "empirical_mean_hardest_margin": float(emp["mean_hardest_margin_mean"]),
                "empirical_p05_hardest_margin": float(emp["p05_hardest_margin_mean"]),
                "empirical_mean_pairwise_margin": float(emp["mean_pairwise_margin_mean"]),
                "empirical_p05_pairwise_margin": float(emp["p05_pairwise_margin_mean"]),
                "empirical_self_score": float(emp["self_score_mean"]),
                "empirical_strongest_other_score": float(emp["strongest_other_score_mean"]),
                "drop_noise_mult_acc90": drop90,
                "drop_snr_acc90": snr90,
                "drop_noise_mult_acc75": drop75,
                "drop_snr_acc75": snr75,
                "stress_accuracy_auc_noise_mult_ge1": compute_auc_log_x(
                    stress["noise_mult"], stress["accuracy_mean"]
                ),
                "stress_pairwise_dominance_auc_noise_mult_ge1": compute_auc_log_x(
                    stress["noise_mult"], stress["pairwise_dominance_mean"]
                ),
                "stress_mean_hardest_margin_auc_noise_mult_ge1": compute_auc_log_x(
                    stress["noise_mult"], stress["mean_hardest_margin_mean"]
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_space_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in summary.columns
        if col.startswith("empirical_")
        or col.startswith("drop_")
        or col.startswith("stress_")
    ]
    rows = []
    for keys, grp in summary.groupby(["method_id", "method_label", "subset_type", "noise_model"], sort=False):
        method_id, label, subset_type, noise_model = keys
        raw = grp[grp["eval_space"] == "Raw"]
        if not raw.empty:
            row = {
                "method_id": method_id,
                "method_label": label,
                "subset_type": subset_type,
                "noise_model": noise_model,
                "eval_space": "Raw",
                "n_tracks": int(raw["track"].nunique()),
            }
            row.update({col: float(raw[col].mean()) for col in metric_cols})
            rows.append(row)
        enc = grp[grp["eval_space"] == "Encoded"]
        if not enc.empty:
            row = {
                "method_id": method_id,
                "method_label": label,
                "subset_type": subset_type,
                "noise_model": noise_model,
                "eval_space": "Encoded mean",
                "n_tracks": int(enc["track"].nunique()),
            }
            row.update({col: float(enc[col].mean()) for col in metric_cols})
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#777777", linewidth=7.0, alpha=0.88, label="Selected"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=1.1,
            markersize=6,
            label="Random mean",
        ),
    ]


def plot_metric_panel(
    space_summary: pd.DataFrame,
    out_dir: Path,
    *,
    methods: list[str],
) -> list[Path]:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plot_metrics = [
        ("empirical_mean_hardest_margin", "Self minus strongest other\nat empirical SNR"),
        ("empirical_mean_pairwise_margin", "Mean pairwise margin\nat empirical SNR"),
        ("empirical_p05_pairwise_margin", "5th percentile pairwise margin\nat empirical SNR"),
        ("stress_accuracy_auc_noise_mult_ge1", "Stress recovery AUC\nnoise_mult >= 1"),
        ("drop_snr_acc90", "SNR where accuracy drops below 90%"),
    ]
    noise_models = ["rdm", "feature"]
    eval_spaces = ["Raw", "Encoded mean"]
    fig, axes = plt.subplots(
        len(plot_metrics),
        len(noise_models) * len(eval_spaces),
        figsize=(15.5, 12.2),
        constrained_layout=True,
        sharey="row",
    )
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (metric_col, metric_label) in enumerate(plot_metrics):
        metric_values = space_summary[metric_col].to_numpy(dtype=float)
        finite = metric_values[np.isfinite(metric_values)]
        if finite.size:
            xmin = min(0.0, float(finite.min()) - 0.05 * max(1e-6, float(np.ptp(finite))))
            xmax = float(finite.max()) + 0.08 * max(1e-6, float(np.ptp(finite)))
            if metric_col.startswith("drop_snr"):
                xmin = 0.0
        else:
            xmin, xmax = 0.0, 1.0
        if abs(xmax - xmin) < 1e-9:
            xmax = xmin + 1.0

        for col_idx, (noise_model, eval_space) in enumerate(
            (nm, sp) for nm in noise_models for sp in eval_spaces
        ):
            ax = axes[row_idx, col_idx]
            selected = space_summary[
                (space_summary["subset_type"] == "selected")
                & (space_summary["noise_model"] == noise_model)
                & (space_summary["eval_space"] == eval_space)
            ].set_index("method_id")
            random = space_summary[
                (space_summary["subset_type"] == "random")
                & (space_summary["noise_model"] == noise_model)
                & (space_summary["eval_space"] == eval_space)
            ].set_index("method_id")
            y = np.arange(len(methods))
            selected_vals = np.array([selected.loc[m, metric_col] if m in selected.index else np.nan for m in methods])
            random_vals = np.array([random.loc[m, metric_col] if m in random.index else np.nan for m in methods])
            colors = [method_color(m) for m in methods]
            ax.barh(y, selected_vals, color=colors, alpha=0.88)
            if np.isfinite(random_vals).any():
                ax.scatter(
                    random_vals,
                    y,
                    marker="o",
                    s=22,
                    facecolor="white",
                    edgecolor="#333333",
                    linewidth=1.0,
                    zorder=3,
                )
            ax.set_yticks(y)
            ax.set_yticklabels([method_label(m) for m in methods])
            ax.invert_yaxis()
            ax.set_xlim(xmin, xmax)
            ax.grid(axis="x", color="#E5E5E5", linewidth=0.6, alpha=0.8)
            if row_idx == 0:
                ax.set_title(f"{noise_model.upper()} noise\n{eval_space}")
            ax.set_xlabel(metric_label)

    fig.suptitle("Recovery robustness: RDM-level vs feature-level noise", fontsize=12, fontweight="bold")
    fig.legend(
        handles=comparison_legend_handles(),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        fontsize=7,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "feature_method_sweep_recovery_robustness_metrics.pdf"
    png = out_dir / "feature_method_sweep_recovery_robustness_metrics.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def plot_stress_curves(
    curves_agg: pd.DataFrame,
    out_dir: Path,
    *,
    methods: list[str],
) -> list[Path]:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    curves = curves_agg.copy()
    curves["snr"] = 1.0 / curves["noise_mult"].astype(float)
    curves["eval_space"] = np.where(curves["track"] == "raw", "Raw", "Encoded mean")
    encoded = curves[curves["eval_space"] == "Encoded mean"].copy()
    raw = curves[curves["eval_space"] == "Raw"].copy()
    enc_keys = ["method_id", "subset_type", "noise_model", "noise_mult", "snr"]
    encoded_mean = encoded.groupby(enc_keys, as_index=False)["accuracy_mean"].mean()
    raw_mean = raw.groupby(enc_keys, as_index=False)["accuracy_mean"].mean()
    raw_mean["eval_space"] = "Raw"
    encoded_mean["eval_space"] = "Encoded mean"
    plot_df = pd.concat([raw_mean, encoded_mean], ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True, sharey=True)
    for ax, noise_model, eval_space in [
        (axes[0, 0], "rdm", "Raw"),
        (axes[0, 1], "rdm", "Encoded mean"),
        (axes[1, 0], "feature", "Raw"),
        (axes[1, 1], "feature", "Encoded mean"),
    ]:
        for method in methods:
            for subset_type, linestyle, alpha, linewidth in [
                ("selected", "-", 0.95, 1.9),
                ("random", "--", 0.42, 1.2),
            ]:
                sub = plot_df[
                    (plot_df["method_id"] == method)
                    & (plot_df["noise_model"] == noise_model)
                    & (plot_df["eval_space"] == eval_space)
                    & (plot_df["subset_type"] == subset_type)
                ].sort_values("snr")
                if sub.empty:
                    continue
                ax.plot(
                    sub["snr"],
                    sub["accuracy_mean"],
                    color=method_color(method),
                    linestyle=linestyle,
                    alpha=alpha,
                    linewidth=linewidth,
                )
        ax.axvline(1.0, color="#444444", linewidth=0.8, alpha=0.35)
        ax.axhline(0.90, color="#999999", linewidth=0.7, linestyle=":", alpha=0.8)
        ax.axhline(0.75, color="#BBBBBB", linewidth=0.7, linestyle=":", alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.009, 11.0)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0.01, 0.1, 1, 10])
        ax.set_xticklabels(["0.01", "0.1", "1", "10"])
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.7)
        ax.set_title(f"{noise_model.upper()} noise: {eval_space}")
        ax.set_xlabel("Relative SNR")
    axes[0, 0].set_ylabel("Recovery accuracy")
    axes[1, 0].set_ylabel("Recovery accuracy")

    method_handles = [
        Line2D([0], [0], color=method_color(method), linewidth=2.0, label=method_label(method))
        for method in methods
    ]
    style_handles = [
        Line2D([0], [0], color="#222222", linestyle="-", linewidth=1.8, label="Selected"),
        Line2D([0], [0], color="#222222", linestyle="--", linewidth=1.2, alpha=0.55, label="Random"),
    ]
    fig.legend(
        handles=method_handles + style_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=7,
    )
    fig.suptitle("Recovery stress curves: RDM-level vs feature-level noise", fontsize=12, fontweight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "feature_method_sweep_recovery_robustness_curves.pdf"
    png = out_dir / "feature_method_sweep_recovery_robustness_curves.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--tracks", default=",".join(TRACKS))
    parser.add_argument("--env", default="raven", choices=["raven", "iris"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-feature-dir", type=Path, default=ROOT / "shared" / "cache_or_heavy" / "natural_pool_subset_100k_seed42")
    parser.add_argument("--n-random-subsets", type=int, default=8)
    parser.add_argument("--n-random-images", type=int, default=5000)
    parser.add_argument("--n-noise-samples", type=int, default=64)
    parser.add_argument("--feature-calib-repeats", type=int, default=12)
    parser.add_argument("--feature-calib-iters", type=int, default=10)
    parser.add_argument("--unique-encodings", action="store_true", default=True)
    parser.add_argument("--shared-encodings", action="store_false", dest="unique_encodings")
    parser.add_argument("--shared-encoding-root", type=Path, default=paths.shared_encoding_root())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    methods = parse_csv_list(args.methods) or list(DEFAULT_METHODS)
    tracks = parse_csv_list(args.tracks) or list(TRACKS)
    args.run_dir = args.run_dir.resolve()
    args.out_dir = (
        args.out_dir
        or (RECOVERY_RESULTS_ROOT / args.run_dir.name / "robustness")
    ).resolve()
    args.fig_dir = args.fig_dir.resolve()
    args.random_feature_dir = args.random_feature_dir.resolve() if args.random_feature_dir else None
    args.shared_encoding_root = args.shared_encoding_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    noise_levels = get_default_noise_level_multipliers()
    all_rows: list[pd.DataFrame] = []

    encoding_root_map = None
    if args.unique_encodings:
        encoding_root_map = {
            key: Path(value).resolve() for key, value in paths.unique_encoding_dirs().items()
        }
        print(f"Using unique encoding roots: {list(encoding_root_map)}")
    else:
        if not args.shared_encoding_root.exists():
            raise FileNotFoundError(f"Shared encoding root does not exist: {args.shared_encoding_root}")
        encoding_root_map = {key: args.shared_encoding_root for key in ENCODING_TRACKS}
        print(f"Using shared encoding root: {args.shared_encoding_root}")

    metadata = {
        "run_dir": str(args.run_dir),
        "methods": methods,
        "tracks": tracks,
        "noise_levels": [float(x) for x in noise_levels],
        "n_random_subsets": args.n_random_subsets,
        "n_random_images": args.n_random_images,
        "n_noise_samples": args.n_noise_samples,
        "feature_calib_repeats": args.feature_calib_repeats,
        "feature_calib_iters": args.feature_calib_iters,
        "random_feature_dir": str(args.random_feature_dir) if args.random_feature_dir else None,
        "unique_encodings": bool(args.unique_encodings),
        "encoding_mode": "unique" if args.unique_encodings else "shared",
        "encoding_roots": {key: str(value) for key, value in encoding_root_map.items()},
    }
    with (args.out_dir / "recovery_robustness_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    payload_root = (
        RECOVERY_RESULTS_ROOT
        / args.run_dir.name
        / "cross_eval_full_tracks"
        / "payloads"
    )
    if not payload_root.exists():
        payload_root = args.run_dir / "payloads"

    for method_idx, method_id in enumerate(methods):
        payload_path = payload_root / method_id / "selected_stimuli_data.pkl"
        if not payload_path.exists():
            raise FileNotFoundError(f"Missing payload: {payload_path}")
        with payload_path.open("rb") as f:
            payload = pickle.load(f)
        payload = apply_env_paths(payload, args.env)
        if args.random_feature_dir is not None:
            available = available_random_models(args.random_feature_dir, payload["model_names"])
            if set(available) != set(payload["model_names"]):
                missing = sorted(set(payload["model_names"]) - set(available))
                raise RuntimeError(f"{method_id}: random feature cache missing models: {missing}")
            payload = filter_payload_to_models(payload, available)

        config = payload.get("config", {})
        metric = config.get("metric", "cosine")
        corr_type = config.get("corr_type", "correlation")
        target_self_corr = float(config.get("noise_ceiling_target", 0.46))
        track_defs = [
            track for track in get_all_tracks_for_evaluation(payload)
            if track["name"] in set(tracks)
        ]
        encoding_params_cache: dict[str, Any] = {}
        for track_idx, track in enumerate(track_defs):
            print(
                f"[{method_idx + 1}/{len(methods)}] {method_id} "
                f"track {track_idx + 1}/{len(track_defs)}: {track['name']}"
            )
            curves = compute_track_curves(
                method_id=method_id,
                payload=payload,
                track=track,
                device=device,
                random_feature_dir=args.random_feature_dir,
                n_random_subsets=args.n_random_subsets,
                n_random_images=args.n_random_images,
                n_noise_samples=args.n_noise_samples,
                noise_levels=noise_levels,
                metric=metric,
                corr_type=corr_type,
                target_self_corr=target_self_corr,
                feature_calib_repeats=args.feature_calib_repeats,
                feature_calib_iters=args.feature_calib_iters,
                seed=args.seed + method_idx * 10000 + track_idx * 1000,
                encoding_params_cache=encoding_params_cache,
                encoding_root_map=encoding_root_map,
            )
            all_rows.append(curves)
            partial = pd.concat(all_rows, ignore_index=True)
            partial.to_csv(args.out_dir / "recovery_robustness_curves_raw.csv", index=False)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    raw_curves = pd.concat(all_rows, ignore_index=True)
    curves_agg = aggregate_curves(raw_curves)
    summary = summarize_curves(curves_agg)
    space_summary = aggregate_space_summary(summary)

    raw_curves.to_csv(args.out_dir / "recovery_robustness_curves_raw.csv", index=False)
    curves_agg.to_csv(args.out_dir / "recovery_robustness_curves_summary.csv", index=False)
    summary.to_csv(args.out_dir / "recovery_robustness_track_summary.csv", index=False)
    space_summary.to_csv(args.out_dir / "recovery_robustness_space_summary.csv", index=False)

    paths = []
    paths.extend(plot_metric_panel(space_summary, args.fig_dir, methods=methods))
    paths.extend(plot_stress_curves(curves_agg, args.fig_dir, methods=methods))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
