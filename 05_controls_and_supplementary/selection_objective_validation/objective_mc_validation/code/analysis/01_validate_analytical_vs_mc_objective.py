#!/usr/bin/env python3
"""Validate analytical selection utility against RDM-space Monte Carlo sampling.

The production objective uses the attenuation approximation

    E[corr(RDM + noise, RDM')] ~= attenuation * corr(RDM, RDM')

and then aggregates the expected correlation matrix. This script compares that
calculation against the direct alternative: sample additive Gaussian noise in RDM
space, compute the utility for each sample, and average utilities.

The implementation is NumPy-only so the supplemental analysis can be reproduced
from the cached natural-pool feature arrays without requiring a CUDA/PyTorch
environment. It mirrors the semantics in cstims.selection.primitives and
cstims.selection.utility for Pearson RSA correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats
from scipy.special import logsumexp


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SECTION = Path(__file__).resolve().parents[1]
DATA_DIR = SECTION / "data"

MODEL_NAMES = [
    "vissl_resnet50_supervised",
    "vissl_resnet50_barlowtwins",
    "vissl_resnet50_mocov2",
    "vicreg_resnet50",
    "robustness_imagenet_l2_eps3",
]

NC_TARGETS = [0.3, 0.46, 0.7, 0.9]
AGGREGATIONS = [
    ("mean", "mean", "Mean / mean"),
    ("mean", "min", "Mean / hard min"),
    ("smooth_min", "smooth_min", "Smooth min / smooth min"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "natural_pool_subset_10k",
        help="Directory with cached per-model .npz feature files.",
    )
    parser.add_argument("--metric", choices=["cosine", "correlation"], default="cosine")
    parser.add_argument("--target-nc", type=float, default=0.46)
    parser.add_argument("--n-calib", type=int, default=1000)
    parser.add_argument("--n-selected", type=int, default=64)
    parser.add_argument("--n-candidates", type=int, default=300)
    parser.add_argument("--n-mc", type=int, default=512)
    parser.add_argument("--mc-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_rows(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def load_cached_features(
    cache_dir: Path,
    model_names: list[str],
    n_total: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Load aligned cached natural-pool features for the requested models."""
    arrays: dict[str, np.ndarray] = {}
    indices_ref: np.ndarray | None = None

    for model in model_names:
        path = cache_dir / f"{model}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached features for {model}: {path}")
        payload = np.load(path)
        features = payload["features"].astype(np.float32, copy=False)
        indices = payload["indices"]
        if indices_ref is None:
            indices_ref = indices
        elif not np.array_equal(indices_ref, indices):
            raise ValueError(f"Feature cache indices are not aligned for {model}")
        arrays[model] = features

    n_available = min(arr.shape[0] for arr in arrays.values())
    if n_total > n_available:
        raise ValueError(f"Requested {n_total} images but cache has only {n_available}")

    rng = np.random.default_rng(seed)
    rows = rng.choice(n_available, size=n_total, replace=False)
    return {model: features[rows] for model, features in arrays.items()}


def _normalize_rows(x: np.ndarray, metric: str) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if metric == "correlation":
        x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


def condensed_pairwise_distances(features: np.ndarray, metric: str) -> np.ndarray:
    x = _normalize_rows(features, metric)
    sim = np.clip(x @ x.T, -1.0, 1.0)
    dist = 1.0 - sim
    tri = np.triu_indices(features.shape[0], k=1)
    return dist[tri].astype(np.float32, copy=False)


def candidate_to_selected_distances(
    candidates: np.ndarray,
    selected: np.ndarray,
    metric: str,
) -> np.ndarray:
    cand = _normalize_rows(candidates, metric)
    sel = _normalize_rows(selected, metric)
    sim = np.clip(cand @ sel.T, -1.0, 1.0)
    return (1.0 - sim).astype(np.float32, copy=False)


def make_augmented_rdms(
    features: dict[str, np.ndarray],
    *,
    n_calib: int,
    n_selected: int,
    n_candidates: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return augmented candidate RDMs [B, M, P] and calibration variances [M]."""
    current_rdms = []
    new_dissims = []
    calib_vars = []

    for model in MODEL_NAMES:
        arr = features[model]
        calib = arr[:n_calib]
        selected = arr[n_calib : n_calib + n_selected]
        candidates = arr[n_calib + n_selected : n_calib + n_selected + n_candidates]

        calib_rdm = condensed_pairwise_distances(calib, metric)
        calib_vars.append(float(np.var(calib_rdm)))
        current_rdms.append(condensed_pairwise_distances(selected, metric))
        new_dissims.append(candidate_to_selected_distances(candidates, selected, metric))

    current = np.stack(current_rdms, axis=0)  # [M, P0]
    new = np.stack(new_dissims, axis=1)  # [B, M, n_selected]
    current_expanded = np.broadcast_to(
        current[None, :, :],
        (n_candidates, current.shape[0], current.shape[1]),
    )
    augmented = np.concatenate([current_expanded, new], axis=2).astype(np.float32, copy=False)
    return augmented, np.asarray(calib_vars, dtype=np.float32)


def noise_vars_from_target(calib_vars: np.ndarray, target_nc: float) -> np.ndarray:
    if target_nc >= 1:
        return np.zeros_like(calib_vars, dtype=np.float32)
    return (calib_vars * (1.0 / (target_nc * target_nc) - 1.0)).astype(np.float32)


def zscore_last(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / (centered.std(axis=-1, keepdims=True) + 1e-8)


def batch_corr_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xz = zscore_last(x)
    yz = zscore_last(y)
    return np.einsum("bmp,bnp->bmn", xz, yz, optimize=True) / x.shape[-1]


def smooth_min(x: np.ndarray, axis: int, beta: float = 5.0) -> np.ndarray:
    return -logsumexp(-beta * x, axis=axis) / beta


def utilities_from_correlations(
    correlations: np.ndarray,
    aggregation_within: str,
    aggregation_across: str,
    *,
    beta_within: float = 5.0,
    beta_across: float = 5.0,
) -> np.ndarray:
    """Mirror cstims.selection.primitives utility aggregation."""
    n_models = correlations.shape[-1]
    idx = np.arange(n_models)
    r_self = np.diagonal(correlations, axis1=-2, axis2=-1)

    if aggregation_within == "mean":
        r_others = (correlations.sum(axis=-1) - r_self) / (n_models - 1)
    elif aggregation_within == "min":
        masked = correlations.copy()
        masked[..., idx, idx] = np.inf
        r_others = masked.min(axis=-1)
    elif aggregation_within == "smooth_min":
        masked = correlations.copy()
        masked[..., idx, idx] = 1e9
        r_others = smooth_min(masked, axis=-1, beta=beta_within)
    else:
        raise ValueError(f"Unknown aggregation_within: {aggregation_within}")

    utilities_per_model = r_self - r_others

    if aggregation_across == "mean":
        return utilities_per_model.mean(axis=-1)
    if aggregation_across == "min":
        return utilities_per_model.min(axis=-1)
    if aggregation_across == "smooth_min":
        return smooth_min(utilities_per_model, axis=-1, beta=beta_across)
    raise ValueError(f"Unknown aggregation_across: {aggregation_across}")


def analytical_utility(
    augmented_rdms: np.ndarray,
    noise_vars: np.ndarray,
    aggregation_within: str,
    aggregation_across: str,
) -> np.ndarray:
    rdm_vars = augmented_rdms.var(axis=2)
    attenuation = np.sqrt(rdm_vars / (rdm_vars + noise_vars[None, :] + 1e-8))
    base_corr = batch_corr_matrix(augmented_rdms, augmented_rdms)
    expected_corr = base_corr * attenuation[:, :, None]
    return utilities_from_correlations(expected_corr, aggregation_within, aggregation_across)


def mc_utility(
    augmented_rdms: np.ndarray,
    noise_stds: np.ndarray,
    aggregation_within: str,
    aggregation_across: str,
    *,
    n_mc: int,
    mc_batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Average utility after directly adding Gaussian RDM-space noise."""
    bsz, n_models, n_pairs = augmented_rdms.shape
    clean_norm = zscore_last(augmented_rdms)
    utility_sum = np.zeros(bsz, dtype=np.float64)
    utility_sumsq = np.zeros(bsz, dtype=np.float64)
    n_done = 0

    while n_done < n_mc:
        batch = min(mc_batch_size, n_mc - n_done)
        noise = rng.standard_normal(
            (batch, bsz, n_models, n_pairs),
            dtype=np.float32,
        )
        noise *= noise_stds[None, None, :, None]
        noisy_norm = zscore_last(augmented_rdms[None, :, :, :] + noise)
        correlations = (
            np.einsum("sbmp,bnp->sbmn", noisy_norm, clean_norm, optimize=True) / n_pairs
        )
        utilities = utilities_from_correlations(
            correlations,
            aggregation_within=aggregation_within,
            aggregation_across=aggregation_across,
        )
        utility_sum += utilities.sum(axis=0)
        utility_sumsq += (utilities * utilities).sum(axis=0)
        n_done += batch

    mean = utility_sum / n_mc
    var = np.maximum(utility_sumsq / n_mc - mean * mean, 0.0)
    sem = np.sqrt(var / n_mc)
    return mean.astype(np.float32), sem.astype(np.float32)


def top_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, len(a))
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


def rank_of_index(scores: np.ndarray, index: int) -> int:
    return int(np.where(np.argsort(-scores) == index)[0][0]) + 1


def summarize_comparison(
    analytical: np.ndarray,
    mc: np.ndarray,
    mc_sem: np.ndarray,
    *,
    nc_target: float,
    aggregation_within: str,
    aggregation_across: str,
    label: str,
) -> dict[str, object]:
    diff = mc - analytical
    analytical_top_idx = int(np.argmax(analytical))
    mc_top_idx = int(np.argmax(mc))
    mc_best = float(mc[mc_top_idx])
    mc_at_analytical_top = float(mc[analytical_top_idx])
    mc_regret = mc_best - mc_at_analytical_top
    mc_range = float(mc.max() - mc.min())
    combined_top_sem = float(
        np.sqrt(mc_sem[mc_top_idx] ** 2 + mc_sem[analytical_top_idx] ** 2)
    )
    return {
        "nc_target": nc_target,
        "aggregation_within": aggregation_within,
        "aggregation_across": aggregation_across,
        "label": label,
        "pearson_r": float(stats.pearsonr(analytical, mc).statistic),
        "spearman_rho": float(stats.spearmanr(analytical, mc).statistic),
        "kendall_tau": float(stats.kendalltau(analytical, mc).statistic),
        "top1_match": int(np.argmax(analytical) == np.argmax(mc)),
        "top10_overlap": float(top_overlap(analytical, mc, 10)),
        "top20_overlap": float(top_overlap(analytical, mc, 20)),
        "top50_overlap": float(top_overlap(analytical, mc, 50)),
        "analytical_top_idx": analytical_top_idx,
        "mc_top_idx": mc_top_idx,
        "analytical_top_rank_under_mc": rank_of_index(mc, analytical_top_idx),
        "mc_top_rank_under_analytical": rank_of_index(analytical, mc_top_idx),
        "mc_best_utility": mc_best,
        "mc_utility_at_analytical_top": mc_at_analytical_top,
        "mc_regret_analytical_choice": float(mc_regret),
        "mc_regret_fraction_of_mc_range": float(mc_regret / mc_range) if mc_range > 0 else 0.0,
        "mc_regret_fraction_of_mc_mean": float(mc_regret / abs(mc.mean())) if mc.mean() != 0 else 0.0,
        "mc_regret_over_combined_top_sem": float(mc_regret / combined_top_sem)
        if combined_top_sem > 0
        else 0.0,
        "mean_analytical": float(analytical.mean()),
        "mean_mc": float(mc.mean()),
        "mean_bias_mc_minus_analytical": float(diff.mean()),
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mean_mc_sem": float(mc_sem.mean()),
        "utility_cv_analytical": float(analytical.std(ddof=1) / abs(analytical.mean())),
        "utility_cv_mc": float(mc.std(ddof=1) / abs(mc.mean())),
    }


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    n_total = args.n_calib + args.n_selected + args.n_candidates
    print(f"Loading {n_total} cached natural-pool images for {len(MODEL_NAMES)} models")
    features = load_cached_features(args.feature_cache, MODEL_NAMES, n_total, args.seed)
    augmented, calib_vars = make_augmented_rdms(
        features,
        n_calib=args.n_calib,
        n_selected=args.n_selected,
        n_candidates=args.n_candidates,
        metric=args.metric,
    )
    print(f"Augmented RDM tensor: {augmented.shape}")

    summary_noise: list[dict[str, object]] = []
    summary_agg: list[dict[str, object]] = []
    metadata = {
        "metric": args.metric,
        "corr_type": "correlation",
        "n_calib": args.n_calib,
        "n_selected": args.n_selected,
        "n_candidates": args.n_candidates,
        "n_mc": args.n_mc,
        "mc_batch_size": args.mc_batch_size,
        "seed": args.seed,
        "model_names": MODEL_NAMES,
        "calibration_rdm_vars": {
            model: float(var) for model, var in zip(MODEL_NAMES, calib_vars)
        },
    }

    # Main paper objective across noise ceilings: mean within-model, hard-min across models.
    for nc in NC_TARGETS:
        print(f"Noise ceiling {nc:g}: mean/min objective")
        noise_vars = noise_vars_from_target(calib_vars, nc)
        noise_stds = np.sqrt(noise_vars)
        ana = analytical_utility(augmented, noise_vars, "mean", "min")
        mc, mc_sem = mc_utility(
            augmented,
            noise_stds,
            "mean",
            "min",
            n_mc=args.n_mc,
            mc_batch_size=args.mc_batch_size,
            rng=np.random.default_rng(args.seed + int(round(nc * 1000))),
        )
        summary_noise.append(
            summarize_comparison(
                ana,
                mc,
                mc_sem,
                nc_target=nc,
                aggregation_within="mean",
                aggregation_across="min",
                label="Mean / hard min",
            )
        )
        if abs(nc - args.target_nc) < 1e-8:
            _write_rows(
                DATA_DIR / "candidate_utilities_nc0.46_mean_min.csv",
                [
                    {
                        "candidate_idx": i,
                        "utility_analytical": float(a),
                        "utility_mc": float(m),
                        "utility_mc_sem": float(s),
                        "nc_target": nc,
                    }
                    for i, (a, m, s) in enumerate(zip(ana, mc, mc_sem))
                ],
                [
                    "candidate_idx",
                    "utility_analytical",
                    "utility_mc",
                    "utility_mc_sem",
                    "nc_target",
                ],
            )

    # Aggregation controls at the paper noise ceiling.
    noise_vars = noise_vars_from_target(calib_vars, args.target_nc)
    noise_stds = np.sqrt(noise_vars)
    for agg_within, agg_across, label in AGGREGATIONS:
        print(f"Aggregation {label} at NC={args.target_nc:g}")
        ana = analytical_utility(augmented, noise_vars, agg_within, agg_across)
        mc, mc_sem = mc_utility(
            augmented,
            noise_stds,
            agg_within,
            agg_across,
            n_mc=args.n_mc,
            mc_batch_size=args.mc_batch_size,
            rng=np.random.default_rng(args.seed + 10_000 + len(summary_agg)),
        )
        summary_agg.append(
            summarize_comparison(
                ana,
                mc,
                mc_sem,
                nc_target=args.target_nc,
                aggregation_within=agg_within,
                aggregation_across=agg_across,
                label=label,
            )
        )

    fields = [
        "nc_target",
        "aggregation_within",
        "aggregation_across",
        "label",
        "pearson_r",
        "spearman_rho",
        "kendall_tau",
        "top1_match",
        "top10_overlap",
        "top20_overlap",
        "top50_overlap",
        "analytical_top_idx",
        "mc_top_idx",
        "analytical_top_rank_under_mc",
        "mc_top_rank_under_analytical",
        "mc_best_utility",
        "mc_utility_at_analytical_top",
        "mc_regret_analytical_choice",
        "mc_regret_fraction_of_mc_range",
        "mc_regret_fraction_of_mc_mean",
        "mc_regret_over_combined_top_sem",
        "mean_analytical",
        "mean_mc",
        "mean_bias_mc_minus_analytical",
        "mae",
        "rmse",
        "mean_mc_sem",
        "utility_cv_analytical",
        "utility_cv_mc",
    ]
    _write_rows(DATA_DIR / "summary_by_noise_ceiling.csv", summary_noise, fields)
    _write_rows(DATA_DIR / "summary_by_aggregation.csv", summary_agg, fields)
    (DATA_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote results to {DATA_DIR}")


if __name__ == "__main__":
    main()
