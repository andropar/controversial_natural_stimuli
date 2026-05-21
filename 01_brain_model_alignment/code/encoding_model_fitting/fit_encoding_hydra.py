#!/usr/bin/env python3
"""Fit DeepVision encoding models with Hydra config and SLURM parallelization.

Supports parallel execution via SLURM by splitting work across:
- Subjects (5 subjects)
- Model batches (configurable, default 10 models per batch)

Usage:
    # Run all evaluations for all subjects and models
    python fit_encoding_hydra.py

    # Run single subject, single batch (for SLURM parallelization)
    python fit_encoding_hydra.py parallel.subject=sub-01 parallel.model_batch=0

    # Use split-both for alpha stability validation
    python fit_encoding_hydra.py fitting.split=both fitting.n_random_splits=3

    # Resume from existing run
    python fit_encoding_hydra.py resume_from=/path/to/existing/run
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_share_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "cstims").exists():
            return path
    return start.parents[2]


PROJECT_ROOT = _find_share_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from cstims.datasets.deepvision import DeepVisionBenchmark, DEFAULT_CVE_THRESHOLD
from cstims.encoding import (
    LinearEncodingModel,
    fit_voxelwise_ridgecv,
    fit_voxelwise_ridgecv_fast,
    refit_with_chosen_alphas,
    refit_with_chosen_alphas_fast,
    create_encoding_model,
    compute_versa,
    compute_voxel_r,
    compute_image_r,
)
from cstims.feature_extraction.universal_extractor import UniversalFeatureExtractor


def log(msg: str):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass(frozen=True)
class ModelCfg:
    model: str
    source: str
    layer: str | int
    aggregation: str = "auto"


def sanitize_layer_name(layer: str | int) -> str:
    """Convert layer identifier to filesystem-safe string."""
    if isinstance(layer, int):
        return str(layer)
    return (
        str(layer)
        .replace(".", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("/", "_")
    )


def read_model_list(csv_path: Path) -> List[ModelCfg]:
    models: List[ModelCfg] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model"):
                layer_raw = row.get("layer", row.get("layer_uid", ""))
                try:
                    layer = int(layer_raw)
                except (ValueError, TypeError):
                    layer = layer_raw
                aggregation = row.get("aggregation", "auto")
                models.append(
                    ModelCfg(
                        model=row["model"],
                        source=row["source"],
                        layer=layer,
                        aggregation=aggregation,
                    )
                )
    return models


def get_model_batch(models: List[ModelCfg], batch_idx: int, models_per_batch: int) -> List[ModelCfg]:
    """Get a subset of models for a given batch index."""
    start = batch_idx * models_per_batch
    end = start + models_per_batch
    return models[start:end]


def _load_and_preprocess(args):
    path, preprocess = args
    img = Image.open(path).convert("RGB")
    return preprocess(img)


def extract_features_batched(
    extractor: UniversalFeatureExtractor,
    image_paths: List[str],
    batch_size: int,
    device: str,
    n_workers: int = 32,
) -> np.ndarray:
    """Extract features for all images in batches."""
    import concurrent.futures

    feats_list: List[np.ndarray] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for start in tqdm(
            range(0, len(image_paths), batch_size),
            desc="extracting features",
            leave=False,
        ):
            end = min(start + batch_size, len(image_paths))
            batch_paths = image_paths[start:end]

            tensors = list(pool.map(
                _load_and_preprocess,
                [(p, extractor.preprocess) for p in batch_paths],
            ))
            batch = torch.stack(tensors).to(device)

            # Retry with halved batch size on OOM
            current_batch_size = len(batch_paths)
            while True:
                try:
                    with torch.no_grad():
                        feats = extractor.extract(batch)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    log(f"    OOM - retrying with batch_size={current_batch_size}")
                    sub_feats = []
                    for sub_start in range(0, len(batch_paths), current_batch_size):
                        sub_batch = batch[sub_start:sub_start + current_batch_size]
                        with torch.no_grad():
                            sub_feats.append(extractor.extract(sub_batch))
                        torch.cuda.empty_cache()
                    feats = torch.cat(sub_feats, dim=0)
                    break

            if isinstance(feats, torch.Tensor):
                feats = feats.detach().cpu().numpy()
            feats = np.asarray(feats)
            feats = feats.reshape(feats.shape[0], -1).astype(np.float32)
            feats_list.append(feats)

    features = np.concatenate(feats_list, axis=0)
    return features


def zscore_targets_by_voxel(
    targets: np.ndarray, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each voxel (column) across images (rows)."""
    mean = targets.mean(axis=0, dtype=np.float64)
    std = targets.std(axis=0, dtype=np.float64, ddof=0)
    std = np.maximum(std, eps)
    standardized = (targets - mean) / std
    return (
        standardized.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def apply_srp(
    features: np.ndarray, eps: float = 0.1, seed: int = 42
) -> Tuple[np.ndarray, Optional[object]]:
    """Apply sparse random projection for dimensionality reduction.

    Args:
        features: (n_samples, n_features) input features
        eps: Johnson-Lindenstrauss epsilon (lower = higher dimensions)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (projected_features, srp_object or None if skipped)
    """
    from sklearn.random_projection import (
        SparseRandomProjection,
        johnson_lindenstrauss_min_dim,
    )

    n_samples, n_features = features.shape
    n_components = johnson_lindenstrauss_min_dim(n_samples, eps=eps)

    if n_features <= n_components:
        return features, None

    srp = SparseRandomProjection(n_components=n_components, random_state=seed)
    projected = srp.fit_transform(features).astype(np.float32)
    return projected, srp


def get_completed_models(run_dir: Path, subject: str) -> set:
    """Get set of already completed model names."""
    completed = set()
    pattern = f"{subject}_*"
    for subdir in run_dir.glob(pattern):
        if (subdir / "encoding_model.npz").exists():
            # Extract model name from directory name
            model_name = subdir.name.replace(f"{subject}_", "").rsplit(".layer", 1)[0]
            completed.add(model_name)
    return completed


def fit_model_for_cfg(
    cfg: ModelCfg,
    hydra_cfg: DictConfig,
    benchmark: DeepVisionBenchmark,
    subject: str,
    run_dir: Path,
) -> Optional[Dict]:
    """Fit encoding model for a single model configuration."""
    layer_safe = sanitize_layer_name(cfg.layer)
    model_dir = run_dir / f"{subject}_{cfg.model}.layer{layer_safe}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Skip if exists
    out_npz = model_dir / "encoding_model.npz"
    if hydra_cfg.output.skip_existing and out_npz.exists():
        log(f"  Skipping (exists): {cfg.model}")
        return None

    log(f"  Fitting: {cfg.model} (layer={cfg.layer})")
    start_time = time.time()

    # Init extractor
    extractor = UniversalFeatureExtractor(
        model_name=cfg.model,
        layer=cfg.layer,
        source=cfg.source,
        device=hydra_cfg.extraction.device,
        aggregation=cfg.aggregation,
    )

    # Extract features (cache on disk)
    features_cache_fp = model_dir / "features.npz"
    if features_cache_fp.exists():
        with np.load(features_cache_fp) as z:
            features = z["features"]
        log(f"    Loaded cached features: {features.shape}")
    else:
        image_paths = benchmark.stimulus_data.image_path.tolist()
        features = extract_features_batched(
            extractor=extractor,
            image_paths=image_paths,
            batch_size=int(hydra_cfg.extraction.batch_size),
            device=hydra_cfg.extraction.device,
        )
        log(f"    Extracted features: {features.shape}")
        np.savez_compressed(features_cache_fp, features=features.astype(np.float32))

    # Prepare response matrix
    responses = benchmark.response_data.to_numpy().T  # (n_images, n_voxels)

    # Optional response z-scoring
    if hydra_cfg.preprocessing.response_zscore:
        Y_full, Y_mean, Y_std = zscore_targets_by_voxel(responses)
        log(f"    Z-scored responses: mean range [{Y_mean.min():.2f}, {Y_mean.max():.2f}]")
    else:
        Y_full = responses.astype(np.float32)
        Y_mean = responses.mean(axis=0).astype(np.float32)
        Y_std = responses.std(axis=0).astype(np.float32)
        log("    Using raw responses (no z-scoring)")

    # Optional SRP on features
    if hydra_cfg.preprocessing.srp:
        X_full, srp_obj = apply_srp(
            features,
            eps=hydra_cfg.preprocessing.srp_eps,
            seed=hydra_cfg.fitting.seed,
        )
        log(f"    Applied SRP: {features.shape[1]} -> {X_full.shape[1]} features")
    else:
        X_full = features
        srp_obj = None

    # Get hlvis mask for dual-ROI evaluation
    hlvis_mask = benchmark.get_roi_mask("hlvis")
    n_voxels_hlvis = int(hlvis_mask.sum())
    log(f"    hlvis subset: {n_voxels_hlvis} voxels")

    # Alpha grid
    alpha_grid = np.logspace(
        np.log10(hydra_cfg.ridge.alpha_min),
        np.log10(hydra_cfg.ridge.alpha_max),
        int(hydra_cfg.ridge.n_alphas),
    )

    n_images = X_full.shape[0]
    n_voxels = Y_full.shape[1]
    seed = hydra_cfg.fitting.seed
    eval_method = hydra_cfg.fitting.eval_method

    # Get dataset labels for GroupKFold (if needed)
    if "dataset" in benchmark.stimulus_data.columns:
        dataset_labels = benchmark.stimulus_data["dataset"].values
        unique_datasets = sorted(set(dataset_labels))
        log(f"    Found {len(unique_datasets)} datasets: {unique_datasets}")
    else:
        dataset_labels = None
        unique_datasets = None

    # Initialize evaluation results
    per_dataset_metrics = None
    fold_results = None
    cross_dataset_info = None
    r_versa = None
    r_voxel = None
    r_image = None
    alpha_cv = None
    fold_alphas_array = None
    alpha_stability = None
    veRSA_fold_mean = None
    r_versa_hlvis = None
    r_voxel_hlvis = None

    # =========================================================================
    # Evaluation: Choose method based on config
    # =========================================================================
    if eval_method == "odd_even":
        # ---------------------------------------------------------------------
        # Simple odd/even split evaluation
        # ---------------------------------------------------------------------
        log("    Running odd/even split evaluation...")

        # Split: even indices = train, odd indices = test
        train_idx = np.arange(0, n_images, 2)  # [0, 2, 4, ...]
        test_idx = np.arange(1, n_images, 2)   # [1, 3, 5, ...]

        X_train, X_test = X_full[train_idx], X_full[test_idx]
        Y_train, Y_test = Y_full[train_idx], Y_full[test_idx]

        log(f"      Train: {len(train_idx)} images (even), Test: {len(test_idx)} images (odd)")

        # Fit on train, predict on test
        Y_test_pred, eval_alphas, _ = fit_voxelwise_ridgecv_fast(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            alphas=alpha_grid,
            verbose=False,
            scale_features=hydra_cfg.fitting.scale_features,
        )

        # Compute metrics on held-out test set
        r_versa = compute_versa(Y_test, Y_test_pred)
        r_voxel = compute_voxel_r(Y_test, Y_test_pred)
        r_image = compute_image_r(Y_test, Y_test_pred)

        log(f"      veRSA={r_versa:.4f}, voxel_r={np.median(r_voxel):.4f}, alpha_median={np.median(eval_alphas):.1f}")

        # Store eval_alphas so the final-model step reuses them (skips expensive LOO-CV on all data)
        fold_alphas_array = eval_alphas[np.newaxis, :]  # (1, n_voxels)
        alphas_from_folds = eval_alphas

    elif eval_method == "group_kfold" and dataset_labels is not None:
        # ---------------------------------------------------------------------
        # GroupKFold Evaluation: Leave one dataset out at a time
        # ---------------------------------------------------------------------
        log("    Running GroupKFold evaluation (leave-one-dataset-out)...")

        # Store predictions for all images (aggregated across folds)
        Y_pred_all = np.full((n_images, n_voxels), np.nan, dtype=np.float32)
        fold_results = []
        fold_alphas = []

        for held_out_dataset in unique_datasets:
            # Create train/test split based on dataset
            test_mask = dataset_labels == held_out_dataset
            train_mask = ~test_mask
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            X_train, X_test = X_full[train_idx], X_full[test_idx]
            Y_train, Y_test = Y_full[train_idx], Y_full[test_idx]

            log(f"      Fold: hold out {held_out_dataset} ({len(test_idx)} images), train on {len(train_idx)} images")

            # Fit on training datasets with LOO-CV for alpha selection
            Y_test_pred, alphas, _ = fit_voxelwise_ridgecv_fast(
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                alphas=alpha_grid,
                verbose=False,
                scale_features=hydra_cfg.fitting.scale_features,
            )

            # Store predictions
            Y_pred_all[test_idx] = Y_test_pred

            # Compute per-fold metrics
            r_versa_fold = compute_versa(Y_test, Y_test_pred)
            r_voxel_fold = compute_voxel_r(Y_test, Y_test_pred)

            fold_results.append({
                "dataset": held_out_dataset,
                "n_images": len(test_idx),
                "veRSA": r_versa_fold,
                "voxel_r_median": float(np.median(r_voxel_fold)),
                "voxel_r_mean": float(np.mean(r_voxel_fold)),
                "alpha_median": float(np.median(alphas)),
            })
            fold_alphas.append(alphas)

            log(f"        veRSA={r_versa_fold:.4f}, voxel_r={np.median(r_voxel_fold):.4f}, alpha_median={np.median(alphas):.1f}")

        # Compute overall metrics from aggregated predictions
        r_versa = compute_versa(Y_full, Y_pred_all)
        r_voxel = compute_voxel_r(Y_full, Y_pred_all)
        r_image = compute_image_r(Y_full, Y_pred_all)

        # Report per-dataset summary
        log("    --- Per-Dataset Summary ---")
        for fr in fold_results:
            log(f"      {fr['dataset']:>10}: veRSA={fr['veRSA']:.4f}, voxel_r={fr['voxel_r_median']:.4f}, n={fr['n_images']}")
        log(f"      {'OVERALL':>10}: veRSA={r_versa:.4f}, voxel_r={np.median(r_voxel):.4f}")

        # Store fold alphas and compute stability metrics
        fold_alphas_array = np.stack(fold_alphas, axis=0)  # (n_folds, n_voxels)

        # Aggregate alphas across folds for final model fitting
        if hydra_cfg.fitting.alpha_aggregation == "median":
            alphas_from_folds = np.median(fold_alphas_array, axis=0)
        else:
            alphas_from_folds = np.mean(fold_alphas_array, axis=0)

        alpha_cv = np.std(fold_alphas_array, axis=0) / (np.mean(fold_alphas_array, axis=0) + 1e-10)
        veRSA_fold_mean = float(np.mean([fr["veRSA"] for fr in fold_results]))

        # Compute fold-fold correlations on log alpha
        from scipy.stats import spearmanr
        fold_corrs = []
        for i in range(len(fold_alphas)):
            for j in range(i + 1, len(fold_alphas)):
                r, _ = spearmanr(
                    np.log10(fold_alphas[i] + 1e-6),
                    np.log10(fold_alphas[j] + 1e-6)
                )
                fold_corrs.append(r)

        alpha_stability = {
            "cv_median": float(np.median(alpha_cv)),
            "cv_mean": float(np.mean(alpha_cv)),
            "cv_std": float(np.std(alpha_cv)),
            "n_folds": len(fold_alphas),
            "fold_correlation_mean": float(np.mean(fold_corrs)) if fold_corrs else None,
            "fold_correlation_min": float(np.min(fold_corrs)) if fold_corrs else None,
        }
        log(f"    Alpha stability: median CV={alpha_stability['cv_median']:.3f}, fold corr={alpha_stability['fold_correlation_mean']:.3f}")

        per_dataset_metrics = {fr["dataset"]: fr for fr in fold_results}

    elif eval_method == "random_kfold":
        # ---------------------------------------------------------------------
        # Random KFold: N random 50/50 splits for alpha stability analysis
        # ---------------------------------------------------------------------
        n_folds = hydra_cfg.fitting.n_folds
        log(f"    Running random_kfold evaluation ({n_folds} folds)...")

        rng = np.random.RandomState(seed)
        fold_results = []
        fold_alphas = []
        Y_pred_per_fold = []  # Store predictions for each fold

        for fold_idx in range(n_folds):
            # Random 50/50 split
            indices = rng.permutation(n_images)
            train_idx = indices[:n_images // 2]
            test_idx = indices[n_images // 2:]

            X_train, X_test = X_full[train_idx], X_full[test_idx]
            Y_train, Y_test = Y_full[train_idx], Y_full[test_idx]

            # Fit with LOO-CV alpha selection
            Y_test_pred, alphas, _ = fit_voxelwise_ridgecv_fast(
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                alphas=alpha_grid,
                verbose=False,
                scale_features=hydra_cfg.fitting.scale_features,
            )

            # Compute per-fold metrics
            r_versa_fold = compute_versa(Y_test, Y_test_pred)
            r_voxel_fold = compute_voxel_r(Y_test, Y_test_pred)

            fold_results.append({
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_idx": test_idx.tolist(),  # Store for prediction aggregation
                "veRSA": float(r_versa_fold),
                "voxel_r_median": float(np.median(r_voxel_fold)),
                "alpha_median": float(np.median(alphas)),
            })
            fold_alphas.append(alphas)
            Y_pred_per_fold.append((test_idx, Y_test_pred))

            log(f"      Fold {fold_idx}: veRSA={r_versa_fold:.4f}, voxel_r={np.median(r_voxel_fold):.4f}")

        # Aggregate predictions: average when image appears in multiple test folds
        Y_pred_sum = np.zeros((n_images, n_voxels), dtype=np.float64)
        Y_pred_count = np.zeros(n_images, dtype=np.int32)
        for test_idx, Y_test_pred in Y_pred_per_fold:
            Y_pred_sum[test_idx] += Y_test_pred
            Y_pred_count[test_idx] += 1
        # Handle images that never appeared in test set (shouldn't happen with n_folds >= 2)
        Y_pred_count = np.maximum(Y_pred_count, 1)
        Y_pred_avg = (Y_pred_sum / Y_pred_count[:, None]).astype(np.float32)

        # Compute overall metrics from averaged predictions
        r_versa = compute_versa(Y_full, Y_pred_avg)
        r_voxel = compute_voxel_r(Y_full, Y_pred_avg)
        r_image = compute_image_r(Y_full, Y_pred_avg)
        veRSA_fold_mean = float(np.mean([fr["veRSA"] for fr in fold_results]))

        log(f"    Overall: veRSA={r_versa:.4f} (fold mean={veRSA_fold_mean:.4f}), voxel_r={np.median(r_voxel):.4f}")

        # Aggregate alphas across folds
        fold_alphas_array = np.stack(fold_alphas, axis=0)  # (n_folds, n_voxels)
        if hydra_cfg.fitting.alpha_aggregation == "median":
            alphas_from_folds = np.median(fold_alphas_array, axis=0)
        else:
            alphas_from_folds = np.mean(fold_alphas_array, axis=0)

        # Compute alpha stability metrics
        alpha_cv = np.std(fold_alphas_array, axis=0) / (np.mean(fold_alphas_array, axis=0) + 1e-10)

        from scipy.stats import spearmanr
        fold_corrs = []
        for i in range(len(fold_alphas)):
            for j in range(i + 1, len(fold_alphas)):
                r, _ = spearmanr(
                    np.log10(fold_alphas[i] + 1e-6),
                    np.log10(fold_alphas[j] + 1e-6)
                )
                fold_corrs.append(r)

        alpha_stability = {
            "cv_median": float(np.median(alpha_cv)),
            "cv_mean": float(np.mean(alpha_cv)),
            "cv_std": float(np.std(alpha_cv)),
            "n_folds": n_folds,
            "fold_correlation_mean": float(np.mean(fold_corrs)) if fold_corrs else None,
            "fold_correlation_min": float(np.min(fold_corrs)) if fold_corrs else None,
        }
        log(f"    Alpha stability: median CV={alpha_stability['cv_median']:.3f}, fold corr={alpha_stability['fold_correlation_mean']:.3f}")

    else:
        # No evaluation possible - will use LOO-CV metrics only
        if eval_method == "group_kfold" and dataset_labels is None:
            log("    WARNING: group_kfold requested but no dataset labels found, skipping evaluation")
        else:
            log(f"    Skipping evaluation (eval_method={eval_method})")

    # =========================================================================
    # Dual-ROI evaluation: compute hlvis metrics if configured
    # =========================================================================
    if r_versa is not None and "hlvis" in hydra_cfg.evaluation.rois:
        # Use the test predictions from the last evaluation to compute hlvis metrics
        # For odd_even: Y_test and Y_test_pred are already defined
        # For group_kfold/random_kfold: use Y_full and Y_pred_all/Y_pred_avg

        if eval_method == "odd_even":
            # Use the test set predictions
            Y_test_hlvis = Y_test[:, hlvis_mask]
            Y_pred_hlvis = Y_test_pred[:, hlvis_mask]
        elif eval_method in ("group_kfold", "random_kfold"):
            # Use the aggregated predictions
            Y_test_hlvis = Y_full[:, hlvis_mask]
            if eval_method == "random_kfold":
                Y_pred_hlvis = Y_pred_avg[:, hlvis_mask]
            else:
                Y_pred_hlvis = Y_pred_all[:, hlvis_mask]
        else:
            Y_test_hlvis = None
            Y_pred_hlvis = None

        if Y_test_hlvis is not None and Y_pred_hlvis is not None:
            r_versa_hlvis = compute_versa(Y_test_hlvis, Y_pred_hlvis)
            r_voxel_hlvis = compute_voxel_r(Y_test_hlvis, Y_pred_hlvis)
            log(f"    hlvis metrics: veRSA={r_versa_hlvis:.4f}, voxel_r={np.median(r_voxel_hlvis):.4f}")

    # =========================================================================
    # Final Model: Use aggregated fold alphas or LOO-CV on ALL data
    # =========================================================================
    scale_features = hydra_cfg.fitting.scale_features

    # Check if we have aggregated alphas from kfold evaluation
    if fold_alphas_array is not None:
        # Use alphas aggregated from folds (median/mean per voxel)
        alphas_chosen = alphas_from_folds.astype(np.float32)
        log(f"    Using aggregated fold alphas ({hydra_cfg.fitting.alpha_aggregation}): "
            f"median={np.median(alphas_chosen):.1f}, range=[{alphas_chosen.min():.1f}, {alphas_chosen.max():.1f}]")

        # Refit on all images with chosen alphas
        W_raw, b_raw, feature_mean, feature_scale = refit_with_chosen_alphas_fast(
            X=X_full, Y=Y_full, alphas=alphas_chosen, verbose=True,
            scale_features=scale_features,
        )

        # Create metrics (no LOO-CV scores since we used fold alphas)
        metrics = {
            "alpha_source": "fold_aggregation",
            "alpha_aggregation": hydra_cfg.fitting.alpha_aggregation,
        }
    else:
        # Fall back to LOO-CV on all data for alpha selection (odd_even or no folds)
        log(f"    Fitting final model on all data with LOO-CV alpha selection (scale_features={scale_features})...")

        from cstims.encoding.ridge_gcv_fast import RidgeCVFast
        from sklearn.preprocessing import StandardScaler

        # Optionally scale features
        if scale_features:
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(X_full).astype(np.float64)
        else:
            X_proc = X_full.astype(np.float64)
        Y_f64 = Y_full.astype(np.float64)

        ridge_final = RidgeCVFast(
            alphas=alpha_grid,
            scoring="pearson_r",
            alpha_per_target=True,
            gcv_mode="svd",
        )
        ridge_final.fit(X_proc, Y_f64)
        alphas_chosen = ridge_final.alpha_.astype(np.float32)

        log(f"    Final LOO-CV score: mean={ridge_final.best_score_.mean():.4f}, median={np.median(ridge_final.best_score_):.4f}")
        log(f"    Final alpha: median={np.median(alphas_chosen):.1f}, range=[{alphas_chosen.min():.1f}, {alphas_chosen.max():.1f}]")

        # Refit on all images with chosen alphas (fast version)
        W_raw, b_raw, feature_mean, feature_scale = refit_with_chosen_alphas_fast(
            X=X_full, Y=Y_full, alphas=alphas_chosen, verbose=True,
            scale_features=scale_features,
        )

        # Create metrics with LOO-CV scores
        metrics = {
            "alpha_source": "loo_cv",
            "loo_cv_score_mean": float(ridge_final.best_score_.mean()),
            "loo_cv_score_median": float(np.median(ridge_final.best_score_)),
        }

    # Add evaluation metrics if available
    if r_versa is not None:
        metrics["veRSA_pearson_r"] = float(r_versa)  # Backwards compat
        metrics["veRSA_visual"] = float(r_versa)     # New explicit name
        metrics["voxel_r_median"] = float(np.median(r_voxel))
        metrics["voxel_r_mean"] = float(np.mean(r_voxel))
        metrics["image_r_median"] = float(np.median(r_image))

    # Add fold mean veRSA if available
    if veRSA_fold_mean is not None:
        metrics["veRSA_fold_mean"] = veRSA_fold_mean

    # Add hlvis metrics if computed
    if r_versa_hlvis is not None:
        metrics["veRSA_hlvis"] = float(r_versa_hlvis)
        metrics["voxel_r_median_hlvis"] = float(np.median(r_voxel_hlvis))
        metrics["n_voxels_hlvis"] = n_voxels_hlvis

    # Add per-dataset metrics if available
    if per_dataset_metrics is not None:
        metrics["per_dataset"] = per_dataset_metrics

    encoding_model = create_encoding_model(
        weights=W_raw,
        intercept=b_raw,
        alphas=alphas_chosen,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        brain_space_info=benchmark.get_brain_space_info(),
        subject=subject,
        model_name=cfg.model,
        layer=str(cfg.layer),
        source=cfg.source,
        cve_threshold=hydra_cfg.benchmark.cve_threshold,
        metrics=metrics,
    )

    encoding_model.save(out_npz)

    # Build cross-dataset evaluation info (only for group_kfold)
    cross_dataset_info = None
    if fold_results is not None and eval_method == "group_kfold":
        cross_dataset_info = {
            "n_datasets": len(fold_results),
            "datasets": [fr["dataset"] for fr in fold_results],
            "veRSA_per_dataset": {fr["dataset"]: fr["veRSA"] for fr in fold_results},
            "veRSA_overall": float(r_versa),
            "veRSA_mean": float(np.mean([fr["veRSA"] for fr in fold_results])),
            "veRSA_std": float(np.std([fr["veRSA"] for fr in fold_results])),
            "alpha_cv_median": float(np.median(alpha_cv)) if alpha_cv is not None else None,
        }

    # Save fold alphas if available and configured
    if fold_alphas_array is not None and hydra_cfg.fitting.store_fold_alphas:
        np.save(model_dir / "fold_alphas.npy", fold_alphas_array.astype(np.float32))
        log(f"    Saved fold alphas: {fold_alphas_array.shape}")

    # Save metadata
    meta = {
        "model": cfg.model,
        "source": cfg.source,
        "layer": str(cfg.layer),
        "subject": subject,
        "voxel_set": hydra_cfg.benchmark.voxel_set,
        "eval_method": eval_method,
        "preprocessing": {
            "response_zscore": hydra_cfg.preprocessing.response_zscore,
            "srp": hydra_cfg.preprocessing.srp,
            "srp_eps": hydra_cfg.preprocessing.srp_eps if hydra_cfg.preprocessing.srp else None,
            "scale_features": scale_features,
        },
        "metrics": metrics,
        "alpha_stability": alpha_stability,
        "fold_metrics": fold_results,
        "cross_dataset_evaluation": cross_dataset_info,
        "alphas": {
            "grid_min": float(alpha_grid.min()),
            "grid_max": float(alpha_grid.max()),
            "n": len(alpha_grid),
            "chosen_median": float(np.median(alphas_chosen)),
            "chosen_min": float(alphas_chosen.min()),
            "chosen_max": float(alphas_chosen.max()),
        },
        "date": datetime.now().isoformat(),
        "elapsed_sec": time.time() - start_time,
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # ridge_final only exists when LOO-CV path was used (no fold alphas)
    loo_cv_score = float(ridge_final.best_score_.mean()) if 'ridge_final' in dir() else None

    if r_versa is not None:
        log(f"    Done in {time.time() - start_time:.1f}s | veRSA={r_versa:.4f}, voxel_r={np.median(r_voxel):.4f}")
    else:
        log(f"    Done in {time.time() - start_time:.1f}s | LOO-CV={loo_cv_score}")

    # Build flattened result dict (parquet doesn't handle nested dicts well)
    result = {
        "model": cfg.model,
        "layer": str(cfg.layer),
        "subject": subject,
        "veRSA": r_versa if r_versa is not None else loo_cv_score,
        "voxel_r_median": float(np.median(r_voxel)) if r_voxel is not None else None,
        "loo_cv_score": loo_cv_score,
        "alpha_median": float(np.median(alphas_chosen)),
    }

    # Add per-dataset veRSA as separate columns
    if cross_dataset_info is not None:
        for dataset, versa_val in cross_dataset_info.get("veRSA_per_dataset", {}).items():
            result[f"veRSA_{dataset}"] = versa_val
        result["veRSA_mean_across_datasets"] = cross_dataset_info.get("veRSA_mean")
        result["veRSA_std_across_datasets"] = cross_dataset_info.get("veRSA_std")

    return result


@hydra.main(version_base=None, config_path="config", config_name="encoding_fit")
def main(cfg: DictConfig):
    print("=" * 60)
    print("DeepVision Encoding Model Fitting")
    print("=" * 60)
    print(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")

    # Resolve paths
    model_list_csv = Path(cfg.paths.model_list_csv)
    cache_root = Path(cfg.paths.cache_root)
    deepvision_root = Path(cfg.paths.deepvision_root)

    # Determine output directory
    if cfg.resume_from:
        run_dir = Path(cfg.resume_from)
        log(f"Resuming from: {run_dir}")
    else:
        run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        log(f"Output directory: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Read model list
    all_models = read_model_list(model_list_csv)
    log(f"Found {len(all_models)} models in {model_list_csv}")

    # Filter models if needed
    if cfg.parallel.model_filter:
        all_models = [m for m in all_models if cfg.parallel.model_filter in m.model]
        log(f"Filtered to {len(all_models)} models matching '{cfg.parallel.model_filter}'")

    # Get batch of models if specified
    if cfg.parallel.model_batch is not None:
        all_models = get_model_batch(
            all_models, cfg.parallel.model_batch, cfg.parallel.models_per_batch
        )
        log(f"Processing batch {cfg.parallel.model_batch}: {len(all_models)} models")

    # Determine subjects
    if cfg.parallel.subject:
        subjects = [cfg.parallel.subject]
    else:
        subjects = list(cfg.subjects)
    log(f"Subjects: {subjects}")

    # Process each subject
    all_results = []
    for subject in subjects:
        log(f"\n{'=' * 60}")
        log(f"Subject: {subject}")
        log(f"{'=' * 60}")

        # Load benchmark data
        log("Loading benchmark data...")
        benchmark = DeepVisionBenchmark(
            cache_root=cache_root,
            deepvision_fmri_root=deepvision_root,
            subject=subject,
            voxel_set=cfg.benchmark.voxel_set,
            cve_threshold=cfg.benchmark.cve_threshold,
            input_source=cfg.benchmark.input_source,
            image_set=cfg.benchmark.get("image_set", "shared"),
            build_rdms=False,
        )
        log(f"Loaded {benchmark.n_stimuli} stimuli, {benchmark.response_data.shape[0]} voxels")

        # Check completed
        completed = get_completed_models(run_dir, subject)
        if completed:
            log(f"Already completed: {len(completed)} models")

        # Process models
        for model_idx, model_cfg in enumerate(all_models):
            if model_cfg.model in completed:
                log(f"[{model_idx+1}/{len(all_models)}] Skipping {model_cfg.model} (completed)")
                continue

            try:
                log(f"[{model_idx+1}/{len(all_models)}] {model_cfg.model}")
                result = fit_model_for_cfg(model_cfg, cfg, benchmark, subject, run_dir)
                if result:
                    all_results.append(result)
            except Exception as e:
                log(f"  ERROR: {e}")
                traceback.print_exc()
                continue

    # Save summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        batch_suffix = f"_batch{cfg.parallel.model_batch}" if cfg.parallel.model_batch is not None else ""
        subject_suffix = f"_{cfg.parallel.subject}" if cfg.parallel.subject else ""
        results_df.to_parquet(run_dir / f"results{subject_suffix}{batch_suffix}.parquet", index=False)
        log(f"\nSaved {len(results_df)} results")

        # Summary
        log("\n=== Summary ===")
        summary = results_df.groupby("subject")["veRSA"].agg(["mean", "std", "count"])
        print(summary)

    log("\nDone!")


if __name__ == "__main__":
    main()
