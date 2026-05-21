#!/usr/bin/env python3
"""Compare LOO evaluation vs train/test split evaluation.

This script compares two approaches for encoding model evaluation:
1. Train/test split: Fit on 50% of data, evaluate on held-out 50%
2. LOO on full data: Use LOO predictions from fitting on all data

The goal is to validate that LOO evaluation gives similar/better results
while using all available data.
"""

import sys
from pathlib import Path
import os
import numpy as np
import time

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_share_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "cstims").exists():
            return path
    return start.parents[1]


PROJECT_ROOT = _find_share_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from cstims.datasets.deepvision import DeepVisionBenchmark
from cstims.encoding.ridge_gcv_fast import RidgeCVFast
from cstims.encoding.fitting import compute_versa, compute_voxel_r
from cstims.paths import deepvision_fmri_root
from sklearn.preprocessing import StandardScaler


def run_comparison(
    subject: str = "sub-01",
    model_features_path: str = None,
    n_voxels_subset: int = None,  # Use subset for faster testing
):
    """Run comparison between LOO and split-based evaluation."""

    print("=" * 60)
    print("LOO vs Train/Test Split Comparison")
    print("=" * 60)

    # Load benchmark data
    print(f"\nLoading data for {subject}...")
    benchmark = DeepVisionBenchmark(
        cache_root=Path(
            os.environ.get(
                "CSTIMS_DEEPVISION_CACHE_ROOT",
                PROJECT_ROOT / "01_brain_model_alignment/cache_or_heavy/brain_data",
            )
        ),
        deepvision_fmri_root=deepvision_fmri_root(),
        subject=subject,
        voxel_set="visual",
        cve_threshold=0.2,
        input_source="finalinterp",
        build_rdms=False,
    )

    # Get responses
    Y_full = benchmark.response_data.to_numpy().T  # (n_images, n_voxels)
    n_images, n_voxels = Y_full.shape
    print(f"Loaded {n_images} images, {n_voxels} voxels")

    # Z-score responses
    Y_mean = Y_full.mean(axis=0)
    Y_std = Y_full.std(axis=0) + 1e-6
    Y_full = (Y_full - Y_mean) / Y_std

    # Load or create dummy features
    if model_features_path:
        with np.load(model_features_path) as z:
            X_full = z["features"]
    else:
        # Use cached features from a previous run if available
        cache_path = Path(
            os.environ.get(
                "CSTIMS_ENCODING_OUTPUT_ROOT",
                PROJECT_ROOT / "01_brain_model_alignment/inputs/encoding_models",
            )
        )
        feature_files = list(cache_path.glob("**/vissl_resnet50_supervised*/features.npz"))
        if feature_files:
            print(f"Using cached features from {feature_files[0]}")
            with np.load(feature_files[0]) as z:
                X_full = z["features"]
        else:
            print("No cached features found, using random features for demo")
            X_full = np.random.randn(n_images, 512).astype(np.float32)

    print(f"Features shape: {X_full.shape}")

    # Optionally subset voxels for faster testing
    if n_voxels_subset and n_voxels_subset < n_voxels:
        print(f"Using subset of {n_voxels_subset} voxels for faster testing")
        voxel_idx = np.random.choice(n_voxels, n_voxels_subset, replace=False)
        Y_full = Y_full[:, voxel_idx]
        n_voxels = n_voxels_subset

    alphas = np.logspace(-1, 6, 20)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full).astype(np.float64)
    Y_f64 = Y_full.astype(np.float64)

    results = {}

    # =========================================================================
    # Approach 1: Even/Odd Split
    # =========================================================================
    print("\n" + "-" * 40)
    print("Approach 1: Even/Odd Split")
    print("-" * 40)

    train_idx = np.arange(0, n_images, 2)
    test_idx = np.arange(1, n_images, 2)

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    Y_train, Y_test = Y_f64[train_idx], Y_f64[test_idx]

    start = time.time()
    model_split = RidgeCVFast(
        alphas=alphas,
        scoring="pearson_r",
        alpha_per_target=True,
        gcv_mode="svd",
    )
    model_split.fit(X_train, Y_train)
    Y_test_pred = model_split.predict(X_test)
    elapsed_split = time.time() - start

    versa_split = compute_versa(Y_test, Y_test_pred)
    voxel_r_split = compute_voxel_r(Y_test, Y_test_pred)

    print(f"Time: {elapsed_split:.1f}s")
    print(f"LOO-CV score (on train): mean={model_split.best_score_.mean():.4f}")
    print(f"veRSA (on test): {versa_split:.4f}")
    print(f"Voxel r (on test): median={np.median(voxel_r_split):.4f}, mean={np.mean(voxel_r_split):.4f}")

    results["split"] = {
        "versa": versa_split,
        "voxel_r_median": np.median(voxel_r_split),
        "loo_cv_score": model_split.best_score_.mean(),
        "time": elapsed_split,
    }

    # =========================================================================
    # Approach 2: LOO on Full Data
    # =========================================================================
    print("\n" + "-" * 40)
    print("Approach 2: LOO on Full Data")
    print("-" * 40)

    start = time.time()
    model_full = RidgeCVFast(
        alphas=alphas,
        scoring="pearson_r",
        alpha_per_target=True,
        gcv_mode="svd",
        store_cv_values=True,  # Store LOO predictions for each alpha
    )
    model_full.fit(X_scaled, Y_f64)
    elapsed_loo = time.time() - start

    # Extract LOO predictions for chosen alphas
    # cv_results_ has shape (n_samples, n_targets, n_alphas)
    cv_values = model_full.cv_results_  # (n_samples, n_voxels, n_alphas)

    # For each voxel, get the LOO predictions for its chosen alpha
    Y_loo_pred = np.empty_like(Y_f64)
    for v in range(n_voxels):
        best_alpha = model_full.alpha_[v]
        alpha_idx = np.argmin(np.abs(alphas - best_alpha))
        Y_loo_pred[:, v] = cv_values[:, v, alpha_idx]

    versa_loo = compute_versa(Y_f64, Y_loo_pred)
    voxel_r_loo = compute_voxel_r(Y_f64, Y_loo_pred)

    print(f"Time: {elapsed_loo:.1f}s")
    print(f"LOO-CV score: mean={model_full.best_score_.mean():.4f}")
    print(f"veRSA (from LOO preds): {versa_loo:.4f}")
    print(f"Voxel r (from LOO preds): median={np.median(voxel_r_loo):.4f}, mean={np.mean(voxel_r_loo):.4f}")

    results["loo_full"] = {
        "versa": versa_loo,
        "voxel_r_median": np.median(voxel_r_loo),
        "loo_cv_score": model_full.best_score_.mean(),
        "time": elapsed_loo,
    }

    # =========================================================================
    # Approach 3: Multiple Random Splits (for reference)
    # =========================================================================
    print("\n" + "-" * 40)
    print("Approach 3: Multiple Random Splits (3 seeds)")
    print("-" * 40)

    versa_random = []
    for seed in [42, 43, 44]:
        np.random.seed(seed)
        indices = np.random.permutation(n_images)
        train_idx = indices[:n_images//2]
        test_idx = indices[n_images//2:]

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        Y_train, Y_test = Y_f64[train_idx], Y_f64[test_idx]

        model = RidgeCVFast(
            alphas=alphas,
            scoring="pearson_r",
            alpha_per_target=True,
            gcv_mode="svd",
        )
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        v = compute_versa(Y_test, Y_pred)
        versa_random.append(v)
        print(f"  Seed {seed}: veRSA={v:.4f}")

    print(f"Random splits: mean={np.mean(versa_random):.4f}, std={np.std(versa_random):.4f}")

    results["random_splits"] = {
        "versa_mean": np.mean(versa_random),
        "versa_std": np.std(versa_random),
        "versa_values": versa_random,
    }

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<25} {'veRSA':>10} {'Voxel r (med)':>15} {'Time':>10}")
    print("-" * 60)
    print(f"{'Even/Odd Split':<25} {results['split']['versa']:>10.4f} {results['split']['voxel_r_median']:>15.4f} {results['split']['time']:>10.1f}s")
    print(f"{'LOO on Full Data':<25} {results['loo_full']['versa']:>10.4f} {results['loo_full']['voxel_r_median']:>15.4f} {results['loo_full']['time']:>10.1f}s")
    print(f"{'Random Splits (mean)':<25} {results['random_splits']['versa_mean']:>10.4f} {'N/A':>15} {'N/A':>10}")
    print(f"{'Random Splits (std)':<25} {results['random_splits']['versa_std']:>10.4f}")

    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    print("-" * 60)
    print("""
- LOO on full data uses ALL images for both fitting and evaluation
- Each prediction is still unbiased (made without seeing that sample)
- Lower variance than random splits (no split randomness)
- Uses 2x more training data → potentially better alpha selection
    """)

    return results


if __name__ == "__main__":
    # Run with subset for quick testing, set to None for full comparison
    results = run_comparison(
        subject="sub-01",
        n_voxels_subset=5000,  # Use 5000 voxels for faster testing
    )
