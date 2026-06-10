"""Utilities for fitting linear encoding models.

This module provides functions for fitting voxel-wise Ridge regression models
to map visual features to brain responses.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .model import LinearEncodingModel
from .ridge_gcv_fast import RidgeCVFast

_LOG = logging.getLogger(__name__)

__all__ = [
    "fit_voxelwise_ridgecv",
    "fit_voxelwise_ridgecv_fast",
    "refit_with_chosen_alphas",
    "refit_with_chosen_alphas_fast",
    "pearsonr_metric",
    "compute_versa",
    "compute_voxel_r",
    "compute_image_r",
]


def _pearsonr_vec(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Pearson correlation between two 1D arrays safely."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    num = np.dot(yt, yp)
    den = np.sqrt((np.dot(yt, yt) + eps) * (np.dot(yp, yp) + eps))
    if den <= eps or not np.isfinite(den):
        return 0.0
    r = num / den
    return float(r) if np.isfinite(r) else 0.0


def pearsonr_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Metric function that returns Pearson r for use with sklearn make_scorer."""
    return _pearsonr_vec(y_true, y_pred)


def compute_versa(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Compute veRSA as Pearson r between upper triangles of RSMs.

    veRSA (vectorized encoding RSA) measures how well the predicted brain
    responses preserve the representational structure of the true responses.

    Args:
        Y_true: True responses, shape (n_images, n_voxels)
        Y_pred: Predicted responses, shape (n_images, n_voxels)

    Returns:
        Pearson correlation between upper triangles of the RSMs
    """
    def zscore_rows(Y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        Y = np.asarray(Y, dtype=np.float64)
        mean = Y.mean(axis=1, keepdims=True)
        std = Y.std(axis=1, ddof=0, keepdims=True)
        std = np.maximum(std, eps)
        return (Y - mean) / std

    Yt = zscore_rows(Y_true)
    Yp = zscore_rows(Y_pred)
    V = Yt.shape[1]

    R_true = (Yt @ Yt.T) / (V - 1)
    R_pred = (Yp @ Yp.T) / (V - 1)

    idx = np.triu_indices(R_true.shape[0], k=1)
    return _pearsonr_vec(R_true[idx], R_pred[idx])


def compute_voxel_r(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Compute per-voxel Pearson r across images.

    Args:
        Y_true: True responses, shape (n_images, n_voxels)
        Y_pred: Predicted responses, shape (n_images, n_voxels)

    Returns:
        Array of Pearson r values, shape (n_voxels,)
    """
    n_voxels = Y_true.shape[1]
    r_v = np.zeros(n_voxels, dtype=np.float64)
    for v in range(n_voxels):
        r_v[v] = _pearsonr_vec(Y_true[:, v], Y_pred[:, v])
    return r_v


def compute_image_r(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Compute per-image Pearson r across voxels.

    Args:
        Y_true: True responses, shape (n_images, n_voxels)
        Y_pred: Predicted responses, shape (n_images, n_voxels)

    Returns:
        Array of Pearson r values, shape (n_images,)
    """
    n_images = Y_true.shape[0]
    r_img = np.zeros(n_images, dtype=np.float64)
    for i in range(n_images):
        r_img[i] = _pearsonr_vec(Y_true[i, :], Y_pred[i, :])
    return r_img


def fit_voxelwise_ridgecv(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    alphas: np.ndarray,
    n_splits: int = 5,
    n_jobs: int = 1,
    batch_size: int = 512,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit per-voxel RidgeCV on training data and predict on test set.

    Args:
        X_train: Training features, shape (n_train, n_features)
        Y_train: Training targets, shape (n_train, n_voxels)
        X_test: Test features, shape (n_test, n_features)
        alphas: Array of alpha values to search over
        n_splits: Number of CV folds for alpha selection
        n_jobs: Number of parallel jobs
        batch_size: Number of voxels to process in parallel per batch
        seed: Random seed for CV splits
        verbose: Whether to show progress bar

    Returns:
        Tuple of:
            - Y_test_pred: Predictions on test set, shape (n_test, n_voxels)
            - alphas_chosen: Chosen alpha per voxel, shape (n_voxels,)
            - scaler: Fitted StandardScaler for features
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    n_voxels = Y_train.shape[1]
    n_test = X_test.shape[0]
    Y_test_pred = np.empty((n_test, n_voxels), dtype=np.float32)
    alphas_chosen = np.empty(n_voxels, dtype=np.float32)

    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scorer = make_scorer(pearsonr_metric, greater_is_better=True)

    def _fit_one_voxel(v: int) -> Tuple[int, np.ndarray, float]:
        y_tr = Y_train[:, v]
        ridge_cv = RidgeCV(
            alphas=alphas,
            scoring=scorer,
            cv=inner_cv,
            fit_intercept=True,
        )
        ridge_cv.fit(X_train_scaled, y_tr)
        y_te_hat = ridge_cv.predict(X_test_scaled)
        return v, y_te_hat, ridge_cv.alpha_

    results = []
    iterator = range(0, n_voxels, batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Fitting voxel-wise RidgeCV", leave=False)

    for start in iterator:
        end = min(start + batch_size, n_voxels)
        batch_results = Parallel(n_jobs=n_jobs if n_jobs != 0 else 1)(
            delayed(_fit_one_voxel)(v) for v in range(start, end)
        )
        results.extend(batch_results)

    for v, y_te_hat, alpha in results:
        Y_test_pred[:, v] = y_te_hat
        alphas_chosen[v] = alpha

    return Y_test_pred, alphas_chosen, scaler


def fit_voxelwise_ridgecv_fast(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    alphas: np.ndarray,
    verbose: bool = True,
    scale_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """Fast per-voxel RidgeCV using analytical LOO-CV (vectorized).

    This is much faster than fit_voxelwise_ridgecv because it:
    1. Uses analytical Leave-One-Out CV (no refitting per fold)
    2. Fits all voxels simultaneously via matrix operations
    3. Selects optimal alpha per voxel in a single pass

    Args:
        X_train: Training features, shape (n_train, n_features)
        Y_train: Training targets, shape (n_train, n_voxels)
        X_test: Test features, shape (n_test, n_features)
        alphas: Array of alpha values to search over
        verbose: Whether to log progress
        scale_features: Whether to z-score features before fitting (default True)

    Returns:
        Tuple of:
            - Y_test_pred: Predictions on test set, shape (n_test, n_voxels)
            - alphas_chosen: Chosen alpha per voxel, shape (n_voxels,)
            - scaler: Fitted StandardScaler for features (None if scale_features=False)
    """
    if verbose:
        _LOG.info(f"Fitting RidgeCVFast: {Y_train.shape[1]} voxels, {len(alphas)} alphas, "
                  f"scale_features={scale_features}")

    # Optionally standardize features
    if scale_features:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_proc = scaler.fit_transform(X_train).astype(np.float64)
        X_test_proc = scaler.transform(X_test).astype(np.float64)
    else:
        scaler = None
        X_train_proc = X_train.astype(np.float64)
        X_test_proc = X_test.astype(np.float64)

    # Ensure Y is float64 for numerical stability
    Y_train_f64 = Y_train.astype(np.float64)

    # Fit all voxels at once with per-voxel alpha selection
    ridge = RidgeCVFast(
        alphas=alphas,
        scoring="pearson_r",
        alpha_per_target=True,
        fit_intercept=True,
        gcv_mode=None,
    )
    ridge.fit(X_train_proc, Y_train_f64)

    if verbose:
        _LOG.info(f"Best LOO-CV scores: mean={ridge.best_score_.mean():.4f}, "
                  f"median={np.median(ridge.best_score_):.4f}")

    # Predict on test set
    Y_test_pred = ridge.predict(X_test_proc).astype(np.float32)
    alphas_chosen = ridge.alpha_.astype(np.float32)

    return Y_test_pred, alphas_chosen, scaler


def refit_with_chosen_alphas(
    X: np.ndarray,
    Y: np.ndarray,
    alphas: np.ndarray,
    n_jobs: int = 1,
    batch_size: int = 512,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refit Ridge on full data with per-voxel alphas and return raw-space weights.

    The returned weights are in raw feature space (not standardized), so predictions
    can be made as: y_pred = X @ W + b

    Args:
        X: Features, shape (n_samples, n_features)
        Y: Targets, shape (n_samples, n_voxels)
        alphas: Per-voxel alpha values, shape (n_voxels,)
        n_jobs: Number of parallel jobs
        batch_size: Number of voxels to process per batch
        seed: Random seed
        verbose: Whether to show progress bar

    Returns:
        Tuple of:
            - W_raw: Weights in raw feature space, shape (n_features, n_voxels)
            - b_raw: Intercepts in raw feature space, shape (n_voxels,)
            - feature_mean: Feature means used for standardization, shape (n_features,)
            - feature_scale: Feature scales used for standardization, shape (n_features,)
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    n_voxels = Y.shape[1]
    n_features = X.shape[1]

    W_scaled = np.empty((n_voxels, n_features), dtype=np.float32)
    b_scaled = np.empty(n_voxels, dtype=np.float32)

    def _fit_one_voxel(v: int) -> Tuple[int, np.ndarray, float]:
        y = Y[:, v]
        alpha_v = alphas[v]
        ridge = Ridge(alpha=alpha_v, fit_intercept=True, random_state=seed)
        ridge.fit(X_scaled, y)
        return v, ridge.coef_, ridge.intercept_

    results = []
    iterator = range(0, n_voxels, batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Refitting on full data", leave=False)

    for start in iterator:
        end = min(start + batch_size, n_voxels)
        batch_results = Parallel(n_jobs=n_jobs if n_jobs != 0 else 1)(
            delayed(_fit_one_voxel)(v) for v in range(start, end)
        )
        results.extend(batch_results)

    for v, coef, intercept in results:
        W_scaled[v, :] = coef
        b_scaled[v] = intercept

    # Convert to raw feature space
    # If X_scaled = (X - mean) / scale, then y = X_scaled @ W_scaled.T + b_scaled
    # = ((X - mean) / scale) @ W_scaled.T + b_scaled
    # = X @ (W_scaled.T / scale) + (b_scaled - mean @ (W_scaled.T / scale))
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)

    W_raw = (W_scaled.T / scale[:, None]).astype(np.float32)  # (n_features, n_voxels)
    b_raw = (b_scaled - (mean / scale) @ W_scaled.T).astype(np.float32)  # (n_voxels,)

    return W_raw, b_raw, mean, scale


def refit_with_chosen_alphas_fast(
    X: np.ndarray,
    Y: np.ndarray,
    alphas: np.ndarray,
    verbose: bool = True,
    scale_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast refit Ridge on full data with per-voxel alphas.

    Groups voxels by their alpha value and fits each group in one shot
    using multi-output Ridge regression.

    Args:
        X: Features, shape (n_samples, n_features)
        Y: Targets, shape (n_samples, n_voxels)
        alphas: Per-voxel alpha values, shape (n_voxels,)
        verbose: Whether to log progress
        scale_features: Whether to z-score features before fitting (default True)

    Returns:
        Tuple of:
            - W_raw: Weights in raw feature space, shape (n_features, n_voxels)
            - b_raw: Intercepts in raw feature space, shape (n_voxels,)
            - feature_mean: Feature means used for standardization, shape (n_features,)
                           (zeros if scale_features=False)
            - feature_scale: Feature scales used for standardization, shape (n_features,)
                            (ones if scale_features=False)
    """
    n_voxels = Y.shape[1]
    n_features = X.shape[1]

    # Optionally standardize features
    if scale_features:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_proc = scaler.fit_transform(X).astype(np.float64)
        feature_mean = scaler.mean_.astype(np.float32)
        feature_scale = scaler.scale_.astype(np.float32)
    else:
        X_proc = X.astype(np.float64)
        feature_mean = np.zeros(n_features, dtype=np.float32)
        feature_scale = np.ones(n_features, dtype=np.float32)

    Y_f64 = Y.astype(np.float64)

    W_fit = np.empty((n_voxels, n_features), dtype=np.float64)
    b_fit = np.empty(n_voxels, dtype=np.float64)

    # Group voxels by alpha value for efficient batch fitting
    unique_alphas = np.unique(alphas)
    if verbose:
        _LOG.info(f"Refitting with {len(unique_alphas)} unique alpha values, "
                  f"scale_features={scale_features}...")

    for alpha_val in unique_alphas:
        voxel_mask = alphas == alpha_val
        voxel_indices = np.where(voxel_mask)[0]
        Y_subset = Y_f64[:, voxel_mask]

        # Fit multi-output Ridge for all voxels with this alpha
        ridge = Ridge(alpha=float(alpha_val), fit_intercept=True)
        ridge.fit(X_proc, Y_subset)

        # Store results
        W_fit[voxel_indices, :] = ridge.coef_
        b_fit[voxel_indices] = ridge.intercept_

    # Convert to raw feature space if we scaled
    if scale_features:
        W_raw = (W_fit.T / feature_scale[:, None]).astype(np.float32)
        b_raw = (b_fit - (feature_mean / feature_scale) @ W_fit.T).astype(np.float32)
    else:
        # Already in raw space
        W_raw = W_fit.T.astype(np.float32)
        b_raw = b_fit.astype(np.float32)

    return W_raw, b_raw, feature_mean, feature_scale


def create_encoding_model(
    weights: np.ndarray,
    intercept: np.ndarray,
    alphas: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    brain_space_info: Dict,
    subject: str,
    model_name: str,
    layer: str,
    source: str,
    cve_threshold: float,
    metrics: Optional[Dict[str, float]] = None,
) -> LinearEncodingModel:
    """Create a LinearEncodingModel from fitted parameters.

    Args:
        weights: Weight matrix, shape (n_features, n_voxels)
        intercept: Intercept vector, shape (n_voxels,)
        alphas: Per-voxel alphas, shape (n_voxels,)
        feature_mean: Feature means, shape (n_features,)
        feature_scale: Feature scales, shape (n_features,)
        brain_space_info: Dict from DeepVisionBenchmark.get_brain_space_info()
        subject: Subject identifier
        model_name: Model name
        layer: Layer identifier
        source: Model source
        cve_threshold: CVE threshold used
        metrics: Optional evaluation metrics

    Returns:
        LinearEncodingModel instance
    """
    n_voxels = weights.shape[1]

    # Create ROI masks from brain_space_info
    roi_masks = {
        "visual": np.ones(n_voxels, dtype=np.bool_),
        "hlvis": brain_space_info["hlvis_mask"].astype(np.bool_),
    }

    return LinearEncodingModel(
        weights=weights,
        intercept=intercept,
        alphas=alphas,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        volume_shape=brain_space_info["volume_shape"],
        affine=brain_space_info["affine"],
        voxel_indices=brain_space_info["voxel_indices"],
        roi_masks=roi_masks,
        subject=subject,
        model_name=model_name,
        layer=layer,
        source=source,
        cve_threshold=cve_threshold,
        metrics=metrics or {},
    )
