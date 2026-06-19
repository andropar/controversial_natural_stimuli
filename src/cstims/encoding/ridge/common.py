from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve, pinvh

ArrayLike = np.ndarray | Sequence[float]


ArrayLike = np.ndarray | Sequence[float]


def _as_2d_targets(y: np.ndarray) -> tuple[np.ndarray, bool]:
    y = np.asarray(y)
    if y.ndim == 1:
        return y.reshape(-1, 1), True
    if y.ndim != 2:
        raise ValueError(f"Expected y to be 1D or 2D, got shape {y.shape}")
    return y, False


def _matrix(x: Any, name: str) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}")
    return array


def _check_2d_array(name: str, array: np.ndarray) -> np.ndarray:
    return _matrix(array, name)


def _check_sample_weight(
    sample_weight: np.ndarray | None,
    n_samples: int,
) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weight = np.asarray(sample_weight, dtype=np.float64)
    if weight.shape != (n_samples,):
        raise ValueError(
            f"sample_weight must have shape ({n_samples},), got {weight.shape}"
        )
    if np.any(weight < 0):
        raise ValueError("sample_weight cannot contain negative values")
    if float(weight.sum()) <= 0.0:
        raise ValueError("sample_weight must have positive total weight")
    return weight


@dataclass(frozen=True)
class StandardizationStats:
    """Column-wise affine transform used by ridge preprocessing."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        *,
        weight: np.ndarray | None = None,
        center: bool = False,
        standardize: bool = False,
        eps: float = 0.0,
        scale_factor: float = 1.0,
    ) -> "StandardizationStats":
        x = _matrix(x, "x")
        n_features = x.shape[1]
        if center or standardize:
            mean = np.average(x, axis=0, weights=weight)
        else:
            mean = np.zeros(n_features, dtype=np.float64)

        scale = np.ones(n_features, dtype=np.float64)
        if standardize:
            variance = np.average((x - mean[None, :]) ** 2, axis=0, weights=weight)
            scale = np.sqrt(np.maximum(variance, 0.0))
            invalid = scale < eps if eps > 0 else scale == 0.0
            scale[invalid] = 1.0
            scale *= float(scale_factor)

        return cls(mean=mean.astype(np.float64), scale=scale.astype(np.float64))

    @classmethod
    def identity(cls, n_features: int) -> "StandardizationStats":
        return cls(
            mean=np.zeros(n_features, dtype=np.float64),
            scale=np.ones(n_features, dtype=np.float64),
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean[None, :]) / self.scale[None, :]

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x * self.scale[None, :] + self.mean[None, :]


def _identity_stats(n_features: int) -> StandardizationStats:
    return StandardizationStats.identity(n_features)


def _fit_stats(
    x: np.ndarray,
    *,
    sample_weight: np.ndarray | None,
    center: bool,
    standardize: bool,
    eps: float,
    scale_by_sqrt_features: bool = False,
) -> StandardizationStats:
    x = _matrix(x, "x")
    return StandardizationStats.fit(
        x,
        weight=sample_weight,
        center=center,
        standardize=standardize,
        eps=eps,
        scale_factor=math.sqrt(x.shape[1]) if scale_by_sqrt_features else 1.0,
    )


def _symmetric_inverse_spd(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    try:
        factor = cho_factor(matrix, lower=True, check_finite=False)
        inverse = cho_solve(
            factor,
            np.eye(matrix.shape[0], dtype=np.float64),
            check_finite=False,
        )
    except Exception:
        inverse = pinvh(matrix, check_finite=False)
    return np.asarray(0.5 * (inverse + inverse.T), dtype=np.float64)


@dataclass
class _ProcessedFit:
    weights: np.ndarray
    intercept: np.ndarray
    alphas: np.ndarray
    score: np.ndarray | None
    estimator: Any = None


def _coef_to_weights(
    coef: np.ndarray,
    n_features: int,
    n_targets: int,
) -> np.ndarray:
    coef = np.asarray(coef, dtype=np.float64)
    if coef.ndim == 1:
        weights = coef.reshape(-1, 1)
    else:
        weights = coef.T
    if weights.shape != (n_features, n_targets):
        raise ValueError(
            f"Expected coefficient weights {(n_features, n_targets)}, got {weights.shape}"
        )
    return weights


def _pearson_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch for column correlations: {x.shape} vs {y.shape}")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    out = np.full(x.shape[1], np.nan, dtype=np.float64)
    ok = denom > 0
    out[ok] = np.sum(x[:, ok] * y[:, ok], axis=0) / denom[ok]
    return out


def _normalize_scoring(scoring: str | None) -> str:
    if scoring is None:
        return "neg_mean_squared_error"
    if scoring in {"neg_mean_squared_error", "neg_mse", "mse"}:
        return "neg_mean_squared_error"
    if scoring == "pearson_r":
        return "pearson_r"
    raise ValueError(
        "scoring must be one of 'neg_mean_squared_error', 'neg_mse', 'mse', "
        "'pearson_r', or None"
    )


def _score_columns(prediction: np.ndarray, target: np.ndarray, scoring: str) -> np.ndarray:
    scoring = _normalize_scoring(scoring)
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch for scoring: {prediction.shape} vs {target.shape}"
        )
    if scoring == "pearson_r":
        return _pearson_columns(prediction, target)
    return -np.mean((target - prediction) ** 2, axis=0)


