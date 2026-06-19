from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.linear_model import Ridge, RidgeCV

from ..model import LinearEncodingModel
from ..ridge_gcv_fast import RidgeCVFast
from .backends import _torch_fit_processed, _torch_refit_processed
from .common import (
    ArrayLike,
    StandardizationStats,
    _ProcessedFit,
    _as_2d_targets,
    _check_2d_array,
    _check_sample_weight,
    _coef_to_weights,
    _fit_stats,
    _matrix,
    _normalize_scoring,
)

@dataclass(frozen=True)
class EncodingRidgeSpec:
    """Statistical contract for feature-to-response ridge regression."""

    alphas: tuple[float, ...]
    alpha_per_target: bool = True
    scoring: str | None = "pearson_r"
    fit_intercept: bool = True
    standardize_x: bool = True
    standardize_y: bool = False
    feature_scale_mode: str = "zscore"
    x_scale_eps: float = 0.0
    y_scale_eps: float = 0.0
    gcv_mode: str | None = None

    def __post_init__(self) -> None:
        alphas = tuple(float(alpha) for alpha in self.alphas)
        if not alphas:
            raise ValueError("alphas must contain at least one value")
        if any(alpha <= 0.0 for alpha in alphas):
            raise ValueError("alphas must be strictly positive")
        if self.feature_scale_mode not in {"zscore", "zscore_sqrt_features"}:
            raise ValueError(
                "feature_scale_mode must be 'zscore' or 'zscore_sqrt_features'"
            )
        object.__setattr__(self, "alphas", alphas)
        object.__setattr__(self, "scoring", _normalize_scoring(self.scoring))

    @classmethod
    def from_params(
        cls,
        *,
        alphas: ArrayLike,
        alpha_per_target: bool = True,
        scoring: str | None = "pearson_r",
        fit_intercept: bool = True,
        standardize_x: bool = True,
        standardize_y: bool = False,
        feature_scale_mode: str = "zscore",
        x_scale_eps: float = 0.0,
        y_scale_eps: float = 0.0,
        gcv_mode: str | None = None,
    ) -> "EncodingRidgeSpec":
        return cls(
            alphas=tuple(float(alpha) for alpha in np.atleast_1d(alphas)),
            alpha_per_target=alpha_per_target,
            scoring=scoring,
            fit_intercept=fit_intercept,
            standardize_x=standardize_x,
            standardize_y=standardize_y,
            feature_scale_mode=feature_scale_mode,
            x_scale_eps=x_scale_eps,
            y_scale_eps=y_scale_eps,
            gcv_mode=gcv_mode,
        )


class EncodingRidgeCV(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Sklearn-like ridge estimator for image-feature encoding models.

    ``fit`` selects alphas with analytical LOO via :class:`RidgeCVFast` for
    Pearson scoring, or sklearn ``RidgeCV`` for explicit neg-MSE scoring.
    ``fit_with_alphas`` performs the final fixed-alpha refit used by existing
    encoding-model scripts.
    """

    def __init__(
        self,
        alphas: ArrayLike = (0.1, 1.0, 10.0),
        *,
        alpha_per_target: bool = True,
        scoring: str | None = "pearson_r",
        fit_intercept: bool = True,
        standardize_x: bool = True,
        standardize_y: bool = False,
        feature_scale_mode: str = "zscore",
        x_scale_eps: float = 0.0,
        y_scale_eps: float = 0.0,
        gcv_mode: str | None = None,
        backend: str = "sklearn",
        torch_device: str | None = None,
        torch_dtype: str = "float32",
    ):
        self.alphas = alphas
        self.alpha_per_target = alpha_per_target
        self.scoring = scoring
        self.fit_intercept = fit_intercept
        self.standardize_x = standardize_x
        self.standardize_y = standardize_y
        self.feature_scale_mode = feature_scale_mode
        self.x_scale_eps = x_scale_eps
        self.y_scale_eps = y_scale_eps
        self.gcv_mode = gcv_mode
        self.backend = backend
        self.torch_device = torch_device
        self.torch_dtype = torch_dtype

    def _spec(self) -> EncodingRidgeSpec:
        return EncodingRidgeSpec.from_params(
            alphas=self.alphas,
            alpha_per_target=self.alpha_per_target,
            scoring=self.scoring,
            fit_intercept=self.fit_intercept,
            standardize_x=self.standardize_x,
            standardize_y=self.standardize_y,
            feature_scale_mode=self.feature_scale_mode,
            x_scale_eps=self.x_scale_eps,
            y_scale_eps=self.y_scale_eps,
            gcv_mode=self.gcv_mode,
        )

    def _prepare_fit_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
        spec = self._spec()
        X = _check_2d_array("X", np.asarray(X, dtype=np.float64))
        y_2d, y_was_1d = _as_2d_targets(np.asarray(y, dtype=np.float64))
        if X.shape[0] != y_2d.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows: {X.shape[0]} vs {y_2d.shape[0]}"
            )
        weight = _check_sample_weight(sample_weight, X.shape[0])

        self.spec_ = spec
        self.n_features_in_ = X.shape[1]
        self.n_targets_ = y_2d.shape[1]
        self._y_was_1d = y_was_1d

        scale_by_sqrt = spec.feature_scale_mode == "zscore_sqrt_features"
        self.x_stats_ = _fit_stats(
            X,
            sample_weight=weight,
            center=spec.standardize_x,
            standardize=spec.standardize_x,
            eps=spec.x_scale_eps,
            scale_by_sqrt_features=scale_by_sqrt,
        )
        X_proc = self.x_stats_.transform(X)

        self.y_stats_ = _fit_stats(
            y_2d,
            sample_weight=weight,
            center=spec.standardize_y,
            standardize=spec.standardize_y,
            eps=spec.y_scale_eps,
        )
        y_proc = self.y_stats_.transform(y_2d) if spec.standardize_y else y_2d
        return X_proc, y_proc, weight, y_was_1d

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "EncodingRidgeCV":
        """Fit ridge and choose alphas by analytical leave-one-out CV."""
        X_proc, y_proc, weight, _y_was_1d = self._prepare_fit_arrays(
            X,
            y,
            sample_weight,
        )
        spec = self.spec_
        if self.backend in {"torch", "gpu", "cuda"}:
            if weight is not None:
                raise ValueError("The torch ridge backend does not support sample_weight yet")
            result = _torch_fit_processed(
                X_proc,
                y_proc,
                spec.alphas,
                fit_intercept=spec.fit_intercept,
                alpha_per_target=spec.alpha_per_target,
                scoring=spec.scoring,
                device=self.torch_device if self.backend == "torch" else self.backend,
                dtype=self.torch_dtype,
            )
            self.backend_ = "torch"
            return self._finish_fit(result, alpha_source="loo_cv")

        if self.backend not in {"auto", "sklearn"}:
            raise ValueError("backend must be 'sklearn', 'auto', 'torch', 'gpu', or 'cuda'")
        if spec.scoring == "pearson_r":
            ridge = RidgeCVFast(
                alphas=np.asarray(spec.alphas, dtype=np.float64),
                scoring=spec.scoring,
                alpha_per_target=spec.alpha_per_target,
                fit_intercept=spec.fit_intercept,
                gcv_mode=spec.gcv_mode,
            )
        else:
            ridge = RidgeCV(
                alphas=np.asarray(spec.alphas, dtype=np.float64),
                scoring=None,
                alpha_per_target=spec.alpha_per_target,
                fit_intercept=spec.fit_intercept,
                gcv_mode=spec.gcv_mode,
            )
        ridge.fit(X_proc, y_proc, sample_weight=weight)

        self.backend_ = "sklearn"
        alpha = np.asarray(ridge.alpha_, dtype=np.float64)
        if alpha.ndim == 0:
            alpha = np.full(self.n_targets_, float(alpha), dtype=np.float64)
        score = np.asarray(ridge.best_score_, dtype=np.float64)
        result = _ProcessedFit(
            weights=_coef_to_weights(ridge.coef_, self.n_features_in_, self.n_targets_),
            intercept=np.asarray(ridge.intercept_, dtype=np.float64).reshape(-1),
            alphas=alpha,
            score=score,
            estimator=ridge,
        )
        return self._finish_fit(result, alpha_source="loo_cv")

    def fit_with_alphas(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alphas: ArrayLike,
        sample_weight: np.ndarray | None = None,
    ) -> "EncodingRidgeCV":
        """Fit ridge with fixed scalar or per-target alphas.

        This mirrors the final refit stage of the existing encoding scripts.
        """
        X_proc, y_proc, weight, _y_was_1d = self._prepare_fit_arrays(
            X,
            y,
            sample_weight,
        )
        alpha_arr = np.asarray(alphas, dtype=np.float64)
        if alpha_arr.ndim == 0:
            alpha_arr = np.full(self.n_targets_, float(alpha_arr), dtype=np.float64)
        if alpha_arr.shape != (self.n_targets_,):
            raise ValueError(
                f"alphas must be scalar or have shape ({self.n_targets_},), got {alpha_arr.shape}"
            )
        if np.any(alpha_arr <= 0.0):
            raise ValueError("alphas must be strictly positive")

        if self.backend in {"torch", "gpu", "cuda"}:
            if weight is not None:
                raise ValueError("The torch ridge backend does not support sample_weight yet")
            W_proc, b_proc = _torch_refit_processed(
                X_proc,
                y_proc,
                alpha_arr,
                fit_intercept=self.spec_.fit_intercept,
                device=self.torch_device if self.backend == "torch" else self.backend,
                dtype=self.torch_dtype,
            )
            self.backend_ = "torch"
        else:
            if self.backend not in {"auto", "sklearn"}:
                raise ValueError("backend must be 'sklearn', 'auto', 'torch', 'gpu', or 'cuda'")
            ridge = Ridge(alpha=alpha_arr, fit_intercept=self.spec_.fit_intercept)
            ridge.fit(X_proc, y_proc, sample_weight=weight)
            W_proc = _coef_to_weights(ridge.coef_, self.n_features_in_, self.n_targets_)
            b_proc = np.asarray(ridge.intercept_, dtype=np.float64).reshape(-1)
            self.backend_ = "sklearn"

        return self._finish_fit(
            _ProcessedFit(
                weights=W_proc,
                intercept=b_proc,
                alphas=alpha_arr,
                score=None,
                estimator=None,
            ),
            alpha_source="fixed",
        )

    def _finish_fit(
        self,
        result: _ProcessedFit,
        *,
        alpha_source: str,
    ) -> "EncodingRidgeCV":
        self.alpha_ = np.asarray(result.alphas, dtype=np.float64)
        if self.alpha_.ndim == 0:
            self.alpha_ = np.full(self.n_targets_, float(self.alpha_), dtype=np.float64)
        self.best_score_ = (
            None if result.score is None else np.asarray(result.score, dtype=np.float64)
        )
        self.estimator_ = result.estimator
        self.alpha_source_ = alpha_source
        self._store_processed_weights(result.weights, result.intercept)
        return self

    def _store_processed_coefficients(
        self,
        coef: np.ndarray,
        intercept: np.ndarray | float,
    ) -> None:
        self._store_processed_weights(
            _coef_to_weights(coef, self.n_features_in_, self.n_targets_),
            np.asarray(intercept, dtype=np.float64).reshape(-1),
        )

    def _store_processed_weights(
        self,
        weights: np.ndarray,
        intercept: np.ndarray | float,
    ) -> None:
        W_proc = np.asarray(weights, dtype=np.float64)
        b_proc = np.asarray(intercept, dtype=np.float64).reshape(-1)
        if b_proc.shape == (1,) and self.n_targets_ > 1:
            b_proc = np.repeat(b_proc, self.n_targets_)
        if W_proc.shape != (self.n_features_in_, self.n_targets_):
            raise ValueError(
                "Processed coefficient shape mismatch: "
                f"expected {(self.n_features_in_, self.n_targets_)}, got {W_proc.shape}"
            )
        if b_proc.shape != (self.n_targets_,):
            raise ValueError(
                f"Processed intercept shape mismatch: expected {(self.n_targets_,)}, got {b_proc.shape}"
            )

        x_mean = self.x_stats_.mean
        x_scale = self.x_stats_.scale
        W_model = W_proc / x_scale[:, None]
        b_model = b_proc - (x_mean / x_scale) @ W_proc

        self.weights_model_space_ = W_model.astype(np.float64)
        self.intercept_model_space_ = b_model.astype(np.float64)

        if self.spec_.standardize_y:
            y_scale = self.y_stats_.scale
            y_mean = self.y_stats_.mean
            W_target = W_model * y_scale[None, :]
            b_target = b_model * y_scale + y_mean
            self.weights_ = W_target.astype(np.float64)
            self.intercept_ = b_target.astype(np.float64)
        else:
            self.weights_ = self.weights_model_space_
            self.intercept_ = self.intercept_model_space_
        self.coef_ = self.weights_.T

        self.feature_mean_ = self.x_stats_.mean.astype(np.float64)
        self.feature_scale_ = self.x_stats_.scale.astype(np.float64)
        self.response_mean_ = self.y_stats_.mean.astype(np.float64)
        self.response_scale_ = self.y_stats_.scale.astype(np.float64)

    def predict(
        self,
        X: np.ndarray,
        *,
        output_space: str = "target",
    ) -> np.ndarray:
        """Predict responses.

        Parameters
        ----------
        output_space:
            ``"target"`` returns the input target scale.  ``"model"`` returns
            the internal target space used for ridge fitting, which differs only
            when ``standardize_y=True``.
        """
        if not hasattr(self, "weights_"):
            raise RuntimeError("EncodingRidgeCV is not fitted")
        X = _check_2d_array("X", np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features_in_}, got {X.shape[1]}"
            )
        if output_space == "target":
            pred = X @ self.weights_ + self.intercept_[None, :]
        elif output_space in {"model", "standardized"}:
            pred = X @ self.weights_model_space_ + self.intercept_model_space_[None, :]
        else:
            raise ValueError("output_space must be 'target', 'model', or 'standardized'")
        if self._y_was_1d:
            return pred.ravel()
        return pred

    def to_linear_encoding_model(
        self,
        *,
        brain_space_info: Mapping[str, Any],
        subject: str,
        model_name: str,
        layer: str,
        source: str,
        cve_threshold: float,
        metrics: Mapping[str, Any] | None = None,
        output_space: str = "target",
    ) -> LinearEncodingModel:
        """Export the fitted estimator as the existing persisted artifact."""
        if output_space == "target":
            weights = self.weights_
            intercept = self.intercept_
        elif output_space in {"model", "standardized"}:
            weights = self.weights_model_space_
            intercept = self.intercept_model_space_
        else:
            raise ValueError("output_space must be 'target', 'model', or 'standardized'")

        n_voxels = weights.shape[1]
        hlvis_mask = np.asarray(
            brain_space_info.get("hlvis_mask", np.ones(n_voxels, dtype=np.bool_)),
            dtype=np.bool_,
        )
        roi_masks = {
            "visual": np.ones(n_voxels, dtype=np.bool_),
            "hlvis": hlvis_mask,
        }
        model_metrics = dict(metrics or {})
        model_metrics.setdefault("ridge_alpha_source", self.alpha_source_)
        model_metrics.setdefault("ridge_standardize_x", bool(self.spec_.standardize_x))
        model_metrics.setdefault("ridge_standardize_y", bool(self.spec_.standardize_y))
        model_metrics.setdefault("ridge_fit_intercept", bool(self.spec_.fit_intercept))
        model_metrics.setdefault("ridge_output_space", output_space)

        return LinearEncodingModel(
            weights=np.asarray(weights, dtype=np.float32),
            intercept=np.asarray(intercept, dtype=np.float32),
            alphas=np.asarray(self.alpha_, dtype=np.float32),
            feature_mean=np.asarray(self.feature_mean_, dtype=np.float32),
            feature_scale=np.asarray(self.feature_scale_, dtype=np.float32),
            volume_shape=tuple(brain_space_info["volume_shape"]),
            affine=np.asarray(brain_space_info["affine"], dtype=np.float64),
            voxel_indices=np.asarray(brain_space_info["voxel_indices"], dtype=np.int64),
            roi_masks=roi_masks,
            subject=subject,
            model_name=model_name,
            layer=layer,
            source=source,
            cve_threshold=float(cve_threshold),
            metrics=model_metrics,
        )

    def build_readout_cache(
        self,
        *,
        X_train: np.ndarray,
        X_val: np.ndarray,
        eval_sets: Mapping[str, np.ndarray],
        X_base: np.ndarray | None = None,
        mode: str = "independent",
        backend: str = "auto",
    ) -> "EncodingRidgeReadoutCache":
        return EncodingRidgeReadoutCache.from_features(
            spec=self._spec(),
            X_train=X_train,
            X_val=X_val,
            eval_sets=eval_sets,
            X_base=X_base,
            mode=mode,
            backend=backend,
        )

