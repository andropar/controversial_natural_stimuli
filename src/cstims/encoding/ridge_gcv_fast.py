"""Fast Ridge regression with analytical Leave-One-Out Cross-Validation.

This module provides RidgeCVFast, a vectorized Ridge regression that:
1. Uses analytical LOO-CV via SVD decomposition (no retraining per fold)
2. Supports alpha_per_target=True for per-voxel optimal alpha selection
3. Fits all voxels simultaneously instead of one-by-one

Based on DeepNSD's ridge_gcv_mod.py, which modifies sklearn's RidgeCV.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from sklearn.base import RegressorMixin, MultiOutputMixin, is_classifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.linear_model._ridge import (
    safe_sparse_dot,
    _RidgeGCV,
    _BaseRidgeCV,
    _rescale_data,
    _preprocess_data,
    _check_gcv_mode,
    _check_sample_weight,
)
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import validate_data


def _pearson_r_score(y_true, y_pred, multioutput=None):
    """Compute Pearson R score column-wise."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    y_true_c = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred_c = y_pred - y_pred.mean(axis=0, keepdims=True)
    numerator = np.sum(y_true_c * y_pred_c, axis=0)
    denominator = np.sqrt(np.sum(y_true_c * y_true_c, axis=0) * np.sum(y_pred_c * y_pred_c, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = numerator / denominator
    score[~np.isfinite(score)] = np.nan
    return score


class _RidgeGCVFast(_RidgeGCV):
    """Ridge regression with built-in Leave-one-out Cross-Validation.

    Uses analytical LOO-CV formula via SVD/eigendecomposition, which is
    much faster than k-fold CV since it doesn't require refitting.
    """

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        scoring=None,
        copy_X=True,
        gcv_mode=None,
        store_cv_values=False,
        is_clf=False,
        alpha_per_target=False,
    ):
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.copy_X = copy_X
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
        self.is_clf = is_clf
        self.alpha_per_target = alpha_per_target

    def fit(self, X, y, sample_weight=None):
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=["csr", "csc", "coo"],
            dtype=[np.float64],
            multi_output=True,
            y_numeric=True,
        )

        assert not (self.is_clf and self.alpha_per_target)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        if np.any(self.alphas <= 0):
            raise ValueError(
                f"alphas must be strictly positive. Got {self.alphas} containing some "
                "negative or null value instead."
            )

        unscaled_y = y
        try:
            preprocessed = _preprocess_data(
                X,
                y,
                fit_intercept=self.fit_intercept,
                copy=self.copy_X,
                sample_weight=sample_weight,
                rescale_with_sw=True,
            )
        except TypeError:
            preprocessed = _preprocess_data(
                X,
                y,
                fit_intercept=self.fit_intercept,
                copy=self.copy_X,
                sample_weight=sample_weight,
            )

        # sklearn >= 1.7 returns sample_weight_sqrt and already rescales
        # X/y when sample_weight is provided. Older versions returned only the
        # first five values and need explicit rescaling below.
        X, y, X_offset, y_offset, X_scale = preprocessed[:5]
        sqrt_sw = preprocessed[5] if len(preprocessed) >= 6 else None

        gcv_mode = _check_gcv_mode(X, self.gcv_mode)

        if gcv_mode == "eigen":
            decompose = self._eigen_decompose_gram
            solve = self._solve_eigen_gram
        elif gcv_mode == "svd":
            if sparse.issparse(X):
                decompose = self._eigen_decompose_covariance
                solve = self._solve_eigen_covariance
            else:
                decompose = self._svd_decompose_design_matrix
                solve = self._solve_svd_design_matrix

        n_samples = X.shape[0]

        if sample_weight is not None and sqrt_sw is None:
            rescaled = _rescale_data(X, y, sample_weight)
            X, y = rescaled[:2]
            if len(rescaled) >= 3:
                sqrt_sw = rescaled[2]
            else:
                sqrt_sw = np.sqrt(sample_weight)
        elif sqrt_sw is None:
            sqrt_sw = np.ones(n_samples, dtype=X.dtype)

        X_mean, *decomposition = decompose(X, y, sqrt_sw)

        if self.scoring not in ["pearson_r", "explained_variance"]:
            raise ValueError(
                "RidgeCVFast scoring requires one of ['pearson_r', 'explained_variance']"
            )

        n_y = 1 if len(y.shape) == 1 else y.shape[1]
        n_alphas = 1 if np.ndim(self.alphas) == 0 else len(self.alphas)

        if self.store_cv_values:
            self.cv_values_ = np.empty((n_samples * n_y, n_alphas), dtype=X.dtype)

        best_coef, best_score, best_alpha = None, None, None

        for i, alpha in enumerate(np.atleast_1d(self.alphas)):
            G_inverse_diag, c = solve(float(alpha), y, sqrt_sw, X_mean, *decomposition)
            predictions = y - (c / G_inverse_diag)
            if sample_weight is not None:
                if predictions.ndim > 1:
                    predictions = predictions / sqrt_sw[:, None]
                else:
                    predictions = predictions / sqrt_sw
            predictions = predictions + y_offset
            if self.store_cv_values:
                self.cv_values_[:, i] = predictions.ravel()

            if self.alpha_per_target:
                if self.scoring == "pearson_r":
                    alpha_score = _pearson_r_score(unscaled_y, predictions)
                elif self.scoring == "explained_variance":
                    alpha_score = explained_variance_score(
                        unscaled_y, predictions, multioutput="raw_values"
                    )
            else:
                if self.scoring == "pearson_r":
                    alpha_score = _pearson_r_score(unscaled_y, predictions).mean()
                elif self.scoring == "explained_variance":
                    alpha_score = explained_variance_score(
                        unscaled_y, predictions, multioutput="uniform_average"
                    )

            # Keep track of the best model
            if best_score is None:
                if self.alpha_per_target and n_y > 1:
                    best_coef = c
                    best_score = np.atleast_1d(alpha_score)
                    best_alpha = np.full(n_y, alpha)
                else:
                    best_coef = c
                    best_score = alpha_score
                    best_alpha = alpha
            else:
                if self.alpha_per_target and n_y > 1:
                    to_update = alpha_score > best_score
                    best_coef[:, to_update] = c[:, to_update]
                    best_score[to_update] = alpha_score[to_update]
                    best_alpha[to_update] = alpha
                elif alpha_score > best_score:
                    best_coef, best_score, best_alpha = c, alpha_score, alpha

        self.alpha_ = best_alpha
        self.best_score_ = best_score
        self.dual_coef_ = best_coef
        dual_T = self.dual_coef_.T if self.dual_coef_.ndim > 1 else self.dual_coef_
        self.coef_ = safe_sparse_dot(dual_T, X)
        if y.ndim == 1 or y.shape[1] == 1:
            self.coef_ = self.coef_.ravel()

        if sparse.issparse(X):
            X_offset = X_mean * X_scale
        else:
            X_offset += X_mean * X_scale
        self._set_intercept(X_offset, y_offset, X_scale)

        if self.store_cv_values:
            if len(y.shape) == 1:
                cv_values_shape = n_samples, n_alphas
            else:
                cv_values_shape = n_samples, n_y, n_alphas
            self.cv_values_ = self.cv_values_.reshape(cv_values_shape)

        return self


class _BaseRidgeCVFast(_BaseRidgeCV):
    def fit(self, X, y, sample_weight=None):
        cv = self.cv
        if cv is None:
            estimator = _RidgeGCVFast(
                self.alphas,
                fit_intercept=self.fit_intercept,
                scoring=self.scoring,
                gcv_mode=self.gcv_mode,
                store_cv_values=self.store_cv_values,
                is_clf=is_classifier(self),
                alpha_per_target=self.alpha_per_target,
            )
            estimator.fit(X, y, sample_weight=sample_weight)
            self.alpha_ = estimator.alpha_
            self.best_score_ = estimator.best_score_
            if self.store_cv_values:
                self.__dict__["cv_results_"] = estimator.cv_values_
        else:
            if self.store_cv_values:
                raise ValueError(
                    "cv is not None and store_cv_values=True are incompatible"
                )
            if self.alpha_per_target:
                raise ValueError(
                    "cv is not None and alpha_per_target=True are incompatible"
                )

            parameters = {"alpha": self.alphas}
            solver = "sparse_cg" if sparse.issparse(X) else "auto"
            model = RidgeClassifier if is_classifier(self) else Ridge
            gs = GridSearchCV(
                model(
                    fit_intercept=self.fit_intercept,
                    solver=solver,
                ),
                parameters,
                cv=cv,
                scoring=self.scoring,
            )
            gs.fit(X, y, sample_weight=sample_weight)
            estimator = gs.best_estimator_
            self.alpha_ = gs.best_estimator_.alpha
            self.best_score_ = gs.best_score_

        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_
        self.n_features_in_ = estimator.n_features_in_
        if hasattr(estimator, "feature_names_in_"):
            self.feature_names_in_ = estimator.feature_names_in_

        return self


class RidgeCVFast(MultiOutputMixin, RegressorMixin, _BaseRidgeCVFast):
    """Fast Ridge regression with built-in cross-validation.

    This is a drop-in replacement for sklearn's RidgeCV that uses analytical
    Leave-One-Out cross-validation via SVD decomposition. Key features:

    1. Much faster than k-fold CV (no refitting required per fold)
    2. Supports alpha_per_target=True for per-output optimal alpha selection
    3. Fits all outputs (voxels) simultaneously

    Parameters
    ----------
    alphas : array-like of shape (n_alphas,), default=(0.1, 1.0, 10.0)
        Array of alpha values to try.
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    scoring : {'pearson_r', 'explained_variance'}
        Scoring metric. Must be one of these two options.
    gcv_mode : {'auto', 'svd', 'eigen'}, default=None
        Flag indicating which strategy to use for LOO-CV.
    store_cv_values : bool, default=False
        Flag indicating if CV values should be stored.
    alpha_per_target : bool, default=False
        If True, optimize alpha for each target (voxel) independently.
    cv : None
        Must be None to use the fast GCV mode. If not None, falls back to
        GridSearchCV which is slower.

    Attributes
    ----------
    alpha_ : float or ndarray of shape (n_targets,)
        Estimated best alpha. If alpha_per_target=True, this is an array.
    best_score_ : float or ndarray of shape (n_targets,)
        Score of the best alpha(s).
    coef_ : ndarray of shape (n_targets, n_features)
        Weight vectors.
    intercept_ : float or ndarray of shape (n_targets,)
        Intercept term(s).

    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> Y = np.random.randn(100, 1000)  # 1000 voxels
    >>> model = RidgeCVFast(
    ...     alphas=np.logspace(-1, 6, 20),
    ...     scoring='pearson_r',
    ...     alpha_per_target=True
    ... )
    >>> model.fit(X, Y)  # Fits all 1000 voxels at once with per-voxel alpha
    """

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        scoring=None,
        cv=None,
        gcv_mode=None,
        store_cv_values=False,
        alpha_per_target=False,
    ):
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.cv = cv
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
        self.alpha_per_target = alpha_per_target
