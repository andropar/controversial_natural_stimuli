from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .common import (
    StandardizationStats,
    _as_2d_targets,
    _check_2d_array,
    _fit_stats,
    _identity_stats,
    _symmetric_inverse_spd,
)

@dataclass(frozen=True)
class EncodingRidgeAlphaSelection:
    """Per-target alpha choices for a reusable readout cache."""

    alpha_values: tuple[float, ...]
    best_alpha_idx: np.ndarray
    target_stats: StandardizationStats
    selected_coefficients: np.ndarray | None = None

    @property
    def selected_alphas(self) -> np.ndarray:
        values = np.asarray(self.alpha_values, dtype=np.float64)
        return values[self.best_alpha_idx]

    def transform_targets(self, y: np.ndarray) -> np.ndarray:
        y_2d, _ = _as_2d_targets(np.asarray(y, dtype=np.float64))
        return self.target_stats.transform(y_2d)

    def inverse_transform_predictions(self, y: np.ndarray) -> np.ndarray:
        return self.target_stats.inverse_transform(np.asarray(y, dtype=np.float64))


@dataclass
class _IndependentReadoutCache:
    alphas: tuple[float, ...]
    backend: str
    eigvals: np.ndarray
    train_projected: np.ndarray
    val_projected: np.ndarray
    eval_projected: dict[str, np.ndarray]

    @classmethod
    def build(
        cls,
        X_train: np.ndarray,
        X_val: np.ndarray,
        eval_sets: Mapping[str, np.ndarray],
        alphas: Sequence[float],
        *,
        backend: str,
    ) -> "_IndependentReadoutCache":
        if backend not in {"auto", "kernel", "feature"}:
            raise ValueError(f"Unsupported backend: {backend}")
        X_train64 = np.asarray(X_train, dtype=np.float64)
        resolved = backend
        if resolved == "auto":
            resolved = "kernel" if X_train64.shape[0] <= X_train64.shape[1] else "feature"
        if resolved == "kernel":
            k_train = X_train64 @ X_train64.T
            eigvals, eigvecs = np.linalg.eigh(k_train)
            eigvals = np.maximum(eigvals, 0.0)
            train_projected = eigvecs
            val_projected = np.asarray(X_val, dtype=np.float64) @ X_train64.T @ eigvecs
            eval_projected = {
                key: np.asarray(X_eval, dtype=np.float64) @ X_train64.T @ eigvecs
                for key, X_eval in eval_sets.items()
            }
        else:
            covariance = X_train64.T @ X_train64
            eigvals, eigvecs = np.linalg.eigh(covariance)
            eigvals = np.maximum(eigvals, 0.0)
            train_projected = X_train64 @ eigvecs
            val_projected = np.asarray(X_val, dtype=np.float64) @ eigvecs
            eval_projected = {
                key: np.asarray(X_eval, dtype=np.float64) @ eigvecs
                for key, X_eval in eval_sets.items()
            }
        return cls(
            alphas=tuple(float(alpha) for alpha in alphas),
            backend=resolved,
            eigvals=np.asarray(eigvals, dtype=np.float64),
            train_projected=np.asarray(train_projected, dtype=np.float64),
            val_projected=np.asarray(val_projected, dtype=np.float64),
            eval_projected={
                key: np.asarray(value, dtype=np.float64)
                for key, value in eval_projected.items()
            },
        )

    def project_targets(self, y_train: np.ndarray) -> np.ndarray:
        return self.train_projected.T @ np.asarray(y_train, dtype=np.float64)

    def coefficients_from_projected_targets(
        self,
        alpha: float,
        projected_y: np.ndarray,
    ) -> np.ndarray:
        denom = self.eigvals + float(alpha)
        return np.asarray(projected_y, dtype=np.float64) / denom[:, None]

    def validation_prediction_from_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        return self.val_projected @ np.asarray(coefficients, dtype=np.float64)

    def eval_prediction_from_coefficients(
        self,
        eval_key: str,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        return self.eval_projected[eval_key] @ np.asarray(coefficients, dtype=np.float64)


@dataclass
class _AugmentedLooCache:
    alphas: tuple[float, ...]
    backend: str
    alpha_selector: _IndependentReadoutCache
    base_right: np.ndarray
    base_left: dict[float, dict[str, np.ndarray]]
    s_inv: dict[float, dict[str, np.ndarray]]

    @property
    def eval_keys(self) -> tuple[str, ...]:
        return tuple(next(iter(self.base_left.values())))

    def base_prediction(
        self,
        alpha: float,
        eval_key: str,
        y_base_fit: np.ndarray,
    ) -> np.ndarray:
        return self.base_left[float(alpha)][eval_key] @ (
            self.base_right @ np.asarray(y_base_fit, dtype=np.float64)
        )

    def eval_prediction(
        self,
        alpha: float,
        eval_key: str,
        y_base_fit: np.ndarray,
        eval_y_fit: np.ndarray,
    ) -> np.ndarray:
        alpha = float(alpha)
        eval_y = np.asarray(eval_y_fit, dtype=np.float64)
        base_pred = self.base_prediction(alpha, eval_key, y_base_fit)
        s_inv = self.s_inv[alpha][eval_key]
        return eval_y + (s_inv @ (base_pred - eval_y)) / np.diag(s_inv)[:, None]

    def materialized_eval_ops(
        self,
        alpha: float,
        eval_key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = float(alpha)
        s_inv = self.s_inv[alpha][eval_key]
        diag = np.diag(s_inv)
        base_op = (s_inv @ (self.base_left[alpha][eval_key] @ self.base_right)) / diag[
            :,
            None,
        ]
        eval_op = -s_inv / diag[:, None]
        idx = np.arange(eval_op.shape[0])
        eval_op[idx, idx] = 0.0
        return np.asarray(base_op, dtype=np.float64), np.asarray(eval_op, dtype=np.float64)


def _build_feature_eval_augmented_loo_cache(
    *,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_base: np.ndarray,
    eval_sets: Mapping[str, np.ndarray],
    alphas: Sequence[float],
    backend: str,
) -> _AugmentedLooCache:
    alpha_selector = _IndependentReadoutCache.build(
        X_train,
        X_val,
        {},
        alphas,
        backend=backend,
    )
    X_base64 = np.asarray(X_base, dtype=np.float64)
    covariance = X_base64.T @ X_base64
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, 0.0)
    base_projected = X_base64 @ eigvecs
    eval_projected = {
        key: np.asarray(X_eval, dtype=np.float64) @ eigvecs
        for key, X_eval in eval_sets.items()
    }
    base_right = np.asarray(base_projected.T, dtype=np.float64)
    base_left: dict[float, dict[str, np.ndarray]] = {}
    s_inv: dict[float, dict[str, np.ndarray]] = {}
    for alpha in alphas:
        alpha = float(alpha)
        denom = eigvals + alpha
        left_by_key: dict[str, np.ndarray] = {}
        inv_by_key: dict[str, np.ndarray] = {}
        for key, projected in eval_projected.items():
            leverage_kernel = (projected / denom[None, :]) @ projected.T
            system = np.eye(projected.shape[0], dtype=np.float64) + leverage_kernel
            system = 0.5 * (system + system.T)
            left_by_key[key] = np.asarray(projected / denom[None, :], dtype=np.float64)
            inv_by_key[key] = _symmetric_inverse_spd(system)
        base_left[alpha] = left_by_key
        s_inv[alpha] = inv_by_key
    return _AugmentedLooCache(
        alphas=tuple(float(alpha) for alpha in alphas),
        backend="feature",
        alpha_selector=alpha_selector,
        base_right=base_right,
        base_left=base_left,
        s_inv=s_inv,
    )


def _build_kernel_eval_augmented_loo_cache(
    *,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_base: np.ndarray,
    eval_sets: Mapping[str, np.ndarray],
    alphas: Sequence[float],
    backend: str,
) -> _AugmentedLooCache:
    alpha_selector = _IndependentReadoutCache.build(
        X_train,
        X_val,
        {},
        alphas,
        backend=backend,
    )
    X_base64 = np.asarray(X_base, dtype=np.float64)
    k_base = X_base64 @ X_base64.T
    eigvals, eigvecs = np.linalg.eigh(k_base)
    eigvals = np.maximum(eigvals, 0.0)
    base_left: dict[float, dict[str, np.ndarray]] = {
        float(alpha): {} for alpha in alphas
    }
    s_inv: dict[float, dict[str, np.ndarray]] = {float(alpha): {} for alpha in alphas}
    base_right = np.eye(X_base64.shape[0], dtype=np.float64)
    for key, X_eval in eval_sets.items():
        X_eval64 = np.asarray(X_eval, dtype=np.float64)
        k_be = X_base64 @ X_eval64.T
        k_ee = X_eval64 @ X_eval64.T
        qtu = eigvecs.T @ k_be
        eye_eval = np.eye(k_ee.shape[0], dtype=np.float64)
        for alpha in alphas:
            alpha = float(alpha)
            denom = eigvals + alpha
            a_inv_u = eigvecs @ (qtu / denom[:, None])
            schur = k_ee + alpha * eye_eval - k_be.T @ a_inv_u
            schur = 0.5 * (schur + schur.T)
            base_left[alpha][key] = np.asarray(a_inv_u.T, dtype=np.float64)
            s_inv[alpha][key] = np.asarray(alpha * _symmetric_inverse_spd(schur), dtype=np.float64)
    return _AugmentedLooCache(
        alphas=tuple(float(alpha) for alpha in alphas),
        backend="kernel",
        alpha_selector=alpha_selector,
        base_right=base_right,
        base_left=base_left,
        s_inv=s_inv,
    )


def _build_kernel_eval_augmented_nested_loo_cache(
    *,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_base: np.ndarray,
    eval_sets: Mapping[str, np.ndarray],
    alphas: Sequence[float],
    backend: str,
) -> _AugmentedLooCache:
    return _build_kernel_eval_augmented_loo_cache(
        X_train=X_train,
        X_val=X_val,
        X_base=X_base,
        eval_sets=eval_sets,
        alphas=alphas,
        backend=backend,
    )


class EncodingRidgeReadoutCache:
    """Reusable ridge readout cache for many target matrices.

    The cache stores feature-only eigendecompositions/projections and exposes
    target-time methods for alpha selection and eval prediction.  It is meant for
    teacher-student and stimulus-selection workloads where the same feature
    design is reused across many noisy target draws.
    """

    def __init__(
        self,
        *,
        spec: EncodingRidgeSpec,
        mode: str,
        feature_stats: StandardizationStats,
        low_level: _IndependentReadoutCache | _AugmentedLooCache,
    ):
        self.spec = spec
        self.mode = mode
        self.feature_stats = feature_stats
        self.low_level = low_level
        self.backend = low_level.backend

    @classmethod
    def from_features(
        cls,
        *,
        spec: EncodingRidgeSpec,
        X_train: np.ndarray,
        X_val: np.ndarray,
        eval_sets: Mapping[str, np.ndarray],
        X_base: np.ndarray | None = None,
        mode: str = "independent",
        backend: str = "auto",
    ) -> "EncodingRidgeReadoutCache":
        if mode not in {"independent", "eval_augmented_loo", "eval_augmented_nested_loo"}:
            raise ValueError(
                "mode must be 'independent', 'eval_augmented_loo', or "
                "'eval_augmented_nested_loo'"
            )
        X_train = _check_2d_array("X_train", np.asarray(X_train, dtype=np.float64))
        X_val = _check_2d_array("X_val", np.asarray(X_val, dtype=np.float64))
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError("X_train and X_val must have the same number of features")
        for key, X_eval in eval_sets.items():
            X_eval = _check_2d_array(f"eval_sets[{key!r}]", np.asarray(X_eval))
            if X_eval.shape[1] != X_train.shape[1]:
                raise ValueError(
                    f"eval set {key!r} has {X_eval.shape[1]} features, expected {X_train.shape[1]}"
                )
        if mode in {"eval_augmented_loo", "eval_augmented_nested_loo"}:
            if X_base is None:
                raise ValueError(f"X_base is required for mode={mode!r}")
            X_base = _check_2d_array("X_base", np.asarray(X_base, dtype=np.float64))
            if X_base.shape[1] != X_train.shape[1]:
                raise ValueError("X_base and X_train must have the same number of features")

        scale_by_sqrt = spec.feature_scale_mode == "zscore_sqrt_features"
        feature_stats = _fit_stats(
            X_train,
            sample_weight=None,
            center=spec.standardize_x or spec.fit_intercept,
            standardize=spec.standardize_x,
            eps=spec.x_scale_eps,
            scale_by_sqrt_features=scale_by_sqrt,
        )
        X_train_proc = feature_stats.transform(X_train)
        X_val_proc = feature_stats.transform(X_val)
        eval_proc = {
            key: feature_stats.transform(np.asarray(X_eval, dtype=np.float64))
            for key, X_eval in eval_sets.items()
        }
        if mode == "independent":
            low_level = _IndependentReadoutCache.build(
                X_train_proc,
                X_val_proc,
                eval_proc,
                spec.alphas,
                backend=backend,
            )
        else:
            X_base_proc = feature_stats.transform(np.asarray(X_base, dtype=np.float64))
            resolved = backend
            if resolved == "auto":
                resolved = (
                    "kernel"
                    if X_base_proc.shape[0] <= X_base_proc.shape[1]
                    else "feature"
                )
            if resolved == "feature":
                low_level = _build_feature_eval_augmented_loo_cache(
                    X_train=X_train_proc,
                    X_val=X_val_proc,
                    X_base=X_base_proc,
                    eval_sets=eval_proc,
                    alphas=spec.alphas,
                    backend=backend,
                )
            elif resolved == "kernel":
                if mode == "eval_augmented_nested_loo":
                    low_level = _build_kernel_eval_augmented_nested_loo_cache(
                        X_train=X_train_proc,
                        X_val=X_val_proc,
                        X_base=X_base_proc,
                        eval_sets=eval_proc,
                        alphas=spec.alphas,
                        backend=backend,
                    )
                else:
                    low_level = _build_kernel_eval_augmented_loo_cache(
                        X_train=X_train_proc,
                        X_val=X_val_proc,
                        X_base=X_base_proc,
                        eval_sets=eval_proc,
                        alphas=spec.alphas,
                        backend=backend,
                    )
            else:
                raise ValueError(f"Unsupported backend: {backend}")
        return cls(
            spec=spec,
            mode=mode,
            feature_stats=feature_stats,
            low_level=low_level,
        )

    @property
    def alpha_values(self) -> tuple[float, ...]:
        return self.spec.alphas

    @property
    def eval_keys(self) -> tuple[str, ...]:
        if isinstance(self.low_level, _IndependentReadoutCache):
            return tuple(self.low_level.eval_projected)
        return self.low_level.eval_keys

    def _target_stats(self, y_train: np.ndarray) -> StandardizationStats:
        y_train = _check_2d_array("y_train", np.asarray(y_train, dtype=np.float64))
        return _fit_stats(
            y_train,
            sample_weight=None,
            center=self.spec.standardize_y or self.spec.fit_intercept,
            standardize=self.spec.standardize_y,
            eps=self.spec.y_scale_eps,
        )

    def select_targetwise_alphas(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> EncodingRidgeAlphaSelection:
        y_train_2d, _ = _as_2d_targets(np.asarray(y_train, dtype=np.float64))
        y_val_2d, _ = _as_2d_targets(np.asarray(y_val, dtype=np.float64))
        if y_train_2d.shape[1] != y_val_2d.shape[1]:
            raise ValueError("y_train and y_val must have the same number of targets")
        target_stats = self._target_stats(y_train_2d)
        y_train_proc = target_stats.transform(y_train_2d)
        y_val_proc = target_stats.transform(y_val_2d)

        selector = (
            self.low_level
            if isinstance(self.low_level, _IndependentReadoutCache)
            else self.low_level.alpha_selector
        )
        projected_y = selector.project_targets(y_train_proc)
        n_targets = y_train_proc.shape[1]
        scores = np.full((len(self.alpha_values), n_targets), -np.inf, dtype=np.float64)
        if self.spec.scoring == "pearson_r":
            y_val_centered = y_val_proc - y_val_proc.mean(axis=0, keepdims=True)
            y_val_norm_sq = np.sum(y_val_centered * y_val_centered, axis=0)
        else:
            y_val_centered = None
            y_val_norm_sq = None

        for alpha_idx, alpha in enumerate(self.alpha_values):
            coeff = selector.coefficients_from_projected_targets(alpha, projected_y)
            pred_val = selector.validation_prediction_from_coefficients(coeff)
            if self.spec.scoring == "pearson_r":
                pred_centered = pred_val - pred_val.mean(axis=0, keepdims=True)
                denom = np.sqrt(
                    np.sum(pred_centered * pred_centered, axis=0) * y_val_norm_sq
                )
                ok = denom > 0
                scores[alpha_idx, ok] = (
                    np.sum(pred_centered[:, ok] * y_val_centered[:, ok], axis=0)
                    / denom[ok]
                )
            else:
                scores[alpha_idx] = -np.mean((y_val_proc - pred_val) ** 2, axis=0)
        scores = np.nan_to_num(scores, nan=-np.inf)
        best_alpha_idx = np.argmax(scores, axis=0).astype(np.int32)
        selected_alpha = np.asarray(self.alpha_values, dtype=np.float64)[best_alpha_idx]
        selected_coefficients = projected_y / (
            selector.eigvals[:, None] + selected_alpha[None, :]
        )
        return EncodingRidgeAlphaSelection(
            alpha_values=self.alpha_values,
            best_alpha_idx=best_alpha_idx,
            target_stats=target_stats,
            selected_coefficients=np.asarray(selected_coefficients, dtype=np.float64),
        )

    def predict_eval(
        self,
        *,
        eval_key: str,
        alpha_selection: EncodingRidgeAlphaSelection,
        y_train: np.ndarray | None = None,
        y_base_fit: np.ndarray | None = None,
        eval_y_fit: np.ndarray | None = None,
        output_space: str = "target",
    ) -> np.ndarray:
        if eval_key not in self.eval_keys:
            raise ValueError(f"Unknown eval_key={eval_key!r}; available={self.eval_keys}")
        if output_space not in {"target", "model", "standardized"}:
            raise ValueError("output_space must be 'target', 'model', or 'standardized'")
        if self.mode == "eval_augmented_nested_loo":
            raise ValueError(
                "Use predict_eval_augmented_nested_loo for mode='eval_augmented_nested_loo'"
            )

        if isinstance(self.low_level, _IndependentReadoutCache):
            if y_train is None:
                raise ValueError("y_train is required for independent readout prediction")
            if alpha_selection.selected_coefficients is None:
                y_train_proc = alpha_selection.transform_targets(y_train)
                projected_y = self.low_level.project_targets(y_train_proc)
                selected_alpha = alpha_selection.selected_alphas
                coeff = projected_y / (
                    self.low_level.eigvals[:, None] + selected_alpha[None, :]
                )
            else:
                coeff = alpha_selection.selected_coefficients
            pred_proc = self.low_level.eval_prediction_from_coefficients(eval_key, coeff)
        else:
            if y_base_fit is None or eval_y_fit is None:
                raise ValueError(
                    "y_base_fit and eval_y_fit are required for eval_augmented_loo prediction"
                )
            y_base_proc = alpha_selection.transform_targets(y_base_fit)
            eval_y_proc = alpha_selection.transform_targets(eval_y_fit)
            n_eval = self.low_level.base_left[alpha_selection.alpha_values[0]][
                eval_key
            ].shape[0]
            n_targets = y_base_proc.shape[1]
            pred_proc = np.empty((n_eval, n_targets), dtype=np.float64)
            for alpha_idx, alpha in enumerate(alpha_selection.alpha_values):
                cols = np.flatnonzero(alpha_selection.best_alpha_idx == alpha_idx)
                if cols.size == 0:
                    continue
                pred_proc[:, cols] = self.low_level.eval_prediction(
                    alpha,
                    eval_key,
                    y_base_proc[:, cols],
                    eval_y_proc[:, cols],
                )

        if output_space == "target":
            return alpha_selection.inverse_transform_predictions(pred_proc).astype(np.float32)
        return pred_proc.astype(np.float32)

    def predict_eval_augmented_nested_loo(
        self,
        *,
        eval_key: str,
        y_base_fit: np.ndarray,
        eval_y_fit: np.ndarray,
        target_stats: StandardizationStats | None = None,
        y_train_for_stats: np.ndarray | None = None,
        output_space: str = "target",
        return_alpha_indices: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict eval-set targets with strict nested eval-augmented LOO.

        This matches ``predict_eval_augmented_nested_loo`` in
        ``cstims.evaluation.teacher_student``.  Alpha is selected independently
        for each outer eval row and target by scoring inner eval-LOO predictions.
        """
        if self.mode != "eval_augmented_nested_loo":
            raise ValueError(
                "predict_eval_augmented_nested_loo requires "
                "mode='eval_augmented_nested_loo'"
            )
        if not isinstance(self.low_level, _AugmentedLooCache):
            raise RuntimeError("Unexpected cache type for nested eval-augmented LOO")
        if eval_key not in self.eval_keys:
            raise ValueError(f"Unknown eval_key={eval_key!r}; available={self.eval_keys}")
        if output_space not in {"target", "model", "standardized"}:
            raise ValueError("output_space must be 'target', 'model', or 'standardized'")

        eval_y_raw, _ = _as_2d_targets(np.asarray(eval_y_fit, dtype=np.float64))
        base_y_raw, _ = _as_2d_targets(np.asarray(y_base_fit, dtype=np.float64))
        if eval_y_raw.shape[1] != base_y_raw.shape[1]:
            raise ValueError("y_base_fit and eval_y_fit must have the same target count")
        n_eval_images, n_targets = eval_y_raw.shape
        if n_eval_images < 3:
            raise ValueError("Nested eval LOO alpha selection requires at least 3 eval images")

        if target_stats is None:
            if y_train_for_stats is not None:
                y_stats_train, _ = _as_2d_targets(
                    np.asarray(y_train_for_stats, dtype=np.float64)
                )
                target_stats = self._target_stats(y_stats_train)
            elif self.spec.standardize_y:
                raise ValueError(
                    "target_stats or y_train_for_stats is required when standardize_y=True"
                )
            else:
                target_stats = _identity_stats(n_targets)

        eval_y = target_stats.transform(eval_y_raw)
        base_y = target_stats.transform(base_y_raw)

        pred_by_alpha = []
        score_by_alpha = []
        all_idx = np.arange(n_eval_images)
        for alpha in self.alpha_values:
            alpha = float(alpha)
            base_pred = self.low_level.base_prediction(alpha, eval_key, base_y)
            s_inv = self.low_level.s_inv[alpha][eval_key]

            residual = s_inv @ (eval_y - base_pred)
            diag = np.diag(s_inv).astype(np.float64, copy=False)
            pred_by_alpha.append(eval_y - residual / diag[:, None])

            scores_for_outer = np.empty((n_eval_images, n_targets), dtype=np.float64)
            for outer_idx in range(n_eval_images):
                inner_idx = all_idx[all_idx != outer_idx]
                a = diag[outer_idx]
                b = s_inv[outer_idx, inner_idx]
                c = diag[inner_idx]
                det = a * c - b * b
                correction = (
                    -b[:, None] * residual[outer_idx : outer_idx + 1, :]
                    + a * residual[inner_idx, :]
                ) / det[:, None]
                inner_pred = eval_y[inner_idx] - correction
                scores_for_outer[outer_idx] = _score_columns(
                    inner_pred,
                    eval_y[inner_idx],
                    self.spec.scoring,
                )
            score_by_alpha.append(scores_for_outer)

        score_stack = np.nan_to_num(np.stack(score_by_alpha, axis=0), nan=-np.inf)
        best_alpha_idx = np.argmax(score_stack, axis=0).astype(np.int32)
        pred_stack = np.stack(pred_by_alpha, axis=0)
        row_idx = np.arange(n_eval_images)[:, None]
        col_idx = np.arange(n_targets)[None, :]
        pred_proc = pred_stack[best_alpha_idx, row_idx, col_idx]

        if output_space == "target":
            pred = target_stats.inverse_transform(pred_proc).astype(np.float32)
        else:
            pred = pred_proc.astype(np.float32)
        if return_alpha_indices:
            return pred, best_alpha_idx
        return pred


