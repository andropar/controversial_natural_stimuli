"""Shared ridge-regression operators for evaluation simulations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np


def standardize_from_train(
    train: np.ndarray,
    *others: np.ndarray,
    scale_by_sqrt_features: bool = False,
) -> tuple[np.ndarray, ...]:
    """Z-score arrays using statistics from the training array."""
    train = np.asarray(train, dtype=np.float32)
    others = tuple(np.asarray(arr, dtype=np.float32) for arr in others)
    mean = train.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    if scale_by_sqrt_features:
        scale *= np.float32(math.sqrt(train.shape[1]))
    out = [(train - mean) / scale]
    out.extend((arr - mean) / scale for arr in others)
    return tuple(np.asarray(arr, dtype=np.float32) for arr in out)


def ridge_ops_for_eval_sets(
    x_train: np.ndarray,
    x_val: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    alphas: list[float],
) -> dict[float, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Kernel ridge prediction operators, reusing one eigensolve across alphas."""
    x_train64 = np.asarray(x_train, dtype=np.float64)
    k_train = x_train64 @ x_train64.T
    eigvals, eigvecs = np.linalg.eigh(k_train)
    eigvals = np.maximum(eigvals, 0.0)

    k_val_u = (np.asarray(x_val, dtype=np.float64) @ x_train64.T) @ eigvecs
    eval_u = {
        key: (np.asarray(x_eval, dtype=np.float64) @ x_train64.T) @ eigvecs
        for key, x_eval in eval_sets.items()
    }
    out: dict[float, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for alpha in alphas:
        denom = eigvals + float(alpha)
        val_op = (k_val_u / denom) @ eigvecs.T
        eval_ops = {
            key: np.asarray((mat / denom) @ eigvecs.T, dtype=np.float32)
            for key, mat in eval_u.items()
        }
        out[float(alpha)] = (np.asarray(val_op, dtype=np.float32), eval_ops)
    return out


@dataclass
class IndependentRidgeOps:
    """Alpha-independent prediction factors for independent ridge refits.

    The object exposes predictions directly instead of materializing one
    ``n_eval x n_train`` operator per alpha/eval-set.  For larger refit pools
    this avoids the kernel-ridge cubic scaling and the largest temporary
    operator matrices.
    """

    alphas: tuple[float, ...]
    backend: str
    eigvals: np.ndarray
    train_projected: np.ndarray
    val_projected: np.ndarray
    eval_projected: dict[str, np.ndarray]
    eval_projected_all: np.ndarray | None = None
    eval_lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.eval_projected_all is None and self.eval_projected:
            keys = tuple(self.eval_projected)
            self.eval_lengths = tuple(
                int(self.eval_projected[key].shape[0]) for key in keys
            )
            self.eval_projected_all = np.concatenate(
                [self.eval_projected[key] for key in keys],
                axis=0,
            ).astype(np.float32, copy=False)
        elif self.eval_projected_all is None:
            self.eval_projected_all = np.empty(
                (0, self.train_projected.shape[1]),
                dtype=np.float32,
            )

    @property
    def alpha_values(self) -> list[float]:
        return list(self.alphas)

    @property
    def eval_keys(self) -> list[str]:
        return list(self.eval_projected)

    def project_targets(self, y_train: np.ndarray) -> np.ndarray:
        return self.train_projected.T @ np.asarray(y_train, dtype=np.float32)

    def coefficients_from_projected_targets(
        self,
        alpha: float,
        projected_y: np.ndarray,
    ) -> np.ndarray:
        denom = self.eigvals.astype(np.float32, copy=False) + np.float32(alpha)
        return np.asarray(projected_y / denom[:, None], dtype=np.float32)

    def target_coefficients(self, alpha: float, y_train: np.ndarray) -> np.ndarray:
        projected_y = self.project_targets(y_train)
        return self.coefficients_from_projected_targets(alpha, projected_y)

    def validation_prediction(self, alpha: float, y_train: np.ndarray) -> np.ndarray:
        coeff = self.target_coefficients(alpha, y_train)
        return self.validation_prediction_from_coefficients(coeff)

    def validation_prediction_from_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        return np.asarray(self.val_projected @ coefficients, dtype=np.float32)

    def eval_prediction_from_coefficients(
        self,
        eval_key: str,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(self.eval_projected[eval_key] @ coefficients, dtype=np.float32)

    def eval_prediction(
        self,
        alpha: float,
        eval_key: str,
        y_train: np.ndarray,
        *,
        y_base_fit: np.ndarray | None = None,
        eval_y_fit: np.ndarray | None = None,
    ) -> np.ndarray:
        del y_base_fit, eval_y_fit
        coeff = self.target_coefficients(alpha, y_train)
        return self.eval_prediction_from_coefficients(eval_key, coeff)


@dataclass
class EvalAugmentedLooRidgeOps:
    """Compact feature-space operators for eval-augmented LOO ridge.

    The alpha selection still uses an independent train/validation ridge split.
    Final eval predictions use the exact linear-ridge leave-one-out formula for
    fits on ``base_fit + eval_set`` without materializing ``n_eval x n_base``
    operators for every alpha and eval set.
    """

    alphas: tuple[float, ...]
    backend: str
    alpha_selector: IndependentRidgeOps
    base_eigvals: np.ndarray
    base_projected: np.ndarray
    eval_projected: dict[str, np.ndarray]
    eval_s_inv: dict[float, dict[str, np.ndarray]]
    eval_lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.eval_lengths:
            self.eval_lengths = tuple(
                int(self.eval_projected[key].shape[0]) for key in self.eval_projected
            )

    @property
    def alpha_values(self) -> list[float]:
        return list(self.alphas)

    @property
    def eval_keys(self) -> list[str]:
        return list(self.eval_projected)

    def validation_prediction(self, alpha: float, y_train: np.ndarray) -> np.ndarray:
        return self.alpha_selector.validation_prediction(alpha, y_train)

    def validation_prediction_from_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        return self.alpha_selector.validation_prediction_from_coefficients(coefficients)

    def project_targets(self, y_train: np.ndarray) -> np.ndarray:
        return self.alpha_selector.project_targets(y_train)

    def coefficients_from_projected_targets(
        self,
        alpha: float,
        projected_y: np.ndarray,
    ) -> np.ndarray:
        return self.alpha_selector.coefficients_from_projected_targets(
            alpha,
            projected_y,
        )

    def eval_prediction(
        self,
        alpha: float,
        eval_key: str,
        y_base_fit: np.ndarray,
        eval_y_fit: np.ndarray,
    ) -> np.ndarray:
        alpha = float(alpha)
        base_y = np.asarray(y_base_fit, dtype=np.float32)
        eval_y = np.asarray(eval_y_fit, dtype=np.float32)
        projected_y = self.base_projected.T @ base_y
        denom = self.base_eigvals.astype(np.float32, copy=False) + np.float32(alpha)
        coeff = projected_y / denom[:, None]
        base_pred = self.eval_projected[eval_key] @ coeff
        s_inv = self.eval_s_inv[alpha][eval_key]
        loo_denom = np.diag(s_inv).astype(np.float32, copy=False)
        return np.asarray(
            eval_y + (s_inv @ (base_pred - eval_y)) / loo_denom[:, None],
            dtype=np.float32,
        )


@dataclass
class EvalAugmentedNestedLooKernelOps:
    """Kernel-space operators for nested eval-augmented LOO ridge.

    For each alpha/eval set this stores the base-only prediction operator and
    the inverse residual operator for the eval block.  That is enough to compute
    both final outer-LOO predictions and inner two-left-out predictions for
    nested alpha selection without materializing a full augmented-system inverse.
    """

    alphas: tuple[float, ...]
    backend: str
    alpha_selector: IndependentRidgeOps
    eval_base_ops: dict[float, dict[str, np.ndarray]]
    eval_s_inv: dict[float, dict[str, np.ndarray]]
    eval_lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.eval_lengths:
            return
        first_alpha = self.alphas[0]
        self.eval_lengths = tuple(
            int(self.eval_base_ops[first_alpha][key].shape[0])
            for key in self.eval_base_ops[first_alpha]
        )

    @property
    def alpha_values(self) -> list[float]:
        return list(self.alphas)

    @property
    def eval_keys(self) -> list[str]:
        return list(self.eval_base_ops[self.alphas[0]])

    def base_prediction(
        self,
        alpha: float,
        eval_key: str,
        y_base_fit: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            self.eval_base_ops[float(alpha)][eval_key]
            @ np.asarray(y_base_fit, dtype=np.float32),
            dtype=np.float32,
        )


def _kernel_independent_ridge_ops(
    x_train: np.ndarray,
    x_val: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    alphas: Sequence[float],
) -> IndependentRidgeOps:
    x_train64 = np.asarray(x_train, dtype=np.float64)
    k_train = x_train64 @ x_train64.T
    eigvals, eigvecs = np.linalg.eigh(k_train)
    eigvals = np.maximum(eigvals, 0.0)
    train_projected = np.asarray(eigvecs, dtype=np.float32)
    val_projected = np.asarray(
        (np.asarray(x_val, dtype=np.float64) @ x_train64.T) @ eigvecs,
        dtype=np.float32,
    )
    eval_projected = {
        key: np.asarray((np.asarray(x_eval, dtype=np.float64) @ x_train64.T) @ eigvecs, dtype=np.float32)
        for key, x_eval in eval_sets.items()
    }
    return IndependentRidgeOps(
        alphas=tuple(float(alpha) for alpha in alphas),
        backend="kernel",
        eigvals=np.asarray(eigvals, dtype=np.float32),
        train_projected=train_projected,
        val_projected=val_projected,
        eval_projected=eval_projected,
    )


def _feature_independent_ridge_ops(
    x_train: np.ndarray,
    x_val: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    alphas: Sequence[float],
) -> IndependentRidgeOps:
    x_train64 = np.asarray(x_train, dtype=np.float64)
    covariance = x_train64.T @ x_train64
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, 0.0)
    train_projected = np.asarray(x_train64 @ eigvecs, dtype=np.float32)
    val_projected = np.asarray(np.asarray(x_val, dtype=np.float64) @ eigvecs, dtype=np.float32)
    eval_projected = {
        key: np.asarray(np.asarray(x_eval, dtype=np.float64) @ eigvecs, dtype=np.float32)
        for key, x_eval in eval_sets.items()
    }
    return IndependentRidgeOps(
        alphas=tuple(float(alpha) for alpha in alphas),
        backend="feature",
        eigvals=np.asarray(eigvals, dtype=np.float32),
        train_projected=train_projected,
        val_projected=val_projected,
        eval_projected=eval_projected,
    )


def build_independent_ridge_ops(
    x_train: np.ndarray,
    x_val: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    alphas: Sequence[float],
    *,
    backend: str = "auto",
) -> IndependentRidgeOps:
    """Build reusable ridge-prediction factors for independent readout refits.

    ``backend='auto'`` uses dual/kernel ridge when the refit split has fewer
    samples than feature dimensions and primal/feature-space ridge otherwise.
    The two backends are algebraically equivalent for the standardized features
    used by these simulations, but have very different runtime scaling.
    """
    if backend not in {"auto", "kernel", "feature"}:
        raise ValueError(f"Unsupported ridge backend: {backend}")
    n_train, n_features = x_train.shape
    resolved = backend
    if resolved == "auto":
        resolved = "kernel" if n_train <= n_features else "feature"
    if resolved == "kernel":
        return _kernel_independent_ridge_ops(x_train, x_val, eval_sets, alphas)
    return _feature_independent_ridge_ops(x_train, x_val, eval_sets, alphas)


def ridge_eval_augmented_loo_ops(
    x_base: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    alphas: list[float],
) -> dict[float, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Kernel ridge LOO operators for eval-set-augmented final fits."""
    x_base64 = np.asarray(x_base, dtype=np.float64)
    k_base = x_base64 @ x_base64.T
    eigvals, eigvecs = np.linalg.eigh(k_base)
    eigvals = np.maximum(eigvals, 0.0)

    out: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        float(alpha): {} for alpha in alphas
    }
    for key, x_eval in eval_sets.items():
        x_eval64 = np.asarray(x_eval, dtype=np.float64)
        k_be = x_base64 @ x_eval64.T
        k_ee = x_eval64 @ x_eval64.T
        qtu = eigvecs.T @ k_be
        eye_eval = np.eye(k_ee.shape[0], dtype=np.float64)
        for alpha in alphas:
            alpha = float(alpha)
            denom = eigvals + alpha
            a_inv_u = eigvecs @ (qtu / denom[:, None])
            schur = k_ee + alpha * eye_eval - k_be.T @ a_inv_u
            schur = 0.5 * (schur + schur.T)
            try:
                schur_inv = np.linalg.inv(schur)
            except np.linalg.LinAlgError:
                schur_inv = np.linalg.pinv(schur)

            h_eval_base = alpha * (schur_inv @ a_inv_u.T)
            h_eval_eval = eye_eval - alpha * schur_inv
            diag = np.diag(h_eval_eval).copy()
            loo_denom = 1.0 - diag
            loo_denom[np.abs(loo_denom) < 1e-8] = np.sign(
                loo_denom[np.abs(loo_denom) < 1e-8]
            ) * 1e-8
            loo_denom[loo_denom == 0] = 1e-8

            eval_part = h_eval_eval.copy()
            idx = np.arange(eval_part.shape[0])
            eval_part[idx, idx] -= diag
            base_op = h_eval_base / loo_denom[:, None]
            eval_op = eval_part / loo_denom[:, None]
            out[alpha][key] = (
                np.asarray(base_op, dtype=np.float32),
                np.asarray(eval_op, dtype=np.float32),
            )
    return out


def _kernel_eval_augmented_nested_loo_ops(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_base: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    alphas: Sequence[float],
) -> EvalAugmentedNestedLooKernelOps:
    alpha_selector = build_independent_ridge_ops(x_train, x_val, {}, alphas)

    x_base64 = np.asarray(x_base, dtype=np.float64)
    k_base = x_base64 @ x_base64.T
    eigvals, eigvecs = np.linalg.eigh(k_base)
    eigvals = np.maximum(eigvals, 0.0)

    eval_base_ops: dict[float, dict[str, np.ndarray]] = {
        float(alpha): {} for alpha in alphas
    }
    eval_s_inv: dict[float, dict[str, np.ndarray]] = {
        float(alpha): {} for alpha in alphas
    }
    for key, x_eval in eval_sets.items():
        x_eval64 = np.asarray(x_eval, dtype=np.float64)
        k_be = x_base64 @ x_eval64.T
        k_ee = x_eval64 @ x_eval64.T
        qtu = eigvecs.T @ k_be
        eye_eval = np.eye(k_ee.shape[0], dtype=np.float64)
        for alpha in alphas:
            alpha = float(alpha)
            denom = eigvals + alpha
            a_inv_u = eigvecs @ (qtu / denom[:, None])
            base_pred_op = a_inv_u.T
            schur = k_ee + alpha * eye_eval - k_be.T @ a_inv_u
            schur = 0.5 * (schur + schur.T)
            try:
                schur_inv = np.linalg.inv(schur)
            except np.linalg.LinAlgError:
                schur_inv = np.linalg.pinv(schur)
            eval_base_ops[alpha][key] = np.asarray(base_pred_op, dtype=np.float32)
            eval_s_inv[alpha][key] = np.asarray(alpha * schur_inv, dtype=np.float32)

    return EvalAugmentedNestedLooKernelOps(
        alphas=tuple(float(alpha) for alpha in alphas),
        backend="kernel",
        alpha_selector=alpha_selector,
        eval_base_ops=eval_base_ops,
        eval_s_inv=eval_s_inv,
    )


def _feature_eval_augmented_loo_ops(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_base: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    alphas: Sequence[float],
) -> EvalAugmentedLooRidgeOps:
    alpha_selector = build_independent_ridge_ops(x_train, x_val, {}, alphas)

    x_base64 = np.asarray(x_base, dtype=np.float64)
    covariance = x_base64.T @ x_base64
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, 0.0)

    base_projected = np.asarray(x_base64 @ eigvecs, dtype=np.float32)
    eval_projected = {
        key: np.asarray(np.asarray(x_eval, dtype=np.float64) @ eigvecs, dtype=np.float32)
        for key, x_eval in eval_sets.items()
    }

    eval_s_inv: dict[float, dict[str, np.ndarray]] = {}
    for alpha in alphas:
        alpha = float(alpha)
        denom = eigvals + alpha
        by_key: dict[str, np.ndarray] = {}
        for key, projected in eval_projected.items():
            projected64 = np.asarray(projected, dtype=np.float64)
            leverage_kernel = (projected64 / denom[None, :]) @ projected64.T
            system = np.eye(projected64.shape[0], dtype=np.float64) + leverage_kernel
            system = 0.5 * (system + system.T)
            try:
                s_inv = np.linalg.inv(system)
            except np.linalg.LinAlgError:
                s_inv = np.linalg.pinv(system)
            by_key[key] = np.asarray(s_inv, dtype=np.float32)
        eval_s_inv[alpha] = by_key

    return EvalAugmentedLooRidgeOps(
        alphas=tuple(float(alpha) for alpha in alphas),
        backend="feature",
        alpha_selector=alpha_selector,
        base_eigvals=np.asarray(eigvals, dtype=np.float32),
        base_projected=base_projected,
        eval_projected=eval_projected,
        eval_s_inv=eval_s_inv,
    )


def build_eval_augmented_loo_ops(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_base: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    alphas: Sequence[float],
    *,
    backend: str = "auto",
) -> EvalAugmentedLooRidgeOps | dict[float, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    """Build prediction operators for eval-set-augmented LOO ridge.

    ``backend='auto'`` mirrors independent ridge: use kernel space when the
    base refit set is smaller than the feature dimension, otherwise use the
    compact feature-space operator.
    """
    if backend not in {"auto", "kernel", "feature"}:
        raise ValueError(f"Unsupported ridge backend: {backend}")
    n_base, n_features = np.asarray(x_base).shape
    resolved = backend
    if resolved == "auto":
        resolved = "kernel" if n_base <= n_features else "feature"
    if resolved == "feature":
        return _feature_eval_augmented_loo_ops(
            x_train,
            x_val,
            x_base,
            eval_sets,
            alphas,
        )

    val_ops = ridge_ops_for_eval_sets(x_train, x_val, {}, list(alphas))
    loo_ops = ridge_eval_augmented_loo_ops(x_base, eval_sets, list(alphas))
    return {
        float(alpha): (val_ops[float(alpha)][0], loo_ops[float(alpha)])
        for alpha in alphas
    }


def build_eval_augmented_nested_loo_ops(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_base: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    alphas: Sequence[float],
    *,
    backend: str = "auto",
) -> EvalAugmentedLooRidgeOps | EvalAugmentedNestedLooKernelOps:
    """Build operators for strict nested eval-set-augmented LOO ridge."""
    if backend not in {"auto", "kernel", "feature"}:
        raise ValueError(f"Unsupported ridge backend: {backend}")
    n_base, n_features = np.asarray(x_base).shape
    resolved = backend
    if resolved == "auto":
        resolved = "kernel" if n_base <= n_features else "feature"
    if resolved == "feature":
        return _feature_eval_augmented_loo_ops(
            x_train,
            x_val,
            x_base,
            eval_sets,
            alphas,
        )
    return _kernel_eval_augmented_nested_loo_ops(
        x_train,
        x_val,
        x_base,
        eval_sets,
        alphas,
    )
