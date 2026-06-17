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
