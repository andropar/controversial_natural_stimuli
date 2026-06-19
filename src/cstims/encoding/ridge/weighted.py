from __future__ import annotations

import numpy as np

from .common import ArrayLike, _as_2d_targets, _check_2d_array, _check_sample_weight

class WeightedEncodingRidgeRefitCache:
    """Weighted ridge refit cache for held-out and target-LOSO predictions.

    This covers the weighted target-adaptation path where the fit design has
    sample weights and predictions are needed for target samples that may be
    included in the weighted fit.  Inputs are expected to already be in the
    desired feature/target space; the cache performs the weighted centering
    required by an unpenalized intercept.
    """

    def __init__(
        self,
        *,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        sample_weight: np.ndarray,
    ):
        X = _check_2d_array("X_fit", np.asarray(X_fit, dtype=np.float64))
        y, _ = _as_2d_targets(np.asarray(y_fit, dtype=np.float64))
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_fit and y_fit must have the same rows: {X.shape[0]} vs {y.shape[0]}"
            )
        w = _check_sample_weight(sample_weight, X.shape[0])
        if w is None:
            raise RuntimeError("sample_weight validation unexpectedly returned None")
        sqrt_w = np.sqrt(w)
        weight_total = float(w.sum())

        x_bar = (X * w[:, None]).sum(axis=0) / weight_total
        y_bar = (y * w[:, None]).sum(axis=0) / weight_total
        Xc = X - x_bar[None, :]
        Yc = y - y_bar[None, :]
        Xw = Xc * sqrt_w[:, None]
        Yw = Yc * sqrt_w[:, None]
        kernel = Xw @ Xw.T
        kernel = 0.5 * (kernel + kernel.T)
        eigvals, eigvecs = np.linalg.eigh(kernel)

        eigvals = np.maximum(eigvals.astype(np.float64), 0.0)
        eigvecs = eigvecs.astype(np.float64, copy=False)

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.y = y
        self.sample_weight = w
        self.sqrt_weight = sqrt_w
        self.weight_total = weight_total
        self.x_bar = x_bar
        self.y_bar = y_bar
        self.y_projected = eigvecs.T @ Yw
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.feature_basis = Xw.T @ eigvecs

    @classmethod
    def fit(
        cls,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "WeightedEncodingRidgeRefitCache":
        return cls(X_fit=X_fit, y_fit=y_fit, sample_weight=sample_weight)

    def predict_heldout(
        self,
        X_eval: np.ndarray,
        alphas: ArrayLike,
    ) -> np.ndarray:
        X_eval = _check_2d_array("X_eval", np.asarray(X_eval, dtype=np.float64))
        alpha_arr = self._normalize_target_alphas(alphas)
        X_eval_centered = X_eval - self.x_bar[None, :]
        eval_u = X_eval_centered @ self.feature_basis
        unique_alphas = np.unique(alpha_arr)
        if unique_alphas.size <= max(4, alpha_arr.size // 16):
            pred = np.empty((X_eval.shape[0], self.y_projected.shape[1]), dtype=np.float32)
            for alpha in unique_alphas:
                cols = np.flatnonzero(alpha_arr == alpha)
                inv_alpha = 1.0 / (self.eigvals + float(alpha))
                coef_q = self.y_projected[:, cols] * inv_alpha[:, None]
                pred[:, cols] = (
                    self.y_bar[cols][None, :] + eval_u @ coef_q
                ).astype(np.float32)
            return pred
        inv = 1.0 / (self.eigvals[:, None] + alpha_arr[None, :])
        coef_q = self.y_projected * inv
        return (self.y_bar[None, :] + eval_u @ coef_q).astype(np.float32)

    def predict_target_loso(
        self,
        *,
        target_indices: np.ndarray | None,
        X_target_eval: np.ndarray,
        alphas: ArrayLike,
        X_extra_eval: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Predict target LOSO plus optional held-out extra eval predictions."""
        alpha_arr = self._normalize_target_alphas(alphas)
        if target_indices is None:
            target_pred = self.predict_heldout(X_target_eval, alpha_arr)
            extra_pred = (
                self.predict_heldout(X_extra_eval, alpha_arr)
                if X_extra_eval is not None
                else None
            )
            return target_pred, extra_pred, np.zeros_like(target_pred, dtype=np.float32)

        target_indices = np.asarray(target_indices, dtype=int)
        if target_indices.ndim != 1:
            raise ValueError("target_indices must be a 1D array")
        if np.any(target_indices < 0) or np.any(target_indices >= self.n_samples):
            raise ValueError("target_indices contains out-of-range rows")
        if np.any(self.sample_weight[target_indices] <= 0):
            raise ValueError("target_indices require strictly positive sample weights")

        X_target_eval = _check_2d_array(
            "X_target_eval",
            np.asarray(X_target_eval, dtype=np.float64),
        )
        if X_target_eval.shape[0] != target_indices.shape[0]:
            raise ValueError(
                "X_target_eval must have one row per target index for LOSO prediction"
            )

        target_eval = X_target_eval - self.x_bar[None, :]
        target_weight = self.sample_weight[target_indices]
        target_eval_u = target_eval @ self.feature_basis
        q_target = self.eigvecs[target_indices] / self.sqrt_weight[target_indices, None]
        target_fit_u = q_target * self.eigvals[None, :]
        target_eval_dot_fit = np.einsum("ij,ij->i", target_eval_u, q_target)

        n_target = len(target_indices)
        n_targets = self.y.shape[1]
        unique_alphas = np.unique(alpha_arr)
        if unique_alphas.size <= max(4, alpha_arr.size // 16):
            target_loso = np.empty((n_target, n_targets), dtype=np.float32)
            target_hat = np.empty((n_target, n_targets), dtype=np.float32)
            for alpha in unique_alphas:
                cols = np.flatnonzero(alpha_arr == alpha)
                inv_alpha = 1.0 / (self.eigvals + float(alpha))
                coef_q = self.y_projected[:, cols] * inv_alpha[:, None]
                target_fit_pred = self.y_bar[cols][None, :] + target_fit_u @ coef_q
                target_eval_pred = self.y_bar[cols][None, :] + target_eval_u @ coef_q
                residual = self.y[target_indices[:, None], cols] - target_fit_pred
                hat_feature = (self.eigvecs[target_indices] ** 2) @ (
                    self.eigvals * inv_alpha
                )
                hat = np.clip(
                    target_weight / self.weight_total + hat_feature,
                    0.0,
                    0.999999,
                )
                dual_cross = np.sum(
                    target_eval_u * inv_alpha[None, :] * target_fit_u,
                    axis=1,
                )
                cross = target_weight / self.weight_total + target_weight * (
                    (target_eval_dot_fit - dual_cross) / float(alpha)
                )
                target_loso[:, cols] = (
                    target_eval_pred
                    - cross[:, None] * residual / (1.0 - hat[:, None])
                ).astype(np.float32)
                target_hat[:, cols] = hat[:, None].astype(np.float32)
            extra_pred = (
                self.predict_heldout(X_extra_eval, alpha_arr)
                if X_extra_eval is not None
                else None
            )
            return target_loso, extra_pred, target_hat

        inv = 1.0 / (self.eigvals[:, None] + alpha_arr[None, :])
        coef_q = self.y_projected * inv

        target_fit_pred = self.y_bar[None, :] + target_fit_u @ coef_q
        target_eval_pred = self.y_bar[None, :] + target_eval_u @ coef_q
        residual = self.y[target_indices] - target_fit_pred
        hat_feature = (self.eigvecs[target_indices] ** 2) @ (
            self.eigvals[:, None] * inv
        )
        target_hat = np.clip(
            target_weight[:, None] / self.weight_total + hat_feature,
            0.0,
            0.999999,
        )
        dual_cross = (target_eval_u * target_fit_u) @ inv
        cross = (
            target_weight[:, None] / self.weight_total
            + target_weight[:, None]
            * (target_eval_dot_fit[:, None] - dual_cross)
            / alpha_arr[None, :]
        )
        target_loso = (
            target_eval_pred - cross * residual / (1.0 - target_hat)
        ).astype(np.float32)

        if X_extra_eval is not None:
            X_extra_eval = _check_2d_array(
                "X_extra_eval",
                np.asarray(X_extra_eval, dtype=np.float64),
            )
            extra_eval = X_extra_eval - self.x_bar[None, :]
            extra_eval_u = extra_eval @ self.feature_basis
            extra_pred = (self.y_bar[None, :] + extra_eval_u @ coef_q).astype(
                np.float32
            )
        else:
            extra_pred = None
        return target_loso, extra_pred, target_hat.astype(np.float32)

    def _normalize_target_alphas(self, alphas: ArrayLike) -> np.ndarray:
        alpha_arr = np.asarray(alphas, dtype=np.float64)
        if alpha_arr.ndim == 0:
            alpha_arr = np.full(self.y.shape[1], float(alpha_arr), dtype=np.float64)
        if alpha_arr.shape != (self.y.shape[1],):
            raise ValueError(
                f"alphas must be scalar or have shape ({self.y.shape[1]},), got {alpha_arr.shape}"
            )
        if np.any(alpha_arr <= 0.0):
            raise ValueError("alphas must be strictly positive")
        return alpha_arr
