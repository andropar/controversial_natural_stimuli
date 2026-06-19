from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import ArrayLike, _check_2d_array, _symmetric_inverse_spd

@dataclass
class SchurCandidateReadoutCache:
    """Schur-update cache for adding one candidate to a selected eval set.

    This is the reusable version of the refit-robust stimulus-selection math.
    All feature/kernel inputs must already be in the same processed feature
    space.  The cache is target-free: once built, ``materialize_candidate`` emits
    alpha-wise linear readout operators for any target matrix.
    """

    alphas: tuple[float, ...]
    selected_inverse: np.ndarray
    selected_inverse_diag: np.ndarray
    selected_base_numerator: np.ndarray
    candidate_q: np.ndarray
    candidate_delta: np.ndarray
    candidate_z: np.ndarray

    @classmethod
    def from_feature_blocks(
        cls,
        *,
        X_base: np.ndarray,
        X_selected: np.ndarray,
        X_candidates: np.ndarray,
        alphas: ArrayLike,
    ) -> "SchurCandidateReadoutCache":
        """Build from processed base, selected-eval, and candidate features."""
        X_base = _check_2d_array("X_base", np.asarray(X_base, dtype=np.float64))
        X_selected = _check_2d_array(
            "X_selected",
            np.asarray(X_selected, dtype=np.float64),
        )
        X_candidates = _check_2d_array(
            "X_candidates",
            np.asarray(X_candidates, dtype=np.float64),
        )
        if X_selected.shape[1] != X_base.shape[1]:
            raise ValueError("X_selected and X_base must have the same feature count")
        if X_candidates.shape[1] != X_base.shape[1]:
            raise ValueError("X_candidates and X_base must have the same feature count")

        k_base = X_base @ X_base.T
        k_base = 0.5 * (k_base + k_base.T)
        base_eigvals, base_eigvecs = np.linalg.eigh(k_base)
        return cls.from_kernel_blocks(
            base_eigvals=np.maximum(base_eigvals, 0.0),
            base_eigvecs=base_eigvecs,
            k_base_selected=X_base @ X_selected.T,
            k_base_candidates=X_base @ X_candidates.T,
            k_selected_selected=X_selected @ X_selected.T,
            k_selected_candidates=X_selected @ X_candidates.T,
            k_candidate_diagonal=np.einsum("ij,ij->i", X_candidates, X_candidates),
            alphas=alphas,
        )

    @classmethod
    def from_kernel_blocks(
        cls,
        *,
        base_eigvals: np.ndarray,
        base_eigvecs: np.ndarray,
        k_base_selected: np.ndarray,
        k_base_candidates: np.ndarray,
        k_selected_selected: np.ndarray,
        k_selected_candidates: np.ndarray,
        k_candidate_diagonal: np.ndarray,
        alphas: ArrayLike,
    ) -> "SchurCandidateReadoutCache":
        """Build from kernel blocks around a fixed base eigendecomposition."""
        alpha_array = np.asarray(np.atleast_1d(alphas), dtype=np.float64)
        if alpha_array.ndim != 1 or alpha_array.size == 0:
            raise ValueError("alphas must be a non-empty 1D array")
        if np.any(alpha_array <= 0.0):
            raise ValueError("alphas must be strictly positive")

        base_eigvals = np.maximum(np.asarray(base_eigvals, dtype=np.float64), 0.0)
        base_eigvecs = _check_2d_array(
            "base_eigvecs",
            np.asarray(base_eigvecs, dtype=np.float64),
        )
        n_base = base_eigvals.shape[0]
        if base_eigvecs.shape != (n_base, n_base):
            raise ValueError(
                "base_eigvecs must have shape "
                f"({n_base}, {n_base}), got {base_eigvecs.shape}"
            )

        k_base_selected = _check_2d_array(
            "k_base_selected",
            np.asarray(k_base_selected, dtype=np.float64),
        )
        k_base_candidates = _check_2d_array(
            "k_base_candidates",
            np.asarray(k_base_candidates, dtype=np.float64),
        )
        if k_base_selected.shape[0] != n_base:
            raise ValueError("k_base_selected has incompatible base dimension")
        if k_base_candidates.shape[0] != n_base:
            raise ValueError("k_base_candidates has incompatible base dimension")

        n_selected = k_base_selected.shape[1]
        n_candidates = k_base_candidates.shape[1]
        k_selected_selected = _check_2d_array(
            "k_selected_selected",
            np.asarray(k_selected_selected, dtype=np.float64),
        )
        k_selected_candidates = _check_2d_array(
            "k_selected_candidates",
            np.asarray(k_selected_candidates, dtype=np.float64),
        )
        k_candidate_diagonal = np.asarray(k_candidate_diagonal, dtype=np.float64)
        if k_selected_selected.shape != (n_selected, n_selected):
            raise ValueError("k_selected_selected has incompatible shape")
        if k_selected_candidates.shape != (n_selected, n_candidates):
            raise ValueError("k_selected_candidates has incompatible shape")
        if k_candidate_diagonal.shape != (n_candidates,):
            raise ValueError("k_candidate_diagonal has incompatible shape")

        qtu_selected = base_eigvecs.T @ k_base_selected
        eye_selected = np.eye(n_selected, dtype=np.float64)
        selected_inverse_list: list[np.ndarray] = []
        a_inv_u_selected_list: list[np.ndarray] = []
        for alpha in alpha_array:
            denom = base_eigvals + float(alpha)
            a_inv_u_selected = base_eigvecs @ (qtu_selected / denom[:, None])
            if n_selected:
                schur_selected = (
                    k_selected_selected
                    + float(alpha) * eye_selected
                    - k_base_selected.T @ a_inv_u_selected
                )
                schur_selected = 0.5 * (schur_selected + schur_selected.T)
                selected_inverse = _symmetric_inverse_spd(schur_selected)
            else:
                selected_inverse = np.empty((0, 0), dtype=np.float64)
            selected_inverse_list.append(np.asarray(selected_inverse, dtype=np.float64))
            a_inv_u_selected_list.append(
                np.asarray(a_inv_u_selected, dtype=np.float64)
            )

        qtu_candidates = base_eigvecs.T @ k_base_candidates
        inverse_denominators = 1.0 / (base_eigvals[None, :] + alpha_array[:, None])
        a_inv_u_candidates = np.matmul(
            base_eigvecs[None, :, :],
            qtu_candidates[None, :, :] * inverse_denominators[:, :, None],
        )
        selected_inverse = np.stack(selected_inverse_list, axis=0)
        a_inv_u_selected = np.stack(a_inv_u_selected_list, axis=0)
        schur_diagonal = (
            k_candidate_diagonal[None, :]
            + alpha_array[:, None]
            - np.sum(k_base_candidates[None, :, :] * a_inv_u_candidates, axis=1)
        )

        if n_selected:
            cross = (
                k_selected_candidates[None, :, :]
                - np.matmul(k_base_selected.T[None, :, :], a_inv_u_candidates)
            )
            candidate_q = np.matmul(selected_inverse, cross)
            candidate_delta = schur_diagonal - np.sum(cross * candidate_q, axis=1)
            candidate_z = a_inv_u_candidates - np.matmul(
                a_inv_u_selected,
                candidate_q,
            )
            selected_base_numerator = np.matmul(
                selected_inverse,
                np.transpose(a_inv_u_selected, (0, 2, 1)),
            )
        else:
            candidate_q = np.empty(
                (alpha_array.size, 0, n_candidates),
                dtype=np.float64,
            )
            candidate_delta = schur_diagonal
            candidate_z = a_inv_u_candidates
            selected_base_numerator = np.empty(
                (alpha_array.size, 0, n_base),
                dtype=np.float64,
            )

        return cls(
            alphas=tuple(float(alpha) for alpha in alpha_array),
            selected_inverse=np.asarray(selected_inverse, dtype=np.float64),
            selected_inverse_diag=np.diagonal(
                selected_inverse,
                axis1=1,
                axis2=2,
            ).copy(),
            selected_base_numerator=np.asarray(
                selected_base_numerator,
                dtype=np.float64,
            ),
            candidate_q=np.asarray(candidate_q, dtype=np.float64),
            candidate_delta=np.asarray(candidate_delta, dtype=np.float64),
            candidate_z=np.asarray(candidate_z, dtype=np.float64),
        )

    def materialize_candidate(
        self,
        candidate_pos: int,
        *,
        delta_tol: float = 1e-10,
        dtype: Any = np.float32,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(base_ops, eval_ops)`` for selected rows plus one candidate.

        For a chosen alpha index, predictions are
        ``base_ops[alpha_idx] @ y_base + eval_ops[alpha_idx] @ y_eval`` where
        ``y_eval`` is ordered as ``selected`` rows followed by the candidate row.
        ``None`` is returned when the candidate Schur complement is numerically
        invalid for any alpha.
        """
        n_candidates = self.candidate_delta.shape[1]
        if candidate_pos < 0 or candidate_pos >= n_candidates:
            raise ValueError(
                f"candidate_pos must be in [0, {n_candidates}), got {candidate_pos}"
            )
        delta = self.candidate_delta[:, candidate_pos]
        if np.any(delta <= delta_tol) or not np.all(np.isfinite(delta)):
            return None

        n_alphas = self.selected_inverse.shape[0]
        n_selected = self.selected_inverse.shape[1]
        n_base = self.selected_base_numerator.shape[2]
        n_eval = n_selected + 1
        out_dtype = np.dtype(dtype)
        base_ops = np.empty((n_alphas, n_eval, n_base), dtype=out_dtype)
        eval_ops = np.zeros((n_alphas, n_eval, n_eval), dtype=out_dtype)

        q = self.candidate_q[:, :, candidate_pos]
        z = self.candidate_z[:, :, candidate_pos]
        inv_delta = 1.0 / self.candidate_delta[:, candidate_pos]

        if n_selected:
            inverse_diag = self.selected_inverse_diag + q * q * inv_delta[:, None]
            base_selected = (
                self.selected_base_numerator
                - q[:, :, None] * z[:, None, :] * inv_delta[:, None, None]
            ) / inverse_diag[:, :, None]

            inverse_rows = (
                self.selected_inverse
                + q[:, :, None] * q[:, None, :] * inv_delta[:, None, None]
            )
            eval_selected = -inverse_rows / inverse_diag[:, :, None]
            diag_idx = np.arange(n_selected)
            eval_selected[:, diag_idx, diag_idx] = 0.0

            base_ops[:, :n_selected, :] = base_selected.astype(out_dtype, copy=False)
            eval_ops[:, :n_selected, :n_selected] = eval_selected.astype(
                out_dtype,
                copy=False,
            )
            eval_ops[:, :n_selected, n_selected] = (
                q * inv_delta[:, None] / inverse_diag
            ).astype(out_dtype, copy=False)

        base_ops[:, n_selected, :] = z.astype(out_dtype, copy=False)
        eval_ops[:, n_selected, :n_selected] = q.astype(out_dtype, copy=False)
        return base_ops, eval_ops

    def predict_candidate(
        self,
        candidate_pos: int,
        *,
        y_base: np.ndarray,
        y_eval: np.ndarray,
        delta_tol: float = 1e-10,
        dtype: Any = np.float32,
    ) -> np.ndarray | None:
        """Predict selected eval rows plus one candidate without materializing ops.

        Returns an array of shape ``(n_alphas, n_selected + 1, n_targets)``.
        ``y_eval`` must be ordered as selected rows followed by the candidate row.
        """
        n_candidates = self.candidate_delta.shape[1]
        if candidate_pos < 0 or candidate_pos >= n_candidates:
            raise ValueError(
                f"candidate_pos must be in [0, {n_candidates}), got {candidate_pos}"
            )
        delta = self.candidate_delta[:, candidate_pos]
        if np.any(delta <= delta_tol) or not np.all(np.isfinite(delta)):
            return None

        y_base = np.asarray(y_base, dtype=np.float64)
        y_eval = np.asarray(y_eval, dtype=np.float64)
        n_alphas, n_selected, _n_candidates = self.candidate_q.shape
        n_eval = n_selected + 1
        if y_base.shape[0] != self.selected_base_numerator.shape[2]:
            raise ValueError("y_base has incompatible row count")
        if y_eval.shape[0] != n_eval:
            raise ValueError(f"y_eval must have {n_eval} rows")

        q = self.candidate_q[:, :, candidate_pos]
        z = self.candidate_z[:, :, candidate_pos]
        inv_delta = 1.0 / delta
        pred = np.empty((n_alphas, n_eval, y_base.shape[1]), dtype=np.float64)

        if n_selected:
            inverse_diag = self.selected_inverse_diag + q * q * inv_delta[:, None]
            base_numerator = (
                self.selected_base_numerator
                - q[:, :, None] * z[:, None, :] * inv_delta[:, None, None]
            )
            inverse_rows = (
                self.selected_inverse
                + q[:, :, None] * q[:, None, :] * inv_delta[:, None, None]
            )
            selected_base = np.einsum("asb,bt->ast", base_numerator, y_base)
            selected_eval = -np.einsum(
                "ase,et->ast",
                inverse_rows,
                y_eval[:n_selected],
            )
            diag_idx = np.arange(n_selected)
            selected_eval[:, diag_idx] += (
                inverse_rows[:, diag_idx, diag_idx, None]
                * y_eval[diag_idx][None, :, :]
            )
            selected_eval += (
                q[:, :, None]
                * inv_delta[:, None, None]
                * y_eval[n_selected][None, None, :]
            )
            pred[:, :n_selected] = (selected_base + selected_eval) / inverse_diag[
                :,
                :,
                None,
            ]

        pred[:, n_selected] = (
            np.einsum("ab,bt->at", z, y_base)
            + np.einsum("as,st->at", q, y_eval[:n_selected])
        )
        return pred.astype(dtype, copy=False)

