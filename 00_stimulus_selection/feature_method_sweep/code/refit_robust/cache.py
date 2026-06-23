"""Round-level ridge caches for refit-robust candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration dependency
    njit = None

NUMBA_AVAILABLE = njit is not None

from .data import FitContext, TeacherNoiseState, apply_standardization


@dataclass
class SelectedAlphaState:
    a_inv_u_selected: np.ndarray
    selected_inverse: np.ndarray


@dataclass
class StudentRoundCache:
    selected_inverse: np.ndarray
    selected_inverse_diag: np.ndarray
    selected_base_numerator: np.ndarray
    candidate_q: np.ndarray
    candidate_delta: np.ndarray
    candidate_z: np.ndarray


@dataclass
class PredictionPath:
    order: np.ndarray
    offsets: np.ndarray


@dataclass
class RoundCache:
    student_caches: dict[str, StudentRoundCache]
    paths: dict[str, PredictionPath]
    blocks: list[tuple[str, int]]
    candidate_pos: dict[int, int]


if njit is not None:

    @njit(cache=False, nogil=True)
    def _materialize_candidate_ops_numba(
        selected_inverse: np.ndarray,
        selected_inverse_diag: np.ndarray,
        selected_base_numerator: np.ndarray,
        candidate_q: np.ndarray,
        candidate_delta: np.ndarray,
        candidate_z: np.ndarray,
        candidate_pos: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Materialize LOO base and eval operators for one shortlist candidate."""
        n_alphas = selected_inverse.shape[0]
        n_selected = selected_inverse.shape[1]
        n_base = selected_base_numerator.shape[2]
        n_eval = n_selected + 1
        base_ops = np.empty((n_alphas, n_eval, n_base), dtype=np.float32)
        eval_ops = np.empty((n_alphas, n_eval, n_eval), dtype=np.float32)

        for alpha_idx in range(n_alphas):
            delta = candidate_delta[alpha_idx, candidate_pos]
            inv_delta = 1.0 / delta
            for row in range(n_selected):
                q_row = candidate_q[alpha_idx, row, candidate_pos]
                inverse_diag = (
                    selected_inverse_diag[alpha_idx, row]
                    + q_row * q_row * inv_delta
                )
                inv_inverse_diag = 1.0 / inverse_diag
                for base_idx in range(n_base):
                    numerator = (
                        selected_base_numerator[alpha_idx, row, base_idx]
                        - q_row
                        * candidate_z[alpha_idx, base_idx, candidate_pos]
                        * inv_delta
                    )
                    base_ops[alpha_idx, row, base_idx] = np.float32(
                        numerator * inv_inverse_diag
                    )
                for col in range(n_selected):
                    if row == col:
                        eval_ops[alpha_idx, row, col] = np.float32(0.0)
                    else:
                        inverse_value = (
                            selected_inverse[alpha_idx, row, col]
                            + q_row
                            * candidate_q[alpha_idx, col, candidate_pos]
                            * inv_delta
                        )
                        eval_ops[alpha_idx, row, col] = np.float32(
                            -inverse_value * inv_inverse_diag
                        )
                eval_ops[alpha_idx, row, n_selected] = np.float32(
                    q_row * inv_delta * inv_inverse_diag
                )

            for base_idx in range(n_base):
                base_ops[alpha_idx, n_selected, base_idx] = np.float32(
                    candidate_z[alpha_idx, base_idx, candidate_pos]
                )
            for col in range(n_selected):
                eval_ops[alpha_idx, n_selected, col] = np.float32(
                    candidate_q[alpha_idx, col, candidate_pos]
                )
            eval_ops[alpha_idx, n_selected, n_selected] = np.float32(0.0)
        return base_ops, eval_ops


else:  # pragma: no cover - exercised only when Numba is unavailable
    _materialize_candidate_ops_numba = None


def build_round_cache(
    *,
    selected_indices: list[int],
    shortlist: np.ndarray,
    raw_features_np: dict[str, np.ndarray],
    fit_context: FitContext,
    noise_states: dict[str, list[TeacherNoiseState]],
    model_names: list[str],
    alphas: list[float],
    build_paths: bool = True,
) -> RoundCache:
    """Build per-student Schur-update caches for the current selected set and shortlist."""
    selected_array = np.asarray(selected_indices, dtype=np.int64)
    shortlist = np.asarray(shortlist, dtype=np.int64)
    alpha_array = np.asarray(alphas, dtype=np.float64)
    student_caches: dict[str, StudentRoundCache] = {}

    for student in model_names:
        ops = fit_context.student_ops[student]
        x_selected = apply_standardization(
            raw_features_np[student][selected_array],
            ops.train_mean,
            ops.train_scale,
        )
        x_candidates = apply_standardization(
            raw_features_np[student][shortlist],
            ops.train_mean,
            ops.train_scale,
        )
        x_selected64 = x_selected.astype(np.float64, copy=False)
        x_candidates64 = x_candidates.astype(np.float64, copy=False)
        x_base64 = ops.x_base.astype(np.float64, copy=False)

        if ops.k_base_pool is not None:
            k_base_selected = np.asarray(ops.k_base_pool[:, selected_array], dtype=np.float64)
            k_base_candidates = np.asarray(ops.k_base_pool[:, shortlist], dtype=np.float64)
        else:
            k_base_selected = x_base64 @ x_selected64.T
            k_base_candidates = x_base64 @ x_candidates64.T
        k_selected_selected = x_selected64 @ x_selected64.T
        k_selected_candidates = x_selected64 @ x_candidates64.T
        k_candidate_diagonal = np.einsum("cf,cf->c", x_candidates64, x_candidates64)

        qtu_selected = ops.base_eigvecs.T @ k_base_selected
        eye_selected = np.eye(len(selected_array), dtype=np.float64)
        alpha_states: list[SelectedAlphaState] = []
        for alpha in alphas:
            alpha = float(alpha)
            denom = ops.base_eigvals + alpha
            a_inv_u_selected = ops.base_eigvecs @ (qtu_selected / denom[:, None])
            schur_selected = (
                k_selected_selected
                + alpha * eye_selected
                - k_base_selected.T @ a_inv_u_selected
            )
            schur_selected = 0.5 * (schur_selected + schur_selected.T)
            selected_inverse = np.linalg.inv(schur_selected)
            alpha_states.append(
                SelectedAlphaState(
                    a_inv_u_selected=np.asarray(a_inv_u_selected, dtype=np.float64),
                    selected_inverse=np.asarray(selected_inverse, dtype=np.float64),
                )
            )
        qtu_candidates = ops.base_eigvecs.T @ k_base_candidates
        inverse_denominators = 1.0 / (ops.base_eigvals[None, :] + alpha_array[:, None])
        a_inv_u_candidates = np.matmul(
            ops.base_eigvecs[None, :, :],
            qtu_candidates[None, :, :] * inverse_denominators[:, :, None],
        )
        selected_inverse = np.stack(
            [state.selected_inverse for state in alpha_states],
            axis=0,
        )
        a_inv_u_selected = np.stack(
            [state.a_inv_u_selected for state in alpha_states],
            axis=0,
        )
        cross = (
            k_selected_candidates[None, :, :]
            - np.matmul(k_base_selected.T[None, :, :], a_inv_u_candidates)
        )
        schur_diagonal = (
            k_candidate_diagonal[None, :]
            + alpha_array[:, None]
            - np.sum(k_base_candidates[None, :, :] * a_inv_u_candidates, axis=1)
        )
        candidate_q = np.matmul(selected_inverse, cross)
        candidate_delta = schur_diagonal - np.sum(cross * candidate_q, axis=1)
        candidate_z = a_inv_u_candidates - np.matmul(a_inv_u_selected, candidate_q)
        selected_inverse_diag = np.diagonal(selected_inverse, axis1=1, axis2=2).copy()
        selected_base_numerator = np.matmul(
            selected_inverse,
            np.transpose(a_inv_u_selected, (0, 2, 1)),
        )
        student_caches[student] = StudentRoundCache(
            selected_inverse=np.asarray(selected_inverse, dtype=np.float64),
            selected_inverse_diag=np.asarray(selected_inverse_diag, dtype=np.float64),
            selected_base_numerator=np.asarray(selected_base_numerator, dtype=np.float64),
            candidate_q=np.asarray(candidate_q, dtype=np.float64),
            candidate_delta=np.asarray(candidate_delta, dtype=np.float64),
            candidate_z=np.asarray(candidate_z, dtype=np.float64),
        )

    blocks = [
        (teacher, noise_idx)
        for teacher in model_names
        for noise_idx in range(len(noise_states[teacher]))
    ]
    paths: dict[str, PredictionPath] = {}
    if build_paths:
        target_dim = fit_context.teacher_targets[model_names[0]].y_base_clean.shape[1]
        for student in model_names:
            alpha_indices: list[np.ndarray] = []
            for alpha_idx in range(len(alphas)):
                packed_indices: list[np.ndarray] = []
                for block_idx, (teacher, noise_idx) in enumerate(blocks):
                    noise_state = noise_states[teacher][noise_idx]
                    _alpha_values, best_alpha_idx = noise_state.alpha_choices[student]
                    cols = np.flatnonzero(best_alpha_idx == alpha_idx)
                    packed_indices.append(block_idx * target_dim + cols)
                alpha_indices.append(np.concatenate(packed_indices))
            order = np.concatenate(alpha_indices).astype(np.int64, copy=False)
            offsets = np.zeros(len(alphas) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(
                np.asarray([indices.size for indices in alpha_indices], dtype=np.int64)
            )
            paths[student] = PredictionPath(
                order=order,
                offsets=offsets,
            )

    return RoundCache(
        student_caches=student_caches,
        paths=paths,
        blocks=blocks,
        candidate_pos={int(candidate): pos for pos, candidate in enumerate(shortlist)},
    )


def materialize_candidate_ops(
    cache: StudentRoundCache,
    candidate_pos: int,
    *,
    delta_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return dense candidate operators when the Schur update is numerically stable."""
    delta = cache.candidate_delta[:, candidate_pos]
    if np.any(delta <= delta_tol) or not np.all(np.isfinite(delta)):
        return None
    assert _materialize_candidate_ops_numba is not None
    return _materialize_candidate_ops_numba(
        cache.selected_inverse,
        cache.selected_inverse_diag,
        cache.selected_base_numerator,
        cache.candidate_q,
        cache.candidate_delta,
        cache.candidate_z,
        candidate_pos,
    )
