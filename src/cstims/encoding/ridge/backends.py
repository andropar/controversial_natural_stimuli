from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .common import _ProcessedFit, _normalize_scoring

def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The torch ridge backend requires PyTorch. Install the full project "
            "environment or use backend='sklearn'."
        ) from exc
    return torch


def _torch_dtype(torch: Any, dtype: str | Any) -> Any:
    if not isinstance(dtype, str):
        return dtype
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError("torch_dtype must be 'float32' or 'float64'")


def _resolve_torch_device(torch: Any, device: str | None) -> Any:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "gpu":
        device = "cuda"
    return torch.device(device)


def _torch_pearson_scores(y_true: Any, y_pred: Any) -> Any:
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    numerator = (yt * yp).sum(dim=0)
    denominator = ((yt * yt).sum(dim=0) * (yp * yp).sum(dim=0)).sqrt()
    return numerator / denominator


def _torch_score_columns(y_true: Any, y_pred: Any, scoring: str) -> Any:
    scoring = _normalize_scoring(scoring)
    if scoring == "pearson_r":
        return _torch_pearson_scores(y_true, y_pred)
    return -((y_true - y_pred) ** 2).mean(dim=0)


def _torch_select_alphas_processed(
    X_proc: np.ndarray,
    y_proc: np.ndarray,
    alphas: Sequence[float],
    *,
    fit_intercept: bool,
    alpha_per_target: bool,
    scoring: str,
    device: str | None,
    dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Torch implementation of RidgeCVFast's Gram-mode LOO alpha selection."""
    if X_proc.shape[0] > X_proc.shape[1]:
        raise ValueError(
            "The torch ridge backend currently implements the Gram/GCV path only; "
            "expected n_samples <= n_features for alpha selection."
        )
    torch = _import_torch()
    torch_device = _resolve_torch_device(torch, device)
    torch_dtype = _torch_dtype(torch, dtype)
    X = torch.as_tensor(X_proc, device=torch_device, dtype=torch_dtype)
    Y = torch.as_tensor(y_proc, device=torch_device, dtype=torch_dtype)

    if fit_intercept:
        X_offset = X.mean(dim=0)
        y_offset = Y.mean(dim=0)
        X_fit = X - X_offset
        y_fit = Y - y_offset
        sqrt_sw = torch.ones(X.shape[0], device=torch_device, dtype=torch_dtype)
        gram = X_fit @ X_fit.T + torch.outer(sqrt_sw, sqrt_sw)
    else:
        y_offset = torch.zeros(Y.shape[1], device=torch_device, dtype=torch_dtype)
        X_fit = X
        y_fit = Y
        gram = X_fit @ X_fit.T
        sqrt_sw = None

    eigvals, Q = torch.linalg.eigh(gram)
    eigvals = torch.clamp(eigvals, min=0)
    QT_Y = Q.T @ y_fit

    intercept_dim = None
    if fit_intercept and sqrt_sw is not None:
        normalized_sw = sqrt_sw / torch.linalg.vector_norm(sqrt_sw)
        intercept_dim = torch.argmax(torch.abs(normalized_sw @ Q))

    scores = []
    for alpha in np.asarray(alphas, dtype=np.float64):
        inv = 1.0 / (eigvals + float(alpha))
        if intercept_dim is not None:
            inv = inv.clone()
            inv[intercept_dim] = 0.0
        dual_coef = Q @ (inv[:, None] * QT_Y)
        g_inverse_diag = (Q * Q) @ inv
        predictions = y_fit - dual_coef / g_inverse_diag[:, None] + y_offset
        scores.append(_torch_score_columns(Y, predictions, scoring))

    score_stack = torch.stack(scores, dim=0)
    if alpha_per_target:
        best_idx = torch.argmax(score_stack, dim=0)
        alpha_tensor = torch.as_tensor(
            np.asarray(alphas, dtype=np.float64),
            device=torch_device,
            dtype=torch_dtype,
        )
        best_alpha = alpha_tensor[best_idx]
        best_score = score_stack.gather(0, best_idx.unsqueeze(0)).squeeze(0)
        return (
            best_alpha.detach().cpu().numpy().astype(np.float64),
            best_score.detach().cpu().numpy().astype(np.float64),
        )

    mean_scores = score_stack.mean(dim=1)
    best_idx = int(torch.argmax(mean_scores).detach().cpu().item())
    best_alpha = np.full(Y.shape[1], float(np.asarray(alphas)[best_idx]), dtype=np.float64)
    best_score = np.full(Y.shape[1], float(mean_scores[best_idx].detach().cpu().item()), dtype=np.float64)
    return best_alpha, best_score


def _torch_fit_processed(
    X_proc: np.ndarray,
    y_proc: np.ndarray,
    alphas: Sequence[float],
    *,
    fit_intercept: bool,
    alpha_per_target: bool,
    scoring: str,
    device: str | None,
    dtype: str,
) -> _ProcessedFit:
    """Torch Gram/GCV fit that reuses the alpha-selection decomposition."""
    if X_proc.shape[0] > X_proc.shape[1]:
        raise ValueError(
            "The torch ridge backend currently implements the Gram/GCV path only; "
            "expected n_samples <= n_features for alpha selection."
        )
    torch = _import_torch()
    torch_device = _resolve_torch_device(torch, device)
    torch_dtype = _torch_dtype(torch, dtype)
    X = torch.as_tensor(X_proc, device=torch_device, dtype=torch_dtype)
    Y = torch.as_tensor(y_proc, device=torch_device, dtype=torch_dtype)

    if fit_intercept:
        X_offset = X.mean(dim=0)
        y_offset = Y.mean(dim=0)
        X_fit = X - X_offset
        y_fit = Y - y_offset
        sqrt_sw = torch.ones(X.shape[0], device=torch_device, dtype=torch_dtype)
        gram = X_fit @ X_fit.T + torch.outer(sqrt_sw, sqrt_sw)
    else:
        X_offset = torch.zeros(X.shape[1], device=torch_device, dtype=torch_dtype)
        y_offset = torch.zeros(Y.shape[1], device=torch_device, dtype=torch_dtype)
        X_fit = X
        y_fit = Y
        gram = X_fit @ X_fit.T
        sqrt_sw = None

    eigvals, Q = torch.linalg.eigh(gram)
    eigvals = torch.clamp(eigvals, min=0)
    QT_Y = Q.T @ y_fit

    intercept_dim = None
    if fit_intercept and sqrt_sw is not None:
        normalized_sw = sqrt_sw / torch.linalg.vector_norm(sqrt_sw)
        intercept_dim = torch.argmax(torch.abs(normalized_sw @ Q))

    alpha_values_np = np.asarray(alphas, dtype=np.float64)
    alpha_tensor = torch.as_tensor(
        alpha_values_np,
        device=torch_device,
        dtype=torch_dtype,
    )
    scores = []
    for alpha in alpha_values_np:
        inv = 1.0 / (eigvals + float(alpha))
        if intercept_dim is not None:
            inv = inv.clone()
            inv[intercept_dim] = 0.0
        dual_coef = Q @ (inv[:, None] * QT_Y)
        g_inverse_diag = (Q * Q) @ inv
        predictions = y_fit - dual_coef / g_inverse_diag[:, None] + y_offset
        scores.append(_torch_score_columns(Y, predictions, scoring))

    score_stack = torch.stack(scores, dim=0)
    if alpha_per_target:
        best_idx = torch.argmax(score_stack, dim=0)
        best_alpha_tensor = alpha_tensor[best_idx]
        best_score = score_stack.gather(0, best_idx.unsqueeze(0)).squeeze(0)
    else:
        mean_scores = score_stack.mean(dim=1)
        best_idx_scalar = int(torch.argmax(mean_scores).detach().cpu().item())
        best_alpha_tensor = torch.full(
            (Y.shape[1],),
            float(alpha_values_np[best_idx_scalar]),
            device=torch_device,
            dtype=torch_dtype,
        )
        best_score = torch.full(
            (Y.shape[1],),
            float(mean_scores[best_idx_scalar].detach().cpu().item()),
            device=torch_device,
            dtype=torch_dtype,
        )

    inv = 1.0 / (eigvals[:, None] + best_alpha_tensor[None, :])
    if intercept_dim is not None:
        inv = inv.clone()
        inv[intercept_dim, :] = 0.0
    dual_coef = Q @ (QT_Y * inv)
    W_proc = X_fit.T @ dual_coef
    b_proc = y_offset - X_offset @ W_proc
    return _ProcessedFit(
        weights=W_proc.detach().cpu().numpy().astype(np.float64),
        intercept=b_proc.detach().cpu().numpy().astype(np.float64),
        alphas=best_alpha_tensor.detach().cpu().numpy().astype(np.float64),
        score=best_score.detach().cpu().numpy().astype(np.float64),
        estimator=None,
    )


def _torch_refit_processed(
    X_proc: np.ndarray,
    y_proc: np.ndarray,
    alphas: np.ndarray,
    *,
    fit_intercept: bool,
    device: str | None,
    dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    torch = _import_torch()
    torch_device = _resolve_torch_device(torch, device)
    torch_dtype = _torch_dtype(torch, dtype)
    X = torch.as_tensor(X_proc, device=torch_device, dtype=torch_dtype)
    Y = torch.as_tensor(y_proc, device=torch_device, dtype=torch_dtype)

    if fit_intercept:
        X_offset = X.mean(dim=0)
        y_offset = Y.mean(dim=0)
        X_fit = X - X_offset
        y_fit = Y - y_offset
    else:
        X_offset = torch.zeros(X.shape[1], device=torch_device, dtype=torch_dtype)
        y_offset = torch.zeros(Y.shape[1], device=torch_device, dtype=torch_dtype)
        X_fit = X
        y_fit = Y

    gram = X_fit @ X_fit.T
    eigvals, Q = torch.linalg.eigh(gram)
    eigvals = torch.clamp(eigvals, min=0)

    alpha = torch.as_tensor(
        np.asarray(alphas, dtype=np.float64),
        device=torch_device,
        dtype=torch_dtype,
    )
    projected_y = Q.T @ y_fit
    dual_coef = Q @ (projected_y / (eigvals[:, None] + alpha[None, :]))
    W_proc = X_fit.T @ dual_coef
    b_proc = y_offset - X_offset @ W_proc

    return (
        W_proc.detach().cpu().numpy().astype(np.float64),
        b_proc.detach().cpu().numpy().astype(np.float64),
    )


