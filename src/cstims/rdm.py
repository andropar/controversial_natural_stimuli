"""Reference RDM and RSA helpers.

This module is the backend-neutral home for small, deterministic RDM/RSA
operations. GPU/torch optimized code can live behind this API, but the
functions here intentionally preserve the historical NumPy/SciPy semantics used
by the paper analysis scripts.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from scipy import stats
from scipy.spatial import distance

RDMMetric = Literal["correlation", "cosine", "euclidean"]
CorrelationMethod = Literal["spearman", "pearson", "correlation"]
EPSILON = 1e-9
_TRIU_INDEX_CACHE: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def compute_rdm(features: np.ndarray, metric: RDMMetric = "correlation") -> np.ndarray:
    """Compute a square representational dissimilarity matrix.

    Parameters
    ----------
    features:
        Array shaped ``(n_items, n_features)``.
    metric:
        Distance metric. ``"correlation"`` matches the previous
        ``cstims.paper.utils.compute_rdm_correlation`` implementation:
        ``1 - np.corrcoef(features)`` with the diagonal set to zero.
    """
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")

    if metric == "correlation":
        corr = np.corrcoef(features)
        rdm = 1.0 - corr
        np.fill_diagonal(rdm, 0.0)
        return rdm

    if metric in {"cosine", "euclidean"}:
        return distance.squareform(distance.pdist(features, metric=metric))

    raise ValueError(f"Unsupported RDM metric: {metric}")


def compute_rdm_correlation(features: np.ndarray) -> np.ndarray:
    """Compute a correlation-distance RDM."""
    return compute_rdm(features, metric="correlation")


def rdm_to_vector(rdm: np.ndarray) -> np.ndarray:
    """Extract the upper-triangular RDM values, excluding the diagonal."""
    rdm = np.asarray(rdm)
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError(f"Expected square RDM, got shape {rdm.shape}")
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def correlate_vectors(
    x: np.ndarray,
    y: np.ndarray,
    method: CorrelationMethod = "spearman",
) -> float:
    """Correlate two vectors with the requested correlation method."""
    x = np.asarray(x)
    y = np.asarray(y)
    if method == "spearman":
        r, _ = stats.spearmanr(x, y)
    elif method in {"pearson", "correlation"}:
        r, _ = stats.pearsonr(x, y)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return float(r)


def compute_rsa_score(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    method: CorrelationMethod = "spearman",
) -> float:
    """Compute RSA as a correlation between upper-triangular RDM vectors."""
    return correlate_vectors(rdm_to_vector(rdm1), rdm_to_vector(rdm2), method=method)


def _device_key(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}"
    return str(device)


def _triu_indices_cached(
    n_images: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(n_images), _device_key(device))
    cached = _TRIU_INDEX_CACHE.get(key)
    if cached is None:
        idx = torch.triu_indices(n_images, n_images, offset=1, device=device)
        cached = (idx[0], idx[1])
        _TRIU_INDEX_CACHE[key] = cached
    return cached


def get_rdm_vector(activations, metric: RDMMetric = "euclidean"):
    """
    Calculate an RDM vector with PyTorch.

    Supports features shaped ``[N, D]`` and batched features shaped
    ``[B, N, D]``. Returns the upper triangle, excluding the diagonal.
    """
    if activations.dim() == 2:
        if activations.shape[0] < 2:
            return torch.tensor([], device=activations.device, dtype=torch.float32)

        if metric == "cosine":
            norm = torch.norm(activations.float(), p=2, dim=1, keepdim=True)
            activations_norm = activations.float() / (norm + EPSILON)
            similarity_matrix = torch.matmul(activations_norm, activations_norm.t())
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
            indices = _triu_indices_cached(rdm_matrix.shape[0], activations.device)
            return rdm_matrix[indices[0], indices[1]]

        if metric == "correlation":
            mean_val = torch.mean(activations.float(), dim=1, keepdim=True)
            std_dev = torch.std(activations.float(), dim=1, keepdim=True)
            activations_norm = (activations.float() - mean_val) / (std_dev + EPSILON)
            similarity_matrix = (
                torch.matmul(activations_norm, activations_norm.t())
                / activations.shape[1]
            )
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
            indices = _triu_indices_cached(rdm_matrix.shape[0], activations.device)
            return rdm_matrix[indices[0], indices[1]]

        return torch.pdist(activations.float(), p=2)

    if activations.dim() == 3:
        batch_size, n_images, n_features = activations.shape
        if n_images < 2:
            return torch.zeros(
                (batch_size, 0), device=activations.device, dtype=torch.float32
            )

        if metric == "cosine":
            norm = torch.norm(activations.float(), p=2, dim=2, keepdim=True)
            activations_norm = activations.float() / (norm + EPSILON)
            similarity_matrix = torch.bmm(
                activations_norm, activations_norm.transpose(1, 2)
            )
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
        elif metric == "correlation":
            mean_val = torch.mean(activations.float(), dim=2, keepdim=True)
            std_dev = torch.std(activations.float(), dim=2, keepdim=True)
            activations_norm = (activations.float() - mean_val) / (std_dev + EPSILON)
            similarity_matrix = (
                torch.bmm(activations_norm, activations_norm.transpose(1, 2))
                / n_features
            )
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
        else:
            rdm_matrix = torch.cdist(activations.float(), activations.float(), p=2)

        indices = _triu_indices_cached(n_images, activations.device)
        return rdm_matrix[:, indices[0], indices[1]]

    raise ValueError(f"Expected 2D or 3D input, got {activations.shape}")


def rank_standardize_batch(vectors: torch.Tensor) -> torch.Tensor:
    """Rank-standardize batched vectors using ordinal ranks."""
    values = vectors.float()
    order = torch.argsort(values, dim=-1)
    ranks = torch.empty_like(values, dtype=torch.float32)
    base = torch.arange(
        values.shape[-1],
        device=values.device,
        dtype=torch.float32,
    ).expand_as(values)
    ranks.scatter_(dim=-1, index=order, src=base)
    return (ranks - ranks.mean(dim=-1, keepdim=True)) / (
        ranks.std(dim=-1, unbiased=False, keepdim=True) + EPSILON
    )


def vector_standardize_batch(vectors: torch.Tensor) -> torch.Tensor:
    values = vectors.float()
    return (values - values.mean(dim=-1, keepdim=True)) / (
        values.std(dim=-1, unbiased=False, keepdim=True) + EPSILON
    )


def prepare_correlation_reference_batch(
    vectors: torch.Tensor,
    corr_type: str = "spearman",
) -> torch.Tensor:
    if corr_type == "spearman":
        return rank_standardize_batch(vectors)
    return vector_standardize_batch(vectors)


def correlate_vector_batches(
    vectors: torch.Tensor,
    reference: torch.Tensor,
    corr_type: str = "spearman",
) -> torch.Tensor:
    """Correlate each row in ``vectors`` with pre-standardized references."""
    if corr_type == "spearman":
        standardized = rank_standardize_batch(vectors)
    else:
        standardized = vector_standardize_batch(vectors)
    return torch.mean(standardized * reference, dim=-1)


def calculate_correlation(vec_A, vec_B, corr_type: str = "correlation"):
    """Calculate Pearson or Spearman correlation using PyTorch."""
    if vec_A.numel() == 0 or vec_B.numel() == 0:
        return torch.tensor(0.0, device=vec_A.device, dtype=torch.float32)

    vec_A_float = vec_A.float()
    vec_B_float = vec_B.float()

    var_A, var_B = torch.var(vec_A_float), torch.var(vec_B_float)
    if var_A < EPSILON and var_B < EPSILON:
        return torch.tensor(1.0, device=vec_A.device, dtype=torch.float32)
    if var_A < EPSILON or var_B < EPSILON:
        return torch.tensor(0.0, device=vec_A.device, dtype=torch.float32)

    if corr_type == "spearman":
        rank_A = torch.argsort(torch.argsort(vec_A_float)).float()
        rank_B = torch.argsort(torch.argsort(vec_B_float)).float()

        rank_A_std = (rank_A - rank_A.mean()) / (
            rank_A.std(unbiased=False) + EPSILON
        )
        rank_B_std = (rank_B - rank_B.mean()) / (
            rank_B.std(unbiased=False) + EPSILON
        )

        r = torch.mean(rank_A_std * rank_B_std)
    else:
        r_matrix = torch.corrcoef(torch.stack([vec_A_float, vec_B_float]))
        r = r_matrix[0, 1]

    return torch.nan_to_num(r, nan=0.0)


def get_rdm_vector_np(activations, metric: RDMMetric = "euclidean", device=None):
    """Compute an RDM vector with ``get_rdm_vector`` and return a CPU NumPy array."""
    if isinstance(activations, torch.Tensor):
        tensor = (
            activations.to(device=device, dtype=torch.float32)
            if device is not None
            else activations.float()
        )
    else:
        tensor = torch.as_tensor(activations, dtype=torch.float32, device=device)
    return get_rdm_vector(tensor, metric=metric).detach().cpu().numpy()


def calculate_correlation_value(
    vec_A,
    vec_B,
    corr_type: str = "spearman",
    device=None,
) -> float:
    """Compute an RDM-vector correlation with ``calculate_correlation`` as a float."""
    if isinstance(vec_A, torch.Tensor):
        a = (
            vec_A.to(device=device, dtype=torch.float32)
            if device is not None
            else vec_A.float()
        )
    else:
        a = torch.as_tensor(vec_A, dtype=torch.float32, device=device)
    if isinstance(vec_B, torch.Tensor):
        b = (
            vec_B.to(device=device, dtype=torch.float32)
            if device is not None
            else vec_B.float()
        )
    else:
        b = torch.as_tensor(vec_B, dtype=torch.float32, device=device)
    return float(
        calculate_correlation(a.reshape(-1), b.reshape(-1), corr_type)
        .detach()
        .cpu()
        .item()
    )


__all__ = [
    "RDMMetric",
    "CorrelationMethod",
    "compute_rdm",
    "compute_rdm_correlation",
    "rdm_to_vector",
    "correlate_vectors",
    "compute_rsa_score",
    "get_rdm_vector",
    "rank_standardize_batch",
    "vector_standardize_batch",
    "prepare_correlation_reference_batch",
    "correlate_vector_batches",
    "calculate_correlation",
    "get_rdm_vector_np",
    "calculate_correlation_value",
]
