from __future__ import annotations

import torch

EPSILON = 1e-9

def get_rdm_vector(activations, metric="euclidean"):
    """
    Calculates the RDM vector (upper triangle, excluding diagonal) using PyTorch.
    Supports both single input [N, D] and batched input [B, N, D].

    Args:
        activations (torch.Tensor): Features (n_samples, n_features) or (batch, n_samples, n_features) on GPU.
        metric (str): Distance metric ('euclidean', 'cosine', etc.). torch.pdist supports fewer than scipy.

    Returns:
        torch.Tensor: RDM vector on GPU (float32).
                      If input is [N, D], returns [P].
                      If input is [B, N, D], returns [B, P].
    """
    if activations.dim() == 2:
        # Standard unbatched case
        if activations.shape[0] < 2:
            return torch.tensor([], device=activations.device, dtype=torch.float32)
        
        if metric == "cosine":
            norm = torch.norm(activations.float(), p=2, dim=1, keepdim=True)
            activations_norm = activations.float() / (norm + 1e-9)
            similarity_matrix = torch.matmul(activations_norm, activations_norm.t())
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
            indices = torch.triu_indices(rdm_matrix.shape[0], rdm_matrix.shape[1], offset=1)
            return rdm_matrix[indices[0], indices[1]]
            
        elif metric == "correlation":
            mean_val = torch.mean(activations.float(), dim=1, keepdim=True)
            std_dev = torch.std(activations.float(), dim=1, keepdim=True)
            activations_norm = (activations.float() - mean_val) / (std_dev + 1e-9)
            similarity_matrix = (
                torch.matmul(activations_norm, activations_norm.t()) / activations.shape[1]
            )
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
            indices = torch.triu_indices(rdm_matrix.shape[0], rdm_matrix.shape[1], offset=1)
            return rdm_matrix[indices[0], indices[1]]
            
        else:  # Assume euclidean or other torch.pdist compatible metric
            return torch.pdist(activations.float(), p=2)

    elif activations.dim() == 3:
        # Batched case [B, N, D]
        B, N, D = activations.shape
        if N < 2:
            return torch.zeros((B, 0), device=activations.device, dtype=torch.float32)
            
        if metric == "cosine":
            norm = torch.norm(activations.float(), p=2, dim=2, keepdim=True) # [B, N, 1]
            activations_norm = activations.float() / (norm + 1e-9)
            # bmm: [B, N, D] x [B, D, N] -> [B, N, N]
            similarity_matrix = torch.bmm(activations_norm, activations_norm.transpose(1, 2))
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
            
        elif metric == "correlation":
            mean_val = torch.mean(activations.float(), dim=2, keepdim=True)
            std_dev = torch.std(activations.float(), dim=2, keepdim=True)
            activations_norm = (activations.float() - mean_val) / (std_dev + 1e-9)
            similarity_matrix = (
                torch.bmm(activations_norm, activations_norm.transpose(1, 2)) / D
            )
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            rdm_matrix = 1.0 - similarity_matrix
            
        else: # Euclidean
            # cdist supports batching: [B, N, D] -> [B, N, N]
            rdm_matrix = torch.cdist(activations.float(), activations.float(), p=2)
            
        # Extract upper triangle for each batch
        # triu_indices returns [2, P]
        indices = torch.triu_indices(N, N, offset=1, device=activations.device)
        # We can gather from [B, N, N] using these indices
        # rdm_matrix[:, indices[0], indices[1]] -> [B, P]
        return rdm_matrix[:, indices[0], indices[1]]

    else:
        raise ValueError(f"Expected 2D or 3D input, got {activations.shape}")

def calculate_correlation(vec_A, vec_B, corr_type="correlation"):
    """
    Calculates Pearson or Spearman correlation using PyTorch.

    Args:
        vec_A (torch.Tensor): First vector (GPU).
        vec_B (torch.Tensor): Second vector (GPU).
        corr_type (str): 'correlation' (Pearson) or 'spearman'.

    Returns:
        torch.Tensor: Correlation coefficient (scalar tensor on GPU).
    """
    if vec_A.numel() == 0 or vec_B.numel() == 0:
        return torch.tensor(0.0, device=vec_A.device, dtype=torch.float32)

    # Ensure inputs are float32 for variance and correlation calculations
    vec_A_float = vec_A.float()
    vec_B_float = vec_B.float()

    var_A, var_B = torch.var(vec_A_float), torch.var(vec_B_float)
    # Use a small epsilon to prevent division by zero if variance is exactly zero
    if var_A < EPSILON and var_B < EPSILON:
        return torch.tensor(1.0, device=vec_A.device, dtype=torch.float32)
    if var_A < EPSILON or var_B < EPSILON:
        return torch.tensor(0.0, device=vec_A.device, dtype=torch.float32)

    if corr_type == "spearman":
        # Calculate Spearman using ranks
        rank_A = torch.argsort(torch.argsort(vec_A_float)).float()
        rank_B = torch.argsort(torch.argsort(vec_B_float)).float()

        # Standardize the ranks
        rank_A_std = (rank_A - rank_A.mean()) / (rank_A.std(unbiased=False) + EPSILON)
        rank_B_std = (rank_B - rank_B.mean()) / (rank_B.std(unbiased=False) + EPSILON)

        # Pearson correlation of standardized ranks
        r = torch.mean(rank_A_std * rank_B_std)
    else:  # Pearson correlation
        # Using torch.corrcoef requires stacking
        r_matrix = torch.corrcoef(torch.stack([vec_A_float, vec_B_float]))
        r = r_matrix[0, 1]

    return torch.nan_to_num(r, nan=0.0)