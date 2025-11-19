# ric/topology.py
# Build similarity graph A_ij = 1/(1+||x_i - x_j||^2)
from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np

def build_features_matrix(sitm: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """
    sitm: {ue_id: x_i in R^4} with x_i = [S,I,T,M]
    returns (X[N,4], ue_ids[list])
    """
    ue_ids = list(sitm.keys())
    X = np.stack([sitm[uid].astype(np.float32) for uid in ue_ids], axis=0)
    return X, ue_ids

def build_similarity_adjacency(X: np.ndarray, self_loop: bool=True, sparsify_eps: float=1e-3) -> np.ndarray:
    """
    Compute dense A with optional light sparsification for stability.
    A_ij = 1 / (1 + ||x_i - x_j||^2)     # Eq.(1)
    """
    # pairwise squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i x_j^T
    norm2 = np.sum(X*X, axis=1, keepdims=True)  # [N,1]
    d2 = norm2 + norm2.T - 2.0 * (X @ X.T)
    A = 1.0 / (1.0 + np.maximum(d2, 0.0))
    if not self_loop:
        np.fill_diagonal(A, 0.0)
    # small-value sparsification
    if sparsify_eps > 0:
        A[A < sparsify_eps] = 0.0
    return A.astype(np.float32)

def normalize_adj(A: np.ndarray, add_self_loop: bool=True) -> np.ndarray:
    """
    Symmetric normalization:  D^{-1/2} (A + I) D^{-1/2}
    Used by GCN encoder (Eq.(2) aggregation term), p.3.  :contentReference[oaicite:8]{index=8}
    """
    N = A.shape[0]
    if add_self_loop:
        A = A + np.eye(N, dtype=A.dtype)
    d = np.sum(A, axis=1)
    d_inv_sqrt = np.power(np.maximum(d, 1e-12), -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)
