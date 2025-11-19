# ric/fl_server.py
# Membership-weighted aggregation (Eq.(9)) and personalized redistribution (Eq.(10)), p.4.  :contentReference[oaicite:13]{index=13}
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch

Weights = Dict[str, torch.Tensor]

def _zeros_like(w: Weights) -> Weights:
    return {k: torch.zeros_like(v) for k, v in w.items()}

def _axpy_(dst: Weights, src: Weights, alpha: float):
    for k in dst.keys():
        dst[k] += float(alpha) * src[k]

def _scale_(w: Weights, alpha: float):
    for k in w.keys():
        w[k] *= float(alpha)

def _clone(w: Weights) -> Weights:
    return {k: v.clone() for k, v in w.items()}

def aggregate_membership_weighted(
    deltas: Dict[str, Weights],                 # {ue_id: Δw_i}
    ue_ids: List[str],                          # row order for gamma
    gamma: np.ndarray,                          # [N,K], γ_ik
    prev_cluster_models: List[Weights],         # cluster baselines \bar w^{r-1}_k
) -> List[Weights]:
    """
    Compute Δ\bar w_k = (Σ_i γ_ik Δw_i) / (Σ_i γ_ik); then \bar w^r_k = \bar w^{r-1}_k + Δ\bar w_k   (Eq.(9)).
    """
    N, K = gamma.shape
    assert len(ue_ids) == N
    sums: List[Weights] = [_zeros_like(prev_cluster_models[0]) for _ in range(K)]
    denom = np.zeros(K, dtype=np.float64)

    for row, ue in enumerate(ue_ids):
        if ue not in deltas:   # no update uploaded
            continue
        for k in range(K):
            w = float(gamma[row, k])
            if w == 0.0:
                continue
            _axpy_(sums[k], deltas[ue], alpha=w)
            denom[k] += w

    out: List[Weights] = []
    for k in range(K):
        if denom[k] > 0:
            _scale_(sums[k], alpha=1.0 / denom[k])  # Δ\bar w_k
        # \bar w^r_k = \bar w^{r-1}_k + Δ\bar w_k
        newk = _clone(prev_cluster_models[k])
        _axpy_(newk, sums[k], alpha=1.0)
        out.append(newk)
    return out

def personalize_by_membership(cluster_models: List[Weights], ue_ids: List[str], gamma: np.ndarray) -> Dict[str, Weights]:
    """
    w^{r+1}_i = Σ_k γ_ik \bar w^r_k   (Eq.(10)).
    """
    K = len(cluster_models)
    out: Dict[str, Weights] = {}
    for row, ue in enumerate(ue_ids):
        w_i = _zeros_like(cluster_models[0])
        for k in range(K):
            _axpy_(w_i, cluster_models[k], alpha=float(gamma[row, k]))
        out[ue] = w_i
    return out
