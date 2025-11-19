# ric/transfer.py
# Warm-start: w_i^0 = δ w_prev_i + (1-δ) w_{j*}, j* = argmax_j cos(x_i, x_j)
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import torch

Weights = Dict[str, torch.Tensor]

def _clone(w: Weights) -> Weights:
    return {k: v.clone() for k, v in w.items()}

def _fuse(w1: Optional[Weights], w2: Weights, delta: float) -> Weights:
    if w1 is None:
        return _clone(w2)
    fused = {}
    for k in w2.keys():
        fused[k] = float(delta) * w1[k] + float(1.0 - delta) * w2[k]
    return fused

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na*nb))

def warm_start_from_neighbors(
    ue_id: str,
    x_i: np.ndarray,                          # SITM of i
    neighbor_feats: Dict[str, np.ndarray],    # {ue_j: x_j}
    personalized_models: Dict[str, Weights],  # latest personalized weights for neighbors
    prev_w_i: Optional[Weights],
    delta: float = 0.5
) -> Weights:
    """
    Choose j* with max cosine(x_i, x_j) among available neighbors, then fuse.
    """
    best_j, best_s = None, -1.0
    for j, xj in neighbor_feats.items():
        if j == ue_id or j not in personalized_models:
            continue
        s = _cos_sim(x_i, xj)
        if s > best_s:
            best_s, best_j = s, j
    if best_j is None:
        # fallback: average of all cluster models (approx by random neighbor)
        any_w = next(iter(personalized_models.values()))
        return _fuse(prev_w_i, any_w, delta)
    return _fuse(prev_w_i, personalized_models[best_j], delta)
