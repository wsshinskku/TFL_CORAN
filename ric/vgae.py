# ric/vgae.py
# Offline-trained VGAE encoder; online inference only
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # h^{(l)} = ρ( A_hat @ h^{(l-1)} @ W^{(l)} )   (Eq.(2))
        return F.relu(A_hat @ self.lin(X))

class VGAEEncoder(nn.Module):
    """
    Two-layer GCN encoder → μ, logσ; z = μ + σ ⊙ ε (reparam), Eq.(3).
    Decoder is inner product (used only for pretraining; not needed online).
    """
    def __init__(self, in_dim: int=4, gcn_dims=(64,32), latent_dim: int=32):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, gcn_dims[0])
        self.gcn2 = GCNLayer(gcn_dims[0], gcn_dims[1])
        self.mu = nn.Linear(gcn_dims[1], latent_dim)
        self.logvar = nn.Linear(gcn_dims[1], latent_dim)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.logvar.weight)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor, reparam: bool=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.gcn1(X, A_hat)
        h = self.gcn2(h, A_hat)
        mu = self.mu(h)
        logvar = self.logvar(h)
        if not reparam:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps  # Eq.(3)
        return z, mu, logvar

    @torch.no_grad()
    def infer(self, X: np.ndarray, A_hat: np.ndarray, device: str="cpu") -> np.ndarray:
        self.eval()
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        At = torch.as_tensor(A_hat, dtype=torch.float32, device=device)
        z, _, _ = self.forward(Xt, At, reparam=False)
        return z.detach().cpu().numpy()

def infer_embeddings(encoder: VGAEEncoder, X: np.ndarray, A_hat: np.ndarray, device: str="cpu") -> np.ndarray:
    """
    Online inference of z_i = f_VGAE(x_i; Θ*). Encoder is assumed pre-trained and frozen (p.3).  :contentReference[oaicite:10]{index=10}
    """
    return encoder.infer(X, A_hat, device=device)
