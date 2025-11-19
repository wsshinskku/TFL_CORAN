# tests/test_vgae.py
import numpy as np
import torch
from ric.vgae import VGAEEncoder
from ric.topology import build_similarity_adjacency, normalize_adj
import torch.nn.functional as F

def _elbo(z, mu, logvar, A):
    logits = z @ z.t()
    recon = F.binary_cross_entropy_with_logits(logits, A, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

def test_vgae_shapes_and_inference_cpu():
    N = 30
    X = torch.randn(N, 4)
    A = build_similarity_adjacency(X.numpy(), self_loop=True)
    A_bin = torch.from_numpy((A > 0).astype(np.float32))
    Ahat = torch.from_numpy(normalize_adj(A, add_self_loop=True))

    enc = VGAEEncoder(in_dim=4, gcn_dims=(16, 8), latent_dim=8)  # 경량 설정
    z, mu, logvar = enc(X, Ahat, reparam=True)
    assert z.shape == (N, 8) and mu.shape == (N, 8) and logvar.shape == (N, 8)

    # 간단 학습으로 ELBO 감소 확인 (빠른 수렴)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    enc.train()
    with torch.enable_grad():
        loss0 = None
        for _ in range(30):
            z, mu, logvar = enc(X, Ahat, reparam=True)
            loss = _elbo(z, mu, logvar, A_bin)
            opt.zero_grad(); loss.backward(); opt.step()
            if loss0 is None: loss0 = float(loss.item())
        lossT = float(loss.item())
    assert lossT < loss0 * 0.95  # 최소 5% 이상 감소
