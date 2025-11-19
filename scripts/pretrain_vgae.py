# scripts/pretrain_vgae.py
# VGAE 오프라인 학습: GCN 인코더 → μ, logσ → reparam z, 디코더 σ(z z^T)로 A 복원, ELBO 최소화
# 데이터: data/historical/*.npz (X[N,4], ue_ids[]) 또는 env에서 합성 수집
from __future__ import annotations
import argparse, os, glob, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ric.vgae import VGAEEncoder
from ric.topology import build_similarity_adjacency, normalize_adj
from data.historical_dataset import collect_snapshots_from_env

def bce_recon_loss_from_logits(logits: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    # logits: [N,N], A: [N,N] in {0,1}
    return F.binary_cross_entropy_with_logits(logits, A, reduction="mean")

def vgae_elbo(z, mu, logvar, A) -> torch.Tensor:
    # Decoder: σ(z z^T), Recon + KL(q||N(0,I))
    logits = z @ z.t()
    recon = bce_recon_loss_from_logits(logits, A)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl

def load_snapshots_npz(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    for f in files:
        dat = np.load(f, allow_pickle=True)
        X = dat["X"].astype(np.float32)                 # [N,4]
        # A는 매 스냅샷마다 SITM으로 재계산
        A = build_similarity_adjacency(X, self_loop=True)
        A = (A > 0).astype(np.float32)                 # 이진화(간소화)
        yield X, A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/checkpoints/vgae_encoder.pt")
    ap.add_argument("--snapshots", type=str, default="data/historical")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gen-if-missing", action="store_true",
                    help="스냅샷 폴더가 비어있으면 env에서 합성 수집")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 스냅샷 로드 또는 합성 생성
    snapshots_iter = list(load_snapshots_npz(args.snapshots))
    if len(snapshots_iter) == 0 and args.gen_if_missing:
        ds = collect_snapshots_from_env(num_snapshots=20, steps_between=200) 
        os.makedirs(args.snapshots, exist_ok=True)
        # 저장
        for i in range(len(ds)):
            X, A_hat = ds[i]
            A = (A_hat.numpy() > 0).astype(np.float32)
            np.savez(os.path.join(args.snapshots, f"snapshot_{i:04d}.npz"), ue_ids=np.arange(X.shape[0]), X=X.numpy())
        snapshots_iter = list(load_snapshots_npz(args.snapshots))

    assert len(snapshots_iter) > 0, f"No snapshots in {args.snapshots}. 먼저 data/generate_historical.py를 실행하세요."

    # VGAE 인코더
    enc = VGAEEncoder(in_dim=4, gcn_dims=(64,32), latent_dim=32).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        elbo_sum = recon_sum = kl_sum = n_graph = 0
        for X_np, A_np in snapshots_iter:
            X = torch.from_numpy(X_np).to(device)
            A = torch.from_numpy(A_np).to(device)

            # 정규화 인접행렬 A_hat 
            A_hat_np = normalize_adj(A_np, add_self_loop=True)
            A_hat = torch.from_numpy(A_hat_np).to(device)

            z, mu, logvar = enc(X, A_hat, reparam=True)
            loss, recon, kl = vgae_elbo(z, mu, logvar, A)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=5.0)
            opt.step()

            elbo_sum += float(loss.item()); recon_sum += float(recon.item()); kl_sum += float(kl.item()); n_graph += 1

        print(f"[epoch {ep:03d}] ELBO={elbo_sum/n_graph:.4f} | recon={recon_sum/n_graph:.4f} | KL={kl_sum/n_graph:.4f}")

    torch.save(enc.state_dict(), args.out)
    print(f"[OK] Saved VGAE encoder to {args.out}")

if __name__ == "__main__":
    main()
