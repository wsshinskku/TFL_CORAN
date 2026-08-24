from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn


def build_knn_graph(features: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a symmetric kNN graph with the paper's Eq. (7) edge weights.

    Symmetrizing the directed k-nearest-neighbor relation can make the final
    average degree larger than ``neighbors``.
    """
    x = np.asarray(features, dtype=np.float32)
    n = x.shape[0]
    if n < 2:
        raise ValueError("at least two nodes are required to build a graph")
    k = min(max(1, int(neighbors)), n - 1)
    search = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(x)
    distances, indices = search.kneighbors(x)
    weighted = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].reshape(-1)
    values = 1.0 / (1.0 + distances[:, 1:].reshape(-1))
    weighted[rows, cols] = values.astype(np.float32)
    weighted = np.maximum(weighted, weighted.T)
    np.fill_diagonal(weighted, 0.0)
    binary = (weighted > 0.0).astype(np.float32)
    return weighted, binary


def normalize_adjacency(weighted: np.ndarray, self_loops: bool = True) -> np.ndarray:
    adjacency = np.asarray(weighted, dtype=np.float32).copy()
    if self_loops:
        adjacency += np.eye(adjacency.shape[0], dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.zeros_like(degree)
    positive = degree > 0.0
    inverse_sqrt[positive] = np.power(degree[positive], -0.5)
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return adjacency @ self.linear(features)


class VGAE(nn.Module):
    """Two-layer VGAE encoder and inner-product decoder (paper Eqs. 8-12)."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.hidden = GraphConvolution(input_dim, hidden_dim)
        self.mu = GraphConvolution(hidden_dim, latent_dim)
        self.log_std = GraphConvolution(hidden_dim, latent_dim)

    def encode(self, features: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.hidden(features, adjacency))
        return self.mu(hidden, adjacency), self.log_std(hidden, adjacency)

    def reparameterize(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(log_std)
        return mu + std * torch.randn_like(std)

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self.encode(features, adjacency)
        z = self.reparameterize(mu, log_std)
        logits = z @ z.T
        return logits, mu, log_std, z


def vgae_loss(
    logits: torch.Tensor,
    binary_adjacency: torch.Tensor,
    mu: torch.Tensor,
    log_std: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = binary_adjacency.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=binary_adjacency.device)
    targets = binary_adjacency[mask]
    predictions = logits[mask]
    positives = targets.sum().clamp_min(1.0)
    negatives = (targets.numel() - targets.sum()).clamp_min(1.0)
    pos_weight = (negatives / positives).detach()
    reconstruction = nn.functional.binary_cross_entropy_with_logits(
        predictions, targets, pos_weight=pos_weight
    )
    # The manuscript parameterizes the posterior with log(sigma), rather than
    # the more common log variance.
    kl = -0.5 * torch.mean(1.0 + 2.0 * log_std - mu.square() - torch.exp(2.0 * log_std))
    total = reconstruction + float(kl_weight) * kl
    return total, reconstruction, kl


@dataclass
class VGAEArtifacts:
    model: VGAE
    scaler: StandardScaler
    losses: list[dict[str, float]]

    def embeddings(self, contexts: np.ndarray, graph_neighbors: int, device: torch.device) -> np.ndarray:
        scaled = self.scaler.transform(np.asarray(contexts, dtype=np.float32)).astype(np.float32)
        weighted, _ = build_knn_graph(scaled, graph_neighbors)
        normalized = normalize_adjacency(weighted)
        self.model.eval()
        with torch.no_grad():
            features = torch.as_tensor(scaled, device=device)
            adjacency = torch.as_tensor(normalized, device=device)
            mu, _ = self.model.encode(features, adjacency)
        return mu.cpu().numpy().astype(np.float64)


def train_vgae(
    context_snapshots: list[np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> VGAEArtifacts:
    if not context_snapshots:
        raise ValueError("at least one historical context snapshot is required")
    scaler = StandardScaler().fit(np.concatenate(context_snapshots, axis=0))
    scaled_snapshots = [scaler.transform(snapshot).astype(np.float32) for snapshot in context_snapshots]
    torch.manual_seed(seed)
    model = VGAE(
        input_dim=scaled_snapshots[0].shape[1],
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    losses: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    for epoch in range(int(config["epochs"])):
        model.train()
        epoch_values: list[tuple[float, float, float]] = []
        for index in rng.permutation(len(scaled_snapshots)):
            snapshot = scaled_snapshots[int(index)]
            weighted, binary = build_knn_graph(snapshot, int(config["graph_neighbors"]))
            normalized = normalize_adjacency(weighted)
            features = torch.as_tensor(snapshot, device=device)
            adjacency = torch.as_tensor(normalized, device=device)
            target = torch.as_tensor(binary, device=device)
            logits, mu, log_std, _ = model(features, adjacency)
            total, reconstruction, kl = vgae_loss(
                logits, target, mu, log_std, float(config["kl_weight"])
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            epoch_values.append((float(total.item()), float(reconstruction.item()), float(kl.item())))
        mean = np.mean(epoch_values, axis=0)
        losses.append(
            {
                "epoch": epoch + 1,
                "loss": float(mean[0]),
                "reconstruction": float(mean[1]),
                "kl": float(mean[2]),
            }
        )
    model.eval()
    return VGAEArtifacts(model=model, scaler=scaler, losses=losses)


def save_vgae_checkpoint(artifacts: VGAEArtifacts, config: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": artifacts.model.state_dict(),
            "input_dim": artifacts.model.input_dim,
            "hidden_dim": artifacts.model.hidden_dim,
            "latent_dim": artifacts.model.latent_dim,
            "scaler_mean": torch.as_tensor(artifacts.scaler.mean_, dtype=torch.float64),
            "scaler_scale": torch.as_tensor(artifacts.scaler.scale_, dtype=torch.float64),
            "scaler_var": torch.as_tensor(artifacts.scaler.var_, dtype=torch.float64),
            "n_features_in": int(artifacts.scaler.n_features_in_),
            "config": config,
            "losses": [
                {
                    "epoch": int(row["epoch"]),
                    "loss": float(row["loss"]),
                    "reconstruction": float(row["reconstruction"]),
                    "kl": float(row["kl"]),
                }
                for row in artifacts.losses
            ],
        },
        output,
    )


def load_vgae_checkpoint(path: str | Path, device: torch.device) -> VGAEArtifacts:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    model = VGAE(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"].detach().cpu().numpy().astype(np.float64)
    scaler.scale_ = checkpoint["scaler_scale"].detach().cpu().numpy().astype(np.float64)
    scaler.var_ = checkpoint["scaler_var"].detach().cpu().numpy().astype(np.float64)
    scaler.n_features_in_ = int(checkpoint["n_features_in"])
    return VGAEArtifacts(model=model, scaler=scaler, losses=list(checkpoint.get("losses", [])))
