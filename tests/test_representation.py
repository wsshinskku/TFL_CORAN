from pathlib import Path

import numpy as np
import pytest
import torch

from tfl_coran.clustering import SoftClusterer
from tfl_coran.config import load_config
from tfl_coran.models.vgae import (
    build_knn_graph,
    load_vgae_checkpoint,
    normalize_adjacency,
    save_vgae_checkpoint,
    train_vgae,
    vgae_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def test_knn_graph_is_symmetric_sparse_and_normalized() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(10, 4)).astype(np.float32)
    weighted, binary = build_knn_graph(features, neighbors=3)
    np.testing.assert_allclose(weighted, weighted.T)
    np.testing.assert_array_equal(binary, binary.T)
    assert np.all(np.diag(weighted) == 0.0)
    assert np.all(binary.sum(axis=1) >= 3)
    normalized = normalize_adjacency(weighted)
    assert normalized.shape == (10, 10)
    assert np.all(np.isfinite(normalized))


def test_vgae_and_gmm_produce_finite_memberships(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    snapshots = [rng.normal(loc=index * 0.1, size=(12, 4)).astype(np.float32) for index in range(3)]
    config = load_config(ROOT / "configs" / "smoke.yaml")
    artifacts = train_vgae(snapshots, config["vgae"], torch.device("cpu"), seed=9)
    assert artifacts.losses
    assert all(
        np.isfinite(value)
        for row in artifacts.losses
        for value in row.values()
    )
    first = artifacts.embeddings(snapshots[0], graph_neighbors=2, device=torch.device("cpu"))
    second = artifacts.embeddings(snapshots[0], graph_neighbors=2, device=torch.device("cpu"))
    np.testing.assert_allclose(first, second)
    assert first.shape == (12, 4)
    assert np.all(np.isfinite(first))
    gamma = SoftClusterer(config["gmm"], seed=9).fit_predict(first)
    assert gamma.shape == (12, 3)
    assert np.all(gamma >= 0.0)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0)
    predicted = SoftClusterer(config["gmm"], seed=10)
    predicted.fit_predict(first)
    np.testing.assert_allclose(predicted.predict(first).sum(axis=1), 1.0)
    checkpoint = tmp_path / "vgae.pt"
    save_vgae_checkpoint(artifacts, config["vgae"], checkpoint)
    # Checkpoints must remain compatible with PyTorch's restricted loader; this
    # prevents the public CLI from silently falling back to arbitrary pickle loading.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["input_dim"] == snapshots[0].shape[1]
    loaded = load_vgae_checkpoint(checkpoint, torch.device("cpu"))
    assert loaded.losses == artifacts.losses
    np.testing.assert_allclose(loaded.scaler.mean_, artifacts.scaler.mean_)
    np.testing.assert_allclose(loaded.scaler.scale_, artifacts.scaler.scale_)
    for name, tensor in artifacts.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], tensor)
    np.testing.assert_allclose(
        loaded.embeddings(snapshots[0], 2, torch.device("cpu")), first
    )


def test_vgae_kl_treats_encoder_scale_as_log_standard_deviation() -> None:
    logits = torch.zeros((2, 2), dtype=torch.float32)
    adjacency = torch.zeros((2, 2), dtype=torch.float32)
    mu = torch.zeros((2, 1), dtype=torch.float32)
    log_std = torch.full((2, 1), float(np.log(2.0)), dtype=torch.float32)

    total, reconstruction, kl = vgae_loss(logits, adjacency, mu, log_std, kl_weight=1.0)

    expected_kl = 0.5 * (4.0 - 1.0 - np.log(4.0))
    assert kl.item() == pytest.approx(expected_kl)
    assert torch.isfinite(total)
    assert torch.isfinite(reconstruction)
    assert torch.isfinite(kl)
