import numpy as np
import torch

from tfl_coran.federated.aggregator import aggregate_models


def _state(value: float) -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([value], dtype=torch.float32)}


def test_uniform_membership_reduces_to_fedavg() -> None:
    local = [_state(1.0), _state(3.0)]
    gamma = np.full((2, 2), 0.5)
    result = aggregate_models(local, gamma, _state(0.0))
    assert result.global_model["weight"].item() == 2.0
    assert all(model["weight"].item() == 2.0 for model in result.cluster_models)
    assert all(model["weight"].item() == 2.0 for model in result.personalized_models)


def test_one_hot_membership_matches_hard_cluster_averages() -> None:
    local = [_state(1.0), _state(3.0), _state(10.0), _state(14.0)]
    gamma = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)
    result = aggregate_models(local, gamma, _state(0.0))
    assert result.cluster_models[0]["weight"].item() == 2.0
    assert result.cluster_models[1]["weight"].item() == 12.0
    assert result.global_model["weight"].item() == 7.0
    assert result.personalized_models[0]["weight"].item() == 2.0
    assert result.personalized_models[3]["weight"].item() == 12.0
