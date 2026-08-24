import numpy as np
import pytest
import torch

from tfl_coran.transfer import transfer_initialize


def _state(value: float) -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([value], dtype=torch.float32)}


def test_transfer_excludes_self_and_handles_new_ue() -> None:
    contexts = np.array([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.01, 1.0]])
    cells = np.array([0, 0, 1, 1])
    current = [_state(0), _state(10), _state(20), _state(30)]
    previous = [_state(4), _state(11), _state(22), _state(33)]
    initialized, neighbors = transfer_initialize(
        contexts,
        cells,
        current,
        previous,
        handover_indices=[0],
        new_ue_indices=[2],
        delta=0.25,
    )
    assert neighbors[0] == 1
    assert initialized[0]["weight"].item() == pytest.approx(8.5)
    assert neighbors[2] == 3
    assert initialized[2]["weight"].item() == 30.0
