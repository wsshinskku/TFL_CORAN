# tests/test_transfer.py
import numpy as np
import torch
from ric.transfer import warm_start_from_neighbors

def _w(v): return {"w": torch.tensor(v, dtype=torch.float32)}

def test_transfer_warm_start_chooses_most_similar_and_fuses():
    x_i = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    neighbor_feats = {
        "a": np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32),
        "b": np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    }
    personalized = {"a": _w([1.0, 1.0]), "b": _w([9.0, 9.0])}
    prev = _w([0.0, 0.0])
    fused = warm_start_from_neighbors("i", x_i, neighbor_feats, personalized, prev, delta=0.5)
    # 가장 유사한 이웃은 'a'이고, δ=0.5 혼합 → 0.5 * prev + 0.5 * w_a = [0.5, 0.5]
    assert torch.allclose(fused["w"], torch.tensor([0.5, 0.5]))
