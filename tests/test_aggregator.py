# tests/test_aggregator.py
import numpy as np
import torch
from ric.fl_server import aggregate_membership_weighted, personalize_by_membership

def _weights(v):
    return {"w": torch.tensor(v, dtype=torch.float32)}

def test_membership_weighted_aggregation_and_personalization():
    # 두 UE, 두 클러스터
    ue_ids = ["u0", "u1"]
    gamma = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # u0->k0, u1->k1
    prev_clusters = [_weights([0.0, 0.0]), _weights([0.0, 0.0])]
    deltas = {
        "u0": _weights([1.0, 2.0]),
        "u1": _weights([3.0, 4.0]),
    }
    # 식 (9): 클러스터별 평균 Δw 후 누적
    new_clusters = aggregate_membership_weighted(deltas, ue_ids, gamma, prev_clusters)
    assert torch.allclose(new_clusters[0]["w"], torch.tensor([1.0, 2.0]))
    assert torch.allclose(new_clusters[1]["w"], torch.tensor([3.0, 4.0]))
    # 식 (10): UE 개인화 재분배
    personalized = personalize_by_membership(new_clusters, ue_ids, gamma)
    assert torch.allclose(personalized["u0"]["w"], torch.tensor([1.0, 2.0]))
    assert torch.allclose(personalized["u1"]["w"], torch.tensor([3.0, 4.0]))
