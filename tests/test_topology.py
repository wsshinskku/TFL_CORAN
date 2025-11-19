# tests/test_topology.py
import numpy as np
from ric.topology import build_similarity_adjacency, normalize_adj

def test_similarity_adjacency_symmetry_and_range():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 4).astype(np.float32)  # SITM: [S,I,T,M]
    A = build_similarity_adjacency(X, self_loop=True)
    # 대칭 / 대각 원소 / 범위 검증
    assert A.shape == (20, 20)
    assert np.allclose(A, A.T, atol=1e-6)
    assert np.allclose(np.diag(A), 1.0, atol=1e-6)
    assert (A >= 0.0).all() and (A <= 1.0).all()

def test_normalize_adj_basic_properties():
    rng = np.random.RandomState(1)
    X = rng.rand(10, 4).astype(np.float32)
    A = build_similarity_adjacency(X, self_loop=True)
    Ahat = normalize_adj(A, add_self_loop=True)
    # 정규화 후 여전히 대칭이며 수치적으로 안정
    assert np.allclose(Ahat, Ahat.T, atol=1e-6)
    # 희소하지 않더라도 대각 성분은 양수
    assert (np.diag(Ahat) > 0).all()
