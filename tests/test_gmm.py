# tests/test_gmm.py
import numpy as np
from ric.gmm import SoftGMM

def test_gmm_gamma_rows_sum_to_one_and_separates_clusters():
    rng = np.random.RandomState(0)
    # 두 개의 명확한 클러스터
    Z1 = rng.randn(50, 4) * 0.3 + np.array([2, 0, 0, 0])
    Z2 = rng.randn(50, 4) * 0.3 + np.array([-2, 0, 0, 0])
    Z = np.vstack([Z1, Z2]).astype(np.float32)

    gmm = SoftGMM(n_components=2, max_iter=200, tol=1e-4, covariance_type="full", rng=0)
    gamma = gmm.fit_predict(Z)
    assert gamma.shape == (100, 2)
    # 각 행의 합은 1
    rowsum = np.abs(gamma.sum(axis=1) - 1.0).max()
    assert rowsum < 1e-5
    # 클러스터 분리: 주된 책임도가 0.5를 초과하는 샘플 비율이 충분히 큼
    hard = np.argmax(gamma, axis=1)
    dominant_prob = np.max(gamma, axis=1)
    assert (dominant_prob > 0.5).mean() > 0.8
