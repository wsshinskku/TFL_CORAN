# ue/features.py
# 상태 표준화/필수 유틸
from __future__ import annotations
import numpy as np

class StateNormalizer:
    """Online standardization: x_norm = (x - μ) / (σ + ε)"""
    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-6):
        self.mu = np.zeros(dim, dtype=np.float32)
        self.var = np.ones(dim, dtype=np.float32)
        self.m = momentum
        self.eps = eps
        self._initialized = False

    def update(self, x: np.ndarray):
        x = x.astype(np.float32)
        if not self._initialized:
            self.mu = x.copy()
            self.var = np.ones_like(x, dtype=np.float32)
            self._initialized = True
        else:
            self.mu = (1 - self.m) * self.mu + self.m * x
            self.var = (1 - self.m) * self.var + self.m * (x - self.mu) ** 2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        return (x - self.mu) / (np.sqrt(self.var) + self.eps)
