# ue/replay_buffer.py
# Experience replay
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from collections import deque

@dataclass
class Experience:
    s: np.ndarray
    a: int
    r: float
    s_next: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, seed: int = 42):
        self.buf: deque[Experience] = deque(maxlen=capacity)
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.buf)

    def add(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool):
        self.buf.append(Experience(s.copy(), int(a), float(r), s_next.copy(), bool(done)))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.choice(len(self.buf), size=batch_size, replace=False)
        S, A, R, SN, D = [], [], [], [], []
        for i in idx:
            e = self.buf[i]
            S.append(e.s); A.append(e.a); R.append(e.r); SN.append(e.s_next); D.append(e.done)
        return (np.stack(S), np.asarray(A, dtype=np.int64), np.asarray(R, dtype=np.float32),
                np.stack(SN), np.asarray(D, dtype=np.float32))
