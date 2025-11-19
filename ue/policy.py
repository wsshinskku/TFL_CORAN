# ue/policy.py
# ε-greedy 스케줄러
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class EpsilonScheduler:
    start: float = 1.0
    end: float = 0.01
    decay_episodes: int = 100  # 몇 에피소드에 걸쳐 선형 감소

    def value(self, episode_idx: int) -> float:
        if self.decay_episodes <= 0:
            return self.end
        frac = max(0.0, min(1.0, 1.0 - episode_idx / float(self.decay_episodes)))
        return self.end + (self.start - self.end) * frac
