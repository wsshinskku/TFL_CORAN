# envs/o_ran_sim/traffic.py
# Traffic generator per service class
from __future__ import annotations
from typing import Dict
import numpy as np

class TrafficGenerator:
    """
    Poisson-like arrivals per slot (1 ms).
    Mean arrivals ≈ (0.7~0.9) * R_tar to keep queues non-trivial.
    """
    def __init__(self, qos_targets: Dict[str, tuple], rng: np.random.RandomState):
        self.qos_targets = qos_targets
        self.rng = rng
        # scaling factors per class (slightly below target rate)
        self.scale = {"eMBB": 0.85, "URLLC": 0.9, "mMTC": 0.8}

    def step(self, ue_states) -> Dict[str, int]:
        arrivals = {}
        for ue_id, ue in ue_states.items():
            R_tar, _ = self.qos_targets[ue.service]
            lam = self.scale[ue.service] * R_tar * 0.001  # bits per 1ms
            # Poisson-like with overdispersion via Gamma-Poisson mixture
            rate = self.rng.gamma(shape=2.0, scale=lam/2.0)
            bits = int(max(0.0, self.rng.poisson(rate)))
            arrivals[ue_id] = bits
        return arrivals
