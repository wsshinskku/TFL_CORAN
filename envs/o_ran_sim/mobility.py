# envs/o_ran_sim/mobility.py
# Mobility model p(t+1)=p(t)+v(t)d(t), Voronoi handovers (Sec. II). :contentReference[oaicite:3]{index=3}
from __future__ import annotations
import numpy as np
from typing import Tuple, List

def hexagonal_layout(cells: int, isd_m: int) -> np.ndarray:
    """
    Place gNBs on a simple 3-cell triangle (extendable).
    For cells>3, place them on a hex grid around origin.
    """
    if cells <= 3:
        # triangle vertices
        p0 = np.array([0.0, 0.0], dtype=np.float32)
        p1 = np.array([isd_m, 0.0], dtype=np.float32)
        p2 = np.array([0.5 * isd_m, (np.sqrt(3)/2) * isd_m], dtype=np.float32)
        return np.stack([p0, p1, p2], axis=0)[:cells]
    # basic hex ring for more cells
    pts = [np.array([0.0, 0.0], dtype=np.float32)]
    r = isd_m
    for k in range(6):
        ang = np.deg2rad(60*k)
        pts.append(np.array([r*np.cos(ang), r*np.sin(ang)], dtype=np.float32))
    return np.stack(pts[:cells], axis=0)

class MobilityModel:
    def __init__(self, bounds_xy: Tuple[np.ndarray, np.ndarray], rng: np.random.RandomState):
        self.bounds_min, self.bounds_max = bounds_xy
        self.rng = rng

    def spawn_near(self, center_xy: np.ndarray, radius: float) -> np.ndarray:
        r = self.rng.rand() * radius
        th = self.rng.rand() * 2*np.pi
        return center_xy + np.array([r*np.cos(th), r*np.sin(th)], dtype=np.float32)

    def sample_speed(self) -> float:
        # 0~7 m/s (walk~bike), mean ≈3.5
        return float(self.rng.uniform(0.0, 7.0))

    def sample_heading(self) -> float:
        return float(self.rng.uniform(-np.pi, np.pi))

    def mobility_index(self, speed_mps: float) -> float:
        # normalize by ~7 m/s
        return float(np.clip(speed_mps / 7.0, 0.0, 1.0))

    def step(self, pos_xy: np.ndarray, speed_mps: float, heading_rad: float) -> Tuple[np.ndarray, float]:
        # small random turns / speed jitter
        heading_rad += float(self.rng.normal(scale=np.deg2rad(5)))
        speed_mps = float(np.clip(speed_mps + self.rng.normal(scale=0.1), 0.0, 10.0))
        pos_xy = pos_xy + speed_mps * np.array([np.cos(heading_rad), np.sin(heading_rad)], dtype=np.float32) * 1.0  # 1s slot
        # reflect at bounds
        for i in (0, 1):
            if pos_xy[i] < self.bounds_min[i]:
                pos_xy[i] = self.bounds_min[i] + (self.bounds_min[i] - pos_xy[i])
                heading_rad = -heading_rad if i == 1 else np.pi - heading_rad
            if pos_xy[i] > self.bounds_max[i]:
                pos_xy[i] = self.bounds_max[i] - (pos_xy[i] - self.bounds_max[i])
                heading_rad = -heading_rad if i == 1 else np.pi - heading_rad
        return pos_xy, heading_rad
