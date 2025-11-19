# envs/o_ran_sim/base_env.py
# Multi-agent Open RAN simulator env (UE/gNB/RIC view)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from collections import deque
import math

from .traffic import TrafficGenerator
from .mobility import MobilityModel, hexagonal_layout
from .channel import ChannelModel, McsDecision
from .scheduler import UtilityScheduler, ScheduledTx

# ---- UE state -----------------------------------------------------------------------------------

SERVICE_TO_IDX = {"eMBB": 0, "URLLC": 1, "mMTC": 2}
IDX_TO_SERVICE = {v: k for k, v in SERVICE_TO_IDX.items()}

@dataclass
class UEState:
    ue_id: str
    cell_id: int
    service: str                  # {"eMBB","URLLC","mMTC"}
    pos_xy: np.ndarray            # shape (2,), meters
    speed_mps: float              # speed
    heading_rad: float            # direction
    queue_bits: int = 0
    # rolling stats for obs/reward
    throughput_ma_bits: float = 0.0
    latency_ma_s: float = 0.0
    reliab_ma: float = 1.0
    # last-step instantaneous values (for logging)
    last_bits_tx: int = 0
    last_latency_s: float = 0.0
    last_success: float = 1.0
    # mobility intensity proxy M_i(t) in [0,1]
    mobility_idx: float = 0.0

# ---- ENV ----------------------------------------------------------------------------------------

class ORanSimEnv:
    """
    Multi-agent slot-level simulator matching the paper's protocol:
      - Episode: L slots (Table III: L=200)
      - FL: every τ_E episodes (default 5) at RIC
      - Embedding/Clustering refresh: every τ_F FL rounds (3 or 10)
    UE actions (per slot): a_t = [a^(f), a^(r), a^(p)]
      - a^(f): subband index in [0, N_cf-1]
      - a^(r): MCS index (0..M-1)
      - a^(p): learned service priority (asp) used by scheduler
    Reward (Eq.7): R = r/η_tar + θ/R_tar − l/L_tar
    Observations (per UE): [dist_norm, rate_norm, lat_norm, rel, thr_norm,
                             sinr_pred, svc_idx, mob_idx, subband_load, time_norm]
    """
    def __init__(
        self,
        *,
        cells: int = 3,
        isd_m: int = 500,
        n_subbands: int = 6,
        slot_per_episode: int = 200,
        center_freq_ghz: float = 3.5,
        carrier_mhz: int = 100,
        qos_targets: Dict[str, Tuple[float, float]] = None,  # {svc: (R_tar[bps], L_tar[s])}
        reliability_targets: Dict[str, float] = None,        # {svc: eta_tar}
        seed: int = 42,
        harq_like_retx: bool = True,
        reuse_allowed: bool = True,
        scheduler_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0), # (priority, rate, deficit)
    ):
        self.rng = np.random.RandomState(seed)
        self.cells = cells
        self.isd_m = isd_m
        self.n_subbands = n_subbands
        self.slot_per_episode = slot_per_episode
        self.center_freq_ghz = center_freq_ghz
        self.carrier_mhz = carrier_mhz
        self.harq_like_retx = harq_like_retx
        self.reuse_allowed = reuse_allowed

        # QoS targets (IV-A): eMBB (20Mbps, 15ms) / URLLC (10, 5) / mMTC (5, 10)
        if qos_targets is None:
            qos_targets = {"eMBB": (20e6, 15e-3), "URLLC": (10e6, 5e-3), "mMTC": (5e6, 10e-3)}
        if reliability_targets is None:
            reliability_targets = {"eMBB": 0.99, "URLLC": 0.999, "mMTC": 0.95}
        self.qos_targets = qos_targets
        self.rel_targets = reliability_targets

        # Geometry of cells & gNB anchors (equilateral hex grid)
        self.gnb_xy = hexagonal_layout(cells, isd_m)

        # Subsystems
        self.traffic = TrafficGenerator(qos_targets=self.qos_targets, rng=self.rng)
        self.mobility = MobilityModel(bounds_xy=self._sim_bounds(), rng=self.rng)
        self.channel = ChannelModel(
            center_freq_ghz=center_freq_ghz,
            bandwidth_mhz=carrier_mhz,
            n_subbands=n_subbands,
            reuse_allowed=reuse_allowed,
            rng=self.rng,
        )
        self.scheduler = UtilityScheduler(
            n_subbands=n_subbands, reuse_allowed=reuse_allowed,
            weights=scheduler_weights
        )

        # Runtime
        self.ues: Dict[str, UEState] = {}
        self.time_slot: int = 0
        self._subband_load = np.zeros(n_subbands, dtype=np.int32)  # # contenders per subband

    # --------------------------- Public API ---------------------------

    def seed(self, seed: int):
        self.rng.seed(seed)

    def reset(self, ue_layout: Dict[int, int] = None) -> Dict[str, np.ndarray]:
        """
        ue_layout: {cell_id: num_ues}. Default from Table II counts per cell (150/120/105).
        Returns obs dict per UE.
        """
        if ue_layout is None:
            # Use scaled-down defaults for faster local runs; keep proportions.
            ue_layout = {0: 30, 1: 24, 2: 21}
        self.ues.clear()
        self.time_slot = 0

        # Assign services close to uniform
        services = list(SERVICE_TO_IDX.keys())

        # Initialize users in each cell around its gNB anchor
        for cell_id, num in ue_layout.items():
            for idx in range(num):
                ue_id = f"c{cell_id}_u{idx:03d}"
                svc = services[(idx + cell_id) % len(services)]
                pos = self.mobility.spawn_near(self.gnb_xy[cell_id], radius=self.isd_m/3)
                speed = self.mobility.sample_speed()
                heading = self.mobility.sample_heading()
                self.ues[ue_id] = UEState(
                    ue_id=ue_id, cell_id=cell_id, service=svc,
                    pos_xy=pos, speed_mps=speed, heading_rad=heading,
                    queue_bits=0, throughput_ma_bits=0.0, latency_ma_s=0.0,
                    reliab_ma=1.0, mobility_idx=self.mobility.mobility_index(speed)
                )

        # Warm up with zero traffic/throughput
        return self._build_observations()

    def step(self, actions: Dict[str, Tuple[int, int, int]]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]:
        """
        actions: {ue_id: (subband, mcs, priority)}
        Returns (obs, rewards, dones, info).
        """
        assert len(self.ues) > 0, "Call reset() first."
        self.time_slot += 1

        # 1) Mobility update (p(t+1) = p(t) + v(t) d(t), handovers at Voronoi boundaries)  (Sec. II)
        handovers = self._update_mobility_and_cells()

        # 2) New traffic arrivals (per class) and compute backlog
        arrivals_bits = self.traffic.step(self.ues)  # dict {ue_id: bits}
        for ue_id, bits_in in arrivals_bits.items():
            self.ues[ue_id].queue_bits += int(bits_in)

        # 3) Channel prediction per UE for chosen subbands/MCS (SINR, TB size, BLER)
        #    Also collect per-subband contenders
        contenders: Dict[int, List[str]] = {sb: [] for sb in range(self.n_subbands)}
        decisions: Dict[str, McsDecision] = {}
        for ue_id, ue in self.ues.items():
            sb, mcs, prio = self._normalize_action(actions.get(ue_id, None))
            dec = self.channel.predict_link(
                ue_xy=ue.pos_xy,
                gnb_xy=self.gnb_xy[ue.cell_id],
                mcs_index=mcs,
                subband=sb
            )
            dec.priority = prio
            decisions[ue_id] = dec
            contenders[sb].append(ue_id)

        # 4) Scheduling with reuse (Sec. IV-A; Table III scheduler description)
        scheduled: List[ScheduledTx] = self.scheduler.schedule(
            decisions=decisions,
            contenders=contenders,
            qos_targets=self.qos_targets,
            ue_states=self.ues
        )
        self._subband_load[:] = [len(contenders[sb]) for sb in range(self.n_subbands)]

        # 5) Execute transmissions (HARQ-like): sample success via BLER; serve bits & update queues
        served_bits: Dict[str, int] = {ue_id: 0 for ue_id in self.ues.keys()}
        success_prob: Dict[str, float] = {ue_id: 1.0 for ue_id in self.ues.keys()}

        for tx in scheduled:
            ue = self.ues[tx.ue_id]
            # Interference reuse penalty: degrade SINR if multiple users share this subband
            n_on_sb = len(contenders[tx.subband])
            sinr_db_eff = tx.sinr_db - (0.0 if n_on_sb <= 1 else 3.0 * (n_on_sb - 1))
            tb_bits = self.channel.tb_bits_for_mcs(tx.mcs_index)
            bler = self.channel.bler_from_sinr(tx.mcs_index, sinr_db_eff)

            if self.harq_like_retx:
                succ = (self.channel.rng.rand() > bler)
                bits = tb_bits if succ else 0
                success_prob[ue.ue_id] = 1.0 if succ else 0.0
            else:
                # expected bits
                succ = 1.0 - bler
                bits = int(tb_bits * succ)
                success_prob[ue.ue_id] = float(succ)

            bits = min(bits, ue.queue_bits)  # cannot exceed backlog
            ue.queue_bits -= bits
            ue.last_bits_tx = bits
            ue.last_success = success_prob[ue.ue_id]
            served_bits[ue.ue_id] = bits

        # 6) Update instantaneous latency proxy (queue / target rate) and MA stats
        rewards: Dict[str, float] = {}
        dones: Dict[str, bool] = {}
        for ue_id, ue in self.ues.items():
            R_tar, L_tar = self.qos_targets[ue.service]
            eta_tar = self.rel_targets[ue.service]

            # latency proxy: waiting time if served at R_tar
            lat_s = ue.queue_bits / max(R_tar, 1e-6)
            ue.last_latency_s = lat_s

            # moving averages (simple momentum)
            alpha = 0.2
            ue.throughput_ma_bits = (1 - alpha) * ue.throughput_ma_bits + alpha * served_bits[ue_id]
            ue.latency_ma_s = (1 - alpha) * ue.latency_ma_s + alpha * lat_s
            ue.reliab_ma = (1 - alpha) * ue.reliab_ma + alpha * success_prob[ue_id]

            # Reward (Eq.7): r/eta_tar + θ/R_tar − l/L_tar
            r_now = ue.reliab_ma
            theta_now = (ue.throughput_ma_bits / 0.001)  # bits per second (slot=1ms)
            l_now = ue.latency_ma_s

            rew = (r_now / eta_tar) + (theta_now / R_tar) - (l_now / L_tar)
            rewards[ue_id] = float(rew)

            dones[ue_id] = (self.time_slot >= self.slot_per_episode)

        obs = self._build_observations()
        info = {
            "handovers": handovers,
            "served_bits": served_bits,
            "subband_load": self._subband_load.copy(),
            "time_slot": self.time_slot
        }
        return obs, rewards, dones, info

    # --------------------------- Helper methods ---------------------------

    def _normalize_action(self, a: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        # default: random exploratory action
        if a is None:
            sb = int(self.rng.randint(0, self.n_subbands))
            mcs = int(self.rng.randint(0, self.channel.num_mcs()))
            pr = int(self.rng.randint(0, 3))
            return sb, mcs, pr
        sb, mcs, pr = a
        sb = int(np.clip(sb, 0, self.n_subbands - 1))
        mcs = int(np.clip(mcs, 0, self.channel.num_mcs() - 1))
        pr = int(np.clip(pr, 0, 2))
        return sb, mcs, pr

    def _build_observations(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        # Rough cell radius proxy
        r_cell = self.isd_m / 2
        for ue_id, ue in self.ues.items():
            # features for RL agent (Sec. III-D.1의 상태 변수 요약)
            dist = np.linalg.norm(ue.pos_xy - self.gnb_xy[ue.cell_id])
            dist_norm = np.clip(dist / (r_cell + 1e-6), 0.0, 1.5)
            R_tar, L_tar = self.qos_targets[ue.service]
            thr_norm = (ue.throughput_ma_bits / 0.001) / (R_tar + 1e-9)  # bps / R_tar
            lat_norm = ue.latency_ma_s / (L_tar + 1e-9)
            svc_idx = SERVICE_TO_IDX[ue.service]
            # predicted SINR for current cell center & mid-MCS (proxy observation)
            sinr_pred = self.channel.predict_sinr_only(ue.pos_xy, self.gnb_xy[ue.cell_id])
            subband_load = float(np.mean(self._subband_load) / max(1, len(self._subband_load)))
            time_norm = self.time_slot / max(1, self.slot_per_episode)
            vec = np.array([
                dist_norm, thr_norm, lat_norm, ue.reliab_ma, thr_norm,
                sinr_pred / 30.0,   # normalize by ~30 dB
                svc_idx / 2.0,
                ue.mobility_idx,
                subband_load,
                time_norm
            ], dtype=np.float32)
            obs[ue_id] = vec
        return obs

    def ric_features_SITM(self) -> Dict[str, np.ndarray]:
        """
        Return features used in Eq.(1) for RIC's graph embedding/clustering:
          x_i = [S, I, T, M]  (Sec. II; Table III)
        S: received signal strength proxy (inverse of path loss)
        I: interference proxy (#contenders on chosen subband in last step, normalized)
        T: traffic load proxy (queue bits / slot)
        M: mobility index in [0,1]
        """
        feats = {}
        for ue_id, ue in self.ues.items():
            S = self.channel.rx_power_proxy(ue.pos_xy, self.gnb_xy[ue.cell_id])
            I = float(np.mean(self._subband_load) / max(1, self.n_subbands))
            T = ue.queue_bits / 0.001  # bits per second equivalent
            M = ue.mobility_idx
            feats[ue_id] = np.array([S, I, T, M], dtype=np.float32)
        return feats

    def _update_mobility_and_cells(self) -> List[Tuple[str, int, int]]:
        handovers: List[Tuple[str, int, int]] = []
        for ue in self.ues.values():
            old_cell = ue.cell_id
            new_pos, new_heading = self.mobility.step(ue.pos_xy, ue.speed_mps, ue.heading_rad)
            ue.pos_xy = new_pos
            ue.heading_rad = new_heading
            ue.cell_id = self._nearest_cell(ue.pos_xy)
            if ue.cell_id != old_cell:
                handovers.append((ue.ue_id, old_cell, ue.cell_id))
        return handovers

    def _nearest_cell(self, xy: np.ndarray) -> int:
        d = np.linalg.norm(self.gnb_xy - xy[None, :], axis=1)
        return int(np.argmin(d))

    def _sim_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        # bounding box covering all cells
        mn = np.min(self.gnb_xy, axis=0) - np.array([self.isd_m, self.isd_m])
        mx = np.max(self.gnb_xy, axis=0) + np.array([self.isd_m, self.isd_m])
        return mn, mx

# ------------------------------ smoke test ---------------------------------
if __name__ == "__main__":
    env = ORanSimEnv()
    obs = env.reset()
    for _ in range(10):
        # random multi-agent actions
        act = {ue_id: (np.random.randint(0, env.n_subbands),
                       np.random.randint(0, 16),
                       np.random.randint(0, 3)) for ue_id in obs.keys()}
        obs, rew, done, info = env.step(act)
    print("OK: step loop ran; last time_slot=", info["time_slot"])
