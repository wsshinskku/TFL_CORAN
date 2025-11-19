# envs/o_ran_sim/scheduler.py
# Slot-level scheduler with reuse and utility (asp + predicted rate + QoS deficit)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class ScheduledTx:
    ue_id: str
    subband: int
    mcs_index: int
    sinr_db: float

class UtilityScheduler:
    def __init__(self, n_subbands: int, reuse_allowed: bool, weights: Tuple[float,float,float]=(1.0,1.0,1.0)):
        self.n_subbands = n_subbands
        self.reuse_allowed = reuse_allowed
        self.w_prio, self.w_rate, self.w_def = weights

    def schedule(self,
                 decisions: Dict[str, "McsDecision"],
                 contenders: Dict[int, List[str]],
                 qos_targets: Dict[str, tuple],
                 ue_states) -> List[ScheduledTx]:
        """
        decisions: per-UE predicted (sinr, tb_bits) for current action
        contenders: per subband UE id list
        qos_targets: {svc: (R_tar, L_tar)}
        """
        result: List[ScheduledTx] = []
        # Precompute QoS deficits: deficit↑ when backlog or delay↑, throughput↓
        def deficit(ue) -> float:
            R_tar, L_tar = qos_targets[ue.service]
            thr = (ue.throughput_ma_bits/0.001) / (R_tar+1e-9)
            lat = ue.latency_ma_s / (L_tar+1e-9)
            # larger when low throughput and high delay
            return float(np.clip((1.0 - thr) + 0.5*lat, 0.0, 3.0))

        deficits = {ue_id: deficit(ue) for ue_id, ue in ue_states.items()}

        for sb in range(self.n_subbands):
            cand = contenders.get(sb, [])
            if not cand:
                continue
            if self.reuse_allowed:
                # Select top-K based on utility; allow multiple UEs per subband
                # K: small cap to avoid extreme reuse; here K=2 by default
                K = min(2, len(cand))
                util = []
                for ue_id in cand:
                    dec = decisions[ue_id]
                    rate_norm = dec.tb_bits / (880e3)   # normalize by max TB
                    u = self.w_prio*(dec.priority/2.0) + self.w_rate*rate_norm + self.w_def*deficits[ue_id]
                    util.append((u, ue_id))
                util.sort(key=lambda x: x[0], reverse=True)
                chosen = [ue for _, ue in util[:K]]
                for ue_id in chosen:
                    dec = decisions[ue_id]
                    tx = ScheduledTx(ue_id=ue_id, subband=sb, mcs_index=dec.mcs_index, sinr_db=dec.sinr_db)
                    result.append(tx)
            else:
                # Exclusive: pick one maximizer
                scores = []
                for ue_id in cand:
                    dec = decisions[ue_id]
                    rate_norm = dec.tb_bits / (880e3)
                    s = self.w_prio*(dec.priority/2.0) + self.w_rate*rate_norm + self.w_def*deficits[ue_id]
                    scores.append((s, ue_id))
                ue_id = max(scores, key=lambda x: x[0])[1]
                dec = decisions[ue_id]
                result.append(ScheduledTx(ue_id=ue_id, subband=sb, mcs_index=dec.mcs_index, sinr_db=dec.sinr_db))

        return result
