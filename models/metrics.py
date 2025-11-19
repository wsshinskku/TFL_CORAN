# models/metrics.py
# 표 IV 지표 집계: QoS 만족도(%) / 평균 처리량(Mbps) / 평균 지연(ms) / 적응시간(s)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class QosTargets:
    rate_bps: float
    latency_s: float
    reliability: float

@dataclass
class MetricsAggregator:
    qos_targets: Dict[str, QosTargets]
    # 누적기
    sum_bits: float = 0.0
    sum_latency_s: float = 0.0
    n_steps: int = 0
    # 클래스별 만족도 집계
    satisfied: Dict[str, int] = field(default_factory=lambda: {"eMBB":0,"URLLC":0,"mMTC":0})
    counts: Dict[str, int] = field(default_factory=lambda: {"eMBB":0,"URLLC":0,"mMTC":0})
    # 적응시간 추적(핸드오버/신규 UE)
    activation_time_s: Dict[str, float] = field(default_factory=dict)
    first_meet_t: Dict[str, float] = field(default_factory=dict)

    def update_step(
        self,
        ue_states: Dict[str, Dict[str, float]],
        slot_s: float = 0.001,
    ):
        """
        ue_states: {ue_id: {"service":str, "served_bits":int, "latency_s":float, "reliab":float, "throughput_bps":float}}
        """
        self.n_steps += 1
        for ue_id, st in ue_states.items():
            svc = st["service"]
            self.sum_bits += st["throughput_bps"] * slot_s
            self.sum_latency_s += st["latency_s"] * slot_s
            self.counts[svc] += 1

            tar = self.qos_targets[svc]
            ok = (st["throughput_bps"] >= tar.rate_bps) and (st["latency_s"] <= tar.latency_s) and (st["reliab"] >= tar.reliability)
            self.satisfied[svc] += int(ok)

            # 적응시간: 최초 활성/핸드오버 시점부터 'ok' 상태 달성까지 걸린 시간
            if ue_id in self.activation_time_s:
                if ok and ue_id not in self.first_meet_t:
                    self.first_meet_t[ue_id] = self.activation_time_s[ue_id]

            # 활성 타임스탬프(없으면 시작)
            if ue_id not in self.activation_time_s:
                self.activation_time_s[ue_id] = 0.0
            else:
                self.activation_time_s[ue_id] += slot_s

    def report(self) -> Dict[str, float]:
        total_counts = sum(self.counts.values()) or 1
        qos_satisfaction = 100.0 * (sum(self.satisfied.values()) / float(total_counts))
        avg_thr_mbps = (self.sum_bits / max(self.n_steps,1)) * 8.0 / 1e6  # bits/s -> Mbps
        avg_lat_ms = (self.sum_latency_s / max(self.n_steps,1)) * 1e3      # s -> ms
        # 평균 적응시간(s): 기록된 UE의 평균
        if self.first_meet_t:
            adapt_s = float(np.mean(list(self.first_meet_t.values())))
        else:
            adapt_s = float("nan")
        return {
            "qos_satisfaction_%": qos_satisfaction,
            "avg_throughput_Mbps": avg_thr_mbps,
            "avg_latency_ms": avg_lat_ms,
            "adaptation_time_s": adapt_s
        }
