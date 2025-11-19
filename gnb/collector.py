# gnb/collector.py
# gNB는 UE들의 SITM 특징 및 로컬 모델 Δw를 수집해 RIC로 릴레이합니다.
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

@dataclass
class UplinkPacket:
    round_idx: int
    gnb_id: int
    sitm: Dict[str, np.ndarray]                 # {ue_id: [S,I,T,M]}
    deltas: Dict[str, Dict[str, torch.Tensor]]  # {ue_id: Δw_i}
    events: Dict[str, Any] = field(default_factory=dict)   # handover, attach/detach 등
    compress_meta: Optional[Dict[str, Any]] = None         # 압축/양자화 메타

class GNBCollector:
    """
    - UE 부착/이탈 관리
    - 슬롯/에피소드 단위 SITM 수집 (env에서 계산된 값 사용)
    - FL 라운드마다 Δw 업로드 패킷 구성 (RIC는 클러스터 가중 집계 수행)
    """
    def __init__(self, gnb_id: int, cell_id: int):
        self.gnb_id = gnb_id
        self.cell_id = cell_id
        self._attached: Dict[str, Dict[str, Any]] = {}   # {ue_id: meta}

    # -------- UE life-cycle --------
    def attach(self, ue_id: str, meta: Optional[Dict[str, Any]] = None):
        self._attached[ue_id] = meta or {}

    def detach(self, ue_id: str):
        if ue_id in self._attached:
            del self._attached[ue_id]

    # -------- Telemetry / Δw collection --------
    def gather_sitm(self, env) -> Dict[str, np.ndarray]:
        """
        env.ric_features_SITM() → {ue_id: [S,I,T,M]} 를 받아
        본 gNB에 연결된 UE만 필터링하여 반환. (식 (1) 입력)
        """
        all_feats: Dict[str, np.ndarray] = env.ric_features_SITM()
        return {ue_id: x for ue_id, x in all_feats.items() if ue_id in self._attached}

    def gather_deltas(self, agents: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        UE 에이전트가 보관한 Δw_i 를 수집(RIC 집계 입력). (식 (9)) 
        """
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for ue_id in self._attached.keys():
            ag = agents.get(ue_id, None)
            if ag is None:
                continue
            out[ue_id] = ag.compute_delta()
        return out

    # -------- FL Uplink packaging --------
    def build_uplink_packet(
        self,
        round_idx: int,
        env,
        agents: Dict[str, Any],
        events: Optional[Dict[str, Any]] = None,
        compress: Optional[Dict[str, Any]] = None,
    ) -> UplinkPacket:
        sitm = self.gather_sitm(env)
        deltas = self.gather_deltas(agents)
        pkt = UplinkPacket(
            round_idx=round_idx,
            gnb_id=self.gnb_id,
            sitm=sitm,
            deltas=deltas,
            events=events or {},
            compress_meta=compress or {}
        )
        return pkt
