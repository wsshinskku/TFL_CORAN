# data/historical_dataset.py
# 오프라인 VGAE 학습(식 (4))을 위한 그래프 스냅샷 데이터셋.
# 각 스냅샷은 SITM 행렬 X[N,4]와 유사도 인접행렬 A(N,N)를 포함합니다(식 (1)).  :contentReference[oaicite:12]{index=12}
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from envs.o_ran_sim.base_env import ORanSimEnv
from ric.topology import build_features_matrix, build_similarity_adjacency, normalize_adj

@dataclass
class GraphSnapshot:
    X: np.ndarray      # [N,4]
    A_hat: np.ndarray  # [N,N] normalized adj (for GCN), D^{-1/2}(A+I)D^{-1/2}  :contentReference[oaicite:13]{index=13}

class HistoricalGraphDataset(Dataset):
    """
    시뮬레이터에서 여러 에피소드/라운드를 진행하며 정해진 간격으로 SITM→A_hat을 뽑아낸 스냅샷 집합.
    - 실제 운영망이라면 로그/텔레메트리에서 동일 포맷(X, A_hat)으로 구축.  :contentReference[oaicite:14]{index=14}
    """
    def __init__(self, snapshots: List[GraphSnapshot]):
        self.snaps = snapshots

    def __len__(self): return len(self.snaps)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.snaps[idx]
        Xt = torch.from_numpy(g.X.astype(np.float32))
        At = torch.from_numpy(g.A_hat.astype(np.float32))
        return Xt, At

def collect_snapshots_from_env(
    env: Optional[ORanSimEnv] = None,
    num_snapshots: int = 50,
    steps_between: int = 200,
) -> HistoricalGraphDataset:
    """
    간단한 합성: 시뮬레이터를 돌리며 X/A_hat 스냅샷을 수집하여 VGAE 프리트레이닝에 사용.
    실제 데이터가 있으면 동일 API로 대체.  :contentReference[oaicite:15]{index=15}
    """
    env = env or ORanSimEnv()
    obs = env.reset()
    snaps: List[GraphSnapshot] = []
    for k in range(num_snapshots):
        # 몇 스텝 진행 (무작위 액션)
        for _ in range(steps_between):
            actions = {ue_id: (np.random.randint(0, env.n_subbands),
                               np.random.randint(0, env.channel.num_mcs()),
                               np.random.randint(0, 3)) for ue_id in obs.keys()}
            obs, _, dones, _ = env.step(actions)
            if all(dones.values()):
                obs = env.reset()
        # 스냅샷 생성: X(=SITM), A(식 (1)), A_hat
        sitm = env.ric_features_SITM()                 # {ue: [S,I,T,M]}
        X, _ = build_features_matrix(sitm)             # [N,4]
        A = build_similarity_adjacency(X, self_loop=True)
        A_hat = normalize_adj(A, add_self_loop=True)   # GCN용 정규화(식 (2) 참조)  :contentReference[oaicite:16]{index=16}
        snaps.append(GraphSnapshot(X=X, A_hat=A_hat))
    return HistoricalGraphDataset(snaps)
