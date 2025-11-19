# ue/agent.py
# DDQN 에이전트 (식 (7) 보상 사용, 식 (8) DDQN 업데이트; 표 I 하이퍼파라미터)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import ReplayBuffer
from .policy import EpsilonScheduler
from .features import StateNormalizer

# ---------- 액션 공간: (subband, mcs, priority) → 단일 이산 index ----------

@dataclass
class ActionSpace:
    n_subbands: int
    n_mcs: int
    n_priority: int

    @property
    def n_actions(self) -> int:
        return int(self.n_subbands * self.n_mcs * self.n_priority)

    def encode(self, sb: int, mcs: int, pr: int) -> int:
        sb = int(np.clip(sb, 0, self.n_subbands - 1))
        mcs = int(np.clip(mcs, 0, self.n_mcs - 1))
        pr = int(np.clip(pr, 0, self.n_priority - 1))
        return int(sb + self.n_subbands * (mcs + self.n_mcs * pr))

    def decode(self, idx: int) -> Tuple[int, int, int]:
        idx = int(np.clip(idx, 0, self.n_actions - 1))
        pr = idx // (self.n_subbands * self.n_mcs)
        rem = idx % (self.n_subbands * self.n_mcs)
        mcs = rem // self.n_subbands
        sb = rem % self.n_subbands
        return int(sb), int(mcs), int(pr)

# ------------------------------ Q-Network ------------------------------

class QNetwork(nn.Module):
    """단일-head Q(s, a) with joint discrete action (n_actions)."""
    def __init__(self, state_dim: int, n_actions: int, hidden: Tuple[int, int]=(256, 256)):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.q = nn.Linear(hidden[1], n_actions)
        nn.init.kaiming_uniform_(self.fc1.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=np.sqrt(5))
        nn.init.xavier_uniform_(self.q.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

# ------------------------------ Agent ------------------------------

@dataclass
class DDQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    target_update_episodes: int = 10
    replay_capacity: int = 100_000
    epsilon: EpsilonScheduler = EpsilonScheduler(1.0, 0.01, 100)
    grad_clip_norm: float = 10.0
    device: str = "cuda"

class DDQNAgent:
    """
    DDQN Agent for UE-level control.
    - 행동: (subband, mcs, priority) (논문 섹션 III-D.1)
    - 보상: R = r/η_tar + θ/R_tar - l/L_tar (식 (7), env에서 계산)
    - 업데이트: Double DQN (식 (8))
    """
    def __init__(self, state_dim: int, action_space: ActionSpace, cfg: DDQNConfig, seed: int = 42, ue_id: str = "ue"):
        self.ue_id = ue_id
        self.state_dim = state_dim
        self.aspace = action_space
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
        torch.manual_seed(seed); np.random.seed(seed)

        self.online = QNetwork(state_dim, self.aspace.n_actions).to(self.device)
        self.target = QNetwork(state_dim, self.aspace.n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber

        self.replay = ReplayBuffer(capacity=self.cfg.replay_capacity, seed=seed)
        self.normalizer = StateNormalizer(dim=state_dim)
        self._episode_idx = 0
        self._last_broadcast_state: Optional[Dict[str, torch.Tensor]] = None  # FL용 기준 가중치

    # ------------------------ 상호작용 API ------------------------

    def act(self, obs: np.ndarray, explore: bool = True) -> Tuple[int, int, int]:
        """ε-탐욕으로 액션 선택; 반환형은 (subband, mcs, priority)."""
        self.normalizer.update(obs)
        s = self.normalizer.normalize(obs)
        eps = self.cfg.epsilon.value(self._episode_idx) if explore else 0.0
        if np.random.rand() < eps:
            # random action
            sb = np.random.randint(0, self.aspace.n_subbands)
            mcs = np.random.randint(0, self.aspace.n_mcs)
            pr = np.random.randint(0, self.aspace.n_priority)
            return sb, mcs, pr
        with torch.no_grad():
            st = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(st)  # [1, n_actions]
            a_idx = int(torch.argmax(q, dim=1).item())
        return self.aspace.decode(a_idx)

    def observe(self, s: np.ndarray, a_tuple: Tuple[int,int,int], r: float, s_next: np.ndarray, done: bool):
        a_idx = self.aspace.encode(*a_tuple)
        self.replay.add(s.astype(np.float32), a_idx, float(r), s_next.astype(np.float32), bool(done))

    def learn(self, gradient_steps: int = 1) -> Dict[str, float]:
        """버퍼에서 샘플링하여 DDQN 업데이트."""
        logs = {}
        for _ in range(gradient_steps):
            if len(self.replay) < self.cfg.batch_size:
                break
            S, A, R, SN, D = self.replay.sample(self.cfg.batch_size)
            S = torch.as_tensor(S, dtype=torch.float32, device=self.device)
            A = torch.as_tensor(A, dtype=torch.int64, device=self.device).unsqueeze(1)     # [B,1]
            R = torch.as_tensor(R, dtype=torch.float32, device=self.device)                # [B]
            SN = torch.as_tensor(SN, dtype=torch.float32, device=self.device)
            D = torch.as_tensor(D, dtype=torch.float32, device=self.device)                # [B]

            # 상태 표준화 (on-the-fly) - 관측이 이미 정규화되어 있어도 안전
            S = (S - torch.as_tensor(self.normalizer.mu, device=self.device)) / \
                (torch.sqrt(torch.as_tensor(self.normalizer.var, device=self.device)) + 1e-6)
            SN = (SN - torch.as_tensor(self.normalizer.mu, device=self.device)) / \
                 (torch.sqrt(torch.as_tensor(self.normalizer.var, device=self.device)) + 1e-6)

            # Q(s,a) & Double DQN target
            q = self.online(S).gather(1, A).squeeze(1)  # [B]
            with torch.no_grad():
                a_next = torch.argmax(self.online(SN), dim=1, keepdim=True)                # [B,1]
                q_next = self.target(SN).gather(1, a_next).squeeze(1)                      # [B]
                y = R + self.cfg.gamma * (1.0 - D) * q_next

            loss = self.loss_fn(q, y)
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip_norm)
            self.optim.step()
            logs = {"loss": float(loss.item())}
        return logs

    def end_episode(self):
        """에피소드 종료 시 호출. 타깃 네트 업데이트 스케줄"""
        self._episode_idx += 1
        if (self._episode_idx % self.cfg.target_update_episodes) == 0:
            self.target.load_state_dict(self.online.state_dict())

    # ------------------------ FL 보조 유틸 ------------------------

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().cpu() for k, v in self.online.state_dict().items()}

    def load_weights(self, state: Dict[str, torch.Tensor], set_as_broadcast_baseline: bool = False):
        self.online.load_state_dict(state, strict=True)
        self.target.load_state_dict(state, strict=True)
        if set_as_broadcast_baseline:
            self._last_broadcast_state = {k: v.clone() for k, v in state.items()}

    def compute_delta(self) -> Dict[str, torch.Tensor]:
        """
        FL 업로드용 Δw = w_local - w_broadcast.
        호출 전 _last_broadcast_state가 설정되어 있어야 함.
        """
        if self._last_broadcast_state is None:
            # 초기 라운드: 전체 가중치를 delta로 취급(혹은 0으로 보낼 수도 있음)
            return self.get_weights()
        cur = self.get_weights()
        delta = {}
        for k, v in cur.items():
            delta[k] = v - self._last_broadcast_state[k]
        return delta

    def accept_broadcast_and_reset_delta(self, new_personalized: Dict[str, torch.Tensor]):
        #RIC 개인화 모델 수신 후, 기준선 갱신.
        self.load_weights(new_personalized, set_as_broadcast_baseline=True)
