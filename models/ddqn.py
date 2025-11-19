# models/ddqn.py
# Joint discrete action Q-network (UE 액션: subband×mcs×priority) 
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

class JointActionQNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: Tuple[int,int]=(256,256)):
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

@dataclass
class JointActionSpace:
    n_subbands: int
    n_mcs: int
    n_priority: int
    @property
    def n_actions(self) -> int:
        return int(self.n_subbands * self.n_mcs * self.n_priority)
