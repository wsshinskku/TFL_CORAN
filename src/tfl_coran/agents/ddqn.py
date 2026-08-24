from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from .replay import ReplayBuffer

TensorState = dict[str, torch.Tensor]


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int] | tuple[int, ...]) -> None:
        super().__init__()
        dims = [state_dim, *hidden_dims, action_dim]
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(nn.Linear(in_dim, out_dim))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


class DDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Mapping[str, Any],
        device: torch.device,
        seed: int,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.config = dict(config)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(state_dim, action_dim, self.config["hidden_dims"]).to(device)
        self.target = QNetwork(state_dim, action_dim, self.config["hidden_dims"]).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(self.config["learning_rate"]))
        self.replay = ReplayBuffer(state_dim, int(self.config["replay_capacity"]), seed=seed + 17)
        self.epsilon = float(self.config["epsilon_start"])
        self.learn_steps = 0
        self.episodes_seen = 0

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_dim))
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def act_batch(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        with torch.no_grad():
            tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            greedy = self.online(tensor).argmax(dim=1).cpu().numpy()
        if deterministic or self.epsilon <= 0.0:
            return greedy.astype(np.int64)
        explore = self.rng.random(states.shape[0]) < self.epsilon
        random_actions = self.rng.integers(self.action_dim, size=states.shape[0])
        return np.where(explore, random_actions, greedy).astype(np.int64)

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(state, action, reward, next_state, done)

    def learn(self, gradient_steps: int = 1) -> float | None:
        required = max(int(self.config["batch_size"]), int(self.config["warmup_steps"]))
        if len(self.replay) < required:
            return None
        losses: list[float] = []
        for _ in range(int(gradient_steps)):
            batch = self.replay.sample(int(self.config["batch_size"]))
            states, actions, rewards, next_states, dones = (
                torch.as_tensor(item, device=self.device) for item in batch
            )
            states = states.float()
            next_states = next_states.float()
            actions = actions.long()
            rewards = rewards.float()
            dones = dones.float()

            predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # Equations (3)-(4): online network selects; target network evaluates.
                next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                next_values = self.target(next_states).gather(1, next_actions).squeeze(1)
                targets = rewards + float(self.config["discount"]) * (1.0 - dones) * next_values
            # Paper Eq. (5) specifies the squared temporal-difference error.
            loss = nn.functional.mse_loss(predicted, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), float(self.config["max_grad_norm"]))
            self.optimizer.step()
            self.learn_steps += 1
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    def end_episode(self, episode_index: int, sync_target: bool = True) -> None:
        self.episodes_seen += 1
        decay_episodes = max(1, int(self.config["epsilon_decay_episodes"]))
        fraction = min(1.0, self.episodes_seen / decay_episodes)
        start = float(self.config["epsilon_start"])
        end = float(self.config["epsilon_end"])
        self.epsilon = start + fraction * (end - start)
        if sync_target and (episode_index + 1) % int(self.config["target_update_episodes"]) == 0:
            self.sync_target()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def get_weights(self) -> TensorState:
        return {name: tensor.detach().cpu().clone() for name, tensor in self.online.state_dict().items()}

    def load_weights(self, weights: Mapping[str, torch.Tensor], sync_target: bool = False) -> None:
        state = {name: tensor.detach().to(self.device) for name, tensor in weights.items()}
        self.online.load_state_dict(state, strict=True)
        if sync_target:
            self.target.load_state_dict(state, strict=True)

    def reset_for_new_client(self, weights: Mapping[str, torch.Tensor]) -> None:
        self.load_weights(weights, sync_target=True)
        self.replay.clear()
        self.optimizer.state.clear()
        self.epsilon = float(self.config["epsilon_start"])
        self.episodes_seen = 0
