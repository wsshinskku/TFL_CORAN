from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """A lazily growing NumPy ring buffer.

    Lazy allocation matters for the paper-scale 375-client run: declaring a
    capacity of 100,000 for every UE should not allocate all storage up front.
    """

    def __init__(self, state_dim: int, capacity: int, seed: int, initial_capacity: int = 1024) -> None:
        self.state_dim = int(state_dim)
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._allocated = min(self.capacity, max(1, int(initial_capacity)))
        self._states = np.empty((self._allocated, self.state_dim), dtype=np.float32)
        self._next_states = np.empty((self._allocated, self.state_dim), dtype=np.float32)
        self._actions = np.empty(self._allocated, dtype=np.int64)
        self._rewards = np.empty(self._allocated, dtype=np.float32)
        self._dones = np.empty(self._allocated, dtype=np.float32)
        self._size = 0
        self._position = 0

    def __len__(self) -> int:
        return self._size

    def _grow(self) -> None:
        if self._allocated >= self.capacity:
            return
        new_capacity = min(self.capacity, self._allocated * 2)
        states = np.empty((new_capacity, self.state_dim), dtype=np.float32)
        next_states = np.empty((new_capacity, self.state_dim), dtype=np.float32)
        actions = np.empty(new_capacity, dtype=np.int64)
        rewards = np.empty(new_capacity, dtype=np.float32)
        dones = np.empty(new_capacity, dtype=np.float32)
        states[: self._size] = self._states[: self._size]
        next_states[: self._size] = self._next_states[: self._size]
        actions[: self._size] = self._actions[: self._size]
        rewards[: self._size] = self._rewards[: self._size]
        dones[: self._size] = self._dones[: self._size]
        self._states, self._next_states = states, next_states
        self._actions, self._rewards, self._dones = actions, rewards, dones
        self._allocated = new_capacity
        self._position = self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if self._size == self._allocated and self._allocated < self.capacity:
            self._grow()
        index = self._position
        self._states[index] = state
        self._next_states[index] = next_state
        self._actions[index] = action
        self._rewards[index] = reward
        self._dones[index] = float(done)
        self._position = (self._position + 1) % self._allocated
        self._size = min(self._size + 1, self._allocated)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        if batch_size > self._size:
            raise ValueError(f"cannot sample {batch_size} items from a buffer of size {self._size}")
        indices = self.rng.choice(self._size, size=batch_size, replace=False)
        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
        )

    def clear(self) -> None:
        self._size = 0
        self._position = 0
