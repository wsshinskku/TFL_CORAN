import numpy as np
import pytest
import torch

from tfl_coran.agents import DDQNAgent, ReplayBuffer


def _agent(done: bool) -> DDQNAgent:
    config = {
        "hidden_dims": [],
        "learning_rate": 0.0,
        "discount": 0.5,
        "batch_size": 1,
        "target_update_episodes": 10,
        "replay_capacity": 4,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "epsilon_decay_episodes": 1,
        "warmup_steps": 1,
        "max_grad_norm": 10.0,
    }
    agent = DDQNAgent(1, 2, config, torch.device("cpu"), seed=1)
    online = agent.online.network[0]
    target = agent.target.network[0]
    with torch.no_grad():
        online.weight.copy_(torch.tensor([[1.0], [2.0]]))
        online.bias.zero_()
        target.weight.copy_(torch.tensor([[10.0], [1.0]]))
        target.bias.zero_()
    agent.observe(np.array([0.0], np.float32), 0, 1.0, np.array([1.0], np.float32), done)
    return agent


def test_ddqn_online_selects_and_target_evaluates() -> None:
    # online selects action 1 at s'=1; target evaluates action 1 as 1.
    # y=1 + .5*1 = 1.5, current Q=0, squared TD error=2.25.
    assert _agent(done=False).learn() == pytest.approx(2.25)
    # Terminal transition must not bootstrap.
    assert _agent(done=True).learn() == pytest.approx(1.0)


def test_replay_buffer_grows_lazily_and_samples() -> None:
    replay = ReplayBuffer(state_dim=2, capacity=8, seed=2, initial_capacity=2)
    for index in range(5):
        state = np.array([index, index + 1], dtype=np.float32)
        replay.add(state, index % 2, float(index), state + 1, False)
    assert len(replay) == 5
    batch = replay.sample(3)
    assert batch[0].shape == (3, 2)
    all_items = replay.sample(5)
    assert set(all_items[2].tolist()) == {0.0, 1.0, 2.0, 3.0, 4.0}
