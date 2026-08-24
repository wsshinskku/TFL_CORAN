from copy import deepcopy
from pathlib import Path

import numpy as np

from tfl_coran.config import load_config
from tfl_coran.envs import ActionCodec, OranTrafficEnv

ROOT = Path(__file__).resolve().parents[1]


def test_action_codec_round_trip() -> None:
    codec = ActionCodec(subbands=6, rate_levels=3, priority_levels=3)
    indices = []
    triples = []
    for frequency in range(6):
        for rate in range(3):
            for priority in range(3):
                indices.append(codec.encode(frequency, rate, priority))
                triples.append((frequency, rate, priority))
    decoded = np.column_stack(codec.decode(indices))
    np.testing.assert_array_equal(decoded, np.asarray(triples))
    assert sorted(indices) == list(range(codec.size))


def test_environment_state_reward_and_event_counts() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    env = OranTrafficEnv(config["environment"], seed=11)
    states = env.states()
    assert states.shape == (12, env.state_dim)
    assert np.all(np.isfinite(states))
    result = env.step(env.heuristic_actions())
    expected = (
        result.metrics["reliability"] / env._target_for("reliability")
        + result.metrics["throughput_mbps"] / env._target_for("throughput_mbps")
        - result.metrics["latency_ms"] / env._target_for("latency_ms")
    )
    np.testing.assert_allclose(result.rewards, expected, rtol=1e-6, atol=1e-6)
    assert set(np.unique(result.metrics["qos_satisfied"])).issubset({0.0, 1.0})

    events = env.apply_round_dynamics(0.25, 1.0 / 12.0)
    assert len(events.handover_indices) == 3
    assert len(events.new_ue_indices) == 1
    assert not set(events.handover_indices) & set(events.new_ue_indices)
    assert env.num_ues == 12

    continuing_env = OranTrafficEnv(config["environment"], seed=12)
    for _ in range(config["environment"]["episode_slots"]):
        terminal = continuing_env.step(continuing_env.heuristic_actions()).terminated
    assert not np.any(terminal)


def test_heuristic_action_query_does_not_advance_environment_rng() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    first = OranTrafficEnv(config["environment"], seed=21)
    second = OranTrafficEnv(config["environment"], seed=21)
    fixed_actions = np.zeros(first.num_ues, dtype=np.int64)
    first.heuristic_actions()
    result_after_query = first.step(fixed_actions)
    result_without_query = second.step(fixed_actions)
    for name in ("throughput_mbps", "latency_ms", "reliability"):
        np.testing.assert_allclose(
            result_after_query.metrics[name], result_without_query.metrics[name]
        )


def test_distribution_mode_controls_new_ue_services() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    environment = deepcopy(config["environment"])
    environment["service_distribution"] = [1.0, 0.0, 0.0]
    environment["new_service_assignment"] = "distribution"
    env = OranTrafficEnv(environment, seed=22)
    events = env.apply_round_dynamics(0.0, 0.5)
    assert np.all(env.services[events.new_ue_indices] == 0)
