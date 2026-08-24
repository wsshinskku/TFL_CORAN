from copy import deepcopy
from pathlib import Path

import numpy as np

from tfl_coran.config import load_config
from tfl_coran.experiments import ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]


def _run_one_training_slot(runner: ExperimentRunner) -> None:
    states = runner.env.states()
    actions = runner._actions(states)
    result = runner.env.step(actions)
    runner._train_step(
        states,
        actions,
        result.rewards,
        result.states,
        result.terminated,
    )


def test_end_to_end_smoke_is_deterministic_and_writes_artifacts(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    first = ExperimentRunner(config, tmp_path / "first").run()
    second = ExperimentRunner(config, tmp_path / "second").run()
    for metric in ("qos_satisfaction_pct", "throughput_mbps", "latency_ms", "reliability"):
        assert first["evaluation"][metric] == second["evaluation"][metric]
        assert np.isfinite(first["evaluation"][metric])
    for relative in (
        "resolved_config.yaml",
        "summary.json",
        "training_metrics.csv",
        "evaluation_by_group.csv",
        "adaptation_events.csv",
        "checkpoints/vgae.pt",
        "checkpoints/final_global.pt",
    ):
        assert (tmp_path / "first" / relative).is_file()


def test_centralized_and_local_agents_share_the_same_warmup_horizon(tmp_path: Path) -> None:
    base = load_config(ROOT / "configs" / "smoke.yaml")
    centralized_config = deepcopy(base)
    centralized_config["method"] = "drl"
    federated_config = deepcopy(base)
    federated_config["method"] = "fdrl"
    centralized = ExperimentRunner(centralized_config, tmp_path / "centralized")
    federated = ExperimentRunner(federated_config, tmp_path / "federated")
    warmup_slots = int(base["agent"]["warmup_steps"])

    for _ in range(warmup_slots - 1):
        _run_one_training_slot(centralized)
        _run_one_training_slot(federated)
    assert centralized.central_agent is not None
    assert centralized.central_agent.learn_steps == 0
    assert all(agent.learn_steps == 0 for agent in federated.agents)

    _run_one_training_slot(centralized)
    _run_one_training_slot(federated)
    gradient_steps = int(base["agent"]["gradient_steps_per_slot"])
    assert centralized.central_agent.learn_steps == gradient_steps * centralized.num_ues
    assert all(agent.learn_steps == gradient_steps for agent in federated.agents)
