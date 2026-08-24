from __future__ import annotations

import csv
import importlib.metadata
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from tfl_coran.agents import DDQNAgent
from tfl_coran.clustering import HardKMeansClusterer, SoftClusterer
from tfl_coran.config import dump_config
from tfl_coran.envs.oran_env import SERVICE_NAMES, OranTrafficEnv
from tfl_coran.federated.aggregator import (
    aggregate_models,
    clone_state,
    mix_states,
    serialized_parameter_bytes,
)
from tfl_coran.models.vgae import (
    VGAEArtifacts,
    load_vgae_checkpoint,
    save_vgae_checkpoint,
    train_vgae,
)
from tfl_coran.transfer import transfer_initialize
from tfl_coran.utils import ensure_dir, resolve_device, seed_everything, write_json

from .metrics import AdaptationTracker, GroupMetricAccumulator, MetricAccumulator

TensorState = dict[str, torch.Tensor]


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_historical_contexts(config: dict[str, Any], seed: int) -> list[np.ndarray]:
    env = OranTrafficEnv(config["environment"], seed=seed + 100_003)
    snapshots: list[np.ndarray] = []
    steps_between = max(2, min(20, int(config["environment"]["episode_slots"]) // 4))
    for snapshot_index in range(int(config["vgae"]["snapshots"])):
        for _ in range(steps_between):
            # Randomized frequency offsets prevent a degenerate, identical load feature.
            actions = env.heuristic_actions()
            offset = snapshot_index % env.subbands
            frequency, rate, priority = env.action_codec.decode(actions)
            frequency = (frequency + offset) % env.subbands
            actions = np.array(
                [
                    env.action_codec.encode(int(f), int(r), int(p))
                    for f, r, p in zip(frequency, rate, priority, strict=True)
                ],
                dtype=np.int64,
            )
            env.step(actions)
        snapshots.append(env.contexts().copy())
        env.apply_round_dynamics(
            float(config["federation"]["handover_fraction"]),
            float(config["federation"]["new_ue_fraction"]),
        )
    return snapshots


class ExperimentRunner:
    def __init__(
        self,
        config: dict[str, Any],
        output_dir: str | Path,
        vgae_checkpoint: str | Path | None = None,
    ) -> None:
        self.config = deepcopy(config)
        self.output_dir = ensure_dir(output_dir)
        self.seed = int(config["seed"])
        seed_everything(self.seed)
        self.device = resolve_device(str(config["device"]))
        self.env = OranTrafficEnv(config["environment"], seed=self.seed)
        self.method = str(config["method"])
        self.num_ues = self.env.num_ues
        self.vgae_checkpoint = Path(vgae_checkpoint) if vgae_checkpoint else None
        self.artifacts: VGAEArtifacts | None = None
        self.raw_scaler: StandardScaler | None = None
        self.clusterer: SoftClusterer | None = None
        self.hard_clusterer: HardKMeansClusterer | None = None
        self.memberships: np.ndarray | None = None
        self.membership_changes: list[float] = []
        self.training_rows: list[dict[str, Any]] = []
        self.agents: list[DDQNAgent] = []
        self.central_agent: DDQNAgent | None = None
        self.global_model: TensorState | None = None
        self.dispatch_bases: list[TensorState] = []
        self._prepare_output()
        self._prepare_representation()
        self._prepare_agents()

    def _prepare_output(self) -> None:
        ensure_dir(self.output_dir / "checkpoints")
        dump_config(self.config, self.output_dir / "resolved_config.yaml")
        package_names = ["torch", "numpy", "scikit-learn", "scipy", "PyYAML", "tfl-coran"]
        packages: dict[str, str | None] = {}
        for name in package_names:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        repository_root = Path(__file__).resolve().parents[3]
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            git_dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repository_root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            git_commit = None
            git_dirty = None
        metadata = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "packages": packages,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "device": str(self.device),
            "seed": self.seed,
            "method": self.method,
        }
        write_json(metadata, self.output_dir / "run_metadata.json")

    def _uses_context_groups(self) -> bool:
        toggles = self.config["toggles"]
        return self.method in {"cfdrl", "tfl_coran"} and bool(
            toggles["gmm"] or toggles["vgae"]
        )

    def _prepare_representation(self) -> None:
        if self.method not in {"cfdrl", "tfl_coran"}:
            return
        toggles = self.config["toggles"]
        if not bool(toggles["vgae"] or toggles["gmm"] or toggles["transfer"]):
            return
        history = generate_historical_contexts(self.config, self.seed)
        np.savez_compressed(
            self.output_dir / "historical_contexts.npz",
            **{f"snapshot_{index:03d}": value for index, value in enumerate(history)},
        )
        if bool(self.config["toggles"]["vgae"]):
            if self.vgae_checkpoint is not None:
                self.artifacts = load_vgae_checkpoint(self.vgae_checkpoint, self.device)
            else:
                self.artifacts = train_vgae(history, self.config["vgae"], self.device, self.seed + 701)
                save_vgae_checkpoint(
                    self.artifacts, self.config["vgae"], self.output_dir / "checkpoints" / "vgae.pt"
                )
            _write_csv(self.artifacts.losses, self.output_dir / "vgae_training.csv")
        else:
            self.raw_scaler = StandardScaler().fit(np.concatenate(history, axis=0))
        if bool(self.config["toggles"]["gmm"]):
            self.clusterer = SoftClusterer(self.config["gmm"], seed=self.seed + 991)
        elif bool(self.config["toggles"]["vgae"]):
            self.hard_clusterer = HardKMeansClusterer(self.config["gmm"], seed=self.seed + 991)

    def _new_agent(self, seed_offset: int) -> DDQNAgent:
        return DDQNAgent(
            self.env.state_dim,
            self.env.action_codec.size,
            self.config["agent"],
            self.device,
            seed=self.seed + seed_offset,
        )

    def _prepare_agents(self) -> None:
        # Representation training/loading must not perturb the common RL model initialization.
        seed_everything(self.seed + 20_000)
        if self.method == "heuristic":
            return
        if self.method == "drl":
            central_config = deepcopy(self.config["agent"])
            # A pooled policy keeps the same total replay capacity as N local
            # buffers; lazy growth prevents up-front paper-scale allocation.
            central_config["replay_capacity"] = int(central_config["replay_capacity"]) * self.num_ues
            self.central_agent = DDQNAgent(
                self.env.state_dim,
                self.env.action_codec.size,
                central_config,
                self.device,
                seed=self.seed + 10_000,
            )
            self.global_model = self.central_agent.get_weights()
            return
        prototype = self._new_agent(20_000)
        initial = prototype.get_weights()
        self.agents = [prototype]
        for index in range(1, self.num_ues):
            agent = self._new_agent(20_000 + index)
            agent.load_weights(initial, sync_target=True)
            self.agents.append(agent)
        self.global_model = clone_state(initial)
        self.dispatch_bases = [clone_state(initial) for _ in range(self.num_ues)]
        self._refresh_memberships(initial=True)

    def _scaled_contexts(self, contexts: np.ndarray) -> np.ndarray:
        if self.artifacts is not None:
            return self.artifacts.scaler.transform(contexts)
        if self.raw_scaler is None:
            self.raw_scaler = StandardScaler().fit(contexts)
        return self.raw_scaler.transform(contexts)

    def _embeddings(self, contexts: np.ndarray) -> np.ndarray:
        if self.artifacts is not None:
            return self.artifacts.embeddings(
                contexts, int(self.config["vgae"]["graph_neighbors"]), self.device
            )
        return self._scaled_contexts(contexts)

    def _refresh_memberships(self, initial: bool = False) -> None:
        if self.method in {"heuristic", "drl"}:
            return
        toggles = self.config["toggles"]
        if self.method == "fdrl" or not bool(toggles["gmm"] or toggles["vgae"]):
            updated = np.ones((self.num_ues, 1), dtype=np.float64)
        else:
            embeddings = self._embeddings(self.env.contexts())
            if bool(toggles["gmm"]):
                assert self.clusterer is not None
                updated = self.clusterer.fit_predict(embeddings, hard=self.method == "cfdrl")
            else:
                assert self.hard_clusterer is not None
                updated = self.hard_clusterer.fit_predict(embeddings)
        if self.memberships is not None and self.memberships.shape == updated.shape and not initial:
            self.membership_changes.extend(np.abs(updated - self.memberships).sum(axis=1).tolist())
        self.memberships = updated
        np.save(self.output_dir / "memberships_latest.npy", updated)

    def _assign_new_memberships(self, new_indices: np.ndarray) -> None:
        """Assign new UEs between structural refreshes without a VGAE forward pass."""
        assert self.memberships is not None
        if not new_indices.size or not self._uses_context_groups():
            return
        scaled = self._scaled_contexts(self.env.contexts())
        excluded = {int(index) for index in new_indices}
        fallback = self.memberships.mean(axis=0)
        for index in map(int, new_indices):
            candidates = [
                int(candidate)
                for candidate in np.flatnonzero(self.env.cell_ids == self.env.cell_ids[index])
                if int(candidate) not in excluded
            ]
            if not candidates:
                self.memberships[index] = fallback
                continue
            query = scaled[index]
            values = scaled[candidates]
            denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(query)
            similarity = np.full(len(candidates), -np.inf, dtype=np.float64)
            valid = denominator > 1.0e-12
            similarity[valid] = (values[valid] @ query) / denominator[valid]
            neighbor = candidates[int(np.argmax(similarity))] if np.any(valid) else candidates[0]
            self.memberships[index] = self.memberships[neighbor]
        row_sum = self.memberships[new_indices].sum(axis=1, keepdims=True)
        self.memberships[new_indices] /= np.maximum(row_sum, 1.0e-12)
        np.save(self.output_dir / "memberships_latest.npy", self.memberships)

    def _actions(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.method == "heuristic":
            return self.env.heuristic_actions()
        if self.method == "drl":
            assert self.central_agent is not None
            return self.central_agent.act_batch(states, deterministic=deterministic)
        return np.array(
            [
                agent.act(state, deterministic=deterministic)
                for agent, state in zip(self.agents, states, strict=True)
            ],
            dtype=np.int64,
        )

    def _train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float | None:
        gradient_steps = int(self.config["agent"]["gradient_steps_per_slot"])
        if self.method == "heuristic":
            return None
        if self.method == "drl":
            assert self.central_agent is not None
            for transition in zip(states, actions, rewards, next_states, dones, strict=True):
                self.central_agent.observe(*transition)
            local_horizon = max(
                int(self.config["agent"]["batch_size"]),
                int(self.config["agent"]["warmup_steps"]),
            )
            if len(self.central_agent.replay) < local_horizon * self.num_ues:
                return None
            # Match the aggregate number of local optimizer steps used by N federated clients.
            return self.central_agent.learn(gradient_steps * self.num_ues)
        losses: list[float] = []
        transitions = zip(states, actions, rewards, next_states, dones, strict=True)
        for agent, transition in zip(self.agents, transitions, strict=True):
            agent.observe(*transition)
            loss = agent.learn(gradient_steps)
            if loss is not None:
                losses.append(loss)
        return float(np.mean(losses)) if losses else None

    def _end_episode(self, episode: int, sync_target: bool = True) -> None:
        if self.central_agent is not None:
            self.central_agent.end_episode(episode, sync_target=sync_target)
        for agent in self.agents:
            agent.end_episode(episode, sync_target=sync_target)

    def _sync_targets(self) -> None:
        if self.central_agent is not None:
            self.central_agent.sync_target()
        for agent in self.agents:
            agent.sync_target()

    def _federate(
        self, round_index: int, *, apply_events: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.method in {"heuristic", "drl"}:
            if apply_events:
                events = self.env.apply_round_dynamics(
                    float(self.config["federation"]["handover_fraction"]),
                    float(self.config["federation"]["new_ue_fraction"]),
                )
                return events.handover_indices, events.new_ue_indices
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        assert self.global_model is not None and self.memberships is not None
        local_models = [agent.get_weights() for agent in self.agents]
        aggregation = aggregate_models(
            local_models,
            self.memberships,
            self.global_model,
            self.dispatch_bases,
            str(self.config["federation"]["delta_reference"]),
        )
        self.global_model = aggregation.global_model

        # Local train -> aggregate -> mobility/churn -> scheduled refresh -> transfer.
        if apply_events:
            events = self.env.apply_round_dynamics(
                float(self.config["federation"]["handover_fraction"]),
                float(self.config["federation"]["new_ue_fraction"]),
            )
            new_indices = events.new_ue_indices
            handovers = events.handover_indices
        else:
            new_indices = np.empty(0, dtype=np.int64)
            handovers = np.empty(0, dtype=np.int64)
        refresh_period = int(self.config["federation"]["cluster_refresh_rounds"])
        if (round_index + 1) % refresh_period == 0:
            self._refresh_memberships()
        else:
            self._assign_new_memberships(new_indices)

        assert self.memberships is not None
        current_models = [
            mix_states(aggregation.cluster_models, self.memberships[index])
            for index in range(self.num_ues)
        ]
        sync_target = bool(self.config["federation"]["sync_target_on_dispatch"])
        for agent, weights in zip(self.agents, current_models, strict=True):
            agent.load_weights(weights, sync_target=sync_target)
            if bool(self.config["federation"]["reset_optimizer_on_dispatch"]):
                agent.optimizer.state.clear()
        self.dispatch_bases = [clone_state(state) for state in current_models]
        if self.method == "tfl_coran" and bool(self.config["toggles"]["transfer"]):
            scaled_contexts = self._scaled_contexts(self.env.contexts())
            transfer_models = list(current_models)
            for index in map(int, new_indices):
                # If a destination cell has no eligible incumbent, a new UE
                # falls back to the shared model rather than a predecessor's
                # stale personalized slot.
                transfer_models[index] = self.global_model
            initialized, _ = transfer_initialize(
                scaled_contexts,
                self.env.cell_ids,
                transfer_models,
                local_models,
                handovers,
                new_indices,
                float(self.config["federation"]["transfer_delta"]),
            )
            for index, weights in initialized.items():
                if index in set(map(int, new_indices)):
                    self.agents[index].reset_for_new_client(weights)
                else:
                    self.agents[index].load_weights(weights, sync_target=True)
                self.dispatch_bases[index] = clone_state(weights)
        else:
            for index in map(int, new_indices):
                # A transfer-disabled activation is a genuine cold start from
                # the shared model, distinct from neighbor-assisted transfer.
                self.agents[index].reset_for_new_client(self.global_model)
                self.dispatch_bases[index] = clone_state(self.global_model)

        checkpoint_every = int(self.config["experiment"]["checkpoint_every_rounds"])
        if checkpoint_every > 0 and (round_index + 1) % checkpoint_every == 0:
            torch.save(
                {"round": round_index + 1, "global_model": self.global_model},
                self.output_dir / "checkpoints" / f"round_{round_index + 1:04d}.pt",
            )
        return handovers, new_indices

    def train(self) -> tuple[AdaptationTracker, float]:
        states = self.env.states()
        tracker = AdaptationTracker(
            int(self.config["experiment"]["adaptation_consecutive_slots"]),
            float(self.config["environment"]["slot_duration_ms"]),
        )
        episodes = int(self.config["experiment"]["episodes"])
        slots = int(self.config["environment"]["episode_slots"])
        fl_period = int(self.config["federation"]["episodes_per_round"])
        started = time.perf_counter()
        round_index = 0
        for episode in range(episodes):
            episode_metrics = MetricAccumulator()
            losses: list[float] = []
            for _ in range(slots):
                actions = self._actions(states)
                result = self.env.step(actions)
                loss = self._train_step(states, actions, result.rewards, result.states, result.terminated)
                if loss is not None:
                    losses.append(loss)
                episode_metrics.update(result.metrics)
                tracker.update(result.metrics["qos_satisfied"], self.env.slot)
                states = result.states
            target_due = (episode + 1) % int(self.config["agent"]["target_update_episodes"]) == 0
            self._end_episode(episode, sync_target=False)
            row: dict[str, Any] = {"episode": episode + 1, **episode_metrics.summary()}
            row["loss"] = float(np.mean(losses)) if losses else None
            row["epsilon"] = (
                self.central_agent.epsilon
                if self.central_agent is not None
                else (self.agents[0].epsilon if self.agents else 0.0)
            )
            if (episode + 1) % fl_period == 0:
                # The last boundary aggregates learned weights but does not
                # inject clients that would receive no subsequent local-update window.
                handovers, activations = self._federate(
                    round_index, apply_events=(episode + 1) < episodes
                )
                round_index += 1
                row["fl_round"] = round_index
                row["handovers"] = int(len(handovers))
                row["activations"] = int(len(activations))
                tracker.start(handovers, "handover", self.env.slot)
                tracker.start(activations, "activation", self.env.slot)
                states = self.env.states()
            if target_due:
                # If FL also occurs here, synchronize to the received model, not the pre-FL local model.
                self._sync_targets()
            self.training_rows.append(row)
        return tracker, time.perf_counter() - started

    def evaluate(
        self, tracker: AdaptationTracker | None = None
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        states = self.env.states()
        overall = MetricAccumulator()
        groups = GroupMetricAccumulator()
        episodes = int(self.config["experiment"]["evaluation_episodes"])
        slots = int(self.config["environment"]["episode_slots"])
        for _ in range(episodes * slots):
            actions = self._actions(states, deterministic=True)
            result = self.env.step(actions)
            overall.update(result.metrics)
            groups.update(result.metrics, self.env.cell_labels, SERVICE_NAMES)
            if tracker is not None:
                tracker.update(result.metrics["qos_satisfied"], self.env.slot)
            states = result.states
        return overall.summary(), groups.rows()

    def run(self) -> dict[str, Any]:
        tracker, training_seconds = self.train()
        tracker.finalize()
        # Headline evaluation is frozen and separate from online adaptation tracking.
        evaluation, group_rows = self.evaluate()
        _write_csv(self.training_rows, self.output_dir / "training_metrics.csv")
        _write_csv(group_rows, self.output_dir / "evaluation_by_group.csv")
        _write_csv(tracker.records, self.output_dir / "adaptation_events.csv")
        if self.central_agent is not None:
            self.global_model = self.central_agent.get_weights()
        if self.global_model is not None:
            torch.save(self.global_model, self.output_dir / "checkpoints" / "final_global.pt")
            model_bytes = serialized_parameter_bytes(self.global_model)
        else:
            model_bytes = 0
        if self.agents and bool(self.config["experiment"]["save_client_models"]):
            torch.save(
                {"client_models": [agent.get_weights() for agent in self.agents]},
                self.output_dir / "checkpoints" / "final_clients.pt",
            )
        summary: dict[str, Any] = {
            "method": self.method,
            "seed": self.seed,
            "num_ues": self.num_ues,
            "training_seconds": training_seconds,
            "evaluation": evaluation,
            "adaptation": tracker.summary(),
            "median_membership_l1_change": (
                float(np.median(self.membership_changes)) if self.membership_changes else None
            ),
            "model_parameter_bytes": model_bytes,
            "paper_result_reproduced": False,
        }
        write_json(summary, self.output_dir / "summary.json")
        return summary


def run_benchmark(
    config: dict[str, Any],
    output_dir: str | Path,
    methods: Sequence[str],
    vgae_checkpoint: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = ensure_dir(output_dir)
    summaries: list[dict[str, Any]] = []
    for method in methods:
        method_config = deepcopy(config)
        method_config["method"] = method
        runner = ExperimentRunner(method_config, root / method, vgae_checkpoint=vgae_checkpoint)
        summaries.append(runner.run())
    _write_csv(
        [
            {
                "method": row["method"],
                "seed": row["seed"],
                "qos_satisfaction_pct": row["evaluation"].get("qos_satisfaction_pct"),
                "throughput_mbps": row["evaluation"].get("throughput_mbps"),
                "latency_ms": row["evaluation"].get("latency_ms"),
                "handover_adaptation_s": row["adaptation"].get("handover_adaptation_s"),
                "handover_completion_rate": row["adaptation"].get(
                    "handover_completion_rate"
                ),
                "handover_adaptation_penalized_s": row["adaptation"].get(
                    "handover_adaptation_penalized_s"
                ),
                "activation_adaptation_s": row["adaptation"].get("activation_adaptation_s"),
                "activation_completion_rate": row["adaptation"].get(
                    "activation_completion_rate"
                ),
                "activation_adaptation_penalized_s": row["adaptation"].get(
                    "activation_adaptation_penalized_s"
                ),
                "training_seconds": row["training_seconds"],
            }
            for row in summaries
        ],
        root / "benchmark_summary.csv",
    )
    write_json(summaries, root / "benchmark_summary.json")
    return summaries


def run_reproduction(
    config: dict[str, Any],
    output_dir: str | Path,
    methods: Sequence[str],
    seeds: Sequence[int],
    vgae_checkpoint: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = ensure_dir(output_dir)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        seeded = deepcopy(config)
        seeded["seed"] = int(seed)
        runs.extend(
            run_benchmark(
                seeded,
                root / f"seed_{int(seed):03d}",
                methods,
                vgae_checkpoint=vgae_checkpoint,
            )
        )
    aggregate: list[dict[str, Any]] = []
    for method in methods:
        selected = [row for row in runs if row["method"] == method]
        row: dict[str, Any] = {"method": method, "seeds": len(selected)}
        for metric in ("qos_satisfaction_pct", "throughput_mbps", "latency_ms", "reliability"):
            values = np.asarray([entry["evaluation"][metric] for entry in selected], dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = float(1.96 * std / np.sqrt(max(1, values.size)))
        for metric in (
            "handover_adaptation_s",
            "handover_completion_rate",
            "handover_adaptation_penalized_s",
            "activation_adaptation_s",
            "activation_completion_rate",
            "activation_adaptation_penalized_s",
        ):
            values = np.asarray(
                [
                    entry["adaptation"][metric]
                    for entry in selected
                    if entry["adaptation"].get(metric) is not None
                ],
                dtype=np.float64,
            )
            if values.size == 0:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_ci95"] = None
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = float(1.96 * std / np.sqrt(values.size))
        aggregate.append(row)
    _write_csv(aggregate, root / "reproduction_summary.csv")
    write_json(aggregate, root / "reproduction_summary.json")
    write_json(runs, root / "reproduction_runs.json")
    return aggregate
