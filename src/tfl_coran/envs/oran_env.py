from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SERVICE_NAMES = np.array(["embb", "urllc", "mmtc"])


@dataclass(frozen=True)
class EventBatch:
    handover_indices: np.ndarray
    new_ue_indices: np.ndarray


@dataclass(frozen=True)
class StepResult:
    states: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    metrics: dict[str, np.ndarray]


class ActionCodec:
    """Maps the paper's (subband, rate, priority) action tuple to one DDQN index."""

    def __init__(self, subbands: int, rate_levels: int = 3, priority_levels: int = 3) -> None:
        self.subbands = int(subbands)
        self.rate_levels = int(rate_levels)
        self.priority_levels = int(priority_levels)
        self.size = self.subbands * self.rate_levels * self.priority_levels

    def encode(self, frequency: int, rate: int, priority: int) -> int:
        if not 0 <= frequency < self.subbands:
            raise ValueError("frequency out of range")
        if not 0 <= rate < self.rate_levels:
            raise ValueError("rate out of range")
        if not 0 <= priority < self.priority_levels:
            raise ValueError("priority out of range")
        return (frequency * self.rate_levels + rate) * self.priority_levels + priority

    def decode(self, indices: np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action = np.asarray(indices, dtype=np.int64)
        if np.any(action < 0) or np.any(action >= self.size):
            raise ValueError("action index out of range")
        priority = action % self.priority_levels
        quotient = action // self.priority_levels
        rate = quotient % self.rate_levels
        frequency = quotient // self.rate_levels
        return frequency, rate, priority


class OranTrafficEnv:
    """Lightweight multi-cell O-RAN simulator matching the paper's control interface.

    It is intentionally self-contained. UERANSIM/Open5GS/QuaDRiGa traces can be
    adapted through the same state/context API, while this backend keeps CI and
    algorithmic reproduction runnable on a laptop.
    """

    state_dim = 9
    context_dim = 4

    def __init__(self, config: dict[str, Any], seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.cell_counts = np.asarray(config["cell_ue_counts"], dtype=np.int64)
        self.cell_labels = list(config["cell_labels"])
        self.num_cells = len(self.cell_counts)
        self.num_ues = int(self.cell_counts.sum())
        self.isd = float(config["inter_site_distance_m"])
        self.frequency_ghz = float(config["carrier_frequency_ghz"])
        self.bandwidth_mhz = float(config["bandwidth_mhz"])
        self.subbands = int(config["subbands"])
        self.slot_duration_s = float(config["slot_duration_ms"]) / 1000.0
        self.episode_slots = int(config["episode_slots"])
        self.max_latency_ms = float(config["max_latency_ms"])
        self.max_queue_mbit = float(config["max_queue_mbit"])
        self.action_codec = ActionCodec(self.subbands)
        self.max_ues_per_subband = int(config["max_ues_per_subband"])
        self.tx_power_dbm = float(config["tx_power_dbm"])
        self.noise_figure_db = float(config["noise_figure_db"])
        self.channel_model = config["channel_model"]
        self.scheduler_model = config["scheduler_model"]
        self.traffic_model = config["traffic_model"]
        self.rate_factors = np.asarray(self.scheduler_model["rate_factors"], dtype=np.float32)
        self.service_bias = np.array(
            [self.scheduler_model["service_bias"][name] for name in SERVICE_NAMES], dtype=np.float32
        )
        self.base_latency_ms = np.array(
            [self.traffic_model["base_latency_ms"][name] for name in SERVICE_NAMES], dtype=np.float32
        )
        self.cell_centers = np.array(
            [[0.0, 0.0], [self.isd, 0.0], [0.5 * self.isd, np.sqrt(3.0) * 0.5 * self.isd]],
            dtype=np.float32,
        )
        self.cell_radius = self.isd / np.sqrt(3.0)
        self._service_targets = self._target_arrays(config["service_targets"])
        self.arrival_rates = np.array(
            [config["arrival_rate_mbps"][name] for name in SERVICE_NAMES], dtype=np.float32
        )
        self.speed_ranges = np.array(
            [config["speed_mps"][label] for label in self.cell_labels], dtype=np.float32
        )
        self._initialize_population()

    @staticmethod
    def _target_arrays(targets: dict[str, dict[str, float]]) -> dict[str, np.ndarray]:
        return {
            key: np.array([targets[name][key] for name in SERVICE_NAMES], dtype=np.float32)
            for key in ("throughput_mbps", "latency_ms", "reliability")
        }

    def _initialize_population(self) -> None:
        self.cell_ids = np.repeat(np.arange(self.num_cells), self.cell_counts)
        self.positions = self._sample_positions(self.cell_ids)
        probabilities = np.asarray(self.config["service_distribution"], dtype=np.float64)
        self.services = self.rng.choice(3, size=self.num_ues, p=probabilities).astype(np.int64)
        self.speeds = self._sample_speeds(self.cell_ids)
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=self.num_ues)
        self.velocities = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        self.velocities *= self.speeds[:, None]
        self.queues = np.zeros(self.num_ues, dtype=np.float32)
        self.throughput = np.zeros(self.num_ues, dtype=np.float32)
        self.allocated_rate = np.zeros(self.num_ues, dtype=np.float32)
        self.latency = self._target_for("latency_ms").copy()
        self.reliability = np.ones(self.num_ues, dtype=np.float32)
        self.signal_dbm, self.interference_dbm, self.sinr_db = self._radio_conditions()
        self.slot = 0

    def _sample_positions(self, cell_ids: np.ndarray) -> np.ndarray:
        n = len(cell_ids)
        radius = self.cell_radius * np.sqrt(self.rng.uniform(0.02, 0.85, size=n))
        angle = self.rng.uniform(0.0, 2.0 * np.pi, size=n)
        offset = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
        return (self.cell_centers[cell_ids] + offset).astype(np.float32)

    def _sample_speeds(self, cell_ids: np.ndarray) -> np.ndarray:
        ranges = self.speed_ranges[cell_ids]
        return self.rng.uniform(ranges[:, 0], ranges[:, 1]).astype(np.float32)

    def _target_for(self, name: str) -> np.ndarray:
        return self._service_targets[name][self.services]

    def reset(self) -> np.ndarray:
        self._initialize_population()
        return self.states()

    def states(self) -> np.ndarray:
        max_x = self.isd + self.cell_radius
        max_y = np.sqrt(3.0) * 0.5 * self.isd + self.cell_radius
        position = self.positions / np.array([max_x, max_y], dtype=np.float32)
        one_hot = np.eye(3, dtype=np.float32)[self.services]
        numerical = np.column_stack(
            (
                self.allocated_rate / 100.0,
                self.latency / self.max_latency_ms,
                self.reliability,
                self.throughput / 100.0,
            )
        ).astype(np.float32)
        return np.concatenate((position.astype(np.float32), numerical, one_hot), axis=1)

    def contexts(self) -> np.ndarray:
        """Return normalized [signal, interference, traffic load, mobility] contexts."""
        signal = np.clip((self.signal_dbm + 120.0) / 80.0, 0.0, 1.0)
        interference = np.clip((self.interference_dbm + 120.0) / 80.0, 0.0, 1.0)
        traffic = np.clip(self.queues / self.max_queue_mbit, 0.0, 1.0)
        mobility = np.clip(self.speeds / max(float(self.speed_ranges[:, 1].max()), 1.0), 0.0, 1.0)
        return np.column_stack((signal, interference, traffic, mobility)).astype(np.float32)

    def _radio_conditions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        distances = np.linalg.norm(self.positions[:, None, :] - self.cell_centers[None, :, :], axis=2)
        distances = np.maximum(distances, 10.0)
        path_loss = (
            32.4
            + 20.0 * np.log10(self.frequency_ghz)
            + float(self.channel_model["path_loss_distance_coefficient"]) * np.log10(distances)
        )
        shadowing = self.rng.normal(
            0.0, float(self.channel_model["shadowing_std_db"]), size=path_loss.shape
        )
        received = self.tx_power_dbm - 10.0 * np.log10(self.subbands) - path_loss + shadowing
        serving = received[np.arange(self.num_ues), self.cell_ids]
        linear = np.power(10.0, received / 10.0)
        serving_linear = linear[np.arange(self.num_ues), self.cell_ids]
        interference_linear = np.maximum(linear.sum(axis=1) - serving_linear, 1.0e-12)
        interference_dbm = 10.0 * np.log10(interference_linear)
        subband_hz = self.bandwidth_mhz * 1.0e6 / self.subbands
        noise_dbm = -174.0 + 10.0 * np.log10(subband_hz) + self.noise_figure_db
        noise_linear = 10.0 ** (noise_dbm / 10.0)
        sinr = serving_linear / (interference_linear + noise_linear)
        return serving.astype(np.float32), interference_dbm.astype(np.float32), (10.0 * np.log10(sinr)).astype(np.float32)

    def _advance_mobility(self) -> None:
        self.positions += self.velocities * self.slot_duration_s
        # Keep routine motion near the current cell; explicit round events create handovers.
        delta = self.positions - self.cell_centers[self.cell_ids]
        distance = np.linalg.norm(delta, axis=1)
        outside = distance > self.cell_radius * 0.95
        if np.any(outside):
            normal = delta[outside] / distance[outside, None]
            velocity = self.velocities[outside]
            reflected = velocity - 2.0 * (velocity * normal).sum(axis=1, keepdims=True) * normal
            self.velocities[outside] = reflected
            self.positions[outside] = (
                self.cell_centers[self.cell_ids[outside]] + normal * self.cell_radius * 0.94
            )

    def step(self, action_indices: np.ndarray | list[int]) -> StepResult:
        actions = np.asarray(action_indices, dtype=np.int64)
        if actions.shape != (self.num_ues,):
            raise ValueError(f"expected {self.num_ues} actions, got shape {actions.shape}")
        frequency, rate_level, priority = self.action_codec.decode(actions)
        self._advance_mobility()
        self.signal_dbm, self.interference_dbm, self.sinr_db = self._radio_conditions()
        sinr_db = self.sinr_db

        scheduled = np.zeros(self.num_ues, dtype=bool)
        target_rate = self._target_for("throughput_mbps")
        requested_factor = self.rate_factors[rate_level]
        self.allocated_rate = target_rate * requested_factor
        priority_score = priority.astype(np.float32) / 2.0
        queue_pressure = np.clip(self.queues / np.maximum(self.max_queue_mbit, 1.0e-6), 0.0, 1.0)
        service_bias = self.service_bias[self.services]
        predicted_efficiency = np.clip(
            np.log2(1.0 + np.power(10.0, sinr_db / 10.0)),
            0.05,
            float(self.channel_model["spectral_efficiency_max"]),
        )
        predicted_rate = self.bandwidth_mhz / self.subbands * predicted_efficiency * requested_factor
        normalized_prediction = np.clip(predicted_rate / target_rate, 0.0, 3.0)
        utility = (
            float(self.scheduler_model["priority_weight"]) * priority_score
            + float(self.scheduler_model["predicted_rate_weight"]) * normalized_prediction
            + float(self.scheduler_model["queue_weight"]) * queue_pressure
            + service_bias
        )
        utility += self.rng.normal(0.0, 0.01, size=self.num_ues)
        for cell in range(self.num_cells):
            for subband in range(self.subbands):
                candidates = np.flatnonzero((self.cell_ids == cell) & (frequency == subband))
                if candidates.size:
                    order = candidates[np.argsort(utility[candidates])[::-1]]
                    scheduled[order[: self.max_ues_per_subband]] = True

        sharing = np.ones(self.num_ues, dtype=np.float32)
        for cell in range(self.num_cells):
            for subband in range(self.subbands):
                selected = np.flatnonzero(scheduled & (self.cell_ids == cell) & (frequency == subband))
                sharing[selected] = max(1, selected.size)
        # MU-MIMO/spatial reuse is represented as a modest SINR penalty, not bandwidth division.
        effective_sinr = sinr_db - float(
            self.channel_model["multiuser_penalty_db_per_doubling"]
        ) * np.log2(sharing)
        spectral_efficiency = np.clip(
            np.log2(1.0 + np.power(10.0, effective_sinr / 10.0)),
            0.05,
            float(self.channel_model["spectral_efficiency_max"]),
        )
        fast_fading = self.rng.lognormal(
            mean=float(self.channel_model["fast_fading_log_mean"]),
            sigma=float(self.channel_model["fast_fading_log_sigma"]),
            size=self.num_ues,
        )
        capacity_mbps = (
            self.bandwidth_mhz / self.subbands * spectral_efficiency * requested_factor * fast_fading
        )
        capacity_mbps = np.where(scheduled, capacity_mbps, 0.0).astype(np.float32)
        threshold = np.asarray(self.channel_model["bler_thresholds_db"], dtype=np.float32)[rate_level]
        bler = 1.0 / (
            1.0
            + np.exp(
                np.clip(
                    (effective_sinr - threshold) / float(self.channel_model["bler_slope_db"]),
                    -40.0,
                    40.0,
                )
            )
        )
        reliability = np.where(scheduled, 1.0 - bler, 0.0).astype(np.float32)
        # Expected HARQ goodput: unreliable transmissions retain queue backlog.
        delivery_capacity_mbps = capacity_mbps * reliability

        arrivals = self.arrival_rates[self.services] * self.slot_duration_s
        arrivals *= self.rng.lognormal(
            mean=float(self.traffic_model["arrival_log_mean"]),
            sigma=float(self.traffic_model["arrival_log_sigma"]),
            size=self.num_ues,
        )
        self.queues = np.minimum(self.queues + arrivals, self.max_queue_mbit)
        served_mbit = np.minimum(self.queues, delivery_capacity_mbps * self.slot_duration_s)
        self.queues -= served_mbit
        throughput = (served_mbit / self.slot_duration_s).astype(np.float32)
        base_latency = self.base_latency_ms[self.services]
        queue_delay = 1000.0 * self.queues / np.maximum(delivery_capacity_mbps, 0.1)
        contention_penalty = np.where(
            scheduled,
            float(self.scheduler_model["sharing_latency_penalty_ms"]) * (sharing - 1.0),
            float(self.scheduler_model["unscheduled_latency_penalty_ms"]),
        )
        latency = np.clip(base_latency + queue_delay + contention_penalty, 0.1, self.max_latency_ms)

        self.throughput = throughput
        self.latency = latency.astype(np.float32)
        self.reliability = reliability

        throughput_target = self._target_for("throughput_mbps")
        latency_target = self._target_for("latency_ms")
        reliability_target = self._target_for("reliability")
        rewards = (
            reliability / reliability_target
            + throughput / throughput_target
            - self.latency / latency_target
        ).astype(np.float32)
        satisfied = (
            (throughput >= throughput_target)
            & (self.latency <= latency_target)
            & (reliability >= reliability_target)
        )
        self.slot += 1
        # Episodes segment optimization; the radio process itself is continuing.
        terminated = np.zeros(self.num_ues, dtype=bool)
        metrics = {
            "throughput_mbps": throughput.copy(),
            "latency_ms": self.latency.copy(),
            "reliability": reliability.copy(),
            "qos_satisfied": satisfied.astype(np.float32),
            "scheduled": scheduled.astype(np.float32),
            "cell_id": self.cell_ids.copy(),
            "service_id": self.services.copy(),
        }
        return StepResult(self.states(), rewards, terminated, metrics)

    def apply_round_dynamics(self, handover_fraction: float, new_fraction: float) -> EventBatch:
        num_handover = int(round(self.num_ues * handover_fraction))
        num_new = int(round(self.num_ues * new_fraction))
        order = self.rng.permutation(self.num_ues)
        handover = np.sort(order[:num_handover])
        new = np.sort(order[num_handover : num_handover + num_new])

        if handover.size:
            old_cells = self.cell_ids[handover].copy()
            offsets = self.rng.integers(1, self.num_cells, size=handover.size)
            target_cells = (old_cells + offsets) % self.num_cells
            self.cell_ids[handover] = target_cells
            self.positions[handover] = self._sample_positions(target_cells)
            self.speeds[handover] = self._sample_speeds(target_cells)

        if new.size:
            target_cells = self.rng.integers(0, self.num_cells, size=new.size)
            self.cell_ids[new] = target_cells
            self.positions[new] = self._sample_positions(target_cells)
            if self.config["new_service_assignment"] == "balanced":
                service_offset = int(self.rng.integers(0, 3))
                new_services = (np.arange(new.size, dtype=np.int64) + service_offset) % 3
                self.rng.shuffle(new_services)
            else:
                probabilities = np.asarray(self.config["service_distribution"], dtype=np.float64)
                new_services = self.rng.choice(3, size=new.size, p=probabilities)
            self.services[new] = new_services
            self.speeds[new] = self._sample_speeds(target_cells)
            self.queues[new] = 0.0
            self.throughput[new] = 0.0
            self.allocated_rate[new] = 0.0
            self.latency[new] = self._target_for("latency_ms")[new]
            self.reliability[new] = 1.0

        changed = np.concatenate((handover, new))
        if changed.size:
            angles = self.rng.uniform(0.0, 2.0 * np.pi, size=changed.size)
            self.velocities[changed, 0] = np.cos(angles) * self.speeds[changed]
            self.velocities[changed, 1] = np.sin(angles) * self.speeds[changed]
        self.signal_dbm, self.interference_dbm, self.sinr_db = self._radio_conditions()
        return EventBatch(handover, new)

    def heuristic_actions(self) -> np.ndarray:
        frequency = np.arange(self.num_ues, dtype=np.int64) % self.subbands
        thresholds = np.asarray(self.channel_model["bler_thresholds_db"], dtype=np.float32)
        rate = np.digitize(self.sinr_db, thresholds[1:]).astype(np.int64)
        priority = np.choose(self.services, [1, 2, 0]).astype(np.int64)
        return np.array(
            [
                self.action_codec.encode(int(f), int(r), int(p))
                for f, r, p in zip(frequency, rate, priority, strict=True)
            ],
            dtype=np.int64,
        )
