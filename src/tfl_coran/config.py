from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "method": "tfl_coran",
    "device": "auto",
    "environment": {
        "cell_ue_counts": [150, 120, 105],
        "cell_labels": ["urban", "commercial", "school"],
        "inter_site_distance_m": 500.0,
        "carrier_frequency_ghz": 3.5,
        "bandwidth_mhz": 100.0,
        "subbands": 6,
        "slot_duration_ms": 1.0,
        "episode_slots": 200,
        "max_ues_per_subband": 8,
        "tx_power_dbm": 46.0,
        "noise_figure_db": 7.0,
        "max_latency_ms": 100.0,
        "max_queue_mbit": 2.0,
        "service_distribution": [0.34, 0.33, 0.33],
        "new_service_assignment": "balanced",
        "service_targets": {
            "embb": {"throughput_mbps": 20.0, "latency_ms": 15.0, "reliability": 0.95},
            "urllc": {"throughput_mbps": 10.0, "latency_ms": 5.0, "reliability": 0.999},
            "mmtc": {"throughput_mbps": 5.0, "latency_ms": 10.0, "reliability": 0.90},
        },
        "arrival_rate_mbps": {"embb": 24.0, "urllc": 12.0, "mmtc": 6.0},
        "speed_mps": {"urban": [3.0, 16.0], "commercial": [1.0, 8.0], "school": [0.2, 3.0]},
        "channel_model": {
            "path_loss_distance_coefficient": 30.0,
            "shadowing_std_db": 4.0,
            "spectral_efficiency_max": 7.4,
            "multiuser_penalty_db_per_doubling": 1.8,
            "fast_fading_log_mean": -0.04,
            "fast_fading_log_sigma": 0.16,
            "bler_thresholds_db": [-1.0, 5.0, 11.0],
            "bler_slope_db": 1.6,
        },
        "scheduler_model": {
            "rate_factors": [0.60, 0.82, 1.0],
            "priority_weight": 1.6,
            "predicted_rate_weight": 1.0,
            "queue_weight": 1.0,
            "service_bias": {"embb": 0.45, "urllc": 1.0, "mmtc": 0.25},
            "sharing_latency_penalty_ms": 0.15,
            "unscheduled_latency_penalty_ms": 2.0,
        },
        "traffic_model": {
            "arrival_log_mean": -0.03,
            "arrival_log_sigma": 0.25,
            "base_latency_ms": {"embb": 3.0, "urllc": 1.0, "mmtc": 4.0},
        },
    },
    "agent": {
        "hidden_dims": [128, 128],
        "learning_rate": 0.001,
        "discount": 0.99,
        "batch_size": 64,
        "target_update_episodes": 10,
        "replay_capacity": 100000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay_episodes": 100,
        "warmup_steps": 128,
        "gradient_steps_per_slot": 1,
        "max_grad_norm": 10.0,
    },
    "vgae": {
        "hidden_dim": 64,
        "latent_dim": 32,
        "epochs": 100,
        "learning_rate": 0.01,
        "snapshots": 12,
        "graph_neighbors": 6,
        "kl_weight": 1.0,
    },
    "gmm": {
        "clusters": 3,
        "covariance_type": "full",
        "max_iter": 100,
        "reg_covar": 1.0e-6,
        "n_init": 3,
    },
    "federation": {
        "episodes_per_round": 5,
        "cluster_refresh_rounds": 10,
        "handover_fraction": 0.10,
        "new_ue_fraction": 0.03,
        "transfer_delta": 0.5,
        "delta_reference": "dispatch_base",
        "sync_target_on_dispatch": False,
        "reset_optimizer_on_dispatch": False,
    },
    "experiment": {
        "episodes": 50,
        "evaluation_episodes": 3,
        "adaptation_consecutive_slots": 3,
        "checkpoint_every_rounds": 10,
        "save_client_models": False,
    },
    "toggles": {"transfer": True, "vgae": True, "gmm": True},
}


def _deep_merge(
    base: dict[str, Any], override: dict[str, Any], path: tuple[str, ...] = ()
) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key not in merged:
            dotted = ".".join((*path, key))
            raise KeyError(f"unknown configuration key: {dotted}")
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value, (*path, key))
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate(config: dict[str, Any]) -> None:
    method = config["method"]
    allowed = {"heuristic", "drl", "fdrl", "cfdrl", "tfl_coran"}
    if method not in allowed:
        raise ValueError(f"method must be one of {sorted(allowed)}, got {method!r}")
    env = config["environment"]
    counts = env["cell_ue_counts"]
    labels = env["cell_labels"]
    if len(counts) != 3 or len(labels) != 3:
        raise ValueError("this reproduction currently models exactly three O-RAN cells")
    if any(int(n) <= 0 for n in counts):
        raise ValueError("all cell_ue_counts must be positive")
    positive_environment_values = {
        "inter_site_distance_m": env["inter_site_distance_m"],
        "carrier_frequency_ghz": env["carrier_frequency_ghz"],
        "bandwidth_mhz": env["bandwidth_mhz"],
        "slot_duration_ms": env["slot_duration_ms"],
        "max_latency_ms": env["max_latency_ms"],
        "max_queue_mbit": env["max_queue_mbit"],
    }
    if any(float(value) <= 0.0 for value in positive_environment_values.values()):
        raise ValueError("environment distances, bandwidth, slot duration, and limits must be positive")
    if (
        int(env["subbands"]) <= 0
        or int(env["episode_slots"]) <= 0
        or int(env["max_ues_per_subband"]) <= 0
    ):
        raise ValueError("subbands, episode_slots, and max_ues_per_subband must be positive")
    targets = env["service_targets"]
    for service in ("embb", "urllc", "mmtc"):
        values = targets[service]
        if float(values["throughput_mbps"]) <= 0.0 or float(values["latency_ms"]) <= 0.0:
            raise ValueError("service throughput and latency targets must be positive")
        if not 0.0 < float(values["reliability"]) <= 1.0:
            raise ValueError("service reliability targets must lie in (0, 1]")
        if float(env["arrival_rate_mbps"][service]) < 0.0:
            raise ValueError("traffic arrival rates must be non-negative")
    for label in labels:
        speed_range = env["speed_mps"][label]
        if (
            len(speed_range) != 2
            or float(speed_range[0]) < 0.0
            or float(speed_range[1]) < float(speed_range[0])
        ):
            raise ValueError("speed ranges must be non-negative [minimum, maximum] pairs")
    channel = env["channel_model"]
    if len(channel["bler_thresholds_db"]) != 3:
        raise ValueError("channel_model.bler_thresholds_db must contain three values")
    if (
        float(channel["path_loss_distance_coefficient"]) <= 0.0
        or float(channel["shadowing_std_db"]) < 0.0
        or float(channel["spectral_efficiency_max"]) <= 0.0
        or float(channel["multiuser_penalty_db_per_doubling"]) < 0.0
        or float(channel["fast_fading_log_sigma"]) < 0.0
        or float(channel["bler_slope_db"]) <= 0.0
    ):
        raise ValueError("channel model scales must be physically valid")
    scheduler = env["scheduler_model"]
    if len(scheduler["rate_factors"]) != 3 or any(
        float(value) <= 0.0 for value in scheduler["rate_factors"]
    ):
        raise ValueError("scheduler_model.rate_factors must contain three positive values")
    if any(float(scheduler[key]) < 0.0 for key in ("priority_weight", "predicted_rate_weight", "queue_weight")):
        raise ValueError("scheduler utility weights must be non-negative")
    if any(float(value) < 0.0 for value in scheduler["service_bias"].values()):
        raise ValueError("scheduler service biases must be non-negative")
    if any(float(value) <= 0.0 for value in env["traffic_model"]["base_latency_ms"].values()):
        raise ValueError("traffic base latencies must be positive")
    if float(env["traffic_model"]["arrival_log_sigma"]) < 0.0:
        raise ValueError("traffic arrival_log_sigma must be non-negative")
    gmm = config["gmm"]
    if int(gmm["clusters"]) <= 0:
        raise ValueError("gmm.clusters must be positive")
    if sum(counts) < gmm["clusters"]:
        raise ValueError("number of UEs must be at least the number of GMM clusters")
    if int(gmm["max_iter"]) <= 0 or int(gmm["n_init"]) <= 0 or float(gmm["reg_covar"]) < 0.0:
        raise ValueError("GMM iteration counts must be positive and regularization non-negative")
    if gmm["covariance_type"] not in {"full", "tied", "diag", "spherical"}:
        raise ValueError("unsupported GMM covariance_type")
    probs = env["service_distribution"]
    if len(probs) != 3 or any(float(value) < 0.0 for value in probs) or abs(sum(probs) - 1.0) > 1.0e-6:
        raise ValueError("service_distribution must contain three probabilities summing to one")
    if env["new_service_assignment"] not in {"balanced", "distribution"}:
        raise ValueError("new_service_assignment must be balanced or distribution")
    fed = config["federation"]
    if not 0.0 <= fed["transfer_delta"] <= 1.0:
        raise ValueError("federation.transfer_delta must lie in [0, 1]")
    if fed["delta_reference"] not in {"dispatch_base", "paper_global"}:
        raise ValueError("federation.delta_reference must be dispatch_base or paper_global")
    event_fraction = float(fed["handover_fraction"]) + float(fed["new_ue_fraction"])
    if (
        not 0.0 <= float(fed["handover_fraction"]) <= 1.0
        or not 0.0 <= float(fed["new_ue_fraction"]) <= 1.0
        or event_fraction > 1.0
    ):
        raise ValueError("handover_fraction and new_ue_fraction must be non-negative and sum to <= 1")
    if int(fed["episodes_per_round"]) <= 0 or int(fed["cluster_refresh_rounds"]) <= 0:
        raise ValueError("federation cadences must be positive")
    agent = config["agent"]
    if not agent["hidden_dims"] or any(int(width) <= 0 for width in agent["hidden_dims"]):
        raise ValueError("agent.hidden_dims must contain positive layer widths")
    if int(agent["batch_size"]) <= 0 or int(agent["replay_capacity"]) < int(agent["batch_size"]):
        raise ValueError("agent replay_capacity must be at least batch_size")
    if (
        float(agent["learning_rate"]) <= 0.0
        or not 0.0 <= float(agent["discount"]) < 1.0
        or int(agent["target_update_episodes"]) <= 0
        or int(agent["epsilon_decay_episodes"]) <= 0
        or int(agent["warmup_steps"]) < 0
        or int(agent["gradient_steps_per_slot"]) <= 0
        or float(agent["max_grad_norm"]) <= 0.0
    ):
        raise ValueError("agent learning rates, cadences, budgets, and limits must be valid")
    epsilon_start = float(agent["epsilon_start"])
    epsilon_end = float(agent["epsilon_end"])
    if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
        raise ValueError("agent epsilon values must satisfy 0 <= end <= start <= 1")
    vgae = config["vgae"]
    if any(
        int(vgae[key]) <= 0
        for key in ("hidden_dim", "latent_dim", "epochs", "snapshots", "graph_neighbors")
    ) or float(vgae["learning_rate"]) <= 0.0:
        raise ValueError("VGAE dimensions, snapshots, epochs, neighbors, and learning rate must be positive")
    if float(vgae["kl_weight"]) < 0.0:
        raise ValueError("vgae.kl_weight must be non-negative")
    experiment = config["experiment"]
    if int(experiment["episodes"]) <= 0 or int(experiment["evaluation_episodes"]) <= 0:
        raise ValueError("training and evaluation episode counts must be positive")
    if int(experiment["adaptation_consecutive_slots"]) <= 0:
        raise ValueError("adaptation_consecutive_slots must be positive")
    if int(experiment["checkpoint_every_rounds"]) < 0:
        raise ValueError("checkpoint_every_rounds must be non-negative")


def load_configs(
    paths: list[str | Path] | tuple[str | Path, ...], overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Merge YAML files from left to right over the documented defaults."""
    config = deepcopy(DEFAULT_CONFIG)
    for path in paths:
        with Path(path).expanduser().open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if loaded:
            if not isinstance(loaded, dict):
                raise TypeError("the YAML root must be a mapping")
            config = _deep_merge(config, loaded)
    if overrides:
        config = _deep_merge(config, overrides)
    _validate(config)
    return config


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load one YAML configuration over the documented defaults."""
    return load_configs([] if path is None else [path], overrides=overrides)


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
