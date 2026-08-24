from pathlib import Path

import pytest

from tfl_coran.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_merges_and_validates() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    assert config["environment"]["cell_ue_counts"] == [4, 4, 4]
    assert config["environment"]["bandwidth_mhz"] == 100.0
    assert config["gmm"]["clusters"] == 3


def test_invalid_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="method"):
        load_config(overrides={"method": "not-a-method"})


def test_unknown_key_is_rejected_instead_of_silently_ignored() -> None:
    with pytest.raises(KeyError, match="unknown configuration key"):
        load_config(overrides={"topology": {"cells": 3}})


@pytest.mark.parametrize(
    "overrides",
    [
        {"agent": {"target_update_episodes": 0}},
        {"experiment": {"evaluation_episodes": 0}},
        {"vgae": {"snapshots": 0}},
        {"environment": {"slot_duration_ms": 0.0}},
        {"environment": {"service_targets": {"embb": {"throughput_mbps": 0.0}}}},
        {"environment": {"service_targets": {"urllc": {"latency_ms": -1.0}}}},
        {"environment": {"service_targets": {"mmtc": {"reliability": 1.01}}}},
    ],
    ids=[
        "target-update-cadence",
        "evaluation-episodes",
        "vgae-snapshots",
        "slot-duration",
        "throughput-target",
        "latency-target",
        "reliability-target",
    ],
)
def test_invalid_runtime_parameters_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        load_config(overrides=overrides)
