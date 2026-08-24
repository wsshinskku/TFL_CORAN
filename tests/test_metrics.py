import numpy as np
import pytest

from tfl_coran.experiments.metrics import AdaptationTracker


def test_adaptation_requires_consecutive_slots_and_keeps_censoring() -> None:
    tracker = AdaptationTracker(consecutive_slots=2, slot_duration_ms=1.0)
    tracker.start(np.array([0, 1]), "handover", slot=10)
    tracker.update(np.array([True, False]), slot=11)
    tracker.update(np.array([True, False]), slot=12)
    records = tracker.finalize()
    completed = next(row for row in records if row["ue_index"] == 0)
    censored = next(row for row in records if row["ue_index"] == 1)
    assert completed["adaptation_s"] == 0.002
    assert not completed["censored"]
    assert censored["censored"]


def test_adaptation_summary_penalizes_censored_followup_horizon() -> None:
    tracker = AdaptationTracker(consecutive_slots=1, slot_duration_ms=1.0)
    tracker.start(np.array([0, 1]), "handover", slot=10)
    tracker.update(np.array([True, False]), slot=11)
    tracker.update(np.array([False, False]), slot=14)

    records = tracker.finalize()
    completed = next(row for row in records if row["ue_index"] == 0)
    censored = next(row for row in records if row["ue_index"] == 1)
    assert completed == {
        "ue_index": 0,
        "event_type": "handover",
        "adaptation_s": pytest.approx(0.001),
        "followup_s": pytest.approx(0.001),
        "censored": False,
    }
    assert censored == {
        "ue_index": 1,
        "event_type": "handover",
        "adaptation_s": None,
        "followup_s": pytest.approx(0.004),
        "censored": True,
    }

    summary = tracker.summary()
    assert summary["handover_adaptation_s"] == pytest.approx(0.001)
    assert summary["handover_events"] == 2
    assert summary["handover_censored"] == 1
    assert summary["handover_completion_rate"] == pytest.approx(0.5)
    assert summary["handover_adaptation_penalized_s"] == pytest.approx(0.0025)
    assert summary["activation_events"] == 0
    assert summary["activation_completion_rate"] is None
    assert summary["activation_adaptation_penalized_s"] is None
