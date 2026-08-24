from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MetricAccumulator:
    sums: dict[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, metrics: dict[str, np.ndarray]) -> None:
        n = int(len(metrics["throughput_mbps"]))
        for name in ("throughput_mbps", "latency_ms", "reliability", "qos_satisfied", "scheduled"):
            self.sums[name] = self.sums.get(name, 0.0) + float(np.asarray(metrics[name]).sum())
        self.count += n

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        result = {name: value / self.count for name, value in self.sums.items()}
        result["qos_satisfaction_pct"] = 100.0 * result.pop("qos_satisfied")
        result["scheduled_pct"] = 100.0 * result.pop("scheduled")
        return result


@dataclass
class _PendingEvent:
    event_type: str
    start_slot: int
    consecutive: int = 0


class AdaptationTracker:
    def __init__(self, consecutive_slots: int, slot_duration_ms: float) -> None:
        self.required = max(1, int(consecutive_slots))
        self.slot_duration_s = float(slot_duration_ms) / 1000.0
        self.pending: dict[int, _PendingEvent] = {}
        self.records: list[dict[str, Any]] = []
        self.last_slot = 0

    def _record_censored(self, index: int, event: _PendingEvent, censor_slot: int) -> None:
        followup_slots = max(0, int(censor_slot) - event.start_slot)
        self.records.append(
            {
                "ue_index": index,
                "event_type": event.event_type,
                "adaptation_s": None,
                "followup_s": followup_slots * self.slot_duration_s,
                "censored": True,
            }
        )

    def start(self, indices: np.ndarray, event_type: str, slot: int) -> None:
        self.last_slot = max(self.last_slot, int(slot))
        for index in map(int, indices):
            if index in self.pending:
                self._record_censored(index, self.pending[index], int(slot))
            self.pending[index] = _PendingEvent(event_type=event_type, start_slot=slot)

    def update(self, satisfied: np.ndarray, slot: int) -> None:
        self.last_slot = max(self.last_slot, int(slot))
        completed: list[int] = []
        for index, event in self.pending.items():
            event.consecutive = event.consecutive + 1 if bool(satisfied[index]) else 0
            if event.consecutive >= self.required:
                elapsed_slots = max(1, slot - event.start_slot)
                self.records.append(
                    {
                        "ue_index": index,
                        "event_type": event.event_type,
                        "adaptation_s": elapsed_slots * self.slot_duration_s,
                        "followup_s": elapsed_slots * self.slot_duration_s,
                        "censored": False,
                    }
                )
                completed.append(index)
        for index in completed:
            self.pending.pop(index, None)

    def finalize(self) -> list[dict[str, Any]]:
        for index, event in list(self.pending.items()):
            self._record_censored(index, event, self.last_slot)
        self.pending.clear()
        return self.records

    def summary(self) -> dict[str, float | int | None]:
        output: dict[str, float | int | None] = {}
        for event_type in ("handover", "activation"):
            rows = [row for row in self.records if row["event_type"] == event_type]
            observed = [float(row["adaptation_s"]) for row in rows if not row["censored"]]
            penalized = [
                float(row["followup_s"] if row["censored"] else row["adaptation_s"])
                for row in rows
            ]
            censored = sum(bool(row["censored"]) for row in rows)
            output[f"{event_type}_adaptation_s"] = float(np.mean(observed)) if observed else None
            output[f"{event_type}_events"] = len(rows)
            output[f"{event_type}_censored"] = censored
            output[f"{event_type}_completion_rate"] = (
                float((len(rows) - censored) / len(rows)) if rows else None
            )
            output[f"{event_type}_adaptation_penalized_s"] = (
                float(np.mean(penalized)) if penalized else None
            )
        return output


class GroupMetricAccumulator:
    def __init__(self) -> None:
        self._groups: dict[tuple[str, str], MetricAccumulator] = {}

    def update(
        self,
        metrics: dict[str, np.ndarray],
        cell_labels: list[str],
        service_names: np.ndarray,
    ) -> None:
        cell_ids = np.asarray(metrics["cell_id"], dtype=np.int64)
        service_ids = np.asarray(metrics["service_id"], dtype=np.int64)
        for kind, ids, labels in (
            ("cell", cell_ids, np.asarray(cell_labels)),
            ("service", service_ids, service_names),
        ):
            for value in np.unique(ids):
                mask = ids == value
                key = (kind, str(labels[int(value)]))
                accumulator = self._groups.setdefault(key, MetricAccumulator())
                accumulator.update({name: np.asarray(array)[mask] for name, array in metrics.items() if name not in {"cell_id", "service_id"}})

    def rows(self) -> list[dict[str, Any]]:
        return [
            {"group_type": key[0], "group": key[1], **accumulator.summary()}
            for key, accumulator in sorted(self._groups.items())
        ]
