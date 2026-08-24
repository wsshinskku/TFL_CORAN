# External O-RAN and channel integration

The included environment is an algorithmic simulator. UERANSIM, Open5GS and
QuaDRiGa are not imported or silently emulated.

A real/trace adapter should provide, per active UE and slot:

```text
ue_id, serving_cell, x_m, y_m, service_class,
allocated_rate_mbps, throughput_mbps, latency_ms, reliability,
received_signal_dbm, interference_dbm, traffic_load_mbit, speed_mps
```

It must accept `(subband, rate_level, priority)` requests and return the next
local state, Eq. (1) reward, termination flag and QoS metrics. The RIC boundary
must expose only model tensors and SITM summaries; replay/raw packet traces stay
at the UE-side adapter.

Recommended deployment boundary:

```text
QuaDRiGa trace files -> channel adapter -> Python environment
UERANSIM/Open5GS telemetry -> UE/gNB adapter -> Python environment
RIC process <-> serialized model update + SITM DTO
```

Pin exact upstream revisions and preserve their license obligations. Do not
vendor those projects into this MIT repository. See `THIRD_PARTY_NOTICES.md`.
