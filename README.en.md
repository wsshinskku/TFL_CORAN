# TFL-CORAN

[한국어](README.md) | English

TFL-CORAN is a federated reinforcement-learning framework for UE-level traffic control in 5G O-RAN. Each UE trains a local DDQN, while the non-RT RIC derives soft cluster memberships from VGAE embeddings and a GMM. Cluster and personalized models are aggregated at the RIC, and model transfer is used to initialize UEs after handover or activation.

This repository contains the TFL-CORAN algorithms, simulation environment, baselines, ablations, and multi-seed evaluation scripts.

## Components

- UE-local DDQN with online action selection and target-network evaluation
- QoS reward: `reliability / target + throughput / target - latency / target`
- SITM context: signal, interference, traffic load, and mobility
- symmetric kNN graph and VGAE encoder
- GMM soft memberships for clustered and personalized aggregation
- destination-cell model transfer for handovers and newly activated UEs
- separate slot, episode, FL-round, and cluster-refresh time scales
- Heuristic, DRL, FDRL, CFDRL, and TFL-CORAN experiments

See [`docs/ALGORITHM.md`](docs/ALGORITHM.md) for the equation-to-code mapping.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[dev]'
tfl-coran doctor
```

## Quick check

`configs/smoke.yaml` exercises the complete pipeline with a small CPU configuration.

```bash
tfl-coran run -c configs/smoke.yaml -o runs/smoke
pytest -q
```

A run writes the resolved configuration, environment metadata, historical contexts, VGAE loss, training and evaluation metrics, adaptation events, memberships, and model checkpoints.

## Experiments

The paper profile contains 375 UE-local learners. Estimate the workload before starting a full run.

```bash
tfl-coran estimate -c configs/paper.yaml

tfl-coran pretrain-vgae \
  -c configs/paper.yaml \
  -o runs/shared/vgae.pt

tfl-coran benchmark \
  -c configs/paper.yaml \
  --methods heuristic drl fdrl cfdrl tfl_coran \
  --vgae-checkpoint runs/shared/vgae.pt \
  -o runs/paper_seed42
```

Use `configs/paper_fast.yaml` for a shorter functional run.

```bash
tfl-coran ablate -c configs/paper_fast.yaml -o runs/ablation

tfl-coran reproduce \
  -c configs/paper.yaml \
  --methods heuristic drl fdrl cfdrl tfl_coran \
  --seeds 0 1 2 3 4 \
  -o runs/reproduction
```

## Methods

| Method | Definition |
|---|---|
| Heuristic | SINR-based rate level, service priority, and round-robin subband assignment |
| DRL | One shared DDQN with pooled UE transitions |
| FDRL | UE-local DDQN with FedAvg |
| CFDRL | Hard-cluster FL using the maximum GMM posterior |
| TFL-CORAN | VGAE-GMM soft memberships, personalized FL, and model transfer |

The component ablations use the following rules:

- Variant A: transfer off, VGAE/GMM on
- Variant B: GMM on standardized raw contexts
- Variant C: deterministic hard KMeans on VGAE embeddings
- Variant D: uniform memberships and FedAvg

The manuscript does not specify replacement operations for every disabled component. The rules above define the implementation used by the ablation runner.

## Reproducibility

The manuscript specifies the main equations, topology, and training cadence, but does not fix every channel, traffic, reliability, network, and action-space parameter required for an executable simulator. Parameters not fixed by the manuscript remain configurable. Their defaults and rationale are documented in [`configs/paper.yaml`](configs/paper.yaml) and [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md). Each run writes the effective configuration to `resolved_config.yaml`.

The values under `paper_reported/` are references transcribed from the manuscript tables and are not used as inputs to the experiments. Generated results are written under `runs/`. The original UERANSIM, Open5GS, and QuaDRiGa configurations and traces are not included; the default backend is therefore a self-contained Python simulator.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`VALIDATION.md`](VALIDATION.md) for the experiment protocol and test coverage.

## External systems

The telemetry schema and adapter boundary for external simulators or a testbed are documented in [`docs/EXTERNAL_SIMULATORS.md`](docs/EXTERNAL_SIMULATORS.md). UERANSIM, Open5GS, and QuaDRiGa are not vendored and remain subject to their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

MIT License, Copyright (c) 2025 Wooseok Daniel Shin.
