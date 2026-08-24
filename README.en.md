# TFL-CORAN

[한국어](README.md) | English

This is the completed, executable release of the code accompanying
**“Transfer-enhanced Federated Learning with Dynamic Clustering for Traffic
Management in 5G Open RAN.”** It upgrades the original prototype by wiring UE
DDQN, offline VGAE, soft GMM clustering, personalized federated aggregation,
handover/new-UE transfer, baselines, ablations, metrics, tests and CI into one
multi-timescale pipeline.

> **Scope:** this is an algorithmic and protocol reproduction. The manuscript
> does not provide the channel/traffic traces, reliability targets, DDQN hidden
> layers, full action grid or VGAE training data needed for exact Table 3/4
> replication. Reported values and major implementation assumptions that can
> affect results are marked in `configs/paper.yaml` and
> `docs/ASSUMPTIONS.md`. The run's `resolved_config.yaml` is authoritative for
> every effective value, including inherited defaults; generated metrics are
> never presented as the paper's reported results.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[dev]'

tfl-coran doctor
tfl-coran run -c configs/smoke.yaml -o runs/smoke
pytest -q
```

## Main commands

```bash
# Inspect full-profile work and memory scale
tfl-coran estimate -c configs/paper.yaml

# Train the offline graph encoder
tfl-coran pretrain-vgae -c configs/paper.yaml -o runs/shared/vgae.pt

# Compare all paper baselines under one seed
tfl-coran benchmark -c configs/paper_fast.yaml \
  --methods heuristic drl fdrl cfdrl tfl_coran -o runs/benchmark

# Run the paper component toggles
tfl-coran ablate -c configs/paper_fast.yaml -o runs/ablation

# Five-seed mean/std/95% CI workflow
tfl-coran reproduce -c configs/paper.yaml \
  --methods heuristic drl fdrl cfdrl tfl_coran \
  --seeds 0 1 2 3 4 -o runs/reproduction
```

Each run writes its resolved configuration, software metadata, historical
contexts, VGAE losses, training/evaluation CSVs, adaptation events, JSON
summary and checkpoints. Adaptation reports completed-event time, completion
rate, and a censor-aware horizon-penalized value.

## Implemented methods

| Method | Definition |
|---|---|
| Heuristic | Non-learning SINR-binned-rate, service-priority and static round-robin frequency-spreading policy |
| DRL | One shared DDQN and pooled replay |
| FDRL | Per-UE DDQN with uniform memberships/FedAvg |
| CFDRL | Hard GMM assignments and per-cluster aggregation |
| TFL-CORAN | VGAE-GMM soft memberships, personalized FL and transfer |

The manuscript does not define every substitute used when an ablation disables
a component. This release fixes the interpretations as follows: Variant A
disables transfer only; Variant B uses standardized raw context with GMM;
Variant C uses VGAE embeddings with seeded deterministic hard KMeans; and
Variant D uses uniform memberships/FedAvg. Transfer is disabled in A-D. The
hard-KMeans choice for Variant C is an implementation interpretation, not a
claim about an otherwise unspecified manuscript detail.

The representation graph starts with six directed nearest-neighbor queries per
node and then takes a symmetric union. Its realized average and maximum degree
can therefore exceed six; it approximates, but does not enforce, the
manuscript's reported average degree of about six.

The default environment is a self-contained algorithmic simulator. It does not
claim to be a hidden UERANSIM/Open5GS/QuaDRiGa integration. See
`docs/EXTERNAL_SIMULATORS.md` for the adapter boundary and
`THIRD_PARTY_NOTICES.md` for licensing.

## Documentation

- `docs/ALGORITHM.md`: equation-to-code mapping
- `docs/ASSUMPTIONS.md`: reported values versus implementation choices
- `docs/REPRODUCIBILITY.md`: multi-seed protocol and provenance
- `VALIDATION.md`: release checks and scientific validation boundary
- `MIGRATION.md`: changes from the earlier repository layout
- `paper_reported/`: reference-only transcriptions of Tables 3 and 4

MIT License, Copyright (c) 2025 Wooseok Daniel Shin.
