# Reproducibility guide

## Profiles

- `smoke.yaml`: correctness/CI only; 12 UEs and a few slots.
- `paper_fast.yaml`: laptop demonstration with reduced population and budget.
- `paper.yaml`: reported topology/cadence/model values plus major marked
  assumptions; inherited defaults are captured in each run's resolved config.

Check cost before a full run:

```bash
tfl-coran estimate -c configs/paper.yaml
```

The replay buffers grow lazily, but 375 local DDQN models and millions of local
optimizer steps can still require hours and substantial memory.

## Five-seed protocol

```bash
tfl-coran reproduce \
  -c configs/paper.yaml \
  --methods heuristic drl fdrl cfdrl tfl_coran \
  --seeds 0 1 2 3 4 \
  -o runs/reproduction
```

The command writes every seed/method run plus means, sample standard deviations
and normal-approximation 95% confidence intervals. Early stopping is not used.

## Saved provenance

Every run stores the fully merged resolved config, seed, Git commit and dirty
flag when available, Python/OS/package versions, historical contexts, VGAE losses,
metrics, adaptation events and checkpoints. The resolved config, rather than
the profile file alone, is authoritative for inherited channel, scheduler and
traffic defaults.
Generated metrics are kept separate from `paper_reported/` reference values.

The implementation seeds Python, NumPy, PyTorch, environment, client replay,
GMM and historical-data streams. CPU runs with the same environment and package
versions are deterministic in the test suite. Hardware/backend differences may
still produce small floating-point changes.

## Fairness notes

All methods use one environment implementation, topology, action space, DDQN
backbone, episode count and event ratios. They are independently seeded with
the same seed. This is common-random initialization, but not a fully precomputed
exogenous scenario tape: action-dependent queues and scheduling necessarily
diverge.

Central DRL receives pooled transitions, so it waits until each UE-equivalent
stream has reached the same warmup horizon as a local client and then performs
the same aggregate number of optimizer steps and exposes the same total replay
capacity as all federated clients. Online
adaptation events stop before the final aggregation/evaluation boundary;
adaptation is finalized from the training window, while headline evaluation is
deterministic and frozen. Always report completion rate and the censor-aware
horizon-penalized value alongside the completed-event adaptation mean.

The smoke profile is intended for functional checks. Quantitative comparisons
should use multiple seeds and report variance and censoring. The paper's Table 3 averages are not all
recoverable from displayed subgroup rows, so this code uses transparent
UE-slot weighted reductions.

The heuristic uses an explicit SINR-bin/service-priority/static-frequency
mapping because the manuscript does not publish its exact action rule. Table 4
ablations also require documented substitutes: raw-context GMM for B,
VGAE-embedding deterministic hard KMeans for C, and uniform FedAvg for D. See
`ASSUMPTIONS.md` before interpreting cross-method or ablation differences.

Graph construction queries six outgoing neighbors and then takes a symmetric
union. The realized average and maximum degree can exceed six, so simulator
communication/graph overhead should not be presented as an exact realization
of the manuscript's reported average degree of about six.
