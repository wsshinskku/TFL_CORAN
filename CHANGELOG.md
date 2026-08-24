# Changelog

## 1.0.0 - 2026-08-19

- Connected DDQN, VGAE, GMM, personalized FL, transfer, mobility, and activation end to end.
- Added paper-scale and smoke profiles with explicit reported/assumed settings.
- Corrected episode termination, 1 ms mobility, per-cell scheduling, stale unscheduled metrics, and CPU handling.
- Replaced scale-unstable dense topology with standardized, sparsified symmetric kNN graphs.
- Added shared global aggregation and a configurable personalized-delta interpretation.
- Added Heuristic, DRL, FDRL, CFDRL, ablation, evaluation, checkpointing, tests, and CI.
- Added reproducibility, limitation, migration, citation, and third-party licensing documentation.
- Added strict config validation, safe VGAE checkpoints, censor-aware adaptation reporting,
  and matched centralized/federated warmup, update, and replay-capacity budgets.
