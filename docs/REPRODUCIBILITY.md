# Reproducibility guide

## Profiles

- `smoke.yaml`: CPU integration checks with 12 UEs and short training windows.
- `paper_fast.yaml`: laptop experiment with 75 UEs and a reduced training budget.
- `paper.yaml`: 375 UEs, 500 episodes, and 100 federated rounds.

Estimate the full workload before execution:

```bash
tfl-coran estimate -c configs/paper.yaml
```

Replay buffers grow lazily. The full profile trains 375 local DDQN models and performs millions of local optimizer steps, so runtime and memory depend strongly on hardware.

## Five-seed protocol

```bash
tfl-coran reproduce -c configs/paper.yaml --methods heuristic drl fdrl cfdrl tfl_coran --seeds 0 1 2 3 4 -o runs/reproduction
```

The command saves each seed/method run and aggregate means, sample standard deviations, and normal-approximation 95% confidence intervals. All configured training episodes are executed.

## Saved provenance

Each run stores its effective configuration, seed, Git commit and dirty flag when available, Python/OS/package versions, historical contexts, VGAE losses, metrics, adaptation events, and checkpoints. The complete channel, scheduler, and traffic values appear in `resolved_config.yaml`.

Python, NumPy, PyTorch, environment, client replay, GMM, and historical-data streams are seeded. The test suite checks repeated CPU execution with the same environment and package versions. Record hardware and backend versions with experiments.

## Comparison protocol

Methods share topology, environment, action space, DDQN architecture, episode count, event ratios, and initialization seed. Queue evolution and scheduling follow each method's decisions.

Centralized DRL waits for the same per-UE-equivalent warmup horizon as local clients, then receives the same aggregate optimizer-step count and total replay capacity. Adaptation events occur during training and stop before the final aggregation/evaluation boundary. Evaluation uses frozen policies.

Report the completed-event adaptation mean together with completion rate and the horizon-penalized mean, which includes the observed follow-up duration of censored events. Service metrics use UE-slot-weighted reductions. Multi-seed comparisons should include variance and event censoring.

The heuristic and component-ablation rules are defined in the [algorithm guide](ALGORITHM.md). Full-profile parameters are listed in [experiment settings](ASSUMPTIONS.md).

## Graph measurements

Graph construction queries six outgoing neighbors and takes their symmetric union. Record realized node and edge counts when measuring communication or graph overhead; incoming neighbors can raise average and maximum degree above six.
