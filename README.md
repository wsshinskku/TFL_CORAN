# TFL-CORAN

**Official implementation** of **Transfer-enhanced Federated Learning with Dynamic Clustering for Traffic Management in 5G Open RAN**.

[English](README.md) | [한국어](README.ko.md)

**Authors:** Wooseok Shin, Janghoon Yang, Zhiqiang Shen, Minseok Choi, and Jitae Shin.

TFL-CORAN combines UE-level double deep Q-learning (DDQN), graph representation learning, soft client clustering, and model transfer for traffic management across heterogeneous 5G Open RAN cells.

## Method

- **Local control:** each UE chooses a subband, rate level, and priority using DDQN.
- **Context representation:** a variational graph autoencoder (VGAE) encodes signal, interference, traffic, and mobility (SITM) summaries.
- **Dynamic clustering:** Gaussian mixture responsibilities provide soft memberships for personalized federated aggregation.
- **Transfer:** handover and newly activated UEs initialize from a similar active UE in the destination cell.
- **Evaluation:** the runner measures service throughput, latency, reliability, QoS satisfaction, and adaptation during training, followed by frozen-policy evaluation.

The default backend is a self-contained Python environment with configurable channel, mobility, traffic, queue, and scheduling models.

## Installation

Use **Python 3.10+**. Dependencies include PyTorch, NumPy, SciPy, scikit-learn, and PyYAML. CPU and CUDA devices are supported.

```bash
git clone https://github.com/wsshinskku/TFL_CORAN.git
cd TFL_CORAN
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell:

```bash
python -m pip install -e ".[dev]"
tfl-coran doctor
```

## Quick start

```bash
tfl-coran run -c configs/smoke.yaml -o runs/smoke
tfl-coran report runs/smoke
python -m pytest
```

The smoke configuration runs 12 UEs over two short training episodes on CPU and exercises DDQN, VGAE, clustering, aggregation, transfer events, evaluation, and checkpoint saving.

## Experiment profiles

| Profile | Use |
|---|---|
| [smoke.yaml](configs/smoke.yaml) | Fast end-to-end checks with 12 UEs |
| [paper_fast.yaml](configs/paper_fast.yaml) | Laptop experiment with 75 UEs and a reduced training budget |
| [paper.yaml](configs/paper.yaml) | 375 UEs across three cells, 500 episodes, and 100 FL rounds |

The full profile uses cell populations 150/120/105, a 3.5 GHz carrier, 100 MHz bandwidth, six subbands, and 1 ms slots. Service throughput/latency targets are eMBB 20 Mbps/15 ms, URLLC 10 Mbps/5 ms, and mMTC 5 Mbps/10 ms. Check the computational budget before running 375 local DDQN agents:

```bash
tfl-coran estimate -c configs/paper.yaml
tfl-coran run -c configs/paper_fast.yaml -o runs/demo --seed 42
tfl-coran run -c configs/paper.yaml -o runs/full --device cuda
```

Each run saves all effective settings, including inherited defaults, in `resolved_config.yaml`. See [experiment settings](docs/ASSUMPTIONS.md) for the state/action encoding and model parameters.

## Comparisons and ablations

| Method | CLI name | Behavior |
|---|---|---|
| Heuristic | `heuristic` | SINR-based rate selection, service priorities, and static frequency spread |
| DRL | `drl` | Centralized DDQN with pooled replay |
| FDRL | `fdrl` | UE-local DDQN with FedAvg |
| CFDRL | `cfdrl` | Hard cluster membership and federated aggregation |
| TFL-CORAN | `tfl_coran` | Soft personalized aggregation and transfer |

```bash
tfl-coran benchmark -c configs/paper_fast.yaml --methods all -o runs/benchmark
tfl-coran ablate -c configs/paper_fast.yaml -o runs/ablations
tfl-coran reproduce -c configs/paper.yaml --methods heuristic drl fdrl cfdrl tfl_coran --seeds 0 1 2 3 4 -o runs/reproduction
```

Ablations include the full method and four variants: A disables transfer; B uses GMM on standardized raw SITM; C uses hard KMeans on VGAE embeddings; D uses uniform memberships/FedAvg.

For a shared pretrained representation:

```bash
tfl-coran pretrain-vgae -c configs/paper_fast.yaml -o runs/shared/vgae.pt
tfl-coran benchmark -c configs/paper_fast.yaml --methods all --vgae-checkpoint runs/shared/vgae.pt -o runs/shared-benchmark
```

Centralized and federated methods use matched aggregate optimizer-step and replay-capacity budgets. Adaptation reports include the completed-event mean, completion rate, and a horizon-penalized mean for censored events.

## Outputs

| Artifact | Contents |
|---|---|
| `resolved_config.yaml` | Complete experiment configuration |
| `run_metadata.json` | Seed, package/platform versions, and Git state |
| `historical_contexts.npz` | SITM observations used for representation learning |
| `memberships_latest.npy` | Latest client-to-cluster memberships |
| `vgae_training.csv` | Representation-learning losses |
| `training_metrics.csv` | Training metrics over time |
| `evaluation_by_group.csv` | Frozen-policy metrics by cell/service group |
| `adaptation_events.csv` | Handover and activation adaptation records |
| `summary.json` | Run-level metrics |
| `checkpoints/` | VGAE and global-model checkpoints |

Benchmark, ablation, and multi-seed commands also write their corresponding aggregate JSON summaries. Multi-seed summaries include means, sample standard deviations, and normal-approximation 95% confidence intervals.

## Repository guide

| Path | Purpose |
|---|---|
| [src/tfl_coran](src/tfl_coran) | Agents, environment, VGAE, clustering, FL, transfer, and runners |
| [configs](configs) | Experiment profiles |
| [scripts](scripts) | Compatibility entry points and smoke runners |
| [tests](tests) | Mathematical, lifecycle, and end-to-end checks |
| [Algorithm guide](docs/ALGORITHM.md) | Equation-to-code mapping and update order |
| [Reproducibility guide](docs/REPRODUCIBILITY.md) | Seeds, comparison protocol, and saved provenance |
| [External integration](docs/EXTERNAL_SIMULATORS.md) | Telemetry and action interfaces for external systems |
| [Paper tables](paper_reported/README.md) | Tables 3 and 4 with source provenance |
| [Validation record](VALIDATION.md) | Recorded validation and test coverage |

## Citation

```bibtex
@unpublished{shin2026tflcoran,
  title  = {Transfer-enhanced Federated Learning with Dynamic Clustering for Traffic Management in 5G Open RAN},
  author = {Shin, Wooseok and Yang, Janghoon and Shen, Zhiqiang and Choi, Minseok and Shin, Jitae},
  note   = {Manuscript under revision at Computer Communications},
  year   = {2026}
}
```

See [CITATION.cff](CITATION.cff). Source code is distributed under the [MIT License](LICENSE); third-party licensing information is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
