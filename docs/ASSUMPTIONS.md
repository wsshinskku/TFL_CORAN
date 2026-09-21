# Experiment settings

[configs/paper.yaml](../configs/paper.yaml) configures the full TFL-CORAN experiment. Values omitted from a profile inherit defaults from `tfl_coran.config`. Each run's `resolved_config.yaml` records every effective value.

| Setting | Full-profile value or behavior |
|---|---|
| Cells and population | Three cells with 150/120/105 UEs; 500 m inter-site distance |
| Carrier and spectrum | 3.5 GHz, 100 MHz, six subbands |
| Throughput/latency targets | eMBB 20 Mbps/15 ms; URLLC 10 Mbps/5 ms; mMTC 5 Mbps/10 ms |
| Reliability targets | 0.95 / 0.999 / 0.90 for eMBB / URLLC / mMTC |
| Slot duration | 1 ms |
| UE state | x/y, allocated rate, latency, reliability, throughput, and one-hot service; nine dimensions with fixed shared scaling |
| Actions | Six subbands × three rate levels × three priorities = 54 actions |
| DDQN | 128/128 ReLU hidden layers, Adam, learning rate .001, discount .99, batch 64 |
| Replay and local updates | Capacity 100,000 per UE; one minibatch update per slot after warmup |
| Centralized DRL budget | Same per-UE warmup horizon, aggregate optimizer-step count, and total replay capacity as federated clients |
| Target synchronization | Every ten episodes; transfer and new-client resets synchronize immediately |
| Optimizer after FL dispatch | Retain existing UE moments; newly activated UEs reset; configurable |
| Channel | Configurable path loss, shadowing, fading, interference, and BLER model |
| Scheduling | Priority, predicted requested rate, queue pressure, and service bias; configurable per-cell multiplexing |
| Traffic | Service-specific lognormal arrivals and bounded queues |
| VGAE history | Separately seeded simulator snapshots |
| Graph | Standardized context, six outgoing nearest neighbors, symmetric edge union, weight 1/(1 + distance); realized degree can exceed six |
| VGAE | Hidden dimension 64, latent dimension 32, Adam lr .01, 100 epochs, reconstruction and KL loss |
| Runtime embedding | Frozen VGAE posterior mean |
| GMM | Three components, full covariance, regularization 1e-6, three initializations |
| Heuristic | Static round-robin frequency allocation, SINR-bin rate selection, service priorities |
| Ablation A | VGAE and GMM; transfer disabled |
| Ablation B | Standardized raw SITM with GMM; transfer disabled |
| Ablation C | VGAE embedding with seeded hard KMeans; transfer disabled |
| Ablation D | Uniform memberships and FedAvg; transfer disabled |
| Handover transfer | Mix old and destination-neighbor models with delta .5 |
| New UE initialization | Destination-cell neighbor, then personalized/global fallback |
| Aggregation mode | `dispatch_base` averages post-local absolute weights; `paper_global` uses global-model-relative deltas |
| Adaptation | Consecutive QoS-satisfying slots during training; completed mean, completion rate, and censored follow-up penalty |
| Evaluation | Frozen deterministic policies; service metrics aggregated over UE-slots |
| Training duration | 500 episodes, FL every five episodes, 100 FL rounds |

Channel, scheduler, traffic, and device settings are configurable. The final training boundary aggregates models before evaluation; handovers and activations are injected only at earlier boundaries so adaptation events have a subsequent local-training window.
