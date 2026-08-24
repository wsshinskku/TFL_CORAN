# Reported parameters and implementation assumptions

The manuscript defines the learning equations and high-level test protocol,
but not every value needed to execute a simulator. `configs/paper.yaml` marks
reported values and the major result-sensitive assumptions it overrides.
Additional simulator constants are inherited from `tfl_coran.config` defaults;
the `resolved_config.yaml` written by every run is the complete, authoritative
record of all effective values.

| Item | Manuscript | Default setting |
|---|---|---|
| Cells / population | 3; 150/120/105; ISD 500 m | Exact |
| Carrier | 3.5 GHz, 100 MHz, 6 subbands | Exact |
| Service rate/latency | eMBB 20 Mbps/15 ms; URLLC 10/5; mMTC 5/10 | Exact |
| Reliability targets | Not reported | 0.95 / 0.999 / 0.90, configurable |
| Slot duration | Not reported | 1 ms |
| UE state encoding | Semantic fields only | x/y + four QoS fields + one-hot service (9-D), fixed shared scaling |
| Action cardinality | 6 subbands; rate/priority spaces unnamed | 3 rate levels x 3 priority levels (54 actions) |
| DDQN hidden layers | Not reported | 128/128 ReLU |
| Local updates | Not reported | One minibatch update per UE/slot after per-UE warmup; centralized DRL waits the same per-UE horizon and receives the same aggregate optimizer-step and total replay-capacity budget |
| Target after FL dispatch | Not reported | Preserve the reported 10-episode sync; transfer/new-client resets synchronize immediately |
| Adam state after FL dispatch | Not reported | Retain moments for existing UEs; reset for newly activated UEs; configurable |
| RF/channel | QuaDRiGa named, no trace/config | Synthetic path loss, shadowing/fading, BLER and interference approximation; detailed model constants appear under `environment.channel_model`, with transmit/noise values elsewhere under `environment` in the resolved config |
| Scheduler utility | Qualitative | Priority + predicted requested rate + queue pressure + service bias; weights appear under `environment.scheduler_model` in the resolved config |
| Same-subband service | Reuse/contenders described | Per-cell configurable MU multiplex cap and SINR penalty |
| Traffic arrivals | Not reported | Service-specific lognormal arrivals and bounded queues; mean service rates appear under `environment.arrival_rate_mbps`, with stochastic-shape/base-latency constants under `environment.traffic_model` in the resolved config |
| VGAE history | Historical data named, not released | Independently seeded simulator snapshots |
| Graph | Eq. 7 dense formula; overhead reports average degree about 6 | Standardized directed kNN query with `k=6`, followed by a symmetric union and Eq. 7 message weights; reciprocal-union edges mean realized average/max degree can exceed 6, so degree 6 is approximated rather than enforced |
| VGAE optimization | GCN 64/32, latent 32 only | Adam, lr .01, 100 epochs, full BCE + KL |
| Runtime embedding | `z` named | Posterior mean `mu` to avoid sampling jitter |
| GMM details | K=3 only | Full covariance, regularization 1e-6, deterministic multi-init |
| Heuristic baseline | Service priority and SINR-based allocation are described; exact mapping is not reported | Static round-robin frequency spread, SINR-threshold rate bin, and service-class priority |
| Ablation A | Transfer off; VGAE/GMM on | Same; no neighbor transfer |
| Ablation B | Transfer/VGAE off; GMM on; no replacement representation specified | Standardized raw SITM context + GMM |
| Ablation C | Transfer/GMM off; VGAE on; no replacement grouping specified | VGAE embedding + seeded deterministic hard KMeans |
| Ablation D | Transfer/VGAE/GMM off; aggregation fallback unspecified | Uniform memberships/FedAvg |
| Transfer delta | Not reported | 0.5 for handover; effectively 0 for a brand-new UE |
| New UE previous model | Undefined | Destination-cell neighbor, then global fallback |
| Delta baseline | Equations use global although clients receive personalized models | `dispatch_base`/absolute aggregation default; literal `paper_global` available |
| Adaptation criterion | Time to satisfy targets | Consecutive configurable QoS slots during online training; completed-only mean, completion rate, and censored follow-up horizon penalty are retained; final boundary injects no event before frozen evaluation |
| QoS evaluation window | Not reported | Instantaneous slot condition, averaged over UE-slots |
| Training duration | Not reported | 100 FL rounds / 500 episodes, inherited from the prototype README |

These choices support algorithm testing. Exact numerical comparison with the
paper requires the original traces and simulator parameters. This table lists
the result-sensitive choices; each run's resolved config contains the complete
effective configuration.
