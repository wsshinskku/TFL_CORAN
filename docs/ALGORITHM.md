# Algorithm-to-code mapping

## UE control

`envs/oran_env.py` exposes the state
`[x, y, allocated_rate, latency, reliability, throughput, one_hot(service)]`.
`ActionCodec` flattens `(subband, rate_level, priority)` into one DDQN action.
The environment computes Eq. (1) exactly:

```text
reward = reliability / reliability_target
       + throughput / throughput_target
       - latency / latency_target
```

`agents/ddqn.py` implements Eqs. (3)-(5): the online network chooses the next
action, the target network evaluates it, terminal transitions omit bootstrap,
and the online network minimizes squared TD error. The included environment is
a continuing MDP: episode boundaries are optimization/evaluation windows and do
not emit terminal transitions, although the agent supports terminal masks.

The non-learning heuristic maps SINR to one of the three rate levels, maps the
service class to a priority, and spreads UEs across frequencies with a static
round-robin index. This is an explicit approximation because the manuscript
does not publish the exact heuristic action mapping.

## Representation and clustering

The RIC receives only normalized SITM summaries, not replay transitions.
`models/vgae.py` standardizes historical SITM, queries `k` directed nearest
neighbors per node and then takes the symmetric edge union. It assigns edge
weight `1/(1 + Euclidean_distance)` as in Eq.
(7), applies normalized graph convolution, and trains the inner-product VGAE
with reconstruction + KL loss (Eqs. 8-12). Runtime inference uses frozen
weights and the posterior mean. Because the union retains incoming as well as
outgoing neighbors, `k=6` does not cap or guarantee average degree six; the
realized average and maximum can be larger.

`clustering/gmm.py` fits K Gaussian components and returns normalized posterior
responsibilities from Eqs. (13)-(14). CFDRL converts them to one-hot values;
TFL-CORAN keeps them soft. For the Table 4 component ablations, A retains
VGAE+GMM without transfer, B fits GMM to standardized raw context, C replaces
the disabled GMM with seeded deterministic hard KMeans over VGAE embeddings,
and D uses uniform memberships/FedAvg. The manuscript does not specify the
replacement grouping for C, so hard KMeans is an implementation interpretation.

## Federated learning and transfer

`federated/aggregator.py` computes cluster models, cluster mass, the shared
global model, and membership-blended personalized models (Eqs. 15-17). Empty
clusters fall back to the previous shared model.

The manuscript defines a client update against one global model even though
each client receives a distinct personalized model. The default
`delta_reference: dispatch_base` averages post-local absolute weights, avoiding
re-adding old personalization offsets. `paper_global` is available for a
literal equation experiment.

After a round, `transfer/initializer.py` restricts Eq. (18)'s cosine search to
active UEs in the destination cell and excludes self. A handover mixes the
previous local model and the neighbor according to Eq. (19). A new UE has no
previous model, so it receives the neighbor model; if no neighbor exists it
falls back to its current personalized/global model.

## Timing

The runner performs:

1. slot-level action, scheduling, QoS outcome, reward and local DDQN update;
2. target-network synchronization every configured number of episodes;
3. FL every `episodes_per_round` episodes;
4. 10% handover/reassignment and 3% activation at training round boundaries;
5. VGAE inference and GMM/hard-KMeans refit every `cluster_refresh_rounds` FL rounds;
6. transfer initialization for affected UEs using standardized context for the
   Eq. (18) cosine search, without an unscheduled VGAE graph pass.

At the final training boundary, the runner still aggregates but deliberately
does not inject handovers/activations: those clients would otherwise enter
immediately before frozen evaluation and receive no subsequent local-update
window. Adaptation is measured during online training, then finalized before a
separate deterministic evaluation. It reports the completed-event mean,
completion rate, and a horizon-penalized mean that uses each censored event's
observed follow-up duration.
