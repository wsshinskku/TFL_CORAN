# ric/orchestrator.py
# Multi-timescale orchestration: RL per slot/episode, FL every τE episodes, VGAE+GMM every τF FL rounds.
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

from envs.o_ran_sim.base_env import ORanSimEnv
from ue import DDQNAgent, DDQNConfig, ActionSpace

from .topology import build_features_matrix, build_similarity_adjacency, normalize_adj
from .vgae import VGAEEncoder, infer_embeddings
from .gmm import SoftGMM
from .fl_server import aggregate_membership_weighted, personalize_by_membership

class Orchestrator:
    def __init__(
        self,
        env: ORanSimEnv,
        *,
        tau_E: int = 5,                     # FL every τE episodes (Table II)
        tau_F: int = 10,                    # VGAE/GMM refresh every τF FL rounds (Table II; Table III shows 3 as variant)
        device: str = "cpu",
        vgae_ckpt: Optional[str] = None,
        gmm_K: int = 3
    ):
        self.env = env
        self.tau_E = tau_E
        self.tau_F = tau_F
        self.device = device

        # Build UE agents after first reset
        self.obs = self.env.reset()
        state_dim = len(next(iter(self.obs.values())))
        aspace = ActionSpace(n_subbands=self.env.n_subbands, n_mcs=self.env.channel.num_mcs(), n_priority=3)
        self.agents: Dict[str, DDQNAgent] = {
            ue_id: DDQNAgent(state_dim=state_dim, action_space=aspace, cfg=DDQNConfig(device=device), ue_id=ue_id)
            for ue_id in self.obs.keys()
        }

        # Initial broadcast model w^(0): copy from any agent
        self.init_weights = next(iter(self.agents.values())).get_weights()
        for ag in self.agents.values():
            ag.load_weights(self.init_weights, set_as_broadcast_baseline=True)

        # RIC side models
        self.encoder = VGAEEncoder(in_dim=4, gcn_dims=(64,32), latent_dim=32).to(device)
        if vgae_ckpt:
            self.encoder.load_state_dict(torch.load(vgae_ckpt, map_location=device))
        self.encoder.eval()

        self.gmm = SoftGMM(n_components=gmm_K, covariance_type="full", max_iter=100, tol=1e-3)

        # Cluster model baselines \bar w_k (init as w^(0))
        self.K = gmm_K
        self.cluster_models = [ {k: v.clone() for k, v in self.init_weights.items()} for _ in range(self.K) ]

        self.round_idx = 0

    # ----------------------------- main loop -----------------------------

    def run(self, num_fl_rounds: int = 5):
        """
        Run FL rounds; each round = τE episodes; refresh γ via VGAE+GMM every τF rounds.
        """
        for r in range(num_fl_rounds):
            # 1) RL at UEs for τE episodes
            for ep in range(self.tau_E):
                self._run_episode_train()
                for ag in self.agents.values(): ag.end_episode()

            # 2) RIC: refresh memberships every τF rounds (slow cadence)
            if (self.round_idx % self.tau_F) == 0:
                gamma, ue_ids = self._refresh_memberships()

            # 3) RIC: aggregate (Eq.(9)) and personalize (Eq.(10))
            deltas = {ue: self.agents[ue].compute_delta() for ue in self.agents.keys()}
            self.cluster_models = aggregate_membership_weighted(
                deltas=deltas, ue_ids=ue_ids, gamma=gamma, prev_cluster_models=self.cluster_models
            )
            personalized = personalize_by_membership(self.cluster_models, ue_ids, gamma)

            # 4) Broadcast back to UEs, reset delta baselines
            for ue_id, weights in personalized.items():
                self.agents[ue_id].accept_broadcast_and_reset_delta(weights)

            self.round_idx += 1

    # ----------------------------- helpers ------------------------------

    def _run_episode_train(self):
        obs = self.obs
        for _ in range(self.env.slot_per_episode):
            actions = {ue_id: self.agents[ue_id].act(o) for ue_id, o in obs.items()}
            next_obs, rewards, dones, info = self.env.step(actions)
            # log single-step to each UE
            for ue_id in obs.keys():
                self.agents[ue_id].observe(obs[ue_id], actions[ue_id], rewards[ue_id], next_obs[ue_id], dones[ue_id])
                self.agents[ue_id].learn(gradient_steps=1)
            obs = next_obs
        self.obs = obs

    def _refresh_memberships(self) -> Tuple[np.ndarray, List[str]]:
        # Gather SITM features (Sec. II; Figure explanation), build A by Eq.(1), then VGAE→Z and GMM→γ.
        sitm = self.env.ric_features_SITM()                              # {ue: [S,I,T,M]}
        X, ue_ids = build_features_matrix(sitm)                          # [N,4]
        A = build_similarity_adjacency(X, self_loop=True)
        A_hat = normalize_adj(A, add_self_loop=True)
        Z = infer_embeddings(self.encoder, X, A_hat, device=self.device) # [N,d], Eq.(2)-(4)
        gamma = self.gmm.fit_predict(Z)                                  # [N,K], Eq.(5)-(6)
        return gamma, ue_ids

# ------------------------------ smoke test ------------------------------ #
if __name__ == "__main__":
    env = ORanSimEnv()
    orch = Orchestrator(env, tau_E=2, tau_F=3, device="cpu")
    orch.run(num_fl_rounds=3)
    print("Orchestrator smoke: ran 3 FL rounds.")
