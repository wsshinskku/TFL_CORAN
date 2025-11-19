# scripts/run_sim.py
# 전체 파이프라인 실행: UE(DDQN)→FL→VGAE/GMM 주기적 리프레시
# 기본 주기: L=200 슬롯/에피소드, τE=5 에피소드/FL 라운드, τF=3~10 라운드
from __future__ import annotations
import argparse, os, random, numpy as np, torch

from configs.loader import load_config, save_resolved_config
from envs.o_ran_sim.base_env import ORanSimEnv
from ric.orchestrator import Orchestrator

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def make_env(cfg) -> ORanSimEnv:
    return ORanSimEnv(
        cells=cfg.topology.cells,
        isd_m=cfg.topology.isd_m,
        n_subbands=cfg.carrier.n_subbands,
        slot_per_episode=cfg.time_scales.episode_len,
        center_freq_ghz=cfg.carrier.center_freq_ghz,
        carrier_mhz=cfg.carrier.bandwidth_mhz,
        harq_like_retx=cfg.env.scheduler.harq_like_retx,
        reuse_allowed=cfg.env.scheduler.reuse_allowed,
        scheduler_weights=(
            cfg.env.scheduler.utility_weights["priority"],
            cfg.env.scheduler.utility_weights["predicted_rate"],
            cfg.env.scheduler.utility_weights["qos_deficit"],
        ),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, action="append", required=True, help="YAML 파일을 여러 개 넘겨 병합 가능")
    ap.add_argument("--rounds", type=int, default=5, help="FL 라운드 수")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg.run.output_dir, exist_ok=True)
    save_resolved_config(cfg, os.path.join(cfg.run.output_dir, "resolved_config.yaml"))
    set_seed(cfg.run.seed)

    env = make_env(cfg)
    orch = Orchestrator(
        env,
        tau_E=cfg.time_scales.tau_E,
        tau_F=cfg.time_scales.tau_F,
        device=cfg.run.device,
        vgae_ckpt=cfg.paths.vgae_checkpoint,
        gmm_K=cfg.models.gmm.K
    )

    print(f"[RUN] FL rounds={args.rounds} | τE={cfg.time_scales.tau_E} | τF={cfg.time_scales.tau_F}")
    orch.run(num_fl_rounds=args.rounds)
    print("[DONE] Simulation finished.")

if __name__ == "__main__":
    main()
