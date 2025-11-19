# scripts/run_fl_round.py
# 간단 실행기: 지정 라운드만 수행하여 집계/개인화 로그 확인
from __future__ import annotations
import argparse
from configs.loader import load_config
from envs.o_ran_sim.base_env import ORanSimEnv
from ric.orchestrator import Orchestrator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, action="append", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = ORanSimEnv(slot_per_episode=cfg.time_scales.episode_len, n_subbands=cfg.carrier.n_subbands)
    orch = Orchestrator(env, tau_E=cfg.time_scales.tau_E, tau_F=cfg.time_scales.tau_F,
                        device=cfg.run.device, vgae_ckpt=cfg.paths.vgae_checkpoint, gmm_K=cfg.models.gmm.K)
    orch.run(num_fl_rounds=args.rounds)
    print(f"[OK] Ran {args.rounds} FL round(s).")

if __name__ == "__main__":
    main()
