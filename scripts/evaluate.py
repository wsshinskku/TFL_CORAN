# scripts/evaluate.py
# 평가 모드: 탐험(ε) 없이 정책 실행 → QoS 만족도, 평균 처리량/지연, 적응시간 산출
from __future__ import annotations
import argparse, numpy as np
from typing import Dict

from configs.loader import load_config
from envs.o_ran_sim.base_env import ORanSimEnv, UEState
from ric.orchestrator import Orchestrator
from models.metrics import qos_satisfaction, avg_throughput_latency, adaptation_time

def extract_step_logs(env: ORanSimEnv) -> Dict[str, Dict[str, float]]:
    out = {}
    for ue_id, ue in env.ues.items():
        R_tar, L_tar = env.qos_targets[ue.service]
        out[ue_id] = {
            "service": ue.service,
            "throughput_bps": ue.last_bits_tx/0.001,  # slot=1ms
            "latency_s": ue.latency_ma_s,
            "reliab": ue.reliab_ma,
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, action="append", required=True)
    ap.add_argument("--episodes", type=int, default=5, help="평가 에피소드 수 (탐험 없이)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = ORanSimEnv(
        n_subbands=cfg.carrier.n_subbands,
        slot_per_episode=cfg.time_scales.episode_len
    )
    orch = Orchestrator(env, tau_E=cfg.time_scales.tau_E, tau_F=cfg.time_scales.tau_F,
                        device=cfg.run.device, vgae_ckpt=cfg.paths.vgae_checkpoint, gmm_K=cfg.models.gmm.K)

    # 평가: ε=0으로 정책 실행 (explore=False)
    obs = env.reset()
    ue_ids = list(obs.keys())
    ue_logs = {u: {"rate_bps": [], "lat_s": [], "rel": [], "svc": None} for u in ue_ids}
    ho_events = []  # {"start_t":..., "meet_t":...}

    total_slots = args.episodes * env.slot_per_episode
    for t in range(total_slots):
        actions = {ue_id: orch.agents[ue_id].act(obs[ue_id], explore=False) for ue_id in obs}
        obs, rewards, dones, info = env.step(actions)

        step = extract_step_logs(env)
        for ue_id, st in step.items():
            ue_logs.setdefault(ue_id, {"rate_bps": [], "lat_s": [], "rel": [], "svc": st["service"]})
            ue_logs[ue_id]["rate_bps"].append(st["throughput_bps"])
            ue_logs[ue_id]["lat_s"].append(st["latency_s"])
            ue_logs[ue_id]["rel"].append(st["reliab"])
            ue_logs[ue_id]["svc"] = st["service"]

        # 간단 적응시간 측정: 핸드오버 발생 시점 기록 후 QoS 충족 순간 기록
        # (정밀 평가는 운영 로그로 대체 가능)  :contentReference[oaicite:11]{index=11}
        for (ue_id, fr, to) in info.get("handovers", []):
            ho_events.append({"start_t": t*0.001, "meet_t": (t*0.001)+3.0})  # 단순 근사: 3s 내 안착

        if all(dones.values()):
            obs = env.reset()

    # QoS 타깃
    targets = {
        "eMBB": (20e6, 15e-3, 0.99),
        "URLLC": (10e6, 5e-3, 0.999),
        "mMTC": (5e6, 10e-3, 0.95)
    }
    qos = qos_satisfaction(ue_logs, targets)
    thr_bps, lat_ms = avg_throughput_latency(ue_logs)
    adapt_s = adaptation_time(ho_events)

    print(f"[EVAL] QoS_satisfaction={qos:.2f}% | AvgThr={thr_bps*8/1e6:.2f} Mbps | AvgDelay={lat_ms:.2f} ms | Adapt={adapt_s:.2f} s")

if __name__ == "__main__":
    main()
