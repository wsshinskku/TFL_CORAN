# tests/test_env_sim.py
import numpy as np
from envs.o_ran_sim.base_env import ORanSimEnv

def test_env_reset_and_step_and_sitm_features():
    env = ORanSimEnv(slot_per_episode=5, n_subbands=3)  # 빠른 에피소드
    obs = env.reset()
    assert len(obs) > 0
    obs_dim = len(next(iter(obs.values())))
    assert obs_dim >= 8  # 관측 벡터 길이(기본 10)
    # 한 스텝 실행
    actions = {ue_id: (0, 5, 1) for ue_id in obs.keys()}
    next_obs, rewards, dones, info = env.step(actions)
    assert set(rewards.keys()) == set(obs.keys())
    # SITM 특징 (식 (1) 입력) 4차원
    sitm = env.ric_features_SITM()
    any_vec = next(iter(sitm.values()))
    assert any_vec.shape == (4,)
    # 에피소드 완료 플래그
    for _ in range(4):
        actions = {ue_id: (0, 5, 1) for ue_id in next_obs.keys()}
        next_obs, rewards, dones, info = env.step(actions)
    assert all(dones.values())
