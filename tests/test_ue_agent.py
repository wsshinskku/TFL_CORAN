# tests/test_ue_agent.py
import numpy as np
from envs.o_ran_sim.base_env import ORanSimEnv
from ue.agent import DDQNAgent, ActionSpace, DDQNConfig

def test_ddqn_agent_interaction_and_learn_step():
    env = ORanSimEnv(slot_per_episode=3, n_subbands=3)
    obs = env.reset()
    ue_id = next(iter(obs.keys()))
    state_dim = len(obs[ue_id])
    aspace = ActionSpace(n_subbands=3, n_mcs=env.channel.num_mcs(), n_priority=3)
    agent = DDQNAgent(state_dim=state_dim, action_space=aspace, cfg=DDQNConfig(device="cpu"), ue_id=ue_id)

    s = obs[ue_id]
    a = agent.act(s, explore=True)
    actions = {ue_id: a}
    s_next, r, d, info = env.step(actions)
    agent.observe(s, a, r[ue_id], s_next[ue_id], d[ue_id])
    logs = agent.learn(gradient_steps=1)
    # 한 번의 학습 후 손실 로그가 존재
    assert "loss" in logs
