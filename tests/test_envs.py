import numpy as np

from rl.envs import ECAParamEnv, LifeParamEnv


def test_eca_env_perfect_action():
    env = ECAParamEnv(width=16, num_pairs=4, rule_number=110, max_attempts=1, reward_mode="acc", seed=7)
    obs = env.reset()
    res = env.step(110)
    assert 0.9 <= res.info["acc"] <= 1.0
    assert res.done is True


def test_life_env_perfect_action():
    env = LifeParamEnv(height=8, width=8, num_pairs=4, born_set={3}, survive_set={2, 3}, max_attempts=1, reward_mode="acc", seed=13)
    obs = env.reset()
    bits = np.zeros(18, dtype=np.uint8)
    bits[3] = 1
    bits[9 + 2] = 1
    bits[9 + 3] = 1
    res = env.step(bits)
    assert 0.9 <= res.info["acc"] <= 1.0
    assert res.done is True

