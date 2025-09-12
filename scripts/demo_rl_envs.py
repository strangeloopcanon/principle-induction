import os
import sys
import numpy as np

# Ensure repository root is on path when run as a script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl.envs import ECAParamEnv, LifeParamEnv


def demo_eca():
    env = ECAParamEnv(width=32, num_pairs=8, rule_number=110, max_attempts=2, reward_mode="acc", seed=42)
    obs = env.reset()
    action = 110
    res = env.step(action)
    print("ECA demo — acc:", res.info["acc"], "done:", res.done)


def demo_life():
    env = LifeParamEnv(height=8, width=8, num_pairs=4, born_set={3}, survive_set={2, 3}, max_attempts=1, reward_mode="acc", seed=123)
    obs = env.reset()
    bits = np.zeros(18, dtype=np.uint8)
    bits[3] = 1  # B3
    bits[9 + 2] = 1  # S2
    bits[9 + 3] = 1  # S3
    res = env.step(bits)
    print("Life demo — acc:", res.info["acc"], "done:", res.done)


if __name__ == "__main__":
    demo_eca()
    demo_life()
