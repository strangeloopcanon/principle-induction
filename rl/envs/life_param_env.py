from __future__ import annotations

import numpy as np
from typing import Optional, Set
from .base import BaseEnv, StepResult
from .spaces import MultiBinary
from tools.life import life_step_rule, bs_bits_to_sets


class LifeParamEnv(BaseEnv):
    """RL env for inferring Life-like B/S rules from pairs of observations.

    Observation: array of shape (N, 2, H, W), where obs[i,0] is x_t and obs[i,1] is x_{t+1}
    under the hidden B/S rule with toroidal wrap.

    Action: MultiBinary(18) — bits [B0..B8, S0..S8].

    Reward: delta improvement in accuracy over best so far (default).
    Done: after max_attempts or perfect accuracy.
    """

    def __init__(
        self,
        height: int = 16,
        width: int = 16,
        num_pairs: int = 16,
        born_set: Optional[Set[int]] = None,
        survive_set: Optional[Set[int]] = None,
        max_attempts: int = 1,
        reward_mode: str = "delta",  # or "acc"
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.height = int(height)
        self.width = int(width)
        self.num_pairs = int(num_pairs)
        self.fixed_born = born_set
        self.fixed_survive = survive_set
        self.max_attempts = int(max_attempts)
        self.reward_mode = reward_mode
        self.action_space = MultiBinary(18)
        self.observation_space = None  # implicit: (num_pairs, 2, H, W)

        self._target_born = None
        self._target_survive = None
        self._pairs = None
        self._attempts = 0
        self._best_acc = 0.0

    def _gen_pairs(self, born_set: Set[int], survive_set: Set[int]):
        xs = self.np_random.integers(0, 2, size=(self.num_pairs, self.height, self.width), dtype=np.uint8)
        ys = np.stack([life_step_rule(x, born_set, survive_set, wrap=True) for x in xs], axis=0)
        obs = np.stack([xs, ys], axis=1)  # (N, 2, H, W)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        b = self.fixed_born if self.fixed_born is not None else {3}
        s = self.fixed_survive if self.fixed_survive is not None else {2, 3}
        self._target_born = set(b)
        self._target_survive = set(s)
        self._pairs = self._gen_pairs(self._target_born, self._target_survive)
        self._attempts = 0
        self._best_acc = 0.0
        return self._pairs.copy()

    def step(self, action_bits: np.ndarray) -> StepResult:
        if not (isinstance(action_bits, np.ndarray) and action_bits.shape == (18,)):
            raise ValueError("action must be np.ndarray of shape (18,)")
        self._attempts += 1

        born_set, survive_set = bs_bits_to_sets(action_bits.astype(np.uint8))
        xs = self._pairs[:, 0, :, :]
        ys = self._pairs[:, 1, :, :]
        y_pred = np.stack([life_step_rule(x, born_set, survive_set, wrap=True) for x in xs], axis=0)
        acc = float(np.mean(y_pred == ys))

        if self.reward_mode == "delta":
            reward = acc - self._best_acc
        else:
            reward = acc
        if acc > self._best_acc:
            self._best_acc = acc

        done = (self._attempts >= self.max_attempts) or (acc == 1.0)
        info = {
            "acc": acc,
            "best_acc": self._best_acc,
            "attempts": self._attempts,
            "target_born": sorted(self._target_born),
            "target_survive": sorted(self._target_survive),
        }
        return StepResult(obs=self._pairs.copy(), reward=reward, done=done, info=info)
