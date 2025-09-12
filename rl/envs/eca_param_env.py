from __future__ import annotations

import numpy as np
from typing import Optional
from .base import BaseEnv, StepResult
from .spaces import Discrete
from tools.eca import eca_step


class ECAParamEnv(BaseEnv):
    """RL env for inferring an ECA rule from pairs of observations.

    Observation: array of shape (N, 2, W), where obs[i,0] is x_t and obs[i,1] is x_{t+1}
    under the hidden rule. Wrap is toroidal.

    Action: Discrete(256) — propose a rule number in [0,255].

    Reward: by default, delta improvement in accuracy over best so far.
    Done: after max_attempts or if accuracy == 1.0.
    """

    def __init__(
        self,
        width: int = 64,
        num_pairs: int = 16,
        rule_number: Optional[int] = None,
        max_attempts: int = 1,
        reward_mode: str = "delta",  # or "acc"
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.width = int(width)
        self.num_pairs = int(num_pairs)
        self.fixed_rule = rule_number
        self.max_attempts = int(max_attempts)
        self.reward_mode = reward_mode
        self.action_space = Discrete(256)
        self.observation_space = None  # implicit: (num_pairs, 2, width) binary

        self._target_rule = None
        self._pairs = None
        self._attempts = 0
        self._best_acc = 0.0

    def _gen_pairs(self, rule_num: int):
        xs = self.np_random.integers(0, 2, size=(self.num_pairs, self.width), dtype=np.uint8)
        ys = np.stack([eca_step(x, rule_number=rule_num) for x in xs], axis=0)
        obs = np.stack([xs, ys], axis=1)  # (N, 2, W)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self._target_rule = self.fixed_rule if self.fixed_rule is not None else int(self.np_random.integers(256))
        self._pairs = self._gen_pairs(self._target_rule)
        self._attempts = 0
        self._best_acc = 0.0
        return self._pairs.copy()

    def step(self, action: int) -> StepResult:
        if not (0 <= int(action) <= 255):
            raise ValueError("action must be rule number in [0,255]")
        self._attempts += 1

        xs = self._pairs[:, 0, :]
        ys = self._pairs[:, 1, :]
        y_pred = np.stack([eca_step(x, rule_number=int(action)) for x in xs], axis=0)
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
            "target_rule": self._target_rule,
        }
        # Observation remains the same across attempts (stationary dataset)
        return StepResult(obs=self._pairs.copy(), reward=reward, done=done, info=info)
