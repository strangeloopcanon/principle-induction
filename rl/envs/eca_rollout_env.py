from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseEnv, StepResult
from tools.eca import eca_step


class ECARolloutEnv(BaseEnv):
    """RL env for predicting a multi-step ECA rollout.

    Observation: dict with keys:
      - x0: np.ndarray shape (W,) in {0,1}
      - rule: int in [0,255]
      - steps: int >= 1

    Action: np.ndarray shape (W,) in {0,1}, representing Y after `steps` rolls.

    Reward: by default, accuracy vs ground-truth Y_T.
    Done: after max_attempts or perfect accuracy.
    """

    def __init__(
        self,
        width: int = 32,
        steps: int = 10,
        rule_number: Optional[int] = None,
        max_attempts: int = 1,
        reward_mode: str = "acc",  # or "delta"
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.width = int(width)
        self.steps = int(steps)
        self.fixed_rule = rule_number
        self.max_attempts = int(max_attempts)
        self.reward_mode = reward_mode

        self._x0 = None
        self._rule = None
        self._yT = None
        self._attempts = 0
        self._best_acc = 0.0

    def _rollout(self, x0: np.ndarray, rule: int, T: int) -> np.ndarray:
        x = x0.astype(np.uint8)
        for _ in range(T):
            x = eca_step(x, rule_number=int(rule))
        return x

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self._rule = self.fixed_rule if self.fixed_rule is not None else int(self.np_random.integers(256))
        self._x0 = self.np_random.integers(0, 2, size=(self.width,), dtype=np.uint8)
        self._yT = self._rollout(self._x0, self._rule, self.steps)
        self._attempts = 0
        self._best_acc = 0.0
        return {"x0": self._x0.copy(), "rule": int(self._rule), "steps": int(self.steps)}

    def step(self, action_bits: np.ndarray) -> StepResult:
        if not (isinstance(action_bits, np.ndarray) and action_bits.shape == (self.width,)):
            raise ValueError(f"action must be np.ndarray of shape ({self.width},)")
        a = action_bits.astype(np.uint8)
        self._attempts += 1

        acc = float(np.mean(a == self._yT))

        if self.reward_mode == "delta":
            reward = acc - self._best_acc
        else:
            reward = acc
        if acc > self._best_acc:
            self._best_acc = acc

        done = (self._attempts >= self.max_attempts) or (acc == 1.0)
        obs = {"x0": self._x0.copy(), "rule": int(self._rule), "steps": int(self.steps)}
        info = {
            "acc": acc,
            "best_acc": self._best_acc,
            "attempts": self._attempts,
        }
        return StepResult(obs=obs, reward=reward, done=done, info=info)

