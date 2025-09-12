from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict
import numpy as np


@dataclass
class StepResult:
    obs: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class BaseEnv:
    observation_space = None
    action_space = None

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed = int(seed)
            self.np_random = np.random.default_rng(self._seed)
        return None

    def step(self, action) -> StepResult:
        raise NotImplementedError

