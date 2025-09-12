from __future__ import annotations

import numpy as np
from typing import Optional, Set, Tuple

from tools.life import life_step_rule


def make_life_pairs(
    born_set: Set[int],
    survive_set: Set[int],
    height: int,
    width: int,
    num_pairs: int,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return pairs array (N, 2, H, W) under B/S rule with wrap."""
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, 2, size=(num_pairs, height, width), dtype=np.uint8)
    ys = np.stack([life_step_rule(x, born_set, survive_set, wrap=True) for x in xs], axis=0)
    obs = np.stack([xs, ys], axis=1)
    return obs


def make_life_trajectories(
    born_set: Set[int],
    survive_set: Set[int],
    height: int,
    width: int,
    horizon: int,
    num_seqs: int,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return trajectories array (N, T, H, W) under B/S rule with wrap."""
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, 2, size=(num_seqs, height, width), dtype=np.uint8)
    traj = np.empty((num_seqs, horizon, height, width), dtype=np.uint8)
    traj[:, 0, :, :] = xs
    for t in range(1, horizon):
        traj[:, t, :, :] = np.stack(
            [life_step_rule(traj[i, t - 1, :, :], born_set, survive_set, wrap=True) for i in range(num_seqs)], axis=0
        )
    return traj

