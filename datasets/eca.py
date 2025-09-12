from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from tools.eca import eca_step


def make_eca_pairs(
    rule_number: int,
    width: int,
    num_pairs: int,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return pairs array of shape (N, 2, W) with toroidal wrap.

    obs[i,0] is x_t and obs[i,1] is x_{t+1}.
    """
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, 2, size=(num_pairs, width), dtype=np.uint8)
    ys = np.stack([eca_step(x, rule_number=rule_number) for x in xs], axis=0)
    obs = np.stack([xs, ys], axis=1)
    return obs


def make_eca_trajectories(
    rule_number: int,
    width: int,
    horizon: int,
    num_seqs: int,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return trajectories array (N, T, W) under ECA rule with wrap."""
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, 2, size=(num_seqs, width), dtype=np.uint8)
    traj = np.empty((num_seqs, horizon, width), dtype=np.uint8)
    traj[:, 0, :] = xs
    for t in range(1, horizon):
        traj[:, t, :] = np.stack([eca_step(traj[i, t - 1, :], rule_number=rule_number) for i in range(num_seqs)], axis=0)
    return traj

