import numpy as np
from typing import Set, Tuple


def neighbor_counts(board: np.ndarray, wrap: bool = True) -> np.ndarray:
    """Compute 8-neighbor counts for a 2D binary board.

    Args:
        board: 2D array of shape (H, W) with values {0,1}.
        wrap: If True, toroidal wraparound; otherwise zero-padded edges.

    Returns:
        2D array of shape (H, W) with integer counts in [0, 8].
    """
    if board.ndim != 2:
        raise ValueError("board must be 2D")
    b = board.astype(np.uint8)
    if wrap:
        up = np.roll(b, -1, axis=0)
        down = np.roll(b, 1, axis=0)
        left = np.roll(b, -1, axis=1)
        right = np.roll(b, 1, axis=1)
        up_left = np.roll(up, -1, axis=1)
        up_right = np.roll(up, 1, axis=1)
        down_left = np.roll(down, -1, axis=1)
        down_right = np.roll(down, 1, axis=1)
        counts = (
            up + down + left + right + up_left + up_right + down_left + down_right
        )
        return counts.astype(np.int16)
    else:
        H, W = b.shape
        padded = np.pad(b, 1, mode="constant")
        counts = (
            padded[0:H, 0:W]
            + padded[0:H, 1:W+1]
            + padded[0:H, 2:W+2]
            + padded[1:H+1, 0:W]
            + padded[1:H+1, 2:W+2]
            + padded[2:H+2, 0:W]
            + padded[2:H+2, 1:W+1]
            + padded[2:H+2, 2:W+2]
        )
        return counts.astype(np.int16)


def life_step(board: np.ndarray, wrap: bool = True) -> np.ndarray:
    """Canonical Conway's Life step (B3/S23).

    Args:
        board: 2D array (H, W) with values {0,1}.
        wrap: Toroidal wraparound if True.

    Returns:
        Next board (uint8) with values {0,1}.
    """
    counts = neighbor_counts(board, wrap=wrap)
    live = board.astype(np.uint8) == 1
    born = (~live) & (counts == 3)
    survive = live & ((counts == 2) | (counts == 3))
    next_board = (born | survive).astype(np.uint8)
    return next_board


def life_step_rule(
    board: np.ndarray,
    born_set: Set[int],
    survive_set: Set[int],
    wrap: bool = True,
) -> np.ndarray:
    """General B/S rule step.

    Args:
        board: 2D array (H, W) in {0,1}.
        born_set: set of neighbor counts that cause birth when cell dead.
        survive_set: set of neighbor counts that keep a live cell alive.
        wrap: Toroidal wraparound if True.

    Returns:
        Next board (uint8) with values {0,1}.
    """
    counts = neighbor_counts(board, wrap=wrap)
    live = board.astype(np.uint8) == 1
    born = (~live) & np.isin(counts, list(born_set))
    survive = live & np.isin(counts, list(survive_set))
    next_board = (born | survive).astype(np.uint8)
    return next_board


def bs_bits_to_sets(bits: np.ndarray) -> Tuple[Set[int], Set[int]]:
    """Convert 18-bit vector [B0..B8, S0..S8] to (born_set, survive_set)."""
    if bits.shape != (18,):
        raise ValueError("bits must have shape (18,)")
    b = set(np.nonzero(bits[:9])[0].tolist())
    s = set(np.nonzero(bits[9:])[0].tolist())
    return b, s


def sets_to_bs_bits(born_set: Set[int], survive_set: Set[int]) -> np.ndarray:
    """Convert (born_set, survive_set) to 18-bit vector [B0..B8, S0..S8]."""
    out = np.zeros(18, dtype=np.uint8)
    for v in born_set:
        if 0 <= v <= 8:
            out[v] = 1
    for v in survive_set:
        if 0 <= v <= 8:
            out[9 + v] = 1
    return out

