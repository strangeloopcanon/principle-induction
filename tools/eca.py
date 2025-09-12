import numpy as np
from typing import Optional


def rule_number_to_table(rule_number: int) -> np.ndarray:
    """Return 8-bit table for ECA rule number.

    Bit i corresponds to neighborhood pattern with binary value i = (l<<2)|(c<<1)|r,
    where 0=000, 7=111. Table entries are in {0,1} (uint8).
    """
    if not (0 <= rule_number <= 255):
        raise ValueError("rule_number must be in [0,255]")
    table = np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)
    return table


def eca_step(state: np.ndarray, rule_number: Optional[int] = None, rule_table: Optional[np.ndarray] = None) -> np.ndarray:
    """One ECA step with toroidal wrap.

    Args:
        state: 1D array (W,) or 2D array (B, W) with values {0,1}.
        rule_number: optional integer in [0,255].
        rule_table: optional length-8 array in {0,1}. If provided, used over rule_number.

    Returns:
        Next state with same shape and dtype uint8.
    """
    if (rule_table is None) == (rule_number is None):
        raise ValueError("Provide exactly one of rule_number or rule_table")
    table = rule_table if rule_table is not None else rule_number_to_table(int(rule_number))

    x = np.asarray(state, dtype=np.uint8)
    if x.ndim == 1:
        left = np.roll(x, 1)
        right = np.roll(x, -1)
        idx = (left << 2) | (x << 1) | right
        return table[idx].astype(np.uint8)
    elif x.ndim == 2:
        left = np.roll(x, 1, axis=1)
        right = np.roll(x, -1, axis=1)
        idx = (left << 2) | (x << 1) | right
        # table[idx] with table shape (8,) broadcasts over batch/width
        return table[idx].astype(np.uint8)
    else:
        raise ValueError("state must be 1D or 2D")

