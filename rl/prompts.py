from __future__ import annotations

import re
import numpy as np
from typing import Tuple


def format_eca_obs(pairs: np.ndarray) -> str:
    """Format ECA (x,y) pairs into a prompt with strict RULE= answer instruction.

    pairs: (N, 2, W) with binary values.
    """
    lines = [
        "You are given pairs (x, y) of a 1D binary cellular automaton where y = step_ECA(x) under a fixed, unknown rule.",
        "Infer the Elementary Cellular Automaton rule number (0-255) that maps x->y.",
        "Answer strictly in the form: RULE=<int>",
        "",
    ]
    for i in range(pairs.shape[0]):
        x = ''.join(map(str, pairs[i, 0].tolist()))
        y = ''.join(map(str, pairs[i, 1].tolist()))
        lines.append(f"Pair {i+1}:")
        lines.append(f"x={x}")
        lines.append(f"y={y}")
    lines.append("")
    lines.append("RULE=")
    return "\n".join(lines)


def parse_eca_action(text: str) -> Tuple[int, str]:
    """Parse RULE=<int> and return (rule_number, canonical_text)."""
    m = re.search(r"RULE\s*=\s*(\d{1,3})", text)
    if not m:
        return 0, "RULE=0"
    v = int(m.group(1))
    v = max(0, min(255, v))
    return v, f"RULE={v}"


def format_life_obs(pairs: np.ndarray) -> str:
    """Format Life-like (X,Y) pairs into a prompt with B/S answer instruction.

    pairs: (N, 2, H, W) with binary values.
    """
    H, W = pairs.shape[2], pairs.shape[3]
    lines = [
        "You are given pairs (X, Y) of 2D binary grids where Y = step_BS(X) under a fixed, unknown Life-like B/S rule.",
        "Infer the B (birth) and S (survive) neighbor-count sets of the rule.",
        "Answer strictly as: B={...} S={...} using digits 0..8 in ascending order.",
        "",
    ]
    for i in range(pairs.shape[0]):
        lines.append(f"Pair {i+1}:")
        lines.append("X:")
        for r in range(H):
            lines.append(''.join(map(str, pairs[i, 0, r].tolist())))
        lines.append("Y:")
        for r in range(H):
            lines.append(''.join(map(str, pairs[i, 1, r].tolist())))
    lines.append("")
    lines.append("B={ } S={ }")
    return "\n".join(lines)


def parse_life_action(text: str) -> Tuple[np.ndarray, str]:
    """Parse B{digits}/S{digits} or B={..} S={..} into 18-bit vector and canonical text.

    Returns (bits, canonical_text) where bits is np.ndarray shape (18,).
    """
    b_set = set()
    s_set = set()
    m1 = re.search(r"B\s*=\s*\{\s*([0-8,\s]*)\}\s*.*S\s*=\s*\{\s*([0-8,\s]*)\}", text)
    if m1:
        b_digits = re.findall(r"[0-8]", m1.group(1))
        s_digits = re.findall(r"[0-8]", m1.group(2))
        b_set = set(int(d) for d in b_digits)
        s_set = set(int(d) for d in s_digits)
    else:
        m2 = re.search(r"B\s*([0-8]*)\s*/\s*S\s*([0-8]*)", text)
        if m2:
            b_set = set(int(d) for d in list(m2.group(1)))
            s_set = set(int(d) for d in list(m2.group(2)))
    if not b_set and not s_set:
        b_set = {3}
        s_set = {2, 3}
    bits = np.zeros(18, dtype=np.uint8)
    for d in sorted(b_set):
        bits[d] = 1
    for d in sorted(s_set):
        bits[9 + d] = 1
    btxt = ''.join(str(d) for d in sorted(b_set))
    stxt = ''.join(str(d) for d in sorted(s_set))
    return bits, f"B{btxt}/S{stxt}"

