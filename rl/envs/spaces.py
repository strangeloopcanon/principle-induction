from __future__ import annotations

import numpy as np


class Space:
    def sample(self, rng: np.random.Generator | None = None):
        raise NotImplementedError


class Discrete(Space):
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = int(n)

    def sample(self, rng: np.random.Generator | None = None):
        g = rng or np.random.default_rng()
        return int(g.integers(self.n))


class MultiBinary(Space):
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = int(n)

    def sample(self, rng: np.random.Generator | None = None):
        g = rng or np.random.default_rng()
        return g.integers(0, 2, size=(self.n,), dtype=np.uint8)


class Box(Space):
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self, rng: np.random.Generator | None = None):
        g = rng or np.random.default_rng()
        return g.uniform(self.low, self.high, size=self.shape).astype(self.dtype)


