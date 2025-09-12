import numpy as np

from tools.life import neighbor_counts, life_step, life_step_rule


def test_neighbor_counts_simple():
    x = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ], dtype=np.uint8)
    c = neighbor_counts(x, wrap=False)
    # Center (1,1) has neighbors: (0,0)=0,(0,1)=1,(0,2)=0,(1,0)=1,(1,2)=0,(2,0)=0,(2,1)=0,(2,2)=0 => 2
    assert int(c[1, 1]) == 2


def test_life_still_life_block():
    block = np.array([
        [1, 1],
        [1, 1],
    ], dtype=np.uint8)
    nxt = life_step(block, wrap=False)
    assert np.array_equal(block, nxt)


def test_life_blinker_oscillator():
    # Horizontal blinker in middle of 5x5 zero-padded
    x = np.zeros((5, 5), dtype=np.uint8)
    x[2, 1:4] = 1
    y = life_step(x, wrap=False)
    # Vertical blinker expected at col=2, rows 1..3
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:4, 2] = 1
    assert np.array_equal(y, expected)

