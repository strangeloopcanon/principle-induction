import numpy as np

from tools.eca import rule_number_to_table, eca_step


def test_rule_table_bits():
    # Rule 30 binary: 00011110 (LSB=pattern 000)
    tbl = rule_number_to_table(30)
    assert tbl.shape == (8,)
    # Check a couple of patterns: 111->0 (idx 7), 110->0 (6), 101->0 (5), 100->1 (4)
    assert int(tbl[7]) == 0
    assert int(tbl[6]) == 0
    assert int(tbl[5]) == 0
    assert int(tbl[4]) == 1


def test_eca_step_deterministic():
    x = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    y1 = eca_step(x, rule_number=110)
    y2 = eca_step(x, rule_number=110)
    assert np.array_equal(y1, y2)

