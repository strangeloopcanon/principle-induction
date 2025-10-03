import numpy as np

from rl.prompts import (
    format_eca_obs,
    parse_eca_action,
    format_life_obs,
    parse_life_action,
    format_eca_rollout_obs,
    parse_eca_rollout_action,
)


def test_parse_eca_action():
    rule, txt = parse_eca_action("RULE=110")
    assert rule == 110
    assert txt == "RULE=110"
    rule, txt = parse_eca_action("some text RULE=999 trailing")
    assert rule == 255


def test_parse_life_action():
    bits, txt = parse_life_action("B3/S23")
    assert bits[3] == 1 and bits[9 + 2] == 1 and bits[9 + 3] == 1
    assert txt == "B3/S23"
    bits2, txt2 = parse_life_action("B={3} S={2,3}")
    assert (bits == bits2).all()
    assert txt2 == "B3/S23"


def test_parse_eca_rollout_action():
    x0 = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    prompt = format_eca_rollout_obs(x0, rule=110, steps=10)
    bits, canon = parse_eca_rollout_action("Y=10101", width=5)
    assert bits.shape == (5,)
    assert canon == "Y=10101"
    bits2, canon2 = parse_eca_rollout_action("Y=101", width=5)
    # Pads to width
    assert canon2 == "Y=10100"
