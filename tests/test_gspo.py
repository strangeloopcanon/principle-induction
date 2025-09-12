import numpy as np

from rl.algo.gspo import compute_group_weights


def test_compute_group_weights_softmax():
    q, _ = compute_group_weights([0.1, 0.2, 0.3], mode="softmax", eta=5.0)
    assert q.shape == (3,)
    assert np.isclose(q.sum(), 1.0, atol=1e-6)
    assert q.argmax() == 2


def test_compute_group_weights_zscore_invariant_shift():
    q1, _ = compute_group_weights([0.1, 0.2, 0.3], mode="zscore", eta=5.0)
    q2, _ = compute_group_weights([1.1, 1.2, 1.3], mode="zscore", eta=5.0)
    assert np.allclose(q1, q2)


def test_compute_group_weights_rank():
    q, extras = compute_group_weights([0.1, 0.9, 0.5], mode="rank", eta=5.0)
    assert np.isclose(q.sum(), 1.0, atol=1e-6)
    assert extras["ranks"].shape == (3,)


def test_compute_group_weights_baseline_signs():
    q, extras = compute_group_weights([0.0, 1.0], mode="baseline", eta=1.0)
    # Baseline mode can return negative weights; sum of abs normalized to 1
    assert np.isclose(np.abs(q).sum(), 1.0)

