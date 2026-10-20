import numpy as np
import pytest

from rarag.calibration import (ABSTAIN, PARAM, RAG, evaluate_splits, grid_counts, ltt_calibrate, make_grid,
                               outcome_stats, route)


def synthetic(n, seed=0):
    rng = np.random.default_rng(seed)
    s0, s1 = rng.uniform(size=n), rng.uniform(size=n)
    y0 = (rng.uniform(size=n) < s0).astype(int)
    y1 = (rng.uniform(size=n) < 0.3 + 0.6 * s1).astype(int)
    return s0, s1, y0, y1


def test_route_order():
    a = route([0.9, 0.1, 0.1], [0.9, 0.9, 0.1], 0.5, 0.5)
    assert a.tolist() == [PARAM, RAG, ABSTAIN]


def test_grid_counts_match_loop():
    s0, s1, y0, y1 = synthetic(400)
    g1, g2 = make_grid(s0, 12), make_grid(s1, 12)
    n_acc, k_err, n_par = grid_counts(s0, s1, y0, y1, g1, g2)
    for i, l1 in enumerate(g1):
        for j, l2 in enumerate(g2):
            st = outcome_stats(route(s0, s1, l1, l2), y0, y1)
            assert n_acc[i, j] == st["n_acc"]
            assert k_err[i, j] == st["k_err"]
            assert n_par[i, j] == np.sum(s0 >= l1)


def test_infinite_thresholds_abstain():
    s0, s1, y0, y1 = synthetic(50)
    st = outcome_stats(route(s0, s1, np.inf, np.inf), y0, y1)
    assert st["coverage"] == 0 and st["risk"] == 0 and st["retrieval_rate"] == 1


def test_never_retrieve_counts_no_retrieval():
    s0, s1, y0, y1 = synthetic(50)
    st = outcome_stats(route(s0, s1, np.inf, np.inf), y0, y1, retrieval_on_abstain=False)
    assert st["retrieval_rate"] == 0


def test_unknown_method():
    s0, s1, y0, y1 = synthetic(50)
    with pytest.raises(ValueError):
        ltt_calibrate(s0, s1, y0, y1, method="holm")


@pytest.mark.parametrize("method", ["fixed_sequence", "bonferroni"])
def test_violation_rate_below_delta(method):
    s0, s1, y0, y1 = synthetic(3000, seed=1)
    r = evaluate_splits(s0, s1, y0, y1, n_cal=750, n_trials=100, seed=0,
                        alpha=0.3, delta=0.1, mu=0.3, grid_size=20, method=method)
    assert r["violation"] <= 0.1 + 0.05  # binomial slack over 100 splits
    assert r["coverage"] > 0


def test_fixed_sequence_not_worse_than_bonferroni_on_average():
    s0, s1, y0, y1 = synthetic(3000, seed=2)
    kw = dict(n_cal=750, n_trials=50, seed=0, alpha=0.3, delta=0.1, mu=0.3, grid_size=20)
    fs = evaluate_splits(s0, s1, y0, y1, method="fixed_sequence", **kw)
    bf = evaluate_splits(s0, s1, y0, y1, method="bonferroni", **kw)
    assert fs["coverage"] >= bf["coverage"] - 0.02
