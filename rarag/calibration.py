"""Threshold calibration for the three-way policy (PARAM / RAG / ABSTAIN).

The policy answers from the closed-book draft if s0 >= l1, otherwise retrieves
and returns the RAG answer if s1 >= l2, and abstains otherwise. The pair
(l1, l2) is chosen with Learn-then-Test so that

    P( R(l1, l2) <= alpha ) >= 1 - delta,

where R is the error rate among answered questions (selective risk).

All candidate threshold pairs are evaluated at once with two matrix products,
which keeps a single calibration in the millisecond range.
"""
import numpy as np
from scipy.stats import binom

PARAM, RAG, ABSTAIN = 0, 1, 2


def route(s0, s1, l1, l2):
    """Apply the policy and return one action per question."""
    s0, s1 = np.asarray(s0), np.asarray(s1)
    actions = np.full(s0.shape, ABSTAIN, dtype=np.int8)
    actions[s1 >= l2] = RAG
    actions[s0 >= l1] = PARAM
    return actions


def outcome_stats(actions, y0, y1, retrieval_on_abstain=True):
    """Coverage, selective risk and retrieval rate of a set of actions.

    With retrieval_on_abstain=True an abstention is counted as a retrieval
    call, since the system only abstains after the RAG answer was rejected.
    The never-retrieve baseline sets it to False.
    """
    y0, y1 = np.asarray(y0), np.asarray(y1)
    err = np.where(actions == PARAM, 1 - y0, np.where(actions == RAG, 1 - y1, 0))
    answered = actions != ABSTAIN
    n, n_acc = len(actions), int(answered.sum())
    retrieved = (actions == RAG) | ((actions == ABSTAIN) & retrieval_on_abstain)
    return {
        "n_acc": n_acc,
        "k_err": int(err[answered].sum()),
        "coverage": n_acc / n,
        "retrieval_rate": float(retrieved.mean()),
        "risk": (err[answered].sum() / n_acc) if n_acc else 0.0,
    }


def make_grid(scores, size, fixed=None):
    """Quantiles of the scores plus +inf (which switches the gate off)."""
    if fixed is not None:
        return np.array([fixed], dtype=float)
    q = np.quantile(scores, np.linspace(0.0, 1.0, size))
    return np.unique(np.concatenate([q, [np.inf]]))


def grid_counts(s0, s1, y0, y1, g1, g2):
    """Answered count, error count and PARAM count for every (l1, l2).

    Returns three arrays of shape (len(g1), len(g2)).
    """
    s0, s1 = np.asarray(s0, float), np.asarray(s1, float)
    e0, e1 = 1.0 - np.asarray(y0, float), 1.0 - np.asarray(y1, float)
    a0 = (s0[None, :] >= g1[:, None]).astype(np.float64)   # passes gate 1
    a1 = (s1[None, :] >= g2[:, None]).astype(np.float64)   # passes gate 2
    n_param = a0.sum(1)
    k_param = a0 @ e0
    not_a0 = 1.0 - a0
    n_rag = not_a0 @ a1.T
    k_rag = not_a0 @ (a1 * e1[None, :]).T
    n_acc = n_param[:, None] + n_rag
    k_err = k_param[:, None] + k_rag
    n_par = np.broadcast_to(n_param[:, None], n_acc.shape)
    return n_acc, k_err, n_par


def binom_pvalues(k_err, n_acc, alpha):
    """Conditional binomial p-values for H0: R > alpha (p = 1 if nothing is answered)."""
    n = np.asarray(n_acc)
    p = binom.cdf(np.asarray(k_err), np.maximum(n, 1), alpha)
    return np.where(n > 0, p, 1.0)


def _objective(n_acc, n_par, n, mu, retrieval_on_abstain):
    coverage = n_acc / n
    retrieval = (n - n_par) / n if retrieval_on_abstain else (n_acc - n_par) / n
    return coverage - mu * retrieval


def ltt_calibrate(s0, s1, y0, y1, alpha=0.2, delta=0.1, mu=0.3, grid_size=30,
                  method="fixed_sequence", order_frac=0.3, seed=0,
                  fix_l1=None, fix_l2=None, retrieval_on_abstain=True):
    """Select (l1, l2) on a labelled calibration set.

    method:
        "fixed_sequence"  order learned on a random `order_frac` part, fixed-sequence
                          testing on the rest (the proposed procedure)
        "bonferroni"      every pair tested at level delta / |grid|
        "naive"           best pair among those with empirical risk <= alpha
                          (no guarantee; reference only)

    Among certified pairs the one maximising coverage - mu * retrieval_rate is
    returned. If nothing is certified, (inf, inf) is returned and the system
    abstains on every question.
    """
    s0, s1, y0, y1 = (np.asarray(v) for v in (s0, s1, y0, y1))
    n = len(s0)
    g1 = make_grid(s0, grid_size, fix_l1)
    g2 = make_grid(s1, grid_size, fix_l2)

    if method == "fixed_sequence":
        perm = np.random.default_rng(seed).permutation(n)
        k = int(order_frac * n)
        ord_idx, tst_idx = perm[:k], perm[k:]
        na_o, ke_o, _ = grid_counts(s0[ord_idx], s1[ord_idx], y0[ord_idx], y1[ord_idx], g1, g2)
        order = np.argsort(binom_pvalues(ke_o, na_o, alpha).ravel(), kind="stable")
        na, ke, npar = grid_counts(s0[tst_idx], s1[tst_idx], y0[tst_idx], y1[tst_idx], g1, g2)
        p = binom_pvalues(ke, na, alpha).ravel()[order]
        fail = np.nonzero(p > delta)[0]
        n_ok = fail[0] if len(fail) else len(p)
        valid = np.zeros(p.size, bool)
        valid[order[:n_ok]] = True
        n_eval = len(tst_idx)
    elif method == "bonferroni":
        na, ke, npar = grid_counts(s0, s1, y0, y1, g1, g2)
        valid = (binom_pvalues(ke, na, alpha) <= delta / (len(g1) * len(g2))).ravel()
        n_eval = n
    elif method == "naive":
        na, ke, npar = grid_counts(s0, s1, y0, y1, g1, g2)
        risk = np.where(na > 0, ke / np.maximum(na, 1), 0.0)
        valid = (risk <= alpha).ravel()
        n_eval = n
    else:
        raise ValueError(f"unknown method: {method}")

    if not valid.any():
        return (np.inf, np.inf), {"n_valid": 0}
    obj = _objective(na, npar, n_eval, mu, retrieval_on_abstain).ravel()
    obj = np.where(valid, obj, -np.inf)
    j = int(np.argmax(obj))
    i1, i2 = np.unravel_index(j, na.shape)
    return (float(g1[i1]), float(g2[i2])), {"n_valid": int(valid.sum()), "objective": float(obj[j])}


def evaluate_splits(s0, s1, y0, y1, n_cal, n_trials=300, seed=0, retrieval_on_abstain=True, **kw):
    """Repeat calibration on random calibration/test splits of a pool.

    Returns the violation rate (fraction of splits with test risk > alpha) and
    the mean test risk, coverage and retrieval rate. Extra keyword arguments are
    passed to `ltt_calibrate`.
    """
    s0, s1, y0, y1 = (np.asarray(v) for v in (s0, s1, y0, y1))
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trials):
        perm = rng.permutation(len(s0))
        cal, test = perm[:n_cal], perm[n_cal:]
        (l1, l2), _ = ltt_calibrate(s0[cal], s1[cal], y0[cal], y1[cal], seed=t,
                                    retrieval_on_abstain=retrieval_on_abstain, **kw)
        st = outcome_stats(route(s0[test], s1[test], l1, l2), y0[test], y1[test], retrieval_on_abstain)
        rows.append((st["risk"], st["coverage"], st["retrieval_rate"]))
    r = np.array(rows)
    alpha = kw.get("alpha", 0.2)
    return {
        "violation": float((r[:, 0] > alpha).mean()),
        "risk": float(r[:, 0].mean()),
        "coverage": float(r[:, 1].mean()),
        "retrieval": float(r[:, 2].mean()),
        "coverage_sd": float(r[:, 1].std()),
        "retrieval_sd": float(r[:, 2].std()),
    }
