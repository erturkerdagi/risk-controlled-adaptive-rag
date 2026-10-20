"""Distribution shift between datasets and threshold-only recalibration.

Scorers are trained and the thresholds calibrated on a source dataset, then the
policy is applied to a target dataset of the same model. In the second setting
the scorers are kept and only (l1, l2) are recalibrated with m labelled target
questions.
"""
import numpy as np

from . import config as C
from .calibration import ltt_calibrate, outcome_stats, route
from .data import load_pair
from .scorers import Scorer, train_pool_split


def _violation_and_coverage(s0c, s1c, y0c, y1c, s0e, s1e, y0e, y1e, alpha, seed):
    (l1, l2), _ = ltt_calibrate(s0c, s1c, y0c, y1c, alpha=alpha, delta=C.DELTA, mu=C.MU,
                                grid_size=C.GRID, seed=seed)
    st = outcome_stats(route(s0e, s1e, l1, l2), y0e, y1e)
    return st["risk"] > alpha, st["coverage"]


def shift_experiment(cache_root, model, datasets, m_values=(50, 100, 200, 400), n_trials=100,
                     alpha=C.ALPHA_MAIN, run_tag="full", progress=None):
    data = {d: load_pair(cache_root, model, d, run_tag) for d in datasets}
    splits = {d: train_pool_split(len(data[d][0]), C.TRAIN_FRAC, C.SEED) for d in datasets}
    rows = []
    for src in (progress(datasets) if progress else datasets):
        df, H0, H1, _ = data[src]
        tr, te = splits[src]
        scorer = Scorer(seed=C.SEED).fit(df, H0, H1, tr)
        s_src = scorer.scores(df.iloc[te], H0[te], H1[te])
        y0s, y1s = df.y0.values[te], df.y1.values[te]
        n_cal = int(C.CAL_FRAC * len(te))
        for tgt in datasets:
            if tgt == src:
                continue
            dft, H0t, H1t, _ = data[tgt]
            _, tet = splits[tgt]
            s_tgt = scorer.scores(dft.iloc[tet], H0t[tet], H1t[tet])
            y0t, y1t = dft.y0.values[tet], dft.y1.values[tet]
            rng = np.random.default_rng(C.SEED)

            # m = 0: thresholds from the source pool, evaluated on the whole target pool
            out = []
            for t in range(n_trials):
                c = rng.permutation(len(te))[:n_cal]
                out.append(_violation_and_coverage(s_src["s0"][c], s_src["s1"][c], y0s[c], y1s[c],
                                                   s_tgt["s0"], s_tgt["s1"], y0t, y1t, alpha, t))
            v, cov = np.mean(out, axis=0)
            rows.append({"model": model, "source": src, "target": tgt, "alpha": alpha, "m": 0,
                         "violation": v, "coverage": cov})

            # m > 0: thresholds from m labelled target questions, evaluated on the rest
            for m in m_values:
                if m >= len(tet) - 50:
                    continue
                out = []
                for t in range(n_trials):
                    perm = rng.permutation(len(tet))
                    c, e = perm[:m], perm[m:]
                    out.append(_violation_and_coverage(s_tgt["s0"][c], s_tgt["s1"][c], y0t[c], y1t[c],
                                                       s_tgt["s0"][e], s_tgt["s1"][e], y0t[e], y1t[e], alpha, t))
                v, cov = np.mean(out, axis=0)
                rows.append({"model": model, "source": src, "target": tgt, "alpha": alpha, "m": m,
                             "violation": v, "coverage": cov})
    return rows
