"""Share of PARAM / RAG / ABSTAIN decisions of the proposed method at alpha = 0.30.

    python scripts/action_shares.py    # -> results/tables/action_shares.csv
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rarag import config as C
from rarag.calibration import ABSTAIN, PARAM, RAG, ltt_calibrate, route
from rarag.data import list_score_files, load_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores")
    ap.add_argument("--out", default="results/tables/action_shares.csv")
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    rows = []
    for model, dataset in list_score_files(args.scores):
        p = load_scores(args.scores, model, dataset)
        s0, s1, y0, y1 = p.s0.values, p.s1.values, p.y0.values, p.y1.values
        n = len(p)
        rng = np.random.default_rng(C.SEED)
        acc = []
        for t in range(args.trials):
            perm = rng.permutation(n)
            cal, test = perm[: n // 2], perm[n // 2:]
            (l1, l2), _ = ltt_calibrate(s0[cal], s1[cal], y0[cal], y1[cal], alpha=C.ALPHA_MAIN,
                                        delta=C.DELTA, mu=C.MU, grid_size=C.GRID, seed=t)
            a = route(s0[test], s1[test], l1, l2)
            par, rag = a == PARAM, a == RAG
            acc.append([par.mean(), rag.mean(), (a == ABSTAIN).mean(),
                        y0[test][par].mean() if par.any() else np.nan,
                        y1[test][rag].mean() if rag.any() else np.nan])
        with warnings.catch_warnings():  # no PARAM (or RAG) answer in any split
            warnings.simplefilter("ignore", RuntimeWarning)
            r = np.nanmean(np.array(acc, dtype=float), axis=0)
        rows.append(dict(model=model, dataset=dataset, param_share=r[0], rag_share=r[1],
                         abstain_share=r[2], acc_param=r[3], acc_rag=r[4]))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
