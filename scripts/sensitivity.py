"""Change one setting at a time for the proposed method (alpha = 0.30).

    python scripts/sensitivity.py      # -> results/tables/sensitivity.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rarag import config as C
from rarag.calibration import evaluate_splits
from rarag.data import list_score_files, load_scores

GRID = {
    "n_cal": [100, 200, 400, 750],
    "mu": [0.0, 0.1, 0.3, 0.5, 1.0],
    "order_frac": [0.1, 0.2, 0.3, 0.5],
    "grid_size": [10, 20, 30, 50],
    "delta": [0.05, 0.1, 0.2],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores")
    ap.add_argument("--out", default="results/tables/sensitivity.csv")
    ap.add_argument("--trials", type=int, default=C.N_TRIALS)
    args = ap.parse_args()

    base = dict(alpha=C.ALPHA_MAIN, delta=C.DELTA, mu=C.MU, grid_size=C.GRID, order_frac=0.3)
    rows = []
    for model, dataset in list_score_files(args.scores):
        pool = load_scores(args.scores, model, dataset)
        for param, values in GRID.items():
            for v in values:
                kw, n_cal = dict(base), len(pool) // 2
                if param == "n_cal":
                    n_cal = v
                else:
                    kw[param] = v
                r = evaluate_splits(pool.s0.values, pool.s1.values, pool.y0.values, pool.y1.values,
                                    n_cal=n_cal, n_trials=args.trials, seed=C.SEED, **kw)
                rows.append(dict(model=model, dataset=dataset, param=param, value=v,
                                 delta_used=kw["delta"], **r))
        print(model, dataset, flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)

    d = pd.DataFrame(rows)
    print(d.groupby(["param", "value"])[["violation", "coverage", "retrieval"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
