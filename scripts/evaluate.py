"""Evaluate all methods on the released score files.

    python scripts/evaluate.py                     # all pairs -> results/tables/results_all.csv
    python scripts/evaluate.py --pairs Qwen2.5-7B-Instruct:triviaqa --trials 50

Runs on a CPU; the full table takes about ten minutes.
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rarag import config as C
from rarag.calibration import evaluate_splits
from rarag.data import list_score_files, load_scores
from rarag.scorers import method_inputs


def evaluate_pair(pool, model, dataset, alphas, n_trials):
    y0, y1 = pool.y0.values, pool.y1.values
    n_cal = int(C.CAL_FRAC * len(pool))
    rows = []
    for method, (s0, s1, kw) in method_inputs(pool).items():
        kw = dict(kw)
        roa = kw.pop("retrieval_on_abstain", True)
        for a in alphas:
            r = evaluate_splits(s0, s1, y0, y1, n_cal=n_cal, n_trials=n_trials, seed=C.SEED,
                                retrieval_on_abstain=roa, alpha=a, delta=C.DELTA, mu=C.MU,
                                grid_size=C.GRID, **kw)
            rows.append({"model": model, "dataset": dataset, "method": method, "alpha": a,
                         "n_pool": len(pool), "n_cal": n_cal, **r,
                         "utility": r["coverage"] - C.MU * r["retrieval"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores")
    ap.add_argument("--out", default="results/tables/results_all.csv")
    ap.add_argument("--pairs", nargs="*", help="model:dataset, default: every score file")
    ap.add_argument("--trials", type=int, default=C.N_TRIALS)
    args = ap.parse_args()

    pairs = [tuple(p.split(":")) for p in args.pairs] if args.pairs else list_score_files(args.scores)
    rows = []
    for model, dataset in pairs:
        t = time.time()
        rows += evaluate_pair(load_scores(args.scores, model, dataset), model, dataset, C.ALPHAS, args.trials)
        print(f"{model:28s} {dataset:16s} {time.time() - t:5.1f}s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
