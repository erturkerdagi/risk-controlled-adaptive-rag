"""Reading the generation cache written by notebook 01 and the released score files."""
import glob
import os

import numpy as np
import pandas as pd

from .labels import contains_answer

# Token-level uncertainty features stored for the closed-book (_0) and RAG (_1) answers.
U0 = ["ent_first_0", "ent_mean_0", "ent_max_0", "logp_mean_0", "margin_first_0", "n_tokens_0"]
U1 = [c.replace("_0", "_1") for c in U0]

# Retrieval features phi(x); the cache also stores the candidate count as a 7th entry.
PHI = ["bm25_top1", "bm25_gap12", "e5_top1", "e5_gap12", "e5_mean5", "e5_std5"]


def pair_dir(cache_root, model, dataset, run_tag="full"):
    return os.path.join(cache_root, model, run_tag, dataset)


def available_pairs(cache_root, models, datasets, run_tag="full"):
    return [(m, d) for m in models for d in datasets
            if glob.glob(os.path.join(pair_dir(cache_root, m, d, run_tag), "chunk_*.parquet"))]


def load_pair(cache_root, model, dataset, run_tag="full"):
    """Return (records, H0, H1, layer_ids) for one model-dataset pair.

    Labels are recomputed from the stored answers so that every pair uses the
    same rule. H0 and H1 have shape (n, n_layers, hidden_size).
    """
    parts = sorted(p for p in glob.glob(os.path.join(pair_dir(cache_root, model, dataset, run_tag),
                                                     "chunk_*.parquet")) if ".tmp" not in p)
    if not parts:
        raise FileNotFoundError(f"no cache for {model} / {dataset}")
    dfs, h0s, h1s = [], [], []
    for p in parts:
        dfs.append(pd.read_parquet(p))
        z = np.load(p.replace(".parquet", ".npz"))
        h0s.append(z["H0"])
        h1s.append(z["H1"])
        layers = z["layer_ids"]
    df = pd.concat(dfs, ignore_index=True)
    df["y0"] = [contains_answer(p, g) for p, g in zip(df.pred0, df.golden_answers)]
    df["y1"] = [contains_answer(p, g) for p, g in zip(df.pred1, df.golden_answers)]
    return df, np.concatenate(h0s), np.concatenate(h1s), np.asarray(layers)


def score_file(scores_dir, model, dataset):
    return os.path.join(scores_dir, f"{model}__{dataset}.csv.gz")


def load_scores(scores_dir, model, dataset):
    """Held-out pool with labels, entropy baselines and all scorer outputs."""
    return pd.read_csv(score_file(scores_dir, model, dataset), dtype={"id": str},
                       float_precision="round_trip")


def list_score_files(scores_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(scores_dir, "*__*.csv.gz"))):
        model, dataset = os.path.basename(f)[: -len(".csv.gz")].split("__")
        out.append((model, dataset))
    return out
