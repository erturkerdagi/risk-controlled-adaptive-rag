"""Numbers quoted in the text of the paper, computed from results/.

    python scripts/summarize.py

Paired comparisons use the exact two-sided Wilcoxon signed-rank test on the
pairs with a non-zero difference.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rarag import config as C
from rarag.data import list_score_files, load_scores
from rarag.scorers import safe_auc

T = "results/tables"
GATES = ["UE gate (mean entropy)", "UE gate (margin)", "Probe gate", "X-Router-style gate"]
KEY = ["model", "dataset"]


def paired(diff, label):
    d = np.asarray(diff, float)
    nz = d[np.abs(d) > 1e-9]
    p = wilcoxon(nz, method="exact").pvalue if len(nz) else np.nan
    return (f"{label}: mean {d.mean():+.3f}; better {np.sum(d > 1e-9)}, equal {np.sum(np.abs(d) <= 1e-9)}, "
            f"worse {np.sum(d < -1e-9)}; Wilcoxon p = {p:.2g}")


def main():
    res = pd.read_csv(f"{T}/results_all.csv")
    inv = pd.read_csv(f"{T}/inventory.csv").set_index(KEY)
    cov = res.pivot_table(index=KEY + ["alpha"], columns="method", values="coverage")
    cov["best gate"] = cov[GATES].max(axis=1)
    c3 = cov.xs(C.ALPHA_MAIN, level="alpha")
    cav = cov.groupby(level=KEY).mean()
    out = []

    out.append("## Guarantee")
    prop = res[res.method == "Proposed"]
    naive = res[res.method == "Proposed, naive thresholds"]
    guaranteed = res[res.method != "Proposed, naive thresholds"]
    out.append(f"Proposed: worst violation {prop.violation.max():.3f}; mean by alpha "
               + ", ".join(f"{a:g}: {v:.3f}" for a, v in prop.groupby("alpha").violation.mean().items()))
    out.append(f"Guaranteed methods, configurations above delta: {(guaranteed.violation > C.DELTA).sum()} "
               f"of {len(guaranteed)} (worst {guaranteed.violation.max():.3f})")
    out.append(f"Naive thresholds: mean violation {naive.violation.mean():.3f}, "
               f"above delta in {(naive.violation > C.DELTA).mean():.0%} of configurations")

    out.append(f"\n## Coverage at alpha = {C.ALPHA_MAIN}")
    out.append(paired(c3["Proposed"] - c3["Always-RAG + CA"], "Proposed - Always-RAG + CA"))
    out.append(paired(c3["Proposed"] - c3["best gate"], "Proposed - best gate"))
    out.append(paired(cav["Proposed"] - cav["Always-RAG + CA"], "Proposed - Always-RAG + CA (mean over alpha)"))
    out.append(paired(cav["Proposed"] - cav["best gate"], "Proposed - best gate (mean over alpha)"))

    out.append("\n## Ablations")
    for other in ["Proposed, Bonferroni", "Proposed (no hidden states)"]:
        out.append(paired(cav["Proposed"] - cav[other], f"Proposed - {other} (mean over alpha)"))
        out.append(paired(c3["Proposed"] - c3[other], f"Proposed - {other} (alpha = {C.ALPHA_MAIN})"))
    out.append(f"Coverage at alpha = 0.10: Proposed {cov.xs(0.1, level='alpha')['Proposed'].mean():.3f}, "
               f"Bonferroni {cov.xs(0.1, level='alpha')['Proposed, Bonferroni'].mean():.3f}")

    out.append(f"\n## Retrieval saving at alpha = {C.ALPHA_MAIN}")
    p3 = prop[prop.alpha == C.ALPHA_MAIN].set_index(KEY)
    saving = (1 - p3.retrieval).reindex(inv.index)
    rho, p = spearmanr(inv.acc_closed_book, saving)
    out.append(f"mean {saving.mean():.1%}, median {saving.median():.1%}, max {saving.max():.1%} "
               f"({' / '.join(saving.idxmax())})")
    out.append(f"Spearman rho with closed-book accuracy = {rho:.2f} (p = {p:.2g}, n = {len(saving)})")

    out.append("\n## AUROC on the held-out pools")
    rows = []
    for model, dataset in list_score_files("results/scores"):
        s = load_scores("results/scores", model, dataset)
        rows.append({"y0 entropy": safe_auc(s.y0, -s.ent_mean_0), "y0 probe": safe_auc(s.y0, s.s0_probe),
                     "y0 no hidden": safe_auc(s.y0, s.s0_nohidden), "y0 proposed": safe_auc(s.y0, s.s0),
                     "y1 entropy": safe_auc(s.y1, -s.ent_mean_1), "y1 no hidden": safe_auc(s.y1, s.s1_nohidden),
                     "y1 proposed": safe_auc(s.y1, s.s1)})
    a = pd.DataFrame(rows)
    out.append("means: " + ", ".join(f"{k} {v:.3f}" for k, v in a.mean().items()))
    out.append(paired(a["y0 proposed"] - a["y0 entropy"], "y0: proposed - entropy"))
    out.append(paired(a["y1 proposed"] - a["y1 entropy"], "y1: proposed - entropy"))
    out.append(paired(a["y0 proposed"] - a["y0 no hidden"], "y0: hidden-state contribution"))
    out.append(paired(a["y1 proposed"] - a["y1 no hidden"], "y1: hidden-state contribution"))
    out.append(paired(a["y0 proposed"] - a["y0 probe"], "y0: proposed - probe"))

    print("\n".join(out))


if __name__ == "__main__":
    main()
