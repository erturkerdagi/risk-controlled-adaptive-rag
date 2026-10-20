"""Tables 3-7 of the paper as CSV files in results/tables/paper/.

    python scripts/make_tables.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rarag import config as C

T = "results/tables"
OUT = os.path.join(T, "paper")
GATES = {"UE gate (mean entropy)": "UE-ent", "UE gate (margin)": "UE margin", "Probe gate": "Probe",
         "X-Router-style gate": "XR"}
KEY = ["model", "dataset"]


def table3(inv):
    cols = ["model", "dataset", "n", "acc_closed_book", "acc_rag", "oracle", "retrieval_helps", "retrieval_hurts"]
    return inv[cols].round(3)


def table4(res, inv):
    q = res[res.alpha == C.ALPHA_MAIN]
    cov = q.pivot_table(index=KEY, columns="method", values="coverage")
    prop = q[q.method == "Proposed"].set_index(KEY)
    best = cov[list(GATES)].max(axis=1)
    best_name = cov[list(GATES)].idxmax(axis=1).map(GATES).where(best > 0, "")
    t = pd.DataFrame({
        "closed_book_acc": inv.set_index(KEY).acc_closed_book.round(2),
        "coverage_always_rag_ca": cov["Always-RAG + CA"].round(3),
        "coverage_best_gate": best.round(3),
        "best_gate": best_name,
        "coverage_proposed": cov["Proposed"].round(3),
        "retrieval_proposed": prop.retrieval.round(3),
        "saving_pct": (100 * (1 - prop.retrieval)).round(1),
        "violation_proposed": prop.violation.round(3),
    })
    return t.reindex(inv.set_index(KEY).index).reset_index()


def table5(res):
    rows = []
    for m in ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]:
        for a in [0.20, 0.30]:
            q = res[(res.model == m) & (res.dataset == "triviaqa") & np.isclose(res.alpha, a)].set_index("method")
            p, b = q.loc["Proposed"], q.loc["Always-RAG + CA"]
            ratio, saving = p.coverage / b.coverage, 1 - p.retrieval
            rows.append({"model": m, "alpha": a, "coverage_proposed": round(p.coverage, 3),
                         "coverage_always_rag_ca": round(b.coverage, 3), "coverage_ratio": round(ratio, 2),
                         "saving_pct": round(100 * saving, 1), "violation": round(p.violation, 3),
                         # rule fixed after the pilot: >= 15% saving, >= 95% of the coverage, violation <= delta
                         "criterion_met": bool(saving >= 0.15 and ratio >= 0.95 and p.violation <= C.DELTA)})
    return pd.DataFrame(rows)


def table6(shift, res):
    rows = []
    for m, g in shift.groupby("m"):
        row = {"m": m}
        for model in ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]:
            s = g[g.model == model]
            row[f"{model} violation mean"] = round(s.violation.mean(), 3)
            row[f"{model} violation worst"] = round(s.violation.max(), 2)
            row[f"{model} coverage"] = round(s.coverage.mean(), 3)
        rows.append(row)
    in_domain = {"m": "in-domain"}
    for model in ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]:
        q = res[(res.model == model) & (res.method == "Proposed") & (res.alpha == C.ALPHA_MAIN)]
        in_domain[f"{model} coverage"] = round(q.coverage.mean(), 3)
    return pd.DataFrame(rows + [in_domain])


def table7(cost):
    cost = cost.assign(seconds=cost.n * cost.sec_per_q)
    g = cost.groupby("model").agg(datasets=("dataset", "size"), questions=("n", "sum"), batch=("batch", "first"),
                                  seconds=("seconds", "sum"), s_min=("sec_per_q", "min"),
                                  s_max=("sec_per_q", "max"), gpu_hours=("gpu_hours", "sum"))
    g["s_per_question"] = (g.seconds / g.questions).round(3)
    order = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3",
             "Qwen2.5-14B-Instruct"]
    g = g.drop(columns="seconds").reindex(order)
    g["gpu_hours"] = g.gpu_hours.round(2)
    return g.reset_index()


def main():
    os.makedirs(OUT, exist_ok=True)
    res = pd.read_csv(f"{T}/results_all.csv")
    inv = pd.read_csv(f"{T}/inventory.csv")
    tables = {
        "table3_data_summary": table3(inv),
        "table4_main_results": table4(res, inv),
        "table5_success_criterion": table5(res),
        "table6_distribution_shift": table6(pd.read_csv(f"{T}/shift.csv"), res),
        "table7_compute_cost": table7(pd.read_csv(f"{T}/compute_cost.csv")),
    }
    pd.set_option("display.width", 200)
    for name, t in tables.items():
        t.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"\n{name}\n{t.to_string(index=False)}")


if __name__ == "__main__":
    main()
