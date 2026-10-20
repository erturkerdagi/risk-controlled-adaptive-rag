"""Figures 2-13 of the paper from the tables in results/.

    python scripts/make_figures.py            # all figures -> figures/
    python scripts/make_figures.py 4 11       # only Figures 4 and 11

Figure 1 (the flow of the method) is a drawing and is not generated here.
"""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rarag import config as C
from rarag.data import load_scores

T, SC, OUT = "results/tables", "results/scores", "figures"

mpl.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a85",
    "axes.labelcolor": "#2f2f2c", "xtick.color": "#5c5c57", "ytick.color": "#5c5c57",
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.axisbelow": True,
    "mathtext.fontset": "dejavusans",
})
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#2f2f2c"

# method -> (colour, marker, legend label)
STYLE = {
    "Proposed": (PAL[0], "o", "Proposed"),
    "Always-RAG + CA": (PAL[1], "s", "Always-RAG + CA"),
    "UE gate (mean entropy)": (PAL[2], "^", "UE gate (mean entropy)"),
    "Probe gate": (PAL[3], "D", "Probe gate"),
    "X-Router-style gate": (PAL[4], "v", "X-Router-style gate"),
    "Proposed (no hidden states)": (PAL[5], "P", "Proposed (no hidden states)"),
    "Never-retrieve + CA": (PAL[6], "X", "Never-retrieve + CA"),
    "Proposed, naive thresholds": (PAL[7], "x", "Naive thresholds (no guarantee)"),
    "UE gate (margin)": ("#8a8a85", "<", "UE gate (margin)"),
    "Proposed, Bonferroni": ("#5c5c57", ">", "Proposed, Bonferroni"),
}
DS = {"triviaqa": "TriviaQA", "nq": "NQ", "popqa": "PopQA", "squad": "SQuAD", "hotpotqa": "HotpotQA",
      "2wikimultihopqa": "2WikiMQA", "musique": "MuSiQue"}
MOD = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
       "Mistral-7B-Instruct-v0.3": "Mistral-7B", "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
       "Qwen2.5-14B-Instruct": "Qwen2.5-14B"}
MAIN = C.MAIN_MODELS
A = C.ALPHA_MAIN


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


def sci(p):
    """2.3e-06 -> 2.3\\times10^{-6} for mathtext."""
    m, e = f"{p:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def fig2(R):
    inv = R["inv"]
    f, axs = plt.subplots(2, 3, figsize=(9.6, 5.0), sharey="row", layout="constrained")
    x, w = np.arange(len(DS)), 0.27
    for j, m in enumerate(MAIN):
        s = inv[inv.model == m].set_index("dataset").loc[list(DS)]
        ax = axs[0, j]
        ax.bar(x - w, s.acc_closed_book, w, color=PAL[6], label="Closed-book")
        ax.bar(x, s.acc_rag, w, color=PAL[1], label="RAG")
        ax.bar(x + w, s.oracle, w, color=PAL[0], label="Oracle (max)")
        ax.set_title(MOD[m], fontsize=9, color=INK)
        ax.set_xticks(x, list(DS.values()), rotation=40, ha="right", fontsize=7.5)
        ax.set_ylim(0, 0.85)
        ax = axs[1, j]
        ax.bar(x - w / 2, s.retrieval_helps, w, color=PAL[2], label=r"Retrieval helps ($y_0=0,\ y_1=1$)")
        ax.bar(x + w / 2, s.retrieval_hurts, w, color=PAL[7], label=r"Retrieval hurts ($y_0=1,\ y_1=0$)")
        ax.set_xticks(x, list(DS.values()), rotation=40, ha="right", fontsize=7.5)
    axs[0, 0].set_ylabel("accuracy")
    axs[1, 0].set_ylabel("fraction of questions")
    h1, l1 = axs[0, 0].get_legend_handles_labels()
    h2, l2 = axs[1, 0].get_legend_handles_labels()
    f.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=5, fontsize=7.5)
    save(f, "Figure2")


def fig3(R):
    g = R["res"].groupby(["method", "alpha"]).violation.mean().reset_index()
    f, ax = plt.subplots(figsize=(3.9, 2.8), layout="constrained")
    for m in ["Proposed, naive thresholds", "Proposed", "Proposed, Bonferroni"]:
        s = g[g.method == m]
        c, mk, lab = STYLE[m]
        ax.plot(s.alpha, s.violation, color=c, marker=mk, ms=6, lw=2 if m == "Proposed" else 1.5,
                ls=":" if "Bonferroni" in m else "-", label=lab)
    ax.axhline(C.DELTA, color=INK, lw=1, ls="--")
    ax.text(0.1, C.DELTA + 0.02, rf"target $\delta = {C.DELTA:.2f}$", color=INK, fontsize=8)
    ax.set_xlabel(r"risk level $\alpha$")
    ax.set_ylabel("mean violation rate")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    save(f, "Figure3")


def fig4(R):
    res = R["res"]
    order = ["Proposed, naive thresholds", "Proposed", "Proposed, Bonferroni", "Proposed (no hidden states)",
             "Probe gate", "X-Router-style gate", "UE gate (mean entropy)", "UE gate (margin)",
             "Always-RAG + CA", "Never-retrieve + CA"]
    f, ax = plt.subplots(figsize=(7.4, 3.0), layout="constrained")
    rng = np.random.default_rng(0)  # horizontal jitter only
    for i, m in enumerate(order):
        v = res[res.method == m].violation.values
        c, _, _ = STYLE[m]
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(v)), v, s=7, color=c, alpha=0.55, linewidth=0)
        ax.plot([i - 0.32, i + 0.32], [v.mean()] * 2, color=INK, lw=1.6)
        ax.text(i, max(v.max(), 0.1) + 0.03, f"{v.max():.3f}", ha="center", fontsize=7, color=INK)
    ax.axhline(C.DELTA, color=INK, lw=1, ls="--")
    ax.set_xticks(range(len(order)), [STYLE[m][2] for m in order], rotation=30, ha="right", fontsize=7.5)
    n_conf = len(res) // res.method.nunique()
    ax.set_ylabel(f"violation rate ({n_conf} configurations)")
    ax.set_ylim(-0.02, 0.85)
    save(f, "Figure4")


def fig5(R):
    meths = ["Proposed", "Proposed, Bonferroni", "Proposed (no hidden states)", "Probe gate",
             "UE gate (mean entropy)", "X-Router-style gate", "Always-RAG + CA", "Never-retrieve + CA"]
    g = R["res"].groupby(["method", "alpha"]).coverage.mean().reset_index()
    f, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
    for m in meths:
        s = g[g.method == m]
        c, mk, lab = STYLE[m]
        ax.plot(s.alpha, s.coverage, color=c, marker=mk, ms=5, lw=2.2 if m == "Proposed" else 1.3,
                ls=":" if "Bonferroni" in m else "-", label=lab)
    ax.set_xlabel(r"risk level $\alpha$")
    ax.set_ylabel(f"mean coverage ({R['res'][['model', 'dataset']].drop_duplicates().shape[0]} pairs)")
    ax.legend(fontsize=7, loc="upper left")
    save(f, "Figure5")


def fig6(R):
    meths = ["Proposed", "Always-RAG + CA", "UE gate (mean entropy)", "Probe gate", "X-Router-style gate",
             "Proposed (no hidden states)"]
    dss = ["triviaqa", "popqa", "squad", "2wikimultihopqa"]
    res = R["res"]
    q = res[(res.alpha == A) & res.method.isin(meths)]
    f, axs = plt.subplots(3, 4, figsize=(9.2, 6.4), sharex=True, sharey=True, layout="constrained")
    for i, mo in enumerate(MAIN):
        for j, d in enumerate(dss):
            ax = axs[i, j]
            s = q[(q.model == mo) & (q.dataset == d)]
            for m in meths:
                r = s[s.method == m]
                c, mk, lab = STYLE[m]
                ax.scatter(r.retrieval, r.coverage, color=c, marker=mk, s=44, zorder=5 if m == "Proposed" else 3,
                           edgecolor="white", linewidth=1.1, label=lab)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(DS[d], fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(MOD[mo], fontsize=9)
            if i == 2:
                ax.set_xlabel("retrieval rate", fontsize=9)
    f.supylabel("coverage", fontsize=9)
    h, lab = axs[0, 0].get_legend_handles_labels()
    f.legend(h, lab, loc="outside lower center", ncol=3, fontsize=8)
    save(f, "Figure6")


def fig7(R):
    act = R["act"]
    rows = []
    for m in MAIN:
        for d in DS:
            r = act[(act.model == m) & (act.dataset == d)].iloc[0]
            rows.append((f"{MOD[m]} · {DS[d]}", r.param_share, r.rag_share, r.abstain_share))
    rows = rows[::-1]
    f, ax = plt.subplots(figsize=(6.4, 5.6), layout="constrained")
    y = np.arange(len(rows))
    a = np.array([r[1:] for r in rows])
    ax.barh(y, a[:, 0], color=PAL[2], label="PARAM (answer without retrieval)")
    ax.barh(y, a[:, 1], left=a[:, 0], color=PAL[0], label="RAG (answer with retrieval)")
    ax.barh(y, a[:, 2], left=a[:, 0] + a[:, 1], color="#d8d7d2", label="ABSTAIN (withhold answer)")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(rf"fraction of test questions per action ($\alpha = {A:.2f}$)")
    ax.grid(axis="y", visible=False)
    for k in range(1, len(MAIN)):
        ax.axhline(len(DS) * k - 0.5, color="white", lw=3)
    f.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=7.5)
    save(f, "Figure7")


def fig8(R):
    mods = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Llama-3.1-8B-Instruct",
            "Mistral-7B-Instruct-v0.3"]
    g = R["auc"].groupby("model").mean(numeric_only=True).loc[mods]
    colour = {"−entropy": PAL[2], r"Probe only ($h_0$)": PAL[3], "Proposed (no hidden states)": PAL[5],
              "Proposed": PAL[0]}
    panels = [
        (["y0: -entropy", "y0: hidden probe", "y0: Proposed (no hidden states)", "y0: Proposed"],
         ["−entropy", r"Probe only ($h_0$)", "Proposed (no hidden states)", "Proposed"],
         r"(a) Closed-book answer ($y_0$)"),
        (["y1: -entropy", "y1: Proposed (no hidden states)", "y1: Proposed"],
         ["−entropy", "Proposed (no hidden states)", "Proposed"], r"(b) RAG answer ($y_1$)"),
    ]
    f, axs = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=True, layout="constrained")
    x = np.arange(len(mods))
    for ax, (cols, labs, title) in zip(axs, panels):
        w = 0.8 / len(cols)
        for i, (c, lab) in enumerate(zip(cols, labs)):
            ax.bar(x - 0.4 + w * (i + 0.5), g[c], w, color=colour[lab], label=lab)
        ax.set_xticks(x, [MOD[m] for m in mods], rotation=25, ha="right", fontsize=7.5)
        ax.set_ylim(0.6, 0.9)
        ax.set_title(title, fontsize=9)
    axs[0].set_ylabel("mean AUROC")
    f.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=7.5)
    save(f, "Figure8")


def fig9(R):
    pairs = [("Qwen2.5-7B-Instruct", "squad"), ("Llama-3.1-8B-Instruct", "popqa"),
             ("Mistral-7B-Instruct-v0.3", "2wikimultihopqa")]
    f, axs = plt.subplots(1, 3, figsize=(9.0, 2.9), sharey=True, layout="constrained")
    for ax, (m, d) in zip(axs, pairs):
        p = load_scores(SC, m, d)
        y1 = p.y1.values
        for sc, lab, c in [(-p.ent_mean_1.values, "−entropy", PAL[2]),
                           (p.s1_nohidden.values, "Proposed (no hidden states)", PAL[5]),
                           (p.s1.values, "Proposed", PAL[0])]:
            idx = np.argsort(-sc)
            k = np.arange(1, len(y1) + 1)
            risk = np.cumsum(1 - y1[idx]) / k
            ax.plot((k / len(y1))[9:], risk[9:], color=c, lw=1.8 if lab == "Proposed" else 1.3, label=lab)
        ax.axhline(A, color=INK, lw=1, ls="--")
        ax.set_title(f"{MOD[m]} · {DS[d]}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.8)
    f.supxlabel("coverage (fraction of most confident questions)", fontsize=9)
    axs[0].set_ylabel("selective risk (error rate)")
    axs[0].legend(fontsize=7.5, loc="upper left")
    axs[2].text(0.72, A - 0.05, rf"$\alpha = {A:.2f}$", fontsize=8, color=INK)
    save(f, "Figure9")


def fig10(R):
    g = R["sens"].groupby(["param", "value"])[["violation", "coverage", "retrieval"]].mean().reset_index()
    panels = [("n_cal", r"(a) Calibration set size $n$", r"$n$"), ("mu", r"(b) Cost weight $\mu$", r"$\mu$"),
              ("delta", r"(c) Tolerance $\delta$", r"$\delta$"),
              ("order_frac", r"(d) Ordering fraction $\rho$", r"$\rho$"),
              ("grid_size", r"(e) Grid size $G$", r"$G$")]
    f, axs = plt.subplots(1, 5, figsize=(11, 2.7), sharey=True, layout="constrained")
    for ax, (p, title, xl) in zip(axs, panels):
        s = g[g.param == p].sort_values("value")
        x = np.arange(len(s))
        ax.plot(x, s.coverage, color=PAL[0], marker="o", lw=2, label="coverage")
        ax.plot(x, s.retrieval, color=PAL[1], marker="s", lw=1.6, label="retrieval rate")
        ax.plot(x, s.violation, color=PAL[7], marker="^", lw=1.6, label="violation rate")
        if p == "delta":
            ax.plot(x, s.value, color=INK, lw=1, ls="--", label=r"$\delta$ (target)")
        else:
            ax.axhline(C.DELTA, color=INK, lw=1, ls="--")
        ax.set_xticks(x, [f"{v:g}" for v in s.value])
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
    axs[0].set_ylabel(rf"mean (27 pairs, $\alpha = {A:.2f}$)")
    h, lab = axs[2].get_legend_handles_labels()
    f.legend(h, lab, loc="outside lower center", ncol=4, fontsize=8)
    save(f, "Figure10")


def fig11(R):
    res, inv = R["res"], R["inv"]
    q = res[(res.method == "Proposed") & (res.alpha == A)].merge(inv[["model", "dataset", "acc_closed_book"]],
                                                                 on=["model", "dataset"])
    q["saving"] = 1 - q.retrieval
    f, ax = plt.subplots(figsize=(4.2, 3.0), layout="constrained")
    for i, m in enumerate(MOD):
        s = q[q.model == m]
        ax.scatter(s.acc_closed_book, s.saving, color=PAL[i], marker="osD^v"[i], s=40, edgecolor="white",
                   linewidth=1.1, label=MOD[m], zorder=3)
    rho, p = spearmanr(q.acc_closed_book, q.saving)
    ax.text(0.02, 0.95, rf"Spearman $\rho = {rho:.2f}$ ($p = {sci(p)}$)", transform=ax.transAxes, va="top",
            fontsize=8, color=INK)
    ax.set_xlabel("closed-book accuracy (no retrieval)")
    ax.set_ylabel(rf"retrieval saving ($\alpha = {A:.2f}$)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.02, 0.58), scatterpoints=1)
    save(f, "Figure11")


def fig12(R):
    res, inv = R["res"], R["inv"]
    sizes = C.SCALE_MODELS
    dss = ["triviaqa", "nq", "popqa"]

    def row(m, d, method):
        return res[(res.model == m) & (res.dataset == d) & (res.alpha == A) & (res.method == method)].iloc[0]

    f, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9.0, 2.7), layout="constrained")
    x, w = np.arange(len(sizes)), 0.26
    for i, d in enumerate(dss):
        acc = [inv[(inv.model == m) & (inv.dataset == d)].acc_closed_book.iloc[0] for m in sizes]
        prop = [row(m, d, "Proposed") for m in sizes]
        base = [row(m, d, "Always-RAG + CA").coverage for m in sizes]
        xs = x + (i - 1) * w
        a1.bar(xs, acc, w, color=PAL[i], label=DS[d])
        a2.bar(xs, [1 - r.retrieval for r in prop], w, color=PAL[i], label=DS[d])
        a3.bar(xs, [r.coverage for r in prop], w, color=PAL[i], label=DS[d])
        a3.scatter(xs, base, marker="_", s=120, color=INK, zorder=4,
                   label="Always-RAG + CA (coverage)" if i == 0 else None)
    for ax, title in [(a1, "(a) Closed-book accuracy"), (a2, "(b) Retrieval saving"), (a3, "(c) Coverage")]:
        ax.set_xticks(x, ["3B", "7B", "14B"])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Qwen2.5 model size")
    h, lab = a1.get_legend_handles_labels()
    h3, l3 = a3.get_legend_handles_labels()
    extra = [(hh, ll) for hh, ll in zip(h3, l3) if ll.startswith("Always")]
    f.legend(h + [e[0] for e in extra], lab + [e[1] for e in extra], loc="outside lower center", ncol=4,
             fontsize=7.5)
    save(f, "Figure12")


def fig13(R):
    sh = R["shift"]
    model = "Qwen2.5-7B-Instruct"
    s = sh[(sh.model == model) & (sh.m == 0)]
    ds = list(DS)
    M = s.pivot(index="source", columns="target", values="violation").reindex(index=ds, columns=ds)
    f, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.2), gridspec_kw={"width_ratios": [1.25, 1]},
                               layout="constrained")
    cmap = mpl.colors.LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#3987e5", "#0d366b"])
    im = a1.imshow(M.values, cmap=cmap, vmin=0, vmax=1)
    a1.set_xticks(range(len(ds)), [DS[d] for d in ds], rotation=45, ha="right")
    a1.set_yticks(range(len(ds)), [DS[d] for d in ds])
    a1.grid(False)
    a1.set_xlabel("target")
    a1.set_ylabel("source (calibrated on)")
    for i in range(len(ds)):
        for j in range(len(ds)):
            v = M.values[i, j]
            if np.isfinite(v):
                a1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else INK)
    f.colorbar(im, ax=a1, shrink=0.85, label="violation rate")
    a1.set_title(f"(a) No recalibration ({MOD[model]})", fontsize=9)
    for k, mo in enumerate(["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]):
        g = sh[sh.model == mo].groupby("m")[["violation", "coverage"]].mean().reset_index()
        ls = "-" if k == 0 else "--"
        a2.plot(g.m, g.violation, color=PAL[0], marker="o", lw=1.8, ls=ls, label=f"violation, {MOD[mo]}")
        a2.plot(g.m, g.coverage, color=PAL[1], marker="s", lw=1.8, ls=ls, label=f"coverage, {MOD[mo]}")
    a2.axhline(C.DELTA, color=INK, lw=1, ls=":")
    a2.set_xlabel(r"labelled target questions $m$ (0 = none)")
    a2.set_ylabel("rate")
    a2.set_ylim(0, 1)
    a2.legend(fontsize=7)
    a2.set_title("(b) Recalibrating thresholds only", fontsize=9)
    save(f, "Figure13")


FIGURES = {2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8, 9: fig9, 10: fig10, 11: fig11,
           12: fig12, 13: fig13}


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {
        "res": pd.read_csv(f"{T}/results_all.csv"),
        "inv": pd.read_csv(f"{T}/inventory.csv"),
        "auc": pd.read_csv(f"{T}/auroc.csv"),
        "act": pd.read_csv(f"{T}/action_shares.csv"),
        "sens": pd.read_csv(f"{T}/sensitivity.csv"),
        "shift": pd.read_csv(f"{T}/shift.csv"),
    }
    todo = [int(a) for a in sys.argv[1:]] or list(FIGURES)
    for k in todo:
        FIGURES[k](R)
        print(f"Figure {k}")


if __name__ == "__main__":
    main()
