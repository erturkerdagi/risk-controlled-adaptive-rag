# Risk-controlled adaptive retrieval for LLMs

Code and results for the paper

> E. Erdağı, *A Risk-Controlled Three-Way Method for Adaptive Retrieval-Augmented Generation in Large Language Models*.

For every question the method chooses one of three actions:

- **PARAM**: return the closed-book answer, if `s0(x) >= λ1`;
- **RAG**: otherwise retrieve and return the RAG answer, if `s1(x) >= λ2`;
- **ABSTAIN**: otherwise withhold the answer.

`s0` and `s1` are logistic scorers on output uncertainty, retrieval features and (optionally) hidden states.
The thresholds `(λ1, λ2)` are chosen on a labelled calibration set with Learn-then-Test, so that the selective
risk (error rate among answered questions) is at most α with probability at least 1 − δ. Among the certified
threshold pairs, the one with the best coverage − μ · retrieval rate is used.

## Repository layout

```
rarag/                 package
  calibration.py       policy, grid evaluation, LTT calibration (fixed-sequence, Bonferroni, naive)
  scorers.py           s0 / s1 scorers and the inputs of all compared methods
  labels.py            answer normalisation and the containment label
  data.py              readers for the generation cache and the score files
  shift.py             cross-dataset shift and threshold-only recalibration
  config.py            settings used in the paper
notebooks/
  01_retrieval_and_generation.ipynb   GPU step (Colab A100): retrieval, answers, features, hidden states
  02_analysis.ipynb                   scorers, evaluation, shift, tables and figures (CPU)
scripts/               evaluation, sensitivity, action shares, tables, figures, summary
results/scores/        held-out pools with labels and all scores (27 model-dataset pairs)
results/tables/        result tables; results/tables/paper/ holds Tables 3-7
figures/               Figures 2-13
tests/
```

## Reproducing the results

Everything after the generation step runs on a CPU from the files in `results/scores`:

```bash
pip install -r requirements.txt
python scripts/evaluate.py        # results/tables/results_all.csv (about 10 min)
python scripts/sensitivity.py     # results/tables/sensitivity.csv
python scripts/action_shares.py   # results/tables/action_shares.csv
python scripts/make_tables.py     # results/tables/paper/
python scripts/make_figures.py    # figures/
python scripts/summarize.py       # numbers quoted in the text, with Wilcoxon and Spearman tests
pytest
```

To start from scratch, run `notebooks/01_retrieval_and_generation.ipynb` on a Colab A100 runtime (about 3.8
GPU-hours for all 79,251 questions) and then `notebooks/02_analysis.ipynb`. Llama-3.1 and Mistral need accepted
licences on Hugging Face; the token is read from the Colab secret `HF_TOKEN`. Hidden states are not included in
this repository because of their size; they are only needed to retrain the scorers and for the shift experiment.

`results/tables/results_all.csv` is the file used in the paper. Rerunning `scripts/evaluate.py` gives the same
values for α ≤ 0.40. At α = 0.50 about 60 of the 270 rows can differ in the fourth decimal of coverage (and by at
most one split in the violation rate) depending on the SciPy version: binomial p-values at p = 0.5 contain many
near ties, and the order of tied hypotheses in fixed-sequence testing follows their last bits. None of the numbers
reported in the paper changes.

## Score files

`results/scores/{model}__{dataset}.csv.gz` contains the held-out half of each pair (1,500 questions; 1,209 for
MuSiQue), which is the pool split into calibration and test sets:

| column | meaning |
|---|---|
| `id` | FlashRAG question id |
| `y0`, `y1` | correctness of the closed-book and RAG answer (containment rule) |
| `ent_mean_0`, `ent_mean_1` | mean token entropy of the two answers |
| `margin_first_0` | first-token probability margin of the closed-book answer |
| `s0`, `s1` | proposed scorers |
| `s0_nohidden`, `s1_nohidden` | proposed scorers without hidden states |
| `s0_probe` | hidden-state probe (Probing-RAG style gate) |
| `s0_xrouter` | closed-book NLL + retrieval features (X-Router style gate) |

## Setup used in the paper

| | |
|---|---|
| Models | Qwen2.5-3B/7B/14B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 |
| Datasets | TriviaQA, NQ, PopQA, SQuAD, HotpotQA, 2WikiMultihopQA, MuSiQue (FlashRAG versions, up to 3,000 questions each) |
| Retrieval | BM25 top-50 on Wikipedia DPR 100-word passages (Pyserini), re-ranked with e5-base-v2, top-5 in the prompt |
| Generation | greedy, at most 32 new tokens, bf16 |
| Calibration | α ∈ {0.10, 0.15, 0.20, 0.30, 0.40, 0.50}, δ = 0.10, μ = 0.30, 30 × 30 threshold grid, ordering fraction 0.3 |
| Evaluation | 50% of each pair trains the scorers; the other half is split 300 times into equal calibration and test sets |

## Citation

See `CITATION.cff`. The entry will be updated with the publication details.

## License

MIT. Models and datasets are subject to their own licences.
