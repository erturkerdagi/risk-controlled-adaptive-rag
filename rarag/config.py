"""Experiment settings used for every number in the paper."""

MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3",
          "Qwen2.5-3B-Instruct", "Qwen2.5-14B-Instruct"]
MAIN_MODELS = MODELS[:3]
SCALE_MODELS = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"]
DATASETS = ["triviaqa", "nq", "popqa", "squad", "hotpotqa", "2wikimultihopqa", "musique"]

ALPHAS = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
ALPHA_MAIN = 0.30
DELTA = 0.10          # P(risk > alpha) <= delta
MU = 0.30             # weight of the retrieval rate in the objective
TRAIN_FRAC = 0.5      # share of questions used to train the scorers
CAL_FRAC = 0.5        # share of the remaining pool used for calibration
N_TRIALS = 300        # random calibration/test splits
GRID = 30             # thresholds per axis
SEED = 2026
