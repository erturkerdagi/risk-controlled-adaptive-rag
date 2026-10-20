"""Scorers s0 (closed-book answer) and s1 (RAG answer), and the compared methods.

Both scorers are L2-regularised logistic regressions on standardised features.
The hidden-state block (PCA-64 of one layer) is kept only if it improves
cross-validated AUROC on the training half.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import U0, U1

# Output name -> (feature blocks, target: 0 = closed-book answer, 1 = RAG answer)
SPECS = {
    "s0": (["u0", "h0"], 0),
    "s1": (["u1", "phi", "h1"], 1),
    "s0_nohidden": (["u0"], 0),
    "s1_nohidden": (["u1", "phi"], 1),
    "s0_probe": (["h0"], 0),
    "s0_xrouter": (["nll0", "phi"], 0),
}


def _classifier():
    return make_pipeline(StandardScaler(),
                         LogisticRegressionCV(Cs=[0.01, 0.1, 1.0], cv=3, scoring="roc_auc", max_iter=3000))


def safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < np.mean(y) < 1 else np.nan


class Scorer:
    def __init__(self, n_components=64, seed=2026):
        self.n_components = n_components
        self.seed = seed

    def fit(self, df, H0, H1, idx):
        y0 = df.y0.values[idx].astype(int)
        y1 = df.y1.values[idx].astype(int)
        self.l0 = self._pick_layer(H0[idx], y0)
        self.l1 = self._pick_layer(H1[idx], y1)
        self.p0 = self._fit_pca(H0[idx][:, self.l0])
        self.p1 = self._fit_pca(H1[idx][:, self.l1])
        X = self._blocks(df.iloc[idx], H0[idx], H1[idx])

        # Keep the hidden-state block only if it helps on the training half.
        self.choice = {}
        for key, base, hidden, y in [("s0", ["u0"], "h0", y0), ("s1", ["u1", "phi"], "h1", y1)]:
            if 0 < y.mean() < 1:
                cv = StratifiedKFold(3, shuffle=True, random_state=self.seed)
                auc_base = cross_val_score(_classifier(), np.hstack([X[c] for c in base]), y,
                                           cv=cv, scoring="roc_auc").mean()
                auc_hidden = cross_val_score(_classifier(), np.hstack([X[c] for c in base + [hidden]]), y,
                                             cv=cv, scoring="roc_auc").mean()
                self.choice[key] = base + [hidden] if auc_hidden > auc_base else base
            else:
                self.choice[key] = base + [hidden]

        self.models = {}
        for name, (cols, target) in SPECS.items():
            cols = self.choice.get(name, cols)
            y = y0 if target == 0 else y1
            if 0 < y.mean() < 1:
                self.models[name] = _classifier().fit(np.hstack([X[c] for c in cols]), y)
            else:  # constant label on the training half
                self.models[name] = float(y.mean())
        return self

    def scores(self, df, H0, H1):
        X = self._blocks(df, H0, H1)
        out = {}
        for name, (cols, _) in SPECS.items():
            cols = self.choice.get(name, cols)
            model = self.models[name]
            if isinstance(model, float):
                out[name] = np.full(len(df), model)
            else:
                out[name] = model.predict_proba(np.hstack([X[c] for c in cols]))[:, 1]
        return out

    def _blocks(self, df, H0, H1):
        phi = np.nan_to_num(np.array(df.phi.tolist(), dtype=float)[:, :6])
        return {
            "u0": df[U0].values,
            "u1": df[U1].values,
            "phi": phi,
            "nll0": df[["logp_mean_0"]].values,
            "h0": self._project(self.p0, H0[:, self.l0]),
            "h1": self._project(self.p1, H1[:, self.l1]),
        }

    def _fit_pca(self, X):
        sc = StandardScaler().fit(X.astype(np.float32))
        k = int(min(self.n_components, len(X) - 1, X.shape[1]))
        return sc, PCA(k, random_state=self.seed).fit(sc.transform(X.astype(np.float32)))

    @staticmethod
    def _project(fitted, X):
        sc, pca = fitted
        return pca.transform(sc.transform(X.astype(np.float32)))

    def _pick_layer(self, H, y):
        """Layer with the best 3-fold AUROC of a small probe; last layer if y is constant."""
        if not 0 < y.mean() < 1:
            return H.shape[1] - 1
        best, best_auc = 0, -1
        for li in range(H.shape[1]):
            Z = self._project(self._fit_pca(H[:, li]), H[:, li])
            probe = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.1], cv=3, max_iter=2000))
            cv = StratifiedKFold(3, shuffle=True, random_state=self.seed)
            auc = cross_val_score(probe, Z, y, cv=cv, scoring="roc_auc").mean()
            if auc > best_auc:
                best, best_auc = li, auc
        return best


def train_pool_split(n, train_frac=0.5, seed=2026):
    perm = np.random.default_rng(seed).permutation(n)
    k = int(train_frac * n)
    return perm[:k], perm[k:]


METHODS = [
    "Always-RAG + CA",
    "Never-retrieve + CA",
    "UE gate (mean entropy)",
    "UE gate (margin)",
    "Probe gate",
    "X-Router-style gate",
    "Proposed (no hidden states)",
    "Proposed",
    "Proposed, naive thresholds",
    "Proposed, Bonferroni",
]


def method_inputs(pool):
    """(s0, s1, calibration options) for every compared method.

    `pool` is a score table as written by `notebooks/02_analysis.ipynb`
    (see results/scores). All methods go through the same calibration.
    """
    e0, e1 = -pool.ent_mean_0.values, -pool.ent_mean_1.values
    zero = np.zeros(len(pool))
    return {
        "Always-RAG + CA": (zero, e1, dict(fix_l1=np.inf)),
        "Never-retrieve + CA": (e0, zero, dict(fix_l2=np.inf, retrieval_on_abstain=False)),
        "UE gate (mean entropy)": (e0, e1, {}),
        "UE gate (margin)": (pool.margin_first_0.values, e1, {}),
        "Probe gate": (pool.s0_probe.values, e1, {}),
        "X-Router-style gate": (pool.s0_xrouter.values, e1, {}),
        "Proposed (no hidden states)": (pool.s0_nohidden.values, pool.s1_nohidden.values, {}),
        "Proposed": (pool.s0.values, pool.s1.values, {}),
        "Proposed, naive thresholds": (pool.s0.values, pool.s1.values, dict(method="naive")),
        "Proposed, Bonferroni": (pool.s0.values, pool.s1.values, dict(method="bonferroni")),
    }
