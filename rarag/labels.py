"""Answer normalisation and the word-boundary containment rule."""
import re
import string
from collections import Counter

_PUNCT = re.compile("[" + re.escape(string.punctuation) + "]")


def normalize(text):
    """Lower-case, replace punctuation with spaces, drop articles, squeeze spaces."""
    s = _PUNCT.sub(" ", str(text).lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_answer(pred, golds):
    """1.0 if a normalised gold answer occurs in the prediction as a whole word sequence.

    The word boundary keeps "Lille" from matching "Lillehammer".
    """
    p = f" {normalize(pred)} "
    return float(any(g and f" {g} " in p for g in (normalize(x) for x in golds)))


def score_answer(pred, golds):
    """Containment accuracy (the label used in the paper), exact match and token F1."""
    p = normalize(pred)
    gs = [normalize(g) for g in golds if str(g).strip()]
    acc = float(any(g and f" {g} " in f" {p} " for g in gs))
    em = float(any(p == g for g in gs))
    f1 = 0.0
    for g in gs:
        pt, gt = p.split(), g.split()
        common = sum((Counter(pt) & Counter(gt)).values())
        if common:
            prec, rec = common / len(pt), common / len(gt)
            f1 = max(f1, 2 * prec * rec / (prec + rec))
    return acc, em, f1
