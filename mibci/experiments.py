"""The three evaluation settings, each with and without re-centering."""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from .align import recenter, reference


def within_session(make, C, y, run):
    """Leave one run out, same person, same day."""
    hits = 0
    for tr, te in LeaveOneGroupOut().split(C, y, run):
        hits += (make().fit(C[tr], y[tr]).predict(C[te]) == y[te]).sum()
    return hits / len(y)


def cross_session(make, train, test, align):
    """Train on day one, test on day two of the same person."""
    (Ctr, ytr), (Cte, yte) = train, test
    if align:
        Ctr, Cte = recenter(Ctr), recenter(Cte)
    return float((make().fit(Ctr, ytr).predict(Cte) == yte).mean())


def cross_subject(make, others, test, align):
    """Train on eight other people, test on someone the decoder has never seen."""
    Cte, yte = test
    if align:
        others = [(recenter(C), y) for C, y in others]
        Cte = recenter(Cte)
    Ctr = np.concatenate([C for C, _ in others]); ytr = np.concatenate([y for _, y in others])
    return float((make().fit(Ctr, ytr).predict(Cte) == yte).mean())


def calibration_curve(make, train, test, counts):
    """How many unlabelled trials of the new day does re-centering need?

    The reference is estimated from the first n trials of day two; accuracy is
    always measured on the second half of day two, so every n is scored on the
    same trials.
    """
    (Ctr, ytr), (Cte, yte) = train, test
    half = len(yte) // 2
    model = make().fit(recenter(Ctr), ytr)
    out = {}
    for n in counts:
        ref = reference(Cte[:n]) if n > 0 else np.eye(Cte.shape[1])
        if n == 0:                                             # no alignment at all
            out[0] = float((make().fit(Ctr, ytr).predict(Cte[half:]) == yte[half:]).mean())
        else:
            out[n] = float((model.predict(recenter(Cte[half:], ref)) == yte[half:]).mean())
    return out
