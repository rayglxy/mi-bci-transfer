"""Run:  pytest -q   (no dataset needed)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mibci.align import covariances, recenter, reference
from mibci.models import tangent_lr, csp_lda
from mibci import experiments as E


def toy(n=120, shift=None, seed=0):
    """Two classes that differ in which of two sources is stronger, mixed into 6 channels."""
    rng = np.random.default_rng(seed)
    A = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    if shift is not None:
        A = shift @ A
    y = np.arange(n) % 2
    S = rng.normal(size=(n, 6, 200))
    S[y == 0, 0] *= 3.0; S[y == 1, 1] *= 3.0
    return covariances(np.einsum('ij,njt->nit', A, S)), y


def test_recentred_mean_is_identity():
    C, _ = toy()
    assert np.allclose(reference(recenter(C)), np.eye(6), atol=1e-6)


def test_recentering_preserves_class_structure():
    C, y = toy()
    acc = (tangent_lr().fit(recenter(C[:80]), y[:80]).predict(recenter(C[80:], reference(C[:80]))) == y[80:]).mean()
    assert acc > 0.9


def test_both_models_learn_the_toy_problem():
    C, y = toy()
    for make in (tangent_lr, csp_lda):
        assert (make().fit(C[:80], y[:80]).predict(C[80:]) == y[80:]).mean() > 0.9


def test_recentering_undoes_a_rescaling_of_the_channels():
    """A second 'day' with different channel gains: the unaligned decoder drops,
    the re-centred one does not."""
    train = toy(seed=1)
    gains = np.diag([4.0, 0.3, 2.0, 0.5, 3.0, 0.4])
    test = toy(seed=2, shift=gains)
    raw = E.cross_session(tangent_lr, train, test, align=False)
    aligned = E.cross_session(tangent_lr, train, test, align=True)
    assert aligned > 0.9 and aligned > raw + 0.1


def test_within_session_uses_every_trial_once():
    C, y = toy()
    run = np.arange(len(y)) // 20
    assert 0.9 < E.within_session(tangent_lr, C, y, run) <= 1.0
