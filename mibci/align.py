"""Covariances and re-centering.

A trial is summarised by its channel covariance matrix. Between days and
between people those matrices shift as a whole (electrode placement, impedance,
anatomy). Re-centering moves each recording's matrices so that their average
becomes the identity. It needs no labels, only the trials themselves.
"""
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann


def covariances(X):
    """Trials x channels x samples -> trials x channels x channels (OAS shrinkage)."""
    return Covariances(estimator='oas').transform(X)


def reference(C):
    """The Riemannian mean of a set of covariance matrices."""
    return mean_riemann(C)


def recenter(C, ref=None):
    """Whiten every matrix by a reference (default: the mean of C itself)."""
    W = invsqrtm(reference(C) if ref is None else ref)
    return W @ C @ W
