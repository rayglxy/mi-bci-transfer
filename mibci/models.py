"""Two standard motor-imagery decoders, both taking covariance matrices."""
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import CSP


def tangent_lr():
    """Map covariances to a flat space around their mean, then logistic regression."""
    return make_pipeline(TangentSpace(metric='riemann'), LogisticRegression(C=1.0, max_iter=1000))


def csp_lda(n_filters=8):
    """Common spatial patterns (log-variance of 8 spatial filters), then LDA."""
    return make_pipeline(CSP(nfilter=n_filters, log=True), LinearDiscriminantAnalysis())


MODELS = {'Tangent space + logistic regression': tangent_lr, 'CSP + LDA': csp_lda}
