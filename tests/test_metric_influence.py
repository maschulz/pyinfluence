"""Tests for callable-metric fairness attribution and cohens_d.

The closed form differentiates a callable metric by finite differences (or
an analytic metric_grad) and chains through the GLM score gradients; the
refit estimator provides exact ground truth for any metric, so every new
metric can be validated the same way the built-ins are.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from pyinfluence.fairness import (
    FairnessInfluenceFunctions,
    RefitFairnessInfluence,
    SubsampledFairnessInfluence,
    cohens_d,
    disparity_removal_curve,
    disparity_value,
)


@pytest.fixture(scope="module")
def clf_problem():
    rng = np.random.default_rng(0)
    n, p, m = 120, 5, 100
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = (X @ beta + 0.5 * rng.normal(size=n) > 0).astype(int)
    Xa = rng.normal(size=(m, p))
    ya = (Xa @ beta + 0.5 * rng.normal(size=m) > 0).astype(int)
    a = (rng.uniform(size=m) < 0.4).astype(int)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    return model, X, y, Xa, ya, a


def _manual_cohens_d(scores, mask):
    s1, s0 = scores[mask], scores[~mask]
    n1, n0 = len(s1), len(s0)
    pooled = ((n1 - 1) * s1.var(ddof=1) + (n0 - 1) * s0.var(ddof=1)) / (
        n1 + n0 - 2
    )
    return (s1.mean() - s0.mean()) / np.sqrt(pooled)


# -----------------------------------------------------------------------------
# cohens_d itself
# -----------------------------------------------------------------------------


def test_cohens_d_matches_manual(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    p1 = model.predict_proba(Xa)[:, 1]
    expected = _manual_cohens_d(p1, a == 1)
    assert np.isclose(cohens_d(p1, a), expected)
    # disparity_value routes callables through the same model scores
    assert np.isclose(
        disparity_value(model, Xa, a, ya, metric=cohens_d), expected
    )


def test_cohens_d_validation():
    with pytest.raises(ValueError, match="two audit samples"):
        cohens_d(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 1]))
    with pytest.raises(ValueError, match="pooled variance"):
        cohens_d(np.ones(10), np.array([0] * 5 + [1] * 5))


# -----------------------------------------------------------------------------
# Closed form vs refit ground truth
# -----------------------------------------------------------------------------


def test_closed_form_matches_refit_cohens_d(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    cf = FairnessInfluenceFunctions(metric=cohens_d).fit(model, X, y)
    s_cf = cf.explain(Xa, y_audit=ya, sensitive_audit=a)
    rf = RefitFairnessInfluence(metric=cohens_d, verbose=0).fit(model, X, y)
    s_rf = rf.explain(Xa, y_audit=ya, sensitive_audit=a)
    r = np.corrcoef(s_cf, s_rf)[0, 1]
    slope = np.polyfit(s_cf, s_rf, 1)[0]
    assert r > 0.95
    assert 0.7 < slope < 1.4


def test_closed_form_matches_refit_cohens_d_regression():
    rng = np.random.default_rng(1)
    n, p, m = 100, 4, 80
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + 0.3 * rng.normal(size=n)
    Xa = rng.normal(size=(m, p))
    a = (rng.uniform(size=m) < 0.5).astype(int)
    model = Ridge(alpha=1.0).fit(X, y)
    cf = FairnessInfluenceFunctions(metric=cohens_d).fit(model, X, y)
    s_cf = cf.explain(Xa, sensitive_audit=a)
    rf = RefitFairnessInfluence(metric=cohens_d, verbose=0).fit(model, X, y)
    s_rf = rf.explain(Xa, sensitive_audit=a)
    assert np.corrcoef(s_cf, s_rf)[0, 1] > 0.95


# -----------------------------------------------------------------------------
# Analytic metric_grad path
# -----------------------------------------------------------------------------


def _cohens_d_grad(scores, sensitive, y=None):
    """Analytic gradient of cohens_d w.r.t. the score vector."""
    scores = np.asarray(scores, dtype=float).ravel()
    svals = np.unique(np.asarray(sensitive).ravel())
    mask = np.asarray(sensitive).ravel() == svals[1]
    s1, s0 = scores[mask], scores[~mask]
    n1, n0 = s1.size, s0.size
    m1, m0 = s1.mean(), s0.mean()
    sp2 = ((n1 - 1) * s1.var(ddof=1) + (n0 - 1) * s0.var(ddof=1)) / (
        n1 + n0 - 2
    )
    sp = np.sqrt(sp2)
    d = (m1 - m0) / sp
    grad = np.empty_like(scores)
    grad[mask] = (1.0 / n1) / sp - d * (s1 - m1) / ((n1 + n0 - 2) * sp2)
    grad[~mask] = (-1.0 / n0) / sp - d * (s0 - m0) / ((n1 + n0 - 2) * sp2)
    return grad


def test_metric_grad_matches_fd(clf_problem):
    """FD gradient and analytic gradient must yield the same scores."""
    model, X, y, Xa, ya, a = clf_problem
    fd = FairnessInfluenceFunctions(metric=cohens_d).fit(model, X, y)
    an = FairnessInfluenceFunctions(
        metric=cohens_d, metric_grad=_cohens_d_grad
    ).fit(model, X, y)
    s_fd = fd.explain(Xa, sensitive_audit=a)
    s_an = an.explain(Xa, sensitive_audit=a)
    np.testing.assert_allclose(s_fd, s_an, rtol=1e-4, atol=1e-12)


def test_metric_grad_bad_shape_raises(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    bad = FairnessInfluenceFunctions(
        metric=cohens_d, metric_grad=lambda s, a_, y_: np.ones(3)
    ).fit(model, X, y)
    with pytest.raises(ValueError, match="one gradient entry per"):
        bad.explain(Xa, sensitive_audit=a)


# -----------------------------------------------------------------------------
# Target handling, other estimators, repair curve
# -----------------------------------------------------------------------------


def test_absolute_target_scales_by_sign(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    val = disparity_value(model, Xa, a, metric=cohens_d)
    s_signed = (
        FairnessInfluenceFunctions(metric=cohens_d, target="signed")
        .fit(model, X, y)
        .explain(Xa, sensitive_audit=a)
    )
    s_abs = (
        FairnessInfluenceFunctions(metric=cohens_d, target="absolute")
        .fit(model, X, y)
        .explain(Xa, sensitive_audit=a)
    )
    np.testing.assert_allclose(s_abs, np.sign(val) * s_signed)


def test_subsampled_callable_metric(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    sub = SubsampledFairnessInfluence(
        metric=cohens_d, n_subsets=40, random_state=0, verbose=0
    ).fit(model, X, y)
    s = sub.explain(Xa, sensitive_audit=a)
    assert s.shape == (len(y),)
    assert np.isfinite(s).any()


def test_removal_curve_callable_metric(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    cf = FairnessInfluenceFunctions(metric=cohens_d, target="absolute").fit(
        model, X, y
    )
    scores = cf.explain(Xa, sensitive_audit=a)
    curve = disparity_removal_curve(
        scores, model, X, y, Xa, a, y_audit=ya, metric=cohens_d,
        fractions=np.linspace(0.0, 0.1, 3), n_random=2, random_state=0,
    )
    assert np.isfinite(curve["disparity"]).all()
    assert np.isclose(
        curve["base_disparity"],
        abs(disparity_value(model, Xa, a, ya, metric=cohens_d)),
    )


def test_nonsmooth_metric_via_refit(clf_problem):
    """Threshold metrics need no gradient through the refit estimator."""
    model, X, y, Xa, ya, a = clf_problem

    def hard_rate_gap(scores, sensitive, y=None):
        dec = (np.asarray(scores) >= 0.5).astype(float)
        m = np.asarray(sensitive) == 1
        return float(dec[m].mean() - dec[~m].mean())

    rf = RefitFairnessInfluence(metric=hard_rate_gap, verbose=0).fit(model, X, y)
    s = rf.explain(Xa, sensitive_audit=a)
    assert s.shape == (len(y),)
    assert np.isfinite(s).all()
