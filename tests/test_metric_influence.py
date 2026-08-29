"""Tests for the cohens_d Functional builder and custom-Functional-metric
fairness attribution.

``pyinfluence.fairness.disparity`` accepts only the named metric strings; a
raw callable now raises. Custom metrics (Cohen's d, or any other statistic)
are built directly with :mod:`pyinfluence.functionals` or a hand-built
:class:`~pyinfluence.Functional` and passed straight to the engine
estimators, or through ``metric=`` on the ``pyinfluence.fairness`` value/
repair utilities (which accept a metric name or a Functional).

The closed form differentiates a functional's ``fn`` by finite differences
(or an analytic ``grad``) and chains through the GLM score gradients; the
refit estimator provides exact ground truth for any functional, so every new
metric can be validated the same way the built-ins are.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from pyinfluence import (
    Functional,
    FunctionalInfluence,
    RefitFunctionalInfluence,
    SubsampledFunctionalInfluence,
    functional_value,
)
from pyinfluence.fairness import disparity, disparity_removal_curve, disparity_value
from pyinfluence.functionals import cohens_d, mean


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
    pooled = ((n1 - 1) * s1.var(ddof=1) + (n0 - 1) * s0.var(ddof=1)) / (n1 + n0 - 2)
    return (s1.mean() - s0.mean()) / np.sqrt(pooled)


# -----------------------------------------------------------------------------
# cohens_d itself (builder form: cohens_d(groups) -> Functional, evaluated on
# a score vector via F(scores))
# -----------------------------------------------------------------------------


def test_cohens_d_matches_manual(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    p1 = model.predict_proba(Xa)[:, 1]
    expected = _manual_cohens_d(p1, a == 1)
    F = cohens_d(a)
    assert np.isclose(F(p1), expected)
    # disparity_value routes a Functional metric through the same model scores
    assert np.isclose(disparity_value(model, Xa, a, ya, metric=F), expected)


def test_cohens_d_validation():
    # errors surface when the functional is evaluated, not when it is built
    F_too_few = cohens_d(np.array([0, 0, 1]))
    with pytest.raises(ValueError, match="at least two reference samples per group"):
        F_too_few(np.array([1.0, 2.0, 3.0]))

    F_zero_var = cohens_d(np.array([0] * 5 + [1] * 5))
    with pytest.raises(ValueError, match="pooled variance"):
        F_zero_var(np.ones(10))


# -----------------------------------------------------------------------------
# Closed form vs refit ground truth
# -----------------------------------------------------------------------------


def test_closed_form_matches_refit_cohens_d(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    F = cohens_d(a)
    cf = FunctionalInfluence(F).fit(model, X, y)
    s_cf = cf.explain(Xa, ya)
    rf = RefitFunctionalInfluence(F, verbose=0).fit(model, X, y)
    s_rf = rf.explain(Xa, ya)
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
    F = cohens_d(a)
    cf = FunctionalInfluence(F).fit(model, X, y)
    s_cf = cf.explain(Xa)
    rf = RefitFunctionalInfluence(F, verbose=0).fit(model, X, y)
    s_rf = rf.explain(Xa)
    assert np.corrcoef(s_cf, s_rf)[0, 1] > 0.95


# -----------------------------------------------------------------------------
# Analytic gradient path (Functional(fn=..., grad=...) vs finite differences)
# -----------------------------------------------------------------------------


def test_functional_grad_matches_fd(clf_problem):
    """FD gradient (forced by stripping the builder's analytic grad) and the
    builder's own analytic gradient must yield the same attribution scores."""
    model, X, y, Xa, ya, a = clf_problem
    F_builder = cohens_d(a)
    F_fd = Functional(fn=F_builder.fn, of="scores")
    fd = FunctionalInfluence(F_fd).fit(model, X, y)
    an = FunctionalInfluence(F_builder).fit(model, X, y)
    s_fd = fd.explain(Xa)
    s_an = an.explain(Xa)
    np.testing.assert_allclose(s_fd, s_an, rtol=1e-4, atol=1e-12)


def test_functional_grad_bad_shape_raises(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    F_builder = cohens_d(a)
    F_bad = Functional(
        fn=F_builder.fn,
        grad=lambda v, yy: np.ones(3),
        of="scores",
    )
    bad = FunctionalInfluence(F_bad).fit(model, X, y)
    with pytest.raises(ValueError, match="one gradient entry per"):
        bad.explain(Xa)


# -----------------------------------------------------------------------------
# Target handling, other estimators, repair curve
# -----------------------------------------------------------------------------


def test_absolute_target_scales_by_sign(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    F = cohens_d(a)
    val = disparity_value(model, Xa, a, metric=F)
    s_signed = FunctionalInfluence(F, target="signed").fit(model, X, y).explain(Xa)
    s_abs = FunctionalInfluence(F, target="absolute").fit(model, X, y).explain(Xa)
    # Away from F=0, |F| is smooth and its removal effect is sign(F) times the
    # signed effect. absolute now routes through perturbation evaluation (exact
    # for the value change) while signed uses the chain rule, so the two agree
    # to first order, not identically.
    expected = np.sign(val) * s_signed
    assert np.corrcoef(s_abs, expected)[0, 1] > 0.99


def test_absolute_near_parity_signs_match_refit():
    """Near F=0, |F| has a kink; a sign(F)*grad linearization gets the sign
    wrong for removals that cross zero. Perturbation evaluation handles the
    crossing, so near parity the closed-form absolute attribution must agree in
    sign with exact refit."""
    rng = np.random.default_rng(26)
    n = 50
    X = rng.normal(size=(n, 4))
    beta = rng.normal(size=4)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ beta))).astype(int)
    a = rng.integers(0, 2, size=n)
    model = LogisticRegression(max_iter=2000).fit(X, y)

    F = cohens_d(a)  # near parity: |F| ~ 5e-3 here
    approx = FunctionalInfluence(F, target="absolute").fit(model, X, y).explain(X)
    exact = RefitFunctionalInfluence(F, target="absolute").fit(model, X, y).explain(X)
    mask = np.abs(exact) > 1e-6
    sign_agree = np.mean(np.sign(approx[mask]) == np.sign(exact[mask]))
    assert sign_agree > 0.9, f"near-parity sign agreement {sign_agree:.2f}"


def test_subsampled_functional_metric(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    F = cohens_d(a)
    sub = SubsampledFunctionalInfluence(F, n_subsets=40, random_state=0, verbose=0).fit(
        model, X, y
    )
    s = sub.explain(Xa)
    assert s.shape == (len(y),)
    assert np.isfinite(s).any()


def test_removal_curve_functional_metric(clf_problem):
    model, X, y, Xa, ya, a = clf_problem
    F = cohens_d(a)
    cf = FunctionalInfluence(F, target="absolute").fit(model, X, y)
    scores = cf.explain(Xa)
    curve = disparity_removal_curve(
        scores,
        model,
        X,
        y,
        Xa,
        a,
        y_audit=ya,
        metric=F,
        fractions=np.linspace(0.0, 0.1, 3),
        n_random=2,
        random_state=0,
    )
    assert np.isfinite(curve["disparity"]).all()
    assert np.isclose(
        curve["base_disparity"],
        abs(disparity_value(model, Xa, a, ya, metric=F)),
    )


def test_nonsmooth_metric_via_refit(clf_problem):
    """Threshold metrics need no gradient through the refit estimator."""
    model, X, y, Xa, ya, a = clf_problem

    def hard_rate_gap(v, y=None):
        dec = (np.asarray(v) >= 0.5).astype(float)
        m = np.asarray(a) == 1
        return float(dec[m].mean() - dec[~m].mean())

    F = Functional(fn=hard_rate_gap, of="scores", name="hard_rate_gap")
    rf = RefitFunctionalInfluence(F, verbose=0).fit(model, X, y)
    s = rf.explain(Xa)
    assert s.shape == (len(y),)
    assert np.isfinite(s).all()


# -----------------------------------------------------------------------------
# disparity()/disparity_value() reject raw callables
# -----------------------------------------------------------------------------


def test_disparity_rejects_callable_metric(clf_problem):
    model, X, y, Xa, ya, a = clf_problem

    def raw_callable(v, y=None):
        return float(np.mean(v))

    with pytest.raises(ValueError, match="Unknown metric"):
        disparity(raw_callable, a)


def test_disparity_value_rejects_raw_callable(clf_problem):
    model, X, y, Xa, ya, a = clf_problem

    def raw_callable(v, y=None):
        return float(np.mean(v))

    with pytest.raises(TypeError, match="Functional"):
        disparity_value(model, Xa, a, metric=raw_callable)


def test_functional_apis_reject_ref_length_mismatch():
    """The functional entry points normalize their own (X_ref, y_ref), so they
    must apply the same alignment check as ordinary explain(): a mismatched or
    scalar y_ref would otherwise broadcast into a finite but wrong result."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)
    from sklearn.linear_model import Ridge

    model = Ridge().fit(X, y)
    F = mean(of="scores")
    X_ref = rng.normal(size=(2, 4))  # two reference rows
    y_ref = np.array([1.0])  # one label: mismatch

    with pytest.raises(ValueError, match="reference row"):
        functional_value(model, X_ref, F, y_ref)
    with pytest.raises(ValueError, match="reference row"):
        FunctionalInfluence(F).fit(model, X, y).explain(X_ref, y_ref)
    with pytest.raises(ValueError, match="reference row"):
        RefitFunctionalInfluence(F).fit(model, X, y).explain(X_ref, y_ref)
    with pytest.raises(ValueError, match="reference row"):
        SubsampledFunctionalInfluence(F, n_subsets=5, random_state=0, verbose=0).fit(
            model, X, y
        ).explain(X_ref, y_ref)
