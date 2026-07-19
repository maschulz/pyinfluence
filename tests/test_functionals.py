"""Tests for pyinfluence.functionals: the domain-neutral Functional builders
(mean, group_gap, cohens_d, worst_group_mean).

Coverage:
  (a) each builder's value matches a manual numpy computation;
  (b) each builder's analytic grad matches central finite differences of its
      fn (group_gap with and without keep, cohens_d, worst_group_mean, mean);
  (c) group_gap with keep and y=None raises ValueError at evaluation time;
  (d) closed-form (FunctionalInfluence) vs RefitFunctionalInfluence agreement
      for functionals not already validated elsewhere: group_gap(..., of=
      "losses") and mean("losses") on a small logistic problem. worst_group_
      mean, cohens_d, and score-valued group_gap (the dp/eopp/fpr gaps) are
      already validated against refit in test_fairness.py and
      test_metric_influence.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pyinfluence import FunctionalInfluence, RefitFunctionalInfluence
from pyinfluence.functionals import cohens_d, group_gap, mean, worst_group_mean


def _central_fd(fn, v, y=None, eps=1e-6):
    """Central finite-difference gradient of fn(v, y) w.r.t. v, independent
    of the engine's own _fd_grad implementation."""
    v = np.asarray(v, dtype=float).ravel()
    grad = np.empty_like(v)
    for i in range(v.size):
        h = eps * max(1.0, abs(v[i]))
        vp = v.copy()
        vp[i] += h
        vm = v.copy()
        vm[i] -= h
        grad[i] = (fn(vp, y) - fn(vm, y)) / (2 * h)
    return grad


def _manual_cohens_d(v, mask):
    s1, s0 = v[mask], v[~mask]
    n1, n0 = s1.size, s0.size
    pooled = ((n1 - 1) * s1.var(ddof=1) + (n0 - 1) * s0.var(ddof=1)) / (n1 + n0 - 2)
    return (s1.mean() - s0.mean()) / np.sqrt(pooled)


# -----------------------------------------------------------------------------
# (a) value matches manual numpy computation
# -----------------------------------------------------------------------------


def test_mean_value_matches_manual():
    rng = np.random.default_rng(0)
    v = rng.normal(size=30)
    F = mean(of="scores")
    assert F(v) == pytest.approx(np.mean(v))


def test_group_gap_value_matches_manual_no_keep():
    rng = np.random.default_rng(1)
    m = 40
    v = rng.normal(size=m)
    groups = (rng.uniform(size=m) < 0.4).astype(int)
    F = group_gap(groups)
    mask = groups == 1
    expected = v[mask].mean() - v[~mask].mean()
    assert F(v) == pytest.approx(expected)


def test_group_gap_value_matches_manual_with_keep():
    rng = np.random.default_rng(2)
    m = 60
    v = rng.normal(size=m)
    y = (rng.uniform(size=m) < 0.5).astype(int)
    groups = (rng.uniform(size=m) < 0.4).astype(int)
    F = group_gap(groups, keep=lambda yy: yy == 1)
    k = y == 1
    vv, gk = v[k], groups[k]
    mask = gk == 1
    expected = vv[mask].mean() - vv[~mask].mean()
    assert F(v, y) == pytest.approx(expected)


def test_cohens_d_value_matches_manual():
    rng = np.random.default_rng(3)
    m = 50
    v = rng.normal(size=m)
    groups = (rng.uniform(size=m) < 0.5).astype(int)
    F = cohens_d(groups)
    expected = _manual_cohens_d(v, groups == 1)
    assert F(v) == pytest.approx(expected)


def test_worst_group_mean_value_matches_manual():
    rng = np.random.default_rng(4)
    m = 60
    v = rng.uniform(size=m)
    groups = rng.integers(0, 3, size=m)
    F = worst_group_mean(groups, of="losses")
    expected = max(v[groups == g].mean() for g in np.unique(groups))
    assert F(v) == pytest.approx(expected)


# -----------------------------------------------------------------------------
# (b) analytic grad matches central finite differences of fn
# -----------------------------------------------------------------------------


def test_mean_grad_matches_fd():
    rng = np.random.default_rng(10)
    v = rng.normal(size=25)
    F = mean()
    np.testing.assert_allclose(F.grad(v), _central_fd(F.fn, v), rtol=1e-5)


def test_group_gap_grad_matches_fd_no_keep():
    rng = np.random.default_rng(11)
    m = 35
    v = rng.normal(size=m)
    groups = (rng.uniform(size=m) < 0.45).astype(int)
    F = group_gap(groups)
    np.testing.assert_allclose(F.grad(v), _central_fd(F.fn, v), rtol=1e-5)


def test_group_gap_grad_matches_fd_with_keep():
    rng = np.random.default_rng(12)
    m = 50
    v = rng.normal(size=m)
    y = (rng.uniform(size=m) < 0.5).astype(int)
    groups = (rng.uniform(size=m) < 0.45).astype(int)
    F = group_gap(groups, keep=lambda yy: yy == 1)
    analytic = F.grad(v, y)
    fd = _central_fd(F.fn, v, y)
    np.testing.assert_allclose(analytic, fd, rtol=1e-5)


def test_cohens_d_grad_matches_fd():
    rng = np.random.default_rng(13)
    m = 40
    v = rng.normal(size=m)
    groups = (rng.uniform(size=m) < 0.5).astype(int)
    F = cohens_d(groups)
    np.testing.assert_allclose(F.grad(v), _central_fd(F.fn, v), rtol=1e-5)


def test_worst_group_mean_grad_matches_fd():
    rng = np.random.default_rng(14)
    m = 45
    v = rng.normal(size=m)
    groups = rng.integers(0, 3, size=m)
    F = worst_group_mean(groups, of="losses")
    np.testing.assert_allclose(F.grad(v), _central_fd(F.fn, v), rtol=1e-5)


# -----------------------------------------------------------------------------
# (c) group_gap with keep and y=None raises ValueError at evaluation time
# -----------------------------------------------------------------------------


def test_group_gap_with_keep_and_no_y_raises():
    rng = np.random.default_rng(15)
    m = 20
    v = rng.normal(size=m)
    groups = (rng.uniform(size=m) < 0.5).astype(int)
    F = group_gap(groups, keep=lambda yy: yy == 1)
    with pytest.raises(ValueError, match="pass y_ref"):
        F(v)
    with pytest.raises(ValueError, match="pass y_ref"):
        F.grad(v)


# -----------------------------------------------------------------------------
# (d) closed-form vs RefitFunctionalInfluence agreement, functionals not
# already validated elsewhere (worst_group_mean, cohens_d, and the score-
# valued dp/eopp/fpr group_gap forms are covered in test_fairness.py /
# test_metric_influence.py)
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_logistic_problem():
    rng = np.random.default_rng(0)
    n, p, m = 100, 4, 60
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = (X @ beta + 0.3 * rng.normal(size=n) > 0).astype(int)
    Xa = rng.normal(size=(m, p))
    ya = (Xa @ beta + 0.3 * rng.normal(size=m) > 0).astype(int)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    return model, X, y, Xa, ya


def test_group_gap_losses_closed_form_matches_refit(small_logistic_problem):
    model, X, y, Xa, ya = small_logistic_problem
    rng = np.random.default_rng(5)
    groups_audit = (rng.uniform(size=len(ya)) < 0.5).astype(int)
    F = group_gap(groups_audit, of="losses")

    cf = FunctionalInfluence(F).fit(model, X, y)
    s_cf = cf.explain(Xa, ya)
    rf = RefitFunctionalInfluence(F, verbose=0).fit(model, X, y)
    s_rf = rf.explain(Xa, ya)

    ok = ~np.isnan(s_rf)
    r = np.corrcoef(s_cf[ok], s_rf[ok])[0, 1]
    assert r > 0.93, f"pearson {r:.3f} too low"


def test_mean_losses_closed_form_matches_refit(small_logistic_problem):
    model, X, y, Xa, ya = small_logistic_problem
    F = mean(of="losses")

    cf = FunctionalInfluence(F).fit(model, X, y)
    s_cf = cf.explain(Xa, ya)
    rf = RefitFunctionalInfluence(F, verbose=0).fit(model, X, y)
    s_rf = rf.explain(Xa, ya)

    ok = ~np.isnan(s_rf)
    r = np.corrcoef(s_cf[ok], s_rf[ok])[0, 1]
    assert r > 0.93, f"pearson {r:.3f} too low"
