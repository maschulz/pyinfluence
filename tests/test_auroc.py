"""Tests for the AUROC functional (exact and smoothed variants)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pyinfluence import (
    FunctionalInfluence,
    RefitFunctionalInfluence,
    functional_value,
)
from pyinfluence.functionals import auroc


@pytest.fixture(scope="module")
def clf_problem():
    rng = np.random.default_rng(0)
    n, p, m = 120, 5, 150
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = (X @ beta + 0.6 * rng.normal(size=n) > 0).astype(int)
    Xa = rng.normal(size=(m, p))
    ya = (Xa @ beta + 0.6 * rng.normal(size=m) > 0).astype(int)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    return model, X, y, Xa, ya


def test_exact_matches_sklearn(clf_problem):
    model, X, y, Xa, ya = clf_problem
    proba = model.predict_proba(Xa)[:, 1]
    assert np.isclose(auroc(1)(proba, ya), roc_auc_score(ya, proba))
    assert np.isclose(
        functional_value(model, Xa, auroc(1), ya), roc_auc_score(ya, proba)
    )


def test_exact_handles_ties():
    scores = np.array([0.2, 0.5, 0.5, 0.8])
    y = np.array([0, 0, 1, 1])
    assert np.isclose(auroc(1)(scores, y), roc_auc_score(y, scores))


def test_smoothed_converges_to_exact(clf_problem):
    model, X, y, Xa, ya = clf_problem
    proba = model.predict_proba(Xa)[:, 1]
    exact = auroc(1)(proba, ya)
    errors = [abs(auroc(1, tau=t)(proba, ya) - exact) for t in (0.2, 0.05, 0.01)]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 5e-3


def test_smoothed_grad_matches_fd(clf_problem):
    model, X, y, Xa, ya = clf_problem
    proba = model.predict_proba(Xa)[:, 1]
    F = auroc(1, tau=0.05)
    g_an = F.grad(proba, ya)
    h = 1e-6
    g_fd = np.empty_like(g_an)
    for i in range(proba.size):
        vp, vm = proba.copy(), proba.copy()
        vp[i] += h
        vm[i] -= h
        g_fd[i] = (F(vp, ya) - F(vm, ya)) / (2 * h)
    np.testing.assert_allclose(g_an, g_fd, atol=1e-8)


def test_closed_form_tracks_exact_refit(clf_problem):
    model, X, y, Xa, ya = clf_problem
    s_cf = (
        FunctionalInfluence(auroc(1, tau=0.05))
        .fit(model, X, y)
        .explain(Xa, ya)
    )
    s_rf = (
        RefitFunctionalInfluence(auroc(1), verbose=0)
        .fit(model, X, y)
        .explain(Xa, ya)
    )
    assert np.corrcoef(s_cf, s_rf)[0, 1] > 0.9


def test_hard_variant_rejected_by_closed_form(clf_problem):
    model, X, y, Xa, ya = clf_problem
    attr = FunctionalInfluence(auroc(1)).fit(model, X, y)
    with pytest.raises(ValueError, match="rank statistic"):
        attr.explain(Xa, ya)


def test_validation(clf_problem):
    model, X, y, Xa, ya = clf_problem
    proba = model.predict_proba(Xa)[:, 1]
    with pytest.raises(ValueError, match="tau must be positive"):
        auroc(1, tau=0.0)
    with pytest.raises(ValueError, match="reference labels"):
        auroc(1)(proba, None)
    with pytest.raises(ValueError, match="both positive and negative"):
        auroc(1)(proba, np.ones_like(ya))
