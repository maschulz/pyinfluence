"""Tests for the AUROC functional and the perturbation-evaluation path."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pyinfluence import (
    FunctionalInfluence,
    RefitFunctionalInfluence,
    functional_value,
)
from pyinfluence.fairness import disparity
from pyinfluence.functionals import auroc


@pytest.fixture(scope="module")
def clf_problem():
    rng = np.random.default_rng(0)
    n, p, m = 150, 5, 200
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = (X @ beta + 1.0 * rng.normal(size=n) > 0).astype(int)
    Xa = rng.normal(size=(m, p))
    ya = (Xa @ beta + 1.0 * rng.normal(size=m) > 0).astype(int)
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


def test_closed_form_tracks_exact_refit(clf_problem):
    """Perturbation evaluation must track exact-refit ground truth closely."""
    model, X, y, Xa, ya = clf_problem
    s_cf = (
        FunctionalInfluence(auroc(1), damping=1e-8)
        .fit(model, X, y)
        .explain(Xa, ya)
    )
    s_rf = (
        RefitFunctionalInfluence(auroc(1), verbose=0)
        .fit(model, X, y)
        .explain(Xa, ya)
    )
    assert np.corrcoef(s_cf, s_rf)[0, 1] > 0.9
    # the estimand is quantized: exact zeros are expected and preserved
    assert (s_cf == 0).any()
    # zero/nonzero structure largely agrees with ground truth
    assert np.mean((s_cf == 0) == (s_rf == 0)) > 0.8


def test_perturbation_matches_gradient_for_smooth_functionals(clf_problem):
    """For a smooth functional both engine paths agree to first order."""
    model, X, y, Xa, ya = clf_problem
    rng = np.random.default_rng(1)
    a = (rng.uniform(size=len(ya)) < 0.5).astype(int)
    F_grad = disparity("dp", a)
    F_pert = dataclasses.replace(F_grad, grad=None, differentiable=False)
    attr = FunctionalInfluence(damping=1e-8).fit(model, X, y)
    s_grad = attr.explain(Xa, functional=F_grad)
    s_pert = attr.explain(Xa, functional=F_pert)
    assert np.corrcoef(s_grad, s_pert)[0, 1] > 0.999
    np.testing.assert_allclose(s_grad, s_pert, rtol=0.2, atol=1e-8)


def test_validation(clf_problem):
    model, X, y, Xa, ya = clf_problem
    proba = model.predict_proba(Xa)[:, 1]
    with pytest.raises(ValueError, match="reference labels"):
        auroc(1)(proba, None)
    with pytest.raises(ValueError, match="both positive and negative"):
        auroc(1)(proba, np.ones_like(ya))
    with pytest.raises(ValueError, match="row-aligned labels"):
        auroc(1)(proba, ya[:-5])
