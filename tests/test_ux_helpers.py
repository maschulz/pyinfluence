"""Tests for supports(), the fit(X, y) typo hint, and warning consolidation."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from pyinfluence import (
    BanzhafInfluence,
    BootstrapInfluence,
    InfluenceFunctions,
    LOOInfluence,
    supports,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X @ np.array([1.0, -1.0, 0.5]) + 0.1 * rng.normal(size=40)
    return X, y


# -----------------------------------------------------------------------------
# supports()
# -----------------------------------------------------------------------------


def test_supports_ok(data):
    X, y = data
    ok, reason = supports(Ridge(alpha=1.0).fit(X, y))
    assert ok and reason is None


def test_supports_unfitted():
    ok, reason = supports(Ridge())
    assert not ok
    assert "not fitted" in reason


def test_supports_unsupported_type(data):
    X, y = data
    ok, reason = supports(DecisionTreeRegressor(max_depth=2).fit(X, y))
    assert not ok
    assert "Unsupported" in reason


def test_supports_class_weight(data):
    X, y = data
    yb = (y > 0).astype(int)
    clf = LogisticRegression(class_weight="balanced", max_iter=500).fit(X, yb)
    ok, reason = supports(clf)
    assert not ok
    assert "class_weight" in reason


def test_supports_never_warns(data):
    """supports() must swallow the no-regularization warning."""
    X, y = data
    model = LinearRegression().fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok, reason = supports(model)
    assert ok


# -----------------------------------------------------------------------------
# fit(X, y) typo hint
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attr_cls",
    [InfluenceFunctions, LOOInfluence, BanzhafInfluence, BootstrapInfluence],
)
def test_fit_signature_typo_hint(data, attr_cls):
    """Passing arrays sklearn-style (fit(X, y)) gets a pointed TypeError."""
    X, y = data
    with pytest.raises(TypeError, match=r"fit\(model, X, y\)"):
        attr_cls().fit(X, y, y)


# -----------------------------------------------------------------------------
# Bootstrap warning consolidation
# -----------------------------------------------------------------------------


def test_bootstrap_single_consolidated_warning(data):
    """Degraded-coverage cases produce ONE warning, not one per condition."""
    X, y = data
    model = Ridge(alpha=1.0).fit(X, y)
    # 5 runs on 40 points: some points get 0 or <3 OOB runs (seeded, stable)
    attr = BootstrapInfluence(
        mode="loss", n_estimators=5, random_state=0, verbose=0
    )
    attr.fit(model, X, y)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        attr.explain(X[:5], y[:5])
    boot = [w for w in rec if "BootstrapInfluence" in str(w.message)]
    assert len(boot) == 1
    # The individual conditions remain greppable in the single message
    assert "OOB" in str(boot[0].message)
