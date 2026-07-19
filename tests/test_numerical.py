"""Tests for numerical stability: ill-conditioned Hessian and near-separable data."""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from pyinfluence import InfluenceFunctions
from tests.helpers import assert_influence_scores_valid


def _fit_collect_warnings(attr, model, X, y):
    """Run attr.fit(model, X, y) with warnings captured; return list of warning messages."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        attr.fit(model, X, y)
    return [str(x.message) for x in w]


def _condition_warnings(wlist):
    """Filter to messages mentioning condition number."""
    return [m for m in wlist if "condition number" in m.lower()]


def _separability_warnings(wlist):
    """Filter to messages about separability or p(1-p)."""
    return [
        m
        for m in wlist
        if any(
            t in m.lower()
            for t in ["separable", "p(1-p)", "near-singular", "probabilities"]
        )
    ]


class TestConditionNumberWarning:
    """Tests for Hessian condition number warning."""

    def test_warns_on_ill_conditioned_hessian(self):
        """Should warn when Hessian condition number exceeds threshold."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 100, 10
        X = rng.standard_normal((n_samples, n_features))
        X[:, -1] = X[:, 0] + rng.standard_normal(n_samples) * 1e-8
        y = rng.standard_normal(n_samples)
        model = LinearRegression().fit(X, y)
        attr = InfluenceFunctions(damping=1e-15, mode="loss")
        wlist = _fit_collect_warnings(attr, model, X, y)
        assert len(_condition_warnings(wlist)) >= 1, (
            "Expected warning about ill-conditioned Hessian"
        )

    def test_no_warning_on_well_conditioned_hessian(self, regression_data):
        """Should not warn when Hessian is well-conditioned."""
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        wlist = _fit_collect_warnings(attr, model, X_train, y_train)
        assert len(_condition_warnings(wlist)) == 0, (
            f"Unexpected: {_condition_warnings(wlist)}"
        )

    def test_higher_damping_reduces_condition_number(self):
        """Increasing damping should reduce condition number and prevent warning."""
        np.random.seed(42)
        n_samples, n_features = 100, 10
        X = np.random.randn(n_samples, n_features)
        X[:, -1] = X[:, 0] + np.random.randn(n_samples) * 1e-8
        y = np.random.randn(n_samples)
        model = LinearRegression().fit(X, y)
        attr = InfluenceFunctions(damping=1e-2, mode="loss")
        wlist = _fit_collect_warnings(attr, model, X, y)
        assert len(_condition_warnings(wlist)) == 0, (
            "Higher damping should prevent warning"
        )

    # Regularization preventing warning: same idea as test_higher_damping_reduces_condition_number; omitted.


class TestNearSeparableData:
    """Tests for logistic regression on near-separable data."""

    def test_warns_on_near_separable_data(self):
        """Should warn when data is near-separable (probs near 0 or 1)."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
        attr = InfluenceFunctions(damping=1e-10, mode="loss")
        wlist = _fit_collect_warnings(attr, model, X, y)
        assert len(_separability_warnings(wlist)) >= 1, (
            "Expected warning about near-separable data"
        )

    def test_does_not_crash_on_separable_data(self):
        """Should compute scores without crashing on separable data."""
        np.random.seed(42)

        # Create linearly separable data
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)

        attr = InfluenceFunctions(damping=1e-5, mode="loss")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings
            attr.fit(model, X, y)

            # Should compute scores without NaN or crash
            X_test = np.random.randn(10, 5)
            y_test = (X_test[:, 0] > 0).astype(int)

            scores = attr.explain(X_test, y_test)

            assert_influence_scores_valid(scores, 10, n_samples)

    def test_no_warning_on_non_separable_data(self, binary_classification_data):
        """Should not warn on well-behaved non-separable data."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = LogisticRegression(C=1.0, max_iter=1000).fit(X_train, y_train)
        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        wlist = _fit_collect_warnings(attr, model, X_train, y_train)
        assert len(_separability_warnings(wlist)) == 0, (
            f"Unexpected: {_separability_warnings(wlist)}"
        )

    def test_higher_regularization_prevents_separability_issues(self):
        """Higher regularization should prevent near-separability issues."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(C=0.1, max_iter=1000).fit(X, y)
        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        wlist = _fit_collect_warnings(attr, model, X, y)
        assert len(_separability_warnings(wlist)) == 0, (
            "Regularization should prevent warning"
        )


@pytest.mark.parametrize(
    "data_fixture,model_cls,kwargs",
    [
        ("regression_data", Ridge, {"alpha": 1.0}),
        (
            "binary_classification_data",
            LogisticRegression,
            {"C": 1.0, "max_iter": 1000},
        ),
    ],
    ids=["regression", "classification"],
)
def test_scores_are_finite(data_fixture, model_cls, kwargs, request):
    """Influence scores should be finite (no NaN or Inf) for regression and classification."""
    X_train, X_test, y_train, y_test = request.getfixturevalue(data_fixture)
    model = model_cls(**kwargs).fit(X_train, y_train)
    attr = InfluenceFunctions(damping=1e-5, mode="loss")
    attr.fit(model, X_train, y_train)
    scores = attr.explain(X_test, y_test)
    assert_influence_scores_valid(
        scores, X_test.shape[0], X_train.shape[0], check_not_all_zero=False
    )


def test_damping_prevents_singular_hessian():
    """Damping should prevent near-singular Hessian from causing issues."""
    np.random.seed(42)
    n_samples, n_features = 100, 10
    X = np.random.randn(n_samples, n_features)
    X[:, -1] = X[:, 0] + np.random.randn(n_samples) * 1e-6
    X[:, -2] = X[:, 1] + np.random.randn(n_samples) * 1e-6
    y = np.random.randn(n_samples)
    model = LinearRegression().fit(X, y)
    attr = InfluenceFunctions(damping=1e-2, mode="loss")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attr.fit(model, X, y)
        X_test = np.random.randn(5, n_features)
        y_test = np.random.randn(5)
        scores = attr.explain(X_test, y_test)
    assert_influence_scores_valid(scores, 5, n_samples, check_not_all_zero=False)


def test_small_damping_warns_on_ill_conditioned():
    """Small damping on ill-conditioned data should warn."""
    np.random.seed(42)
    n_samples, n_features = 100, 10
    X = np.random.randn(n_samples, n_features)
    X[:, -1] = X[:, 0] + np.random.randn(n_samples) * 1e-10
    y = np.random.randn(n_samples)
    model = LinearRegression().fit(X, y)
    attr = InfluenceFunctions(damping=1e-12, mode="loss")
    wlist = _fit_collect_warnings(attr, model, X, y)
    assert len(_condition_warnings(wlist)) >= 1
