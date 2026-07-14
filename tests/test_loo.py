"""Tests for Leave-One-Out influence (method-specific only).

Universal contract (output shape, not_fitted, y_test, scores valid, sign, single point)
lives in test_attributor_contract.py. Here: verbose, parallel, edge cases (failed refit, RF).
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pyinfluence import LOOInfluence
from tests.helpers import assert_influence_scores_valid

pytestmark = pytest.mark.slow


class TestLOOInfluenceBasic:
    """LOO-specific: verbose and prediction-mode requirements."""

    def test_verbose_zero_runs_without_error(self, fitted_ridge):
        """LOO fit and explain with verbose=0 should run without error (no progress bar)."""
        model, X_train, y_train, X_test, y_test = fitted_ridge

        loo = LOOInfluence(mode="loss", verbose=0)
        loo.fit(model, X_train, y_train)
        scores = loo.explain(X_test, y_test)
        assert_influence_scores_valid(scores, X_test.shape[0], X_train.shape[0])

    # prediction mode + classifier requires y_test: test_attributor_contract.test_prediction_mode_classifier_requires_y_test


class TestLOOInfluenceParallel:
    """Tests for parallel execution."""

    def test_parallel_gives_same_results(self, fitted_ridge):
        """Parallel and sequential LOO should give identical results."""
        model, X_train, y_train, X_test, y_test = fitted_ridge

        loo_seq = LOOInfluence(mode="loss", n_jobs=1)
        loo_seq.fit(model, X_train, y_train)
        scores_seq = loo_seq.explain(X_test, y_test)

        loo_par = LOOInfluence(mode="loss", n_jobs=2)
        loo_par.fit(model, X_train, y_train)
        scores_par = loo_par.explain(X_test, y_test)

        np.testing.assert_allclose(scores_seq, scores_par, rtol=1e-10)

class TestLOOInfluenceEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_failed_refit_gracefully(self):
        """
        LOO should handle cases where retraining fails due to class imbalance.

        When removing a sample leaves a class with 0 examples, the refit fails.
        LOO should return NaN for that sample's influence and warn.
        """
        # Create data with extreme imbalance: one class has only 1 sample
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.zeros(50, dtype=int)
        y[0] = 1  # Only 1 sample of class 1

        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X, y)

        X_test = np.random.randn(5, 5)
        y_test = np.array([0, 0, 1, 0, 0])

        loo = LOOInfluence(mode="loss")

        # Should warn about failed refits
        with pytest.warns(UserWarning, match="LOO refit failed"):
            loo.fit(model, X, y)

        scores = loo.explain(X_test, y_test)

        # Score for sample 0 (the only minority class sample) should be NaN
        assert np.isnan(scores[:, 0]).all(), (
            "Failed refits should produce NaN scores"
        )

        # Other scores should be finite
        successful_mask = ~np.isnan(scores[0, :])
        assert np.sum(successful_mask) > 0, "Some refits should succeed"
        assert np.all(np.isfinite(scores[:, successful_mask]))

    # Single test point: contract test_single_test_point_shape in test_attributor_contract.py.

    def test_works_with_random_forest(self, binary_classification_data):
        """LOO should work with any sklearn estimator (e.g., RandomForest)."""
        X_train, X_test, y_train, y_test = binary_classification_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        loo = LOOInfluence(mode="loss")
        loo.fit(model, X_train, y_train)
        scores = loo.explain(X_test, y_test)
        assert_influence_scores_valid(scores, X_test.shape[0], X_train.shape[0])

@pytest.mark.parametrize("fitted_fixture,is_regression", [
    ("fitted_logistic_binary", False),
    ("fitted_ridge", True),
], ids=["classification", "regression"])
def test_prediction_method(fitted_fixture, is_regression, request):
    """Prediction mode: classification requires y_test; regression can omit it. Regression column j = baseline - loo_pred."""
    model, X_train, y_train, X_test, y_test = request.getfixturevalue(fitted_fixture)
    loo = LOOInfluence(mode="prediction")
    loo.fit(model, X_train, y_train)
    scores = loo.explain(X_test, y_test)
    assert_influence_scores_valid(scores, X_test.shape[0], X_train.shape[0])
    if is_regression:
        scores_no_y = loo.explain(X_test)
        np.testing.assert_allclose(scores_no_y, scores)
        j = 0
        assert loo.loo_models_[j] is not None
        baseline = model.predict(X_test)
        loo_pred = loo.loo_models_[j].predict(X_test)
        np.testing.assert_allclose(scores[:, j], baseline - loo_pred)


class TestLOOSelfInfluenceDiag:
    """_self_influence_diag() matches the diagonal of the full score matrix."""

    def test_matches_diagonal_of_explain(self, fitted_ridge):
        model, X_train, y_train, X_test, y_test = fitted_ridge
        loo = LOOInfluence(mode="loss", verbose=0)
        loo.fit(model, X_train, y_train)

        diag_direct = loo._self_influence_diag()
        full = loo.explain(X_train, y_train)
        np.testing.assert_allclose(diag_direct, np.diag(full), equal_nan=True)
