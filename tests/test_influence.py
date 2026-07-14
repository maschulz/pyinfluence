"""Tests for influence function computation (method-specific only).

Universal contract (output shape, not_fitted, y_test, scores valid, sign) is in
test_attributor_contract.py. Here: Hessian/gradient unit tests, Fit, formula,
prediction mode, logistic/intercept specifics.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from pyinfluence import InfluenceFunctions
from tests.helpers import assert_influence_scores_valid
from pyinfluence._linear import (
    _augment_intercept,
    _damping_matrix,
    _get_params,
    _get_params_logistic,
    _gradients_logistic,
    _gradients_ridge,
    _hessian_logistic,
    _hessian_ridge,
)

# Fixture names for parameterized tests over all supported linear(-like) models.
SUPPORTED_LINEAR_MODEL_FIXTURES = [
    "fitted_ridge",
    "fitted_linear_regression",
    "fitted_logistic_binary",
    "fitted_ridge_cv",
    "fitted_logistic_cv",
    "fitted_ridge_classifier",
    "fitted_ridge_classifier_cv",
    "fitted_kernel_ridge",
]


class TestHessianRidge:
    """Tests for Hessian computation on Ridge regression."""

    def test_hessian_shape(self, regression_data):
        """Hessian should be (p, p) matrix."""
        X_train, _, y_train, _ = regression_data
        n, p = X_train.shape
        H = _hessian_ridge(X_train, reg_lambda=1.0, damping=1e-5, has_intercept=False)
        assert H.shape == (p, p)

    def test_hessian_shape_with_intercept(self, regression_data):
        """Hessian with intercept should be (p+1, p+1)."""
        X_train, _, y_train, _ = regression_data
        n, p = X_train.shape
        X_aug = _augment_intercept(X_train)
        H = _hessian_ridge(X_aug, reg_lambda=1.0, damping=1e-5, has_intercept=True)
        assert H.shape == (p + 1, p + 1)

    def test_hessian_symmetric(self, regression_data):
        """Hessian should be symmetric."""
        X_train, _, _, _ = regression_data
        H = _hessian_ridge(X_train, reg_lambda=1.0, damping=1e-5, has_intercept=False)
        np.testing.assert_allclose(H, H.T, rtol=1e-10)

    def test_hessian_positive_definite(self, regression_data):
        """Hessian should be positive definite (all eigenvalues > 0)."""
        X_train, _, _, _ = regression_data
        H = _hessian_ridge(X_train, reg_lambda=1.0, damping=1e-5, has_intercept=False)
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues > 0)

    def test_hessian_manual_computation(self):
        """Compare Hessian to manual calculation on small example."""
        # Small deterministic example
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        n, p = X.shape
        reg_lambda = 0.5
        damping = 1e-5

        # Manual: H = X'X / n + diag([lambda, lambda]) + damping * I
        expected = X.T @ X / n + np.diag([reg_lambda, reg_lambda]) + damping * np.eye(p)

        H = _hessian_ridge(X, reg_lambda=reg_lambda, damping=damping, has_intercept=False)
        np.testing.assert_allclose(H, expected, rtol=1e-10)

    def test_hessian_intercept_not_regularized(self):
        """Intercept column should not have regularization applied."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        n, p = X.shape
        X_aug = _augment_intercept(X)  # Now shape (3, 3)
        reg_lambda = 1.0
        damping = 1e-5

        # Manual: H = X'X / n + diag([lambda, lambda, 0]) + damping_matrix
        # damping_matrix has minimal damping on intercept
        expected_reg = np.diag([reg_lambda, reg_lambda, 0.0])
        expected_damp = _damping_matrix(p + 1, damping, has_intercept=True)
        expected = X_aug.T @ X_aug / n + expected_reg + expected_damp

        H = _hessian_ridge(X_aug, reg_lambda=reg_lambda, damping=damping, has_intercept=True)
        np.testing.assert_allclose(H, expected, rtol=1e-10)


class TestGradientsRidge:
    """Tests for per-sample gradient computation on Ridge regression."""

    def test_gradients_shape(self, fitted_ridge):
        """Gradients should be (n, p) matrix."""
        model, X_train, y_train, _, _ = fitted_ridge
        theta = _get_params(model)
        X = X_train if not model.fit_intercept else _augment_intercept(X_train)
        grads = _gradients_ridge(X, y_train, theta)
        assert grads.shape == (X_train.shape[0], len(theta))

    def test_gradients_manual_computation(self):
        """Compare gradients to manual calculation on small example."""
        # Small deterministic example
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([1.0, 2.0, 3.0])
        theta = np.array([0.5, 0.3])

        # Per-sample gradient of (1/2)(y_i - x_i'theta)^2 is -(y_i - x_i'theta) * x_i
        # = (x_i'theta - y_i) * x_i
        predictions = X @ theta
        residuals = y - predictions
        expected = -X * residuals[:, np.newaxis]

        grads = _gradients_ridge(X, y, theta)
        np.testing.assert_allclose(grads, expected, rtol=1e-10)

    def test_gradients_sum_to_loss_gradient(self, fitted_ridge):
        """Mean of per-sample gradients should equal full loss gradient (without reg)."""
        model, X_train, y_train, _, _ = fitted_ridge
        theta = _get_params(model)
        X = X_train if not model.fit_intercept else _augment_intercept(X_train)

        grads = _gradients_ridge(X, y_train, theta)
        mean_grad = grads.mean(axis=0)

        # Full gradient of (1/2n) sum (y_i - x_i'theta)^2 is (1/n) sum g_i
        predictions = X @ theta
        residuals = y_train - predictions
        expected_full_grad = -X.T @ residuals / len(y_train)

        np.testing.assert_allclose(mean_grad, expected_full_grad, rtol=1e-8)


class TestAugmentIntercept:
    """Tests for intercept augmentation."""

    def test_augment_adds_ones_column(self):
        """Augment should add column of ones at the end."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_aug = _augment_intercept(X)
        assert X_aug.shape == (2, 3)
        np.testing.assert_array_equal(X_aug[:, -1], np.ones(2))
        np.testing.assert_array_equal(X_aug[:, :-1], X)


class TestGetParams:
    """Tests for parameter extraction from models."""

    def test_get_params_no_intercept(self, regression_data):
        """Get params without intercept."""
        X_train, _, y_train, _ = regression_data
        model = Ridge(alpha=1.0, fit_intercept=False).fit(X_train, y_train)
        theta = _get_params(model)
        np.testing.assert_array_equal(theta, model.coef_)

    def test_get_params_with_intercept(self, fitted_ridge):
        """Get params with intercept appended."""
        model, _, _, _, _ = fitted_ridge
        theta = _get_params(model)
        expected = np.concatenate([model.coef_, [model.intercept_]])
        np.testing.assert_array_equal(theta, expected)


class TestDampingMatrix:
    """Tests for non-uniform damping."""

    def test_damping_uniform_no_intercept(self):
        """Without intercept, damping is uniform."""
        D = _damping_matrix(5, damping=1e-5, has_intercept=False)
        expected = 1e-5 * np.eye(5)
        np.testing.assert_allclose(D, expected)

    def test_damping_minimal_on_intercept(self):
        """With intercept, last diagonal entry should be minimal."""
        D = _damping_matrix(5, damping=1e-5, has_intercept=True)
        # First 4 entries: 1e-5
        np.testing.assert_allclose(np.diag(D)[:4], np.full(4, 1e-5))
        # Last entry: min(1e-10, 1e-5 * 1e-3) = 1e-10
        assert D[-1, -1] == pytest.approx(1e-10)


# Prediction mode regression (no y_test): test_attributor_contract.test_prediction_mode_regression_does_not_require_y_test.


class TestInfluenceFunctionsFit:
    """Tests for InfluenceFunctions.fit() behavior."""

    # test_fit_returns_self: in test_sklearn_compat.py (TestFitReturnsSelf).

    def test_fit_stores_H_inv(self, fitted_ridge):
        """fit() should compute and store H_inv_."""
        model, X_train, y_train, _, _ = fitted_ridge
        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model, X_train, y_train)
        assert hasattr(attr, "H_inv_")
        # H_inv should be square with dimension = n_features (+ 1 if intercept)
        expected_dim = X_train.shape[1] + (1 if model.fit_intercept else 0)
        assert attr.H_inv_.shape == (expected_dim, expected_dim)

    def test_fit_stores_train_grads(self, fitted_ridge):
        """fit() should compute and store train_grads_."""
        model, X_train, y_train, _, _ = fitted_ridge
        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model, X_train, y_train)
        assert hasattr(attr, "train_grads_")
        expected_dim = X_train.shape[1] + (1 if model.fit_intercept else 0)
        assert attr.train_grads_.shape == (X_train.shape[0], expected_dim)

    def test_fit_stores_model_type(self, fitted_ridge):
        """fit() should detect and store model type."""
        model, X_train, y_train, _, _ = fitted_ridge
        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model, X_train, y_train)
        assert hasattr(attr, "model_type_")
        assert attr.model_type_ == "ridge"

    # test_invalid_mode_raises: in test_sklearn_compat.TestInvalidModeRaises.


class TestInfluenceFunctionsNumericalCorrectness:
    """Tests for numerical correctness of influence computation."""

    def test_influence_formula(self, fitted_ridge):
        """
        Verify influence = (test_grad @ H_inv @ train_grads.T) / n
        (removal-calibrated) for a simple case.
        """
        model, X_train, y_train, X_test, y_test = fitted_ridge

        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model, X_train, y_train)
        scores = attr.explain(X_test, y_test)

        # Manual computation
        theta = _get_params(model)
        X_test_aug = _augment_intercept(X_test) if model.fit_intercept else X_test

        # Test gradients for loss mode: -(y_test - prediction) * x_test
        predictions = X_test_aug @ theta
        residuals = y_test - predictions
        test_grads = -X_test_aug * residuals[:, np.newaxis]

        # I = (test_grads @ H_inv @ train_grads.T) / n  (removal weight -1/n)
        n_train = X_train.shape[0]
        expected = test_grads @ attr.H_inv_ @ attr.train_grads_.T / n_train
        np.testing.assert_allclose(scores, expected, rtol=1e-8)

    def test_prediction_mode_formula(self, fitted_ridge):
        """
        Verify prediction mode: grad_theta(prediction) = x_test.
        """
        model, X_train, y_train, X_test, _ = fitted_ridge

        attr = InfluenceFunctions(mode="prediction", damping=1e-5)
        attr.fit(model, X_train, y_train)
        scores = attr.explain(X_test)

        # For linear regression, grad of prediction w.r.t. theta is x_test
        X_test_aug = _augment_intercept(X_test) if model.fit_intercept else X_test
        test_grads = X_test_aug

        n_train = X_train.shape[0]
        expected = -(test_grads @ attr.H_inv_ @ attr.train_grads_.T) / n_train
        np.testing.assert_allclose(scores, expected, rtol=1e-8)


# =============================================================================
# Logistic Regression Tests (Phase 3)
# =============================================================================


class TestHessianLogistic:
    """Tests for Hessian computation on Logistic regression."""

    def test_hessian_shape_binary(self, binary_classification_data):
        """Hessian for binary logistic should be (p, p) matrix."""
        X_train, _, y_train, _ = binary_classification_data
        n, p = X_train.shape
        # Compute probabilities (mock fitted model)
        probs = np.full(n, 0.5)  # Placeholder
        H = _hessian_logistic(X_train, probs, reg_lambda=1.0, damping=1e-5, has_intercept=False)
        assert H.shape == (p, p)

    def test_hessian_shape_with_intercept(self, binary_classification_data):
        """Hessian with intercept should be (p+1, p+1)."""
        X_train, _, y_train, _ = binary_classification_data
        n, p = X_train.shape
        X_aug = _augment_intercept(X_train)
        probs = np.full(n, 0.5)
        H = _hessian_logistic(X_aug, probs, reg_lambda=1.0, damping=1e-5, has_intercept=True)
        assert H.shape == (p + 1, p + 1)

    def test_hessian_symmetric(self, binary_classification_data):
        """Hessian should be symmetric."""
        X_train, _, y_train, _ = binary_classification_data
        n = X_train.shape[0]
        probs = np.random.rand(n) * 0.8 + 0.1  # Random probs in (0.1, 0.9)
        H = _hessian_logistic(X_train, probs, reg_lambda=1.0, damping=1e-5, has_intercept=False)
        np.testing.assert_allclose(H, H.T, rtol=1e-10)

    def test_hessian_positive_definite(self, binary_classification_data):
        """Hessian should be positive definite (all eigenvalues > 0)."""
        X_train, _, y_train, _ = binary_classification_data
        n = X_train.shape[0]
        probs = np.random.rand(n) * 0.8 + 0.1
        H = _hessian_logistic(X_train, probs, reg_lambda=1.0, damping=1e-5, has_intercept=False)
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues > 0)

    def test_hessian_manual_computation(self):
        """Compare Hessian to manual calculation on small example."""
        # Small deterministic example
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        n, p = X.shape
        probs = np.array([0.3, 0.5, 0.7])
        reg_lambda = 0.5
        damping = 1e-5

        # Manual: H = X' diag(p(1-p)) X / n + diag([lambda, lambda]) + damping * I
        weights = probs * (1 - probs)
        expected = X.T @ (X * weights[:, np.newaxis]) / n
        expected += np.diag([reg_lambda, reg_lambda])
        expected += damping * np.eye(p)

        H = _hessian_logistic(X, probs, reg_lambda=reg_lambda, damping=damping, has_intercept=False)
        np.testing.assert_allclose(H, expected, rtol=1e-10)

    def test_hessian_intercept_not_regularized(self):
        """Intercept column should not have regularization applied."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        n, p = X.shape
        X_aug = _augment_intercept(X)  # Now shape (3, 3)
        probs = np.array([0.3, 0.5, 0.7])
        reg_lambda = 1.0
        damping = 1e-5

        # Manual: H = X' diag(p(1-p)) X / n + diag([lambda, lambda, 0]) + damping_matrix
        weights = probs * (1 - probs)
        expected = X_aug.T @ (X_aug * weights[:, np.newaxis]) / n
        expected += np.diag([reg_lambda, reg_lambda, 0.0])
        expected += _damping_matrix(p + 1, damping, has_intercept=True)

        H = _hessian_logistic(X_aug, probs, reg_lambda=reg_lambda, damping=damping, has_intercept=True)
        np.testing.assert_allclose(H, expected, rtol=1e-10)


class TestGradientsLogistic:
    """Tests for per-sample gradient computation on Logistic regression."""

    def test_gradients_shape(self, fitted_logistic_binary):
        """Gradients should be (n, p) matrix."""
        model, X_train, y_train, _, _ = fitted_logistic_binary
        X = X_train if not model.fit_intercept else _augment_intercept(X_train)
        probs = model.predict_proba(X_train)[:, 1]
        grads = _gradients_logistic(X, y_train, probs)
        n_features = X_train.shape[1] + (1 if model.fit_intercept else 0)
        assert grads.shape == (X_train.shape[0], n_features)

    def test_gradients_manual_computation(self):
        """Compare gradients to manual calculation on small example."""
        # Small deterministic example
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 1])
        probs = np.array([0.3, 0.6, 0.8])

        # Per-sample gradient: g_i = -(y_i - p_i) * x_i
        expected = -X * (y - probs)[:, np.newaxis]

        grads = _gradients_logistic(X, y, probs)
        np.testing.assert_allclose(grads, expected, rtol=1e-10)

    def test_gradients_sum_to_loss_gradient(self, fitted_logistic_binary):
        """Mean of per-sample gradients should equal full loss gradient (without reg)."""
        model, X_train, y_train, _, _ = fitted_logistic_binary
        X = X_train if not model.fit_intercept else _augment_intercept(X_train)
        probs = model.predict_proba(X_train)[:, 1]

        grads = _gradients_logistic(X, y_train, probs)
        mean_grad = grads.mean(axis=0)

        # Full gradient of NLL is (1/n) sum_i -(y_i - p_i) * x_i
        expected_full_grad = -X.T @ (y_train - probs) / len(y_train)

        np.testing.assert_allclose(mean_grad, expected_full_grad, rtol=1e-8)


class TestGetParamsLogistic:
    """Tests for parameter extraction from logistic models."""

    def test_get_params_binary(self, fitted_logistic_binary):
        """Get params for binary logistic regression."""
        model, _, _, _, _ = fitted_logistic_binary
        theta = _get_params_logistic(model, class_idx=0)
        expected = np.concatenate([model.coef_.ravel(), [model.intercept_[0]]])
        np.testing.assert_allclose(theta, expected)


class TestInfluenceFunctionsLogisticBinary:
    """Influence-specific: logistic model type and intercept handling."""

    def test_stores_model_type_logistic(self, fitted_logistic_binary):
        """fit() should detect logistic model type."""
        model, X_train, y_train, _, _ = fitted_logistic_binary
        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model, X_train, y_train)
        assert attr.model_type_ == "logistic"


class TestInfluenceFunctionsLogisticIntercept:
    """Tests for intercept handling in logistic regression."""

    def test_intercept_handling_binary(self, binary_classification_data):
        """Verify intercept is properly handled for binary classification."""
        X_train, X_test, y_train, y_test = binary_classification_data

        # Model with intercept
        model_with = LogisticRegression(
            C=1.0, fit_intercept=True, random_state=42
        ).fit(X_train, y_train)

        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model_with, X_train, y_train)
        scores_with = attr.explain(X_test, y_test)

        assert_influence_scores_valid(
            scores_with, X_test.shape[0], X_train.shape[0]
        )

    def test_intercept_handling_no_intercept(self, binary_classification_data):
        """Verify model without intercept works correctly."""
        X_train, X_test, y_train, y_test = binary_classification_data

        # Model without intercept
        model_without = LogisticRegression(
            C=1.0, fit_intercept=False, random_state=42
        ).fit(X_train, y_train)

        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model_without, X_train, y_train)
        scores_without = attr.explain(X_test, y_test)

        assert_influence_scores_valid(
            scores_without, X_test.shape[0], X_train.shape[0]
        )

    def test_intercept_different_from_no_intercept(self, binary_classification_data):
        """Models with/without intercept should produce different influence scores."""
        X_train, X_test, y_train, y_test = binary_classification_data

        model_with = LogisticRegression(
            C=1.0, fit_intercept=True, random_state=42
        ).fit(X_train, y_train)

        model_without = LogisticRegression(
            C=1.0, fit_intercept=False, random_state=42
        ).fit(X_train, y_train)

        attr_with = InfluenceFunctions(mode="loss", damping=1e-5)
        attr_with.fit(model_with, X_train, y_train)
        scores_with = attr_with.explain(X_test, y_test)

        attr_without = InfluenceFunctions(mode="loss", damping=1e-5)
        attr_without.fit(model_without, X_train, y_train)
        scores_without = attr_without.explain(X_test, y_test)

        # Scores should be different (models are different)
        assert not np.allclose(scores_with, scores_without)


# -----------------------------------------------------------------------------
# Loss–prediction relationship, single test point, damping (from integration_sign)
# -----------------------------------------------------------------------------


class TestLossPredictionRelationship:
    """For regression: I_loss = residual * I_pred."""

    def test_loss_prediction_relationship(self, fitted_ridge):
        """I_loss = residual * I_pred for squared loss."""
        model, X_train, y_train, X_test, y_test = fitted_ridge
        attr_loss = InfluenceFunctions(damping=1e-5, mode="loss")
        attr_loss.fit(model, X_train, y_train)
        I_loss = attr_loss.explain(X_test, y_test)
        attr_pred = InfluenceFunctions(damping=1e-5, mode="prediction")
        attr_pred.fit(model, X_train, y_train)
        I_pred = attr_pred.explain(X_test)
        predictions = model.predict(X_test)
        residuals = y_test - predictions
        expected_I_loss = residuals[:, np.newaxis] * I_pred
        np.testing.assert_allclose(I_loss, expected_I_loss, rtol=1e-8)

@pytest.mark.parametrize("fitted_fixture", ["fitted_ridge", "fitted_logistic_binary"])
def test_higher_damping_reduces_influence_magnitude(fitted_fixture, request):
    """Higher damping should reduce influence magnitudes (regression and classification)."""
    model, X_train, y_train, X_test, y_test = request.getfixturevalue(fitted_fixture)
    attr_low = InfluenceFunctions(damping=1e-5, mode="loss")
    attr_low.fit(model, X_train, y_train)
    scores_low = attr_low.explain(X_test, y_test)
    attr_high = InfluenceFunctions(damping=1.0, mode="loss")
    attr_high.fit(model, X_train, y_train)
    scores_high = attr_high.explain(X_test, y_test)
    assert np.mean(np.abs(scores_high)) < np.mean(np.abs(scores_low))


# Single test point and 1D X_test: test_attributor_contract.test_single_test_point_and_1d_input.


class TestInputValidation:
    """InfluenceFunctions accepts list and 2D y input."""

    def test_accepts_list_input(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model, X_train.tolist(), y_train.tolist())
        scores = attr.explain(X_test.tolist(), y_test.tolist())
        assert_influence_scores_valid(
            scores, len(X_test), len(X_train), check_not_all_zero=False
        )

    def test_accepts_2d_y(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model, X_train, y_train.reshape(-1, 1))
        scores = attr.explain(X_test, y_test.reshape(-1, 1))
        assert_influence_scores_valid(
            scores, len(X_test), len(X_train), check_not_all_zero=False
        )


class TestDeterminism:
    """InfluenceFunctions results are deterministic."""

    def test_repeated_fit_explain_identical(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        attr1 = InfluenceFunctions(damping=1e-5, mode="loss")
        attr1.fit(model, X_train, y_train)
        scores1 = attr1.explain(X_test, y_test)
        attr2 = InfluenceFunctions(damping=1e-5, mode="loss")
        attr2.fit(model, X_train, y_train)
        scores2 = attr2.explain(X_test, y_test)
        np.testing.assert_array_equal(scores1, scores2)

    def test_repeated_explain_identical(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        attr.fit(model, X_train, y_train)
        np.testing.assert_array_equal(
            attr.explain(X_test, y_test), attr.explain(X_test, y_test)
        )
