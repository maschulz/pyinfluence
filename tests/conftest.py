"""Shared fixtures and helpers for pyinfluence tests.

Adding a new supported model:
  1. Add a data fixture if needed (e.g. regression_data is shared).
  2. Add fitted_<name> returning (model, X_train, y_train, X_test, y_test).
  3. In pyinfluence._validation: register in LINEAR_MODEL_REGISTRY, extend
     extract_regularization(), check_is_fitted_model(), and validate_model().
  4. In test_validation.py: add to GET_MODEL_TYPE_CASES, VALIDATE_VALID_CASES,
     EXTRACT_REGULARIZATION_CASES, etc.
  5. In test_attributor_contract.py: add CONTRACT_REGISTRY and SIGN_REGISTRY entries
     (and prediction registries if applicable).
  6. In test_influence.py: add fixture name to SUPPORTED_LINEAR_MODEL_FIXTURES.
"""

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
)
from sklearn.model_selection import train_test_split


def _make_small_regression():
    """Small regression dataset (50 samples, 5 features) for fast tests."""
    X, y = make_regression(
        n_samples=50,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------------------------------------------------------
# Data Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def regression_data():
    """Well-behaved regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def binary_classification_data():
    """Well-behaved binary classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------------------------------------------------------
# Fitted Model Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def fitted_ridge(regression_data):
    """Fitted Ridge model with its training/test data."""
    X_train, X_test, y_train, y_test = regression_data
    model = Ridge(alpha=1.0).fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture
def fitted_linear_regression(regression_data):
    """Fitted LinearRegression model (no regularization)."""
    X_train, X_test, y_train, y_test = regression_data
    model = LinearRegression().fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture
def fitted_logistic_binary(binary_classification_data):
    """Fitted LogisticRegression model (binary)."""
    X_train, X_test, y_train, y_test = binary_classification_data
    model = LogisticRegression(C=1.0, random_state=42).fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture
def fitted_ridge_cv(regression_data):
    """Fitted RidgeCV model (auto-selected alpha)."""
    X_train, X_test, y_train, y_test = regression_data
    model = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture
def fitted_logistic_cv(binary_classification_data):
    """Fitted LogisticRegressionCV model (auto-selected C, binary)."""
    X_train, X_test, y_train, y_test = binary_classification_data
    model = LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], random_state=42).fit(
        X_train, y_train
    )
    return model, X_train, y_train, X_test, y_test


@pytest.fixture
def fitted_ridge_classifier(binary_classification_data):
    """Fitted RidgeClassifier model (binary)."""
    X_train, X_test, y_train, y_test = binary_classification_data
    model = RidgeClassifier(alpha=1.0).fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture
def fitted_ridge_classifier_cv(binary_classification_data):
    """Fitted RidgeClassifierCV model (auto-selected alpha, binary)."""
    X_train, X_test, y_train, y_test = binary_classification_data
    model = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0]).fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture
def fitted_kernel_ridge():
    """Fitted KernelRidge model (linear kernel, tiny data for scipy stability)."""
    X, y = make_regression(
        n_samples=30,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = KernelRidge(alpha=1.0, kernel="linear").fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


# -----------------------------------------------------------------------------
# Unsupported Model Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def fitted_sgd_classifier(binary_classification_data):
    """Fitted SGDClassifier (unsupported - no closed-form Hessian)."""
    X_train, X_test, y_train, y_test = binary_classification_data
    model = SGDClassifier(loss="log_loss", random_state=42, max_iter=1000).fit(
        X_train, y_train
    )
    return model, X_train, y_train, X_test, y_test


@pytest.fixture
def fitted_lasso(regression_data):
    """Fitted Lasso (unsupported - L1 breaks smoothness)."""
    X_train, X_test, y_train, y_test = regression_data
    model = Lasso(alpha=0.1).fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture
def unfitted_ridge():
    """Unfitted Ridge model."""
    return Ridge(alpha=1.0)


@pytest.fixture
def unfitted_logistic():
    """Unfitted LogisticRegression model."""
    return LogisticRegression(C=1.0)


@pytest.fixture
def unfitted_ridge_classifier():
    """Unfitted RidgeClassifier model."""
    return RidgeClassifier(alpha=1.0)


@pytest.fixture
def unfitted_kernel_ridge():
    """Unfitted KernelRidge model."""
    return KernelRidge(alpha=1.0, kernel="rbf")


# -----------------------------------------------------------------------------
# Small / fast fixtures (Banzhaf correlation, etc.)
# -----------------------------------------------------------------------------


@pytest.fixture
def small_regression_data():
    """Small regression (X_train, X_test, y_train, y_test) only; no fitted model."""
    return _make_small_regression()


@pytest.fixture
def small_fitted_ridge():
    """Small regression data + fitted Ridge for fast Banzhaf/correlation tests."""
    X_train, X_test, y_train, y_test = _make_small_regression()
    model = Ridge(alpha=1.0).fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


# -----------------------------------------------------------------------------
# Fitted attributor fixtures (reduce fit + explain boilerplate)
# -----------------------------------------------------------------------------


@pytest.fixture
def fitted_influence_ridge(fitted_ridge):
    """Fitted InfluenceFunctions (loss mode) + Ridge and train/test data."""
    from pyinfluence import InfluenceFunctions

    model, X_train, y_train, X_test, y_test = fitted_ridge
    attr = InfluenceFunctions(damping=1e-5, mode="loss")
    attr.fit(model, X_train, y_train)
    return attr, model, X_train, y_train, X_test, y_test


@pytest.fixture
def fitted_influence_logistic(fitted_logistic_binary):
    """Fitted InfluenceFunctions (loss mode) + binary LogisticRegression and data."""
    from pyinfluence import InfluenceFunctions

    model, X_train, y_train, X_test, y_test = fitted_logistic_binary
    attr = InfluenceFunctions(damping=1e-5, mode="loss")
    attr.fit(model, X_train, y_train)
    return attr, model, X_train, y_train, X_test, y_test
