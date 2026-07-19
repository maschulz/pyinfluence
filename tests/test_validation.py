"""Tests for model validation and rejection logic."""

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.svm import SVC

from pyinfluence._validation import (
    check_is_fitted_model,
    extract_regularization,
    get_model_type,
    validate_model,
)


def _get_model(fixture_value):
    """Return model from fixture: (model, ...) or bare model."""
    if isinstance(fixture_value, tuple):
        return fixture_value[0]
    return fixture_value


# (fixture_name, expected get_model_type result)
GET_MODEL_TYPE_CASES = [
    ("fitted_ridge", "ridge"),
    ("fitted_linear_regression", "linear"),
    ("fitted_logistic_binary", "logistic"),
    ("fitted_ridge_cv", "ridge"),
    ("fitted_logistic_cv", "logistic"),
    ("fitted_ridge_classifier", "ridge_classifier"),
    ("fitted_ridge_classifier_cv", "ridge_classifier"),
    ("fitted_kernel_ridge", "kernel_ridge"),
    ("fitted_sgd_classifier", "unsupported"),
    ("fitted_lasso", "unsupported"),
]


@pytest.mark.parametrize("fixture_name,expected_type", GET_MODEL_TYPE_CASES)
def test_get_model_type_from_fixture(fixture_name, expected_type, request):
    """get_model_type() returns expected value for fixture-based models."""
    fixture_value = request.getfixturevalue(fixture_name)
    model = _get_model(fixture_value)
    assert get_model_type(model) == expected_type


def test_get_model_type_unsupported_random_forest(binary_classification_data):
    X_train, _, y_train, _ = binary_classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(
        X_train, y_train
    )
    assert get_model_type(model) == "unsupported"


def test_get_model_type_unsupported_svc(binary_classification_data):
    X_train, _, y_train, _ = binary_classification_data
    model = SVC(kernel="rbf").fit(X_train, y_train)
    assert get_model_type(model) == "unsupported"


def test_get_model_type_unsupported_elasticnet(regression_data):
    X_train, _, y_train, _ = regression_data
    model = ElasticNet(alpha=0.1).fit(X_train, y_train)
    assert get_model_type(model) == "unsupported"


# (fixture_name, should_raise)
CHECK_FITTED_CASES = [
    ("fitted_ridge", False),
    ("fitted_logistic_binary", False),
    ("fitted_ridge_classifier", False),
    ("fitted_kernel_ridge", False),
    ("unfitted_ridge", True),
    ("unfitted_logistic", True),
    ("unfitted_ridge_classifier", True),
    ("unfitted_kernel_ridge", True),
]


@pytest.mark.parametrize("fixture_name,should_raise", CHECK_FITTED_CASES)
def test_check_is_fitted_model(fixture_name, should_raise, request):
    """check_is_fitted_model() passes for fitted, raises for unfitted."""
    fixture_value = request.getfixturevalue(fixture_name)
    model = _get_model(fixture_value)
    if should_raise:
        with pytest.raises(ValueError, match="not fitted"):
            check_is_fitted_model(model)
    else:
        check_is_fitted_model(model)


# (fixture_name,) for valid; validate_model should not raise
VALIDATE_VALID_CASES = [
    "fitted_ridge",
    "fitted_linear_regression",
    "fitted_logistic_binary",
    "fitted_ridge_cv",
    "fitted_logistic_cv",
    "fitted_ridge_classifier",
    "fitted_ridge_classifier_cv",
    "fitted_kernel_ridge",
]


@pytest.mark.parametrize("fixture_name", VALIDATE_VALID_CASES)
def test_validate_model_valid(fixture_name, request):
    """validate_model() accepts supported fitted models."""
    fixture_value = request.getfixturevalue(fixture_name)
    model = _get_model(fixture_value)
    validate_model(model)


# (fixture_name, match_regex)
VALIDATE_REJECTED_CASES = [
    ("fitted_sgd_classifier", "[Uu]nsupported"),
    ("fitted_lasso", "[Uu]nsupported"),
    ("unfitted_ridge", "not fitted"),
    ("unfitted_ridge_classifier", "not fitted"),
    ("unfitted_kernel_ridge", "not fitted"),
]


@pytest.mark.parametrize("fixture_name,match", VALIDATE_REJECTED_CASES)
def test_validate_model_rejected(fixture_name, match, request):
    """validate_model() rejects unsupported or unfitted models."""
    fixture_value = request.getfixturevalue(fixture_name)
    model = _get_model(fixture_value)
    with pytest.raises(ValueError, match=match):
        validate_model(model)


def test_validate_model_logistic_multiclass_rejected():
    """validate_model() rejects multiclass logistic regression."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    model = LogisticRegression(C=1.0, max_iter=500).fit(X, y)
    with pytest.raises(ValueError, match="more than two classes"):
        validate_model(model)


def test_validate_model_ridge_classifier_multiclass_rejected():
    """validate_model() rejects multiclass RidgeClassifier."""
    from sklearn.linear_model import RidgeClassifier

    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    model = RidgeClassifier(alpha=1.0).fit(X, y)
    with pytest.raises(ValueError, match="more than two classes"):
        validate_model(model)


def test_validate_model_random_forest_rejected(binary_classification_data):
    """validate_model() rejects RandomForest (unsupported)."""
    X_train, _, y_train, _ = binary_classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(
        X_train, y_train
    )
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        validate_model(model)


# (fixture_name, expected extract_regularization value). Add row when adding a supported model.
EXTRACT_REGULARIZATION_CASES = [
    ("fitted_ridge", 1.0),
    ("fitted_linear_regression", 0.0),
    ("fitted_logistic_binary", 1.0),
    ("fitted_ridge_classifier", 1.0),
    ("fitted_kernel_ridge", 1.0),
]


@pytest.mark.parametrize("fixture_name,expected_reg", EXTRACT_REGULARIZATION_CASES)
def test_extract_regularization(fixture_name, expected_reg, request):
    """extract_regularization() returns expected value for fixture-based models."""
    fixture_value = request.getfixturevalue(fixture_name)
    model = _get_model(fixture_value)
    assert extract_regularization(model) == expected_reg


def test_extract_regularization_ridge_cv(fitted_ridge_cv):
    """RidgeCV extract_regularization returns CV-selected alpha (alpha_)."""
    model = fitted_ridge_cv[0]
    reg = extract_regularization(model)
    assert reg == model.alpha_
    assert reg in [0.1, 1.0, 10.0]


def test_extract_regularization_logistic_cv(fitted_logistic_cv):
    """LogisticRegressionCV extract_regularization returns 1/C_[0]."""
    model = fitted_logistic_cv[0]
    reg = extract_regularization(model)
    assert reg == pytest.approx(1.0 / model.C_[0])
    assert model.C_[0] in [0.1, 1.0, 10.0]


def test_extract_regularization_ridge_classifier_cv(fitted_ridge_classifier_cv):
    """RidgeClassifierCV extract_regularization returns CV-selected alpha_."""
    model = fitted_ridge_classifier_cv[0]
    reg = extract_regularization(model)
    assert reg == model.alpha_
    assert reg in [0.1, 1.0, 10.0]


def test_extract_regularization_logistic_custom_c(binary_classification_data):
    """LogisticRegression C=0.5 -> lambda = 2.0."""
    X_train, _, y_train, _ = binary_classification_data
    model = LogisticRegression(C=0.5).fit(X_train, y_train)
    assert extract_regularization(model) == 2.0


def test_extract_regularization_logistic_no_penalty(binary_classification_data):
    """LogisticRegression penalty=None -> 0.0."""
    X_train, _, y_train, _ = binary_classification_data
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500).fit(
        X_train, y_train
    )
    assert extract_regularization(model) == 0.0


# (fixture_name, warning_match). Add row when adding a model that should warn.
VALIDATE_WARNINGS_CASES = [
    ("fitted_linear_regression", "no regularization"),
]


@pytest.mark.parametrize("fixture_name,warning_match", VALIDATE_WARNINGS_CASES)
def test_validate_model_warns(fixture_name, warning_match, request):
    """validate_model() warns when appropriate (e.g. unregularized)."""
    fixture_value = request.getfixturevalue(fixture_name)
    model = _get_model(fixture_value)
    with pytest.warns(UserWarning, match=warning_match):
        validate_model(model)


def test_validate_model_warns_no_penalty_logistic(binary_classification_data):
    """validate_model() warns for LogisticRegression with penalty=None."""
    X_train, _, y_train, _ = binary_classification_data
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500).fit(
        X_train, y_train
    )
    with pytest.warns(UserWarning, match="no regularization"):
        validate_model(model)
