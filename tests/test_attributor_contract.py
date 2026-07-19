"""Universal contract tests for all attributors.

Every attributor (InfluenceFunctions, LOOInfluence, BanzhafInfluence, BootstrapInfluence)
must pass these tests. Adding a new method = add an entry to the registries below
and add method-specific tests in test_<method>.py.

Registries:
- CONTRACT_REGISTRY: (attributor_class, fixture_name, attr_kwargs, check_finite, check_not_all_zero).
  Used for: output shape, not_fitted, requires_y_test, scores_valid,
  single-test-point + 1D input (one test).
- SIGN_REGISTRY: (attributor_class, fixture_name, attr_kwargs, is_classifier).
  Used for sign sanity checks (remove helpful → loss increases; remove harmful → loss decreases).
- PREDICTION_REGRESSION_REGISTRY: (cls, fixture_name, kwargs) with mode="prediction" and
  regression fixture. Used for test_prediction_mode_regression_does_not_require_y_test (all methods).
- PREDICTION_CLASSIFICATION_REGISTRY: (cls, fixture_name, kwargs) with mode="prediction" and
  classification fixture. Used for test_prediction_mode_classifier_requires_y_test (LOO, Banzhaf,
  Bootstrap require y_test; InfluenceFunctions does not).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import log_loss, mean_squared_error

from pyinfluence import (
    BanzhafInfluence,
    BootstrapInfluence,
    InfluenceFunctions,
    LOOInfluence,
)
from tests.helpers import assert_influence_scores_valid


def _contract_entry(
    cls, fixture_name, kwargs, check_finite=True, check_not_all_zero=True
):
    return (cls, fixture_name, kwargs, check_finite, check_not_all_zero)


def _slow(entry_tuple):
    """Mark an entry as slow (for parametrize, pass the tuple as single value)."""
    return pytest.param(entry_tuple, marks=pytest.mark.slow)


# (attributor_class, fixture_name, attr_kwargs, check_finite, check_not_all_zero)
# Bootstrap uses check_finite=False because OOB can produce NaNs.
CONTRACT_REGISTRY = [
    _contract_entry(
        InfluenceFunctions, "fitted_ridge", {"mode": "loss", "damping": 1e-5}
    ),
    _contract_entry(
        InfluenceFunctions,
        "fitted_linear_regression",
        {"mode": "loss", "damping": 1e-5},
    ),
    _contract_entry(
        InfluenceFunctions, "fitted_logistic_binary", {"mode": "loss", "damping": 1e-5}
    ),
    _contract_entry(
        InfluenceFunctions, "fitted_ridge_cv", {"mode": "loss", "damping": 1e-5}
    ),
    _contract_entry(
        InfluenceFunctions, "fitted_logistic_cv", {"mode": "loss", "damping": 1e-5}
    ),
    _contract_entry(
        InfluenceFunctions, "fitted_ridge_classifier", {"mode": "loss", "damping": 1e-5}
    ),
    _contract_entry(
        InfluenceFunctions,
        "fitted_ridge_classifier_cv",
        {"mode": "loss", "damping": 1e-5},
    ),
    _contract_entry(
        InfluenceFunctions, "fitted_kernel_ridge", {"mode": "loss", "damping": 1e-5}
    ),
    _slow(_contract_entry(LOOInfluence, "fitted_ridge", {"mode": "loss"})),
    _slow(_contract_entry(LOOInfluence, "fitted_logistic_binary", {"mode": "loss"})),
    _slow(
        _contract_entry(
            BanzhafInfluence,
            "small_fitted_ridge",
            {"mode": "loss", "n_samples": 30, "random_state": 42},
        )
    ),
    _contract_entry(
        BootstrapInfluence,
        "small_fitted_ridge",
        {"mode": "loss", "n_estimators": 8, "random_state": 42, "verbose": 0},
        check_finite=False,
        check_not_all_zero=False,
    ),
]


def _unpack_entry(entry):
    """Unwrap pytest.param(entry) or return entry. Use in parametrized contract tests."""
    return entry.values[0] if hasattr(entry, "values") else entry


def _entry_id(entry):
    t = _unpack_entry(entry)
    return f"{t[0].__name__}-{t[1]}"


CONTRACT_IDS = [_entry_id(e) for e in CONTRACT_REGISTRY]


def _sign_entry(cls, fixture_name, kwargs, is_classifier=False):
    return (cls, fixture_name, kwargs, is_classifier)


# Sign sanity: remove helpful → loss increases; remove harmful → loss decreases.
# Banzhaf/Bootstrap need enough samples/estimators so "most harmful" is stable.
SIGN_REGISTRY = [
    _sign_entry(
        InfluenceFunctions, "fitted_ridge", {"mode": "loss", "damping": 1e-5}, False
    ),
    _sign_entry(
        InfluenceFunctions,
        "fitted_logistic_binary",
        {"mode": "loss", "damping": 1e-5},
        True,
    ),
    _sign_entry(
        InfluenceFunctions, "fitted_ridge_cv", {"mode": "loss", "damping": 1e-5}, False
    ),
    _sign_entry(
        InfluenceFunctions,
        "fitted_ridge_classifier",
        {"mode": "loss", "damping": 1e-5},
        True,
    ),
    _sign_entry(
        InfluenceFunctions,
        "fitted_kernel_ridge",
        {"mode": "loss", "damping": 1e-5},
        False,
    ),
    _slow(_sign_entry(LOOInfluence, "fitted_ridge", {"mode": "loss"}, False)),
    _slow(
        _sign_entry(
            BanzhafInfluence,
            "small_fitted_ridge",
            {"mode": "loss", "n_samples": 150, "random_state": 42},
            False,
        )
    ),
    _slow(
        _sign_entry(
            BootstrapInfluence,
            "fitted_ridge",
            {"mode": "loss", "n_estimators": 500, "random_state": 42, "verbose": 0},
            False,
        )
    ),
]

SIGN_IDS = [_entry_id(e) for e in SIGN_REGISTRY]

# (cls, fixture_name, kwargs) with mode="prediction" and regression fixture.
# All attributors: prediction mode regression does not require y_test.
PREDICTION_REGRESSION_REGISTRY = [
    (InfluenceFunctions, "fitted_ridge", {"mode": "prediction", "damping": 1e-5}),
    (InfluenceFunctions, "fitted_ridge_cv", {"mode": "prediction", "damping": 1e-5}),
    (
        InfluenceFunctions,
        "fitted_kernel_ridge",
        {"mode": "prediction", "damping": 1e-5},
    ),
    _slow((LOOInfluence, "fitted_ridge", {"mode": "prediction"})),
    _slow(
        (
            BanzhafInfluence,
            "small_fitted_ridge",
            {"mode": "prediction", "n_samples": 30, "random_state": 42},
        )
    ),
    (
        BootstrapInfluence,
        "small_fitted_ridge",
        {"mode": "prediction", "n_estimators": 8, "random_state": 42, "verbose": 0},
    ),
]

# (cls, fixture_name, kwargs) with mode="prediction" and classification fixture.
# LOO, Banzhaf, Bootstrap require y_test for prediction+classifier; InfluenceFunctions does not.
PREDICTION_CLASSIFICATION_REGISTRY = [
    _slow((LOOInfluence, "fitted_logistic_binary", {"mode": "prediction"})),
    _slow(
        (
            BanzhafInfluence,
            "fitted_logistic_binary",
            {"mode": "prediction", "n_samples": 30, "random_state": 42},
        )
    ),
    (
        BootstrapInfluence,
        "fitted_logistic_binary",
        {"mode": "prediction", "n_estimators": 8, "random_state": 42, "verbose": 0},
    ),
]


def _pred_class_id(entry):
    t = _unpack_entry(entry)
    return f"{t[0].__name__}-{t[1]}"


PREDICTION_CLASSIFICATION_IDS = [
    _pred_class_id(e) for e in PREDICTION_CLASSIFICATION_REGISTRY
]

PREDICTION_REGRESSION_IDS = [_pred_class_id(e) for e in PREDICTION_REGRESSION_REGISTRY]


def _retrain_without_index(model, X_train, y_train, idx):
    mask = np.ones(len(y_train), dtype=bool)
    mask[idx] = False
    return clone(model).fit(X_train[mask], y_train[mask])


def _model_loss(model, X_test, y_test, is_classifier):
    if is_classifier:
        if hasattr(model, "predict_proba") and callable(model.predict_proba):
            return log_loss(y_test, model.predict_proba(X_test))
        # Fallback for classifiers without predict_proba (e.g. RidgeClassifier):
        # use squared error on decision_function, matching _compute_loss_sklearn.
        decision = model.decision_function(X_test)
        y_binary = np.where(y_test == model.classes_[1], 1.0, -1.0)
        return float(np.mean((y_binary - decision) ** 2))
    return mean_squared_error(y_test, model.predict(X_test))


# -----------------------------------------------------------------------------
# Contract tests (parametrized over CONTRACT_REGISTRY)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("entry", CONTRACT_REGISTRY, ids=CONTRACT_IDS)
def test_not_fitted_raises(entry, request):
    """explain() before fit() must raise NotFittedError (or similar)."""
    cls, fixture_name, kwargs, _cf, _cz = _unpack_entry(entry)
    _, X_train, y_train, X_test, y_test = request.getfixturevalue(fixture_name)
    attr = cls(**kwargs)
    with pytest.raises((NotFittedError, ValueError)):
        attr.explain(X_test, y_test)


@pytest.mark.parametrize("entry", CONTRACT_REGISTRY, ids=CONTRACT_IDS)
def test_loss_mode_requires_y_test(entry, request):
    """With mode='loss', explain() must require y_test."""
    cls, fixture_name, kwargs, _cf, _cz = _unpack_entry(entry)
    model, X_train, y_train, X_test, _ = request.getfixturevalue(fixture_name)
    attr = cls(**kwargs)
    attr.fit(model, X_train, y_train)
    with pytest.raises(ValueError, match="y_test"):
        attr.explain(X_test)


@pytest.mark.parametrize(
    "entry", PREDICTION_CLASSIFICATION_REGISTRY, ids=PREDICTION_CLASSIFICATION_IDS
)
def test_prediction_mode_classifier_requires_y_test(entry, request):
    """With mode='prediction' and a classifier, explain() must require y_test."""
    cls, fixture_name, kwargs = _unpack_entry(entry)
    model, X_train, y_train, X_test, _ = request.getfixturevalue(fixture_name)
    attr = cls(**kwargs)
    attr.fit(model, X_train, y_train)
    with pytest.raises(ValueError, match="y_test"):
        attr.explain(X_test)


@pytest.mark.parametrize(
    "entry", PREDICTION_REGRESSION_REGISTRY, ids=PREDICTION_REGRESSION_IDS
)
def test_prediction_mode_regression_does_not_require_y_test(entry, request):
    """With mode='prediction' and regression, explain(X_test) without y_test must return (n_test, n_train)."""
    cls, fixture_name, kwargs = _unpack_entry(entry)
    model, X_train, y_train, X_test, _ = request.getfixturevalue(fixture_name)
    attr = cls(**kwargs)
    attr.fit(model, X_train, y_train)
    scores = attr.explain(X_test)
    assert scores.shape == (X_test.shape[0], X_train.shape[0])


@pytest.mark.parametrize("entry", CONTRACT_REGISTRY, ids=CONTRACT_IDS)
def test_scores_valid(entry, request):
    """explain() must return (n_test, n_train) and valid scores (finite/not-all-zero where applicable)."""
    cls, fixture_name, kwargs, check_finite, check_not_all_zero = _unpack_entry(entry)
    model, X_train, y_train, X_test, y_test = request.getfixturevalue(fixture_name)
    attr = cls(**kwargs)
    attr.fit(model, X_train, y_train)
    scores = attr.explain(X_test, y_test)
    assert_influence_scores_valid(
        scores,
        X_test.shape[0],
        X_train.shape[0],
        check_finite=check_finite,
        check_not_all_zero=check_not_all_zero,
    )


@pytest.mark.parametrize("entry", CONTRACT_REGISTRY, ids=CONTRACT_IDS)
def test_single_test_point_and_1d_input(entry, request):
    """explain() with one point: (1, n_train); 1D X_test/y_test must match 2D."""
    cls, fixture_name, kwargs, _cf, _cz = _unpack_entry(entry)
    model, X_train, y_train, X_test, y_test = request.getfixturevalue(fixture_name)
    attr = cls(**kwargs)
    attr.fit(model, X_train, y_train)
    scores_one = attr.explain(X_test[:1], y_test[:1])
    assert scores_one.shape == (1, X_train.shape[0])
    x_1d, y_1d = X_test[0], np.asarray(y_test[0])
    scores_1d = attr.explain(x_1d, y_1d)
    scores_2d = attr.explain(x_1d.reshape(1, -1), y_1d.reshape(1))
    assert scores_1d.shape == (1, X_train.shape[0])
    np.testing.assert_allclose(scores_1d, scores_2d, rtol=1e-10, equal_nan=True)


# -----------------------------------------------------------------------------
# Sign convention sanity (parametrized over SIGN_REGISTRY)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("entry", SIGN_REGISTRY, ids=SIGN_IDS)
def test_removing_helpful_point_increases_loss(entry, request):
    """Remove training point with highest positive influence → test loss should increase."""
    cls, fixture_name, kwargs, is_classifier = _unpack_entry(entry)
    model, X_train, y_train, X_test, y_test = request.getfixturevalue(fixture_name)
    attr = cls(**kwargs)
    attr.fit(model, X_train, y_train)
    scores = attr.explain(X_test, y_test)
    total_influence = scores.sum(axis=0)
    most_helpful_idx = int(np.argmax(total_influence))
    model_reduced = _retrain_without_index(model, X_train, y_train, most_helpful_idx)
    loss_full = _model_loss(model, X_test, y_test, is_classifier)
    loss_reduced = _model_loss(model_reduced, X_test, y_test, is_classifier)
    assert loss_reduced > loss_full, (
        f"Removing helpful point should increase loss. "
        f"Full: {loss_full:.6f}, Reduced: {loss_reduced:.6f}"
    )


@pytest.mark.parametrize("entry", SIGN_REGISTRY, ids=SIGN_IDS)
def test_removing_harmful_point_decreases_loss(entry, request):
    """Remove training point with most negative influence → test loss should decrease."""
    cls, fixture_name, kwargs, is_classifier = _unpack_entry(entry)
    model, X_train, y_train, X_test, y_test = request.getfixturevalue(fixture_name)
    attr = cls(**kwargs)
    attr.fit(model, X_train, y_train)
    scores = attr.explain(X_test, y_test)
    total_influence = scores.sum(axis=0)
    most_harmful_idx = int(np.argmin(total_influence))
    model_reduced = _retrain_without_index(model, X_train, y_train, most_harmful_idx)
    loss_full = _model_loss(model, X_test, y_test, is_classifier)
    loss_reduced = _model_loss(model_reduced, X_test, y_test, is_classifier)
    assert loss_reduced < loss_full, (
        f"Removing harmful point should decrease loss. "
        f"Full: {loss_full:.6f}, Reduced: {loss_reduced:.6f}"
    )
