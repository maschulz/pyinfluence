"""Tests for guardrails added around InfluenceFunctions / refit-based
attributors: incompatible model configurations, sparse input, multi-output
y, wrapped estimators, and unfitted models.
"""

import warnings

import numpy as np
import pytest
import scipy.sparse
from sklearn.datasets import make_classification
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyinfluence import (
    BanzhafInfluence,
    BootstrapInfluence,
    InfluenceFunctions,
    LOOInfluence,
    influence,
)
from pyinfluence._linear import _compute_kernel_matrix
from pyinfluence._validation import extract_regularization

REFIT_ATTRIBUTOR_CLASSES = [LOOInfluence, BanzhafInfluence, BootstrapInfluence]


def _kwargs_for(cls):
    if cls is InfluenceFunctions:
        return {}
    if cls is BanzhafInfluence:
        return {"n_samples": 5, "verbose": 0}
    if cls is BootstrapInfluence:
        return {"n_estimators": 3, "verbose": 0}
    return {"verbose": 0}


# -----------------------------------------------------------------------------
# InfluenceFunctions.fit rejects incompatible LogisticRegression configs
# -----------------------------------------------------------------------------


class TestIncompatibleConfigurationsRejected:
    def test_class_weight_balanced_raises(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(
            X_train, y_train
        )
        with pytest.raises(
            ValueError, match="LOOInfluence.*BanzhafInfluence.*BootstrapInfluence"
        ):
            InfluenceFunctions().fit(model, X_train, y_train)

    @pytest.mark.parametrize("solver", ["liblinear", "saga"])
    def test_l1_penalty_raises(self, binary_classification_data, solver):
        X_train, _, y_train, _ = binary_classification_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(penalty="l1", solver=solver, max_iter=2000).fit(
                X_train, y_train
            )
        with pytest.raises(
            ValueError, match="LOOInfluence.*BanzhafInfluence.*BootstrapInfluence"
        ):
            InfluenceFunctions().fit(model, X_train, y_train)


class TestInfluenceAutoFallback:
    def test_incompatible_config_warns_and_falls_back_to_bootstrap(
        self, binary_classification_data
    ):
        X_train, X_test, y_train, y_test = binary_classification_data
        model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(
            X_train, y_train
        )
        with pytest.warns(UserWarning, match="falling back"):
            scores, attributor = influence(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                method="auto",
                return_attributor=True,
                n_estimators=5,
                verbose=0,
            )
        assert isinstance(attributor, BootstrapInfluence)
        assert scores.shape == (X_test.shape[0], X_train.shape[0])

    def test_pipeline_warns_about_wrapping_and_falls_back(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        pipe = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))]).fit(
            X_train, y_train
        )
        with pytest.warns(UserWarning, match="wraps its estimator"):
            scores, attributor = influence(
                pipe,
                X_train,
                y_train,
                X_test,
                y_test,
                method="auto",
                return_attributor=True,
                n_estimators=5,
                verbose=0,
            )
        assert isinstance(attributor, BootstrapInfluence)
        assert scores.shape == (X_test.shape[0], X_train.shape[0])


# -----------------------------------------------------------------------------
# Sparse input rejected by every attributor
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("attr_cls", [InfluenceFunctions] + REFIT_ATTRIBUTOR_CLASSES)
def test_sparse_X_raises_type_error(attr_cls, fitted_ridge):
    model, X_train, y_train, X_test, y_test = fitted_ridge
    X_sparse = scipy.sparse.csr_matrix(X_train)
    attr = attr_cls(**_kwargs_for(attr_cls))
    with pytest.raises(TypeError, match="sparse"):
        attr.fit(model, X_sparse, y_train)


# -----------------------------------------------------------------------------
# Multi-output y rejected
# -----------------------------------------------------------------------------


def test_multi_output_y_raises(fitted_ridge):
    model, X_train, y_train, X_test, y_test = fitted_ridge
    y_2d = np.column_stack([y_train, y_train])
    with pytest.raises(ValueError, match="Multi-output"):
        InfluenceFunctions().fit(model, X_train, y_2d)


# -----------------------------------------------------------------------------
# Unfitted models rejected immediately (before any refitting)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("attr_cls", REFIT_ATTRIBUTOR_CLASSES)
def test_unfitted_model_raises_before_refits(attr_cls, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    attr = attr_cls(**_kwargs_for(attr_cls))
    with pytest.raises(ValueError, match="not fitted"):
        attr.fit(Ridge(alpha=1.0), X_train, y_train)


def test_multiclass_ridge_classifier_without_predict_proba_raises_at_fit():
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_classes=3,
        n_informative=6,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=0,
    )
    model = RidgeClassifier(alpha=1.0).fit(X, y)
    attr = LOOInfluence(mode="loss", verbose=0)
    with pytest.raises(ValueError, match="predict_proba"):
        attr.fit(model, X, y)


# -----------------------------------------------------------------------------
# extract_regularization: array-valued alpha implies multi-output, rejected
# -----------------------------------------------------------------------------


def test_extract_regularization_array_alpha_raises(regression_data):
    X_train, _, y_train, _ = regression_data
    model = Ridge(alpha=1.0).fit(X_train, y_train)
    model.alpha = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Array-valued alpha"):
        extract_regularization(model)


# -----------------------------------------------------------------------------
# liblinear intercept-regularization warning
# -----------------------------------------------------------------------------


def test_liblinear_solver_warns_about_intercept(binary_classification_data):
    X_train, _, y_train, _ = binary_classification_data
    model = LogisticRegression(solver="liblinear", max_iter=1000).fit(X_train, y_train)
    with pytest.warns(UserWarning, match="liblinear"):
        InfluenceFunctions().fit(model, X_train, y_train)


# -----------------------------------------------------------------------------
# KernelRidge with a callable kernel
# -----------------------------------------------------------------------------


def _rbf_like(a, b):
    return np.exp(-0.5 * np.sum((a - b) ** 2))


def test_kernel_ridge_callable_kernel_matches_sklearn_and_does_not_raise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = rng.normal(size=30)
    model = KernelRidge(alpha=1.0, kernel=_rbf_like).fit(X, y)

    K_ours = _compute_kernel_matrix(model, X)
    K_sklearn = model._get_kernel(X)
    np.testing.assert_allclose(K_ours, K_sklearn)

    attr = InfluenceFunctions(mode="loss", damping=1e-5)
    attr.fit(model, X, y)
    scores = attr.explain(X[:5], y[:5])
    assert scores.shape == (5, 30)
