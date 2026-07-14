"""Tests for high-level API: influence()."""

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from pyinfluence import influence
from tests.helpers import assert_influence_scores_valid


# -----------------------------------------------------------------------------
# influence() — method='auto'
# -----------------------------------------------------------------------------


class TestInfluenceAuto:
    def test_ridge_uses_influence_functions(self, fitted_ridge):
        model, X_train, y_train, X_test, y_test = fitted_ridge
        scores = influence(
            model, X_train, y_train, X_test, y_test, method="auto"
        )
        assert_influence_scores_valid(
            scores, X_test.shape[0], X_train.shape[0], check_not_all_zero=False
        )

    @pytest.mark.slow
    def test_auto_with_fallback_loo_for_non_linear(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(
            X_train, y_train
        )
        scores = influence(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            method="auto",
            fallback="loo",
        )
        assert_influence_scores_valid(
            scores, X_test.shape[0], X_train.shape[0], check_not_all_zero=False
        )

    @pytest.mark.slow
    def test_auto_with_fallback_banzhaf_for_non_linear(self, small_fitted_ridge):
        # Use small data so Banzhaf is fast; use RF to trigger fallback
        model, X_train, y_train, X_test, y_test = small_fitted_ridge
        model = RandomForestRegressor(n_estimators=5, random_state=42).fit(
            X_train, y_train
        )
        scores = influence(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            method="auto",
            fallback="banzhaf",
            n_samples=20,
        )
        assert_influence_scores_valid(
            scores, X_test.shape[0], X_train.shape[0], check_not_all_zero=False
        )

    def test_auto_fallback_bootstrap_uses_bootstrap(self, small_fitted_ridge):
        """method='auto' with non-linear model and fallback='bootstrap' uses BootstrapInfluence."""
        model, X_train, y_train, X_test, y_test = small_fitted_ridge
        rf = RandomForestRegressor(n_estimators=5, random_state=42).fit(
            X_train, y_train
        )
        scores = influence(
            rf,
            X_train,
            y_train,
            X_test,
            y_test,
            method="auto",
            fallback="bootstrap",
            n_estimators=8,
            verbose=0,
        )
        assert_influence_scores_valid(
            scores, X_test.shape[0], X_train.shape[0],
            check_finite=False, check_not_all_zero=False,
        )


# -----------------------------------------------------------------------------
# influence() — explicit method
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method,fixture_name,extra_kwargs,check_finite,check_not_all_zero",
    [
        pytest.param("loo", "fitted_ridge", {}, True, False, marks=pytest.mark.slow),
        pytest.param("banzhaf", "small_fitted_ridge", {"n_samples": 30}, True, False, marks=pytest.mark.slow),
        ("bootstrap", "small_fitted_ridge", {"n_estimators": 8, "verbose": 0}, False, False),
    ],
    ids=["loo", "banzhaf", "bootstrap"],
)
def test_explicit_method_returns_valid_scores(
    method, fixture_name, extra_kwargs, check_finite, check_not_all_zero, request
):
    """influence(..., method=X) returns valid scores for each method."""
    model, X_train, y_train, X_test, y_test = request.getfixturevalue(fixture_name)
    scores = influence(
        model, X_train, y_train, X_test, y_test, method=method, **extra_kwargs
    )
    assert_influence_scores_valid(
        scores, X_test.shape[0], X_train.shape[0],
        check_finite=check_finite, check_not_all_zero=check_not_all_zero,
    )


# -----------------------------------------------------------------------------
# influence() — return_attributor
# -----------------------------------------------------------------------------


class TestInfluenceReturnAttributor:
    def test_return_attributor_true_returns_tuple(self, fitted_ridge):
        model, X_train, y_train, X_test, y_test = fitted_ridge
        result = influence(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            method="auto",
            return_attributor=True,
        )
        assert isinstance(result, tuple)
        scores, attributor = result
        assert_influence_scores_valid(
            scores, X_test.shape[0], X_train.shape[0], check_not_all_zero=False
        )
        assert hasattr(attributor, "model_")
        # Can call explain again
        scores2 = attributor.explain(X_test, y_test)
        assert_influence_scores_valid(
            scores2, X_test.shape[0], X_train.shape[0], check_not_all_zero=False
        )

    def test_return_attributor_false_returns_ndarray(self, fitted_ridge):
        model, X_train, y_train, X_test, y_test = fitted_ridge
        result = influence(
            model, X_train, y_train, X_test, y_test, return_attributor=False
        )
        assert not isinstance(result, tuple)
        assert_influence_scores_valid(
            result, X_test.shape[0], X_train.shape[0], check_not_all_zero=False
        )


# -----------------------------------------------------------------------------
# influence() — validation
# -----------------------------------------------------------------------------


def test_mode_loss_without_y_test_raises(fitted_ridge):
    model, X_train, y_train, X_test, _ = fitted_ridge
    with pytest.raises(ValueError, match="y_test is required"):
        influence(model, X_train, y_train, X_test, y_test=None, mode="loss")


def test_X_train_y_train_length_mismatch_raises(fitted_ridge):
    model, X_train, y_train, X_test, y_test = fitted_ridge
    with pytest.raises(ValueError, match="X_train and y_train must have the same length"):
        influence(model, X_train, y_train[: len(y_train) - 1], X_test, y_test)


def test_X_test_y_test_length_mismatch_raises(fitted_ridge):
    model, X_train, y_train, X_test, y_test = fitted_ridge
    with pytest.raises(ValueError, match="X_test and y_test must have the same number of samples"):
        influence(model, X_train, y_train, X_test, y_test[: len(y_test) - 1])


def test_unsupported_model_explicit_influence_functions_raises(fitted_sgd_classifier):
    model, X_train, y_train, X_test, y_test = fitted_sgd_classifier
    with pytest.raises(ValueError, match="Unsupported model type"):
        influence(
            model, X_train, y_train, X_test, y_test,
            method="influence_functions",
        )
