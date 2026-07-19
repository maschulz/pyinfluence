"""Tests for Bootstrap (OOB) influence (method-specific only).

Universal contract and sign convention live in test_attributor_contract.py.
Here: BootstrapIndices (unit), Fit, Explain (modes), WithRandomForest.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from pyinfluence._bootstrap import BootstrapInfluence, _bootstrap_indices
from tests.helpers import assert_influence_scores_valid


class TestBootstrapIndices:
    """Unit tests for bootstrap index sampling."""

    def test_bootstrap_indices_shape(self):
        """Each bootstrap sample should have n_samples indices."""
        rng = np.random.default_rng(42)
        in_bag_list = _bootstrap_indices(n_samples=100, n_estimators=10, rng=rng)
        assert len(in_bag_list) == 10
        for indices in in_bag_list:
            assert indices.shape == (100,)
            assert indices.dtype in (np.intp, np.int64, np.int32)
            assert np.all((indices >= 0) & (indices < 100))

    def test_bootstrap_indices_reproducible(self):
        """Same seed should give same indices."""
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        list1 = _bootstrap_indices(50, 5, rng1)
        list2 = _bootstrap_indices(50, 5, rng2)
        for a, b in zip(list1, list2):
            np.testing.assert_array_equal(a, b)


class TestBootstrapFit:
    """Unit tests for BootstrapInfluence.fit."""

    def test_fit_stores_attributes(self, small_fitted_ridge):
        """fit() should set model_, X_train_, y_train_, bootstrap_models_, in_bag_indices_."""
        model, X_train, y_train, X_test, y_test = small_fitted_ridge
        attr = BootstrapInfluence(
            mode="loss",
            n_estimators=5,
            random_state=42,
            verbose=0,
        )
        attr.fit(model, X_train, y_train)
        assert attr.model_ is model
        np.testing.assert_array_equal(attr.X_train_, X_train)
        np.testing.assert_array_equal(attr.y_train_, y_train)
        assert len(attr.bootstrap_models_) == 5
        assert len(attr.in_bag_indices_) == 5
        assert all(m is not None for m in attr.bootstrap_models_)
        assert attr.is_classifier_ is False

    # test_fit_returns_self: in test_sklearn_compat.py (TestFitReturnsSelf).


class TestBootstrapModes:
    """Both loss and prediction modes."""

    def test_prediction_mode_regression(self, small_fitted_ridge):
        """Prediction mode should run (shape covered by contract test_prediction_mode_regression_does_not_require_y_test)."""
        model, X_train, y_train, X_test, y_test = small_fitted_ridge
        attr = BootstrapInfluence(
            mode="prediction",
            n_estimators=8,
            random_state=42,
            verbose=0,
        )
        attr.fit(model, X_train, y_train)
        scores = attr.explain(X_test)
        assert_influence_scores_valid(
            scores,
            X_test.shape[0],
            X_train.shape[0],
            check_finite=False,
            check_not_all_zero=False,
        )


class TestBootstrapWithRandomForest:
    """Bootstrap with a non-linear model (typical use case)."""

    def test_random_forest_regression(self, small_fitted_ridge):
        """BootstrapInfluence should work with RandomForestRegressor."""
        model, X_train, y_train, X_test, y_test = small_fitted_ridge
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        attr = BootstrapInfluence(
            mode="loss",
            n_estimators=8,
            random_state=43,
            verbose=0,
        )
        attr.fit(rf, X_train, y_train)
        scores = attr.explain(X_test, y_test)
        assert_influence_scores_valid(
            scores,
            X_test.shape[0],
            X_train.shape[0],
            check_finite=False,
            check_not_all_zero=False,
        )


class TestBootstrapScoresStd:
    """attr.scores_std_ after explain()."""

    def test_shape_and_mostly_finite(self, small_fitted_ridge):
        model, X_train, y_train, X_test, y_test = small_fitted_ridge
        attr = BootstrapInfluence(
            mode="loss", n_estimators=40, random_state=0, verbose=0
        )
        attr.fit(model, X_train, y_train)
        scores = attr.explain(X_test, y_test)

        assert hasattr(attr, "scores_std_")
        assert attr.scores_std_.shape == scores.shape
        # Most points get plenty of OOB/in-bag runs with n_estimators=40.
        frac_finite = np.mean(np.isfinite(attr.scores_std_))
        assert frac_finite > 0.5, f"only {frac_finite:.1%} of scores_std_ finite"


class TestBootstrapNeverOOB:
    """A training point that is in-bag in every run gets NaN, with a warning
    naming the in-bag-in-every-run case specifically (not the every-OOB case)."""

    def test_point_never_oob_is_nan_with_warning(self, small_fitted_ridge):
        model, X_train, y_train, X_test, y_test = small_fitted_ridge
        attr = BootstrapInfluence(
            mode="loss", n_estimators=15, random_state=0, verbose=0
        )
        attr.fit(model, X_train, y_train)

        # Force index 0 to be in-bag in every bootstrap run.
        forced_in_bag = []
        for indices in attr.in_bag_indices_:
            if 0 not in indices:
                indices = np.append(indices, 0)
            forced_in_bag.append(indices)
        attr.in_bag_indices_ = forced_in_bag

        with pytest.warns(UserWarning, match="in-bag in every"):
            scores = attr.explain(X_test, y_test)

        assert np.isnan(scores[:, 0]).all()


class TestBootstrapSelfInfluenceDiag:
    """_self_influence_diag() matches the diagonal of the full score matrix."""

    def test_matches_diagonal_of_explain(self, small_fitted_ridge):
        model, X_train, y_train, X_test, y_test = small_fitted_ridge
        attr = BootstrapInfluence(
            mode="loss", n_estimators=25, random_state=0, verbose=0
        )
        attr.fit(model, X_train, y_train)

        diag_direct = attr._self_influence_diag()
        full = attr.explain(X_train, y_train)
        np.testing.assert_allclose(diag_direct, np.diag(full), equal_nan=True)
