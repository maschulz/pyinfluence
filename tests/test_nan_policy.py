"""Tests for the NaN policy shared by pyinfluence._utils helpers.

Refit-based attributors (LOO, Banzhaf, Bootstrap) legitimately produce NaN
scores where a point's effect is unmeasurable. Every utility in
pyinfluence._utils follows one rule: NaN scores are excluded from rankings
and statistics (never silently ranked, averaged, or treated as zero), and
the function warns with the count. These tests use small hand-built score
arrays (and, where a fitted attributor is required, minimal stub
attributors) so the NaN handling can be checked exactly.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone

from pyinfluence._base import BaseAttributor
from pyinfluence._utils import (
    _compute_loss_sklearn,
    compare_attributors,
    find_mislabeled,
    influence_summary,
    removal_curve,
    top_influential,
)

# -----------------------------------------------------------------------------
# top_influential
# -----------------------------------------------------------------------------


def test_top_influential_nan_excluded_from_both_rankings():
    scores = np.array([0.5, np.nan, 0.8, -0.3, np.nan, -0.9, 0.1])
    with pytest.warns(UserWarning, match="NaN"):
        helpful, harmful = top_influential(scores, k=3)

    assert 1 not in helpful and 1 not in harmful
    assert 4 not in helpful and 4 not in harmful
    # Exact ranking among the 5 finite entries (0.5, 0.8, -0.3, -0.9, 0.1).
    assert list(helpful) == [2, 0, 6]
    assert list(harmful) == [5, 3, 6]


def test_top_influential_nan_excluded_2d():
    scores = np.array(
        [
            [0.5, np.nan, 0.8, -0.1, 0.2],
            [-0.1, 0.9, np.nan, 0.3, -0.5],
        ]
    )
    with pytest.warns(UserWarning, match="NaN"):
        helpful, harmful = top_influential(scores, k=2)

    # NaN columns never appear for either test row.
    assert 1 not in helpful[0] and 1 not in harmful[0]
    assert 2 not in helpful[1] and 2 not in harmful[1]


# -----------------------------------------------------------------------------
# influence_summary
# -----------------------------------------------------------------------------


def test_influence_summary_nan_excluded_from_statistics():
    scores = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    with pytest.warns(UserWarning, match="NaN"):
        result = influence_summary(scores)

    finite = np.array([1.0, 2.0, 4.0, 5.0])
    assert result["n_nan"] == 1
    np.testing.assert_allclose(result["mean"], finite.mean())
    np.testing.assert_allclose(result["std"], finite.std())
    np.testing.assert_allclose(result["min"], finite.min())
    np.testing.assert_allclose(result["max"], finite.max())


def test_influence_summary_no_nan_does_not_warn():
    scores = np.array([1.0, 2.0, 3.0])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = influence_summary(scores)
    assert result["n_nan"] == 0
    assert not any("NaN" in str(w.message) for w in record)


# -----------------------------------------------------------------------------
# find_mislabeled: stub attributor exposing a fixed self-influence diagonal
# -----------------------------------------------------------------------------


class _StubSelfInfluenceAttributor(BaseAttributor):
    """Minimal fitted-looking attributor with a fixed ``_self_influence_diag``.

    Only what ``find_mislabeled``/``self_influence`` touch is implemented:
    ``mode``, ``model_``, and ``_self_influence_diag()``.
    """

    def __init__(self, diag):
        self.mode = "loss"
        self.model_ = object()
        self._diag = np.asarray(diag, dtype=float)

    def fit(self, model, X, y):
        return self

    def explain(self, X_test, y_test=None):
        raise NotImplementedError("not exercised by find_mislabeled")

    def _self_influence_diag(self):
        return self._diag


def test_find_mislabeled_nan_self_influence_excluded_but_outliers_flagged():
    rng = np.random.default_rng(0)
    diag = rng.normal(scale=0.1, size=20)
    diag[3] = np.nan
    diag[7] = np.nan
    diag[10] = 5.0  # clear outlier
    diag[15] = -5.0  # clear outlier

    attr = _StubSelfInfluenceAttributor(diag)
    with pytest.warns(UserWarning, match="NaN"):
        suspected = find_mislabeled(attr, threshold="auto")

    assert 10 in suspected
    assert 15 in suspected
    assert 3 not in suspected
    assert 7 not in suspected


# -----------------------------------------------------------------------------
# compare_attributors: stub attributors with fixed (partially NaN) scores
# -----------------------------------------------------------------------------


class _FixedScoresAttributor(BaseAttributor):
    """Attributor stub whose ``explain`` always returns a fixed score array."""

    def __init__(self, scores):
        self.model_ = object()
        self._scores = np.asarray(scores, dtype=float)

    def fit(self, model, X, y):
        return self

    def explain(self, X_test, y_test=None):
        return self._scores


def test_compare_attributors_drops_nan_pairs_and_reports_count():
    scores1 = np.array(
        [
            [1.0, 2.0, np.nan, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ]
    )
    scores2 = np.array(
        [
            [1.1, 2.1, 3.1, 4.1, 5.1],
            [6.1, 7.1, 8.1, np.nan, 10.1],
        ]
    )
    attr1 = _FixedScoresAttributor(scores1)
    attr2 = _FixedScoresAttributor(scores2)

    with pytest.warns(UserWarning, match="NaN"):
        result = compare_attributors(attr1, attr2, X_test=np.zeros((2, 1)))

    assert result["n_nan_dropped"] == 2
    assert np.isfinite(result["pearson"])
    assert np.isfinite(result["spearman"])
    assert np.isfinite(result["kendall"])
    # The two score matrices are nearly identical on the 8 valid pairs.
    assert result["pearson"] > 0.99


# -----------------------------------------------------------------------------
# removal_curve: NaN-scored points are ranked last (never removed first)
# -----------------------------------------------------------------------------


class _FixedAggScoresAttributor(BaseAttributor):
    """Attributor stub with a fixed 1D aggregate score per training point."""

    def __init__(self, model, X_train, y_train, scores):
        self.mode = "loss"
        self.model_ = model
        self.X_train_ = X_train
        self.y_train_ = y_train
        self._scores = np.asarray(scores, dtype=float)

    def fit(self, model, X, y):
        return self

    def explain(self, X_test, y_test=None):
        return self._scores


def test_removal_curve_never_removes_nan_scored_points_first(small_fitted_ridge):
    model, X_train, y_train, X_test, y_test = small_fitted_ridge
    n_train = X_train.shape[0]

    # Ascending scores: indices 0, 1 would be removed first under
    # direction='harmful' if not NaN (most negative = most harmful).
    scores = np.linspace(-1.0, 1.0, n_train)
    nan_positions = [0, 1]
    scores[nan_positions] = np.nan

    attr = _FixedAggScoresAttributor(model, X_train, y_train, scores)
    n_remove = 2
    frac = n_remove / n_train

    with pytest.warns(UserWarning, match="NaN"):
        result = removal_curve(attr, X_test, y_test, fractions=[0.0, frac], n_random=0)

    # Expected removal set: the n_remove smallest-scored *finite* points.
    finite_idx = np.array([i for i in range(n_train) if i not in nan_positions])
    expected_remove = finite_idx[np.argsort(scores[finite_idx])[:n_remove]]
    assert not set(expected_remove) & set(nan_positions)

    keep = np.ones(n_train, dtype=bool)
    keep[expected_remove] = False
    refit = clone(model).fit(X_train[keep], y_train[keep])
    expected_loss = float(
        np.mean(_compute_loss_sklearn(refit, X_test, y_test, is_classifier=False))
    )

    np.testing.assert_allclose(result["by_influence"][1], expected_loss)
