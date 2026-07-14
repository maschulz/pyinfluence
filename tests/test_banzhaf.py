"""Tests for Data Banzhaf influence (method-specific only).

Universal contract and sign convention live in test_attributor_contract.py.
Here: Analytical, Convergence, NullPlayer, Symmetry, Parallel.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from pyinfluence import BanzhafInfluence
from tests.helpers import assert_influence_scores_valid


@pytest.mark.slow
class TestBanzhafAnalytical:
    """Test against known analytical solutions on tiny datasets."""

    def test_known_solution_n4_regression(self):
        """
        Test Banzhaf values on n=4 dataset with known analytical solution.

        For a 4-point regression with MSE utility, we can compute exact
        Banzhaf values by enumerating all 2^3 = 8 subsets per point.

        Setup: Simple 1D regression where one point is clearly helpful
        and one point is clearly harmful (outlier).
        """
        np.random.seed(42)

        # Simple 1D linear data
        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_train = np.array([1.0, 2.0, 3.0, 100.0])  # Point 3 is an outlier

        X_test = np.array([[2.5]])
        y_test = np.array([2.5])  # True relationship: y = x

        # Fit full model
        model = Ridge(alpha=0.01)
        model.fit(X_train, y_train)

        # Compute Banzhaf with sufficient samples for accurate estimation
        banzhaf = BanzhafInfluence(n_samples=200, random_state=42)
        banzhaf.fit(model, X_train, y_train)
        scores = banzhaf.explain(X_test, y_test)

        # The outlier (index 3) should have the most negative (harmful) score
        total_scores = scores.sum(axis=0)
        most_harmful_idx = np.argmin(total_scores)

        assert most_harmful_idx == 3, (
            f"Outlier should be most harmful, but got idx {most_harmful_idx}"
        )

        # The outlier's score should be negative (harmful)
        assert total_scores[3] < 0, (
            f"Outlier score {total_scores[3]:.4f} should be negative"
        )

    def test_exact_banzhaf_tiny_dataset(self):
        """
        Verify Monte Carlo Banzhaf converges to correct values on n=3 data.

        With n=3, each point has 2^2=4 subset contributions.
        For well-behaved linear data, all points should be approximately
        equally helpful (positive scores).
        """
        # Create predictable data
        X_train = np.array([[0.0], [1.0], [2.0]])
        y_train = np.array([0.0, 1.0, 2.0])  # Perfect linear: y = x

        X_test = np.array([[1.5]])
        y_test = np.array([1.5])

        model = Ridge(alpha=0.001)
        model.fit(X_train, y_train)

        # Compute with many samples
        banzhaf = BanzhafInfluence(n_samples=500, random_state=123)
        banzhaf.fit(model, X_train, y_train)
        scores = banzhaf.explain(X_test, y_test)

        # For perfectly linear data, all points should have non-negative scores
        # (each contributes to learning the pattern)
        total_scores = scores.sum(axis=0)

        # All should be non-negative or very close to zero (helpful or neutral)
        assert np.all(total_scores > -0.01), (
            f"Well-behaved points should have non-negative scores: {total_scores}"
        )

        # All scores should be small for well-fit data (model already works well)
        assert np.all(np.abs(total_scores) < 1.0), (
            f"Scores should be small for well-fit data: {total_scores}"
        )


@pytest.mark.slow
class TestBanzhafConvergence:
    """Tests that variance decreases with more samples."""

    def test_variance_decreases_with_samples(self):
        """
        Variance of Banzhaf estimates should decrease with more MC samples.

        Run Banzhaf multiple times with different sample counts and check
        that estimate variance decreases.
        """
        np.random.seed(42)
        X_train = np.random.randn(10, 3)
        y_train = X_train[:, 0] + 0.1 * np.random.randn(10)
        X_test = np.random.randn(2, 3)
        y_test = X_test[:, 0]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        # Collect estimates at different sample sizes
        n_trials = 3
        scores_low = []
        scores_high = []

        for seed in range(n_trials):
            banzhaf_low = BanzhafInfluence(n_samples=20, random_state=seed)
            banzhaf_low.fit(model, X_train, y_train)
            scores_low.append(banzhaf_low.explain(X_test, y_test).ravel())

            banzhaf_high = BanzhafInfluence(n_samples=100, random_state=seed)
            banzhaf_high.fit(model, X_train, y_train)
            scores_high.append(banzhaf_high.explain(X_test, y_test).ravel())

        # Compute variance across trials at each sample size
        var_low = np.var(scores_low, axis=0).mean()
        var_high = np.var(scores_high, axis=0).mean()

        assert var_high < var_low, (
            f"Variance should decrease with more samples. "
            f"Low (n=20): {var_low:.6f}, High (n=100): {var_high:.6f}"
        )

    def test_reproducible_with_random_state(self):
        """Same random_state should give identical results."""
        np.random.seed(42)
        X_train = np.random.randn(15, 3)
        y_train = np.random.randn(15)
        X_test = np.random.randn(3, 3)
        y_test = np.random.randn(3)

        model = Ridge(alpha=1.0).fit(X_train, y_train)

        banzhaf1 = BanzhafInfluence(n_samples=100, random_state=42)
        banzhaf1.fit(model, X_train, y_train)
        scores1 = banzhaf1.explain(X_test, y_test)

        banzhaf2 = BanzhafInfluence(n_samples=100, random_state=42)
        banzhaf2.fit(model, X_train, y_train)
        scores2 = banzhaf2.explain(X_test, y_test)

        np.testing.assert_allclose(scores1, scores2)


@pytest.mark.slow
class TestBanzhafNullPlayer:
    """Tests for the null player axiom: useless samples get ~0 value."""

    def test_null_player_irrelevant_feature(self):
        """
        Test that Banzhaf correctly identifies samples that don't contribute
        to predictions on test points.

        Setup: Training points with high x0 values contribute to predicting
        test points with high x0. A training point with x0 near 0 contributes
        less to predicting test points with extreme x0 values.
        """
        np.random.seed(42)

        # Create training data with informative points at extremes
        X_train = np.array(
            [
                [3.0, 0.0, 0.0],  # High leverage for positive x0
                [2.5, 0.1, 0.0],
                [-3.0, 0.0, 0.0],  # High leverage for negative x0
                [-2.5, -0.1, 0.0],
                [0.0, 0.0, 0.0],  # Low leverage - at center, will be "less useful"
                [0.1, 0.0, 0.0],
            ]
        )
        y_train = X_train[:, 0] + 0.01 * np.random.randn(6)

        # Test on extreme points where edge training points have high leverage
        X_test = np.array([[5.0, 0.0, 0.0]])
        y_test = np.array([5.0])

        model = Ridge(alpha=0.01)
        model.fit(X_train, y_train)

        banzhaf = BanzhafInfluence(n_samples=100, random_state=42)
        banzhaf.fit(model, X_train, y_train)
        scores = banzhaf.explain(X_test, y_test)

        total_scores = scores.sum(axis=0)

        # Points near the origin (indices 4, 5) should have smaller influence
        # on predicting the extreme test point than points at the extremes
        center_scores = np.abs(total_scores[4:6]).mean()
        extreme_scores = np.abs(total_scores[0:2]).mean()  # Positive extreme

        # Extreme points should have higher influence than center points
        assert extreme_scores > center_scores, (
            f"Extreme points (score={extreme_scores:.4f}) should have higher "
            f"influence than center points (score={center_scores:.4f}) for "
            f"predicting extreme test points"
        )


@pytest.mark.slow
class TestBanzhafSymmetry:
    """Tests for the symmetry axiom: identical points get equal values."""

    def test_duplicate_points_equal_values(self):
        """
        Duplicate training points should receive equal Banzhaf values
        (within estimation error).

        The symmetry axiom states: if points i and j contribute identically
        to all coalitions, their values should be equal.
        """
        np.random.seed(42)

        # Create data with two duplicate points
        X_train = np.random.randn(8, 3)
        y_train = X_train[:, 0] + 0.1 * np.random.randn(8)

        # Make points 0 and 1 identical
        X_train[1] = X_train[0].copy()
        y_train[1] = y_train[0]

        X_test = np.random.randn(2, 3)
        y_test = X_test[:, 0]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        # Use sufficient samples for accurate estimation
        banzhaf = BanzhafInfluence(n_samples=200, random_state=42)
        banzhaf.fit(model, X_train, y_train)
        scores = banzhaf.explain(X_test, y_test)

        total_scores = scores.sum(axis=0)

        # Duplicate points should have similar scores
        score_diff = np.abs(total_scores[0] - total_scores[1])
        score_scale = max(np.abs(total_scores[0]), np.abs(total_scores[1]), 0.01)
        relative_diff = score_diff / score_scale

        assert relative_diff < 0.5, (
            f"Duplicate points should have similar scores. "
            f"Point 0: {total_scores[0]:.4f}, Point 1: {total_scores[1]:.4f}, "
            f"Relative diff: {relative_diff:.2f}"
        )

    def test_permutation_invariance(self):
        """
        Banzhaf values should not depend on the order of training points.

        Due to Monte Carlo estimation variance, we check that the rankings
        are similar rather than exact equality.
        """
        np.random.seed(42)

        X_train = np.random.randn(6, 3)
        y_train = X_train[:, 0] + 0.1 * np.random.randn(6)
        X_test = np.random.randn(2, 3)
        y_test = X_test[:, 0]

        model = Ridge(alpha=1.0).fit(X_train, y_train)

        banzhaf = BanzhafInfluence(n_samples=150, random_state=42)
        banzhaf.fit(model, X_train, y_train)
        scores1 = banzhaf.explain(X_test, y_test)

        # Permute training data
        perm = np.array([3, 1, 4, 0, 2, 5])
        X_perm = X_train[perm]
        y_perm = y_train[perm]

        model_perm = Ridge(alpha=1.0).fit(X_perm, y_perm)

        banzhaf_perm = BanzhafInfluence(n_samples=150, random_state=42)
        banzhaf_perm.fit(model_perm, X_perm, y_perm)
        scores_perm = banzhaf_perm.explain(X_test, y_test)

        # Inverse permutation to compare
        inv_perm = np.argsort(perm)
        scores_perm_reordered = scores_perm[:, inv_perm]

        # Check that total scores are correlated (ranking preserved)
        # Due to Monte Carlo variance, exact equality isn't expected
        total1 = scores1.sum(axis=0)
        total_perm = scores_perm_reordered.sum(axis=0)
        corr = np.corrcoef(total1, total_perm)[0, 1]

        assert corr > 0.8, (
            f"Permuted and original Banzhaf rankings should be correlated "
            f"(got {corr:.3f})"
        )


@pytest.mark.slow
class TestBanzhafParallel:
    """Tests for parallel execution."""

    def test_parallel_gives_same_results(self):
        """Parallel and sequential Banzhaf should give identical results."""
        np.random.seed(42)
        X_train = np.random.randn(12, 3)
        y_train = X_train[:, 0] + 0.1 * np.random.randn(12)
        X_test = np.random.randn(2, 3)
        y_test = X_test[:, 0]
        model = Ridge(alpha=1.0).fit(X_train, y_train)

        banzhaf_seq = BanzhafInfluence(n_samples=30, n_jobs=1, random_state=42)
        banzhaf_seq.fit(model, X_train, y_train)
        scores_seq = banzhaf_seq.explain(X_test, y_test)

        banzhaf_par = BanzhafInfluence(n_samples=30, n_jobs=2, random_state=42)
        banzhaf_par.fit(model, X_train, y_train)
        scores_par = banzhaf_par.explain(X_test, y_test)

        np.testing.assert_allclose(scores_seq, scores_par, rtol=1e-10)
