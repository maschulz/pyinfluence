"""Tests for utility functions: top_influential, self_influence, and predict_proba fallback."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier

from pyinfluence import InfluenceFunctions
from pyinfluence._utils import (
    _compute_loss_sklearn,
    _get_prediction_value,
    _value_at_test,
    self_influence,
    top_influential,
)


class TestTopInfluential:
    """Tests for top_influential() function."""

    def test_1d_input_returns_correct_indices(self):
        """1D scores should return top-k helpful and harmful indices."""
        scores = np.array([0.5, -0.3, 0.8, -0.1, 0.2, -0.9, 0.1, 0.3])
        helpful, harmful = top_influential(scores, k=3)

        # Most helpful: indices with highest positive influence
        # scores[2]=0.8, scores[0]=0.5, scores[7]=0.3
        assert list(helpful) == [2, 0, 7]

        # Most harmful: indices with most negative influence
        # scores[5]=-0.9, scores[1]=-0.3, scores[3]=-0.1
        assert list(harmful) == [5, 1, 3]

    def test_2d_input_returns_per_test_indices(self):
        """2D scores should return (n_test, k) arrays."""
        # 3 test samples, 5 train samples
        scores = np.array(
            [
                [0.5, -0.3, 0.8, -0.1, 0.2],
                [-0.1, 0.9, 0.1, 0.3, -0.5],
                [0.4, 0.4, -0.2, 0.1, 0.3],
            ]
        )
        helpful, harmful = top_influential(scores, k=2)

        # Shape should be (n_test, k)
        assert helpful.shape == (3, 2)
        assert harmful.shape == (3, 2)

        # First test sample: most helpful are idx 2 (0.8), idx 0 (0.5)
        assert list(helpful[0]) == [2, 0]
        # First test sample: most harmful are idx 1 (-0.3), idx 3 (-0.1)
        assert list(harmful[0]) == [1, 3]

        # Second test sample: most helpful are idx 1 (0.9), idx 3 (0.3)
        assert list(helpful[1]) == [1, 3]
        # Second test sample: most harmful are idx 4 (-0.5), idx 0 (-0.1)
        assert list(harmful[1]) == [4, 0]

    def test_k_larger_than_samples_clips(self):
        """k > n_samples should return all samples sorted."""
        scores = np.array([0.3, -0.1, 0.5])
        helpful, harmful = top_influential(scores, k=10)

        # Should return all 3 samples, sorted
        assert len(helpful) == 3
        assert len(harmful) == 3
        # Helpful: idx 2 (0.5), idx 0 (0.3), idx 1 (-0.1)
        assert list(helpful) == [2, 0, 1]

    def test_default_k_is_10(self):
        """Default k should be 10."""
        scores = np.arange(20).astype(float)  # 20 samples
        helpful, harmful = top_influential(scores)

        assert len(helpful) == 10
        assert len(harmful) == 10

    def test_handles_ties_deterministically(self):
        """Tied scores should return consistent indices."""
        scores = np.array([0.5, 0.5, 0.5, -0.3, -0.3])
        helpful, harmful = top_influential(scores, k=3)

        # All three 0.5 values should be in helpful
        assert set(helpful) == {0, 1, 2}
        # Top 2 harmful should include the -0.3 values
        assert 3 in harmful and 4 in harmful

    def test_all_positive_scores(self):
        """When all scores are positive, harmful still returns least helpful."""
        scores = np.array([0.1, 0.5, 0.3, 0.8])
        helpful, harmful = top_influential(scores, k=2)

        # Most helpful: idx 3 (0.8), idx 1 (0.5)
        assert list(helpful) == [3, 1]
        # Least helpful (most harmful): idx 0 (0.1), idx 2 (0.3)
        assert list(harmful) == [0, 2]

    def test_all_negative_scores(self):
        """When all scores are negative, helpful still returns least harmful."""
        scores = np.array([-0.1, -0.5, -0.3, -0.8])
        helpful, harmful = top_influential(scores, k=2)

        # Least harmful (most helpful): idx 0 (-0.1), idx 2 (-0.3)
        assert list(helpful) == [0, 2]
        # Most harmful: idx 3 (-0.8), idx 1 (-0.5)
        assert list(harmful) == [3, 1]


class TestSelfInfluence:
    """Tests for self_influence() function."""

    def test_shape_is_n_train(self, regression_data):
        """Self-influence should have shape (n_train,)."""
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)

        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        attr.fit(model, X_train, y_train)

        scores = self_influence(attr)

        assert scores.shape == (X_train.shape[0],)

    def test_values_are_diagonal_of_full_matrix(self, regression_data):
        """Self-influence should equal diagonal of explain(X_train, y_train)."""
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)

        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        attr.fit(model, X_train, y_train)

        self_scores = self_influence(attr)
        full_scores = attr.explain(X_train, y_train)

        # Self-influence is the diagonal: scores[i, i]
        np.testing.assert_allclose(self_scores, np.diag(full_scores), rtol=1e-10)

    def test_works_with_binary_classification(self, binary_classification_data):
        """Self-influence should work with binary classification."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = LogisticRegression(C=1.0, max_iter=1000).fit(X_train, y_train)

        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        attr.fit(model, X_train, y_train)

        self_scores = self_influence(attr)

        assert self_scores.shape == (X_train.shape[0],)
        # Should be finite
        assert np.all(np.isfinite(self_scores))

    def test_prediction_mode(self, regression_data):
        """Self-influence should work in prediction mode."""
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)

        attr = InfluenceFunctions(damping=1e-5, mode="prediction")
        attr.fit(model, X_train, y_train)

        # For prediction mode, we don't need y_train
        self_scores = self_influence(attr)

        assert self_scores.shape == (X_train.shape[0],)

    def test_high_self_influence_on_outliers(self):
        """Outlier samples should have higher self-influence."""
        np.random.seed(42)
        # Normal data
        X_normal = np.random.randn(100, 5)
        y_normal = (
            X_normal @ np.array([1, 2, 0.5, -1, 0.3]) + np.random.randn(100) * 0.1
        )

        # Add outliers with very different target values
        X_outlier = np.random.randn(5, 5)
        y_outlier = np.random.randn(5) * 100  # Very different scale

        X = np.vstack([X_normal, X_outlier])
        y = np.concatenate([y_normal, y_outlier])

        model = Ridge(alpha=1.0).fit(X, y)
        attr = InfluenceFunctions(damping=1e-5, mode="loss").fit(model, X, y)

        self_scores = self_influence(attr)

        # Outliers should have higher absolute self-influence
        normal_influence = np.abs(self_scores[:100])
        outlier_influence = np.abs(self_scores[100:])

        # Mean outlier self-influence should be higher than mean normal
        assert np.mean(outlier_influence) > np.mean(normal_influence)

    def test_raises_on_unfitted_attributor(self, regression_data):
        """Should raise NotFittedError on unfitted attributor."""
        from sklearn.exceptions import NotFittedError

        attr = InfluenceFunctions(damping=1e-5)

        with pytest.raises(NotFittedError):
            self_influence(attr)

    @pytest.mark.parametrize("mode", ["loss", "prediction"])
    def test_self_influence_diag_matches_diagonal_and_self_influence(
        self, fitted_ridge, mode
    ):
        """_self_influence_diag() == diag(explain(X_train, y_train)), and
        self_influence(attr) uses the fast path (same values)."""
        model, X_train, y_train, X_test, y_test = fitted_ridge
        attr = InfluenceFunctions(damping=1e-5, mode=mode)
        attr.fit(model, X_train, y_train)

        diag_direct = attr._self_influence_diag()
        full = attr.explain(X_train, y_train if mode == "loss" else None)
        np.testing.assert_allclose(diag_direct, np.diag(full), equal_nan=True)
        np.testing.assert_allclose(self_influence(attr), diag_direct, equal_nan=True)


# =============================================================================
# Phase 7: Additional Utility Tests
# =============================================================================

from pyinfluence import LOOInfluence  # noqa: E402
from pyinfluence._utils import (  # noqa: E402
    aggregate_influence,
    compare_attributors,
    find_mislabeled,
    influence_by_group,
    influence_summary,
)


class TestInfluenceSummary:
    """Tests for influence_summary() function."""

    def test_returns_correct_statistics(self):
        """influence_summary should return dict with all expected statistics."""
        scores = np.array(
            [
                [0.5, -0.3, 0.8, -0.1, 0.2],
                [-0.1, 0.9, 0.1, 0.3, -0.5],
            ]
        )

        result = influence_summary(scores)

        # Check all expected keys are present
        expected_keys = {
            "mean",
            "std",
            "min",
            "max",
            "percentiles",
            "sparsity",
            "n_nan",
        }
        assert set(result.keys()) == expected_keys
        assert result["n_nan"] == 0

        # Check statistics are correct
        np.testing.assert_allclose(result["mean"], np.mean(scores))
        np.testing.assert_allclose(result["std"], np.std(scores))
        np.testing.assert_allclose(result["min"], np.min(scores))
        np.testing.assert_allclose(result["max"], np.max(scores))

    def test_percentiles_correct(self):
        """Percentiles should contain 25th, 50th, 75th by default."""
        scores = np.arange(100).astype(float)

        result = influence_summary(scores)

        # Default percentiles: 25, 50, 75
        assert 25 in result["percentiles"]
        assert 50 in result["percentiles"]
        assert 75 in result["percentiles"]

        np.testing.assert_allclose(result["percentiles"][50], np.median(scores))

    def test_sparsity_calculation(self):
        """Sparsity should be fraction of near-zero values."""
        # 7 near-zero values out of 10: 0.0 (5 times), 0.001, -0.001
        scores = np.array([0.0, 0.001, -0.001, 0.5, -0.3, 0.0, 0.0, 0.8, 0.0, 0.0])

        result = influence_summary(scores, zero_threshold=0.01)

        # 7 values within threshold of zero
        expected_sparsity = 7 / 10
        np.testing.assert_allclose(result["sparsity"], expected_sparsity)

    def test_1d_input(self):
        """Should work with 1D input."""
        scores = np.array([0.1, -0.2, 0.3, -0.4, 0.5])

        result = influence_summary(scores)

        np.testing.assert_allclose(result["mean"], np.mean(scores))
        np.testing.assert_allclose(result["min"], -0.4)
        np.testing.assert_allclose(result["max"], 0.5)

    def test_custom_percentiles(self):
        """Should support custom percentile values."""
        scores = np.arange(100).astype(float)

        result = influence_summary(scores, percentiles=[10, 90])

        assert 10 in result["percentiles"]
        assert 90 in result["percentiles"]
        assert 25 not in result["percentiles"]


class TestFindMislabeled:
    """Tests for find_mislabeled() function."""

    def test_identifies_injected_label_noise(self, regression_data):
        """Should identify samples with artificially flipped labels."""
        X_train, X_test, y_train, y_test = regression_data

        # Inject label noise: flip sign of target for first 5 samples
        y_noisy = y_train.copy()
        noise_indices = [0, 1, 2, 3, 4]
        y_noisy[noise_indices] = -y_noisy[noise_indices] * 10  # Extreme flip

        model = Ridge(alpha=1.0).fit(X_train, y_noisy)
        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        attr.fit(model, X_train, y_noisy)

        suspected = find_mislabeled(attr, threshold="auto")

        # At least some of the noisy samples should be detected
        detected_noise = set(suspected) & set(noise_indices)
        assert len(detected_noise) >= 2, (
            f"Expected >=2 noisy samples, got {detected_noise}"
        )

    def test_returns_indices_array(self, regression_data):
        """Should return array of indices."""
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        attr.fit(model, X_train, y_train)

        result = find_mislabeled(attr)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.intp or np.issubdtype(result.dtype, np.integer)

    def test_threshold_numeric(self, regression_data):
        """Should accept numeric threshold for z-score."""
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        attr.fit(model, X_train, y_train)

        # Higher threshold = fewer samples detected
        result_low = find_mislabeled(attr, threshold=1.0)
        result_high = find_mislabeled(attr, threshold=3.0)

        assert len(result_high) <= len(result_low)

    def test_threshold_nonpositive_raises(self, fitted_ridge):
        """threshold must be positive when numeric."""
        model, X_train, y_train, X_test, y_test = fitted_ridge
        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        attr.fit(model, X_train, y_train)
        with pytest.raises(ValueError, match="threshold must be positive"):
            find_mislabeled(attr, threshold=0)
        with pytest.raises(ValueError, match="threshold must be positive"):
            find_mislabeled(attr, threshold=-1.0)

    def test_clean_data_returns_few_samples(self, regression_data):
        """Clean data should have few/no suspected mislabeled samples."""
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        attr.fit(model, X_train, y_train)

        suspected = find_mislabeled(attr, threshold=3.0)

        # With clean data and high threshold, very few should be flagged
        assert len(suspected) < len(X_train) * 0.1

    def test_works_with_classification(self, binary_classification_data):
        """Should work with classification models."""
        X_train, X_test, y_train, y_test = binary_classification_data

        # Flip some labels
        y_noisy = y_train.copy()
        flip_idx = np.random.RandomState(42).choice(len(y_train), 5, replace=False)
        y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

        model = LogisticRegression(C=1.0, max_iter=1000).fit(X_train, y_noisy)
        attr = InfluenceFunctions(damping=1e-5, mode="loss")
        attr.fit(model, X_train, y_noisy)

        suspected = find_mislabeled(attr, threshold="auto")

        assert isinstance(suspected, np.ndarray)


class TestCompareAttributors:
    """Tests for compare_attributors() function."""

    def test_returns_correlation_metrics(self, regression_data):
        """Should return dict with pearson, spearman, and kendall correlations."""
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)

        attr1 = InfluenceFunctions(damping=1e-5, mode="loss")
        attr1.fit(model, X_train, y_train)

        attr2 = InfluenceFunctions(damping=1e-4, mode="loss")  # Different damping
        attr2.fit(model, X_train, y_train)

        result = compare_attributors(attr1, attr2, X_test, y_test)

        expected_keys = {"pearson", "spearman", "kendall", "top_k_overlap"}
        assert expected_keys <= set(result.keys())

    def test_identical_attributors_perfect_correlation(self, fitted_influence_ridge):
        """Comparing attributor to itself should give perfect correlation."""
        attr, _, _, _, X_test, y_test = fitted_influence_ridge
        result = compare_attributors(attr, attr, X_test, y_test)
        np.testing.assert_allclose(result["pearson"], 1.0, atol=1e-10)
        np.testing.assert_allclose(result["spearman"], 1.0, atol=1e-10)

    def test_top_k_overlap_jaccard(self, fitted_influence_ridge):
        """top_k_overlap should be Jaccard similarity of top-k sets."""
        attr, _, _, _, X_test, y_test = fitted_influence_ridge
        result = compare_attributors(attr, attr, X_test, y_test, k=10)
        np.testing.assert_allclose(result["top_k_overlap"], 1.0)

    def test_works_with_different_attributor_types(self, regression_data):
        """Should work comparing IF vs LOO."""
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)

        attr_if = InfluenceFunctions(damping=1e-5, mode="loss")
        attr_if.fit(model, X_train, y_train)

        attr_loo = LOOInfluence(mode="loss", n_jobs=1)
        attr_loo.fit(model, X_train, y_train)

        result = compare_attributors(attr_if, attr_loo, X_test[:5], y_test[:5])

        # Should return valid correlation values
        assert -1 <= result["pearson"] <= 1
        assert -1 <= result["spearman"] <= 1


# (scores, axis, method, expected) for aggregate_influence. Add rows when adding behaviors.
AGGREGATE_INFLUENCE_CASES = [
    # axis=0 sum: 3 test x 5 train
    (
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [0.5, 1.5, 2.5, 3.5, 4.5],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        ),
        0,
        "sum",
        np.array([1.5, 4.5, 7.5, 10.5, 13.5]),
    ),
    # axis=1 mean
    (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 1, "mean", np.array([2.0, 5.0])),
    # 2x2 sum
    (np.array([[1.0, 2.0], [3.0, 4.0]]), 0, "sum", np.array([4.0, 6.0])),
    # 2x2 mean
    (np.array([[1.0, 2.0], [3.0, 4.0]]), 0, "mean", np.array([2.0, 3.0])),
    # absmax
    (
        np.array([[1.0, -5.0, 2.0], [-3.0, 2.0, -1.0]]),
        0,
        "absmax",
        np.array([-3.0, -5.0, 2.0]),
    ),
    # 1d sum -> scalar
    (np.array([1.0, 2.0, 3.0]), 0, "sum", np.array(6.0)),
]


@pytest.mark.parametrize(
    "scores,axis,method,expected",
    AGGREGATE_INFLUENCE_CASES,
    ids=[
        "axis0_sum_3x5",
        "axis1_mean",
        "axis0_sum_2x2",
        "axis0_mean_2x2",
        "absmax",
        "1d_sum",
    ],
)
def test_aggregate_influence(scores, axis, method, expected):
    """aggregate_influence(axis=..., method=...) returns expected shape and values."""
    result = aggregate_influence(scores, axis=axis, method=method)
    assert result.shape == expected.shape, (result.shape, expected.shape)
    np.testing.assert_allclose(result, expected)


def test_aggregate_influence_invalid_method_raises():
    """Invalid method raises ValueError."""
    with pytest.raises(ValueError, match="method"):
        aggregate_influence(np.array([[1.0, 2.0]]), axis=0, method="invalid")


class TestInfluenceByGroup:
    """Tests for influence_by_group() function."""

    def test_returns_dict_mapping_groups(self):
        """Should return dict mapping group labels to aggregate influence."""
        scores = np.array([0.5, -0.3, 0.8, -0.1, 0.2])
        groups = np.array(["A", "A", "B", "B", "A"])

        result = influence_by_group(scores, groups)

        assert isinstance(result, dict)
        assert "A" in result
        assert "B" in result

    def test_aggregate_method_sum(self):
        """Default aggregation should sum scores within group."""
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        groups = np.array([0, 0, 1, 1])

        result = influence_by_group(scores, groups, method="sum")

        np.testing.assert_allclose(result[0], 3.0)  # 1 + 2
        np.testing.assert_allclose(result[1], 7.0)  # 3 + 4

    def test_aggregate_method_mean(self):
        """method='mean' should average scores within group."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        groups = np.array(["X", "X", "Y", "Y", "Y"])

        result = influence_by_group(scores, groups, method="mean")

        np.testing.assert_allclose(result["X"], 1.5)  # (1+2)/2
        np.testing.assert_allclose(result["Y"], 4.0)  # (3+4+5)/3

    def test_2d_scores_aggregates_correctly(self):
        """2D scores should aggregate each train sample, then group."""
        # 2 test samples, 4 train samples
        scores = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 1.5, 2.5, 3.5],
            ]
        )
        groups = np.array([0, 0, 1, 1])

        # First aggregate over test (axis=0), then group
        result = influence_by_group(scores, groups, method="sum")

        # Aggregate over test: [1.5, 3.5, 5.5, 7.5]
        # Then group: 0 -> 1.5+3.5=5.0, 1 -> 5.5+7.5=13.0
        np.testing.assert_allclose(result[0], 5.0)
        np.testing.assert_allclose(result[1], 13.0)

    def test_2d_scores_method_mean_matches_mean_then_mean(self):
        """method='mean' should be mean-over-tests then mean-over-group
        (not a sum/mean hybrid)."""
        scores = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 7.0, 8.0],
            ]
        )
        groups = np.array([0, 0, 1, 1])

        result = influence_by_group(scores, groups, method="mean")

        # Mean over test axis: [2.0, 3.0, 5.0, 6.0]; then mean within group.
        per_train_mean = scores.mean(axis=0)
        expected = {
            0: per_train_mean[groups == 0].mean(),
            1: per_train_mean[groups == 1].mean(),
        }
        np.testing.assert_allclose(result[0], expected[0])
        np.testing.assert_allclose(result[1], expected[1])

    def test_groups_with_single_member(self):
        """Groups with single members should work."""
        scores = np.array([1.0, 2.0, 3.0])
        groups = np.array(["A", "B", "C"])

        result = influence_by_group(scores, groups)

        assert len(result) == 3
        np.testing.assert_allclose(result["A"], 1.0)
        np.testing.assert_allclose(result["B"], 2.0)
        np.testing.assert_allclose(result["C"], 3.0)

    def test_integer_group_labels(self):
        """Should work with integer group labels."""
        scores = np.array([0.5, -0.3, 0.8])
        groups = np.array([1, 2, 1])

        result = influence_by_group(scores, groups)

        assert 1 in result
        assert 2 in result


# =============================================================================
# predict_proba fallback tests
# =============================================================================


class TestComputeLossSklearnFallback:
    """Tests for _compute_loss_sklearn decision_function fallback."""

    def test_classifier_with_predict_proba_uses_nll(self, binary_classification_data):
        """Classifier with predict_proba should use NLL (existing behavior)."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = LogisticRegression(C=1.0, random_state=42).fit(X_train, y_train)
        loss = _compute_loss_sklearn(model, X_test, y_test, is_classifier=True)
        assert loss.shape == (X_test.shape[0],)
        assert np.all(loss >= 0)  # NLL is non-negative

    def test_classifier_without_predict_proba_warns(self, binary_classification_data):
        """Classifier without predict_proba should warn and use squared error."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = RidgeClassifier(alpha=1.0).fit(X_train, y_train)
        assert not hasattr(model, "predict_proba") or not callable(
            getattr(model, "predict_proba", None)
        )
        with pytest.warns(UserWarning, match="does not expose predict_proba"):
            loss = _compute_loss_sklearn(model, X_test, y_test, is_classifier=True)
        assert loss.shape == (X_test.shape[0],)
        assert np.all(loss >= 0)  # Squared error is non-negative

    def test_classifier_without_predict_proba_uses_decision_function(
        self, binary_classification_data
    ):
        """Fallback uses HALF squared error on decision_function vs binarized y,
        matching the influence-function loss scale."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = RidgeClassifier(alpha=1.0).fit(X_train, y_train)
        with pytest.warns(UserWarning):
            loss = _compute_loss_sklearn(model, X_test, y_test, is_classifier=True)
        decision = model.decision_function(X_test)
        y_binary = np.where(y_test == model.classes_[1], 1.0, -1.0)
        expected = 0.5 * (y_binary - decision) ** 2
        np.testing.assert_allclose(loss, expected)

    def test_regressor_uses_half_squared_error(self, regression_data):
        """Regression loss is ½(y - ŷ)², matching InfluenceFunctions so refit
        and closed-form loss scores share one scale."""
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        loss = _compute_loss_sklearn(model, X_test, y_test, is_classifier=False)
        expected = 0.5 * (y_test - model.predict(X_test)) ** 2
        np.testing.assert_allclose(loss, expected)


class TestGetPredictionValueFallback:
    """Tests for _get_prediction_value decision_function fallback."""

    def test_with_predict_proba_returns_true_class_prob(
        self, binary_classification_data
    ):
        """Classifier with predict_proba returns P(true class)."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = LogisticRegression(C=1.0, random_state=42).fit(X_train, y_train)
        values = _get_prediction_value(model, X_test, y_test)
        assert values.shape == (X_test.shape[0],)
        assert np.all((values >= 0) & (values <= 1))

    def test_without_predict_proba_warns(self, binary_classification_data):
        """Classifier without predict_proba should warn."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = RidgeClassifier(alpha=1.0).fit(X_train, y_train)
        with pytest.warns(UserWarning, match="decision_function"):
            _get_prediction_value(model, X_test, y_test)

    def test_without_predict_proba_returns_true_class_margin(
        self, binary_classification_data
    ):
        """Fallback returns the decision_function margin toward the TRUE class:
        +decision for positive-class points, -decision for negative-class ones
        (mirroring the predict_proba path's probability of the true class)."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = RidgeClassifier(alpha=1.0).fit(X_train, y_train)
        with pytest.warns(UserWarning):
            values = _get_prediction_value(model, X_test, y_test)
        decision = model.decision_function(X_test).ravel()
        sign = np.where(y_test == model.classes_[1], 1.0, -1.0)
        np.testing.assert_allclose(values, sign * decision)


class TestValueAtTestFallback:
    """Tests for _value_at_test with classifiers lacking predict_proba."""

    def test_loss_mode_fallback(self, binary_classification_data):
        """Loss mode should fall back to squared error on decision_function."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = RidgeClassifier(alpha=1.0).fit(X_train, y_train)
        with pytest.warns(UserWarning, match="does not expose predict_proba"):
            values = _value_at_test(model, X_test, y_test, "loss", is_classifier=True)
        assert values.shape == (X_test.shape[0],)
        assert np.all(values >= 0)

    def test_prediction_mode_fallback(self, binary_classification_data):
        """Prediction mode returns the true-class decision_function margin."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = RidgeClassifier(alpha=1.0).fit(X_train, y_train)
        with pytest.warns(UserWarning, match="decision_function"):
            values = _value_at_test(
                model, X_test, y_test, "prediction", is_classifier=True
            )
        decision = model.decision_function(X_test).ravel()
        sign = np.where(y_test == model.classes_[1], 1.0, -1.0)
        np.testing.assert_allclose(values, sign * decision)

    def test_no_predict_proba_no_decision_function_raises(self):
        """Classifier with neither method should raise ValueError."""

        class BareClassifier:
            _estimator_type = "classifier"
            classes_ = np.array([0, 1])

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        model = BareClassifier()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        with pytest.raises(
            ValueError, match="neither predict_proba.*nor decision_function"
        ):
            _compute_loss_sklearn(model, X, y, is_classifier=True)


# =============================================================================
# removal_curve: mode requirement
# =============================================================================

from pyinfluence._utils import removal_curve  # noqa: E402


class TestRemovalCurveModeRequirement:
    def test_prediction_mode_attributor_raises(self, fitted_ridge):
        model, X_train, y_train, X_test, y_test = fitted_ridge
        attr = InfluenceFunctions(mode="prediction", damping=1e-5)
        attr.fit(model, X_train, y_train)
        with pytest.raises(ValueError, match="mode='loss'"):
            removal_curve(attr, X_test, y_test)


# =============================================================================
# stability_replicates
# =============================================================================

from pyinfluence._utils import stability_replicates  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    from pyinfluence import viz

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class TestStabilityReplicates:
    def test_shape_and_reproducible(self, fitted_ridge):
        model, X_train, y_train, X_test, y_test = fitted_ridge
        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model, X_train, y_train)

        reps1 = stability_replicates(
            attr, X_test, y_test, n_replicates=6, random_state=0
        )
        reps2 = stability_replicates(
            attr, X_test, y_test, n_replicates=6, random_state=0
        )

        assert reps1.shape == (6, X_train.shape[0])
        np.testing.assert_allclose(reps1, reps2)

    @pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
    def test_feeds_plot_top_k_stability_without_error(self, fitted_ridge):
        model, X_train, y_train, X_test, y_test = fitted_ridge
        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model, X_train, y_train)

        reps = stability_replicates(
            attr, X_test, y_test, n_replicates=5, random_state=0
        )
        fig, ax = viz.plot_top_k_stability(reps, k=5)
        assert fig is not None
        assert ax is not None
