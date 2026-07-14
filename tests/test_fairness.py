"""Tests for pyinfluence.fairness.

The load-bearing tests validate the closed-form disparity influence against
exact refitting (RefitFairnessInfluence), requiring correlation ~1 AND
slope ~1, per metric. Unit tests cover metric values, sign conventions, and
input validation.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from pyinfluence.fairness import (
    FairnessInfluenceFunctions,
    RefitFairnessInfluence,
    SubsampledFairnessInfluence,
    disparity_removal_curve,
    disparity_value,
    disparity_value_hard,
    group_removal_effect,
)


def _make_biased_classification(n=400, p=6, seed=0, gap_strength=1.0):
    """Synthetic binary task with a binary sensitive attribute and a real
    disparity: the sensitive group shifts a latent feature that also drives y.
    Returns train and audit splits."""
    rng = np.random.default_rng(seed)
    m = n  # audit same size
    total = n + m
    a = (rng.uniform(size=total) < 0.4).astype(float)
    X = rng.normal(size=(total, p))
    X[:, 0] += gap_strength * a  # proxy feature correlated with group
    logits = X @ rng.normal(size=p) * 0.8 + 0.5 * gap_strength * a - 0.2
    y = (rng.uniform(size=total) < 1 / (1 + np.exp(-logits))).astype(float)
    return (X[:n], y[:n], a[:n]), (X[n:], y[n:], a[n:])


@pytest.fixture(scope="module")
def biased_logistic():
    (Xtr, ytr, atr), (Xau, yau, aau) = _make_biased_classification()
    model = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, ytr)
    return model, Xtr, ytr, atr, Xau, yau, aau


# -----------------------------------------------------------------------------
# Metric values
# -----------------------------------------------------------------------------


class TestDisparityValue:
    def test_dp_sign_convention(self, biased_logistic):
        model, Xtr, ytr, atr, Xau, yau, aau = biased_logistic
        # group a=1 has higher positive rate by construction
        gap = disparity_value(model, Xau, aau, metric="dp")
        assert gap > 0

    def test_absolute_is_abs_of_signed(self, biased_logistic):
        model, *_, Xau, yau, aau = biased_logistic
        signed = disparity_value(model, Xau, aau, metric="dp", target="signed")
        absolute = disparity_value(model, Xau, aau, metric="dp", target="absolute")
        assert absolute == pytest.approx(abs(signed))

    def test_eopp_fpr_need_y(self, biased_logistic):
        model, *_, Xau, yau, aau = biased_logistic
        with pytest.raises(ValueError, match="y is required"):
            disparity_value(model, Xau, aau, metric="eopp")
        v = disparity_value(model, Xau, aau, y=yau, metric="eopp")
        assert np.isfinite(v)

    def test_worst_group_loss_multigroup(self, biased_logistic):
        model, *_, Xau, yau, aau = biased_logistic
        groups3 = (aau + (np.arange(len(aau)) % 2)).astype(int)  # 3 groups
        v = disparity_value(model, Xau, groups3, y=yau, metric="worst_group_loss")
        assert v > 0

    def test_nonbinary_sensitive_raises(self, biased_logistic):
        model, *_, Xau, yau, aau = biased_logistic
        bad = np.arange(len(aau)) % 3
        with pytest.raises(ValueError, match="binary"):
            disparity_value(model, Xau, bad, metric="dp")

    def test_hard_gap_in_range(self, biased_logistic):
        model, *_, Xau, yau, aau = biased_logistic
        v = disparity_value_hard(model, Xau, aau, y=yau, metric="eopp")
        assert -1.0 <= v <= 1.0

    def test_unknown_metric_raises(self, biased_logistic):
        model, *_, Xau, yau, aau = biased_logistic
        with pytest.raises(ValueError, match="Unknown metric"):
            disparity_value(model, Xau, aau, metric="parity")


# -----------------------------------------------------------------------------
# Closed form vs exact refitting (the load-bearing validation)
# -----------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("metric", ["dp", "eopp", "fpr", "worst_group_loss"])
def test_closed_form_matches_refit(biased_logistic, metric):
    model, Xtr, ytr, atr, Xau, yau, aau = biased_logistic
    n = len(ytr)

    attr = FairnessInfluenceFunctions(metric=metric, damping=1e-8)
    attr.fit(model, Xtr, ytr, sensitive=atr)
    pred = attr.explain(Xau, y_audit=yau, sensitive_audit=aau)

    exact = RefitFairnessInfluence(
        metric=metric,
        verbose=0,
        refit_factory=lambda m: LogisticRegression(C=1.0 * n / m, max_iter=5000),
    )
    exact.fit(model, Xtr, ytr)
    true = exact.explain(Xau, y_audit=yau, sensitive_audit=aau)

    ok = ~np.isnan(true)
    r = np.corrcoef(pred[ok], true[ok])[0, 1]
    slope = np.polyfit(pred[ok], true[ok], 1)[0]
    assert r > 0.95, f"{metric}: pearson {r:.3f} too low"
    assert 0.7 < slope < 1.4, f"{metric}: slope {slope:.3f} not ~1"


@pytest.mark.slow
def test_absolute_target_matches_refit(biased_logistic):
    model, Xtr, ytr, atr, Xau, yau, aau = biased_logistic
    n = len(ytr)
    attr = FairnessInfluenceFunctions(metric="dp", target="absolute", damping=1e-8)
    attr.fit(model, Xtr, ytr)
    pred = attr.explain(Xau, sensitive_audit=aau)
    exact = RefitFairnessInfluence(
        metric="dp", target="absolute", verbose=0,
        refit_factory=lambda m: LogisticRegression(C=1.0 * n / m, max_iter=5000),
    )
    exact.fit(model, Xtr, ytr)
    true = exact.explain(Xau, sensitive_audit=aau)
    ok = ~np.isnan(true)
    r = np.corrcoef(pred[ok], true[ok])[0, 1]
    assert r > 0.95


@pytest.mark.slow
def test_ridge_regression_mean_pred_gap():
    rng = np.random.default_rng(3)
    n, p = 300, 5
    a = (rng.uniform(size=2 * n) < 0.5).astype(float)
    X = rng.normal(size=(2 * n, p))
    y = X @ rng.normal(size=p) + 0.8 * a + rng.normal(scale=0.5, size=2 * n)
    Xtr, ytr, atr = X[:n], y[:n], a[:n]
    Xau, aau = X[n:], a[n:]
    alpha = 2.0
    model = Ridge(alpha=alpha).fit(Xtr, ytr)

    attr = FairnessInfluenceFunctions(metric="dp", damping=1e-10)
    attr.fit(model, Xtr, ytr)
    pred = attr.explain(Xau, sensitive_audit=aau)

    exact = RefitFairnessInfluence(
        metric="dp", verbose=0,
        refit_factory=lambda m: Ridge(alpha=alpha * m / n),
    )
    exact.fit(model, Xtr, ytr)
    true = exact.explain(Xau, sensitive_audit=aau)
    r = np.corrcoef(pred, true)[0, 1]
    slope = np.polyfit(pred, true, 1)[0]
    assert r > 0.97
    assert 0.8 < slope < 1.25


# -----------------------------------------------------------------------------
# Subsampled estimator: sign/rank agreement (looser; different estimand scale)
# -----------------------------------------------------------------------------


@pytest.mark.slow
def test_subsampled_rank_agreement(biased_logistic):
    model, Xtr, ytr, atr, Xau, yau, aau = biased_logistic
    attr = SubsampledFairnessInfluence(
        metric="dp", n_subsets=300, subset_frac=0.7, random_state=0, verbose=0
    )
    attr.fit(model, Xtr, ytr)
    mc = attr.explain(Xau, sensitive_audit=aau)

    cf = FairnessInfluenceFunctions(metric="dp", damping=1e-8)
    cf.fit(model, Xtr, ytr)
    pred = cf.explain(Xau, sensitive_audit=aau)

    from scipy.stats import spearmanr

    ok = ~np.isnan(mc)
    rho = spearmanr(mc[ok], pred[ok])[0]
    assert rho > 0.5, f"spearman {rho:.3f} too low"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


@pytest.mark.slow
def test_group_removal_effect_matches_sum_for_small_groups(biased_logistic):
    model, Xtr, ytr, atr, Xau, yau, aau = biased_logistic
    n = len(ytr)
    attr = FairnessInfluenceFunctions(metric="dp", damping=1e-8)
    attr.fit(model, Xtr, ytr)
    pred = attr.explain(Xau, sensitive_audit=aau)
    idx = np.argsort(-pred)[:5]
    actual = group_removal_effect(
        model, Xtr, ytr, idx, Xau, aau,
        refit_factory=lambda m: LogisticRegression(C=1.0 * n / m, max_iter=5000),
    )
    predicted_sum = pred[idx].sum()
    # small-group additivity should hold loosely
    assert np.sign(actual) == np.sign(predicted_sum)
    assert abs(actual - predicted_sum) < max(0.5 * abs(actual), 1e-3)


@pytest.mark.slow
def test_disparity_removal_curve_reduces_gap(biased_logistic):
    model, Xtr, ytr, atr, Xau, yau, aau = biased_logistic
    attr = FairnessInfluenceFunctions(metric="dp", target="absolute", damping=1e-8)
    attr.fit(model, Xtr, ytr)
    scores = attr.explain(Xau, sensitive_audit=aau)
    curve = disparity_removal_curve(
        scores, model, Xtr, ytr, Xau, aau, y_audit=yau,
        metric="dp", target="absolute",
        fractions=np.array([0.0, 0.05, 0.10]), n_random=3, random_state=0,
    )
    # removing the most gap-increasing points should shrink |gap| vs baseline
    assert curve["disparity"][-1] < curve["base_disparity"]
    assert curve["disparity"][-1] < curve["random_disparity_mean"][-1] + 1e-12
    assert np.isfinite(curve["accuracy"]).all()


class TestValidation:
    def test_explain_requires_sensitive(self, biased_logistic):
        model, Xtr, ytr, atr, Xau, yau, aau = biased_logistic
        attr = FairnessInfluenceFunctions(metric="dp").fit(model, Xtr, ytr)
        with pytest.raises(ValueError, match="sensitive_audit"):
            attr.explain(Xau)

    def test_eopp_requires_y_audit(self, biased_logistic):
        model, Xtr, ytr, atr, Xau, yau, aau = biased_logistic
        attr = FairnessInfluenceFunctions(metric="eopp").fit(model, Xtr, ytr)
        with pytest.raises(ValueError, match="y_audit"):
            attr.explain(Xau, sensitive_audit=aau)

    def test_kernel_ridge_rejected(self):
        from sklearn.kernel_ridge import KernelRidge

        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        y = rng.normal(size=50)
        model = KernelRidge(alpha=1.0, kernel="rbf").fit(X, y)
        with pytest.raises(ValueError, match="KernelRidge"):
            FairnessInfluenceFunctions().fit(model, X, y)

    def test_identity_hessian_runs(self, biased_logistic):
        model, Xtr, ytr, atr, Xau, yau, aau = biased_logistic
        attr = FairnessInfluenceFunctions(metric="dp", hessian="identity")
        attr.fit(model, Xtr, ytr)
        scores = attr.explain(Xau, sensitive_audit=aau)
        assert scores.shape == (len(ytr),)
        assert np.isfinite(scores).all()

    def test_scores_shape_and_finite(self, biased_logistic):
        model, Xtr, ytr, atr, Xau, yau, aau = biased_logistic
        attr = FairnessInfluenceFunctions(metric="dp").fit(model, Xtr, ytr)
        scores = attr.explain(Xau, sensitive_audit=aau)
        assert scores.shape == (len(ytr),)
        assert np.isfinite(scores).all()
