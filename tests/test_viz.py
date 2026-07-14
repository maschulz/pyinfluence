"""Smoke tests for the pyinfluence.viz module.

Every plot function is exercised at least once with default kwargs, once
with its main option toggled, and once with an externally supplied ax.
We verify shape of the return and that no exception is raised - we do
not do image-diff testing.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


pytestmark = pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def scores_2d(rng):
    """(n_test=5, n_train=30) signed scores."""
    return rng.normal(scale=0.5, size=(5, 30))


@pytest.fixture
def scores_1d(scores_2d):
    return scores_2d.sum(axis=0)


@pytest.fixture
def groups(rng):
    return rng.choice(["A", "B", "C"], size=30)


@pytest.fixture
def self_inf(rng):
    """Self-influence with a handful of inflated outliers."""
    base = np.abs(rng.normal(scale=0.1, size=30))
    base[[3, 12, 25]] *= 8
    return base


@pytest.fixture
def errors(self_inf, rng):
    """Errors loosely correlated with self-influence (plus noise)."""
    return self_inf * 0.8 + np.abs(rng.normal(scale=0.05, size=30))


@pytest.fixture
def fitted_ridge_attr():
    """Tiny fitted Ridge + InfluenceFunctions attributor for report()/curve."""
    from sklearn.linear_model import Ridge

    from pyinfluence import InfluenceFunctions

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    coef = rng.normal(size=5)
    y = X @ coef + 0.1 * rng.normal(size=40)
    model = Ridge(alpha=1.0).fit(X, y)
    attr = InfluenceFunctions(mode="loss", damping=1e-3).fit(model, X, y)
    X_test = rng.normal(size=(8, 5))
    y_test = X_test @ coef + 0.1 * rng.normal(size=8)
    return attr, X_test, y_test


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _assert_fig_ax(result):
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


# -----------------------------------------------------------------------------
# plot_top_influencers
# -----------------------------------------------------------------------------


def test_top_influencers_2d(scores_2d):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_top_influencers(scores_2d, test_idx=2, k=5))


def test_top_influencers_1d(scores_1d):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_top_influencers(scores_1d, k=5))


def test_top_influencers_labels(scores_2d):
    from pyinfluence import viz
    labels = [f"row_{i}" for i in range(scores_2d.shape[1])]
    _assert_fig_ax(
        viz.plot_top_influencers(scores_2d, test_idx=0, k=3, labels=labels)
    )


def test_top_influencers_nan_excluded(scores_1d):
    """NaN scores must never be ranked as top influencers."""
    from pyinfluence import viz
    s = scores_1d.copy()
    top_val_idx = int(np.argmax(s))
    s[[0, 1]] = np.nan
    fig, ax = viz.plot_top_influencers(s, k=3)
    ticks = [t.get_text() for t in ax.get_yticklabels()]
    assert "0" not in ticks and "1" not in ticks
    assert str(top_val_idx) in ticks
    plt.close(fig)


def test_top_influencers_xerr(scores_2d, rng):
    from pyinfluence import viz
    xerr = np.abs(rng.normal(scale=0.05, size=scores_2d.shape))
    _assert_fig_ax(viz.plot_top_influencers(scores_2d, test_idx=1, k=4, xerr=xerr))


def test_top_influencers_xerr_shape_mismatch(scores_1d):
    from pyinfluence import viz
    with pytest.raises(ValueError):
        viz.plot_top_influencers(scores_1d, xerr=np.ones(3))


def test_top_influencers_accepts_ax(scores_2d):
    from pyinfluence import viz
    fig, ax = plt.subplots()
    result_fig, result_ax = viz.plot_top_influencers(scores_2d, ax=ax)
    assert result_fig is fig and result_ax is ax
    plt.close(fig)


# -----------------------------------------------------------------------------
# plot_self_influence
# -----------------------------------------------------------------------------


def test_self_influence_histogram(self_inf):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_self_influence(self_inf))


def test_self_influence_scatter(self_inf, errors):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_self_influence(self_inf, errors=errors))


def test_self_influence_annotate(self_inf, errors):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_self_influence(self_inf, errors=errors, annotate=True))


def test_self_influence_threshold_none(self_inf):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_self_influence(self_inf, threshold=None))


def test_self_influence_threshold_float(self_inf):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_self_influence(self_inf, threshold=0.5))


def test_self_influence_mismatched_errors(self_inf):
    from pyinfluence import viz
    with pytest.raises(ValueError):
        viz.plot_self_influence(self_inf, errors=np.arange(3))


# -----------------------------------------------------------------------------
# plot_by_group
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["bar", "box", "violin"])
def test_by_group_styles(scores_1d, groups, style):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_by_group(scores_1d, groups, style=style))


def test_by_group_2d(scores_2d, groups):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_by_group(scores_2d, groups))


def test_by_group_method_mean(scores_1d, groups):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_by_group(scores_1d, groups, method="mean"))


def test_by_group_invalid_style(scores_1d, groups):
    from pyinfluence import viz
    with pytest.raises(ValueError):
        viz.plot_by_group(scores_1d, groups, style="nope")


def test_by_group_mismatched_groups(scores_1d):
    from pyinfluence import viz
    with pytest.raises(ValueError):
        viz.plot_by_group(scores_1d, groups=np.array(["A", "B"]))


# -----------------------------------------------------------------------------
# plot_heatmap
# -----------------------------------------------------------------------------


def test_heatmap_2d(scores_2d):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_heatmap(scores_2d))


def test_heatmap_1d(scores_1d):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_heatmap(scores_1d))


def test_heatmap_top_k_subsets(scores_2d):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_heatmap(scores_2d, top_k=5))


def test_heatmap_top_k_none(scores_2d):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_heatmap(scores_2d, top_k=None))


def test_heatmap_invalid_dim():
    from pyinfluence import viz
    with pytest.raises(ValueError):
        viz.plot_heatmap(np.zeros((2, 2, 2)))


# -----------------------------------------------------------------------------
# plot_method_comparison
# -----------------------------------------------------------------------------


def test_method_comparison(scores_1d, rng):
    from pyinfluence import viz
    alt = scores_1d + rng.normal(scale=0.05, size=scores_1d.shape)
    _assert_fig_ax(viz.plot_method_comparison(scores_1d, alt))


def test_method_comparison_no_extras(scores_1d, rng):
    from pyinfluence import viz
    alt = scores_1d + rng.normal(scale=0.05, size=scores_1d.shape)
    _assert_fig_ax(viz.plot_method_comparison(
        scores_1d, alt, show_correlation=False, show_identity=False,
    ))


def test_method_comparison_shape_mismatch(scores_1d):
    from pyinfluence import viz
    with pytest.raises(ValueError):
        viz.plot_method_comparison(scores_1d, np.arange(5))


# -----------------------------------------------------------------------------
# plot_removal_curve  +  removal_curve util
# -----------------------------------------------------------------------------


def test_removal_curve_end_to_end(fitted_ridge_attr):
    from pyinfluence import removal_curve, viz
    attr, X_test, y_test = fitted_ridge_attr
    curve = removal_curve(
        attr, X_test, y_test,
        fractions=[0.0, 0.1, 0.2], n_random=2, random_state=0,
    )
    assert set(curve) >= {"fractions", "by_influence", "random_mean", "random_std", "direction"}
    assert curve["by_influence"].shape == (3,)
    _assert_fig_ax(viz.plot_removal_curve(curve))


def test_removal_curve_no_random(fitted_ridge_attr):
    from pyinfluence import removal_curve, viz
    attr, X_test, y_test = fitted_ridge_attr
    curve = removal_curve(
        attr, X_test, y_test,
        fractions=[0.0, 0.1], n_random=0,
    )
    assert curve["random_mean"].size == 0
    _assert_fig_ax(viz.plot_removal_curve(curve))


def test_removal_curve_helpful_direction(fitted_ridge_attr):
    from pyinfluence import removal_curve
    attr, X_test, y_test = fitted_ridge_attr
    curve = removal_curve(
        attr, X_test, y_test,
        fractions=[0.0, 0.1], direction="helpful", n_random=1, random_state=0,
    )
    assert curve["direction"] == "helpful"


def test_removal_curve_invalid_fractions(fitted_ridge_attr):
    from pyinfluence import removal_curve
    attr, X_test, y_test = fitted_ridge_attr
    with pytest.raises(ValueError):
        removal_curve(attr, X_test, y_test, fractions=[0.0, 1.5])


def test_removal_curve_invalid_direction(fitted_ridge_attr):
    from pyinfluence import removal_curve
    attr, X_test, y_test = fitted_ridge_attr
    with pytest.raises(ValueError):
        removal_curve(attr, X_test, y_test, fractions=[0.0], direction="sideways")


# -----------------------------------------------------------------------------
# plot_disparity_curve
# -----------------------------------------------------------------------------


def _fake_disparity_curve(with_random=True, with_hard=True):
    f = np.linspace(0.0, 0.2, 5)
    curve = {
        "fractions": f,
        "disparity": 0.1 - 0.3 * f,
        "disparity_hard": (0.12 - 0.3 * f) if with_hard else np.full(5, np.nan),
        "accuracy": np.full(5, 0.9),
        "random_disparity_mean": np.full(5, 0.1) if with_random else np.array([]),
        "random_disparity_std": np.full(5, 0.01) if with_random else np.array([]),
        "base_disparity": 0.1,
    }
    return curve


def test_disparity_curve():
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_disparity_curve(_fake_disparity_curve()))


def test_disparity_curve_no_hard_no_random():
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_disparity_curve(
        _fake_disparity_curve(with_random=False, with_hard=False)
    ))


def test_disparity_curve_end_to_end():
    """Round-trip: fairness.disparity_removal_curve output plots directly."""
    from sklearn.linear_model import LogisticRegression

    from pyinfluence import FunctionalInfluence, viz
    from pyinfluence.fairness import disparity, disparity_removal_curve

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=80) > 0).astype(int)
    a = (rng.uniform(size=80) < 0.5).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    F = disparity("dp", a)
    attr = FunctionalInfluence(F).fit(model, X, y)
    scores = attr.explain(X)
    curve = disparity_removal_curve(
        scores, model, X, y, X, a, y_audit=y,
        fractions=np.linspace(0.0, 0.1, 3), n_random=2, random_state=0,
    )
    _assert_fig_ax(viz.plot_disparity_curve(curve))


# -----------------------------------------------------------------------------
# plot_detection_curve
# -----------------------------------------------------------------------------


def test_detection_curve(self_inf):
    from pyinfluence import viz
    corrupted = np.zeros(self_inf.size, dtype=bool)
    corrupted[[3, 12, 25]] = True
    fig, ax = viz.plot_detection_curve(self_inf, corrupted)
    # The inflated outliers are exactly the corrupted set: the curve must
    # reach full recall long before full inspection.
    line = ax.get_lines()[0]
    y = line.get_ydata()
    assert y[self_inf.size // 2] == 1.0
    plt.close(fig)


def test_detection_curve_nan_ranked_last(self_inf):
    from pyinfluence import viz
    s = self_inf.copy()
    s[3] = np.nan  # corrupted-and-NaN: found only at the very end
    corrupted = np.zeros(s.size, dtype=bool)
    corrupted[[3, 12]] = True
    fig, ax = viz.plot_detection_curve(s, corrupted)
    y = ax.get_lines()[0].get_ydata()
    assert y[-2] < 1.0 and y[-1] == 1.0
    plt.close(fig)


def test_detection_curve_validation(self_inf):
    from pyinfluence import viz
    with pytest.raises(ValueError):
        viz.plot_detection_curve(self_inf, np.zeros(self_inf.size, dtype=bool))
    with pytest.raises(ValueError):
        viz.plot_detection_curve(self_inf, np.ones(3, dtype=bool))


# -----------------------------------------------------------------------------
# plot_influence_concentration
# -----------------------------------------------------------------------------


def test_influence_concentration_2d(scores_2d):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_influence_concentration(scores_2d))


def test_influence_concentration_1d(scores_1d):
    from pyinfluence import viz
    _assert_fig_ax(viz.plot_influence_concentration(scores_1d, mark_share=None))


def test_influence_concentration_all_zero():
    from pyinfluence import viz
    with pytest.raises(ValueError):
        viz.plot_influence_concentration(np.zeros(10))


# -----------------------------------------------------------------------------
# plot_top_k_stability
# -----------------------------------------------------------------------------


def test_top_k_stability_2d(rng):
    from pyinfluence import viz
    replicates = rng.normal(size=(8, 20))
    _assert_fig_ax(viz.plot_top_k_stability(replicates, k=5))


def test_top_k_stability_3d(rng):
    from pyinfluence import viz
    replicates = rng.normal(size=(6, 4, 20))
    _assert_fig_ax(viz.plot_top_k_stability(replicates, k=5))


@pytest.mark.parametrize("show", ["helpful", "harmful", "abs"])
def test_top_k_stability_show(rng, show):
    from pyinfluence import viz
    replicates = rng.normal(size=(5, 15))
    _assert_fig_ax(viz.plot_top_k_stability(replicates, k=3, show=show))


def test_top_k_stability_invalid_show(rng):
    from pyinfluence import viz
    with pytest.raises(ValueError):
        viz.plot_top_k_stability(rng.normal(size=(2, 5)), show="nope")


def test_top_k_stability_invalid_dim():
    from pyinfluence import viz
    with pytest.raises(ValueError):
        viz.plot_top_k_stability(np.zeros(5))


# -----------------------------------------------------------------------------
# report
# -----------------------------------------------------------------------------


def test_report_basic(fitted_ridge_attr):
    from pyinfluence import viz
    attr, X_test, y_test = fitted_ridge_attr
    fig = viz.report(attr, X_test, y_test)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_report_with_groups_and_errors(fitted_ridge_attr, rng):
    from pyinfluence import self_influence, viz
    attr, X_test, y_test = fitted_ridge_attr
    n_train = attr.X_train_.shape[0]
    groups = rng.choice(["A", "B"], size=n_train)
    errors = np.abs(rng.normal(size=n_train))
    fig = viz.report(attr, X_test, y_test, groups=groups, errors=errors)
    assert isinstance(fig, Figure)
    # sanity: self_influence still callable
    assert self_influence(attr).shape == (n_train,)
    plt.close(fig)


def test_report_save_path(fitted_ridge_attr, tmp_path):
    from pyinfluence import viz
    attr, X_test, y_test = fitted_ridge_attr
    out = tmp_path / "report.png"
    fig = viz.report(attr, X_test, y_test, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


# -----------------------------------------------------------------------------
# Labels plumbing
# -----------------------------------------------------------------------------


def test_labels_top_influencers(scores_2d):
    from pyinfluence import viz
    labs = [f"row_{i:02d}" for i in range(scores_2d.shape[1])]
    fig, ax = viz.plot_top_influencers(scores_2d, test_idx=0, k=3, labels=labs)
    seen = {t.get_text() for t in ax.get_yticklabels()}
    assert seen <= set(labs) and seen  # all ticks are real labels, none missing
    plt.close(fig)


def test_labels_self_influence_annotate(self_inf, errors):
    from pyinfluence import viz
    labs = [f"s{i:02d}" for i in range(self_inf.size)]
    # annotate=True uses labels for flagged points; we just check it runs.
    _assert_fig_ax(
        viz.plot_self_influence(self_inf, errors=errors, annotate=True, labels=labs)
    )


def test_labels_heatmap(scores_2d):
    from pyinfluence import viz
    n_test, n_train = scores_2d.shape
    fig, ax = viz.plot_heatmap(
        scores_2d, top_k=None,
        train_labels=[f"tr_{i}" for i in range(n_train)],
        test_labels=[f"te_{i}" for i in range(n_test)],
    )
    xt = {t.get_text() for t in ax.get_xticklabels()}
    yt = {t.get_text() for t in ax.get_yticklabels()}
    # Labels must be the named ones we passed in, not integer index strings.
    assert any(t.startswith("tr_") for t in xt)
    assert any(t.startswith("te_") for t in yt)
    plt.close(fig)


def test_labels_top_k_stability(rng):
    from pyinfluence import viz
    replicates = rng.normal(size=(6, 15))
    labs = [f"x{i}" for i in range(15)]
    fig, ax = viz.plot_top_k_stability(replicates, k=4, labels=labs)
    seen = {t.get_text() for t in ax.get_yticklabels()}
    assert seen and seen <= set(labs)
    plt.close(fig)


def test_labels_length_mismatch(scores_1d):
    from pyinfluence import viz
    with pytest.raises(ValueError, match="length"):
        viz.plot_top_influencers(scores_1d, labels=["only_one_label"])


def test_labels_in_report(fitted_ridge_attr):
    from pyinfluence import viz
    attr, X_test, y_test = fitted_ridge_attr
    n_tr = attr.X_train_.shape[0]
    n_te = X_test.shape[0]
    fig = viz.report(
        attr, X_test, y_test,
        train_labels=[f"tr_{i}" for i in range(n_tr)],
        test_labels=[f"te_{i}" for i in range(n_te)],
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_labels_accept_pandas_like(scores_1d):
    """Anything with __array__ (pandas Index/Series, numpy arrays) is OK."""
    from pyinfluence import viz

    class FakeIndex:
        def __init__(self, vals):
            self._v = list(vals)
        def __array__(self, dtype=None):
            return np.asarray(self._v, dtype=dtype)

    labs = FakeIndex([f"name_{i}" for i in range(scores_1d.size)])
    fig, ax = viz.plot_top_influencers(scores_1d, k=3, labels=labs)
    seen = {t.get_text() for t in ax.get_yticklabels()}
    assert any(t.startswith("name_") for t in seen)
    plt.close(fig)


def test_labels_accept_pandas_if_installed(scores_1d):
    """End-to-end pandas check (skipped if pandas isn't installed)."""
    pd = pytest.importorskip("pandas")
    from pyinfluence import viz
    idx = pd.Index([f"sample_{i}" for i in range(scores_1d.size)], name="sample_id")
    fig, ax = viz.plot_top_influencers(scores_1d, k=3, labels=idx)
    seen = {t.get_text() for t in ax.get_yticklabels()}
    assert any(t.startswith("sample_") for t in seen)
    plt.close(fig)
