"""Regression tests for fixes driven by the simulated-user usability round."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pyinfluence import (
    FunctionalInfluence,
    InfluenceFunctions,
    RefitFunctionalInfluence,
    influence,
    influence_by_group,
    stability_replicates,
)
from pyinfluence import functionals as fn
from pyinfluence.fairness import disparity


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4)) * np.array([1000.0, 1.0, 0.01, 1.0])
    y = (X[:, 1] + 0.3 * rng.normal(size=100) > 0).astype(int)
    return X, y


# -----------------------------------------------------------------------------
# Model-data mismatch guard (silent Pipeline-inner-estimator misuse)
# -----------------------------------------------------------------------------


def test_mismatch_guard_warns_on_raw_features(clf_data):
    """Inner estimator of a scaler pipeline + raw features must warn."""
    X, y = clf_data
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]).fit(X, y)
    inner = pipe.named_steps["clf"]
    with pytest.warns(UserWarning, match="trivial baseline"):
        InfluenceFunctions(mode="loss").fit(inner, X, y)


def test_mismatch_guard_silent_on_consistent_data(clf_data):
    X, y = clf_data
    Xs = StandardScaler().fit_transform(X)
    model = LogisticRegression(max_iter=1000).fit(Xs, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        InfluenceFunctions(mode="loss").fit(model, Xs, y)


def test_mismatch_guard_regression():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3)) * 100
    y = X[:, 0] + 0.1 * rng.normal(size=60)
    Xs = StandardScaler().fit_transform(X)
    model = Ridge(alpha=1.0).fit(Xs, y)
    with pytest.warns(UserWarning, match="R\\^2"):
        InfluenceFunctions(mode="loss").fit(model, X, y)


# -----------------------------------------------------------------------------
# Functional row-alignment guards
# -----------------------------------------------------------------------------


def test_functional_row_alignment_error():
    rng = np.random.default_rng(0)
    a = (rng.uniform(size=50) < 0.5).astype(int)
    F = disparity("dp", a)
    with pytest.raises(ValueError, match="row-aligned"):
        F(rng.uniform(size=40))


def test_auroc_y_length_mismatch():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="row-aligned labels"):
        fn.auroc(1)(rng.uniform(size=40), np.zeros(30))


# -----------------------------------------------------------------------------
# influence_by_group composite keys
# -----------------------------------------------------------------------------


def test_influence_by_group_rejects_tuple_keys():
    scores = np.arange(4.0)
    groups = [(0, 1), (0, 0), (1, 1), (1, 0)]
    with pytest.raises(ValueError, match="one-dimensional"):
        influence_by_group(scores, groups)


# -----------------------------------------------------------------------------
# viz guards (skip cleanly without matplotlib)
# -----------------------------------------------------------------------------

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")


def test_report_rejects_functional_attributor():
    from pyinfluence import viz

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    a = (rng.uniform(size=20) < 0.5).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    attr = FunctionalInfluence(disparity("dp", a)).fit(model, X, y)
    with pytest.raises(TypeError, match="per-test-point"):
        viz.report(attr, X[:20], y[:20])


def test_plot_top_influencers_1d_title():
    import matplotlib.pyplot as plt

    from pyinfluence import viz

    fig, ax = viz.plot_top_influencers(np.linspace(-1, 1, 30), k=3)
    assert ax.get_title() == "Top influencers"
    plt.close(fig)


# -----------------------------------------------------------------------------
# auto-fallback warning for unsupported model types
# -----------------------------------------------------------------------------


def test_auto_fallback_unsupported_type_warns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    with pytest.warns(UserWarning, match="fallback"):
        influence(
            model, X, y, X[:5], y[:5], method="auto",
            n_estimators=8, random_state=0, verbose=0,
        )


# -----------------------------------------------------------------------------
# Picklability of functionals and fitted attributors
# -----------------------------------------------------------------------------


def test_functionals_pickle_roundtrip():
    rng = np.random.default_rng(0)
    a = (rng.uniform(size=40) < 0.5).astype(int)
    scores = rng.uniform(size=40)
    ya = (rng.uniform(size=40) < 0.5).astype(int)
    for F in (
        fn.mean("scores"),
        fn.group_gap(a),
        fn.cohens_d(a),
        fn.worst_group_mean(a, of="scores"),
        fn.auroc(1),
        disparity("dp", a),
        disparity("eopp", a, pos_label=1),
    ):
        F2 = pickle.loads(pickle.dumps(F))
        assert F(scores, ya) == F2(scores, ya), F.name


def test_fitted_functional_attributor_pickle_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    a = (rng.uniform(size=25) < 0.5).astype(int)
    Xa = rng.normal(size=(25, 3))
    model = LogisticRegression(max_iter=1000).fit(X, y)
    attr = RefitFunctionalInfluence(disparity("dp", a), verbose=0).fit(model, X, y)
    attr2 = pickle.loads(pickle.dumps(attr))
    np.testing.assert_allclose(attr.explain(Xa), attr2.explain(Xa))


# -----------------------------------------------------------------------------
# stability_replicates warning dedup
# -----------------------------------------------------------------------------


def test_stability_replicates_dedupes_warnings():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(int)  # separable -> near-separability warnings
    model = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
    attr = InfluenceFunctions(mode="loss").fit(model, X, y)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        stability_replicates(attr, X[:10], y[:10], n_replicates=5, random_state=0)
    sep = [w for w in rec if "near-separable" in str(w.message)]
    assert len(sep) <= 1
    if sep:
        assert "across 5 replicates" in str(sep[0].message)
