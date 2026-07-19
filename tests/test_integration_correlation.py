"""Integration tests: correlation between attributor methods.

Parametrized over (method_a, method_b, fixture, threshold) so adding a new method
= add pairs to CORRELATION_PAIRS. All tests here are slow (refit-based methods).
"""

import numpy as np
import pytest

from pyinfluence import BanzhafInfluence, InfluenceFunctions, LOOInfluence

pytestmark = pytest.mark.slow


def _pair(cls_a, cls_b, fixture_name, threshold, kwargs_a=None, kwargs_b=None):
    return (cls_a, cls_b, fixture_name, threshold, kwargs_a or {}, kwargs_b or {})


# (AttributorClass_a, AttributorClass_b, fixture_name, min_correlation, kwargs_a, kwargs_b)
CORRELATION_PAIRS = [
    _pair(
        InfluenceFunctions,
        LOOInfluence,
        "fitted_ridge",
        0.9,
        {"damping": 1e-5, "mode": "loss"},
        {"mode": "loss", "n_jobs": -1},
    ),
    _pair(
        InfluenceFunctions,
        LOOInfluence,
        "fitted_logistic_binary",
        0.8,
        {"damping": 1e-5, "mode": "loss"},
        {"mode": "loss", "n_jobs": -1},
    ),
    _pair(
        InfluenceFunctions,
        BanzhafInfluence,
        "small_fitted_ridge",
        0.5,
        {"damping": 1e-5, "mode": "loss"},
        {"mode": "loss", "n_samples": 100, "random_state": 42},
    ),
    _pair(
        LOOInfluence,
        BanzhafInfluence,
        "small_fitted_ridge",
        0.5,
        {"mode": "loss", "n_jobs": -1},
        {"mode": "loss", "n_samples": 100, "random_state": 42},
    ),
]

CORRELATION_IDS = [f"{p[0].__name__}-{p[1].__name__}-{p[2]}" for p in CORRELATION_PAIRS]


@pytest.mark.parametrize("entry", CORRELATION_PAIRS, ids=CORRELATION_IDS)
def test_method_pair_correlation(entry, request):
    """Scores from two methods on the same data should be correlated above threshold."""
    cls_a, cls_b, fixture_name, threshold, kwargs_a, kwargs_b = entry
    model, X_train, y_train, X_test, y_test = request.getfixturevalue(fixture_name)

    attr_a = cls_a(**kwargs_a)
    attr_a.fit(model, X_train, y_train)
    scores_a = attr_a.explain(X_test, y_test)

    attr_b = cls_b(**kwargs_b)
    attr_b.fit(model, X_train, y_train)
    scores_b = attr_b.explain(X_test, y_test)

    corr = np.corrcoef(scores_a.ravel(), scores_b.ravel())[0, 1]
    assert corr > threshold, (
        f"{cls_a.__name__}-{cls_b.__name__} correlation {corr:.3f} is too low. "
        f"Expected > {threshold}."
    )


@pytest.mark.parametrize("entry", CORRELATION_PAIRS, ids=CORRELATION_IDS)
def test_method_pair_ranking_overlap(entry, request):
    """Top-k influential training samples should have significant overlap between methods."""
    cls_a, cls_b, fixture_name, _threshold, kwargs_a, kwargs_b = entry
    model, X_train, y_train, X_test, y_test = request.getfixturevalue(fixture_name)
    n_train = X_train.shape[0]
    k = min(10, n_train // 2, 8)  # smaller k for small_fitted_ridge

    attr_a = cls_a(**kwargs_a)
    attr_a.fit(model, X_train, y_train)
    scores_a = attr_a.explain(X_test, y_test)

    attr_b = cls_b(**kwargs_b)
    attr_b.fit(model, X_train, y_train)
    scores_b = attr_b.explain(X_test, y_test)

    total_a = scores_a.sum(axis=0)
    total_b = scores_b.sum(axis=0)
    top_a = set(np.argsort(total_a)[-k:])
    top_b = set(np.argsort(total_b)[-k:])
    overlap = len(top_a & top_b)
    min_overlap = max(1, k // 2)
    assert overlap >= min_overlap, (
        f"{cls_a.__name__}-{cls_b.__name__} top-{k} overlap is {overlap}. "
        f"Expected at least {min_overlap}."
    )
