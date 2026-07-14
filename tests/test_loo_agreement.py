"""Slope agreement between InfluenceFunctions and exact LOO retraining.

Correlation alone cannot catch magnitude errors (e.g. mis-scaled Hessian
regularization): predictions can correlate with retraining deltas while being
off by a constant factor. These tests regress the exact LOO deltas on the
influence predictions and require slope ~ 1.

The LOO refits adjust the sklearn regularization parameter so that the
per-sample-average regularization stays fixed when one sample is removed
(alpha -> alpha*(n-1)/n for Ridge, C -> C*n/(n-1) for LogisticRegression);
otherwise the removal effect is confounded with a regularization shift.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from pyinfluence import InfluenceFunctions

pytestmark = pytest.mark.slow


def _loo_deltas(model_factory, X, y, loss_fn, X_test, y_test):
    """Exact per-point removal effects on mean test loss."""
    n = len(y)
    base_model = model_factory(n).fit(X, y)
    base = loss_fn(base_model, X_test, y_test)
    deltas = np.empty(n)
    for j in range(n):
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        m = model_factory(n - 1).fit(X[mask], y[mask])
        deltas[j] = loss_fn(m, X_test, y_test) - base
    return deltas


def _slope_and_r(pred, true):
    slope = np.polyfit(pred, true, 1)[0]
    r = np.corrcoef(pred, true)[0, 1]
    return slope, r


def test_ridge_slope_agreement():
    rng = np.random.default_rng(42)
    n, p, alpha = 300, 10, 5.0
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + rng.normal(scale=1.0, size=n)
    X_test = rng.normal(size=(100, p))
    y_test = X_test @ beta + rng.normal(scale=1.0, size=100)

    model = Ridge(alpha=alpha).fit(X, y)
    attr = InfluenceFunctions(mode="loss", damping=1e-8).fit(model, X, y)
    scores = attr.explain(X_test, y_test)
    # package sign convention: positive = removal increases loss
    pred = scores.mean(axis=0)

    def loss_fn(m, Xt, yt):
        return 0.5 * np.mean((yt - m.predict(Xt)) ** 2)

    # keep per-sample-average regularization fixed: alpha_eff = alpha * m/n
    true = _loo_deltas(lambda m: Ridge(alpha=alpha * m / n), X, y, loss_fn,
                       X_test, y_test)

    slope, r = _slope_and_r(pred, true)
    assert r > 0.97, f"pearson {r:.3f} too low"
    assert 0.8 < slope < 1.25, f"slope {slope:.3f} not ~1 (Hessian mis-scaled?)"


def test_logistic_slope_agreement():
    rng = np.random.default_rng(0)
    n, p, C = 400, 8, 1.0
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    prob = 1.0 / (1.0 + np.exp(-X @ beta))
    y = (rng.uniform(size=n) < prob).astype(float)
    X_test = rng.normal(size=(150, p))
    y_test = (rng.uniform(size=150) < 1.0 / (1.0 + np.exp(-X_test @ beta))).astype(float)

    model = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    attr = InfluenceFunctions(mode="loss", damping=1e-8).fit(model, X, y)
    scores = attr.explain(X_test, y_test)
    pred = scores.mean(axis=0)

    def loss_fn(m, Xt, yt):
        q = np.clip(m.predict_proba(Xt)[:, 1], 1e-12, 1 - 1e-12)
        return float(-np.mean(yt * np.log(q) + (1 - yt) * np.log(1 - q)))

    # keep per-sample-average regularization fixed: C_eff = C * n/m
    true = _loo_deltas(
        lambda m: LogisticRegression(C=C * n / m, max_iter=5000),
        X, y, loss_fn, X_test, y_test,
    )

    slope, r = _slope_and_r(pred, true)
    assert r > 0.97, f"pearson {r:.3f} too low"
    assert 0.8 < slope < 1.25, f"slope {slope:.3f} not ~1 (Hessian mis-scaled?)"


def test_kernel_ridge_slope_agreement():
    from sklearn.kernel_ridge import KernelRidge

    rng = np.random.default_rng(7)
    n, p, alpha = 200, 5, 1.0
    X = rng.normal(size=(n, p))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=n)
    X_test = rng.normal(size=(80, p))
    y_test = np.sin(X_test[:, 0]) + 0.5 * X_test[:, 1] + rng.normal(scale=0.3, size=80)

    model = KernelRidge(alpha=alpha, kernel="rbf", gamma=0.2).fit(X, y)
    attr = InfluenceFunctions(mode="loss", damping=1e-10).fit(model, X, y)
    scores = attr.explain(X_test, y_test)
    pred = scores.mean(axis=0)

    def loss_fn(m, Xt, yt):
        return 0.5 * np.mean((yt - m.predict(Xt)) ** 2)

    # KernelRidge penalty lambda*a'Ka is defined on the function, not
    # per-sample; keep lambda fixed across refits scaled per sample count.
    true = _loo_deltas(
        lambda m: KernelRidge(alpha=alpha * m / n, kernel="rbf", gamma=0.2),
        X, y, loss_fn, X_test, y_test,
    )

    slope, r = _slope_and_r(pred, true)
    assert r > 0.95, f"pearson {r:.3f} too low"
    assert 0.7 < slope < 1.4, f"slope {slope:.3f} not ~1"
