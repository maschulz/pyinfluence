"""Engine-level tests for pyinfluence._functional (Functional,
FunctionalInfluence, RefitFunctionalInfluence, SubsampledFunctionalInfluence).

These exercise the generic engine directly (no fairness domain layer):
bare-callable score functionals, loss functionals, validation, the
functional=/target= explain overrides, and the identity-Hessian baseline.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pyinfluence import Functional, FunctionalInfluence, RefitFunctionalInfluence


@pytest.fixture(scope="module")
def small_logistic_problem():
    rng = np.random.default_rng(0)
    n, p, m = 100, 4, 60
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = (X @ beta + 0.3 * rng.normal(size=n) > 0).astype(int)
    Xa = rng.normal(size=(m, p))
    ya = (Xa @ beta + 0.3 * rng.normal(size=m) > 0).astype(int)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    return model, X, y, Xa, ya


# -----------------------------------------------------------------------------
# (a) bare-callable score functional: closed form vs refit
# -----------------------------------------------------------------------------


def test_bare_callable_score_functional_matches_refit(small_logistic_problem):
    model, X, y, Xa, ya = small_logistic_problem

    def mean_score(v, y):
        return float(np.mean(v))

    cf = FunctionalInfluence(mean_score).fit(model, X, y)
    s_cf = cf.explain(Xa)

    rf = RefitFunctionalInfluence(mean_score, verbose=0).fit(model, X, y)
    s_rf = rf.explain(Xa)

    ok = ~np.isnan(s_rf)
    r = np.corrcoef(s_cf[ok], s_rf[ok])[0, 1]
    assert r > 0.95, f"pearson {r:.3f} too low"


# -----------------------------------------------------------------------------
# (b) loss functional: closed form vs refit, and explain without y_ref raises
# -----------------------------------------------------------------------------


def test_loss_functional_matches_refit_and_requires_y_ref(small_logistic_problem):
    model, X, y, Xa, ya = small_logistic_problem
    F = Functional(fn=lambda v, y: float(np.mean(v)), of="losses")

    cf = FunctionalInfluence(F).fit(model, X, y)
    s_cf = cf.explain(Xa, ya)

    rf = RefitFunctionalInfluence(F, verbose=0).fit(model, X, y)
    s_rf = rf.explain(Xa, ya)

    ok = ~np.isnan(s_rf)
    r = np.corrcoef(s_cf[ok], s_rf[ok])[0, 1]
    assert r > 0.95, f"pearson {r:.3f} too low"

    with pytest.raises(ValueError, match="losses"):
        cf.explain(Xa)


# -----------------------------------------------------------------------------
# (c) Functional validation
# -----------------------------------------------------------------------------


def test_functional_invalid_of_raises():
    with pytest.raises(ValueError):
        Functional(fn=lambda v, y: 0.0, of="bogus")


def test_noncallable_functional_raises_type_error(small_logistic_problem):
    model, X, y, Xa, ya = small_logistic_problem
    with pytest.raises(TypeError, match="must be a Functional or a callable"):
        FunctionalInfluence(123).fit(model, X, y)


# -----------------------------------------------------------------------------
# (d) Refit explain functional= override
# -----------------------------------------------------------------------------


def test_refit_explain_functional_override_matches_dedicated(small_logistic_problem):
    model, X, y, Xa, ya = small_logistic_problem
    F1 = Functional(fn=lambda v, y: float(np.mean(v)), of="scores", name="mean")
    F2 = Functional(fn=lambda v, y: float(np.max(v)), of="scores", name="max")

    shared = RefitFunctionalInfluence(F1, verbose=0).fit(model, X, y)
    s1_override = shared.explain(Xa)
    s2_override = shared.explain(Xa, functional=F2)

    dedicated1 = RefitFunctionalInfluence(F1, verbose=0).fit(model, X, y)
    s1 = dedicated1.explain(Xa)
    dedicated2 = RefitFunctionalInfluence(F2, verbose=0).fit(model, X, y)
    s2 = dedicated2.explain(Xa)

    np.testing.assert_allclose(s1_override, s1, equal_nan=True)
    np.testing.assert_allclose(s2_override, s2, equal_nan=True)


# -----------------------------------------------------------------------------
# (e) invalid target raises
# -----------------------------------------------------------------------------


def test_invalid_target_raises(small_logistic_problem):
    model, X, y, Xa, ya = small_logistic_problem
    with pytest.raises(ValueError, match="target"):
        FunctionalInfluence(
            lambda v, y: float(np.mean(v)), target="bogus"
        ).fit(model, X, y)


# -----------------------------------------------------------------------------
# (f) hessian='identity' runs and differs from exact
# -----------------------------------------------------------------------------


def test_hessian_identity_runs_and_differs_from_exact(small_logistic_problem):
    model, X, y, Xa, ya = small_logistic_problem
    F = lambda v, y: float(np.mean(v))  # noqa: E731

    exact = FunctionalInfluence(F, hessian="exact").fit(model, X, y).explain(Xa)
    identity = (
        FunctionalInfluence(F, hessian="identity").fit(model, X, y).explain(Xa)
    )

    assert exact.shape == identity.shape == (len(y),)
    assert np.isfinite(identity).all()
    assert not np.allclose(exact, identity)


# -----------------------------------------------------------------------------
# (g) KernelRidge rejected by FunctionalInfluence.fit
# -----------------------------------------------------------------------------


def test_kernel_ridge_rejected():
    from sklearn.kernel_ridge import KernelRidge

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    model = KernelRidge(alpha=1.0, kernel="rbf").fit(X, y)
    with pytest.raises(ValueError, match="KernelRidge"):
        FunctionalInfluence(lambda v, y: float(np.mean(v))).fit(model, X, y)
