"""Regression guard for the loss-scale convention.

InfluenceFunctions differentiates the half-squared-error loss ½(y - ŷ)². The
refit-based attributors (LOO/Banzhaf/Bootstrap) measure loss with
_compute_loss_sklearn, which must use the same ½ convention. If it used the
full squared error, the two families would disagree by a factor of two in
magnitude while still correlating, so summed effects and cross-method
comparisons would be silently wrong.
"""

import numpy as np
from sklearn.linear_model import Ridge

from pyinfluence import InfluenceFunctions, LOOInfluence


def test_influence_and_loo_share_loss_scale():
    rng = np.random.default_rng(1)
    n, p, alpha = 150, 6, 2.0
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + rng.normal(scale=1.0, size=n)
    X_test = rng.normal(size=(50, p))
    y_test = X_test @ beta + rng.normal(scale=1.0, size=50)

    model = Ridge(alpha=alpha).fit(X, y)
    if_scores = (
        InfluenceFunctions(mode="loss", damping=1e-8)
        .fit(model, X, y)
        .explain(X_test, y_test)
        .mean(axis=0)
    )
    loo_scores = (
        LOOInfluence(mode="loss", verbose=0)
        .fit(model, X, y)
        .explain(X_test, y_test)
        .mean(axis=0)
    )

    r = float(np.corrcoef(if_scores, loo_scores)[0, 1])
    slope = float(np.polyfit(if_scores, loo_scores, 1)[0])
    assert r > 0.95, f"pearson {r:.3f}: IF and LOO loss scores should track"
    # slope ~1 under the shared ½ convention; the old bug gave ~2 (LOO full).
    assert 0.7 < slope < 1.4, f"slope {slope:.3f}: IF/LOO loss scales disagree"
