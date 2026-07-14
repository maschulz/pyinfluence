"""Shared test helpers for attributor tests.

Use assert_influence_scores_valid() whenever a test checks influence output from
explain(): shape (n_test, n_train), and optionally finite values and not all zero.
Use it in test_loo, test_banzhaf, test_bootstrap, test_influence, test_api, and
test_numerical for consistency. For Bootstrap, use check_finite=False when
some training points have few OOB runs (NaNs are expected).
"""

import numpy as np


def assert_influence_scores_valid(
    scores,
    n_test: int,
    n_train: int,
    *,
    check_finite: bool = True,
    check_not_all_zero: bool = True,
) -> None:
    """Assert that an influence score matrix has valid shape and values.

    Parameters
    ----------
    scores : ndarray
        Output of attributor.explain(X_test, y_test).
    n_test, n_train : int
        Expected dimensions.
    check_finite : bool, default=True
        If True, assert all values are finite (no NaN/inf).
    check_not_all_zero : bool, default=True
        If True, assert scores are not all close to zero.
    """
    assert scores.shape == (n_test, n_train), (
        f"Expected shape ({n_test}, {n_train}), got {scores.shape}"
    )
    if check_finite:
        assert np.all(np.isfinite(scores)), "Influence scores should all be finite"
    if check_not_all_zero:
        assert not np.allclose(scores, 0), "Influence scores should not be all zeros"
