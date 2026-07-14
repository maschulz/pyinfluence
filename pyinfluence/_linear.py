"""Hessian and gradient computations for linear and kernel models."""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics.pairwise import pairwise_kernels

# Type alias for linear models
LinearModel = Union[Ridge, LinearRegression, LogisticRegression]

# Condition number threshold for warning
_CONDITION_NUMBER_THRESHOLD: float = 1e10

# Threshold for detecting near-separable data (p(1-p) near 0)
_SEPARABILITY_THRESHOLD: float = 1e-10


def _augment_intercept(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Augment feature matrix with a column of ones for intercept.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.

    Returns
    -------
    X_aug : ndarray of shape (n_samples, n_features + 1)
        Feature matrix with ones column appended.
    """
    n = X.shape[0]
    return np.hstack([X, np.ones((n, 1))])


def _get_params(model: LinearModel) -> NDArray[np.floating]:
    """
    Extract model parameters, concatenating intercept if present.

    Parameters
    ----------
    model : sklearn linear model
        Fitted model with coef_ and optionally intercept_.

    Returns
    -------
    theta : ndarray of shape (n_features,) or (n_features + 1,)
        Model parameters. If fit_intercept=True, intercept is appended.
    """
    coef = np.atleast_1d(model.coef_).ravel()
    if model.fit_intercept:
        intercept = np.atleast_1d(model.intercept_).ravel()
        return np.concatenate([coef, intercept])
    return coef


def _damping_matrix(
    p: int, damping: float, has_intercept: bool
) -> NDArray[np.floating]:
    """
    Create damping matrix with non-uniform damping for intercept.

    The intercept dimension receives minimal damping to maintain numerical
    stability without distorting influence (intercept is not regularized).

    Parameters
    ----------
    p : int
        Total number of parameters (features + 1 if intercept).
    damping : float
        Damping value for feature dimensions.
    has_intercept : bool
        Whether the last dimension is an intercept.

    Returns
    -------
    D : ndarray of shape (p, p)
        Diagonal damping matrix.
    """
    D = damping * np.eye(p)
    if has_intercept:
        # Minimal damping on intercept: small enough to not distort,
        # but non-zero for numerical stability
        D[-1, -1] = min(1e-10, damping * 1e-3)
    return D


def _hessian_ridge(
    X: NDArray[np.floating],
    reg_lambda: float,
    damping: float,
    has_intercept: bool,
) -> NDArray[np.floating]:
    """
    Compute Hessian matrix for Ridge regression.

    H = (1/n) X'X + diag([λ, ..., λ, 0]) + damping_matrix

    The intercept dimension (if present) is not regularized.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix. If has_intercept=True, this should already include
        the ones column from _augment_intercept().
    reg_lambda : float
        L2 regularization of the per-sample-average objective. For sklearn
        Ridge (total-loss objective with alpha) this is alpha/n.
    damping : float
        Damping value added to diagonal for numerical stability.
    has_intercept : bool
        Whether the last column of X is the intercept.

    Returns
    -------
    H : ndarray of shape (n_features, n_features)
        Hessian matrix.
    """
    n, p = X.shape

    # Regularization diagonal: lambda on features, 0 on intercept
    reg = np.full(p, reg_lambda)
    if has_intercept:
        reg[-1] = 0.0

    # H = X'X / n + regularization + damping
    H = X.T @ X / n + np.diag(reg)
    H += _damping_matrix(p, damping, has_intercept)

    return H


def _gradients_ridge(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    theta: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute per-sample gradients for Ridge regression.

    For per-example loss ℓ_i(θ) = (1/2)(y_i - x_i'θ)²:
    ∇ℓ_i = -(y_i - x_i'θ) x_i = (x_i'θ - y_i) x_i

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (augmented with intercept if applicable).
    y : ndarray of shape (n_samples,)
        Target values.
    theta : ndarray of shape (n_features,)
        Model parameters.

    Returns
    -------
    grads : ndarray of shape (n_samples, n_features)
        Per-sample gradients.
    """
    residuals = y - X @ theta
    return -X * residuals[:, np.newaxis]


def _get_params_logistic(
    model: LogisticRegression, class_idx: int = 0
) -> NDArray[np.floating]:
    """
    Extract model parameters for logistic regression, for a specific class.

    Parameters
    ----------
    model : sklearn LogisticRegression
        Fitted logistic regression model.
    class_idx : int, default=0
        For binary classification, use 0.

    Returns
    -------
    theta : ndarray of shape (n_features,) or (n_features + 1,)
        Model parameters. If fit_intercept=True, intercept is appended.
    """
    # For binary, coef_ is (1, n_features)
    if model.coef_.ndim == 1:
        coef = model.coef_
    else:
        coef = model.coef_[class_idx]

    if model.fit_intercept:
        intercept = np.atleast_1d(model.intercept_)
        if len(intercept) == 1 and class_idx == 0:
            return np.concatenate([coef, intercept])
        else:
            return np.concatenate([coef, [intercept[class_idx]]])
    return coef


def _hessian_logistic(
    X: NDArray[np.floating],
    probs: NDArray[np.floating],
    reg_lambda: float,
    damping: float,
    has_intercept: bool,
) -> NDArray[np.floating]:
    """
    Compute Hessian matrix for Logistic regression.

    H = (1/n) X' diag(p(1-p)) X + diag([λ, ..., λ, 0]) + damping_matrix

    The intercept dimension (if present) is not regularized.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix. If has_intercept=True, this should already include
        the ones column from _augment_intercept().
    probs : ndarray of shape (n_samples,)
        Predicted probabilities for the positive class.
    reg_lambda : float
        L2 regularization of the per-sample-average objective. For sklearn
        LogisticRegression (objective ||theta||^2/2 + C*sum nll_i) this is
        1/(C*n).
    damping : float
        Damping value added to diagonal for numerical stability.
    has_intercept : bool
        Whether the last column of X is the intercept.

    Returns
    -------
    H : ndarray of shape (n_features, n_features)
        Hessian matrix.

    Warns
    -----
    UserWarning
        If data appears near-separable (probabilities close to 0 or 1).
    """
    n, p = X.shape

    # Logistic Hessian weights: p(1-p)
    weights = probs * (1 - probs)

    # Check for near-separable data
    min_weight = np.min(weights)
    if min_weight < _SEPARABILITY_THRESHOLD:
        n_extreme = np.sum(weights < _SEPARABILITY_THRESHOLD)
        warnings.warn(
            f"Data appears near-separable: {n_extreme} samples have "
            f"p(1-p) < {_SEPARABILITY_THRESHOLD:.0e} (min={min_weight:.2e}). "
            "This can cause numerical instability in influence estimates. "
            "Consider increasing regularization (lower C) or damping.",
            UserWarning,
        )

    # Regularization diagonal: lambda on features, 0 on intercept
    reg = np.full(p, reg_lambda)
    if has_intercept:
        reg[-1] = 0.0

    # H = X' diag(w) X / n + regularization + damping
    H = X.T @ (X * weights[:, np.newaxis]) / n + np.diag(reg)
    H += _damping_matrix(p, damping, has_intercept)

    return H


def _gradients_logistic(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    probs: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute per-sample gradients for Logistic regression.

    For per-example negative log-likelihood:
    ∇ℓ_i = -(y_i - p_i) x_i

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (augmented with intercept if applicable).
    y : ndarray of shape (n_samples,)
        Binary labels (0 or 1).
    probs : ndarray of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    grads : ndarray of shape (n_samples, n_features)
        Per-sample gradients.
    """
    return -X * (y - probs)[:, np.newaxis]


def _invert_hessian(H: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Invert Hessian using Cholesky decomposition.

    Uses Cholesky for numerical stability on positive definite matrices.
    Warns if the Hessian is ill-conditioned.

    Parameters
    ----------
    H : ndarray of shape (p, p)
        Hessian matrix (should be positive definite).

    Returns
    -------
    H_inv : ndarray of shape (p, p)
        Inverse of Hessian.

    Warns
    -----
    UserWarning
        If the Hessian condition number exceeds the threshold.
    """
    # Check condition number before inversion
    cond = np.linalg.cond(H)
    if cond > _CONDITION_NUMBER_THRESHOLD:
        warnings.warn(
            f"Hessian condition number is {cond:.2e} "
            f"(threshold: {_CONDITION_NUMBER_THRESHOLD:.0e}). "
            "Influence estimates may be unstable. "
            "Consider increasing damping or regularization.",
            UserWarning,
        )

    p = H.shape[0]
    L = scipy.linalg.cholesky(H, lower=True)
    H_inv = scipy.linalg.cho_solve((L, True), np.eye(p))
    return H_inv


# -------------------------------------------------------------------------
# KernelRidge helpers (dual-space influence functions)
# -------------------------------------------------------------------------


def _compute_kernel_matrix(
    model: KernelRidge,
    X: NDArray[np.floating],
    Y: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """
    Compute kernel matrix using the fitted KernelRidge model's parameters.

    Parameters
    ----------
    model : KernelRidge
        Fitted KernelRidge model whose kernel, gamma, degree, coef0 are used.
    X : ndarray of shape (n_samples_X, n_features)
        First input array.
    Y : ndarray of shape (n_samples_Y, n_features), optional
        Second input array. If None, computes K(X, X).

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        Kernel matrix.
    """
    kernel = model.kernel
    params = {}
    if hasattr(model, "gamma") and model.gamma is not None:
        params["gamma"] = model.gamma
    if hasattr(model, "degree"):
        params["degree"] = model.degree
    if hasattr(model, "coef0"):
        params["coef0"] = model.coef0
    return pairwise_kernels(X, Y, metric=kernel, filter_params=True, **params)


def _get_dual_params(model: KernelRidge) -> NDArray[np.floating]:
    """
    Extract dual parameters from a fitted KernelRidge model.

    Parameters
    ----------
    model : KernelRidge
        Fitted KernelRidge model with dual_coef_.

    Returns
    -------
    alpha : ndarray of shape (n_samples,)
        Dual coefficients.
    """
    return np.atleast_1d(model.dual_coef_).ravel()
