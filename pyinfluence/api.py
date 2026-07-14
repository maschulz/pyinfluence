"""High-level API: one-shot influence computation."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from pyinfluence._banzhaf import BanzhafInfluence
from pyinfluence._base import BaseAttributor
from pyinfluence._bootstrap import BootstrapInfluence
from pyinfluence._influence import InfluenceFunctions
from pyinfluence._loo import LOOInfluence
from pyinfluence._validation import (
    get_model_type,
    influence_function_incompatibility,
)

__all__ = ["influence"]


def influence(
    model: BaseEstimator,
    X_train: np.typing.ArrayLike,
    y_train: np.typing.ArrayLike,
    X_test: np.typing.ArrayLike,
    y_test: np.typing.ArrayLike | None = None,
    *,
    method: Literal["auto", "influence_functions", "loo", "banzhaf", "bootstrap"] = "auto",
    fallback: Literal["bootstrap", "loo", "banzhaf"] = "bootstrap",
    mode: Literal["loss", "prediction"] = "loss",
    return_attributor: bool = False,
    **kwargs,
) -> NDArray[np.floating] | tuple[NDArray[np.floating], BaseAttributor]:
    """
    Compute influence scores in one call, choosing the method from the model or explicitly.

    For ``method='auto'``, uses influence functions when the model is a
    supported linear model (Ridge, RidgeCV, LinearRegression,
    LogisticRegression, LogisticRegressionCV, RidgeClassifier,
    RidgeClassifierCV, or KernelRidge); otherwise uses the ``fallback``
    method (default ``'bootstrap'``; or ``'loo'`` / ``'banzhaf'``).

    Parameters
    ----------
    model : sklearn estimator
        A fitted model.
    X_train : array-like of shape (n_samples, n_features)
        Training features.
    y_train : array-like of shape (n_samples,)
        Training targets.
    X_test : array-like of shape (n_test, n_features)
        Test features to explain.
    y_test : array-like of shape (n_test,) or None, optional
        Test labels. Required when ``mode='loss'``. In ``mode='prediction'``,
        required for classifiers when the resolved method is a refit-based
        one (``loo``, ``banzhaf``, ``bootstrap``), which measure influence on
        the true-class score; not needed for regression or for
        ``InfluenceFunctions`` on binary classifiers.
    method : {'auto', 'influence_functions', 'loo', 'banzhaf', 'bootstrap'}, default='auto'
        Which attribution method to use. ``'auto'`` selects from the model type
        (influence functions for supported linear models, else ``fallback``).
    fallback : {'bootstrap', 'loo', 'banzhaf'}, default='bootstrap'
        Method used when ``method='auto'`` and the model is not a supported
        linear model.
    mode : {'loss', 'prediction'}, default='loss'
        What to measure influence on. ``'loss'`` requires ``y_test``.
    return_attributor : bool, default=False
        If False, return only the scores array. If True, return
        ``(scores, attributor)`` so you can call ``attributor.explain()`` again.
    **kwargs
        Passed to the attributor constructor (e.g. ``damping=1e-5`` for
        InfluenceFunctions, ``n_jobs=-1`` for LOO/Banzhaf).

    Returns
    -------
    scores : ndarray of shape (n_test, n_train)
        Influence of each training sample on each test sample.
        Positive = helpful (upweighting decreases test loss in loss mode).
    attributor : BaseAttributor, optional
        The fitted attributor, only when ``return_attributor=True``.

    Raises
    ------
    ValueError
        If ``mode='loss'`` and ``y_test`` is None, or if the model is
        unsupported and ``method='influence_functions'``.

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from pyinfluence import influence
    >>> X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> model = Ridge(alpha=1.0).fit(X_train, y_train)
    >>> scores = influence(model, X_train, y_train, X_test, y_test)
    >>> scores.shape
    (40, 160)
    """
    if mode == "loss" and y_test is None:
        raise ValueError(
            "y_test is required when mode='loss'. "
            "Pass test labels or use mode='prediction'."
        )

    # Early input validation: consistent lengths
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    X_test = np.asarray(X_test)
    if len(y_train) != len(X_train):
        raise ValueError(
            f"X_train and y_train must have the same length; "
            f"got X_train {X_train.shape[0]} and y_train {len(y_train)}."
        )
    if X_test.ndim == 1:
        n_test = 1
    else:
        n_test = X_test.shape[0]
    if mode == "loss" and y_test is not None:
        y_test_arr = np.asarray(y_test).ravel()
        if len(y_test_arr) != n_test:
            raise ValueError(
                f"X_test and y_test must have the same number of samples; "
                f"got X_test {n_test} and y_test {len(y_test_arr)}."
            )

    # Resolve which attributor class and constructor kwargs to use
    if method == "auto":
        model_type = get_model_type(model)
        # Binary classifiers: need classes_ with length 2
        if model_type in ("logistic", "ridge_classifier"):
            n_classes = len(getattr(model, "classes_", []))
            use_if = n_classes == 2
        else:
            use_if = model_type in ("ridge", "linear", "kernel_ridge")

        # A supported model type can still carry a configuration the
        # closed form cannot represent (class_weight, l1 penalty, ...);
        # fall back rather than produce silently degraded scores.
        if use_if:
            reason = influence_function_incompatibility(model)
            if reason is not None:
                warnings.warn(
                    f"method='auto' is falling back to '{fallback}' because "
                    f"{reason}.",
                    UserWarning,
                )
                use_if = False

        # Wrapped estimators (Pipeline, GridSearchCV, ...) are not unwrapped;
        # tell the user why a linear model inside one is not getting the
        # closed-form treatment.
        if model_type == "unsupported" and (
            hasattr(model, "steps") or hasattr(model, "best_estimator_")
        ):
            warnings.warn(
                f"{type(model).__name__} wraps its estimator, and method='auto' "
                "does not unwrap it; using the refit-based fallback "
                f"'{fallback}'. To use InfluenceFunctions on a wrapped linear "
                "model, pass the fitted inner estimator (e.g. "
                "pipeline[-1] or search.best_estimator_) with correspondingly "
                "transformed features.",
                UserWarning,
            )

        if use_if:
            attributor_cls = InfluenceFunctions
            attr_kwargs = {"mode": mode, **kwargs}
        else:
            if fallback == "bootstrap":
                attributor_cls = BootstrapInfluence
            elif fallback == "loo":
                attributor_cls = LOOInfluence
            elif fallback == "banzhaf":
                attributor_cls = BanzhafInfluence
            else:
                raise ValueError(
                    f"fallback must be 'bootstrap', 'loo', or 'banzhaf'; got {fallback!r}."
                )
            attr_kwargs = {"mode": mode, **kwargs}
    else:
        # Explicit method
        if method == "influence_functions":
            attributor_cls = InfluenceFunctions
        elif method == "loo":
            attributor_cls = LOOInfluence
        elif method == "banzhaf":
            attributor_cls = BanzhafInfluence
        elif method == "bootstrap":
            attributor_cls = BootstrapInfluence
        else:
            raise ValueError(
                f"method must be one of 'auto', 'influence_functions', 'loo', 'banzhaf', 'bootstrap'; got {method!r}."
            )
        attr_kwargs = {"mode": mode, **kwargs}

    attributor = attributor_cls(**attr_kwargs)
    attributor.fit(model, X_train, y_train)
    scores = attributor.explain(X_test, y_test)

    if return_attributor:
        return scores, attributor
    return scores
