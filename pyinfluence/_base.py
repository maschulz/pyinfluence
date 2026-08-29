"""Base class and shared helpers for training data attribution methods.

All attributors (InfluenceFunctions, LOOInfluence, BanzhafInfluence, BootstrapInfluence)
follow the same interface: fit(model, X, y) then explain(X_test, y_test).
Shared helpers: _validate_mode, _prepare_fit_inputs (for fit), _prepare_explain_inputs
(for explain) keep validation and array shaping in one place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

if TYPE_CHECKING:
    from typing import Self


class BaseAttributor(BaseEstimator, ABC):
    """
    Base class for training data attribution methods.

    All attributors inherit from this class and follow sklearn conventions.
    """

    @abstractmethod
    def fit(self, model: BaseEstimator, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Fit the attributor to a trained model and its training data.

        Parameters
        ----------
        model : sklearn estimator
            A fitted sklearn model.
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If model is unsupported type or not fitted.
        """

    @abstractmethod
    def explain(
        self, X_test: ArrayLike, y_test: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        """
        Compute influence scores of training samples on test samples.

        Parameters
        ----------
        X_test : array-like of shape (m_samples, n_features) or (n_features,)
            Test samples to explain. A single test point may be passed as a
            1D array; the result will have shape (1, n_train).
        y_test : array-like of shape (m_samples,) or scalar, optional
            Test labels. Required when mode='loss'. Ignored when mode='prediction'.
            A scalar is allowed when explaining a single test point.

        Returns
        -------
        scores : ndarray of shape (m_samples, n_samples)
            scores[i, j] = influence of training sample j on test sample i.
            For 1D X_test input, shape is (1, n_train).

            For mode='loss':
                Positive = helpful (upweighting decreases test loss).
                Negative = harmful (upweighting increases test loss).

            For mode='prediction':
                Positive = increases the predicted value (or true-class probability).
                Negative = decreases the predicted value (or true-class probability).

        Raises
        ------
        NotFittedError
            If called before fit().
        ValueError
            If mode='loss' and y_test is None.
        """


def _validate_mode(mode: str) -> None:
    """Raise ValueError if mode is not 'loss' or 'prediction'."""
    if mode not in ("loss", "prediction"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'loss' or 'prediction'.")


def _reject_sparse(X, name: str) -> None:
    """Raise a clear TypeError for scipy sparse inputs.

    np.asarray on a sparse matrix produces a 0-d object array, which would
    otherwise surface much later as a cryptic IndexError.
    """
    if scipy.sparse.issparse(X):
        raise TypeError(
            f"{name} is a scipy sparse matrix, which is not supported. "
            f"Densify first with {name}.toarray() (or .todense())."
        )


def _prepare_fit_inputs(
    X: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Convert X and y to numpy arrays; y is raveled. For use in fit().

    Rejects sparse X and multi-output y with clear errors.
    """
    _reject_sparse(X, "X")
    X = np.asarray(X)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        raise ValueError(
            f"Multi-output y (shape {y.shape}) is not supported. "
            "Fit one model per output and attribute each separately."
        )
    y = y.ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]} elements; "
            "X and y must have the same number of training examples."
        )
    return X, y


def _prepare_explain_inputs(
    X_test: ArrayLike,
    y_test: ArrayLike | None,
    needs_y_test: bool,
    *,
    error_msg: str | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
    """
    Normalize X_test and y_test for explain(), and validate y_test when required.

    Parameters
    ----------
    X_test : array-like
        Test samples (may be 1D for a single point).
    y_test : array-like or None
        Test labels.
    needs_y_test : bool
        If True, y_test must not be None.
    error_msg : str, optional
        Message for ValueError when needs_y_test and y_test is None.

    Returns
    -------
    X_test : ndarray of shape (m_samples, n_features)
    y_test : ndarray of shape (m_samples,) or None
    """
    _reject_sparse(X_test, "X_test")
    X_test = np.asarray(X_test)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)
    if y_test is not None:
        y_test = np.asarray(y_test)
        if y_test.ndim == 0:
            y_test = y_test.reshape(1)
        else:
            y_test = y_test.ravel()
        if y_test.shape[0] != X_test.shape[0]:
            raise ValueError(
                f"X_test has {X_test.shape[0]} rows but y_test has "
                f"{y_test.shape[0]} elements; they must align by test point."
            )
    if needs_y_test and y_test is None:
        raise ValueError(
            error_msg or "y_test is required for this mode. Provide y_test."
        )
    return X_test, y_test


def check_is_fitted(
    attributor: BaseAttributor, attributes: list[str] | None = None
) -> None:
    """
    Check if the attributor has been fitted.

    Parameters
    ----------
    attributor : BaseAttributor
        The attributor instance to check.
    attributes : list of str, optional
        List of attribute names to check. If None, checks for 'model_'.

    Raises
    ------
    NotFittedError
        If the attributor has not been fitted.
    """
    if attributes is None:
        attributes = ["model_"]

    for attr in attributes:
        if not hasattr(attributor, attr):
            raise NotFittedError(
                f"This {type(attributor).__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this method."
            )
