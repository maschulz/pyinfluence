"""Leave-One-Out influence implementation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone, is_classifier
from tqdm import tqdm

from pyinfluence._base import (
    BaseAttributor,
    check_is_fitted,
    _prepare_explain_inputs,
    _prepare_fit_inputs,
    _validate_mode,
)
from pyinfluence._utils import tqdm_joblib, _value_at_test

if TYPE_CHECKING:
    from typing import Self


def _fit_loo_model(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    idx: int,
) -> BaseEstimator | None:
    """
    Fit a model with one sample removed.

    Parameters
    ----------
    model : sklearn estimator
        The model to clone and refit.
    X : ndarray
        Training features.
    y : ndarray
        Training targets.
    idx : int
        Index of sample to remove.

    Returns
    -------
    model_loo : sklearn estimator or None
        The refitted model, or None if fitting failed.
    """
    mask = np.ones(len(y), dtype=bool)
    mask[idx] = False
    model_loo = clone(model)
    try:
        model_loo.fit(X[mask], y[mask])
        return model_loo
    except Exception:
        # Fitting failed (e.g., class imbalance after removal)
        return None


class LOOInfluence(BaseAttributor):
    r"""
    Leave-one-out influence via retraining.

    For each training sample, retrains the model without it and measures
    the change in prediction/loss on test samples.

    Parameters
    ----------
    mode : {'loss', 'prediction'}, default='loss'
        How to measure influence:

        - 'loss': influence = L(x; D \ z_i) - L(x; D). Works for both
          classification and regression. Requires y_test.
          Positive = removing sample increases loss = sample is helpful.

        - 'prediction': For classifiers, influence = P(y|x; D) - P(y|x; D \ z_i)
          (requires y_test to identify the true class). For regressors,
          influence = f(x; D) - f(x; D \ z_i), the change in the raw prediction.
          Positive values mean the sample increases the prediction or
          true-class probability.

    n_jobs : int, default=None
        Number of parallel retraining jobs. None means sequential.
        -1 means use all available cores.

    verbose : int, default=1
        If 1, show progress bars during fit and explain. If 0, no progress output.

    Attributes
    ----------
    model_ : sklearn estimator
        Reference to the original fitted model.
    X_train_ : ndarray
        Training features.
    y_train_ : ndarray
        Training targets.
    loo_models_ : list
        List of LOO-refitted models. None entries indicate failed refits.
    is_classifier_ : bool
        Whether the model is a classifier.
    failed_indices_ : list of int
        Indices where LOO refit failed.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> attr = LOOInfluence(mode='loss', n_jobs=-1).fit(model, X_train, y_train)
    >>> scores = attr.explain(X_test, y_test)
    """

    def __init__(
        self,
        mode: Literal["loss", "prediction"] = "loss",
        n_jobs: int | None = None,
        verbose: int = 1,
    ) -> None:
        self.mode = mode
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, model: BaseEstimator, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Fit the attributor by training LOO models.
        
        Parameters
        ----------
        model : sklearn estimator
            A fitted sklearn model that implements fit() and predict()/predict_proba().
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Training labels.
            
        Returns
        -------
        self
        
        """
        _validate_mode(self.mode)
        X, y = _prepare_fit_inputs(X, y)
        self.model_ = model
        self.X_train_ = X
        self.y_train_ = y
        self.is_classifier_ = is_classifier(model)
        n_train = X.shape[0]
        
        # Fit LOO models in parallel
        iterator = range(n_train)
        if self.verbose > 0:
            iterator = tqdm(iterator, desc="Fitting LOO models")
        if self.n_jobs is None or self.n_jobs == 1:
            loo_models = [_fit_loo_model(model, X, y, i) for i in iterator]
        else:
            with tqdm_joblib(tqdm(total=n_train, desc="Fitting LOO models", disable=(self.verbose == 0))):
                loo_models = Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_loo_model)(model, X, y, i) for i in range(n_train)
                )
        
        # Track failed refits
        self.failed_indices_ = [i for i, m in enumerate(loo_models) if m is None]
        if self.failed_indices_:
            warnings.warn(
                f"LOO refit failed for {len(self.failed_indices_)} samples: "
                f"{self.failed_indices_}. "
                "This may happen due to class imbalance when removing samples. "
                "Influence scores for these samples will be NaN.",
                UserWarning
            )
        
        self.loo_models_ = loo_models

        return self

    def explain(
        self, X_test: ArrayLike, y_test: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        """
        Compute LOO influence scores of training samples on test samples.

        Parameters
        ----------
        X_test : array-like of shape (m_samples, n_features) or (n_features,)
            Test samples to explain. A single point may be 1D; result shape (1, n_train).
        y_test : array-like of shape (m_samples,) or scalar, optional
            Test labels. Required when `mode='loss'` and when using
            `mode='prediction'` with classifiers (to identify the true
            class). Optional for regression prediction mode.

        Returns
        -------
        scores : ndarray of shape (m_samples, n_samples)
            scores[i, j] = influence of training sample j on test sample i.
            May contain NaN where LOO refit failed (e.g. class imbalance).

            For mode='loss':
                Positive = helpful (removing sample increases test loss).
                Negative = harmful (removing sample decreases test loss).

            For mode='prediction':
                - Classifiers: Positive = increases the probability of the true class.
                - Regressors: Positive = increases the predicted value.

        Raises
        ------
        NotFittedError
            If called before fit().
        ValueError
            If required y_test labels are missing (loss mode, or prediction mode
            with classifiers).
        """
        check_is_fitted(self, ["model_", "loo_models_"])
        needs_y_test = self.mode == "loss" or (
            self.mode == "prediction" and self.is_classifier_
        )
        X_test, y_test = _prepare_explain_inputs(
            X_test,
            y_test,
            needs_y_test=needs_y_test,
            error_msg=f"y_test is required when mode='{self.mode}'. Provide y_test.",
        )

        n_test = X_test.shape[0]
        n_train = self.X_train_.shape[0]
        baseline = _value_at_test(
            self.model_, X_test, y_test, self.mode, self.is_classifier_
        )

        scores = np.full((n_test, n_train), np.nan)
        loo_iter = self.loo_models_
        if self.verbose > 0:
            loo_iter = tqdm(loo_iter, desc="Computing LOO influence")
        for j, loo_model in enumerate(loo_iter):
            if loo_model is None:
                continue
            loo_value = _value_at_test(
                loo_model, X_test, y_test, self.mode, self.is_classifier_
            )
            # Loss: influence = L(D \ z_j) - L(D); prediction: baseline - loo_value
            if self.mode == "loss":
                scores[:, j] = loo_value - baseline
            else:
                scores[:, j] = baseline - loo_value

        return scores
