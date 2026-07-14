"""Influence function implementation for linear models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator

from pyinfluence._base import (
    BaseAttributor,
    _prepare_explain_inputs,
    _prepare_fit_inputs,
    _validate_mode,
    check_is_fitted,
)
from pyinfluence._linear import (
    _augment_intercept,
    _compute_kernel_matrix,
    _get_dual_params,
    _get_params,
    _gradients_logistic,
    _gradients_ridge,
    _hessian_logistic,
    _hessian_ridge,
    _invert_hessian,
)
from pyinfluence._validation import (
    extract_regularization,
    validate_labels_in_classes,
    validate_model,
)

if TYPE_CHECKING:
    from typing import Self


class InfluenceFunctions(BaseAttributor):
    """
    Influence functions for linear models.

    Computes the effect of upweighting each training sample on the loss
    or prediction at test points, using the closed-form Hessian.

    Parameters
    ----------
    mode : {'loss', 'prediction'}, default='loss'
        What to measure influence on:

        - 'loss': Influence on test loss. Requires y_test in explain().
          Positive = helpful (upweighting decreases test loss).
          Negative = harmful (upweighting increases test loss).

        - 'prediction': Influence on predicted value. No y_test required.
          For regression: influence on model.predict(X_test).
          For LogisticRegression (binary): influence on the positive-class
          probability P(Y=classes_[1] | x). For RidgeClassifier (binary):
          influence on the linear decision value (no probability available).
          The true label is not needed. (LOO/Banzhaf/Bootstrap instead define
          prediction as change in true-class score and thus require y_test.)
          Positive = upweighting increases predicted value.
          Negative = upweighting decreases predicted value.

    damping : float, default=1e-5
        Regularization added to Hessian diagonal for numerical stability.

    Attributes
    ----------
    H_inv_ : ndarray
        Inverse Hessian matrix (computed during fit). Shape (p, p) where
        p = n_features (+1 if the model has an intercept); for KernelRidge
        the Hessian lives in dual space and p = n_train.
    train_grads_ : ndarray
        Per-sample gradients on training data, shape (n_samples, p) with
        p as for ``H_inv_``.
    model_type_ : str
        Detected model type ('ridge', 'linear', 'logistic',
        'ridge_classifier', 'kernel_ridge').
    model_ : sklearn estimator
        Reference to the fitted model.
    X_train_ : ndarray
        Training features (possibly augmented with intercept; the kernel
        matrix for KernelRidge).
    y_train_ : ndarray
        Training targets.
    n_classes_ : int
        2 for binary LogisticRegression; 1 otherwise (regression models and
        RidgeClassifier, which is treated as regression on ±1 targets).

    Supported Models
    ----------------
    - sklearn.linear_model.Ridge / RidgeCV
    - sklearn.linear_model.LinearRegression
    - sklearn.linear_model.LogisticRegression / LogisticRegressionCV (binary only)
    - sklearn.linear_model.RidgeClassifier / RidgeClassifierCV (binary only)
    - sklearn.kernel_ridge.KernelRidge (dual-space; Hessian is n_train x n_train)

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> model = Ridge(alpha=1.0).fit(X_train, y_train)
    >>> attr = InfluenceFunctions(mode='loss', damping=1e-5)
    >>> attr.fit(model, X_train, y_train)
    >>> scores = attr.explain(X_test, y_test)  # y_test required for mode='loss'
    """

    def __init__(
        self, mode: Literal["loss", "prediction"] = "loss", damping: float = 1e-5
    ) -> None:
        self.mode = mode
        self.damping = damping

    def fit(self, model: BaseEstimator, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Fit the attributor to a trained model and its training data.

        Parameters
        ----------
        model : sklearn estimator
            A fitted sklearn model (Ridge, RidgeCV, LinearRegression,
            LogisticRegression, LogisticRegressionCV, RidgeClassifier,
            RidgeClassifierCV, or KernelRidge).
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
            If model is unsupported, not fitted, or mode is invalid.
        """
        _validate_mode(self.mode)
        if self.damping < 0:
            raise ValueError(f"damping must be non-negative; got {self.damping}.")
        self.model_type_ = validate_model(model)
        self.model_ = model
        X, y = _prepare_fit_inputs(X, y)

        # Store original training data
        self.X_train_raw_ = X
        self.y_train_ = y

        # Extract regularization and convert to the per-sample-average
        # convention used by the Hessians below (H = data_term/n + lambda*I).
        # sklearn objectives are total-loss based: Ridge/RidgeClassifier
        # minimize ||y-X0||^2 + alpha*||0||^2 and LogisticRegression minimizes
        # ||0||^2/2 + C*sum(nll_i), so the average-loss lambda is alpha/n and
        # 1/(C*n) respectively. extract_regularization returns alpha or 1/C.
        n_samples = X.shape[0]
        reg_lambda = extract_regularization(model) / n_samples

        # KernelRidge operates in dual space — separate path
        if self.model_type_ == "kernel_ridge":
            self.has_intercept_ = False
            self._fit_kernel_ridge(X, y, reg_lambda)
            return self

        # Handle intercept augmentation (all primal-space models)
        has_intercept = model.fit_intercept
        if has_intercept:
            X_aug = _augment_intercept(X)
        else:
            X_aug = X

        self.X_train_ = X_aug
        self.has_intercept_ = has_intercept

        # Dispatch to appropriate fitting method
        if self.model_type_ in ("ridge", "linear"):
            self._fit_regression(X_aug, y, reg_lambda)
        elif self.model_type_ == "ridge_classifier":
            # Binarize labels to {-1, +1} matching RidgeClassifier internals
            y = validate_labels_in_classes(y, model.classes_, name="y")
            y_binary = np.where(y == model.classes_[1], 1.0, -1.0)
            self.classes_ = model.classes_
            self._fit_regression(X_aug, y_binary, reg_lambda)
        elif self.model_type_ == "logistic":
            self._fit_logistic(X_aug, y, reg_lambda)
        else:
            raise ValueError(f"Model type '{self.model_type_}' not yet supported.")

        return self

    def _fit_regression(
        self,
        X_aug: NDArray[np.floating],
        y: NDArray[np.floating],
        reg_lambda: float,
    ) -> None:
        """Fit for Ridge/Linear regression."""
        self.n_classes_ = 1

        # Get model parameters
        theta = _get_params(self.model_)

        # Compute Hessian and its inverse
        H = _hessian_ridge(X_aug, reg_lambda, self.damping, self.has_intercept_)
        self.H_inv_ = _invert_hessian(H)

        # Compute training gradients
        self.train_grads_ = _gradients_ridge(X_aug, y, theta)

    def _fit_logistic(
        self,
        X_aug: NDArray[np.floating],
        y: NDArray[np.floating],
        reg_lambda: float,
    ) -> None:
        """Fit for Logistic regression (binary only)."""
        n_classes = len(self.model_.classes_)
        self.n_classes_ = n_classes

        if n_classes != 2:
            raise ValueError(
                "LogisticRegression with more than two classes is not supported. "
                "Use a binary classifier or reduce to a binary task."
            )

        # Binary classification: single classifier
        self._fit_logistic_binary(X_aug, y, reg_lambda)

    def _fit_logistic_binary(
        self,
        X_aug: NDArray[np.floating],
        y: NDArray[np.floating],
        reg_lambda: float,
    ) -> None:
        """Fit for binary logistic regression."""
        # Get predicted probabilities for classes_[1]
        probs = self.model_.predict_proba(self.X_train_raw_)[:, 1]

        # Compute Hessian and its inverse
        H = _hessian_logistic(X_aug, probs, reg_lambda, self.damping, self.has_intercept_)
        self.H_inv_ = _invert_hessian(H)

        # Compute training gradients. The NLL gradient needs y as a 0/1
        # indicator of classes_[1] (the class probs refers to); raw label
        # values ({-1,+1}, {1,2}, strings, ...) would silently corrupt it.
        y = validate_labels_in_classes(y, self.model_.classes_, name="y")
        y01 = (y == self.model_.classes_[1]).astype(float)
        self.classes_ = self.model_.classes_
        self.train_grads_ = _gradients_logistic(X_aug, y01, probs)
        self.train_probs_ = probs

    def _fit_kernel_ridge(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        reg_lambda: float,
    ) -> None:
        """
        Fit for KernelRidge (dual-space influence functions).

        The math is identical to Ridge but in dual parameter space:
        X -> K (kernel matrix), coef_ -> dual_coef_.
        The Hessian is n_train x n_train (not p x p).
        """
        self.n_classes_ = 1

        # Compute training kernel matrix
        K_train = _compute_kernel_matrix(self.model_, X)
        self.K_train_ = K_train
        self.X_train_ = K_train  # used by _gradients_ridge

        # Get dual parameters
        alpha = _get_dual_params(self.model_)

        # Hessian of the per-sample-average KRR objective in dual space:
        # (1/n) [sum_i 0.5*(y_i - k_i'a)^2 + (lambda/2) a'Ka] has Hessian
        # K'K/n + (lambda/n) K. With this penalty matrix (K, not I), the
        # bilinear influence formula is exact for the KRR stationarity
        # condition (K + lambda*I) a = y. reg_lambda is already lambda/n here.
        n = K_train.shape[0]
        H = K_train @ K_train / n + reg_lambda * K_train
        H += self.damping * np.eye(n)
        self.H_inv_ = _invert_hessian(H)

        # Training gradients in dual space: -(y_i - k_i'α) * k_i
        self.train_grads_ = _gradients_ridge(K_train, y, alpha)

    def _removal_calibrated_scores(
        self, test_grads: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Bilinear influence scores calibrated to per-point removal.

        The raw upweighting form g_test' H^{-1} g_j is the derivative w.r.t. a
        unit weight on z_j; deleting z_j corresponds to a weight change of
        -1/n, so scores are divided by n_train to approximate the actual
        change from removal (same estimand and scale as LOOInfluence).
        """
        n_train = self.train_grads_.shape[0]
        return (test_grads @ self.H_inv_ @ self.train_grads_.T) / n_train

    def explain(
        self, X_test: ArrayLike, y_test: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        """
        Compute influence scores of training samples on test samples.

        Parameters
        ----------
        X_test : array-like of shape (m_samples, n_features) or (n_features,)
            Test samples to explain. A single point may be 1D; result shape (1, n_train).
        y_test : array-like of shape (m_samples,) or scalar, optional
            Test labels. Required when mode='loss'. Ignored when mode='prediction'.

        Returns
        -------
        scores : ndarray of shape (m_samples, n_samples)
            scores[i, j] = influence of training sample j on test sample i.

            For mode='loss':
                Positive = helpful (upweighting decreases test loss).
                Negative = harmful (upweighting increases test loss).

            For mode='prediction':
                Positive = upweighting increases predicted value.
                Negative = upweighting decreases predicted value.

        Raises
        ------
        NotFittedError
            If called before fit().
        ValueError
            If mode='loss' and y_test is None.
        """
        check_is_fitted(self, ["model_", "H_inv_", "train_grads_"])
        _validate_mode(self.mode)
        X_test, y_test = _prepare_explain_inputs(
            X_test,
            y_test,
            needs_y_test=(self.mode == "loss"),
            error_msg=(
                "y_test is required when mode='loss'. "
                "Provide y_test or use mode='prediction'."
            ),
        )

        # KernelRidge: separate path (dual space, no intercept augmentation)
        if self.model_type_ == "kernel_ridge":
            return self._score_kernel_ridge(X_test, y_test)

        # Augment test data if model has intercept
        if self.has_intercept_:
            X_test_aug = _augment_intercept(X_test)
        else:
            X_test_aug = X_test

        # Dispatch to appropriate scoring method
        if self.model_type_ in ("ridge", "linear"):
            return self._score_regression(X_test_aug, y_test)
        elif self.model_type_ == "ridge_classifier":
            # Binarize test labels for loss mode
            if y_test is not None:
                y_test = validate_labels_in_classes(
                    y_test, self.classes_, name="y_test"
                )
                y_test = np.where(y_test == self.classes_[1], 1.0, -1.0)
            return self._score_regression(X_test_aug, y_test)
        elif self.model_type_ == "logistic":
            return self._score_logistic_binary(X_test_aug, y_test)
        else:
            raise ValueError(f"Model type '{self.model_type_}' not supported.")

    def _score_regression(
        self,
        X_test_aug: NDArray[np.floating],
        y_test: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        """Compute scores for regression models."""
        if self.mode == "loss":
            test_grads = self._compute_test_grads_loss_regression(X_test_aug, y_test)
        else:
            test_grads = self._compute_test_grads_prediction_regression(X_test_aug)

        scores = self._removal_calibrated_scores(test_grads)
        return scores if self.mode == "loss" else -scores

    def _score_logistic_binary(
        self,
        X_test_aug: NDArray[np.floating],
        y_test: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        """Compute scores for binary logistic regression."""
        if self.mode == "loss":
            test_grads = self._compute_test_grads_loss_logistic_binary(X_test_aug, y_test)
        else:
            test_grads = self._compute_test_grads_prediction_logistic_binary(X_test_aug)

        scores = self._removal_calibrated_scores(test_grads)
        return scores if self.mode == "loss" else -scores

    def _compute_test_grads_loss_regression(
        self,
        X_test_aug: NDArray[np.floating],
        y_test: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute test gradients for loss mode (regression).

        For squared loss: ∇L(z_test) = -(y_test - ŷ_test) * x_test
        """
        theta = _get_params(self.model_)
        predictions = X_test_aug @ theta
        residuals = y_test - predictions
        return -X_test_aug * residuals[:, np.newaxis]

    def _compute_test_grads_prediction_regression(
        self, X_test_aug: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Compute test gradients for prediction mode (regression).

        For linear regression: ∇f(x_test) = x_test
        """
        return X_test_aug

    def _compute_test_grads_loss_logistic_binary(
        self,
        X_test_aug: NDArray[np.floating],
        y_test: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute test gradients for loss mode (binary logistic).

        For NLL: ∇L(z_test) = -(1{y_test = classes_[1]} - p_test) * x_test
        """
        # Get raw X_test for predict_proba
        if self.has_intercept_:
            X_test_raw = X_test_aug[:, :-1]
        else:
            X_test_raw = X_test_aug

        probs = self.model_.predict_proba(X_test_raw)[:, 1]
        y_test = validate_labels_in_classes(
            y_test, self.model_.classes_, name="y_test"
        )
        y01 = (y_test == self.model_.classes_[1]).astype(float)
        return _gradients_logistic(X_test_aug, y01, probs)

    def _compute_test_grads_prediction_logistic_binary(
        self, X_test_aug: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Compute test gradients for prediction mode (binary logistic).

        For logistic regression: ∇p(x_test) = p(1-p) * x_test
        """
        if self.has_intercept_:
            X_test_raw = X_test_aug[:, :-1]
        else:
            X_test_raw = X_test_aug

        probs = self.model_.predict_proba(X_test_raw)[:, 1]
        weights = probs * (1 - probs)
        return X_test_aug * weights[:, np.newaxis]

    def _self_influence_diag(self) -> NDArray[np.floating]:
        """
        Diagonal of the train-vs-train influence matrix without forming it.

        Used by ``pyinfluence.self_influence``: O(n p^2) time and O(n p)
        memory instead of the O(n^2) score matrix.
        """
        check_is_fitted(self, ["model_", "H_inv_", "train_grads_"])
        n_train = self.train_grads_.shape[0]
        if self.mode == "loss":
            # The loss gradient at training point j *is* train_grads_[j]
            # (same formula, same labels, same probabilities).
            test_grads = self.train_grads_
            sign = 1.0
        else:
            if self.model_type_ == "kernel_ridge":
                test_grads = self.K_train_
            elif self.model_type_ == "logistic":
                test_grads = self._compute_test_grads_prediction_logistic_binary(
                    self.X_train_
                )
            else:
                test_grads = self._compute_test_grads_prediction_regression(
                    self.X_train_
                )
            sign = -1.0
        diag = (
            np.einsum("ij,ij->i", test_grads @ self.H_inv_, self.train_grads_)
            / n_train
        )
        return sign * diag

    # -----------------------------------------------------------------
    # KernelRidge scoring (dual space)
    # -----------------------------------------------------------------

    def _score_kernel_ridge(
        self,
        X_test: NDArray[np.floating],
        y_test: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        """
        Compute scores for KernelRidge (dual space).

        Same formula as ridge regression with K replacing X and
        dual_coef_ replacing theta.
        """
        # Compute test kernel matrix: K(X_test, X_train)
        K_test = _compute_kernel_matrix(self.model_, X_test, self.X_train_raw_)

        alpha = _get_dual_params(self.model_)

        if self.mode == "loss":
            # Test gradients for loss: -(y_test - K_test @ α) * K_test
            predictions = K_test @ alpha
            residuals = y_test - predictions
            test_grads = -K_test * residuals[:, np.newaxis]
        else:
            # Test gradients for prediction: K_test
            test_grads = K_test

        scores = self._removal_calibrated_scores(test_grads)
        return scores if self.mode == "loss" else -scores
