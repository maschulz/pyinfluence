"""Input validation and model type detection."""

from __future__ import annotations

import warnings
from typing import Literal

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

# Type alias for supported model types
ModelType = Literal[
    "ridge", "linear", "logistic", "ridge_classifier", "kernel_ridge", "unsupported"
]

# Registry for influence-function support: (model_class, model_type).
# get_model_type() returns the first matching entry; register subclasses
# before base classes (e.g. RidgeCV before Ridge).
LINEAR_MODEL_REGISTRY: list[tuple[type, str]] = [
    # CV variants before base classes (LogisticRegressionCV is a subclass of
    # LogisticRegression; RidgeCV is NOT a subclass of Ridge but listed first
    # for consistency).
    (RidgeCV, "ridge"),
    (LogisticRegressionCV, "logistic"),
    (RidgeClassifierCV, "ridge_classifier"),
    # Base classes
    (Ridge, "ridge"),
    (LinearRegression, "linear"),
    (LogisticRegression, "logistic"),
    (RidgeClassifier, "ridge_classifier"),
    (KernelRidge, "kernel_ridge"),
]


def get_model_type(model: BaseEstimator) -> ModelType:
    """
    Detect the model type for influence function computation.

    Parameters
    ----------
    model : sklearn estimator
        A fitted sklearn model.

    Returns
    -------
    model_type : {'ridge', 'logistic', 'linear', 'ridge_classifier',
                  'kernel_ridge', 'unsupported'}
        The detected model type.
    """
    for _cls, model_type in LINEAR_MODEL_REGISTRY:
        if isinstance(model, _cls):
            return model_type  # type: ignore[return-value]
    return "unsupported"


def check_is_fitted_model(model: BaseEstimator) -> None:
    """
    Check if the model has been fitted.

    For supported linear models, checks for coef_ attribute.
    For other models, uses sklearn's check_is_fitted.

    Parameters
    ----------
    model : sklearn estimator
        The model to check.

    Raises
    ------
    ValueError
        If the model is not fitted.
    """
    model_type = get_model_type(model)

    if model_type == "kernel_ridge":
        # KernelRidge stores dual_coef_, not coef_
        if not hasattr(model, "dual_coef_"):
            raise ValueError(
                f"Model {type(model).__name__} is not fitted. "
                "Please call model.fit(X, y) before passing to an attributor."
            )
    elif model_type in ("ridge", "linear", "logistic", "ridge_classifier"):
        # Linear models store coef_
        if not hasattr(model, "coef_"):
            raise ValueError(
                f"Model {type(model).__name__} is not fitted. "
                "Please call model.fit(X, y) before passing to an attributor."
            )
    else:
        # For other models, use sklearn's check_is_fitted
        try:
            sklearn_check_is_fitted(model)
        except NotFittedError:
            raise ValueError(
                f"Model {type(model).__name__} is not fitted. "
                "Please call model.fit(X, y) before passing to an attributor."
            )


def extract_regularization(model: BaseEstimator) -> float:
    """
    Extract the regularization parameter from a fitted model.

    Parameters
    ----------
    model : sklearn estimator
        A fitted supported model (Ridge, RidgeCV, LinearRegression,
        LogisticRegression, LogisticRegressionCV, RidgeClassifier,
        RidgeClassifierCV, or KernelRidge).

    Returns
    -------
    reg_lambda : float
        The L2 regularization parameter (alpha for Ridge-family, 1/C for
        Logistic-family).

    Raises
    ------
    ValueError
        If the model type is unsupported.
    """
    model_type = get_model_type(model)

    if model_type in ("ridge", "ridge_classifier", "kernel_ridge"):
        # RidgeCV / RidgeClassifierCV store CV-selected alpha in alpha_
        if hasattr(model, "alpha_"):
            return float(model.alpha_)
        return float(model.alpha)  # type: ignore[union-attr]
    elif model_type == "linear":
        return 0.0  # No regularization
    elif model_type == "logistic":
        # Check if penalty is None
        penalty = getattr(model, "penalty", "l2")
        if penalty is None or penalty == "none":
            return 0.0
        # LogisticRegressionCV stores CV-selected C in C_ (array, one per class)
        if hasattr(model, "C_"):
            return 1.0 / float(model.C_[0])
        # lambda = 1/C for LogisticRegression
        return 1.0 / float(model.C)  # type: ignore[union-attr]
    else:
        raise ValueError(
            f"Cannot extract regularization from unsupported model type: "
            f"{type(model).__name__}"
        )


def validate_model(model: BaseEstimator) -> ModelType:
    """
    Validate that a model is supported for influence function computation.

    Parameters
    ----------
    model : sklearn estimator
        The model to validate.

    Returns
    -------
    model_type : {'ridge', 'logistic', 'linear', 'ridge_classifier',
                  'kernel_ridge'}
        The detected model type.

    Raises
    ------
    ValueError
        If the model is not fitted, unsupported, or is multiclass.

    Warns
    -----
    UserWarning
        If the model has no regularization (relies on damping for stability).
    """
    # Check model type first (so we reject unsupported before checking fitted)
    model_type = get_model_type(model)

    if model_type == "unsupported":
        supported_names = ", ".join(
            sorted({cls.__name__ for cls, _ in LINEAR_MODEL_REGISTRY})
        )
        raise ValueError(
            f"Unsupported model type: {type(model).__name__}. "
            f"Supported models: {supported_names} "
            "(binary classification only for LogisticRegression / RidgeClassifier)."
        )

    # Check if model is fitted (only for supported models)
    check_is_fitted_model(model)

    # Reject multiclass classifiers
    if model_type == "logistic" and len(model.classes_) > 2:
        raise ValueError(
            "LogisticRegression with more than two classes is not supported. "
            "Use a binary classifier or reduce to a binary task."
        )
    if model_type == "ridge_classifier" and len(model.classes_) > 2:
        raise ValueError(
            "RidgeClassifier with more than two classes is not supported. "
            "Use a binary classifier or reduce to a binary task."
        )

    # Warn if no regularization
    reg_lambda = extract_regularization(model)
    if reg_lambda == 0.0:
        warnings.warn(
            f"Model {type(model).__name__} has no regularization. "
            "Influence estimates will rely entirely on damping for numerical "
            "stability. Consider using Ridge instead of LinearRegression, or "
            "setting C < inf for LogisticRegression.",
            UserWarning,
        )

    return model_type
