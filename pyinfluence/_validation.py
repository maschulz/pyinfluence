"""Input validation and model type detection."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
)
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
            ) from None


def validate_labels_in_classes(
    y: ArrayLike, classes: NDArray, name: str = "y"
) -> NDArray:
    """
    Validate that every label in ``y`` is one of the model's classes.

    Catches silent mistakes like passing a wrong column or mistyped labels,
    which would otherwise be mapped to the negative class without error.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Labels to validate.
    classes : ndarray
        The fitted model's ``classes_``.
    name : str, default='y'
        Name used in the error message ('y' or 'y_test').

    Returns
    -------
    y : ndarray of shape (n_samples,)
        The validated labels as an array.

    Raises
    ------
    ValueError
        If any label is not in ``classes``.
    """
    y = np.asarray(y)
    known = np.isin(y, classes)
    if not np.all(known):
        bad = np.unique(y[~known])
        raise ValueError(
            f"{name} contains labels not among the model's classes_: "
            f"{bad.tolist()!r}. Expected labels from {classes.tolist()!r}."
        )
    return y


def _check_model_not_array(model) -> None:
    """Reject arrays passed in place of the model (the fit(X, y) slip)."""
    if isinstance(model, np.ndarray):
        raise TypeError(
            "model is a numpy array — did you mean fit(model, X, y)? "
            "The attributor signature takes the fitted estimator first, "
            "then the training data."
        )


def supports(model: BaseEstimator) -> tuple[bool, str | None]:
    """
    Check whether ``InfluenceFunctions`` can handle a fitted model.

    Never raises and never warns; a programmatic version of the validation
    that ``InfluenceFunctions.fit`` performs.

    Parameters
    ----------
    model : sklearn estimator
        The (fitted) model to check.

    Returns
    -------
    ok : bool
        True if ``InfluenceFunctions.fit(model, ...)`` would accept it.
    reason : str or None
        Human-readable explanation when ``ok`` is False, else None.

    Notes
    -----
    The refit-based attributors (LOOInfluence, BanzhafInfluence,
    BootstrapInfluence) accept any fitted sklearn estimator regardless of
    this check (multiclass classifiers additionally need predict_proba).

    Examples
    --------
    >>> ok, reason = supports(model)
    >>> if not ok:
    ...     print(f"falling back to LOO: {reason}")
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validate_model(model)
    except (ValueError, TypeError) as e:
        return False, str(e)
    return True, None


def warn_if_data_mismatch(model, X, y) -> None:
    """Warn when the model can't even beat a trivial baseline on (X, y).

    A fitted model scoring worse than the majority class (classifier) or the
    mean predictor (regressor) on the very data passed to ``fit`` is the
    signature of a preprocessing mismatch, most commonly passing *raw*
    features to the inner estimator of a Pipeline that was fit on
    *transformed* features. Influence scores computed from such a pairing
    are meaningless; warning here is preferable to returning them
    silently.
    """
    from pyinfluence._utils import _compute_loss_sklearn, _quiet_sklearn

    try:
        if is_classifier(model) and callable(getattr(model, "predict_proba", None)):
            # NLL against the predict-the-class-frequencies baseline: far
            # more sensitive than accuracy (a mismatched model is
            # confidently wrong, blowing up NLL even at chance accuracy).
            y_arr = np.asarray(y).ravel()
            mean_nll = float(
                _compute_loss_sklearn(model, np.asarray(X), y_arr, True).mean()
            )
            freqs = np.unique(y_arr, return_counts=True)[1] / y_arr.size
            prior_nll = float(-(freqs * np.log(freqs)).sum())
            bad = mean_nll > prior_nll
            detail = (
                f"mean NLL {mean_nll:.3f}, predict-the-class-frequencies "
                f"baseline {prior_nll:.3f}"
            )
        else:
            with _quiet_sklearn():
                score = model.score(X, y)
            if is_classifier(model):
                y_arr = np.asarray(y).ravel()
                _, counts = np.unique(y_arr, return_counts=True)
                baseline = counts.max() / y_arr.size
                bad = score < baseline - 0.05
                detail = f"accuracy {score:.3f}, majority-class baseline {baseline:.3f}"
            else:
                bad = score < -0.05
                detail = f"R^2 = {score:.3f} (0 = predicting the mean)"
    except Exception:
        return
    if bad:
        warnings.warn(
            f"The model scores worse than a trivial baseline on the data "
            f"passed to fit() ({detail}). Influence scores computed from a "
            "mismatched (model, X, y) triple are meaningless. Most common "
            "cause: the model was fit on *transformed* features (e.g. inside "
            "a Pipeline with a scaler) but fit() received the raw ones — "
            "pass the same transformed X the estimator was trained on.",
            UserWarning,
            stacklevel=2,
        )


def validate_refit_model(model: BaseEstimator) -> None:
    """
    Validate a model for the refit-based attributors (LOO/Banzhaf/Bootstrap).

    These are model-agnostic, but two failure modes deserve a clear error
    *before* the expensive refit loop runs:

    - an unfitted model (only detected when explain() calls predict on the
      original model, after all refits are wasted);
    - a multiclass classifier without predict_proba, whose 2-D
      decision_function output breaks the binary squared-error fallback with
      a cryptic broadcasting error (loss mode) or silently wrong shapes
      (prediction mode).

    Raises
    ------
    ValueError
        If the model is not fitted, or is a multiclass classifier without
        predict_proba.
    """
    _check_model_not_array(model)
    check_is_fitted_model(model)

    if is_classifier(model):
        classes = getattr(model, "classes_", None)
        if (
            classes is not None
            and len(classes) > 2
            and not callable(getattr(model, "predict_proba", None))
        ):
            raise ValueError(
                f"{type(model).__name__} is a multiclass classifier without "
                "predict_proba; the decision_function fallback only supports "
                "binary classification. Use a classifier with predict_proba "
                "(or wrap it, e.g. sklearn.calibration.CalibratedClassifierCV)."
            )


def influence_function_incompatibility(model: BaseEstimator) -> str | None:
    """
    Check a supported-type model for configurations that break the
    closed-form influence approximation.

    Returns a human-readable reason string if the model was fit with a
    configuration whose objective does not match the Hessian/gradient
    formulas used by InfluenceFunctions, else None.

    Checked configurations:
    - ``class_weight`` (logistic / ridge classifier): the weighted objective
      requires weighted gradients and Hessian; silently ignoring the weights
      yields degraded scores. Refit-based methods (LOO, Banzhaf, Bootstrap)
      clone the estimator and therefore handle class_weight correctly.
    - ``penalty='l1'`` / ``'elasticnet'`` (logistic): an l1 penalty
      contributes no curvature, so the l2-style Hessian correction is wrong.

    Note: models fit with a ``sample_weight`` argument cannot be detected
    post-hoc; see the README limitations section.
    """
    model_type = get_model_type(model)

    if model_type in ("logistic", "ridge_classifier"):
        class_weight = getattr(model, "class_weight", None)
        if class_weight is not None:
            return (
                f"the model was fit with class_weight={class_weight!r}, but "
                "InfluenceFunctions assumes an unweighted objective and would "
                "return silently degraded scores. Refit the model without "
                "class_weight, or use a refit-based method (LOOInfluence, "
                "BanzhafInfluence, BootstrapInfluence), which honors "
                "class_weight by cloning the estimator"
            )

    if model_type == "logistic":
        # sklearn <=1.7 spells this penalty='l1'/'elasticnet'; sklearn >=1.8
        # deprecates penalty in favor of l1_ratio (0 = l2, 1 = l1).
        penalty = getattr(model, "penalty", "l2")
        l1_ratio = getattr(model, "l1_ratio", None)
        if penalty in ("l1", "elasticnet") or (l1_ratio is not None and l1_ratio > 0):
            descr = (
                f"penalty={penalty!r}"
                if penalty in ("l1", "elasticnet")
                else f"l1_ratio={l1_ratio!r}"
            )
            return (
                f"the model uses {descr}, whose objective does not "
                "match the l2 Hessian correction used by InfluenceFunctions "
                "(an l1 penalty contributes no curvature), so scores would be "
                "biased. Use a pure l2 penalty, or a refit-based method "
                "(LOOInfluence, BanzhafInfluence, BootstrapInfluence)"
            )

    return None


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
        alpha = model.alpha_ if hasattr(model, "alpha_") else model.alpha  # type: ignore[union-attr]
        alpha_arr = np.asarray(alpha, dtype=float)
        if alpha_arr.size != 1:
            raise ValueError(
                f"Array-valued alpha ({alpha_arr.tolist()!r}) implies per-target "
                "regularization (multi-output), which is not supported. "
                "Fit one model per output with a scalar alpha."
            )
        return float(alpha_arr.reshape(()))
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
    _check_model_not_array(model)

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

    # Reject configurations whose objective does not match the closed-form
    # Hessian/gradient formulas (class_weight, l1/elasticnet penalties).
    reason = influence_function_incompatibility(model)
    if reason is not None:
        raise ValueError(f"Cannot use InfluenceFunctions: {reason}.")

    # liblinear regularizes the intercept, which violates the
    # unregularized-intercept assumption in the Hessian (error is O(1/(Cn))).
    if (
        model_type == "logistic"
        and getattr(model, "solver", None) == "liblinear"
        and getattr(model, "fit_intercept", False)
    ):
        warnings.warn(
            "LogisticRegression(solver='liblinear') regularizes the intercept, "
            "which the influence-function Hessian assumes is unregularized. "
            "Estimates carry a small O(1/(C*n)) bias; consider solver='lbfgs' "
            "for exact agreement.",
            UserWarning,
            stacklevel=2,
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
            stacklevel=2,
        )

    return model_type
