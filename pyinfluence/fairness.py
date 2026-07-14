"""Fairness-targeted training data attribution.

Attributes group-disparity functionals (demographic parity, equal
opportunity, false-positive-rate gaps, worst-group loss) to individual
training examples: which training points, if removed, would most change the
disparity a fairness audit measures?

Estimand
--------
Let F(theta) be a smoothed disparity functional evaluated on a fixed audit
set, e.g. the demographic parity gap

    F = mean_{a=a1} p_theta(x) - mean_{a=a0} p_theta(x),

where a is a binary sensitive attribute (a0 < a1 by sort order of the two
unique values). All attributors in this module estimate the per-point
removal effect

    score[j] ~= F(D \\ {z_j}) - F(D),

so positive scores mark training points whose removal *increases* the signed
gap, and negative scores mark points whose removal decreases it. This is the
same removal convention (sign and scale) used by the loss-based attributors
in the rest of the package.

With ``target='absolute'`` the functional is |F|, so negative scores always
mean "removing this point shrinks the disparity magnitude".

Methods
-------
- :class:`FairnessInfluenceFunctions` — closed-form influence functions for
  supported GLMs (binary LogisticRegression / RidgeClassifier, Ridge /
  LinearRegression for group mean-prediction gaps). ``hessian='identity'``
  gives the gradient-dot baseline (no curvature correction).
- :class:`RefitFairnessInfluence` — exact per-point removal effects via
  refitting (model-agnostic ground truth; n refits).
- :class:`SubsampledFairnessInfluence` — Monte-Carlo subset estimator
  (model-agnostic, maximum-sample-reuse Banzhaf-style; T refits total).

Utilities
---------
- :func:`disparity_value` / :func:`disparity_value_hard` — evaluate metrics.
- :func:`group_removal_effect` — actual disparity change from removing an
  index set (refit-based; for validating summed scores).
- :func:`disparity_removal_curve` — retrain-based repair curve: disparity
  and accuracy after removing the top-k scored points, vs a random baseline.
- :func:`cohens_d` — ready-made callable metric (standardized group gap).

Custom metrics
--------------
Every ``metric=`` parameter also accepts a callable
``metric(scores, sensitive, y) -> float`` over the audit-set score vector,
so any group functional (Cohen's d, quantile gaps, calibration gaps, ...)
can be attributed. The closed form differentiates the callable by finite
differences (or an analytic ``metric_grad``), which requires smoothness;
the refit-based estimators only evaluate it, so non-smooth metrics work
there. The refit estimator doubles as ground truth for validating any new
metric's closed-form scores.

Scope note: leverage, not fault
-------------------------------
Disparity-influence scores localize *leverage*: which training records the
measured gap rests on, and what removing them would do. They do not
identify records whose labels or features are wrong. Within a
group-by-outcome cell, every attribution score is a function of the
recorded features alone, so a corrupted record and a legitimate one that
look alike to the model cannot be separated by any attribution score —
empirically, within-cell retrieval of planted label flips is at chance for
these estimands. Use these scores to find where a disparity lives and to
choose repair interventions; use mechanism-matched detectors (per-sample
error for label noise, group-conditional feature residuals for measurement
corruption) to find suspect records.

Terminology note: "fairness influence functions" is used by Ghosh, Basu &
Meel (FAccT 2023) for *feature*-level variance decomposition; this module
attributes disparities to *training examples* instead.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone, is_classifier

from pyinfluence._base import _prepare_fit_inputs, check_is_fitted
from pyinfluence._influence import InfluenceFunctions
from pyinfluence._linear import _augment_intercept
from pyinfluence._utils import _compute_loss_sklearn, tqdm_joblib
from pyinfluence._validation import (
    check_is_fitted_model,
    validate_labels_in_classes,
)

if TYPE_CHECKING:
    from typing import Self

from tqdm import tqdm

__all__ = [
    "FairnessInfluenceFunctions",
    "RefitFairnessInfluence",
    "SubsampledFairnessInfluence",
    "cohens_d",
    "disparity_value",
    "disparity_value_hard",
    "group_removal_effect",
    "disparity_removal_curve",
    "DISPARITY_METRICS",
]

DISPARITY_METRICS = ("dp", "eopp", "fpr", "worst_group_loss")

MetricName = Literal["dp", "eopp", "fpr", "worst_group_loss"]
# A custom metric is any smooth function of the audit-set score vector:
# metric(scores, sensitive, y) -> float. `scores` are the model outputs
# `_model_scores` produces (P(Y=classes_[1]|x) for classifiers with
# predict_proba, decision values or predictions otherwise); `y` is the audit
# label array or None — a metric that needs labels must validate that itself.
MetricLike = "MetricName | Callable[[NDArray, NDArray, NDArray | None], float]"
TargetName = Literal["signed", "absolute"]


# -----------------------------------------------------------------------------
# Sensitive-attribute handling
# -----------------------------------------------------------------------------


def _binarize_sensitive(sensitive: ArrayLike) -> tuple[NDArray[np.bool_], tuple]:
    """
    Map a binary sensitive attribute to boolean (True = higher sorted value).

    Returns (mask_a1, (a0, a1)) where the signed gap convention is
    F = E[. | a1] - E[. | a0] and a0 < a1 in sort order.
    """
    s = np.asarray(sensitive).ravel()
    values = np.unique(s)
    if len(values) != 2:
        raise ValueError(
            f"sensitive must be binary; got {len(values)} unique values. "
            "For multi-group attributes use metric='worst_group_loss' or "
            "compute pairwise gaps."
        )
    return s == values[1], (values[0], values[1])


def _metric_needs_y(metric) -> bool:
    # Callable metrics receive y (or None) and enforce their own requirements
    return metric in ("eopp", "fpr", "worst_group_loss")


def _validate_metric(metric) -> None:
    if callable(metric):
        return
    if metric not in DISPARITY_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Available: {DISPARITY_METRICS}, "
            "or a callable metric(scores, sensitive, y) -> float."
        )


def _validate_target(target: str) -> None:
    if target not in ("signed", "absolute"):
        raise ValueError(f"target must be 'signed' or 'absolute'; got {target!r}.")


def _model_scores(model: BaseEstimator, X: NDArray) -> NDArray[np.floating]:
    """Model output the smoothed metrics average: P(Y=1|x) or prediction."""
    if is_classifier(model):
        if callable(getattr(model, "predict_proba", None)):
            return model.predict_proba(X)[:, 1]
        # decision_function fallback (e.g. RidgeClassifier)
        return np.asarray(model.decision_function(X), dtype=float).ravel()
    return np.asarray(model.predict(X), dtype=float).ravel()


def _positive_label(model: BaseEstimator):
    """The label whose score `_model_scores` averages.

    For classifiers this is ``classes_[1]`` — the class predict_proba[:, 1]
    (or a positive decision value) refers to. Conditioning eopp/fpr on any
    other value would silently compute a different metric (e.g. swap the
    two) whenever labels are not {0, 1}.
    """
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 2:
        return classes[1]
    return 1.0


def _metric_mask(
    metric: str,
    y: NDArray[np.floating] | None,
    pos_label,
) -> NDArray[np.bool_] | None:
    """Which audit rows enter the metric (None = all).

    eopp restricts to true positives (y == pos_label), fpr to true
    negatives. pos_label must be the model's positive class
    (`_positive_label`), not a hard-coded 1.
    """
    if metric == "eopp":
        return np.asarray(y).ravel() == pos_label
    if metric == "fpr":
        return np.asarray(y).ravel() != pos_label
    return None


def _validate_audit_labels(
    model: BaseEstimator, y: NDArray | None, metric
) -> None:
    """For classifier metrics that use y, insist labels come from classes_.

    Skipped for callable metrics: they define their own use of y (which need
    not be a label array at all).
    """
    if y is None or callable(metric) or not is_classifier(model):
        return
    classes = getattr(model, "classes_", None)
    if classes is not None:
        validate_labels_in_classes(y, np.asarray(classes), name="y (audit)")


def cohens_d(
    scores: ArrayLike,
    sensitive: ArrayLike,
    y: ArrayLike | None = None,
) -> float:
    """
    Group-difference Cohen's d of a score vector.

    ``d = (mean_a1 - mean_a0) / pooled_std`` with the same group convention
    as the 'dp' gap (a0 < a1 in sort order) and the pooled standard
    deviation ``sqrt(((n1-1)v1 + (n0-1)v0) / (n1+n0-2))`` with ``ddof=1``
    group variances.

    A ready-made callable metric: pass ``metric=cohens_d`` to any fairness
    attributor, :func:`disparity_value`, or
    :func:`disparity_removal_curve`.

    Parameters
    ----------
    scores : array-like of shape (m,)
        Model scores on the audit set (as produced internally by the
        fairness machinery: probabilities, decision values, or predictions).
    sensitive : array-like of shape (m,)
        Binary sensitive attribute.
    y : ignored
        Present to fit the callable-metric signature.

    Returns
    -------
    d : float

    Notes
    -----
    Because d is normalized by the pooled spread, a training point can
    shrink ``|d|`` by *inflating within-group score variance* rather than by
    closing the gap. When ranking points for removal by Cohen's-d influence,
    inspect the raw-gap ('dp') attribution alongside before acting.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    mask_a1, _ = _binarize_sensitive(sensitive)
    s1, s0 = scores[mask_a1], scores[~mask_a1]
    n1, n0 = s1.size, s0.size
    if n1 < 2 or n0 < 2:
        raise ValueError(
            "cohens_d requires at least two audit samples per group; "
            f"got {n1} and {n0}."
        )
    pooled_var = (
        (n1 - 1) * s1.var(ddof=1) + (n0 - 1) * s0.var(ddof=1)
    ) / (n1 + n0 - 2)
    if pooled_var <= 0:
        raise ValueError(
            "cohens_d is undefined: zero pooled variance "
            "(scores are constant within each group)."
        )
    return float((s1.mean() - s0.mean()) / np.sqrt(pooled_var))


def _fd_metric_grad(
    metric: Callable,
    scores: NDArray[np.floating],
    sensitive: NDArray,
    y: NDArray | None,
    eps: float = 1e-6,
) -> NDArray[np.floating]:
    """Central finite-difference gradient of a callable metric w.r.t. scores.

    2m metric evaluations of an O(m) numpy function — cheap for audit sets
    up to ~10^4 points. Requires the metric to be smooth in the scores;
    thresholded/rank-based metrics yield zero or garbage gradients here and
    must go through the refit-based estimators instead.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    grad = np.empty_like(scores)
    for i in range(scores.size):
        h = eps * max(1.0, abs(scores[i]))
        sp = scores.copy()
        sp[i] += h
        sm = scores.copy()
        sm[i] -= h
        grad[i] = (metric(sp, sensitive, y) - metric(sm, sensitive, y)) / (2 * h)
    return grad


# -----------------------------------------------------------------------------
# Metric values
# -----------------------------------------------------------------------------


def disparity_value(
    model: BaseEstimator,
    X: ArrayLike,
    sensitive: ArrayLike,
    y: ArrayLike | None = None,
    metric: MetricName = "dp",
    target: TargetName = "signed",
) -> float:
    """
    Smoothed disparity of a fitted model on an audit set.

    Parameters
    ----------
    model : fitted sklearn estimator
        Classifier with predict_proba/decision_function, or regressor.
    X : array-like of shape (m, p)
        Audit features.
    sensitive : array-like of shape (m,)
        Binary sensitive attribute (any two values; the gap is
        E[.|a1] - E[.|a0] with a0 < a1 in sort order). For
        'worst_group_loss' any number of groups is allowed.
    y : array-like of shape (m,), optional
        Audit labels. Required for 'eopp', 'fpr', and 'worst_group_loss'.
    metric : {'dp', 'eopp', 'fpr', 'worst_group_loss'} or callable, default='dp'
        - 'dp': gap in mean score (P(Y=1|x) for classifiers, prediction for
          regressors) between sensitive groups.
        - 'eopp': same gap restricted to the model's positive class
          (true-positive-rate analog).
        - 'fpr': same gap restricted to the negative class
          (false-positive-rate analog).
        - 'worst_group_loss': max over groups of mean per-sample loss.
        - callable ``metric(scores, sensitive, y) -> float``: any custom
          functional of the model-score vector, e.g.
          :func:`cohens_d`. Receives the same smoothed scores the built-in
          gaps average, plus the raw sensitive and y arrays.
    target : {'signed', 'absolute'}, default='signed'
        Report the signed gap or its absolute value. Ignored for
        'worst_group_loss' (already nonnegative).

    Returns
    -------
    value : float
    """
    _validate_metric(metric)
    _validate_target(target)
    X = np.asarray(X)
    s = np.asarray(sensitive).ravel()
    if _metric_needs_y(metric) and y is None:
        raise ValueError(f"y is required for metric={metric!r}.")
    _validate_audit_labels(model, y, metric)

    if callable(metric):
        scores = _model_scores(model, X)
        y_arr = None if y is None else np.asarray(y).ravel()
        value = float(metric(scores, s, y_arr))
        return abs(value) if target == "absolute" else value

    if metric == "worst_group_loss":
        yv = np.asarray(y).ravel()
        losses = _compute_loss_sklearn(model, X, yv, is_classifier(model))
        return float(
            max(losses[s == g].mean() for g in np.unique(s))
        )

    mask_a1, _ = _binarize_sensitive(s)
    keep = _metric_mask(metric, y, _positive_label(model))
    scores = _model_scores(model, X)
    if keep is not None:
        scores, mask_a1 = scores[keep], mask_a1[keep]
    if mask_a1.all() or not mask_a1.any():
        raise ValueError(
            "Both sensitive groups must be present in the "
            "(label-restricted) audit set."
        )
    gap = float(scores[mask_a1].mean() - scores[~mask_a1].mean())
    return abs(gap) if target == "absolute" else gap


def disparity_value_hard(
    model: BaseEstimator,
    X: ArrayLike,
    sensitive: ArrayLike,
    y: ArrayLike | None = None,
    metric: MetricName = "dp",
    target: TargetName = "signed",
    threshold: float = 0.5,
) -> float:
    """
    Hard (thresholded-decision) disparity on an audit set.

    Same conventions as :func:`disparity_value` but computed from thresholded
    positive-class probabilities (classifiers only): 'dp' is the selection
    rate gap, 'eopp' the TPR gap, 'fpr' the FPR gap. 'worst_group_loss' uses
    0/1 error instead of the model loss. A callable metric is applied to the
    0/1 decision vector instead of the smoothed scores.

    ``threshold`` applies to ``predict_proba``; for classifiers exposing only
    ``decision_function`` the decision boundary is fixed at 0 and
    ``threshold`` is ignored.
    """
    _validate_metric(metric)
    _validate_target(target)
    if not is_classifier(model):
        raise ValueError("disparity_value_hard requires a classifier.")
    X = np.asarray(X)
    s = np.asarray(sensitive).ravel()
    if _metric_needs_y(metric) and y is None:
        raise ValueError(f"y is required for metric={metric!r}.")
    _validate_audit_labels(model, y, metric)

    # decisions is a 0/1 indicator of predicting the positive class
    # (classes_[1]); label-space comparisons below must use the same encoding.
    if callable(getattr(model, "predict_proba", None)):
        decisions = (model.predict_proba(X)[:, 1] >= threshold).astype(float)
    else:
        decisions = (np.asarray(model.decision_function(X)).ravel() >= 0).astype(float)

    if callable(metric):
        y_arr = None if y is None else np.asarray(y).ravel()
        value = float(metric(decisions, s, y_arr))
        return abs(value) if target == "absolute" else value

    if metric == "worst_group_loss":
        y01 = (np.asarray(y).ravel() == _positive_label(model)).astype(float)
        err = (decisions != y01).astype(float)
        return float(max(err[s == g].mean() for g in np.unique(s)))

    mask_a1, _ = _binarize_sensitive(s)
    keep = _metric_mask(metric, y, _positive_label(model))
    if keep is not None:
        decisions, mask_a1 = decisions[keep], mask_a1[keep]
    gap = float(decisions[mask_a1].mean() - decisions[~mask_a1].mean())
    return abs(gap) if target == "absolute" else gap


# -----------------------------------------------------------------------------
# Closed-form attributor (GLMs)
# -----------------------------------------------------------------------------


class FairnessInfluenceFunctions:
    """
    Closed-form influence of training points on a disparity functional.

    Supported models: binary LogisticRegression(CV), RidgeClassifier(CV)
    (via decision-value gaps), Ridge/RidgeCV/LinearRegression (mean-prediction
    gaps). KernelRidge is not supported.

    Parameters
    ----------
    metric : {'dp', 'eopp', 'fpr', 'worst_group_loss'} or callable, default='dp'
        Disparity functional to attribute (see :func:`disparity_value`).
        A callable ``metric(scores, sensitive, y) -> float`` attributes any
        *smooth* functional of the audit-set score vector (e.g.
        :func:`cohens_d`); its score-gradient is taken from ``metric_grad``
        when given, else by central finite differences. Non-smooth metrics
        (thresholded rates, rank statistics) must use the refit-based
        estimators instead.
    target : {'signed', 'absolute'}, default='signed'
        Attribute the signed gap or its absolute value (gradient scaled by
        sign(F); undefined at F=0). Ignored for 'worst_group_loss'.
    damping : float, default=1e-5
        Hessian damping, as in :class:`~pyinfluence.InfluenceFunctions`.
    hessian : {'exact', 'identity'}, default='exact'
        'identity' replaces H^{-1} with I (gradient-dot baseline).
    metric_grad : callable, optional
        Analytic gradient of a callable ``metric`` w.r.t. the score vector:
        ``metric_grad(scores, sensitive, y) -> ndarray of shape (m,)``.
        Only used with a callable metric; default is finite differences.
    fd_eps : float, default=1e-6
        Relative step for the finite-difference metric gradient.

    Attributes
    ----------
    base_attributor_ : InfluenceFunctions
        Fitted loss-influence attributor providing H^{-1} and train grads.
    sensitive_train_ : ndarray
        Stored training sensitive attribute (for reporting only; scores do
        not require training-set sensitive labels).

    Notes
    -----
    ``explain`` returns a vector of length n_train estimating
    F(D \\ {z_j}) - F(D) on the audit set — removal-calibrated like all
    attributors in this package. Validated against exact refitting in
    ``tests/test_fairness.py``.
    """

    def __init__(
        self,
        metric: MetricName | Callable = "dp",
        target: TargetName = "signed",
        damping: float = 1e-5,
        hessian: Literal["exact", "identity"] = "exact",
        metric_grad: Callable | None = None,
        fd_eps: float = 1e-6,
    ) -> None:
        self.metric = metric
        self.target = target
        self.damping = damping
        self.hessian = hessian
        self.metric_grad = metric_grad
        self.fd_eps = fd_eps

    def fit(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        sensitive: ArrayLike | None = None,
    ) -> Self:
        """
        Fit to a trained model and its training data.

        Parameters
        ----------
        model : fitted sklearn estimator
            Supported GLM (see class docstring).
        X, y : array-like
            Training data the model was fitted on.
        sensitive : array-like of shape (n,), optional
            Training-set sensitive attribute. Not needed for the scores
            (the functional lives on the audit set) but stored for
            group-level reporting.
        """
        _validate_metric(self.metric)
        _validate_target(self.target)
        if self.hessian not in ("exact", "identity"):
            raise ValueError(
                f"hessian must be 'exact' or 'identity'; got {self.hessian!r}."
            )
        base = InfluenceFunctions(mode="prediction", damping=self.damping)
        base.fit(model, X, y)
        if base.model_type_ == "kernel_ridge":
            raise ValueError(
                "KernelRidge is not supported by FairnessInfluenceFunctions."
            )
        self.base_attributor_ = base
        self.model_ = model
        X_arr, y_arr = _prepare_fit_inputs(X, y)
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self.sensitive_train_ = (
            None if sensitive is None else np.asarray(sensitive).ravel()
        )
        return self

    # -- gradient of the functional w.r.t. parameters -------------------------

    def _grad_disparity(
        self,
        X_audit: NDArray[np.floating],
        y_audit: NDArray[np.floating] | None,
        sensitive_audit: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """grad_theta F on the audit set, in augmented-parameter space."""
        base = self.base_attributor_
        model = self.model_
        X_aug = (
            _augment_intercept(X_audit) if base.has_intercept_ else X_audit
        )

        if callable(self.metric):
            # Generic smooth functional of the score vector: chain rule
            # grad_theta F = sum_i (dF/ds_i) * grad_theta s_i, with dF/ds
            # from the user's analytic gradient or central finite
            # differences (the metric is plain numpy on an m-vector, so FD
            # is cheap; smoothness is required — see _fd_metric_grad).
            s_arr = np.asarray(sensitive_audit).ravel()
            y_arr = None if y_audit is None else np.asarray(y_audit).ravel()
            scores = _model_scores(model, X_audit)
            if self.metric_grad is not None:
                dFds = np.asarray(
                    self.metric_grad(scores, s_arr, y_arr), dtype=float
                ).ravel()
                if dFds.shape != scores.shape:
                    raise ValueError(
                        "metric_grad must return one gradient entry per "
                        f"audit sample; got shape {dFds.shape} for "
                        f"{scores.shape[0]} samples."
                    )
            else:
                dFds = _fd_metric_grad(
                    self.metric, scores, s_arr, y_arr, eps=self.fd_eps
                )
            # d score_i / d theta: p(1-p) x for logistic probabilities,
            # x for linear scores (ridge / linear / ridge_classifier)
            if base.model_type_ == "logistic":
                per_sample = X_aug * (scores * (1 - scores))[:, None]
            else:
                per_sample = X_aug
            grad = per_sample.T @ dFds
            if self.target == "absolute":
                value = float(self.metric(scores, s_arr, y_arr))
                if value == 0:
                    warnings.warn(
                        "Metric value is exactly 0; absolute-target gradient "
                        "is undefined. Returning the signed gradient.",
                        UserWarning,
                    )
                else:
                    grad = np.sign(value) * grad
            return grad

        if self.metric == "worst_group_loss":
            # subgradient: gradient of the mean loss of the worst group
            s = np.asarray(sensitive_audit).ravel()
            losses = _compute_loss_sklearn(
                model, X_audit, np.asarray(y_audit).ravel(), is_classifier(model)
            )
            groups = np.unique(s)
            worst = groups[int(np.argmax([losses[s == g].mean() for g in groups]))]
            mask = s == worst
            tg = self._per_sample_loss_grads(
                X_aug[mask],
                None if y_audit is None else np.asarray(y_audit).ravel()[mask],
                X_audit[mask],
            )
            return tg.mean(axis=0)

        mask_a1, _ = _binarize_sensitive(sensitive_audit)
        keep = _metric_mask(self.metric, y_audit, _positive_label(model))
        if keep is not None:
            X_aug, X_audit, mask_a1 = X_aug[keep], X_audit[keep], mask_a1[keep]
        if mask_a1.sum() == 0 or (~mask_a1).sum() == 0:
            raise ValueError(
                "Both sensitive groups must be present in the (label-restricted) "
                "audit set."
            )
        # d score / d theta per audit sample
        if base.model_type_ == "logistic":
            p = model.predict_proba(X_audit)[:, 1]
            per_sample = X_aug * (p * (1 - p))[:, None]
        else:
            # linear score: ridge / linear regression / ridge_classifier
            per_sample = X_aug
        grad = per_sample[mask_a1].mean(axis=0) - per_sample[~mask_a1].mean(axis=0)

        if self.target == "absolute":
            # sign from the smoothed signed gap on the restricted set
            scores = _model_scores(model, X_audit)
            gap = scores[mask_a1].mean() - scores[~mask_a1].mean()
            if gap == 0:
                warnings.warn(
                    "Signed gap is exactly 0; absolute-target gradient is "
                    "undefined. Returning signed-gap gradient.",
                    UserWarning,
                )
            else:
                grad = np.sign(gap) * grad
        return grad

    def _per_sample_loss_grads(
        self,
        X_aug: NDArray[np.floating],
        y: NDArray[np.floating] | None,
        X_raw: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Per-sample loss gradients on arbitrary points (for worst-group)."""
        base = self.base_attributor_
        model = self.model_
        if base.model_type_ == "logistic":
            p = model.predict_proba(X_raw)[:, 1]
            # NLL gradient needs y as a 0/1 indicator of classes_[1] (the
            # class p refers to), not the raw label values.
            y01 = (np.asarray(y).ravel() == model.classes_[1]).astype(float)
            return -X_aug * (y01 - p)[:, None]
        # squared-error models: _compute_loss_sklearn uses the *unhalved*
        # squared error, so the matching gradient carries a factor 2
        theta = np.concatenate(
            [np.atleast_1d(model.coef_).ravel(),
             np.atleast_1d(model.intercept_).ravel()]
        ) if model.fit_intercept else np.atleast_1d(model.coef_).ravel()
        if base.model_type_ == "ridge_classifier":
            yv = np.where(np.asarray(y).ravel() == model.classes_[1], 1.0, -1.0)
        else:
            yv = np.asarray(y).ravel()
        resid = yv - X_aug @ theta
        return -2.0 * X_aug * resid[:, None]

    def explain(
        self,
        X_audit: ArrayLike,
        *,
        y_audit: ArrayLike | None = None,
        sensitive_audit: ArrayLike | None = None,
    ) -> NDArray[np.floating]:
        """
        Estimate each training point's removal effect on the disparity.

        Parameters
        ----------
        X_audit : array-like of shape (m, p)
            Audit set on which the disparity functional is evaluated.
        y_audit : array-like of shape (m,), optional
            Audit labels; required for 'eopp', 'fpr', 'worst_group_loss'.
            Keyword-only, so a sensitive attribute can never silently bind
            to the label slot.
        sensitive_audit : array-like of shape (m,)
            Audit sensitive attribute. Required. Keyword-only.

        Returns
        -------
        scores : ndarray of shape (n_train,)
            scores[j] ~= F(D \\ {z_j}) - F(D). Positive = removing z_j
            increases the (signed or absolute) disparity.
        """
        check_is_fitted(self, ["base_attributor_"])
        if sensitive_audit is None:
            raise ValueError("sensitive_audit is required.")
        if _metric_needs_y(self.metric) and y_audit is None:
            raise ValueError(f"y_audit is required for metric={self.metric!r}.")
        X_audit = np.asarray(X_audit)
        if X_audit.ndim == 1:
            X_audit = X_audit.reshape(1, -1)

        base = self.base_attributor_
        gF = self._grad_disparity(
            X_audit,
            None if y_audit is None else np.asarray(y_audit).ravel(),
            np.asarray(sensitive_audit).ravel(),
        )
        n_train = base.train_grads_.shape[0]
        if self.hessian == "identity":
            direction = gF
        else:
            direction = base.H_inv_ @ gF
        # removal weight -1/n; d theta = (1/n) H^{-1} grad_l_j
        return (base.train_grads_ @ direction) / n_train


# -----------------------------------------------------------------------------
# Exact refit attributor (ground truth)
# -----------------------------------------------------------------------------


def _refit_without(
    model: BaseEstimator,
    X: NDArray,
    y: NDArray,
    remove: NDArray[np.intp] | int,
    refit_factory: Callable[[int], BaseEstimator] | None,
) -> BaseEstimator | None:
    mask = np.ones(len(y), dtype=bool)
    mask[remove] = False
    est = (
        refit_factory(int(mask.sum())) if refit_factory is not None
        else clone(model)
    )
    try:
        return est.fit(X[mask], y[mask])
    except Exception:
        return None


class RefitFairnessInfluence:
    """
    Exact per-point removal effects on a disparity functional via refitting.

    Model-agnostic ground truth: for each training point, refits the
    estimator without it and recomputes the (smoothed) disparity on the audit
    set. Costs n refits.

    Parameters
    ----------
    metric, target : see :class:`FairnessInfluenceFunctions`. Callable
        metrics ``metric(scores, sensitive, y) -> float`` are evaluated
        on refit models directly, so smoothness is NOT required here
        (rank- or threshold-based metrics are fine).
    n_jobs : int, optional
        Parallel refits (joblib). None = sequential.
    verbose : int, default=1
        Progress bar on/off.
    refit_factory : callable(n_remaining) -> estimator, optional
        Estimator constructor for refits. Default clones the original model
        (the practitioner counterfactual). Pass a factory to hold the
        per-sample-average regularization fixed (e.g. C * n/m for
        LogisticRegression), which isolates the removal effect from the
        regularization shift when validating influence estimates.
    """

    def __init__(
        self,
        metric: MetricName = "dp",
        target: TargetName = "signed",
        n_jobs: int | None = None,
        verbose: int = 1,
        refit_factory: Callable[[int], BaseEstimator] | None = None,
    ) -> None:
        self.metric = metric
        self.target = target
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit_factory = refit_factory

    def fit(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        sensitive: ArrayLike | None = None,
    ) -> Self:
        """Fit the leave-one-out models (the expensive step, done once)."""
        _validate_metric(self.metric)
        _validate_target(self.target)
        check_is_fitted_model(model)
        X_arr, y_arr = _prepare_fit_inputs(X, y)
        self.model_ = model
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self.sensitive_train_ = (
            None if sensitive is None else np.asarray(sensitive).ravel()
        )
        n = len(y_arr)

        def one(j: int) -> BaseEstimator | None:
            return _refit_without(model, X_arr, y_arr, j, self.refit_factory)

        if self.n_jobs is None or self.n_jobs == 1:
            it = range(n)
            if self.verbose > 0:
                it = tqdm(it, desc="Fitting LOO models")
            self.loo_models_ = [one(j) for j in it]
        else:
            with tqdm_joblib(
                tqdm(total=n, desc="Fitting LOO models",
                     disable=(self.verbose == 0))
            ):
                self.loo_models_ = Parallel(n_jobs=self.n_jobs)(
                    delayed(one)(j) for j in range(n)
                )
        n_failed = sum(m is None for m in self.loo_models_)
        if n_failed:
            warnings.warn(
                f"Refit failed for {n_failed} points; their scores will be "
                "NaN.",
                UserWarning,
            )
        return self

    def explain(
        self,
        X_audit: ArrayLike,
        *,
        y_audit: ArrayLike | None = None,
        sensitive_audit: ArrayLike | None = None,
        metric: MetricName | Callable | None = None,
        target: TargetName | None = None,
    ) -> NDArray[np.floating]:
        """
        Exact removal effects: scores[j] = F(D \\ {z_j}) - F(D).

        The LOO models fitted in ``fit`` are reused, so different metrics /
        audit sets can be evaluated without refitting (pass ``metric`` /
        ``target`` to override the constructor arguments). NaN where the
        refit failed (e.g. a class disappears).
        """
        check_is_fitted(self, ["model_", "loo_models_"])
        metric = self.metric if metric is None else metric
        target = self.target if target is None else target
        _validate_metric(metric)
        _validate_target(target)
        if sensitive_audit is None:
            raise ValueError("sensitive_audit is required.")
        X_audit = np.asarray(X_audit)
        s_audit = np.asarray(sensitive_audit).ravel()
        y_a = None if y_audit is None else np.asarray(y_audit).ravel()

        base_value = disparity_value(
            self.model_, X_audit, s_audit, y_a, metric, target
        )
        deltas = np.array([
            float("nan") if m is None else
            disparity_value(m, X_audit, s_audit, y_a, metric, target)
            - base_value
            for m in self.loo_models_
        ])
        return deltas


# -----------------------------------------------------------------------------
# Subsampled Monte-Carlo attributor (model-agnostic)
# -----------------------------------------------------------------------------


class SubsampledFairnessInfluence:
    """
    Monte-Carlo subset estimator of disparity influence (model-agnostic).

    Fits ``n_subsets`` models on random subsets (each point included
    independently with probability ``subset_frac``) and scores each training
    point by the difference in mean disparity between subsets that exclude
    and subsets that include it (maximum-sample-reuse, Data-Banzhaf style):

        scores[j] = mean_{S not containing j} F(S) - mean_{S containing j} F(S)

    matching the removal sign convention. Note the estimand is an *average*
    removal effect over subsets of size ~ subset_frac * n, not the
    full-dataset LOO effect; magnitudes are typically larger than LOO deltas.

    Parameters
    ----------
    metric, target : see :class:`FairnessInfluenceFunctions`. Callable
        metrics ``metric(scores, sensitive, y) -> float`` are evaluated
        on refit models directly, so smoothness is NOT required here
        (rank- or threshold-based metrics are fine).
    n_subsets : int, default=200
        Number of subset models to fit.
    subset_frac : float, default=0.5
        Inclusion probability per point.
    n_jobs : int, optional
        Parallel subset fits.
    random_state : int, optional
    verbose : int, default=1
    """

    def __init__(
        self,
        metric: MetricName = "dp",
        target: TargetName = "signed",
        n_subsets: int = 200,
        subset_frac: float = 0.5,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 1,
    ) -> None:
        self.metric = metric
        self.target = target
        self.n_subsets = n_subsets
        self.subset_frac = subset_frac
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        sensitive: ArrayLike | None = None,
    ) -> Self:
        """Fit subset models (the expensive step)."""
        _validate_metric(self.metric)
        _validate_target(self.target)
        if not 0.0 < self.subset_frac < 1.0:
            raise ValueError("subset_frac must be in (0, 1).")
        check_is_fitted_model(model)
        X_arr, y_arr = _prepare_fit_inputs(X, y)
        self.model_ = model
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self.sensitive_train_ = (
            None if sensitive is None else np.asarray(sensitive).ravel()
        )
        n = len(y_arr)
        rng = np.random.default_rng(self.random_state)
        masks = rng.uniform(size=(self.n_subsets, n)) < self.subset_frac

        def fit_one(mask: NDArray[np.bool_]) -> BaseEstimator | None:
            if mask.sum() < 2:
                return None
            try:
                return clone(self.model_).fit(X_arr[mask], y_arr[mask])
            except Exception:
                return None

        if self.n_jobs is None or self.n_jobs == 1:
            it = masks
            if self.verbose > 0:
                it = tqdm(masks, desc="Fitting subset models")
            models = [fit_one(m) for m in it]
        else:
            with tqdm_joblib(
                tqdm(total=self.n_subsets, desc="Fitting subset models",
                     disable=(self.verbose == 0))
            ):
                models = Parallel(n_jobs=self.n_jobs)(
                    delayed(fit_one)(m) for m in masks
                )
        ok = [i for i, m in enumerate(models) if m is not None]
        if len(ok) < self.n_subsets:
            warnings.warn(
                f"{self.n_subsets - len(ok)} subset fits failed and were "
                "dropped.",
                UserWarning,
            )
        self.subset_masks_ = masks[ok]
        self.subset_models_ = [models[i] for i in ok]
        return self

    def explain(
        self,
        X_audit: ArrayLike,
        *,
        y_audit: ArrayLike | None = None,
        sensitive_audit: ArrayLike | None = None,
        metric: MetricName | Callable | None = None,
        target: TargetName | None = None,
    ) -> NDArray[np.floating]:
        """Estimated removal effects (see class docstring for the estimand).

        The subset models fitted in ``fit`` are metric-agnostic, so pass
        ``metric`` / ``target`` to score a different disparity without
        refitting (as with :class:`RefitFairnessInfluence`).
        """
        check_is_fitted(self, ["subset_models_"])
        metric = self.metric if metric is None else metric
        target = self.target if target is None else target
        _validate_metric(metric)
        _validate_target(target)
        if sensitive_audit is None:
            raise ValueError("sensitive_audit is required.")
        X_audit = np.asarray(X_audit)
        s_audit = np.asarray(sensitive_audit).ravel()
        y_a = None if y_audit is None else np.asarray(y_audit).ravel()

        values = np.array([
            disparity_value(m, X_audit, s_audit, y_a, metric, target)
            for m in self.subset_models_
        ])
        inc = self.subset_masks_  # (T, n)
        n_in = inc.sum(axis=0).astype(float)
        n_out = (~inc).sum(axis=0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_in = (values[:, None] * inc).sum(axis=0) / n_in
            mean_out = (values[:, None] * ~inc).sum(axis=0) / n_out
        scores = mean_out - mean_in
        if np.isnan(scores).any():
            warnings.warn(
                "Some points were never included (or never excluded) in any "
                "subset; their scores are NaN. Increase n_subsets.",
                UserWarning,
            )
        return scores


# -----------------------------------------------------------------------------
# Validation / repair utilities
# -----------------------------------------------------------------------------


def group_removal_effect(
    model: BaseEstimator,
    X_train: ArrayLike,
    y_train: ArrayLike,
    indices: ArrayLike,
    X_audit: ArrayLike,
    sensitive_audit: ArrayLike,
    y_audit: ArrayLike | None = None,
    metric: MetricName = "dp",
    target: TargetName = "signed",
    refit_factory: Callable[[int], BaseEstimator] | None = None,
) -> float:
    """
    Actual disparity change from removing a set of training points.

    Refits once without ``indices`` and returns F(D \\ S) - F(D). Use to
    validate summed per-point scores (group effects are not additive in
    general).
    """
    X_arr, y_arr = _prepare_fit_inputs(X_train, y_train)
    idx = np.asarray(indices, dtype=np.intp).ravel()
    s_audit = np.asarray(sensitive_audit).ravel()
    y_a = None if y_audit is None else np.asarray(y_audit).ravel()
    base_value = disparity_value(model, X_audit, s_audit, y_a, metric, target)
    refit = _refit_without(model, X_arr, y_arr, idx, refit_factory)
    if refit is None:
        return float("nan")
    return (
        disparity_value(refit, X_audit, s_audit, y_a, metric, target) - base_value
    )


def disparity_removal_curve(
    scores: ArrayLike,
    model: BaseEstimator,
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_audit: ArrayLike,
    sensitive_audit: ArrayLike,
    y_audit: ArrayLike | None = None,
    metric: MetricName = "dp",
    target: TargetName = "absolute",
    fractions: ArrayLike | None = None,
    n_random: int = 5,
    random_state: int | None = None,
    refit_factory: Callable[[int], BaseEstimator] | None = None,
) -> dict:
    """
    Repair curve: disparity (and accuracy) after removing top-scored points.

    Removes the fraction of training points whose removal is predicted to
    *decrease* the disparity most (most negative removal scores first),
    refits, and recomputes the smoothed disparity, the hard disparity
    (classifiers), and accuracy on the audit set; compares against random
    removal.

    Parameters
    ----------
    scores : array-like of shape (n_train,)
        Disparity-influence scores (removal convention: positive = removal
        increases disparity). Any attributor in this module produces these.
    model, X_train, y_train : the fitted model and its training data.
    X_audit, sensitive_audit, y_audit : audit set.
    metric, target : disparity to track (default absolute gap).
    fractions : array-like, optional
        Removal fractions in [0, 1). Default: 11 points in [0, 0.2].
    n_random : int, default=5
        Random-removal replicates per fraction (0 to skip).
    random_state : int, optional
    refit_factory : callable(n_remaining) -> estimator, optional

    Returns
    -------
    result : dict with keys 'fractions', 'disparity', 'disparity_hard',
        'accuracy', 'random_disparity_mean', 'random_disparity_std',
        'base_disparity'. Classifier-only entries are NaN for regressors.
    """
    from sklearn.utils import check_random_state

    scores = np.asarray(scores, dtype=float).ravel()
    X_arr, y_arr = _prepare_fit_inputs(X_train, y_train)
    n = len(y_arr)
    if len(scores) != n:
        raise ValueError("scores must have length n_train.")
    s_audit = np.asarray(sensitive_audit).ravel()
    y_a = None if y_audit is None else np.asarray(y_audit).ravel()
    X_aud = np.asarray(X_audit)
    is_clf = is_classifier(model)

    if fractions is None:
        fractions = np.linspace(0.0, 0.2, 11)
    fractions = np.asarray(fractions, dtype=float)
    if (fractions < 0).any() or (fractions >= 1).any():
        raise ValueError("fractions must lie in [0, 1)")

    # most negative removal effect first (removal shrinks the disparity);
    # NaN-scored points go last
    order = np.argsort(np.nan_to_num(scores, nan=np.inf))

    def evaluate(remove_idx: NDArray[np.intp]) -> tuple[float, float, float]:
        if len(remove_idx) == 0:
            m = model
        else:
            m = _refit_without(model, X_arr, y_arr, remove_idx, refit_factory)
            if m is None:
                return float("nan"), float("nan"), float("nan")
        disp = disparity_value(m, X_aud, s_audit, y_a, metric, target)
        if is_clf and y_a is not None:
            hard = disparity_value_hard(m, X_aud, s_audit, y_a, metric, target)
            acc = float(np.mean(m.predict(X_aud) == y_a))
        else:
            hard, acc = float("nan"), float("nan")
        return disp, hard, acc

    rng = check_random_state(random_state)
    # Baseline = the full model, independent of which fractions were requested
    # (fractions need not start at 0 or be sorted).
    base_disp, _, _ = evaluate(np.array([], dtype=np.intp))
    disp = np.empty(len(fractions))
    hard = np.empty(len(fractions))
    acc = np.empty(len(fractions))
    rand = (
        np.empty((len(fractions), n_random)) if n_random > 0 else None
    )
    for i, f in enumerate(fractions):
        k = int(round(f * n))
        disp[i], hard[i], acc[i] = evaluate(order[:k])
        if n_random > 0:
            for r in range(n_random):
                if k == 0:
                    rand[i, r] = disp[i]
                else:
                    idx = rng.choice(n, size=k, replace=False)
                    rand[i, r], _, _ = evaluate(idx)

    if n_random > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rmean = np.nanmean(rand, axis=1)
            rstd = np.nanstd(rand, axis=1)
    else:
        rmean = np.array([])
        rstd = np.array([])

    return {
        "fractions": fractions,
        "disparity": disp,
        "disparity_hard": hard,
        "accuracy": acc,
        "random_disparity_mean": rmean,
        "random_disparity_std": rstd,
        "base_disparity": base_disp,
    }
