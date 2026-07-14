"""Fairness auditing vocabulary and workflow over the functional engine.

Which training points, if removed, would most change the disparity a
fairness audit measures? This module contributes exactly two things:

1. **Vocabulary**: :func:`disparity` maps audit metric names — demographic
   parity ('dp'), equal opportunity ('eopp'), FPR gap ('fpr'), worst-group
   loss ('worst_group_loss') — onto the domain-neutral builders in
   :mod:`pyinfluence.functionals`, handling the fairness-specific details
   (sensitive-attribute conventions, the model's positive class for label
   conditioning).
2. **Workflow**: metric evaluation on a fitted model
   (:func:`disparity_value`, :func:`disparity_value_hard`) and
   retrain-based validation/repair (:func:`group_removal_effect`,
   :func:`disparity_removal_curve`, plotted by
   ``viz.plot_disparity_curve``).

Attribution itself is the generic engine:

>>> from pyinfluence import FunctionalInfluence
>>> from pyinfluence.fairness import disparity
>>> F = disparity("eopp", sensitive_audit, target_of=model)
>>> scores = FunctionalInfluence(F, target="absolute").fit(model, X, y).explain(X_audit, y_audit)

Estimand
--------
For a disparity functional F on a fixed audit set — e.g. the demographic
parity gap ``F = mean_{a=a1} p(x) - mean_{a=a0} p(x)`` with the binary
sensitive attribute's values ordered a0 < a1 — every estimator attributes
the per-point removal effect ``score[j] ~= F(D \\ {z_j}) - F(D)``. Positive
scores mark training points whose removal *increases* the gap. With
``target='absolute'`` the functional is |F|, so negative scores always mean
"removing this point shrinks the disparity magnitude".

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

import dataclasses
import warnings
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, is_classifier

from pyinfluence._base import _prepare_fit_inputs
from pyinfluence._functional import (
    Functional,
    TargetName,
    _refit_without,
    _validate_target,
    functional_value,
)
from pyinfluence._validation import validate_labels_in_classes
from pyinfluence.functionals import group_gap, worst_group_mean

__all__ = [
    "DISPARITY_METRICS",
    "disparity",
    "disparity_value",
    "disparity_value_hard",
    "group_removal_effect",
    "disparity_removal_curve",
]

DISPARITY_METRICS = ("dp", "eopp", "fpr", "worst_group_loss")

MetricName = Literal["dp", "eopp", "fpr", "worst_group_loss"]


def _positive_label(model: BaseEstimator):
    """The label whose score the engine's score-functionals consume.

    For classifiers this is ``classes_[1]`` — the class predict_proba[:, 1]
    (or a positive decision value) refers to. Conditioning eopp/fpr on any
    other value would silently compute a different metric (e.g. swap the
    two) whenever labels are not {0, 1}.
    """
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 2:
        return classes[1]
    return 1.0


def _validate_audit_labels(model: BaseEstimator, y: NDArray | None) -> None:
    """For classifier metrics that use y, insist labels come from classes_."""
    if y is None or not is_classifier(model):
        return
    classes = getattr(model, "classes_", None)
    if classes is not None:
        validate_labels_in_classes(y, np.asarray(classes), name="y (audit)")


def _as_disparity_functional(
    metric, sensitive, model: BaseEstimator | None
) -> Functional:
    """Resolve a metric argument (name or Functional) for the utilities."""
    if isinstance(metric, Functional):
        return metric
    if isinstance(metric, str):
        return disparity(metric, sensitive, target_of=model)
    raise TypeError(
        f"metric must be one of {DISPARITY_METRICS} or a Functional (build "
        "custom metrics with pyinfluence.functionals or "
        "pyinfluence.Functional); got "
        f"{type(metric).__name__}."
    )


def disparity(
    metric: MetricName,
    sensitive: ArrayLike,
    *,
    target_of: BaseEstimator | None = None,
    pos_label=None,
) -> Functional:
    """
    Build the functional for a named audit metric, bound to the audit set's
    sensitive attribute.

    Thin vocabulary over :mod:`pyinfluence.functionals`: 'dp' is
    ``group_gap(sensitive)``; 'eopp'/'fpr' are label-conditioned group
    gaps; 'worst_group_loss' is ``worst_group_mean(sensitive, of='losses')``.
    All carry analytic gradients. For anything else (Cohen's d, quantile
    gaps, custom statistics), use the builders in
    :mod:`pyinfluence.functionals` directly.

    Parameters
    ----------
    metric : {'dp', 'eopp', 'fpr', 'worst_group_loss'}
        Audit metric name.
    sensitive : array-like of shape (m,)
        The audit set's sensitive attribute. The returned functional is
        bound to these rows: evaluate it only on the row-aligned audit set.
        Binary for the gap metrics (gap = E[.|a1] - E[.|a0], a0 < a1 in
        sort order); any number of groups for 'worst_group_loss'.
    target_of : fitted classifier, optional
        Convenience for 'eopp'/'fpr': resolves ``pos_label`` from the
        model's ``classes_[1]`` — the class whose score the engine feeds
        the functional.
    pos_label : optional
        Explicit positive label for 'eopp'/'fpr' (which audit rows count as
        true positives/negatives). Required for those metrics unless
        ``target_of`` is given.

    Returns
    -------
    functional : Functional
        ``of='scores'`` for the gap metrics, ``of='losses'`` for
        'worst_group_loss'.

    Examples
    --------
    >>> F = disparity("eopp", a_audit, target_of=model)
    >>> attr = FunctionalInfluence(F, target="absolute").fit(model, X, y)
    >>> scores = attr.explain(X_audit, y_audit)
    """
    if metric not in DISPARITY_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Available: {DISPARITY_METRICS}. "
            "For custom metrics use pyinfluence.functionals."
        )

    if metric == "worst_group_loss":
        func = worst_group_mean(sensitive, of="losses")
        return dataclasses.replace(func, name=metric)

    if metric == "dp":
        keep = None
    else:
        if pos_label is None and target_of is not None:
            pos_label = _positive_label(target_of)
        if pos_label is None:
            raise ValueError(
                f"metric={metric!r} needs the positive label: pass "
                "target_of=model (resolves classes_[1]) or pos_label=..."
            )
        pos = pos_label
        if metric == "eopp":
            def keep(y):
                return np.asarray(y).ravel() == pos
        else:  # fpr
            def keep(y):
                return np.asarray(y).ravel() != pos

    func = group_gap(sensitive, keep=keep, of="scores")
    return dataclasses.replace(func, name=metric)


def _metric_needs_y(metric) -> bool:
    return metric in ("eopp", "fpr", "worst_group_loss")


# -----------------------------------------------------------------------------
# Metric values on a fitted model
# -----------------------------------------------------------------------------


def disparity_value(
    model: BaseEstimator,
    X: ArrayLike,
    sensitive: ArrayLike,
    y: ArrayLike | None = None,
    metric: MetricName | Functional = "dp",
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
        'worst_group_loss' any number of groups is allowed. Ignored when
        ``metric`` is a Functional (which is already bound to its rows).
    y : array-like of shape (m,), optional
        Audit labels. Required for 'eopp', 'fpr', and 'worst_group_loss'.
    metric : metric name or Functional, default='dp'
        A named audit metric (the positive class for 'eopp'/'fpr' is
        resolved from the model's ``classes_``), or any Functional (e.g.
        ``functionals.cohens_d(sensitive)``).
    target : {'signed', 'absolute'}, default='signed'
        Report the signed value or its absolute value.

    Returns
    -------
    value : float
    """
    if isinstance(metric, str) and _metric_needs_y(metric) and y is None:
        raise ValueError(f"y is required for metric={metric!r}.")
    if isinstance(metric, str):
        _validate_audit_labels(model, y)
    func = _as_disparity_functional(metric, sensitive, model)
    return functional_value(model, X, func, y, target)


def disparity_value_hard(
    model: BaseEstimator,
    X: ArrayLike,
    sensitive: ArrayLike,
    y: ArrayLike | None = None,
    metric: MetricName | Functional = "dp",
    target: TargetName = "signed",
    threshold: float = 0.5,
) -> float:
    """
    Hard (thresholded-decision) disparity on an audit set.

    Same conventions as :func:`disparity_value` but computed from thresholded
    positive-class probabilities (classifiers only): 'dp' is the selection
    rate gap, 'eopp' the TPR gap, 'fpr' the FPR gap. 'worst_group_loss' uses
    0/1 error instead of the model loss. A Functional metric is applied to
    the 0/1 decision vector instead of the smoothed scores.

    ``threshold`` applies to ``predict_proba``; for classifiers exposing only
    ``decision_function`` the decision boundary is fixed at 0 and
    ``threshold`` is ignored.
    """
    _validate_target(target)
    if not is_classifier(model):
        raise ValueError("disparity_value_hard requires a classifier.")
    X = np.asarray(X)
    s = np.asarray(sensitive).ravel()
    if isinstance(metric, str) and _metric_needs_y(metric) and y is None:
        raise ValueError(f"y is required for metric={metric!r}.")
    if isinstance(metric, str):
        _validate_audit_labels(model, y)
    y_arr = None if y is None else np.asarray(y).ravel()

    # decisions is a 0/1 indicator of predicting the positive class
    # (classes_[1]); label-space comparisons below must use the same encoding.
    if callable(getattr(model, "predict_proba", None)):
        decisions = (model.predict_proba(X)[:, 1] >= threshold).astype(float)
    else:
        decisions = (
            np.asarray(model.decision_function(X)).ravel() >= 0
        ).astype(float)

    if metric == "worst_group_loss":
        y01 = (y_arr == _positive_label(model)).astype(float)
        err = (decisions != y01).astype(float)
        return float(max(err[s == g].mean() for g in np.unique(s)))

    func = _as_disparity_functional(metric, s, model)
    value = func(decisions, y_arr)
    return abs(value) if target == "absolute" else value


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
    metric: MetricName | Functional = "dp",
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
        disparity_value(refit, X_audit, s_audit, y_a, metric, target)
        - base_value
    )


def disparity_removal_curve(
    scores: ArrayLike,
    model: BaseEstimator,
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_audit: ArrayLike,
    sensitive_audit: ArrayLike,
    y_audit: ArrayLike | None = None,
    metric: MetricName | Functional = "dp",
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
    removal. Plot with ``viz.plot_disparity_curve``.

    Parameters
    ----------
    scores : array-like of shape (n_train,)
        Disparity-influence scores (removal convention: positive = removal
        increases disparity). Any engine estimator produces these.
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
