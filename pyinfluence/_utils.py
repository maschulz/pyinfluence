"""Utility functions for influence analysis.

Internal helpers (used by attributors or other utils): _compute_loss_sklearn,
_get_prediction_value, _value_at_test, tqdm_joblib, _top_influential_1d/2d.
Public API: top_influential, self_influence, influence_summary, find_mislabeled,
compare_attributors, aggregate_influence, influence_by_group, removal_curve,
stability_replicates.

NaN policy
----------
Refit-based attributors (LOO, Banzhaf, Bootstrap) legitimately produce NaN
scores where a point's effect is unmeasurable (failed refits, no OOB runs).
Every utility here follows one rule: NaN scores are *excluded* from rankings
and statistics, and the function warns with the count. NaN is never silently
ranked, averaged, or treated as zero.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Literal

import numpy as np
from joblib import parallel
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from tqdm import tqdm

from pyinfluence._base import BaseAttributor, check_is_fitted

# -----------------------------------------------------------------------------
# Internal helpers (used by LOO/Banzhaf/Bootstrap and/or public utils below)
# -----------------------------------------------------------------------------


def _has_predict_proba(model) -> bool:
    """Check whether *model* exposes a usable ``predict_proba`` method."""
    return callable(getattr(model, "predict_proba", None))


def _has_decision_function(model) -> bool:
    """Check whether *model* exposes a usable ``decision_function`` method."""
    return callable(getattr(model, "decision_function", None))


def _check_classifier_prediction_support(model) -> None:
    """Raise if the classifier has neither predict_proba nor decision_function."""
    if not _has_predict_proba(model) and not _has_decision_function(model):
        raise ValueError(
            f"{type(model).__name__} has neither predict_proba() nor "
            "decision_function(). Wrap it with "
            "sklearn.calibration.CalibratedClassifierCV to add "
            "predict_proba, or use InfluenceFunctions (closed-form) "
            "if the model is supported."
        )


def _true_class_values(
    probs: NDArray[np.floating],
    y: NDArray,
    classes: NDArray,
    missing: float,
) -> NDArray[np.floating]:
    """Pick each sample's true-class column from a (n, n_classes) array.

    The class -> column mapping is built once (not per sample); labels not in
    ``classes`` get ``missing``.
    """
    n_samples = probs.shape[0]
    class_index = {c: i for i, c in enumerate(classes)}
    col = np.fromiter(
        (class_index.get(yi, -1) for yi in y), dtype=int, count=n_samples
    )
    values = np.full(n_samples, missing, dtype=float)
    found = col >= 0
    values[found] = probs[np.arange(n_samples)[found], col[found]]
    return values


def _warn_nan_scores(scores: NDArray, func_name: str, action: str) -> int:
    """Warn (once) about NaN entries in a score array; return the count."""
    n_nan = int(np.isnan(scores).sum())
    if n_nan:
        warnings.warn(
            f"{func_name}: scores contain {n_nan} NaN value(s) (refit-based "
            f"attributors produce NaN where an effect is unmeasurable); "
            f"{action}.",
            UserWarning,
        )
    return n_nan


def _compute_loss_sklearn(
    model,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    is_classifier: bool,
) -> NDArray[np.floating]:
    """
    Compute per-sample loss for a sklearn model.

    Classification with predict_proba: negative log-likelihood of the true class.
    Classification without predict_proba: squared error on decision_function values
    (with a UserWarning, since this changes the loss semantics).
    Regression: squared error.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model with predict() and, for classifiers, predict_proba()
        or decision_function().
    X : ndarray of shape (n_samples, n_features)
        Features.
    y : ndarray of shape (n_samples,)
        True labels.
    is_classifier : bool
        Whether the model is a classifier.

    Returns
    -------
    loss : ndarray of shape (n_samples,)
        Per-sample loss.

    Raises
    ------
    ValueError
        If the classifier has neither predict_proba nor decision_function.
    """
    if is_classifier:
        if _has_predict_proba(model):
            probs = model.predict_proba(X)
            true_class_probs = _true_class_values(
                probs, y, model.classes_, missing=1e-10
            )
            true_class_probs = np.clip(true_class_probs, 1e-10, 1.0 - 1e-10)
            return -np.log(true_class_probs)

        # Fallback: decision_function + squared error
        _check_classifier_prediction_support(model)
        warnings.warn(
            f"{type(model).__name__} does not expose predict_proba(). "
            "Using squared error on decision_function() values instead of "
            "negative log-likelihood. Influence magnitudes will differ from "
            "classifiers that use NLL.",
            UserWarning,
        )
        decision = np.asarray(model.decision_function(X), dtype=float).ravel()
        # Binarize y to {-1, +1} using the model's class ordering
        y_binary = np.where(y == model.classes_[1], 1.0, -1.0)
        return (y_binary - decision) ** 2

    predictions = model.predict(X)
    return (y - predictions) ** 2


def _get_prediction_value(
    model,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Get per-sample prediction value for a classifier.

    If the model has predict_proba, returns the predicted probability of the
    true class. Otherwise falls back to decision_function values.

    Parameters
    ----------
    model : sklearn classifier
        Fitted classifier with predict_proba() or decision_function().
    X : ndarray of shape (n_samples, n_features)
        Features.
    y : ndarray of shape (n_samples,)
        True class labels.

    Returns
    -------
    values : ndarray of shape (n_samples,)
        Predicted probability of the true class, or decision_function values.

    Raises
    ------
    ValueError
        If the classifier has neither predict_proba nor decision_function.
    """
    if _has_predict_proba(model):
        probs = model.predict_proba(X)
        return _true_class_values(probs, y, model.classes_, missing=0.0)

    # Fallback: decision_function
    _check_classifier_prediction_support(model)
    warnings.warn(
        f"{type(model).__name__} does not expose predict_proba(). "
        "Using decision_function() values for prediction mode. "
        "Values represent decision boundary distance, not probabilities.",
        UserWarning,
    )
    return np.asarray(model.decision_function(X), dtype=float).ravel()


def _value_at_test(
    model,
    X_test: NDArray[np.floating],
    y_test: NDArray[np.floating] | None,
    mode: Literal["loss", "prediction"],
    is_classifier: bool,
) -> NDArray[np.floating]:
    """
    Per-test-point value (loss or prediction) for refit-based attributors.

    Used by LOOInfluence, BanzhafInfluence, and BootstrapInfluence to avoid
    duplicating the mode dispatch.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model with predict() and, for classifiers, predict_proba()
        or decision_function().
    X_test : ndarray of shape (n_test, n_features)
        Test features.
    y_test : ndarray of shape (n_test,) or None
        Test labels. Required for mode='loss' and for prediction mode with classifiers.
    mode : {'loss', 'prediction'}
        If 'loss', return per-sample loss. If 'prediction', return predicted value
        or true-class probability (or decision_function value as fallback).
    is_classifier : bool
        Whether the model is a classifier.

    Returns
    -------
    values : ndarray of shape (n_test,)
        Per-test-point value (loss or prediction).
    """
    if mode == "loss":
        if y_test is None:
            raise ValueError("y_test is required for loss mode")
        return _compute_loss_sklearn(model, X_test, y_test, is_classifier)
    if is_classifier:
        if y_test is None:
            raise ValueError("y_test is required for prediction mode with classifiers")
        return _get_prediction_value(model, X_test, y_test)
    return np.asarray(model.predict(X_test), dtype=float)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """
    Context manager to patch joblib to report into tqdm progress bar.

    This allows progress bars to work with joblib.Parallel by patching
    the BatchCompletionCallBack to update the tqdm progress bar as batches
    complete.

    Parameters
    ----------
    tqdm_object : tqdm
        The tqdm progress bar object to update.

    Examples
    --------
    >>> from joblib import Parallel, delayed
    >>> from tqdm import tqdm
    >>> from pyinfluence._utils import tqdm_joblib
    >>>
    >>> items = range(100)
    >>> with tqdm_joblib(tqdm(total=len(items))) as pbar:
    ...     results = Parallel(n_jobs=4)(
    ...         delayed(process_item)(item) for item in items
    ...     )
    """
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def top_influential(
    scores: ArrayLike, k: int = 10
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    Get indices of top-k most influential training samples.

    Parameters
    ----------
    scores : ndarray of shape (n_test, n_train) or (n_train,)
        Influence scores from explain().
    k : int, default=10
        Number of top samples to return.

    Returns
    -------
    helpful : ndarray of shape (n_test, k) or (k,)
        Indices of k most helpful training samples (highest positive influence).
    harmful : ndarray of shape (n_test, k) or (k,)
        Indices of k most harmful training samples (most negative influence).

    Notes
    -----
    NaN scores (failed refits in LOO/Bootstrap/Banzhaf) are excluded from
    both rankings, with a warning. Reversed ``np.argsort`` would otherwise
    rank NaN entries as the *most helpful* points. If exclusions leave fewer
    than k valid entries in some row, k shrinks accordingly.

    Examples
    --------
    >>> scores = attr.explain(X_test, y_test)
    >>> helpful, harmful = top_influential(scores[0], k=5)
    >>> print(f"Most helpful training samples: {helpful}")
    >>> print(f"Most harmful training samples: {harmful}")
    """
    scores = np.asarray(scores, dtype=float)
    _warn_nan_scores(scores, "top_influential", "excluded from the rankings")

    if scores.ndim == 1:
        return _top_influential_1d(scores, k)
    elif scores.ndim == 2:
        return _top_influential_2d(scores, k)
    else:
        raise ValueError(f"scores must be 1D or 2D array, got shape {scores.shape}")


def _top_influential_1d(
    scores: NDArray[np.floating], k: int
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Get top-k indices for 1D scores, ignoring NaN entries."""
    valid = np.where(~np.isnan(scores))[0]
    k = min(k, valid.size)

    order = valid[np.argsort(scores[valid])]

    # Most helpful: highest scores (descending order)
    helpful_indices = order[::-1][:k]

    # Most harmful: lowest scores (ascending order)
    harmful_indices = order[:k]

    return helpful_indices, harmful_indices


def _top_influential_2d(
    scores: NDArray[np.floating], k: int
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Get top-k indices for 2D scores (one per test sample)."""
    n_test, n_train = scores.shape
    # One k for all rows: the smallest per-row valid count caps it so the
    # result stays rectangular. (In practice NaN patterns are per-column.)
    k = min(k, int((~np.isnan(scores)).sum(axis=1).min()))

    # Pre-allocate result arrays
    helpful = np.empty((n_test, k), dtype=np.intp)
    harmful = np.empty((n_test, k), dtype=np.intp)

    for i in range(n_test):
        helpful[i], harmful[i] = _top_influential_1d(scores[i], k)

    return helpful, harmful


def self_influence(
    attributor: BaseAttributor,
    X_train: ArrayLike | None = None,
    y_train: ArrayLike | None = None,
) -> NDArray[np.floating]:
    """
    Compute self-influence: influence of each training sample on itself.

    Self-influence measures how much each training sample affects its own
    prediction/loss. High self-influence suggests the sample is unusual
    (outlier, mislabeled, or from a sparse region of feature space).

    Parameters
    ----------
    attributor : fitted BaseAttributor
        A fitted attributor (InfluenceFunctions or LOOInfluence).
    X_train : array-like, optional
        Training features. If None, uses data stored in attributor.
    y_train : array-like, optional
        Training labels. If None, uses data stored in attributor.

    Returns
    -------
    scores : ndarray of shape (n_train,)
        Self-influence score for each training sample.
        scores[i] = influence of training sample i on test sample i.

    Examples
    --------
    >>> attr = InfluenceFunctions(damping=1e-5, mode='loss')
    >>> attr.fit(model, X_train, y_train)
    >>> self_scores = self_influence(attr)
    >>> outlier_candidates = np.argsort(np.abs(self_scores))[::-1][:10]

    Notes
    -----
    For loss mode, high positive self-influence indicates a sample that
    strongly helps its own prediction (well-fitted). High negative self-influence
    indicates a sample that hurts its own prediction (potential mislabel).

    For prediction mode, the sign indicates whether removing the sample would
    increase or decrease the prediction at that point.

    Attributors that implement a ``_self_influence_diag`` method (currently
    InfluenceFunctions, LOOInfluence, BootstrapInfluence) compute the
    diagonal directly — O(n) memory instead of materializing the full
    (n_train, n_train) score matrix. The generic fallback builds the full
    matrix; for refit-free attributors this costs O(n^2) memory.
    """
    check_is_fitted(attributor, ["model_"])

    # Fast path: the attributor knows how to produce the diagonal directly
    # (only valid for its own stored training data).
    if X_train is None and y_train is None and hasattr(
        attributor, "_self_influence_diag"
    ):
        return attributor._self_influence_diag()

    # Use stored training data if not provided
    if X_train is None:
        if hasattr(attributor, 'X_train_raw_'):
            X_train = attributor.X_train_raw_
        elif hasattr(attributor, 'X_train_'):
            X_train = attributor.X_train_
        else:
            raise ValueError(
                "X_train not provided and attributor has no stored training data."
            )

    if y_train is None:
        if hasattr(attributor, 'y_train_'):
            y_train = attributor.y_train_
        else:
            # For prediction mode, y_train might not be needed
            pass

    # Check if mode requires y_train (all attributors use mode='loss' | 'prediction')
    mode = getattr(attributor, 'mode', None)
    if mode == 'loss' and y_train is None:
        raise ValueError(
            "y_train is required for loss mode but was not provided "
            "and attributor has no stored y_train_."
        )

    # Compute full influence matrix
    X_train = np.asarray(X_train)
    if y_train is not None:
        y_train = np.asarray(y_train).ravel()

    full_scores = attributor.explain(X_train, y_train)

    # Extract diagonal: influence of sample i on itself
    return np.diag(full_scores)


def influence_summary(
    scores: ArrayLike,
    percentiles: list[int] | None = None,
    zero_threshold: float = 1e-6,
) -> dict:
    """
    Compute summary statistics for influence scores.

    Parameters
    ----------
    scores : array-like of shape (n_test, n_train) or (n_train,)
        Influence scores from explain().
    percentiles : list of int, optional
        Percentiles to compute. Default is [25, 50, 75].
    zero_threshold : float, default=1e-6
        Threshold for considering a score as "zero" when computing sparsity.

    Returns
    -------
    summary : dict
        Dictionary containing:
        - mean : float
        - std : float
        - min : float
        - max : float
        - percentiles : dict mapping percentile -> value
        - sparsity : float (fraction of near-zero values, among non-NaN)
        - n_nan : int (number of NaN scores, excluded from all statistics)

    Examples
    --------
    >>> scores = attr.explain(X_test, y_test)
    >>> summary = influence_summary(scores)
    >>> print(f"Mean influence: {summary['mean']:.4f}")
    >>> print(f"Sparsity: {summary['sparsity']:.1%}")
    """
    scores = np.asarray(scores, dtype=float)

    if percentiles is None:
        percentiles = [25, 50, 75]

    n_nan = _warn_nan_scores(
        scores, "influence_summary", "excluded from all statistics"
    )
    valid = scores[~np.isnan(scores)]
    if valid.size == 0:
        nanval = float("nan")
        return {
            'mean': nanval, 'std': nanval, 'min': nanval, 'max': nanval,
            'percentiles': {p: nanval for p in percentiles},
            'sparsity': nanval, 'n_nan': n_nan,
        }

    percentile_values = np.percentile(valid, percentiles)
    percentile_dict = dict(zip(percentiles, percentile_values))

    # Sparsity: fraction of values within threshold of zero
    n_zero = np.sum(np.abs(valid) <= zero_threshold)
    sparsity = n_zero / valid.size

    return {
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
        'percentiles': percentile_dict,
        'sparsity': float(sparsity),
        'n_nan': n_nan,
    }


def find_mislabeled(
    attributor: BaseAttributor,
    threshold: float | Literal['auto'] = 'auto',
) -> NDArray[np.intp]:
    """
    Identify training samples that are likely mislabeled.

    Uses self-influence to detect unusual samples. Mislabeled samples
    typically have high absolute self-influence because they are outliers
    that require significant model capacity to fit.

    Parameters
    ----------
    attributor : fitted BaseAttributor
        A fitted attributor with mode='loss'.
    threshold : float or 'auto', default='auto'
        Threshold for flagging samples. If 'auto', uses z-score > 2.0
        on the absolute self-influence. If float, uses that z-score
        threshold.

    Returns
    -------
    indices : ndarray of shape (n_suspected,)
        Indices of training samples suspected of being mislabeled,
        sorted by absolute self-influence (highest first).

    Notes
    -----
    Mislabeled samples are unusual points that have high self-influence:
    they strongly affect their own predictions because the model must
    distort to accommodate their incorrect labels.

    Self-influence measures *atypicality* — how unusual a record is for
    the model — which is the trace label errors typically leave. Three
    caveats follow:

    - Plain per-sample training loss/error is an equally strong baseline;
      expect self-influence to match it, not beat it.
    - The signal is mechanism-specific: label errors are visible, but
      corrupted *features* (measurement errors) leave a different trace
      that self-influence does not read. A flagged record is "atypical",
      not proven mislabeled; an unflagged record can still be corrupted.
    - Detection degrades sharply when errors are plausible (records near
      the decision boundary) rather than gross.

    Before trusting the ranking on real data, run an injection
    experiment: corrupt a known subset the way you believe your errors
    look (including plausible, near-boundary corruptions), then check
    ``viz.plot_detection_curve(self_influence(attr), is_corrupted)``.

    Examples
    --------
    >>> attr = InfluenceFunctions(damping=1e-5, mode='loss')
    >>> attr.fit(model, X_train, y_train)
    >>> suspected = find_mislabeled(attr)
    >>> print(f"Suspected mislabeled samples: {suspected}")
    """
    check_is_fitted(attributor, ["model_"])

    mode = getattr(attributor, "mode", None)
    if mode != "loss":
        raise ValueError(
            "find_mislabeled requires an attributor with mode='loss'. "
            "Use loss mode to interpret self-influence as an error signal."
        )

    self_scores = self_influence(attributor)
    abs_scores = np.abs(self_scores)

    if threshold == 'auto':
        threshold = 2.0
    elif isinstance(
        threshold, (int, float, np.integer, np.floating)
    ) and threshold <= 0:
        raise ValueError(
            f"threshold must be positive when a number; got {threshold}."
        )

    # NaN self-influence (failed refits) is excluded from the statistics and
    # can never be flagged; without this, one NaN silently turns the result
    # into "no mislabeled samples found".
    _warn_nan_scores(
        abs_scores, "find_mislabeled",
        "excluded from the z-score statistics and never flagged",
    )

    # Compute z-scores on absolute self-influence
    mean_abs = np.nanmean(abs_scores)
    std_abs = np.nanstd(abs_scores)

    if not np.isfinite(std_abs) or std_abs < 1e-10:
        # All scores identical (or none valid), no outliers
        return np.array([], dtype=np.intp)

    z_scores = (abs_scores - mean_abs) / std_abs

    # Flag samples with unusually high absolute self-influence
    # (NaN compares False, so NaN entries are never flagged)
    suspected = np.where(z_scores > threshold)[0]

    # Sort by absolute self-influence (highest first)
    suspected = suspected[np.argsort(abs_scores[suspected])[::-1]]

    return suspected.astype(np.intp)


def compare_attributors(
    attr1: BaseAttributor,
    attr2: BaseAttributor,
    X_test: ArrayLike,
    y_test: ArrayLike | None = None,
    k: int = 10,
) -> dict:
    """
    Compare two attribution methods on the same test data.

    Parameters
    ----------
    attr1 : fitted BaseAttributor
        First fitted attributor.
    attr2 : fitted BaseAttributor
        Second fitted attributor.
    X_test : array-like of shape (n_test, n_features)
        Test samples.
    y_test : array-like of shape (n_test,), optional
        Test labels. Required if either attributor uses loss mode.
    k : int, default=10
        Number of top samples to use for overlap calculation.

    Returns
    -------
    comparison : dict
        Dictionary containing:
        - pearson : float (Pearson correlation)
        - spearman : float (Spearman rank correlation)
        - kendall : float (Kendall's tau)
        - top_k_overlap : float (Jaccard similarity of top-k sets)
        - n_nan_dropped : int (score pairs excluded because either method
          returned NaN, e.g. failed LOO refits)

    Examples
    --------
    >>> attr_if = InfluenceFunctions(damping=1e-5, mode='loss')
    >>> attr_loo = LOOInfluence(mode='loss')
    >>> comparison = compare_attributors(attr_if, attr_loo, X_test, y_test)
    >>> print(f"Pearson r: {comparison['pearson']:.3f}")
    """
    check_is_fitted(attr1, ["model_"])
    check_is_fitted(attr2, ["model_"])

    X_test = np.asarray(X_test)
    if y_test is not None:
        y_test = np.asarray(y_test).ravel()

    scores1 = attr1.explain(X_test, y_test)
    scores2 = attr2.explain(X_test, y_test)

    # Flatten for correlation computations; drop pairs where either side is
    # NaN (a single failed LOO refit would otherwise make every correlation
    # NaN with no explanation).
    flat1 = scores1.ravel()
    flat2 = scores2.ravel()
    pair_valid = ~(np.isnan(flat1) | np.isnan(flat2))
    n_nan_dropped = int((~pair_valid).sum())
    if n_nan_dropped:
        warnings.warn(
            f"compare_attributors: dropped {n_nan_dropped} score pair(s) "
            "where either attributor returned NaN.",
            UserWarning,
        )
    flat1 = flat1[pair_valid]
    flat2 = flat2[pair_valid]

    # Pearson correlation
    pearson_r, _ = stats.pearsonr(flat1, flat2)

    # Spearman correlation
    spearman_r, _ = stats.spearmanr(flat1, flat2)

    # Kendall's tau
    kendall_tau, _ = stats.kendalltau(flat1, flat2)

    # Top-k overlap (Jaccard similarity)
    # Get top-k indices for each test sample
    helpful1, _ = top_influential(scores1, k=k)
    helpful2, _ = top_influential(scores2, k=k)

    # Compute average Jaccard similarity across test samples
    if scores1.ndim == 1:
        set1 = set(helpful1)
        set2 = set(helpful2)
        jaccard = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 1.0
    else:
        jaccards = []
        for i in range(len(helpful1)):
            set1 = set(helpful1[i])
            set2 = set(helpful2[i])
            if set1 | set2:
                jaccards.append(len(set1 & set2) / len(set1 | set2))
            else:
                jaccards.append(1.0)
        jaccard = np.mean(jaccards)

    return {
        'pearson': float(pearson_r),
        'spearman': float(spearman_r),
        'kendall': float(kendall_tau),
        'top_k_overlap': float(jaccard),
        'n_nan_dropped': n_nan_dropped,
    }


def aggregate_influence(
    scores: ArrayLike,
    axis: int = 0,
    method: Literal['sum', 'mean', 'absmax'] = 'sum',
) -> NDArray[np.floating]:
    """
    Aggregate influence scores along an axis.

    Parameters
    ----------
    scores : array-like of shape (n_test, n_train) or (n_train,)
        Influence scores.
    axis : int, default=0
        Axis to aggregate over.
        - axis=0: aggregate over test samples (result shape: (n_train,))
        - axis=1: aggregate over train samples (result shape: (n_test,))
    method : {'sum', 'mean', 'absmax'}, default='sum'
        Aggregation method:
        - 'sum': sum of scores
        - 'mean': mean of scores
        - 'absmax': value with largest absolute magnitude

    Returns
    -------
    aggregated : ndarray
        Aggregated scores.

    Examples
    --------
    >>> scores = attr.explain(X_test, y_test)
    >>> # Total influence of each training sample across all test samples
    >>> total = aggregate_influence(scores, axis=0, method='sum')
    """
    scores = np.asarray(scores)

    if method == 'sum':
        return np.sum(scores, axis=axis)
    elif method == 'mean':
        return np.mean(scores, axis=axis)
    elif method == 'absmax':
        # Select value with largest absolute magnitude
        abs_scores = np.abs(scores)
        idx = np.argmax(abs_scores, axis=axis)
        return np.take_along_axis(scores, np.expand_dims(idx, axis=axis), axis=axis).squeeze(axis=axis)
    else:
        raise ValueError(f"method must be 'sum', 'mean', or 'absmax', got {method!r}")


def influence_by_group(
    scores: ArrayLike,
    groups: ArrayLike,
    method: Literal['sum', 'mean'] = 'sum',
) -> dict:
    """
    Aggregate influence scores by group membership.

    Parameters
    ----------
    scores : array-like of shape (n_test, n_train) or (n_train,)
        Influence scores.
    groups : array-like of shape (n_train,)
        Group label for each training sample.
    method : {'sum', 'mean'}, default='sum'
        Aggregation method, applied to *both* stages: 2D scores are first
        aggregated over the test axis with this method, then within groups.

    Returns
    -------
    group_influence : dict
        Dictionary mapping group label -> aggregate influence.

    Examples
    --------
    >>> # Influence by data source
    >>> sources = ['web', 'survey', 'web', 'api', ...]
    >>> by_source = influence_by_group(scores, sources)
    >>> print(f"Web data influence: {by_source['web']:.4f}")
    """
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    if method not in ('sum', 'mean'):
        raise ValueError(f"method must be 'sum' or 'mean', got {method!r}")

    # If 2D, first aggregate over test samples (same method as within-group,
    # so method='mean' is mean all the way through, not a sum/mean hybrid)
    if scores.ndim == 2:
        scores = aggregate_influence(scores, axis=0, method=method)

    unique_groups = np.unique(groups)
    result = {}

    for group in unique_groups:
        mask = groups == group
        group_scores = scores[mask]

        if method == 'sum':
            result[group] = float(np.sum(group_scores))
        else:
            result[group] = float(np.mean(group_scores))

    return result


def removal_curve(
    attributor: BaseAttributor,
    X_test: ArrayLike,
    y_test: ArrayLike,
    fractions: ArrayLike | None = None,
    direction: Literal["harmful", "helpful"] = "harmful",
    n_random: int = 5,
    random_state: int | None = None,
) -> dict:
    """
    Loss-vs-removal curve for validating an attributor against retraining.

    Refits the underlying model after dropping the top fraction of training
    points ranked by aggregate influence and compares the resulting mean test
    loss to a random-removal baseline. If influence scores are meaningful,
    removing the most harmful points should drop the loss faster than random;
    removing the most helpful points should raise it faster.

    Parameters
    ----------
    attributor : fitted BaseAttributor
        A fitted attributor. ``attributor.model_``, ``X_train_``, ``y_train_``
        are required (all attributors expose these after ``fit``).
    X_test, y_test : array-like
        Test set used to compute mean loss after each refit.
    fractions : array-like, optional
        Removal fractions in [0, 1). Defaults to 11 points in [0, 0.5].
    direction : {'harmful', 'helpful'}, default='harmful'
        Which end of the influence ranking to remove first.
    n_random : int, default=5
        Number of random-removal replicates per fraction. Set 0 to skip.
    random_state : int, optional
        Seed for the random-removal baseline.

    Returns
    -------
    result : dict
        Keys: ``fractions``, ``by_influence`` (mean test loss at each fraction
        when removing by influence), ``random_mean``, ``random_std``
        (mean/std across baseline replicates; empty arrays if n_random=0),
        ``direction``.

    Notes
    -----
    Cost: ``len(fractions) * (1 + n_random)`` refits of the underlying model.
    The first point in ``by_influence`` (fraction=0) equals the base model's
    test loss.
    """
    from sklearn.base import clone
    from sklearn.base import is_classifier as _is_classifier_sk
    from sklearn.utils import check_random_state

    check_is_fitted(attributor, ["model_", "X_train_", "y_train_"])

    # The curve evaluates test *loss*; a prediction-mode attributor's scores
    # rank by influence-on-prediction, whose sign has nothing to do with
    # harmful/helpful for loss — the curve would look valid and mean nothing.
    mode = getattr(attributor, "mode", None)
    if mode != "loss":
        raise ValueError(
            "removal_curve requires an attributor fitted with mode='loss'; "
            f"got mode={mode!r}. The curve ranks points as harmful/helpful "
            "for test loss and evaluates loss after removal."
        )

    # InfluenceFunctions stores intercept-augmented X_train_ and keeps the raw
    # in X_train_raw_. Other attributors store the raw X in X_train_. Refits
    # must use the raw, un-augmented matrix.
    X_train = np.asarray(
        getattr(attributor, "X_train_raw_", attributor.X_train_)
    )
    y_train = np.asarray(attributor.y_train_)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).ravel()
    n_train = X_train.shape[0]
    is_clf = _is_classifier_sk(attributor.model_)

    if fractions is None:
        fractions = np.linspace(0.0, 0.5, 11)
    fractions = np.asarray(fractions, dtype=float)
    if (fractions < 0).any() or (fractions >= 1).any():
        raise ValueError("fractions must lie in [0, 1)")

    # Aggregate influence on the test set: sum over test points → (n_train,)
    scores = attributor.explain(X_test, y_test)
    agg = np.sum(scores, axis=0) if scores.ndim == 2 else np.asarray(scores)

    # NaN-scored points (failed refits) can't be ranked; keep them out of the
    # removal front in either direction.
    _warn_nan_scores(
        agg, "removal_curve", "ranked last (never removed first)"
    )
    valid = np.where(~np.isnan(agg))[0]
    nan_idx = np.where(np.isnan(agg))[0]

    # Most harmful = most negative S (removing helps loss).
    # Most helpful = most positive S (removing hurts loss).
    if direction == "harmful":
        order = valid[np.argsort(agg[valid])]
    elif direction == "helpful":
        order = valid[np.argsort(agg[valid])[::-1]]
    else:
        raise ValueError("direction must be 'harmful' or 'helpful'")
    order = np.concatenate([order, nan_idx])

    base_estimator = attributor.model_

    def _refit_and_loss(remove_idx: np.ndarray) -> float:
        keep = np.ones(n_train, dtype=bool)
        keep[remove_idx] = False
        if keep.sum() < 2:
            return float("nan")
        # Classifier with a class disappearing after deletion can't be refit;
        # NaN that point so the curve is robust on imbalanced subsets.
        if is_clf and len(np.unique(y_train[keep])) < 2:
            return float("nan")
        m = clone(base_estimator)
        m.fit(X_train[keep], y_train[keep])
        return float(np.mean(_compute_loss_sklearn(m, X_test, y_test, is_clf)))

    rng = check_random_state(random_state)
    by_influence = np.empty(len(fractions))
    random_runs = np.empty((len(fractions), n_random)) if n_random > 0 else None

    for i, f in enumerate(fractions):
        n_remove = int(round(f * n_train))
        by_influence[i] = _refit_and_loss(order[:n_remove])
        if n_random > 0:
            for r in range(n_random):
                if n_remove == 0:
                    random_runs[i, r] = by_influence[i]
                else:
                    idx = rng.choice(n_train, size=n_remove, replace=False)
                    random_runs[i, r] = _refit_and_loss(idx)

    if n_random > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            random_mean = np.nanmean(random_runs, axis=1)
            random_std = np.nanstd(random_runs, axis=1)
    else:
        random_mean = np.array([])
        random_std = np.array([])

    return {
        "fractions": fractions,
        "by_influence": by_influence,
        "random_mean": random_mean,
        "random_std": random_std,
        "direction": direction,
    }


def stability_replicates(
    attributor: BaseAttributor,
    X_test: ArrayLike,
    y_test: ArrayLike | None = None,
    n_replicates: int = 20,
    random_state: int | None = None,
) -> NDArray[np.floating]:
    """
    Rerun an attributor on bootstrap-resampled training sets.

    Produces the ``(n_replicates, n_train)`` replicate-score array that
    ``viz.plot_top_k_stability`` consumes: for each replicate, the training
    set is resampled with replacement, the model and a fresh clone of the
    attributor are refit, and the resulting scores (summed over test points)
    are mapped back to the original training indices (duplicates sum).

    A sample that ranks in the top-k in most replicates is a robust finding;
    one that appears once is likely noise.

    Parameters
    ----------
    attributor : fitted BaseAttributor
        A fitted attributor; its ``model_``, ``X_train_``, ``y_train_`` and
        constructor parameters are reused. The attributor itself is not
        modified (a clone is fitted per replicate).
    X_test, y_test : array-like
        Test set passed to each replicate's ``explain``. ``y_test`` is
        required in loss mode.
    n_replicates : int, default=20
        Number of bootstrap replicates. Cost: ``n_replicates`` model refits
        plus attributor fits (which for refit-based attributors are
        themselves expensive).
    random_state : int, optional
        Seed for the bootstrap resampling.

    Returns
    -------
    replicate_scores : ndarray of shape (n_replicates, n_train)
        Aggregate influence per training sample per replicate, indexed by
        the *original* training indices. Samples not drawn in a replicate
        contribute 0 for that replicate. NaN scores within a replicate are
        treated as 0 (with a warning).

    Examples
    --------
    >>> attr = InfluenceFunctions(mode='loss').fit(model, X_train, y_train)
    >>> reps = stability_replicates(attr, X_test, y_test, n_replicates=12)
    >>> viz.plot_top_k_stability(reps, k=10)
    """
    from sklearn.base import clone

    check_is_fitted(attributor, ["model_", "X_train_", "y_train_"])
    X_train = np.asarray(
        getattr(attributor, "X_train_raw_", attributor.X_train_)
    )
    y_train = np.asarray(attributor.y_train_)
    X_test = np.asarray(X_test)
    if y_test is not None:
        y_test = np.asarray(y_test).ravel()
    n_train = X_train.shape[0]

    rng = np.random.default_rng(random_state)
    reps = np.zeros((n_replicates, n_train))
    any_nan = 0
    for r in range(n_replicates):
        boot = rng.choice(n_train, size=n_train, replace=True)
        model_r = clone(attributor.model_).fit(X_train[boot], y_train[boot])
        attr_r = clone(attributor).fit(model_r, X_train[boot], y_train[boot])
        scores_r = attr_r.explain(X_test, y_test)
        agg = np.sum(scores_r, axis=0) if scores_r.ndim == 2 else scores_r
        n_nan = int(np.isnan(agg).sum())
        if n_nan:
            any_nan += n_nan
            agg = np.nan_to_num(agg, nan=0.0)
        np.add.at(reps[r], boot, agg)
    if any_nan:
        warnings.warn(
            f"stability_replicates: {any_nan} NaN replicate score(s) were "
            "treated as 0 for the stability count.",
            UserWarning,
        )
    return reps
