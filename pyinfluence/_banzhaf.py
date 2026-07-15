"""Data Banzhaf influence implementation.

Banzhaf values measure the average marginal contribution of each training
point across all possible subsets. Unlike LOO which only considers the
full dataset minus one point, Banzhaf averages over subsets of all sizes.

Banzhaf trades Shapley's efficiency axiom (values summing to the total
performance) for cheaper estimation: uniform subset sampling instead of
permutation sampling, with lower variance at the same sample budget. It
keeps the symmetry and null-player axioms, which is what data-valuation
use cases typically need.
"""

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
    _prepare_explain_inputs,
    _prepare_fit_inputs,
    _validate_mode,
    check_is_fitted,
)
from pyinfluence._utils import _value_at_test, tqdm_joblib
from pyinfluence._validation import validate_refit_model, warn_if_data_mismatch

if TYPE_CHECKING:
    from typing import Self


def _compute_marginal_contribution(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    X_test: NDArray[np.floating],
    y_test: NDArray[np.floating] | None,
    idx: int,
    subset_mask: NDArray[np.bool_],
    is_classifier_: bool,
    mode: Literal["loss", "prediction"],
) -> NDArray[np.floating] | None:
    """
    Compute marginal contribution of point idx for a given subset.

    For loss mode: Returns v(S ∪ {idx}) - v(S) where v is negative loss.
    For prediction mode: Returns prediction change.

    Parameters
    ----------
    model : sklearn estimator
        The model to clone and refit.
    X : ndarray
        Full training features.
    y : ndarray
        Full training targets.
    X_test : ndarray
        Test features.
    y_test : ndarray or None
        Test targets. Required for loss mode.
    idx : int
        Index of the point whose marginal contribution to compute.
    subset_mask : ndarray of bool
        Boolean mask for subset S (excluding idx).
    is_classifier_ : bool
        Whether the model is a classifier.
    mode : {'loss', 'prediction'}
        Whether to compute influence on loss or prediction.

    Returns
    -------
    marginal : ndarray of shape (n_test,) or None
        Per-test-point marginal contribution, or None if fitting failed.
    """
    # Fit model on S (without idx). Only the empty subset is excluded a
    # priori (no estimator can fit 0 samples); any other unfittable subset
    # (e.g. single-class for a classifier) is dropped by the try/except
    # below. Excluding small-but-fittable subsets would bias the estimator
    # in exactly the small-n regime where Banzhaf is recommended.
    if subset_mask.sum() < 1:
        return None

    try:
        model_without = clone(model)
        model_without.fit(X[subset_mask], y[subset_mask])
    except Exception:
        return None

    # Fit model on S ∪ {idx}
    mask_with = subset_mask.copy()
    mask_with[idx] = True

    try:
        model_with = clone(model)
        model_with.fit(X[mask_with], y[mask_with])
    except Exception:
        return None

    val_without = _value_at_test(
        model_without, X_test, y_test, mode, is_classifier_
    )
    val_with = _value_at_test(model_with, X_test, y_test, mode, is_classifier_)
    # Loss: marginal = loss(without) - loss(with). Prediction: val(with) - val(without)
    if mode == "loss":
        return val_without - val_with
    return val_with - val_without


def _process_sample_batch(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    X_test: NDArray[np.floating],
    y_test: NDArray[np.floating] | None,
    idx: int,
    n_samples: int,
    is_classifier_: bool,
    rng: np.random.Generator,
    mode: Literal["loss", "prediction"],
) -> tuple[NDArray[np.floating], int]:
    """
    Compute Monte Carlo Banzhaf estimates for a single training point.

    Parameters
    ----------
    model : sklearn estimator
        The model to clone and refit.
    X : ndarray
        Full training features.
    y : ndarray
        Full training targets.
    X_test : ndarray
        Test features.
    y_test : ndarray
        Test targets.
    idx : int
        Index of the training point.
    n_samples : int
        Number of Monte Carlo samples.
    is_classifier_ : bool
        Whether the model is a classifier.
    rng : numpy Generator
        Random number generator.

    Returns
    -------
    marginal_sum : ndarray of shape (n_test,)
        Sum of marginal contributions across MC samples.
    marginal_sq_sum : ndarray of shape (n_test,)
        Sum of squared marginal contributions (for the standard error).
    valid_count : int
        Number of valid (non-failed) samples.
    """
    n_train = X.shape[0]
    n_test = X_test.shape[0]

    marginal_sum = np.zeros(n_test)
    marginal_sq_sum = np.zeros(n_test)
    valid_count = 0

    other_indices = np.array([i for i in range(n_train) if i != idx])

    for _ in range(n_samples):
        # Sample random subset S ⊆ {1,...,n} \ {idx}
        # Each other point is included with probability 0.5
        include = rng.random(len(other_indices)) < 0.5

        # Build subset mask
        subset_mask = np.zeros(n_train, dtype=bool)
        subset_mask[other_indices[include]] = True

        # Compute marginal contribution
        marginal = _compute_marginal_contribution(
            model, X, y, X_test, y_test, idx, subset_mask, is_classifier_, mode
        )

        if marginal is not None:
            marginal_sum += marginal
            marginal_sq_sum += marginal**2
            valid_count += 1

    return marginal_sum, marginal_sq_sum, valid_count


class BanzhafInfluence(BaseAttributor):
    r"""
    Data Banzhaf influence via Monte Carlo estimation.

    Banzhaf values measure the average marginal contribution of each training
    point to model performance across random subsets of the training data.

    For each training point i, the Banzhaf value is:

    .. math::

        \phi_i = \frac{1}{2^{n-1}} \sum_{S \subseteq D \setminus \{i\}}
                 [v(S \cup \{i\}) - v(S)]

    where v(S) is the model's performance (negative loss) when trained on S.

    This is estimated via Monte Carlo sampling: for each point i, we sample
    random subsets S, fit models with and without i, and average the marginal
    contributions.

    Parameters
    ----------
    mode : {'loss', 'prediction'}, default='loss'
        How to measure influence:

        - 'loss': influence = L(x; D \ z_i) - L(x; D). Works for both
          classification and regression. Requires y_test.
          Positive = adding sample decreases loss = sample is helpful.

        - 'prediction': For classifiers, influence = P(y|x; D ∪ z_i) - P(y|x; D)
          (requires y_test to identify the true class). For regressors,
          influence = f(x; D ∪ z_i) - f(x; D), the change in the raw prediction.
          Positive values mean the sample increases the prediction or
          true-class probability.

    n_samples : int, default=1000
        Number of random subsets to sample per training point for Monte Carlo
        estimation. More samples give lower variance but higher computational
        cost.

    n_jobs : int, default=None
        Number of parallel jobs for subset evaluation. None means sequential.
        -1 means use all available cores.

    random_state : int, default=None
        Random seed for reproducible subset sampling.

    verbose : int, default=1
        If 1, show progress bar during explain. If 0, no progress output.

    Attributes
    ----------
    model_ : sklearn estimator
        Reference to the original fitted model.
    X_train_ : ndarray
        Training features.
    y_train_ : ndarray
        Training targets.
    is_classifier_ : bool
        Whether the model is a classifier.
    n_train_ : int
        Number of training samples.
    scores_std_ : ndarray of shape (n_test, n_train)
        Monte Carlo standard error of the most recent ``explain`` call's
        scores (NaN where fewer than two subset pairs succeeded). Use it to
        judge whether a ranking is signal or sampling noise given this
        training set; for stability under training-data resampling (a
        different question) see ``pyinfluence.stability_replicates``, e.g. via
        ``viz.plot_top_influencers(scores, xerr=attr.scores_std_[i])``.

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> model = Ridge().fit(X_train, y_train)
    >>> banzhaf = BanzhafInfluence(mode='loss', n_samples=500, n_jobs=-1, random_state=42)
    >>> banzhaf.fit(model, X_train, y_train)
    >>> scores = banzhaf.explain(X_test, y_test)

    Notes
    -----
    Computational complexity is O(n_samples × n_train × T) where T is the time
    to fit a single model. For large datasets or slow models, consider using
    fewer Monte Carlo samples or parallelization.

    Unlike LOOInfluence and BootstrapInfluence, whose ``fit`` caches the
    refitted models, the Banzhaf refit loop depends on nothing that ``fit``
    could precompute per test set, and caching the 2 × n_samples × n_train
    subset models would be prohibitive. The full Monte Carlo cost is
    therefore paid on **every** ``explain`` call. Explain all test points of
    interest in a single call rather than looping.

    Points whose subset refits *all* fail (e.g. the only member of a rare
    class) receive NaN scores and a warning: their value is unmeasurable
    with this configuration, not zero.

    Unlike influence functions, Banzhaf values are model-agnostic and work
    with any sklearn estimator that implements fit() and predict()/predict_proba().
    """

    def __init__(
        self,
        mode: Literal["loss", "prediction"] = "loss",
        n_samples: int = 1000,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 1,
    ) -> None:
        self.mode = mode
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, model: BaseEstimator, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Fit the attributor to a trained model and its training data.

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
        validate_refit_model(model)
        X, y = _prepare_fit_inputs(X, y)
        warn_if_data_mismatch(model, X, y)
        self.model_ = model
        self.X_train_ = X
        self.y_train_ = y
        self.is_classifier_ = is_classifier(model)
        self.n_train_ = X.shape[0]

        return self

    def explain(
        self, X_test: ArrayLike, y_test: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        """
        Compute Banzhaf influence scores of training samples on test samples.

        Parameters
        ----------
        X_test : array-like of shape (m_samples, n_features) or (n_features,)
            Test samples to explain. A single point may be 1D; result shape (1, n_train).
        y_test : array-like of shape (m_samples,) or scalar, optional
            Test labels. Required for loss mode and for prediction mode with classifiers.

        Returns
        -------
        scores : ndarray of shape (m_samples, n_samples)
            scores[i, j] = Banzhaf value of training sample j for test sample i.
            NaN where every subset refit for a point failed (with a warning).
            The Monte Carlo standard error is stored in ``scores_std_``.

            For mode='loss':
                Positive = helpful (adding sample decreases loss).
                Negative = harmful (adding sample increases loss).

            For mode='prediction':
                - Classifiers: Positive = helpful (adding sample increases probability
                  of true class).
                - Regressors: Positive = helpful (adding sample increases prediction).

        Raises
        ------
        NotFittedError
            If called before fit().
        ValueError
            If y_test is None when required (loss mode, or prediction mode with classifiers).
        """
        check_is_fitted(self, ["model_", "X_train_", "y_train_"])
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
        n_train = self.n_train_

        # Initialize RNG and generate per-point seeds for reproducibility
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=n_train)

        if self.n_jobs is None or self.n_jobs == 1:
            # Sequential execution - use same per-point seeds as parallel
            idx_range = range(n_train)
            if self.verbose > 0:
                idx_range = tqdm(idx_range, desc="Computing Banzhaf influence")
            results = [
                _process_sample_batch(
                    self.model_,
                    self.X_train_,
                    self.y_train_,
                    X_test,
                    y_test,
                    idx,
                    self.n_samples,
                    self.is_classifier_,
                    np.random.default_rng(seeds[idx]),
                    self.mode,
                )
                for idx in idx_range
            ]
        else:
            # Parallel execution with optional progress bar
            with tqdm_joblib(
                tqdm(total=n_train, desc="Computing Banzhaf influence", disable=(self.verbose == 0))
            ):
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_process_sample_batch)(
                        self.model_,
                        self.X_train_,
                        self.y_train_,
                        X_test,
                        y_test,
                        idx,
                        self.n_samples,
                        self.is_classifier_,
                        np.random.default_rng(seeds[idx]),
                        self.mode,
                    )
                    for idx in range(n_train)
                )

        scores = np.full((n_test, n_train), np.nan)
        scores_std = np.full((n_test, n_train), np.nan)
        unmeasured = []
        for idx, (marginal_sum, marginal_sq_sum, valid_count) in enumerate(results):
            if valid_count == 0:
                # No subset produced a valid with/without pair: the value is
                # unmeasurable with this configuration, NOT zero.
                unmeasured.append(idx)
                continue
            mean = marginal_sum / valid_count
            scores[:, idx] = mean
            if valid_count >= 2:
                # Standard error of the Monte Carlo mean
                sample_var = (
                    marginal_sq_sum - valid_count * mean**2
                ) / (valid_count - 1)
                scores_std[:, idx] = np.sqrt(
                    np.maximum(sample_var, 0.0) / valid_count
                )
        if unmeasured:
            warnings.warn(
                f"All {self.n_samples} subset refits failed for "
                f"{len(unmeasured)} training point(s) (e.g. every subset "
                "without the point is single-class); their scores are NaN: "
                f"{unmeasured[:10]}{'...' if len(unmeasured) > 10 else ''}",
                UserWarning,
            )
        self.scores_std_ = scores_std

        return scores
