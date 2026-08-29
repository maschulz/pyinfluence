"""Bootstrap (out-of-bag) influence implementation."""

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


def _bootstrap_indices(
    n_samples: int,
    n_estimators: int,
    rng: np.random.Generator,
) -> list[NDArray[np.intp]]:
    """
    Draw bootstrap samples: for each run, sample n_samples indices with replacement.

    Returns
    -------
    in_bag_list : list of ndarray
        in_bag_list[b] is the array of training indices in the in-bag set for run b.
    """
    in_bag_list = []
    for _ in range(n_estimators):
        indices = rng.integers(0, n_samples, size=n_samples)
        in_bag_list.append(indices)
    return in_bag_list


def _fit_bootstrap_model(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    indices_b: NDArray[np.intp],
) -> BaseEstimator | None:
    """
    Fit a clone of the model on the bootstrap sample indices_b.

    Parameters
    ----------
    model : sklearn estimator
        The model to clone and refit.
    X : ndarray
        Full training features.
    y : ndarray
        Full training targets.
    indices_b : ndarray of int
        In-bag indices for this bootstrap run.

    Returns
    -------
    fitted_model : sklearn estimator or None
        The refitted model, or None if fitting failed.
    """
    try:
        cloned = clone(model)
        cloned.fit(X[indices_b], y[indices_b])
        return cloned
    except Exception:
        return None


class BootstrapInfluence(BaseAttributor):
    r"""
    Bootstrap (out-of-bag) influence.

    Trains B models on bootstrap samples. For each training point i, influence
    is the mean loss (or prediction) over runs where i was out-of-bag minus
    the mean over runs where i was in-bag. Positive = helpful.

    This is a presence-vs-absence contrast, but not a clean one-point effect:
    conditional on being in-bag, a point appears with expected multiplicity
    ``1 / (1 - (1 - 1/n) ** n) ≈ 1.58``, so magnitudes are inflated by roughly
    that factor relative to a single-point (leave-one-out) removal, and the
    factor depends on n and the model's nonlinearity. Treat the *ranking* as
    the reliable output; for calibrated magnitudes use ``LOOInfluence`` or
    ``RefitFunctionalInfluence``.

    Parameters
    ----------
    mode : {'loss', 'prediction'}, default='loss'
        How to measure influence:

        - 'loss': influence = mean L over OOB runs minus mean L over in-bag
          runs. Requires y_test. Positive = helpful.

        - 'prediction': Same structure with predicted value or true-class
          probability. Positive = helpful.

    n_estimators : int, default=50
        Number of bootstrap models to fit. More gives lower variance but
        higher cost (50–200 is a typical range).

    random_state : int or None, default=None
        Seed for bootstrap index sampling.

    n_jobs : int or None, default=None
        Number of parallel jobs for fitting bootstrap models. None means
        sequential; -1 means all cores.

    min_oob_runs : int, default=3
        Minimum number of OOB runs required per training point. If a point
        appears in-bag in (nearly) all runs, it will have few OOB runs; we
        warn and use available runs (or NaN if zero).

    verbose : int, default=1
        If 1, show progress during fit and explain. If 0, no progress output.

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
    bootstrap_models_ : list
        Fitted bootstrap models. None entries indicate failed fits.
    in_bag_indices_ : list of ndarray
        in_bag_indices_[b] = indices of training samples in bag for run b.
    failed_estimator_indices_ : list of int
        Indices b where bootstrap fit failed.
    scores_std_ : ndarray of shape (n_test, n_train)
        Standard error of the most recent ``explain`` call's scores,
        combining the OOB and in-bag run variances
        (sqrt(var_oob/n_oob + var_in/n_in)). NaN where either side has
        fewer than two runs. Use to judge whether a ranking is signal or sampling noise given
        this training set; for stability under training-data resampling (a
        different question) see ``pyinfluence.stability_replicates``.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model = RandomForestRegressor().fit(X_train, y_train)
    >>> attr = BootstrapInfluence(mode="loss", n_estimators=50, random_state=42)
    >>> attr.fit(model, X_train, y_train)
    >>> scores = attr.explain(X_test, y_test)
    """

    def __init__(
        self,
        mode: Literal["loss", "prediction"] = "loss",
        n_estimators: int = 50,
        random_state: int | None = None,
        n_jobs: int | None = None,
        min_oob_runs: int = 3,
        verbose: int = 1,
    ) -> None:
        self.mode = mode
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_oob_runs = min_oob_runs
        self.verbose = verbose

    def fit(self, model: BaseEstimator, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Fit the attributor by training bootstrap models.

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
        """
        _validate_mode(self.mode)
        validate_refit_model(model)
        X, y = _prepare_fit_inputs(X, y)
        warn_if_data_mismatch(model, X, y)
        n_train = X.shape[0]
        self.model_ = model
        self.X_train_ = X
        self.y_train_ = y
        self.is_classifier_ = is_classifier(model)

        rng = np.random.default_rng(self.random_state)
        in_bag_list = _bootstrap_indices(n_train, self.n_estimators, rng)
        self.in_bag_indices_ = in_bag_list

        iterator = range(self.n_estimators)
        if self.verbose > 0:
            iterator = tqdm(iterator, desc="Fitting bootstrap models")
        if self.n_jobs is None or self.n_jobs == 1:
            bootstrap_models = [
                _fit_bootstrap_model(model, X, y, in_bag_list[b]) for b in iterator
            ]
        else:
            with tqdm_joblib(
                tqdm(
                    total=self.n_estimators,
                    desc="Fitting bootstrap models",
                    disable=(self.verbose == 0),
                )
            ):
                bootstrap_models = Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_bootstrap_model)(model, X, y, in_bag_list[b])
                    for b in range(self.n_estimators)
                )

        self.failed_estimator_indices_ = [
            b for b, m in enumerate(bootstrap_models) if m is None
        ]
        if self.failed_estimator_indices_:
            warnings.warn(
                f"Bootstrap fit failed for {len(self.failed_estimator_indices_)} "
                "runs. Influence scores will use only successful runs.",
                UserWarning,
                stacklevel=2,
            )
        self.bootstrap_models_ = bootstrap_models
        return self

    def explain(
        self, X_test: ArrayLike, y_test: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        """
        Compute bootstrap (OOB) influence scores.

        Parameters
        ----------
        X_test : array-like of shape (m_samples, n_features) or (n_features,)
            Test samples. A single point may be 1D; result shape (1, n_train).
        y_test : array-like of shape (m_samples,) or scalar, optional
            Test labels. Required when mode='loss' and for prediction with
            classifiers.

        Returns
        -------
        scores : ndarray of shape (m_samples, n_train)
            scores[i, j] = influence of training sample j on test sample i.
            Positive = helpful. May contain NaN for training points with no
            OOB (or no in-bag) runs; both cases warn. The standard error is
            stored in ``scores_std_``.
        """
        check_is_fitted(self, ["model_", "bootstrap_models_"])
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
        B = self.n_estimators
        in_bag_list = self.in_bag_indices_

        # Build (n_test, B) array of loss or prediction per bootstrap run
        # Use only successful runs
        valid_b = [b for b in range(B) if self.bootstrap_models_[b] is not None]
        if not valid_b:
            return np.full((n_test, n_train), np.nan)

        values_per_b = np.full((n_test, B), np.nan)
        for b in valid_b:
            m = self.bootstrap_models_[b]
            values_per_b[:, b] = _value_at_test(
                m, X_test, y_test, self.mode, self.is_classifier_
            )

        # OOB mask: oob_mask[i, b] = True if training point i is OOB in run b
        oob_mask = np.ones((n_train, B), dtype=bool)
        for b in range(B):
            oob_mask[in_bag_list[b], b] = False
        in_bag_mask = ~oob_mask

        # For each training point i: average over b where i is OOB, then subtract baseline
        scores = np.full((n_test, n_train), np.nan)
        scores_std = np.full((n_test, n_train), np.nan)
        valid_b_set = set(valid_b)
        n_no_oob = 0
        n_no_in_bag = 0
        for i in range(n_train):
            oob_b_valid = [b for b in range(B) if oob_mask[i, b] and b in valid_b_set]
            in_bag_b_valid = [
                b for b in range(B) if in_bag_mask[i, b] and b in valid_b_set
            ]
            if len(oob_b_valid) == 0:
                n_no_oob += 1
                continue
            if len(in_bag_b_valid) == 0:
                n_no_in_bag += 1
                continue
            vals = np.mean(values_per_b[:, oob_b_valid], axis=1)
            baseline_i = np.mean(values_per_b[:, in_bag_b_valid], axis=1)
            if self.mode == "loss":
                scores[:, i] = vals - baseline_i
            else:
                scores[:, i] = baseline_i - vals
            if len(oob_b_valid) >= 2 and len(in_bag_b_valid) >= 2:
                var_oob = np.var(values_per_b[:, oob_b_valid], axis=1, ddof=1)
                var_in = np.var(values_per_b[:, in_bag_b_valid], axis=1, ddof=1)
                scores_std[:, i] = np.sqrt(
                    var_oob / len(oob_b_valid) + var_in / len(in_bag_b_valid)
                )

        # Warn once for every degraded case: NaN (no OOB / no in-bag runs)
        # and few-OOB points. A point that is in-bag in every run is the most
        # extreme dropout and must not be the one silent case.
        n_oob_per_i = np.sum(oob_mask[:, valid_b], axis=1)
        few_oob = int(np.sum((n_oob_per_i > 0) & (n_oob_per_i < self.min_oob_runs)))
        issues = []
        if n_no_oob > 0:
            issues.append(
                f"{n_no_oob} training point(s) were in-bag in every "
                "successful run (no OOB runs; scores are NaN)"
            )
        if n_no_in_bag > 0:
            issues.append(
                f"{n_no_in_bag} training point(s) were OOB in every "
                "successful run (no in-bag runs; scores are NaN)"
            )
        if few_oob > 0:
            issues.append(
                f"{few_oob} training point(s) have fewer than "
                f"{self.min_oob_runs} OOB runs (scores use available runs)"
            )
        if issues:
            warnings.warn(
                "BootstrapInfluence: "
                + "; ".join(issues)
                + ". Increase n_estimators for more reliable estimates.",
                UserWarning,
                stacklevel=2,
            )
        self.scores_std_ = scores_std

        return scores

    def _self_influence_diag(self) -> NDArray[np.floating]:
        """
        Diagonal of the train-vs-train influence matrix in O(n x B) memory.

        Evaluates each bootstrap model once on the full training set and
        combines only the diagonal entries, instead of the (n_train, n_train)
        matrix that ``explain(X_train, y_train)`` would build. Used by
        ``pyinfluence.self_influence``.
        """
        check_is_fitted(self, ["model_", "bootstrap_models_"])
        X, y = self.X_train_, self.y_train_
        n_train = X.shape[0]
        B = self.n_estimators
        valid_b = [b for b in range(B) if self.bootstrap_models_[b] is not None]
        if not valid_b:
            return np.full(n_train, np.nan)

        values_per_b = np.full((n_train, B), np.nan)
        for b in valid_b:
            values_per_b[:, b] = _value_at_test(
                self.bootstrap_models_[b], X, y, self.mode, self.is_classifier_
            )
        oob_mask = np.ones((n_train, B), dtype=bool)
        for b in range(B):
            oob_mask[self.in_bag_indices_[b], b] = False

        diag = np.full(n_train, np.nan)
        valid_b_set = set(valid_b)
        for i in range(n_train):
            ob = [b for b in valid_b_set if oob_mask[i, b]]
            ib = [b for b in valid_b_set if not oob_mask[i, b]]
            if not ob or not ib:
                continue
            v = values_per_b[i, ob].mean()
            base = values_per_b[i, ib].mean()
            diag[i] = v - base if self.mode == "loss" else base - v
        return diag
