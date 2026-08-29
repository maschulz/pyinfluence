"""Attribution of scalar functionals of model behavior to training examples.

Estimand
--------
Let F(theta) be a scalar functional of a fitted model, defined through
per-sample values on a fixed reference set: model *scores* (predictions,
probabilities, decision values) or per-sample *losses*. All estimators here
attribute the per-point removal effect

    score[j] ~= F(D \\ {z_j}) - F(D),

the same removal convention (sign and scale) as the rest of the package.

The functional is a plain callable ``fn(values, y) -> float`` over the
reference-set value vector, or a :class:`Functional` bundling the callable
with an analytic gradient and the value kind (``of='scores'`` or
``'losses'``). Group-disparity functionals for fairness audits are built by
:func:`pyinfluence.fairness.disparity`.

Estimators
----------
- :class:`FunctionalInfluence`: closed form for supported GLMs. Smooth
  functionals go through the chain rule grad_theta F = sum_i (dF/dv_i)
  grad_theta v_i (analytic gradient or finite differences); functionals
  marked ``differentiable=False`` (rank statistics such as the exact
  AUROC) are attributed by perturbation evaluation: the exact functional
  re-evaluated on each removal's linearized value change.
- :class:`RefitFunctionalInfluence`: exact removal effects by refitting
  without each point (model-agnostic ground truth; n refits). Evaluates the
  functional directly, so smoothness is not required.
- :class:`SubsampledFunctionalInfluence`: Monte-Carlo subset estimator
  (model-agnostic, maximum-sample-reuse, Data-Banzhaf style; T refits).

The refit estimator doubles as ground truth for validating any new
functional's closed-form scores (correlation and slope ~ 1).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone, is_classifier
from tqdm import tqdm

from pyinfluence._base import _prepare_fit_inputs, check_is_fitted
from pyinfluence._influence import InfluenceFunctions
from pyinfluence._linear import _augment_intercept
from pyinfluence._utils import _compute_loss_sklearn, _quiet_sklearn, tqdm_joblib
from pyinfluence._validation import check_is_fitted_model, warn_if_data_mismatch

if TYPE_CHECKING:
    from typing import Self

__all__ = [
    "Functional",
    "FunctionalInfluence",
    "RefitFunctionalInfluence",
    "SubsampledFunctionalInfluence",
    "functional_value",
]

ValueKind = Literal["scores", "losses"]
TargetName = Literal["signed", "absolute"]


def _validate_target(target: str) -> None:
    if target not in ("signed", "absolute"):
        raise ValueError(f"target must be 'signed' or 'absolute'; got {target!r}.")


@dataclass(frozen=True)
class Functional:
    """
    A scalar functional of per-sample model values on a reference set.

    Parameters
    ----------
    fn : callable
        ``fn(values, y) -> float`` where ``values`` is the per-sample value
        vector on the reference set and ``y`` the reference labels (or None).
        Row-aligned context beyond ``y`` (e.g. a sensitive attribute) is
        closed over. See :func:`pyinfluence.fairness.disparity`.
    grad : callable, optional
        Analytic gradient ``grad(values, y) -> ndarray of shape (m,)``
        w.r.t. the value vector. Default: central finite differences
        (requires ``fn`` to be smooth in ``values``).
    of : {'scores', 'losses'}, default='scores'
        Which per-sample values ``fn`` consumes. ``'scores'``: the model's
        output (positive-class probability for classifiers with
        predict_proba, decision value or prediction otherwise).
        ``'losses'``: per-sample loss (NLL / squared error), which requires
        reference labels.
    differentiable : bool, default=True
        Whether ``fn`` carries usable first-order information in the
        values. Set False for rank statistics and other piecewise-constant
        functionals (e.g. the exact AUROC), whose gradient is zero almost
        everywhere: :class:`FunctionalInfluence` then attributes them by
        evaluating ``fn`` exactly on linearized per-removal value
        perturbations instead of by the chain rule.
    name : str
        Display name.
    """

    fn: Callable[[NDArray, NDArray | None], float]
    grad: Callable[[NDArray, NDArray | None], NDArray] | None = None
    of: ValueKind = "scores"
    differentiable: bool = True
    name: str = field(default="")

    def __post_init__(self):
        if self.of not in ("scores", "losses"):
            raise ValueError(f"of must be 'scores' or 'losses'; got {self.of!r}.")

    def __call__(self, values: NDArray, y: NDArray | None = None) -> float:
        return float(self.fn(values, y))


def as_functional(functional) -> Functional:
    """Coerce a bare callable (a score-functional) to a :class:`Functional`."""
    if isinstance(functional, Functional):
        return functional
    if callable(functional):
        return Functional(
            fn=functional,
            of="scores",
            name=getattr(functional, "__name__", "functional"),
        )
    raise TypeError(
        "functional must be a Functional or a callable fn(values, y) -> "
        f"float; got {type(functional).__name__}."
    )


def _fd_grad(
    fn: Callable,
    values: NDArray[np.floating],
    y: NDArray | None,
    eps: float = 1e-6,
) -> NDArray[np.floating]:
    """Central finite-difference gradient of ``fn`` w.r.t. the value vector.

    2m evaluations of an O(m) numpy function, cheap for reference sets up
    to ~10^4 points. Requires smoothness; thresholded/rank-based
    functionals yield zero or meaningless gradients here. Mark those
    ``differentiable=False`` so the engine uses perturbation evaluation
    instead.
    """
    values = np.asarray(values, dtype=float).ravel()
    grad = np.empty_like(values)
    for i in range(values.size):
        h = eps * max(1.0, abs(values[i]))
        vp = values.copy()
        vp[i] += h
        vm = values.copy()
        vm[i] -= h
        grad[i] = (fn(vp, y) - fn(vm, y)) / (2 * h)
    return grad


def _model_scores(model: BaseEstimator, X: NDArray) -> NDArray[np.floating]:
    """The per-sample model output that score-functionals consume.

    P(Y=classes_[1] | x) for classifiers with predict_proba, decision values
    otherwise (e.g. RidgeClassifier), predictions for regressors.
    """
    with _quiet_sklearn():
        if is_classifier(model):
            if callable(getattr(model, "predict_proba", None)):
                return model.predict_proba(X)[:, 1]
            return np.asarray(model.decision_function(X), dtype=float).ravel()
        return np.asarray(model.predict(X), dtype=float).ravel()


def _model_values(
    model: BaseEstimator,
    X: NDArray,
    y: NDArray | None,
    of: ValueKind,
) -> NDArray[np.floating]:
    """Per-sample values of the requested kind on the reference set."""
    if of == "losses":
        if y is None:
            raise ValueError("y_ref is required for a functional of per-sample losses.")
        return _compute_loss_sklearn(model, X, y, is_classifier(model))
    return _model_scores(model, X)


def _resolve_functional(default, override) -> Functional:
    """Resolve constructor-default vs explain-time functional."""
    functional = default if override is None else override
    if functional is None:
        raise ValueError(
            "No functional to attribute: pass one at construction or to "
            "explain(functional=...)."
        )
    return as_functional(functional)


def functional_value(
    model: BaseEstimator,
    X_ref: ArrayLike,
    functional: Functional | Callable,
    y_ref: ArrayLike | None = None,
    target: TargetName = "signed",
) -> float:
    """
    Evaluate a functional on a fitted model's reference-set values.

    The evaluation counterpart of the attribution estimators: computes the
    per-sample values (scores or losses, per ``functional.of``) and applies
    the functional.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_ref : array-like of shape (m, p)
        Reference set, row-aligned with any context the functional closes
        over.
    functional : Functional or callable
    y_ref : array-like of shape (m,), optional
        Required for loss-functionals and label-conditioned functionals.
    target : {'signed', 'absolute'}, default='signed'

    Returns
    -------
    value : float
    """
    func = as_functional(functional)
    _validate_target(target)
    y_arr = None if y_ref is None else np.asarray(y_ref).ravel()
    values = _model_values(model, np.asarray(X_ref), y_arr, func.of)
    value = func(values, y_arr)
    return abs(value) if target == "absolute" else value


def _refit_without(
    model: BaseEstimator,
    X: NDArray,
    y: NDArray,
    remove: NDArray[np.intp] | int,
    refit_factory: Callable[[int], BaseEstimator] | None,
) -> BaseEstimator | None:
    mask = np.ones(len(y), dtype=bool)
    mask[remove] = False
    est = refit_factory(int(mask.sum())) if refit_factory is not None else clone(model)
    try:
        return est.fit(X[mask], y[mask])
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Closed form (GLMs)
# -----------------------------------------------------------------------------


class FunctionalInfluence:
    """
    Closed-form influence of training points on a scalar functional.

    Supported models: binary LogisticRegression(CV), RidgeClassifier(CV)
    (decision-value scores), Ridge/RidgeCV/LinearRegression (prediction
    scores). KernelRidge is not supported (dual space does not match the
    primal chain rule).

    Parameters
    ----------
    functional : Functional or callable, optional
        The functional to attribute. A bare callable is treated as a smooth
        score-functional ``fn(scores, y) -> float`` differentiated by
        finite differences; wrap it in :class:`Functional` to supply an
        analytic gradient, to consume per-sample losses, or to mark it
        ``differentiable=False`` (rank statistics), in which case the
        engine switches from the chain rule to perturbation evaluation:
        the exact functional on each removal's linearized value change.
        ``fit`` is functional-independent, so this may be omitted at
        construction and supplied per ``explain`` call instead.
    target : {'signed', 'absolute'}, default='signed'
        Attribute F or |F| (gradient scaled by sign(F); undefined at F=0).
    damping : float, default=1e-5
        Hessian damping, as in :class:`~pyinfluence.InfluenceFunctions`.
    hessian : {'exact', 'identity'}, default='exact'
        'identity' replaces H^{-1} with I (gradient-dot baseline).
    fd_eps : float, default=1e-6
        Relative step for the finite-difference functional gradient.

    Attributes
    ----------
    base_attributor_ : InfluenceFunctions
        Fitted loss-influence attributor providing H^{-1} and train grads.

    Notes
    -----
    ``explain`` returns a vector of length n_train estimating
    F(D \\ {z_j}) - F(D), removal-calibrated like all attributors in this
    package. Validate new functionals against
    :class:`RefitFunctionalInfluence` (correlation and slope ~ 1).
    """

    def __init__(
        self,
        functional: Functional | Callable | None = None,
        target: TargetName = "signed",
        damping: float = 1e-5,
        hessian: Literal["exact", "identity"] = "exact",
        fd_eps: float = 1e-6,
    ) -> None:
        self.functional = functional
        self.target = target
        self.damping = damping
        self.hessian = hessian
        self.fd_eps = fd_eps

    def fit(self, model: BaseEstimator, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Fit to a trained model and its training data.

        Parameters
        ----------
        model : fitted sklearn estimator
            Supported GLM (see class docstring).
        X, y : array-like
            Training data the model was fitted on.
        """
        if self.functional is not None:
            as_functional(self.functional)  # fail early on bad input
        _validate_target(self.target)
        if self.hessian not in ("exact", "identity"):
            raise ValueError(
                f"hessian must be 'exact' or 'identity'; got {self.hessian!r}."
            )
        base = InfluenceFunctions(mode="prediction", damping=self.damping)
        base.fit(model, X, y)
        if base.model_type_ == "kernel_ridge":
            raise ValueError("KernelRidge is not supported by FunctionalInfluence.")
        self.base_attributor_ = base
        self.model_ = model
        X_arr, y_arr = _prepare_fit_inputs(X, y)
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        return self

    def explain(
        self,
        X_ref: ArrayLike,
        y_ref: ArrayLike | None = None,
        *,
        functional: Functional | Callable | None = None,
        target: TargetName | None = None,
    ) -> NDArray[np.floating]:
        """
        Estimate each training point's removal effect on the functional.

        Parameters
        ----------
        X_ref : array-like of shape (m, p)
            Reference set on which the functional is evaluated. Must be
            row-aligned with any context the functional closes over (e.g.
            the sensitive attribute bound by ``fairness.disparity``).
        y_ref : array-like of shape (m,), optional
            Reference labels; required for loss-functionals and for any
            functional that uses ``y``.
        functional, target : optional
            Per-call overrides; ``fit`` is functional-independent, so one
            fitted attributor can explain many functionals.

        Returns
        -------
        scores : ndarray of shape (n_train,)
            scores[j] ~= F(D \\ {z_j}) - F(D). Positive = removing z_j
            increases the (signed or absolute) functional.
        """
        check_is_fitted(self, ["base_attributor_"])
        func = _resolve_functional(self.functional, functional)
        target = self.target if target is None else target
        _validate_target(target)
        X_ref = np.asarray(X_ref)
        if X_ref.ndim == 1:
            X_ref = X_ref.reshape(1, -1)
        y_arr = None if y_ref is None else np.asarray(y_ref).ravel()

        # target="absolute" wraps the functional in |.|, which has a kink at
        # F=0. The chain-rule linearization applies sign(F) to the gradient and
        # so mispredicts any removal that crosses zero — common near parity,
        # exactly where absolute fairness gaps are audited. Evaluate the
        # absolute functional exactly via perturbation instead, which sees the
        # crossing. Non-differentiable functionals (rank statistics, and the
        # max in worst_group_mean) use the same exact path.
        if not func.differentiable or target == "absolute":
            return self._perturbation_scores(func, X_ref, y_arr, target)

        base = self.base_attributor_
        gF = self._grad_functional(func, X_ref, y_arr, target)
        n_train = base.train_grads_.shape[0]
        if self.hessian == "identity":
            direction = gF
        else:
            direction = base.H_inv_ @ gF
        # removal weight -1/n; d theta = (1/n) H^{-1} grad_l_j
        return (base.train_grads_ @ direction) / n_train

    def _perturbation_scores(
        self,
        func: Functional,
        X_ref: NDArray[np.floating],
        y: NDArray | None,
        target: TargetName,
    ) -> NDArray[np.floating]:
        """Attribute a non-differentiable functional by exact re-evaluation.

        Linearize the parameter step of each removal (the same
        influence-function step the chain rule uses), propagate it to the
        per-sample values, and evaluate the *exact* functional on each
        perturbed value vector:

            score[j] = F(v + dv_j) - F(v),   dv_j = grad_v @ H^{-1} grad_l_j / n

        (removal weight -1/n gives theta_{-j} - theta = +H^{-1} grad_l_j / n,
        so the value perturbation enters with a plus sign)

        This preserves the discrete structure of rank statistics (removals
        that swap no pairs score exactly 0). On the package's benchmark it
        matches exact-refit ground truth at r > 0.99 for the exact AUROC,
        but the linearization is dataset-dependent and can correlate poorly
        on small or high-leverage problems; validate against
        RefitFunctionalInfluence on your data. Column blocks keep memory at
        O(m x block) instead of O(m x n).
        """
        base = self.base_attributor_
        model = self.model_
        values = _model_values(model, X_ref, y, func.of)
        per_sample = self._per_sample_value_grads(X_ref, y, values, func.of)
        n_train = base.train_grads_.shape[0]

        def evaluate(v: NDArray) -> float:
            out = func(v, y)
            return abs(out) if target == "absolute" else out

        base_value = evaluate(values)
        # dv (m, n) in column blocks: per_sample @ H^{-1} @ train_grads.T / n
        left = per_sample if self.hessian == "identity" else per_sample @ base.H_inv_
        scores = np.empty(n_train)
        block = max(1, int(2**22 // max(1, values.size)))  # ~32 MB blocks
        for start in range(0, n_train, block):
            stop = min(start + block, n_train)
            dv = left @ base.train_grads_[start:stop].T / n_train
            for j in range(stop - start):
                scores[start + j] = evaluate(values + dv[:, j]) - base_value
        return scores

    # -- gradient of the functional w.r.t. parameters -------------------------

    def _grad_functional(
        self,
        func: Functional,
        X_ref: NDArray[np.floating],
        y: NDArray | None,
        target: TargetName,
    ) -> NDArray[np.floating]:
        """grad_theta F on the reference set, in augmented-parameter space.

        Chain rule: grad_theta F = sum_i (dF/dv_i) grad_theta v_i, with
        dF/dv from the functional's analytic gradient or central finite
        differences, and grad_theta v_i the model's per-sample value
        gradients (scores: p(1-p)x for logistic probabilities, x for linear
        scores; losses: the per-sample loss gradients).
        """
        model = self.model_
        values = _model_values(model, X_ref, y, func.of)

        if func.grad is not None:
            dFdv = np.asarray(func.grad(values, y), dtype=float).ravel()
            if dFdv.shape != values.shape:
                raise ValueError(
                    "the functional's grad must return one gradient entry "
                    f"per reference sample; got shape {dFdv.shape} for "
                    f"{values.shape[0]} samples."
                )
        else:
            dFdv = _fd_grad(func.fn, values, y, eps=self.fd_eps)

        per_sample = self._per_sample_value_grads(X_ref, y, values, func.of)
        grad = per_sample.T @ dFdv

        if target == "absolute":
            value = func(values, y)
            if value == 0:
                warnings.warn(
                    "Functional value is exactly 0; absolute-target gradient "
                    "is undefined. Returning the signed gradient.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                grad = np.sign(value) * grad
        return grad

    def _per_sample_value_grads(
        self,
        X_ref: NDArray[np.floating],
        y: NDArray | None,
        values: NDArray[np.floating],
        of: ValueKind,
    ) -> NDArray[np.floating]:
        """grad_theta of each reference sample's value, shape (m, p)."""
        base = self.base_attributor_
        X_aug = _augment_intercept(X_ref) if base.has_intercept_ else X_ref

        if of == "scores":
            if base.model_type_ == "logistic":
                # values are P(Y=classes_[1]|x); d p / d theta = p(1-p) x
                return X_aug * (values * (1 - values))[:, None]
            # linear score: ridge / linear regression / ridge_classifier
            return X_aug

        # of == "losses"
        return self._per_sample_loss_grads(X_aug, y, X_ref)

    def _per_sample_loss_grads(
        self,
        X_aug: NDArray[np.floating],
        y: NDArray | None,
        X_raw: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Per-sample loss gradients on arbitrary points."""
        base = self.base_attributor_
        model = self.model_
        if base.model_type_ == "logistic":
            with _quiet_sklearn():
                p = model.predict_proba(X_raw)[:, 1]
            # NLL gradient needs y as a 0/1 indicator of classes_[1] (the
            # class p refers to), not the raw label values.
            y01 = (np.asarray(y).ravel() == model.classes_[1]).astype(float)
            return -X_aug * (y01 - p)[:, None]
        # squared-error models use the half-squared-error loss ½(y - ŷ)²
        # (matching _compute_loss_sklearn), whose gradient is -(y - ŷ)x
        theta = (
            np.concatenate(
                [
                    np.atleast_1d(model.coef_).ravel(),
                    np.atleast_1d(model.intercept_).ravel(),
                ]
            )
            if model.fit_intercept
            else np.atleast_1d(model.coef_).ravel()
        )
        if base.model_type_ == "ridge_classifier":
            yv = np.where(np.asarray(y).ravel() == model.classes_[1], 1.0, -1.0)
        else:
            yv = np.asarray(y).ravel()
        resid = yv - X_aug @ theta
        return -X_aug * resid[:, None]


# -----------------------------------------------------------------------------
# Exact refit (ground truth)
# -----------------------------------------------------------------------------


class RefitFunctionalInfluence:
    """
    Exact per-point removal effects on a functional via refitting.

    Model-agnostic ground truth: for each training point, refits the
    estimator without it and re-evaluates the functional on the reference
    set. Costs n refits. Evaluates the functional directly, so smoothness is
    NOT required (rank- or threshold-based functionals are fine).

    Parameters
    ----------
    functional : Functional or callable
        See :class:`FunctionalInfluence`.
    target : {'signed', 'absolute'}, default='signed'
    n_jobs : int, optional
        Parallel refits (joblib). None = sequential.
    verbose : int, default=1
        Progress bar on/off.
    refit_factory : callable(n_remaining) -> estimator, optional
        Estimator constructor for refits. Default clones the original model
        (the practitioner counterfactual). Pass a factory to hold the
        per-sample-average regularization fixed, which isolates the removal
        effect from the regularization shift when validating estimates.
    """

    def __init__(
        self,
        functional: Functional | Callable | None = None,
        target: TargetName = "signed",
        n_jobs: int | None = None,
        verbose: int = 1,
        refit_factory: Callable[[int], BaseEstimator] | None = None,
    ) -> None:
        self.functional = functional
        self.target = target
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit_factory = refit_factory

    def fit(self, model: BaseEstimator, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the leave-one-out models (the expensive step, done once)."""
        if self.functional is not None:
            as_functional(self.functional)
        _validate_target(self.target)
        check_is_fitted_model(model)
        X_arr, y_arr = _prepare_fit_inputs(X, y)
        warn_if_data_mismatch(model, X_arr, y_arr)
        self.model_ = model
        self.X_train_ = X_arr
        self.y_train_ = y_arr
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
                tqdm(total=n, desc="Fitting LOO models", disable=(self.verbose == 0))
            ):
                self.loo_models_ = Parallel(n_jobs=self.n_jobs)(
                    delayed(one)(j) for j in range(n)
                )
        n_failed = sum(m is None for m in self.loo_models_)
        if n_failed:
            warnings.warn(
                f"Refit failed for {n_failed} points; their scores will be NaN.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def explain(
        self,
        X_ref: ArrayLike,
        y_ref: ArrayLike | None = None,
        *,
        functional: Functional | Callable | None = None,
        target: TargetName | None = None,
    ) -> NDArray[np.floating]:
        """
        Exact removal effects: scores[j] = F(D \\ {z_j}) - F(D).

        The LOO models fitted in ``fit`` are functional-agnostic, so pass
        ``functional`` / ``target`` to score a different functional or
        reference set without refitting. NaN where the refit failed (e.g. a
        class disappears).
        """
        check_is_fitted(self, ["model_", "loo_models_"])
        func = _resolve_functional(self.functional, functional)
        target = self.target if target is None else target
        _validate_target(target)
        X_ref = np.asarray(X_ref)
        y_arr = None if y_ref is None else np.asarray(y_ref).ravel()

        def evaluate(model: BaseEstimator) -> float:
            values = _model_values(model, X_ref, y_arr, func.of)
            v = func(values, y_arr)
            return abs(v) if target == "absolute" else v

        base_value = evaluate(self.model_)
        return np.array(
            [
                float("nan") if m is None else evaluate(m) - base_value
                for m in self.loo_models_
            ]
        )


# -----------------------------------------------------------------------------
# Subsampled Monte-Carlo (model-agnostic)
# -----------------------------------------------------------------------------


class SubsampledFunctionalInfluence:
    """
    Monte-Carlo subset estimator of functional influence (model-agnostic).

    Fits ``n_subsets`` models on random subsets (each point included
    independently with probability ``subset_frac``) and scores each training
    point by the difference in mean functional value between subsets that
    exclude and subsets that include it (maximum-sample-reuse, Data-Banzhaf
    style):

        scores[j] = mean_{S w/o j} F(S) - mean_{S with j} F(S)

    matching the removal sign convention. The estimand is an *average*
    removal effect over subsets of size ~ subset_frac * n, not the
    full-dataset LOO effect; magnitudes are typically larger than LOO
    deltas. Like the refit estimator, evaluates the functional directly
    (smoothness not required).

    Parameters
    ----------
    functional : Functional or callable
        See :class:`FunctionalInfluence`.
    target : {'signed', 'absolute'}, default='signed'
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
        functional: Functional | Callable | None = None,
        target: TargetName = "signed",
        n_subsets: int = 200,
        subset_frac: float = 0.5,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 1,
    ) -> None:
        self.functional = functional
        self.target = target
        self.n_subsets = n_subsets
        self.subset_frac = subset_frac
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, model: BaseEstimator, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit subset models (the expensive step)."""
        if self.functional is not None:
            as_functional(self.functional)
        _validate_target(self.target)
        if not 0.0 < self.subset_frac < 1.0:
            raise ValueError("subset_frac must be in (0, 1).")
        check_is_fitted_model(model)
        X_arr, y_arr = _prepare_fit_inputs(X, y)
        warn_if_data_mismatch(model, X_arr, y_arr)
        self.model_ = model
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        n = len(y_arr)
        rng = np.random.default_rng(self.random_state)
        masks = rng.uniform(size=(self.n_subsets, n)) < self.subset_frac

        def fit_one(mask: NDArray[np.bool_]) -> BaseEstimator | None:
            if mask.sum() < 1:
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
                tqdm(
                    total=self.n_subsets,
                    desc="Fitting subset models",
                    disable=(self.verbose == 0),
                )
            ):
                models = Parallel(n_jobs=self.n_jobs)(
                    delayed(fit_one)(m) for m in masks
                )
        ok = [i for i, m in enumerate(models) if m is not None]
        if len(ok) < self.n_subsets:
            warnings.warn(
                f"{self.n_subsets - len(ok)} subset fits failed and were dropped.",
                UserWarning,
                stacklevel=2,
            )
        self.subset_masks_ = masks[ok]
        self.subset_models_ = [models[i] for i in ok]
        return self

    def explain(
        self,
        X_ref: ArrayLike,
        y_ref: ArrayLike | None = None,
        *,
        functional: Functional | Callable | None = None,
        target: TargetName | None = None,
    ) -> NDArray[np.floating]:
        """Estimated removal effects (see class docstring for the estimand).

        The subset models fitted in ``fit`` are functional-agnostic, so pass
        ``functional`` / ``target`` to score a different functional without
        refitting.
        """
        check_is_fitted(self, ["subset_models_"])
        func = _resolve_functional(self.functional, functional)
        target = self.target if target is None else target
        _validate_target(target)
        X_ref = np.asarray(X_ref)
        y_arr = None if y_ref is None else np.asarray(y_ref).ravel()

        def evaluate(model: BaseEstimator) -> float:
            values = _model_values(model, X_ref, y_arr, func.of)
            v = func(values, y_arr)
            return abs(v) if target == "absolute" else v

        values = np.array([evaluate(m) for m in self.subset_models_])
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
                stacklevel=2,
            )
        return scores
