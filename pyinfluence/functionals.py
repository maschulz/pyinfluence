"""A library of ready-made functionals for the attribution engine.

Domain-neutral builders: each takes the fixed row-aligned context it needs
(a grouping variable, a conditioning rule) and returns a
:class:`~pyinfluence.Functional` for the engine estimators
(:class:`~pyinfluence.FunctionalInfluence`,
:class:`~pyinfluence.RefitFunctionalInfluence`,
:class:`~pyinfluence.SubsampledFunctionalInfluence`) or for evaluation via
:func:`~pyinfluence.functional_value`.

The smooth builders carry analytic gradients, so the closed form needs no
finite differences; :func:`auroc` is a rank statistic and is marked
``differentiable=False``, which the engine attributes by perturbation
evaluation instead. All returned functionals are picklable (fitted attributors
holding them can be persisted with joblib/pickle, provided any
user-supplied callables, e.g. a custom ``keep=``, are themselves picklable,
i.e. module-level functions rather than lambdas). The fairness vocabulary
(demographic parity, equal opportunity, ...) is a thin naming layer over
these in :mod:`pyinfluence.fairness`.

Builders
--------
- :func:`mean`: average score or loss.
- :func:`group_gap`: difference in group means (optionally restricted to
  rows selected from the labels).
- :func:`cohens_d`: standardized group gap (pooled-SD normalized).
- :func:`worst_group_mean`: max over groups of the group mean.
- :func:`auroc`: ranking quality (exact Mann-Whitney; attributed via
  perturbation evaluation, no smoothing).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyinfluence._functional import Functional, ValueKind

__all__ = [
    "mean",
    "group_gap",
    "cohens_d",
    "worst_group_mean",
    "auroc",
]


def _check_aligned(values, n_bound: int, what: str) -> None:
    """Row-alignment guard: bound context must match the evaluated rows."""
    n = np.asarray(values).shape[0]
    if n != n_bound:
        raise ValueError(
            f"this functional is bound to a {what} array of length "
            f"{n_bound}, but was evaluated on {n} reference rows — "
            "evaluate it only on the row-aligned reference/audit set."
        )


def _two_groups(groups: ArrayLike) -> NDArray[np.bool_]:
    """Boolean mask of the higher-sorted group value (gap = high - low)."""
    g = np.asarray(groups).ravel()
    values = np.unique(g)
    if len(values) != 2:
        raise ValueError(
            f"groups must be binary; got {len(values)} unique values. "
            "For multi-group comparisons use worst_group_mean or pairwise "
            "gaps."
        )
    return g == values[1]


# -----------------------------------------------------------------------------
# Picklable payload classes (bound methods of picklable instances pickle
# cleanly; closures do not)
# -----------------------------------------------------------------------------


class _Mean:
    def value(self, v, y=None):
        return float(np.mean(v))

    def grad(self, v, y=None):
        n = np.asarray(v).size
        return np.full(n, 1.0 / n)


class _GroupGap:
    def __init__(self, mask_g1, keep):
        self.mask_g1 = mask_g1
        self.keep = keep

    def _masks(self, v, y):
        _check_aligned(v, self.mask_g1.size, "groups")
        if self.keep is None:
            return None, self.mask_g1
        if y is None:
            raise ValueError(
                "this group_gap conditions on the labels; pass y_ref."
            )
        k = np.asarray(self.keep(np.asarray(y).ravel()), dtype=bool)
        m1 = self.mask_g1[k]
        if m1.all() or not m1.any():
            raise ValueError(
                "Both groups must be present in the (label-restricted) "
                "reference set."
            )
        return k, m1

    def value(self, v, y=None):
        k, m1 = self._masks(v, y)
        vv = v if k is None else v[k]
        return float(vv[m1].mean() - vv[~m1].mean())

    def grad(self, v, y=None):
        k, m1 = self._masks(v, y)
        g = np.zeros(np.asarray(v).size)
        idx = np.arange(g.size) if k is None else np.where(k)[0]
        g[idx[m1]] = 1.0 / m1.sum()
        g[idx[~m1]] = -1.0 / (~m1).sum()
        return g


class _CohensD:
    def __init__(self, mask):
        self.mask = mask

    def _stats(self, v):
        v = np.asarray(v, dtype=float).ravel()
        _check_aligned(v, self.mask.size, "groups")
        s1, s0 = v[self.mask], v[~self.mask]
        n1, n0 = s1.size, s0.size
        if n1 < 2 or n0 < 2:
            raise ValueError(
                "cohens_d requires at least two reference samples per "
                f"group; got {n1} and {n0}."
            )
        pooled_var = (
            (n1 - 1) * s1.var(ddof=1) + (n0 - 1) * s0.var(ddof=1)
        ) / (n1 + n0 - 2)
        if pooled_var <= 0:
            raise ValueError(
                "cohens_d is undefined: zero pooled variance "
                "(values are constant within each group)."
            )
        return s1, s0, n1, n0, pooled_var

    def value(self, v, y=None):
        s1, s0, n1, n0, pooled_var = self._stats(v)
        return float((s1.mean() - s0.mean()) / np.sqrt(pooled_var))

    def grad(self, v, y=None):
        s1, s0, n1, n0, sp2 = self._stats(v)
        sp = np.sqrt(sp2)
        d = (s1.mean() - s0.mean()) / sp
        g = np.empty(np.asarray(v).size)
        # d(mean gap)/dv +- 1/(n_g sp), minus the pooled-SD term
        g[self.mask] = (1.0 / n1) / sp - d * (s1 - s1.mean()) / (
            (n1 + n0 - 2) * sp2
        )
        g[~self.mask] = (-1.0 / n0) / sp - d * (s0 - s0.mean()) / (
            (n1 + n0 - 2) * sp2
        )
        return g


class _WorstGroupMean:
    def __init__(self, group_masks, n_rows):
        self.group_masks = group_masks
        self.n_rows = n_rows

    def value(self, v, y=None):
        _check_aligned(v, self.n_rows, "groups")
        return float(max(v[m].mean() for m in self.group_masks))

    def grad(self, v, y=None):
        _check_aligned(v, self.n_rows, "groups")
        means = [v[m].mean() for m in self.group_masks]
        m = self.group_masks[int(np.argmax(means))]
        out = np.zeros(np.asarray(v).size)
        out[m] = 1.0 / m.sum()
        return out


class _Auroc:
    def __init__(self, pos_label):
        self.pos_label = pos_label

    def _split(self, v, y):
        if y is None:
            raise ValueError(
                "auroc requires the reference labels; pass y_ref."
            )
        v = np.asarray(v, dtype=float).ravel()
        y_arr = np.asarray(y).ravel()
        if y_arr.size != v.size:
            raise ValueError(
                f"y_ref has length {y_arr.size} but the reference set has "
                f"{v.size} rows — auroc needs row-aligned labels."
            )
        pos = y_arr == self.pos_label
        if pos.all() or not pos.any():
            raise ValueError(
                "auroc needs both positive and negative reference samples; "
                f"got {int(pos.sum())} positives of {pos.size}."
            )
        return v, pos

    def value(self, v, y=None):
        from scipy.stats import rankdata

        v, pos = self._split(v, y)
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        ranks = rankdata(v)  # average ranks handle ties exactly
        u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------


def mean(of: ValueKind = "scores") -> Functional:
    """
    The plain average of per-sample values.

    ``mean('losses')`` is the model's mean reference-set loss; attributing
    it recovers (audit-aggregated) loss influence.
    """
    payload = _Mean()
    return Functional(
        fn=payload.value, grad=payload.grad, of=of, name=f"mean_{of}"
    )


def group_gap(
    groups: ArrayLike,
    keep: Callable[[NDArray], NDArray] | None = None,
    of: ValueKind = "scores",
) -> Functional:
    """
    Difference in group means: ``mean(v | g1) - mean(v | g0)``.

    The two group values are ordered by sort order (g0 < g1). With
    ``of='scores'`` and a classifier this is the (smoothed) demographic
    parity gap; conditioned on the labels it becomes the equal-opportunity
    or FPR gap. See :func:`pyinfluence.fairness.disparity` for those
    named forms.

    Parameters
    ----------
    groups : array-like of shape (m,)
        Binary grouping variable, row-aligned with the reference set.
    keep : callable, optional
        Row filter computed from the reference labels:
        ``keep(y) -> bool mask``. Only kept rows enter the two means (e.g.
        a TPR-style gap keeps ``y == pos``). Requires ``y`` at evaluation
        time. Pass a module-level function (not a lambda) if you need the
        functional to be picklable.
    of : {'scores', 'losses'}, default='scores'
        Value kind the gap is computed over.
    """
    payload = _GroupGap(_two_groups(groups), keep)
    return Functional(
        fn=payload.value, grad=payload.grad, of=of, name="group_gap"
    )


def cohens_d(groups: ArrayLike) -> Functional:
    """
    Standardized group gap of scores: ``(mean_g1 - mean_g0) / pooled_sd``.

    Pooled standard deviation uses ``ddof=1`` group variances:
    ``sqrt(((n1-1)v1 + (n0-1)v0) / (n1+n0-2))``. Groups ordered by sort
    order (g0 < g1), as in :func:`group_gap`.

    Notes
    -----
    Because d is normalized by the pooled spread, a training point can
    shrink ``|d|`` by *inflating within-group score variance* rather than
    by closing the gap. When ranking points for removal by Cohen's-d
    influence, inspect the raw :func:`group_gap` attribution alongside
    before acting.
    """
    payload = _CohensD(_two_groups(groups))
    return Functional(
        fn=payload.value, grad=payload.grad, of="scores", name="cohens_d"
    )


def worst_group_mean(
    groups: ArrayLike,
    of: ValueKind = "losses",
) -> Functional:
    """
    Max over groups of the group's mean value (any number of groups).

    With ``of='losses'`` this is the worst-group loss. The gradient is the
    subgradient at the argmax group (uniform over its members).
    """
    g = np.asarray(groups).ravel()
    payload = _WorstGroupMean([g == u for u in np.unique(g)], g.size)
    return Functional(
        fn=payload.value, grad=payload.grad, of=of, name="worst_group_mean"
    )


def auroc(pos_label) -> Functional:
    """
    Area under the ROC curve of the scores against the reference labels.

    The exact Mann-Whitney AUROC (tie-corrected via average ranks),
    identical to ``sklearn.metrics.roc_auc_score``. Works with every
    estimator: the refit-based ones evaluate it directly, and
    :class:`~pyinfluence.FunctionalInfluence` attributes it by exact
    re-evaluation on linearized per-removal score perturbations (a rank
    statistic is piecewise constant, so it is marked
    ``differentiable=False`` and the engine skips the chain rule). This
    agrees with exact leave-one-out refitting at r > 0.99 (enforced in
    the test suite) and preserves the discreteness of the estimand:
    removals that swap no (positive, negative) pair score exactly 0.

    Parameters
    ----------
    pos_label :
        The label counted as positive (e.g. ``model.classes_[1]``, the
        class whose score the engine feeds the functional). Required
        explicitly; it is not inferred from the data.

    Notes
    -----
    Per-removal effects on an audit-set AUROC are quantized in steps of
    1/(n_pos * n_neg); many training points have exactly zero effect.
    If the question you want answered is "which points change class
    *separation*" (a smooth, densely-attributable quantity), use
    :func:`cohens_d` with the true labels as groups. The two estimands
    are related but distinct (they correlate only moderately per point).

    Attribution cost with FunctionalInfluence is O(m x n) memory-blocked
    matrix work plus n exact evaluations at O(m log m) each.
    """
    payload = _Auroc(pos_label)
    return Functional(
        fn=payload.value, of="scores", differentiable=False, name="auroc"
    )
