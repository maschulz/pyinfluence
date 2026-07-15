"""Plotting helpers for influence analysis.

Thin plotting functions plus a multi-panel ``report`` wrapper. Each plotting
function takes pre-computed arrays (not an attributor), returns
``(fig, ax)``, accepts an optional ``ax=`` to draw on, and never colours
points by sign unless the sign carries meaning for the figure (top-k
explanation and the heatmap).

Matplotlib is an optional dependency (requires matplotlib >= 3.9); importing
this module without it raises a clear error from the first plotting call.

NaN scores (failed refits in LOO/Bootstrap/Banzhaf) are excluded from
rankings, matching the policy of the analysis utilities.

Functions
---------
- plot_top_influencers        : per-instance explanation (top helpful/harmful)
- plot_self_influence         : mislabel-detector view (histogram or vs-error)
- plot_by_group               : aggregate influence by group (bar/box/violin)
- plot_heatmap                : top-k subset of the influence matrix
- plot_method_comparison      : scatter of two attributors' scores
- plot_removal_curve          : loss-after-removal validation curve
- plot_disparity_curve        : fairness repair curve (disparity_removal_curve)
- plot_detection_curve        : mislabel-detection recall vs inspection budget
- plot_influence_concentration: Lorenz-style influence-mass concentration
- plot_top_k_stability        : top-k membership across replicates
- report                      : 2x2 diagnostic dashboard
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _require_mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install 'pyinfluence[viz]'"
        ) from e


def _ax_or_new(ax: Axes | None, **subplots_kwargs) -> tuple[Figure, Axes]:
    plt = _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(**subplots_kwargs)
    else:
        fig = ax.get_figure()
    return fig, ax


def _aggregate_to_1d(
    scores: ArrayLike,
    method: Literal["sum", "mean", "absmax"] = "sum",
) -> np.ndarray:
    """Collapse a 2D (n_test, n_train) score matrix to (n_train,)."""
    s = np.asarray(scores)
    if s.ndim == 1:
        return s
    if method == "sum":
        return s.sum(axis=0)
    if method == "mean":
        return s.mean(axis=0)
    if method == "absmax":
        idx = np.argmax(np.abs(s), axis=0)
        return np.take_along_axis(s, idx[None, :], axis=0).ravel()
    raise ValueError(f"unknown aggregation method: {method!r}")


_HELPFUL = "#2c7fb8"
_HARMFUL = "#d7301f"
_NEUTRAL = "#4d4d4d"


def _as_labels(labels: ArrayLike | None, expected_len: int) -> np.ndarray | None:
    """Coerce labels to a 1D ndarray of strings.

    Accepts list, ndarray, pandas Index, or pandas Series. Returns None when
    the user passed None. Raises if the length doesn't match the data.
    """
    if labels is None:
        return None
    arr = np.asarray(labels).ravel()
    if arr.size != expected_len:
        raise ValueError(
            f"labels has length {arr.size}, expected {expected_len}"
        )
    return arr.astype(str) if arr.dtype.kind != "U" else arr


# -----------------------------------------------------------------------------
# 1. Per-instance explanation
# -----------------------------------------------------------------------------


def plot_top_influencers(
    scores: ArrayLike,
    test_idx: int = 0,
    k: int = 10,
    labels: ArrayLike | None = None,
    xerr: ArrayLike | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Horizontal bar chart of the top-k most helpful and most harmful training
    points for a single test sample.

    Parameters
    ----------
    scores : array-like, shape (n_test, n_train) or (n_train,)
        Influence scores. A 1D vector is treated as a single test point.
        NaN scores (failed refits) are excluded from the ranking.
    test_idx : int, default=0
        Which row of ``scores`` to explain. Ignored when ``scores`` is 1D.
    k : int, default=10
        Number of helpful and harmful samples to show (so 2k bars total,
        capped at n_train).
    labels : array-like, optional
        Per-training-sample labels for the y-axis. Accepts a list, ndarray,
        or pandas Index/Series. Default: integer indices.
    xerr : array-like, optional
        Per-score standard errors, same shape as ``scores`` (or (n_train,)
        matching a 1D ``scores``). Drawn as error bars — pass
        ``attr.scores_std_`` from BanzhafInfluence / BootstrapInfluence to
        show whether the ranking is signal or Monte Carlo noise.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    s_all = np.asarray(scores, dtype=float)
    is_1d = s_all.ndim == 1
    if s_all.ndim == 1:
        row = s_all
        err_row = None if xerr is None else np.asarray(xerr, dtype=float).ravel()
    elif s_all.ndim == 2:
        row = s_all[test_idx]
        if xerr is None:
            err_row = None
        else:
            e = np.asarray(xerr, dtype=float)
            err_row = e[test_idx] if e.ndim == 2 else e.ravel()
    else:
        raise ValueError("scores must be 1D or 2D")
    if err_row is not None and err_row.size != row.size:
        raise ValueError("xerr must match scores in shape")

    n_train = row.size
    # NaN scores can't be ranked; reversed argsort would put them on top.
    valid = np.where(~np.isnan(row))[0]
    k_eff = min(k, valid.size // 2 if valid.size >= 2 else valid.size)
    order = valid[np.argsort(row[valid])]  # ascending (most harmful first)
    # Strict signed-value ranking top -> bottom: most helpful first, then
    # decreasing through 0 to most harmful at the bottom.
    helpful_desc = order[-k_eff:][::-1]   # [most_pos, ..., k-th_pos]
    harmful_desc = order[:k_eff][::-1]    # [k-th_neg, ..., most_neg]
    idx = np.concatenate([helpful_desc, harmful_desc])
    vals = row[idx]
    colors = [_HELPFUL if v >= 0 else _HARMFUL for v in vals]
    labels_arr = _as_labels(labels, n_train)
    if labels_arr is None:
        tick = [str(i) for i in idx]
    else:
        tick = [labels_arr[i] for i in idx]

    fig, ax = _ax_or_new(ax, figsize=(6, max(3, 0.3 * len(idx))))
    y = np.arange(len(idx))
    ax.barh(
        y, vals, color=colors,
        xerr=None if err_row is None else err_row[idx],
        error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(tick)
    ax.invert_yaxis()  # helpful on top
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Influence")
    ax.set_ylabel("Training sample")
    # 1-D input has no "test sample" framing (e.g. functional-influence
    # score vectors)
    ax.set_title(
        title if title is not None
        else ("Top influencers" if is_1d
              else f"Top influencers for test sample {test_idx}")
    )
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# 2. Self-influence diagnostic
# -----------------------------------------------------------------------------


def plot_self_influence(
    self_inf: ArrayLike,
    errors: ArrayLike | None = None,
    threshold: float | Literal["auto"] | None = "auto",
    annotate: bool = False,
    labels: ArrayLike | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Visualise the self-influence diagnostic used by ``find_mislabeled``.

    Two modes, selected by whether per-sample errors are passed in:

    - ``errors=None``: histogram of ``|self_inf|`` with the auto threshold
      (mean + 2*std, matching ``find_mislabeled``) drawn as a vertical line.
    - ``errors`` given: scatter of ``|self_inf|`` vs ``errors`` with both
      thresholds drawn; points in the top-right quadrant are flagged.

    Parameters
    ----------
    self_inf : array-like, shape (n_train,)
        Self-influence scores (see ``pyinfluence.self_influence``).
    errors : array-like, shape (n_train,), optional
        Per-sample errors (residuals, 0/1 misclassification, etc.).
    threshold : float, 'auto', or None, default='auto'
        Threshold on ``|self_inf|``. 'auto' uses ``mean + 2*std`` (matching
        the z>2 cutoff used by ``find_mislabeled``). None hides the cutoff.
    annotate : bool, default=False
        If True, label flagged points (scatter mode only). Uses ``labels`` if
        given, otherwise the integer training-sample index.
    labels : array-like, optional
        Per-sample labels used by ``annotate``. List, ndarray, or pandas
        Index/Series.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    s = np.abs(np.asarray(self_inf))

    if threshold == "auto":
        thr = float(s.mean() + 2.0 * s.std())
    else:
        thr = None if threshold is None else float(threshold)

    fig, ax = _ax_or_new(ax)

    if errors is None:
        # Self-influence is typically heavy-tailed - the outliers we care about
        # are by definition rare and far from the bulk. Log y-axis keeps the
        # tail visible without hiding the bulk.
        n_bins = min(50, max(15, s.size // 4))
        ax.hist(s, bins=n_bins, color=_NEUTRAL, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
        if thr is not None:
            ax.axvline(thr, color=_HARMFUL, linestyle="--",
                       label=f"threshold = {thr:.3g}")
            ax.legend(loc="upper right", frameon=False)
        ax.set_xlabel("|Self-influence|")
        ax.set_ylabel("Count (symlog)")
        ax.set_title(title or "Self-influence distribution")
    else:
        err = np.asarray(errors)
        if err.shape != s.shape:
            raise ValueError("errors must have the same shape as self_inf")
        err_thr = float(np.percentile(err, 75))
        flagged = (s > thr if thr is not None else np.zeros_like(s, bool)) & (err > err_thr)
        ax.scatter(s[~flagged], err[~flagged], c=_NEUTRAL, alpha=0.5,
                   edgecolors="none", label="other")
        ax.scatter(s[flagged], err[flagged], c=_HARMFUL, alpha=0.9,
                   edgecolors="black", s=50, label="flagged")
        if thr is not None:
            ax.axvline(thr, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(err_thr, color="gray", linestyle="--", linewidth=0.8)
        if annotate:
            labels_arr = _as_labels(labels, s.size)
            for i in np.where(flagged)[0]:
                tag = labels_arr[i] if labels_arr is not None else str(i)
                ax.annotate(tag, (s[i], err[i]), fontsize=8, alpha=0.8,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("|Self-influence|")
        ax.set_ylabel("Error")
        if flagged.any():
            ax.legend(loc="upper left", frameon=False)
        ax.set_title(title or "Self-influence vs error")

    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# 3. By-group view (merged bar / box / violin)
# -----------------------------------------------------------------------------


def plot_by_group(
    scores: ArrayLike,
    groups: ArrayLike,
    style: Literal["bar", "box", "violin"] = "bar",
    method: Literal["sum", "mean"] = "sum",
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Aggregate influence by group label on training samples.

    ``style='bar'`` shows a single aggregate per group (sum or mean); ``'box'``
    and ``'violin'`` show the per-sample distribution within each group.

    Parameters
    ----------
    scores : array-like, shape (n_test, n_train) or (n_train,)
        Influence scores. 2D input is summed over test samples first.
    groups : array-like, shape (n_train,)
        Group label for each training sample.
    style : {'bar', 'box', 'violin'}, default='bar'
    method : {'sum', 'mean'}, default='sum'
        Aggregation rule for ``style='bar'``. Ignored otherwise.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    s = _aggregate_to_1d(scores, method="sum")
    g = np.asarray(groups)
    if g.shape != s.shape:
        raise ValueError("groups must have shape (n_train,)")

    uniq = np.unique(g)
    fig, ax = _ax_or_new(ax)

    if style == "bar":
        agg = np.array([
            (s[g == u].sum() if method == "sum" else s[g == u].mean())
            for u in uniq
        ])
        colors = [_HELPFUL if v >= 0 else _HARMFUL for v in agg]
        x = np.arange(len(uniq))
        ax.bar(x, agg, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([str(u) for u in uniq])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(f"{method.capitalize()} influence")
    else:
        data = [s[g == u] for u in uniq]
        if style == "violin":
            parts = ax.violinplot(data, showmeans=False, showmedians=True)
            for body in parts["bodies"]:
                body.set_facecolor(_NEUTRAL)
                body.set_alpha(0.6)
        elif style == "box":
            ax.boxplot(data, tick_labels=[str(u) for u in uniq], patch_artist=True,
                       boxprops=dict(facecolor=_NEUTRAL, alpha=0.5))
        else:
            raise ValueError("style must be 'bar', 'box', or 'violin'")
        if style == "violin":
            ax.set_xticks(np.arange(1, len(uniq) + 1))
            ax.set_xticklabels([str(u) for u in uniq])
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Influence")

    ax.set_xlabel("Group")
    ax.set_title(title or f"Influence by group ({style})")
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# 4. Heatmap (top-k subset by default)
# -----------------------------------------------------------------------------


def plot_heatmap(
    scores: ArrayLike,
    top_k: int | None = 25,
    train_labels: ArrayLike | None = None,
    test_labels: ArrayLike | None = None,
    cmap: str = "RdBu",
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Heatmap of an influence matrix, restricted by default to the rows and
    columns with the largest influence mass so the figure stays readable.

    Parameters
    ----------
    scores : array-like, shape (n_test, n_train) or (n_train,)
        A 1D vector is reshaped to ``(1, n_train)``.
    top_k : int or None, default=25
        Keep the ``top_k`` rows (by max |influence|) and ``top_k`` columns
        (by sum |influence|). Pass ``None`` to show all rows and columns.
    train_labels : array-like, optional
        Length-n_train labels used on the x-axis (training samples). List,
        ndarray, or pandas Index/Series. Default: integer indices.
    test_labels : array-like, optional
        Length-n_test labels used on the y-axis (test samples).
    cmap : str, default='RdBu'
        Divergent colormap centred at 0. Red = harmful (negative),
        blue = helpful (positive); matches the sign convention used by the
        other plotting functions.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    _require_mpl()
    s = np.asarray(scores)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    elif s.ndim != 2:
        raise ValueError("scores must be 1D or 2D")

    n_test, n_train = s.shape
    train_labels_arr = _as_labels(train_labels, n_train)
    test_labels_arr = _as_labels(test_labels, n_test)

    row_keep = np.arange(n_test)
    col_keep = np.arange(n_train)
    if top_k is not None:
        if n_test > top_k:
            row_keep = np.argsort(np.abs(s).max(axis=1))[::-1][:top_k]
            row_keep.sort()
        if n_train > top_k:
            col_keep = np.argsort(np.abs(s).sum(axis=0))[::-1][:top_k]
            col_keep.sort()
    s = s[np.ix_(row_keep, col_keep)]

    fig, ax = _ax_or_new(ax, figsize=(8, 5))
    vmax = float(np.abs(s).max()) if s.size else 1.0
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(s, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Influence")

    # Integer ticks at evenly-spaced cell positions. Tick labels use the
    # original (pre-subset) sample indices so users can map a cell back to the
    # underlying sample.
    def _pick_ticks(n: int, max_ticks: int = 10) -> np.ndarray:
        if n <= max_ticks:
            return np.arange(n)
        step = max(1, n // max_ticks)
        return np.arange(0, n, step)

    xt = _pick_ticks(len(col_keep))
    yt = _pick_ticks(len(row_keep))

    def _tick_label(orig_idx: int, labels_arr: np.ndarray | None) -> str:
        return labels_arr[orig_idx] if labels_arr is not None else str(int(orig_idx))

    ax.set_xticks(xt)
    ax.set_xticklabels([_tick_label(col_keep[t], train_labels_arr) for t in xt],
                        rotation=45 if train_labels_arr is not None else 0,
                        ha="right" if train_labels_arr is not None else "center")
    ax.set_yticks(yt)
    ax.set_yticklabels([_tick_label(row_keep[t], test_labels_arr) for t in yt])

    ax.set_xlabel(f"Training sample ({len(col_keep)} of {n_train})")
    ax.set_ylabel(f"Test sample ({len(row_keep)} of {n_test})")
    ax.set_title(title or "Influence heatmap")
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# 5. Method comparison
# -----------------------------------------------------------------------------


def plot_method_comparison(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    names: tuple[str, str] = ("Method A", "Method B"),
    show_correlation: bool = True,
    show_fit: bool = True,
    show_identity: bool = False,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Scatter plot of two attributors' scores against each other.

    Different influence methods often disagree in absolute scale (e.g.,
    InfluenceFunctions tends to be a factor of ~n larger than LOO) while
    still agreeing on the *ranking* of points. For that reason the y=x
    identity line is off by default; the best-fit line and Spearman ρ are
    the more honest signals of agreement.

    Parameters
    ----------
    scores_a, scores_b : array-like
        Influence scores of matching shape from two attributors.
    names : (str, str), default=('Method A', 'Method B')
        Axis labels.
    show_correlation : bool, default=True
        Print Pearson r and Spearman rho in the upper-left corner.
    show_fit : bool, default=True
        Overlay an ordinary-least-squares fit b ~ a and label its slope.
    show_identity : bool, default=False
        Overlay the y=x identity line. Only meaningful if the two methods
        are calibrated to the same scale.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    a = np.asarray(scores_a).ravel()
    b = np.asarray(scores_b).ravel()
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have the same shape")

    fig, ax = _ax_or_new(ax)
    ax.scatter(a, b, c=_NEUTRAL, alpha=0.5, edgecolors="none", s=18)

    info_lines = []
    if show_correlation:
        from scipy import stats
        pr, _ = stats.pearsonr(a, b)
        sr, _ = stats.spearmanr(a, b)
        info_lines.append(f"Pearson r = {pr:.3f}")
        info_lines.append(f"Spearman ρ = {sr:.3f}")

    if show_fit and a.size >= 2 and np.ptp(a) > 0:
        slope, intercept = np.polyfit(a, b, 1)
        xs = np.array([a.min(), a.max()])
        ax.plot(xs, slope * xs + intercept, color=_HELPFUL,
                linewidth=1.5, label=f"fit: slope = {slope:.3g}")
        info_lines.append(f"slope = {slope:.3g}")

    if show_identity:
        lo = float(min(a.min(), b.min()))
        hi = float(max(a.max(), b.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.6,
                label="y = x")

    if info_lines:
        ax.text(
            0.04, 0.96, "\n".join(info_lines),
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgray", alpha=0.85),
        )

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title(title or f"{names[0]} vs {names[1]}")
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# 6. Removal curve
# -----------------------------------------------------------------------------


def plot_removal_curve(
    curve: dict,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot a removal-curve result from ``pyinfluence.removal_curve``.

    Shows mean test loss as a function of the fraction of training points
    removed (by influence ranking) and overlays a random-removal baseline.

    Parameters
    ----------
    curve : dict
        Result of ``removal_curve(...)``. Required keys:
        ``fractions``, ``by_influence``, ``random_mean``, ``random_std``,
        ``direction``.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = _ax_or_new(ax)
    f = np.asarray(curve["fractions"])
    by_inf = np.asarray(curve["by_influence"])
    direction = curve.get("direction", "harmful")

    ax.plot(f, by_inf, color=_HARMFUL if direction == "harmful" else _HELPFUL,
            marker="o", linewidth=1.8,
            label=f"remove {direction} (by influence)")

    rmean = np.asarray(curve.get("random_mean", []))
    rstd = np.asarray(curve.get("random_std", []))
    if rmean.size:
        ax.plot(f, rmean, color=_NEUTRAL, linestyle="--", marker="s",
                linewidth=1.2, label="random baseline")
        ax.fill_between(f, rmean - rstd, rmean + rstd,
                        color=_NEUTRAL, alpha=0.15)

    ax.set_xlabel("Fraction of training data removed")
    ax.set_ylabel("Mean test loss")
    ax.set_title(title or f"Removal curve ({direction})")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# 7. Fairness repair curve
# -----------------------------------------------------------------------------


def plot_disparity_curve(
    curve: dict,
    show_hard: bool = True,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot a fairness repair curve from ``fairness.disparity_removal_curve``.

    Shows the audit-set disparity as a function of the fraction of training
    points removed (most disparity-driving first), against the random-removal
    baseline and the full-model disparity.

    Accuracy is intentionally not drawn on a second axis; read it from
    ``curve['accuracy']`` if you need the fairness/accuracy trade-off.

    Parameters
    ----------
    curve : dict
        Result of ``disparity_removal_curve(...)``. Required keys:
        ``fractions``, ``disparity``, ``base_disparity``; optional:
        ``disparity_hard``, ``random_disparity_mean``,
        ``random_disparity_std``.
    show_hard : bool, default=True
        Also draw the thresholded-decision disparity when present (NaN for
        regressors, in which case it is skipped).
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = _ax_or_new(ax)
    f = np.asarray(curve["fractions"])
    disp = np.asarray(curve["disparity"])

    ax.plot(f, disp, color=_HELPFUL, marker="o", linewidth=1.8,
            label="smoothed disparity (by influence)")

    hard = np.asarray(curve.get("disparity_hard", []))
    if show_hard and hard.size and not np.isnan(hard).all():
        ax.plot(f, hard, color=_HELPFUL, marker="s", linewidth=1.2,
                linestyle=":", alpha=0.8, label="hard-decision disparity")

    rmean = np.asarray(curve.get("random_disparity_mean", []))
    rstd = np.asarray(curve.get("random_disparity_std", []))
    if rmean.size:
        ax.plot(f, rmean, color=_NEUTRAL, linestyle="--", marker="s",
                linewidth=1.2, label="random baseline")
        ax.fill_between(f, rmean - rstd, rmean + rstd,
                        color=_NEUTRAL, alpha=0.15)

    base = curve.get("base_disparity")
    if base is not None and np.isfinite(base):
        ax.axhline(base, color="black", linewidth=0.6, linestyle="-",
                   alpha=0.5, label="full model")

    ax.set_xlabel("Fraction of training data removed")
    ax.set_ylabel("Disparity")
    ax.set_title(title or "Disparity repair curve")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# 8. Mislabel-detection curve
# -----------------------------------------------------------------------------


def plot_detection_curve(
    self_inf: ArrayLike,
    is_corrupted: ArrayLike,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Cumulative recall of known-corrupted samples vs inspection budget.

    Rank training samples by ``|self_inf|`` (largest first) and plot, for
    every inspection budget "check the top x fraction", the fraction of the
    known corruptions found. The standard validation figure for
    ``find_mislabeled``-style workflows: it requires ground truth, so it is
    used on injection experiments (corrupt some labels on purpose) to decide
    whether self-influence ranking works on your data before trusting it.

    Detection difficulty depends strongly on how plausible the injected
    corruption is: gross corruptions (labels far from anything the model
    would predict) give near-perfect curves that are an *upper bound*,
    while plausible errors on near-boundary records can drive any detector
    toward the diagonal. Inject corruptions that look like the errors you
    actually expect.

    Parameters
    ----------
    self_inf : array-like, shape (n_train,)
        Self-influence scores (see ``pyinfluence.self_influence``). NaN
        entries are ranked last (never inspected first).
    is_corrupted : array-like of bool, shape (n_train,)
        Ground-truth corruption mask (True = this sample was corrupted).
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    s = np.abs(np.asarray(self_inf, dtype=float))
    corrupted = np.asarray(is_corrupted, dtype=bool).ravel()
    if corrupted.shape != s.shape:
        raise ValueError("is_corrupted must have the same shape as self_inf")
    n_corr = int(corrupted.sum())
    if n_corr == 0:
        raise ValueError("is_corrupted has no True entries; nothing to detect.")

    n = s.size
    # NaN self-influence last: never credited with early detection
    order = np.argsort(np.where(np.isnan(s), -np.inf, s))[::-1]
    found = np.cumsum(corrupted[order]) / n_corr
    frac_inspected = np.arange(1, n + 1) / n

    fig, ax = _ax_or_new(ax)
    ax.plot(frac_inspected, found, color=_HELPFUL, linewidth=1.8,
            label="by |self-influence|")
    ax.plot([0, 1], [0, 1], color=_NEUTRAL, linestyle="--", linewidth=1.0,
            label="random inspection")
    ax.plot([0, n_corr / n, 1], [0, 1, 1], color=_NEUTRAL, linestyle=":",
            linewidth=1.0, alpha=0.8, label="perfect ranking")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Fraction of training data inspected")
    ax.set_ylabel(f"Corruptions found (of {n_corr})")
    ax.set_title(title or "Mislabel-detection curve")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# 9. Influence concentration (Lorenz-style)
# -----------------------------------------------------------------------------


def plot_influence_concentration(
    scores: ArrayLike,
    mark_share: float = 0.8,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Lorenz-style concentration of influence mass across training samples.

    Sort training samples by ``|influence|`` (largest first) and plot the
    cumulative share of total ``|influence|`` against the fraction of
    samples. Answers "how many points carry the signal?" — a curve hugging
    the top-left means a few samples dominate (inspect those); the diagonal
    means influence is spread uniformly.

    Parameters
    ----------
    scores : array-like, shape (n_test, n_train) or (n_train,)
        Influence scores; 2D input is summed over test samples first. NaN
        scores are excluded (with the remaining mass renormalized).
    mark_share : float or None, default=0.8
        Annotate the smallest sample fraction whose combined mass reaches
        this share (e.g. "12% of samples carry 80% of influence").
        None disables the annotation.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    agg = _aggregate_to_1d(scores, method="sum")
    mass = np.abs(agg)
    mass = mass[~np.isnan(mass)]
    if mass.size == 0 or mass.sum() == 0:
        raise ValueError("scores carry no influence mass (all zero or NaN).")

    mass_sorted = np.sort(mass)[::-1]
    cum_share = np.cumsum(mass_sorted) / mass_sorted.sum()
    frac_samples = np.arange(1, mass.size + 1) / mass.size

    fig, ax = _ax_or_new(ax)
    ax.plot(frac_samples, cum_share, color=_HELPFUL, linewidth=1.8,
            label="cumulative |influence|")
    ax.plot([0, 1], [0, 1], color=_NEUTRAL, linestyle="--", linewidth=1.0,
            label="uniform")

    if mark_share is not None:
        j = int(np.searchsorted(cum_share, mark_share))
        if j < mass.size:
            fx = frac_samples[j]
            ax.plot([fx, fx, 0], [0, mark_share, mark_share],
                    color=_NEUTRAL, linewidth=0.7, linestyle=":")
            ax.annotate(
                f"{fx:.0%} of samples carry {mark_share:.0%} of influence",
                xy=(fx, mark_share), xytext=(fx + 0.03, mark_share - 0.12),
                fontsize=9,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Fraction of training samples (sorted by |influence|)")
    ax.set_ylabel("Cumulative share of total |influence|")
    ax.set_title(title or "Influence concentration")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# 10. Top-k stability across replicates
# -----------------------------------------------------------------------------


def plot_top_k_stability(
    replicate_scores: ArrayLike,
    k: int = 10,
    show: Literal["helpful", "harmful", "abs"] = "abs",
    max_show: int = 25,
    labels: ArrayLike | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    For each training sample, the fraction of replicates in which it appears
    in the top-k. Useful for noisy attributors (Banzhaf, Bootstrap) where the
    user has run several replicates with different seeds.

    Parameters
    ----------
    replicate_scores : array-like, shape (n_rep, n_train) or (n_rep, n_test, n_train)
        Score arrays from independent runs of the same attributor. 3D inputs
        are summed over the test axis before ranking.
    k : int, default=10
        Top-k size used to rank within each replicate.
    show : {'helpful', 'harmful', 'abs'}, default='abs'
        Which extreme to rank by: most-helpful (highest), most-harmful
        (lowest), or largest |influence|.
    max_show : int, default=25
        Cap the number of samples displayed. Samples are ranked by frequency;
        the long tail of singleton appearances is hidden so the chart stays
        readable.
    labels : array-like, optional
        Length-n_train labels for the y-axis. List, ndarray, or pandas
        Index/Series. Default: integer indices.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    arr = np.asarray(replicate_scores)
    if arr.ndim == 3:
        arr = arr.sum(axis=1)
    if arr.ndim != 2:
        raise ValueError(
            "replicate_scores must be 2D (n_rep, n_train) or "
            "3D (n_rep, n_test, n_train)"
        )

    n_rep, n_train = arr.shape
    labels_arr = _as_labels(labels, n_train)
    if show == "helpful":
        rank_vals = arr
    elif show == "harmful":
        rank_vals = -arr
    elif show == "abs":
        rank_vals = np.abs(arr)
    else:
        raise ValueError("show must be 'helpful', 'harmful', or 'abs'")
    # NaN would sort as largest under argpartition; exclude it from top-k
    rank_vals = np.where(np.isnan(rank_vals), -np.inf, rank_vals)

    # Per-replicate top-k indices
    counts = np.zeros(n_train, dtype=int)
    for r in range(n_rep):
        idx = np.argpartition(rank_vals[r], -k)[-k:]
        counts[idx] += 1
    freq = counts / n_rep

    seen = np.where(freq > 0)[0]
    order = seen[np.argsort(-freq[seen])][:max_show]
    truncated = len(seen) > max_show

    fig, ax = _ax_or_new(ax, figsize=(6.5, max(2.5, 0.28 * len(order))))
    y = np.arange(len(order))
    ax.barh(y, freq[order], color=_NEUTRAL)
    ax.set_yticks(y)
    tick = [labels_arr[i] if labels_arr is not None else str(i) for i in order]
    ax.set_yticklabels(tick)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Fraction of {n_rep} replicates in top-{k}")
    ax.set_ylabel("Training sample")
    suffix = f" (showing {len(order)} of {len(seen)})" if truncated else ""
    ax.set_title(title or f"Top-{k} stability ({show}){suffix}")
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# 11. Report wrapper
# -----------------------------------------------------------------------------


def report(
    attributor,
    X_test: ArrayLike,
    y_test: ArrayLike | None = None,
    groups: ArrayLike | None = None,
    errors: ArrayLike | None = None,
    train_labels: ArrayLike | None = None,
    test_labels: ArrayLike | None = None,
    test_idx: int = 0,
    k: int = 10,
    top_k: int = 25,
    save_path: str | None = None,
) -> Figure:
    """
    Four-panel diagnostic dashboard for a fitted attributor.

    Panels (clockwise from top-left):
    - Self-influence histogram with the auto threshold (or scatter vs
      ``errors`` if those are provided).
    - Top-k helpful/harmful for ``test_idx``.
    - Heatmap of the top-``top_k`` rows/cols of the influence matrix.
    - By-group bar chart if ``groups`` is given; otherwise a sorted bar
      chart of aggregate influence (sum over test points) per training
      sample.

    Parameters
    ----------
    attributor : fitted BaseAttributor
    X_test, y_test : test set passed to ``attributor.explain``.
    groups : array-like (n_train,), optional
        Group labels for the bottom-right panel.
    errors : array-like (n_train,), optional
        Per-training-sample error to use in the self-influence panel.
    train_labels, test_labels : array-like, optional
        Per-sample labels (list, ndarray, or pandas Index/Series). Plumbed
        through to ``plot_top_influencers`` and ``plot_heatmap`` so axis
        ticks and bar labels are readable. Default: integer indices.
    test_idx : int, default=0
        Which test point to explain in the top-right panel.
    k : int, default=10
        ``k`` for the top-influencers panel.
    top_k : int, default=25
        ``top_k`` for the heatmap panel.
    save_path : str, optional
        If given, save the figure to this path.

    Returns
    -------
    fig : matplotlib Figure
    """
    plt = _require_mpl()
    from pyinfluence._utils import self_influence as _self_influence

    scores = np.asarray(attributor.explain(X_test, y_test))
    if scores.ndim != 2:
        raise TypeError(
            "report() is a per-test-point dashboard and needs an attributor "
            "whose explain() returns (n_test, n_train) scores. Functional "
            "attributors return a single (n_train,) vector - plot those "
            "with plot_top_influencers(scores) and plot_disparity_curve(...)."
        )
    self_inf = _self_influence(attributor)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    plot_self_influence(
        self_inf, errors=errors, threshold="auto",
        labels=train_labels,
        ax=axes[0, 0],
        title=("Self-influence vs error" if errors is not None
               else "Self-influence distribution"),
    )

    plot_top_influencers(
        scores, test_idx=test_idx, k=k, labels=train_labels,
        ax=axes[0, 1],
        title=f"Top influencers for test sample {test_idx}",
    )

    plot_heatmap(
        scores, top_k=top_k,
        train_labels=train_labels, test_labels=test_labels,
        ax=axes[1, 0],
        title=f"Influence heatmap (top {top_k})",
    )

    if groups is not None:
        plot_by_group(scores, groups, style="bar", method="sum",
                      ax=axes[1, 1], title="Influence by group")
    else:
        agg = _aggregate_to_1d(scores, method="sum")
        order = np.argsort(agg)
        axes[1, 1].bar(
            np.arange(agg.size), agg[order],
            color=[_HELPFUL if v >= 0 else _HARMFUL for v in agg[order]],
        )
        axes[1, 1].axhline(0, color="black", linewidth=0.5)
        axes[1, 1].set_xlabel("Training sample (sorted)")
        axes[1, 1].set_ylabel("Aggregate influence")
        axes[1, 1].set_title("Aggregate influence (sorted)")

    fig.suptitle(
        f"{type(attributor).__name__}  •  mode={getattr(attributor, 'mode', '?')}"
        f"  •  n_train={self_inf.size}  •  n_test={np.asarray(X_test).shape[0]}",
        y=1.005, fontsize=11,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


__all__ = [
    "plot_top_influencers",
    "plot_self_influence",
    "plot_by_group",
    "plot_heatmap",
    "plot_method_comparison",
    "plot_removal_curve",
    "plot_disparity_curve",
    "plot_detection_curve",
    "plot_influence_concentration",
    "plot_top_k_stability",
    "report",
]
