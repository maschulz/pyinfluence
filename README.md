# pyinfluence

Training data attribution for scikit-learn estimators.

`pyinfluence` computes an **influence matrix** $S \in \mathbb{R}^{m \times n}$ that quantifies how each training example $z_j$ affects a model-derived quantity at each test point $x_i$. It is intended for scientific workflows where you need per-example diagnostics (e.g., dataset debugging, auditing, sensitivity analysis), not for producing end-user-facing explanations.

## Installation

```bash
pip install pyinfluence
```

Optional plotting utilities:

```bash
pip install "pyinfluence[viz]"
```

## Quickstart

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from pyinfluence import influence, top_influential

X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Ridge(alpha=1.0).fit(X_train, y_train)

# scores has shape (n_test, n_train)
scores = influence(model, X_train, y_train, X_test, y_test, method="auto", mode="loss")

helpful_idx, harmful_idx = top_influential(scores[0], k=5)
print("Most helpful training indices:", helpful_idx)
print("Most harmful training indices:", harmful_idx)
```

If you will explain multiple test sets, request the fitted attributor:

```python
scores, attr = influence(
    model,
    X_train, y_train,
    X_test, y_test,
    method="auto",
    mode="loss",
    return_attributor=True,
)
scores2 = attr.explain(X_test[:5], y_test[:5])
```

## Worked examples

### 1) Self-influence as an outlier/mislabel heuristic

```python
import numpy as np
from sklearn.linear_model import Ridge

from pyinfluence import InfluenceFunctions, self_influence, find_mislabeled

model = Ridge(alpha=1.0).fit(X_train, y_train)
attr = InfluenceFunctions(mode="loss", damping=1e-5).fit(model, X_train, y_train)

s_self = self_influence(attr)
suspected = find_mislabeled(attr, threshold="auto")

print("Largest |self-influence| indices:", np.argsort(np.abs(s_self))[-10:][::-1])
print("Suspected mislabeled indices:", suspected[:10])
```

Interpretation: `find_mislabeled` is a ranking heuristic (z-scores on $|S_{ii}|$); it is not a proof of label error.

### 2) Comparing methods on the same problem

```python
import numpy as np
from pyinfluence import InfluenceFunctions, LOOInfluence, compare_attributors

attr_if = InfluenceFunctions(mode="loss", damping=1e-5).fit(model, X_train, y_train)
attr_loo = LOOInfluence(mode="loss", n_jobs=-1, verbose=0).fit(model, X_train, y_train)

stats = compare_attributors(attr_if, attr_loo, X_test, y_test, k=10)
print(stats)  # pearson/spearman/kendall + top_k_overlap

scores_if = attr_if.explain(X_test, y_test)
scores_loo = attr_loo.explain(X_test, y_test)
print("Correlation:", np.corrcoef(scores_if.ravel(), scores_loo.ravel())[0, 1])
```

## Output shape and conventions

- `scores` always has shape `(n_test, n_train)`.
- `scores[i, j]` refers to test point `i` and training point `j`.
- In `mode="loss"`: positive indicates “helpful” under the package’s sign convention (see above).

## What is being estimated?

### Notation

- Training set $D = \{z_j\}_{j=1}^n$, with $z_j = (x_j, y_j)$.
- Test inputs $X_{\mathrm{test}} = \{x_i\}_{i=1}^m$.
- Influence matrix $S \in \mathbb{R}^{m \times n}$, where $S_{ij}$ refers to training point $j$ and test point $i$.

### Modes (two estimands)

All attributors support `mode="loss"` and `mode="prediction"`, but **classification prediction mode is method-dependent**.

- **Loss mode** (`mode="loss"`, default):
  - Estimates how a training point affects *test loss*.
  - **Requires `y_test`** for all methods.
  - Intended interpretation: positive values indicate training points that are helpful for reducing loss at the test point.

- **Prediction mode** (`mode="prediction"`):
  - Regression: influence on the scalar prediction $f(x)$. **Does not require `y_test`**.
  - Classification:
    - `LOOInfluence`, `BanzhafInfluence`, `BootstrapInfluence`: influence on the **true-class score** at $x$, so **requires `y_test`** to identify the true class.
    - `InfluenceFunctions` (binary only): influence on the model’s **positive-class probability** $P_\theta(Y=1 \mid x)$ when available, and otherwise on a linear decision value (e.g., for `RidgeClassifier`). It **does not require `y_test`**.

### Sign convention (loss mode)

Loss-mode signs are aligned to a removal interpretation (and validated by tests):

- $S_{ij} > 0$: training point $j$ is **helpful** for test point $i$; removing $z_j$ increases test loss.
- $S_{ij} < 0$: training point $j$ is **harmful** for test point $i$; removing $z_j$ decreases test loss.

## Methods

All methods share the same interface:

```python
attr.fit(model, X_train, y_train)
scores = attr.explain(X_test, y_test)  # shape (n_test, n_train)
```

### Influence functions (`InfluenceFunctions`)

`InfluenceFunctions` implements the classical influence-function approximation for smooth empirical risk minimization, specialized to closed-form Hessians for specific scikit-learn estimators.

For a scalar per-example loss $\ell(z;\theta)$, scores are the **removal-calibrated** influence-function quantity

$$
S_{ij} \;=\; \frac{1}{n}\,\nabla_\theta \ell(z_i^{\mathrm{test}};\theta)^\top \, H_\theta^{-1} \, \nabla_\theta \ell(z_j;\theta),
$$

with

$$
H_\theta \;=\; \frac{1}{n}\sum_{k=1}^n \nabla_\theta^2 \ell(z_k;\theta) \;+\; \lambda I ,
$$

where $\lambda$ is the model's L2 regularization expressed in the per-sample-average objective ($\alpha/n$ for Ridge-family models, $1/(Cn)$ for LogisticRegression; the intercept dimension is not regularized). For KernelRidge the dual-space penalty matrix is $(\lambda/n)K$, which makes the formula exact at the KRR stationarity condition.

The $1/n$ factor calibrates the classical infinitesimal-upweighting derivative to the finite effect of **deleting** one training point (a weight change of $-1/n$), so `InfluenceFunctions` scores estimate the same quantity as `LOOInfluence` — in both sign and magnitude. Agreement with exact leave-one-out retraining (correlation and regression slope ≈ 1) is tested in this repository (`tests/test_loo_agreement.py`).

**Supported estimators (as implemented):**

- `sklearn.linear_model.Ridge`, `RidgeCV`
- `sklearn.linear_model.LinearRegression`
- `sklearn.linear_model.LogisticRegression`, `LogisticRegressionCV` (**binary only**)
- `sklearn.linear_model.RidgeClassifier`, `RidgeClassifierCV` (**binary only**)
- `sklearn.kernel_ridge.KernelRidge` (dual-space influence functions; Hessian is $n \times n$)

**Stability parameter:**

- `damping`: a diagonal term added to the Hessian for numerical stability. Larger values typically reduce influence magnitudes and can reduce numerical instability on ill-conditioned problems.

### Leave-one-out retraining (`LOOInfluence`)

`LOOInfluence` refits the estimator $n$ times (excluding each training point once) and measures the resulting change in loss or prediction:

- Loss mode: $S_{ij} = L_i(D \setminus \{z_j\}) - L_i(D)$.
- Prediction mode (regression): $S_{ij} = f_i(D) - f_i(D \setminus \{z_j\})$.
- Prediction mode (classification): uses the **true-class score**, so requires `y_test`.

This method is model-agnostic but computationally expensive. For classifiers, refits can fail after deletion (e.g., a class disappears); in that case the corresponding scores are NaN.

### Data Banzhaf (`BanzhafInfluence`)

`BanzhafInfluence` estimates an average marginal contribution using Monte Carlo subset sampling. For each point $j$, it samples subsets $S \subseteq D \setminus \{z_j\}$ by including each other point independently with probability $1/2$, then refits models on $S$ and $S \cup \{z_j\}$:

- Loss mode: $S_{ij} \approx \mathbb{E}_S\left[ L_i(S) - L_i(S \cup \{z_j\}) \right]$ (positive indicates the point reduces loss on average).
- Prediction mode: analogous, with the value-at-test being a prediction (regression) or true-class score (classification).

This construction satisfies symmetry and null-player properties up to Monte Carlo error. It does not enforce Shapley efficiency.

### Bootstrap out-of-bag (`BootstrapInfluence`)

`BootstrapInfluence` fits $B$ bootstrap models and uses out-of-bag (OOB) runs to approximate a point's contribution. For each training point $j$, it compares the mean test value across runs where $j$ is OOB to the mean across runs where $j$ is in-bag. Fair "with vs without" comparison: both use ~63% of data; only difference is whether $j$ is in the sample.

OOB counts can be small (or zero) for some points when $B$ is small; in that case scores may be noisy or NaN.

## Choosing a method (practical guidance)

- If your model is supported by `InfluenceFunctions` and you need fast, deterministic estimates: start there.
- If you need model-agnostic influence and $n$ is small enough to refit many times: use `LOOInfluence`.
- If you want a subset-average notion of contribution (data valuation) and can afford repeated refits: use `BanzhafInfluence`.
- If you want a model-agnostic baseline with controllable compute and are comfortable with OOB variability: use `BootstrapInfluence`.

## High-level API: `influence(...)`

`pyinfluence.influence()` provides a one-shot interface that selects an attributor:

- `method="auto"` uses `InfluenceFunctions` for supported linear(-like) models (including binary classifiers and KernelRidge).
- otherwise, or if unsupported: uses `fallback` (default `bootstrap`; alternatives `loo` or `banzhaf`).

```python
from pyinfluence import influence

scores = influence(
    model,
    X_train, y_train,
    X_test, y_test,
    method="auto",
    fallback="bootstrap",
    mode="loss",
)
```

## Utilities (analysis helpers)

- `top_influential(scores, k=10)`: indices of most helpful / most harmful training points.
- `self_influence(attributor, ...)`: diagonal of the influence matrix $S$ for training points explained against themselves.
- `influence_summary(scores, ...)`: summary statistics (mean/std/percentiles/sparsity).
- `find_mislabeled(attributor, threshold="auto")`: flags outliers based on z-scores of absolute self-influence (heuristic).
- `compare_attributors(attr1, attr2, ...)`: correlations and top-k overlap across two methods.
- `aggregate_influence(scores, axis=0, method=...)`: aggregate across test points or train points.
- `influence_by_group(scores, groups, ...)`: aggregate influence by group labels on training samples.
- `removal_curve(attributor, X_test, y_test, ...)`: retrain after dropping the top-k% most harmful (or helpful) training points and compare to a random-removal baseline — the standard validation of whether influence scores carry signal.

## Fairness attribution (`pyinfluence.fairness`)

Attribute **group-disparity functionals** — demographic parity ("dp"), equal-opportunity ("eopp") and FPR ("fpr") gaps, worst-group loss — to individual training examples: `scores[j] ≈ F(D \ z_j) − F(D)` on a fixed audit set.

```python
from sklearn.linear_model import LogisticRegression
from pyinfluence.fairness import FairnessInfluenceFunctions, disparity_removal_curve

model = LogisticRegression(C=1.0, max_iter=2000).fit(X_train, y_train)

attr = FairnessInfluenceFunctions(metric="dp", target="absolute")
attr.fit(model, X_train, y_train)
scores = attr.explain(X_audit, sensitive_audit=a_audit)  # (n_train,)

# retrain-validated repair curve: drop the most disparity-driving points
curve = disparity_removal_curve(scores, model, X_train, y_train,
                                X_audit, a_audit, y_audit=y_audit)
```

- `FairnessInfluenceFunctions` — closed form for supported GLMs (`hessian="identity"` gives the gradient-dot baseline).
- `RefitFairnessInfluence` — exact removal effects via refitting (model-agnostic ground truth).
- `SubsampledFairnessInfluence` — Monte-Carlo subset estimator for arbitrary sklearn estimators (e.g. gradient boosting).
- `disparity_value` / `disparity_value_hard` — smoothed and thresholded metric values; `group_removal_effect` — actual effect of removing a set.

Closed-form scores are validated against exact refitting (correlation *and* slope ≈ 1) in `tests/test_fairness.py`. Note: "fairness influence functions" as *feature*-level decomposition (Ghosh et al., FAccT 2023) is a different quantity; this module attributes disparities to training examples.

## Visualization (`pyinfluence.viz`)

Plotting requires matplotlib (`pip install "pyinfluence[viz]"`). Each function takes pre-computed score arrays (not an attributor), returns `(fig, ax)`, and accepts an optional `ax=` for composition. The set is intentionally small — eight functions, each one chart — plus one report wrapper.

| Function | When to use |
|---|---|
| `plot_top_influencers(scores, test_idx, k=10)` | Explain a single prediction: top-k helpful + top-k harmful for one test point. |
| `plot_self_influence(self_inf, errors=None)` | Find suspect / mislabeled samples. Histogram when `errors=None`, scatter against errors otherwise. |
| `plot_by_group(scores, groups, style='bar'|'box'|'violin')` | Audit influence by data source / subgroup. |
| `plot_heatmap(scores, top_k=25, cluster=False)` | Inspect the influence matrix. Top-k restriction keeps the figure readable as `n_train` grows. |
| `plot_method_comparison(s1, s2)` | Sanity-check two attributors against each other. Shows Pearson/Spearman + best-fit slope. |
| `plot_removal_curve(curve)` | Validate scores by retraining: compare loss-after-removal to a random baseline. Pair with the `removal_curve(attr, ...)` util. |
| `plot_top_k_stability(replicate_scores, k=10)` | Check rank-stability across bootstrap / resampling replicates. |
| `report(attr, X_test, y_test, ...)` | Four-panel diagnostic dashboard in one call. |

See [`examples/showcase.ipynb`](examples/showcase.ipynb) for an end-to-end walkthrough.

## Limitations and failure modes (explicit)

- **Classifier loss semantics**:
  - If a classifier exposes `predict_proba`, loss mode uses negative log-likelihood of the true class.
  - If it does not, loss mode falls back to squared error on `decision_function` values (with a warning). In that case, magnitudes are not comparable to NLL-based losses.
- **Prediction mode is not uniform for classifiers**:
  - Refit-based methods (`LOOInfluence`, `BanzhafInfluence`, `BootstrapInfluence`) use true-class scores and thus require `y_test`.
  - `InfluenceFunctions` (binary) reports influence on the positive-class probability (or a linear decision value for some estimators) and does not require `y_test`.
- **Numerical conditioning (influence functions)**:
  - Influence-function estimates require Hessian inversion; ill-conditioned problems can yield unstable estimates and warnings. `damping` and/or stronger regularization can improve stability.
  - Near-separable logistic regression can yield extreme probabilities and unstable Hessians; the implementation warns in this regime.
- **Compute**:
  - `LOOInfluence` and `BanzhafInfluence` require repeated refits; they can be infeasible for large $n$ or expensive estimators.
  - `BootstrapInfluence` trades bias/variance via $B$; small $B$ can yield noisy or NaN scores for some points.

## References (selected)

- Cook, R. D., & Weisberg, S. (1980). *Characterizations of an empirical influence function for detecting influential cases in regression*. Technometrics.
- Koh, P. W., & Liang, P. (2017). *Understanding black-box predictions via influence functions*. ICML.
- Shapley, L. S. (1953). *A value for n-person games*. Contributions to the Theory of Games.
- Banzhaf, J. F. (1965). *Weighted voting doesn't work: A mathematical analysis*. Rutgers Law Review.
- Ghorbani, A., & Zou, J. (2019). *Data Shapley: Equitable valuation of data for machine learning*. ICML.
- Jia, R., Dao, D., Wang, B., Hubis, F., Hynes, N., Gurel, N., et al. (2019). *Efficient task-specific data valuation for nearest neighbor algorithms*. VLDB.
- Breiman, L. (1996). *Bagging predictors*. Machine Learning.

## How to cite

If you use `pyinfluence` in academic work, cite the repository and the method-specific references above that correspond to your use case.
