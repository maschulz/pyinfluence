# Changelog

## 0.3.0 (2026-07-15)

Scalar-property attribution is now a first-class, domain-neutral
engine; the fairness module is a thin vocabulary layer over it. No
compatibility shims.

### Added

- Engine (top level): `Functional` (a scalar function of per-sample
  model scores or losses on a reference set, with optional analytic
  gradient), attributed by `FunctionalInfluence` (closed form),
  `RefitFunctionalInfluence` (exact leave-one-out), and
  `SubsampledFunctionalInfluence` (Monte-Carlo subsets); evaluated by
  `functional_value(model, X, F, y)`. `fit` is functional-independent:
  one fitted attributor can explain any number of functionals via
  `explain(..., functional=...)`.
- `pyinfluence.functionals`: builders with analytic gradients —
  `mean(of)`, `group_gap(groups, keep=None, of)`, `cohens_d(groups)`,
  `worst_group_mean(groups, of)`, and `auroc(pos_label)`. Each is
  validated against exact refitting in the test suite.

### Changed (breaking)

- `pyinfluence.fairness` now provides vocabulary and workflow only:
  `disparity(metric, sensitive, *, target_of=... | pos_label=...)` maps
  'dp'/'eopp'/'fpr'/'worst_group_loss' onto the builders;
  `disparity_value(_hard)`, `group_removal_effect`, and
  `disparity_removal_curve` keep their signatures (`metric` accepts a
  name or any `Functional`).
- Removed `FairnessInfluenceFunctions`, `RefitFairnessInfluence`, and
  `SubsampledFairnessInfluence` (replaced by the engine classes plus
  `disparity`), the raw-callable metric form, and the top-level
  re-exports of fairness utilities.
- `cohens_d` moved to `pyinfluence.functionals` as a builder returning
  a `Functional` with an analytic gradient.

## 0.2.0 (2026-07-15)

Correctness release; the numerical claims below are enforced by tests
against exact refitting or brute-force enumeration.

### Breaking changes

- Fairness `explain` audit arguments are keyword-only.
- `viz.plot_heatmap` no longer accepts `cluster=`.
- `BanzhafInfluence` returns NaN (with a warning) instead of 0.0 for
  training points whose subset refits all fail.
- `InfluenceFunctions` rejects configurations it cannot represent
  (`class_weight`, l1/elasticnet penalties) instead of returning
  degraded scores; `influence(method='auto')` falls back with a
  warning.
- `removal_curve` requires `mode='loss'`.
- Labels outside `model.classes_` raise instead of being mapped to the
  negative class.
- viz requires matplotlib >= 3.9.

### Fixed

- Binary label encodings other than {0,1} produced incorrect scores for
  `LogisticRegression` (train- and test-side NLL gradients used raw
  label values). Scores are now invariant across {0,1}, {1,2}, {-1,+1},
  and string encodings; the same fix applies to the fairness closed
  form.
- Fairness `eopp` and `fpr` were swapped for non-{0,1} encodings
  (conditioning on the literal value 1 instead of the model's positive
  class); `disparity_value_hard`'s worst-group error compared decisions
  against raw labels.
- NaN scores corrupted the analysis utilities (`top_influential` ranked
  NaN as most helpful; a single NaN emptied `find_mislabeled` and
  poisoned `influence_summary`/`compare_attributors`). All utilities
  and plots exclude NaN from rankings and statistics, with a warning.
- Small-subset sampling bias in `BanzhafInfluence` (subsets of size one
  were excluded from Monte-Carlo sampling).
- `disparity_removal_curve` reported the wrong `base_disparity` when
  `fractions` did not start at 0.
- Clear early errors replace cryptic failures for sparse inputs,
  multi-output targets, unfitted models passed to refit-based
  attributors, multiclass classifiers without `predict_proba`, and
  wrapped estimators under `method='auto'`.
- `KernelRidge` with a callable kernel failed, and `kernel_params` was
  ignored; the kernel reconstruction now mirrors scikit-learn's.
- `liblinear`-solver logistic models trigger a warning (the regularized
  intercept violates the Hessian assumption).

### Added

- Custom fairness metrics via callables, with analytic or
  finite-difference gradients in the closed form; bundled `cohens_d`.
- Monte-Carlo standard errors: `BanzhafInfluence` and
  `BootstrapInfluence` expose `scores_std_`;
  `viz.plot_top_influencers(..., xerr=...)` draws error bars.
- New plots: `plot_disparity_curve`, `plot_detection_curve`,
  `plot_influence_concentration`.
- `stability_replicates` utility feeding `plot_top_k_stability`.
- `supports(model)`: capability check for `InfluenceFunctions` that
  never raises or warns.
- `self_influence` computes the diagonal directly for
  InfluenceFunctions, LOOInfluence, and BootstrapInfluence (O(n) memory
  instead of the full matrix).
- README figures and an explicit limitations section.

## 0.1.0

Initial release: `InfluenceFunctions`, `LOOInfluence`,
`BanzhafInfluence`, `BootstrapInfluence`, analysis utilities,
`pyinfluence.fairness`, and the `pyinfluence.viz` plotting module.
