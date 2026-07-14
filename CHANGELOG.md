# Changelog

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
