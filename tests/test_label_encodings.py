"""Regression tests: influence scores must not depend on training-label encoding.

A binary classifier fit on {0,1}, {1,2}, {-1,+1}, or string ('neg'/'pos')
labels for the *same* underlying task produces the same fitted decision
function (only ``classes_`` differs); InfluenceFunctions and the fairness
attributors must therefore produce identical scores across encodings. See
``pyinfluence._validation.validate_labels_in_classes`` and the ``y01``
binarization inside ``_fit_logistic_binary`` / the ridge_classifier path in
``pyinfluence._influence.InfluenceFunctions``.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from pyinfluence import InfluenceFunctions
from pyinfluence.fairness import (
    FairnessInfluenceFunctions,
    disparity_value,
    disparity_value_hard,
)

ENCODING_NAMES = ("01", "12", "pm1", "str")


def _encodings(y01):
    """Map a {0,1} label array to equivalent encodings of the same task."""
    y01 = np.asarray(y01)
    return {
        "01": y01.copy(),
        "12": y01 + 1,
        "pm1": np.where(y01 == 1, 1, -1),
        "str": np.where(y01 == 1, "pos", "neg"),
    }


def _make_binary_data(n=150, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    logits = X @ rng.normal(size=p) * 0.9 - 0.1
    y01 = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y01


@pytest.fixture(scope="module")
def binary_encoding_data():
    X, y01 = _make_binary_data()
    n_train = 100
    return X[:n_train], y01[:n_train], X[n_train:], y01[n_train:]


def _make_fairness_data(n=200, p=5, seed=1):
    rng = np.random.default_rng(seed)
    a = (rng.uniform(size=n) < 0.4).astype(float)
    X = rng.normal(size=(n, p))
    X[:, 0] += 1.0 * a
    logits = X @ rng.normal(size=p) * 0.8 + 0.5 * a - 0.2
    y01 = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y01, a


@pytest.fixture(scope="module")
def fairness_encoding_data():
    X, y01, a = _make_fairness_data()
    n_train = 140
    return X[:n_train], y01[:n_train], a[:n_train], X[n_train:], y01[n_train:], a[n_train:]


# -----------------------------------------------------------------------------
# InfluenceFunctions: scores identical across label encodings
# -----------------------------------------------------------------------------


class TestInfluenceFunctionsLabelEncodings:
    @pytest.mark.parametrize("mode", ["loss", "prediction"])
    def test_logistic_regression_scores_identical(self, binary_encoding_data, mode):
        X_train, y_train01, X_test, y_test01 = binary_encoding_data
        encs_train = _encodings(y_train01)
        encs_test = _encodings(y_test01)

        base_scores = None
        for name in ENCODING_NAMES:
            model = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(
                X_train, encs_train[name]
            )
            attr = InfluenceFunctions(mode=mode, damping=1e-5).fit(
                model, X_train, encs_train[name]
            )
            if mode == "loss":
                scores = attr.explain(X_test, encs_test[name])
            else:
                scores = attr.explain(X_test)
            if base_scores is None:
                base_scores = scores
            else:
                np.testing.assert_allclose(
                    scores, base_scores, rtol=1e-8, atol=1e-10,
                    err_msg=f"LogisticRegression encoding {name!r} mismatch (mode={mode})",
                )

    @pytest.mark.parametrize("mode", ["loss", "prediction"])
    def test_ridge_classifier_scores_identical(self, binary_encoding_data, mode):
        X_train, y_train01, X_test, y_test01 = binary_encoding_data
        encs_train = _encodings(y_train01)
        encs_test = _encodings(y_test01)

        base_scores = None
        for name in ENCODING_NAMES:
            model = RidgeClassifier(alpha=1.0).fit(X_train, encs_train[name])
            attr = InfluenceFunctions(mode=mode, damping=1e-5).fit(
                model, X_train, encs_train[name]
            )
            if mode == "loss":
                scores = attr.explain(X_test, encs_test[name])
            else:
                scores = attr.explain(X_test)
            if base_scores is None:
                base_scores = scores
            else:
                np.testing.assert_allclose(
                    scores, base_scores, rtol=1e-8, atol=1e-10,
                    err_msg=f"RidgeClassifier encoding {name!r} mismatch (mode={mode})",
                )


# -----------------------------------------------------------------------------
# Labels outside classes_ raise a clear error, for both fit() and explain()
# -----------------------------------------------------------------------------


class TestLabelValidationErrors:
    @pytest.mark.parametrize("model_cls", [LogisticRegression, RidgeClassifier])
    def test_fit_unknown_label_raises(self, binary_encoding_data, model_cls):
        X_train, y_train01, X_test, y_test01 = binary_encoding_data
        if model_cls is LogisticRegression:
            model = model_cls(max_iter=1000).fit(X_train, y_train01)
        else:
            model = model_cls(alpha=1.0).fit(X_train, y_train01)
        bad_y = y_train01.copy()
        bad_y[0] = 7
        with pytest.raises(ValueError, match="classes_"):
            InfluenceFunctions().fit(model, X_train, bad_y)

    @pytest.mark.parametrize("model_cls", [LogisticRegression, RidgeClassifier])
    def test_explain_unknown_label_raises(self, binary_encoding_data, model_cls):
        X_train, y_train01, X_test, y_test01 = binary_encoding_data
        if model_cls is LogisticRegression:
            model = model_cls(max_iter=1000).fit(X_train, y_train01)
        else:
            model = model_cls(alpha=1.0).fit(X_train, y_train01)
        attr = InfluenceFunctions().fit(model, X_train, y_train01)
        bad_y_test = np.full(X_test.shape[0], 7)
        with pytest.raises(ValueError, match="classes_"):
            attr.explain(X_test, bad_y_test)


# -----------------------------------------------------------------------------
# Fairness attributors: label-encoding invariance and exact metric conventions
# -----------------------------------------------------------------------------


class TestFairnessLabelEncodings:
    @pytest.mark.parametrize("metric", ["dp", "eopp", "fpr", "worst_group_loss"])
    def test_scores_identical_across_encodings(self, fairness_encoding_data, metric):
        X_train, y_train01, a_train, X_test, y_test01, a_test = fairness_encoding_data
        encs_train = _encodings(y_train01)
        encs_test = _encodings(y_test01)

        base = None
        for name in ENCODING_NAMES:
            model = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(
                X_train, encs_train[name]
            )
            attr = FairnessInfluenceFunctions(metric=metric, damping=1e-6).fit(
                model, X_train, encs_train[name], sensitive=a_train
            )
            scores = attr.explain(
                X_test, y_audit=encs_test[name], sensitive_audit=a_test
            )
            if base is None:
                base = scores
            else:
                np.testing.assert_allclose(
                    scores, base, rtol=1e-6, atol=1e-9,
                    err_msg=f"metric={metric!r} encoding {name!r} mismatch",
                )

    def test_disparity_value_eopp_fpr_matches_manual_gap_for_12_encoding(
        self, fairness_encoding_data
    ):
        X_train, y_train01, a_train, X_test, y_test01, a_test = fairness_encoding_data
        y12_train = y_train01 + 1
        y12_test = y_test01 + 1
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(
            X_train, y12_train
        )
        assert list(model.classes_) == [1, 2]

        probs = model.predict_proba(X_test)[:, 1]
        mask_a1 = a_test == 1.0

        for metric, cond in [
            ("eopp", y12_test == model.classes_[1]),
            ("fpr", y12_test != model.classes_[1]),
        ]:
            value = disparity_value(model, X_test, a_test, y=y12_test, metric=metric)
            gap_manual = (
                probs[cond & mask_a1].mean() - probs[cond & ~mask_a1].mean()
            )
            assert value == pytest.approx(gap_manual)

    def test_disparity_value_hard_worst_group_loss_matches_manual_for_pm1_encoding(
        self, fairness_encoding_data
    ):
        X_train, y_train01, a_train, X_test, y_test01, a_test = fairness_encoding_data
        ypm_train = np.where(y_train01 == 1, 1, -1)
        ypm_test = np.where(y_test01 == 1, 1, -1)
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(
            X_train, ypm_train
        )
        assert list(model.classes_) == [-1, 1]

        value = disparity_value_hard(
            model, X_test, a_test, y=ypm_test, metric="worst_group_loss"
        )
        preds = model.predict(X_test)
        err = (preds != ypm_test).astype(float)
        manual = max(err[a_test == g].mean() for g in np.unique(a_test))
        assert value == pytest.approx(manual)
