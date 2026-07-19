"""Tests for sklearn estimator compatibility (get_params, set_params, clone, repr, invalid mode, fit returns self).

Input validation and determinism for InfluenceFunctions are in test_influence.py.
"""

import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge

from pyinfluence import (
    BanzhafInfluence,
    BootstrapInfluence,
    InfluenceFunctions,
    LOOInfluence,
)

# Registry: (attributor_cls, init_kwargs). get_params() must contain init_kwargs.
SKLEARN_ATTRIBUTORS = [
    (InfluenceFunctions, {"mode": "prediction", "damping": 1e-4}),
    (
        LOOInfluence,
        {"mode": "loss", "n_jobs": 1},
    ),  # n_jobs=1 avoids parallel spawn in tests
    (BootstrapInfluence, {"mode": "loss", "n_estimators": 20}),
    (BanzhafInfluence, {"mode": "loss", "n_samples": 30}),
]

# (cls, init_kwargs, set_kwargs): after set_params(**set_kwargs), get_params() contains set_kwargs.
SKLEARN_SET_PARAMS = [
    (
        InfluenceFunctions,
        {"mode": "loss", "damping": 1e-5},
        {"damping": 1e-3, "mode": "prediction"},
    ),
    (LOOInfluence, {"mode": "loss", "n_jobs": 1}, {"mode": "prediction", "n_jobs": 1}),
    (
        BootstrapInfluence,
        {"mode": "loss", "n_estimators": 50},
        {"mode": "prediction", "n_estimators": 30},
    ),
    (
        BanzhafInfluence,
        {"mode": "loss", "n_samples": 20},
        {"mode": "prediction", "n_samples": 50},
    ),
]

# (cls, init_kwargs, substrings that must appear in repr)
SKLEARN_REPR = [
    (
        InfluenceFunctions,
        {"mode": "prediction", "damping": 1e-4},
        ["InfluenceFunctions", "damping", "mode"],
    ),
    (LOOInfluence, {"mode": "prediction", "n_jobs": 4}, ["LOOInfluence", "n_jobs"]),
    (
        BootstrapInfluence,
        {"mode": "prediction", "n_estimators": 30},
        ["BootstrapInfluence", "n_estimators"],
    ),
    (
        BanzhafInfluence,
        {"mode": "loss", "n_samples": 20},
        ["BanzhafInfluence", "n_samples"],
    ),
]

# (cls, fit_kwargs): mode='invalid_mode' + extra kwargs for fit. Raises ValueError on fit.
SKLEARN_INVALID_MODE = [
    (InfluenceFunctions, {"damping": 1e-5, "mode": "invalid_mode"}),
    (LOOInfluence, {"mode": "invalid_mode"}),
    (BootstrapInfluence, {"mode": "invalid_mode", "n_estimators": 5}),
    (BanzhafInfluence, {"mode": "invalid_mode", "n_samples": 10}),
]


def _id_attributor(entry):
    if hasattr(entry, "values"):
        t = entry.values[0]
    else:
        t = entry
    cls = t[0] if isinstance(t, (list, tuple)) else t
    return getattr(cls, "__name__", str(cls))


@pytest.mark.parametrize("entry", SKLEARN_ATTRIBUTORS, ids=_id_attributor)
class TestGetParams:
    """get_params() returns init params."""

    def test_get_params(self, entry):
        cls, kwargs = entry
        attr = cls(**kwargs)
        params = attr.get_params()
        for k, v in kwargs.items():
            assert params.get(k) == v, f"get_params()[{k!r}]"


@pytest.mark.parametrize("entry", SKLEARN_SET_PARAMS, ids=_id_attributor)
class TestSetParams:
    """set_params() updates state and get_params() reflects it."""

    def test_set_params(self, entry):
        cls, init_kwargs, set_kwargs = entry
        attr = cls(**init_kwargs)
        attr.set_params(**set_kwargs)
        params = attr.get_params()
        for k, v in set_kwargs.items():
            assert params.get(k) == v, f"after set_params, get_params()[{k!r}]"


@pytest.mark.parametrize("entry", SKLEARN_ATTRIBUTORS, ids=_id_attributor)
class TestCloneable:
    """clone() produces unfitted copy with same params."""

    def test_cloneable(self, entry):
        cls, kwargs = entry
        attr = cls(**kwargs)
        attr_clone = clone(attr)
        for k, v in kwargs.items():
            assert attr_clone.get_params().get(k) == v
        assert not hasattr(attr_clone, "model_")


@pytest.mark.parametrize("entry", SKLEARN_REPR, ids=_id_attributor)
class TestRepr:
    """__repr__ shows class name and parameters."""

    def test_repr(self, entry):
        cls, kwargs, substrings = entry
        attr = cls(**kwargs)
        repr_str = repr(attr)
        for s in substrings:
            assert s in repr_str, f"{s!r} not in repr"


@pytest.mark.parametrize("entry", SKLEARN_INVALID_MODE, ids=_id_attributor)
class TestInvalidModeRaises:
    """Invalid mode raises ValueError on fit."""

    def test_invalid_mode_raises_on_fit(self, entry, regression_data):
        cls, fit_kwargs = entry
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        attr = cls(**fit_kwargs)
        with pytest.raises(ValueError, match="Invalid mode"):
            attr.fit(model, X_train, y_train)


@pytest.mark.parametrize("entry", SKLEARN_ATTRIBUTORS, ids=_id_attributor)
class TestFitReturnsSelf:
    """fit() returns self per sklearn convention."""

    def test_fit_returns_self(self, entry, regression_data):
        cls, kwargs = entry
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        attr = cls(**kwargs)
        result = attr.fit(model, X_train, y_train)
        assert result is attr


class TestFittedCloneIsUnfitted:
    """Clone of fitted attributor is unfitted."""

    def test_fitted_attributor_clone_is_unfitted(self, regression_data):
        """Clone of fitted attributor should be unfitted."""
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        attr = InfluenceFunctions(mode="loss", damping=1e-5)
        attr.fit(model, X_train, y_train)
        assert hasattr(attr, "model_")
        assert hasattr(attr, "H_inv_")
        attr_clone = clone(attr)
        assert not hasattr(attr_clone, "model_")
        assert not hasattr(attr_clone, "H_inv_")


# Note: We don't use parametrize_with_checks because our estimators have a
# non-standard fit(model, X, y) signature that doesn't match sklearn's fit(X, y).
# The tests above cover the relevant sklearn conventions that do apply.
