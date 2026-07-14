"""
pyinfluence: Training data attribution for scikit-learn models.

This package provides methods to answer: "Which training examples most
influenced this prediction?" — and, through the functional engine, "which
training examples most influence any scalar property of the model?"

Namespaces
----------
- Top level: the per-test-point attributors (InfluenceFunctions,
  LOOInfluence, BanzhafInfluence, BootstrapInfluence), the scalar-functional
  engine (Functional, FunctionalInfluence, RefitFunctionalInfluence,
  SubsampledFunctionalInfluence, functional_value), the one-shot
  ``influence()`` API, and analysis utilities.
- ``pyinfluence.functionals``: ready-made functionals (group_gap, cohens_d,
  worst_group_mean, mean).
- ``pyinfluence.fairness``: audit vocabulary (disparity) and repair
  workflow (disparity_value, disparity_removal_curve, ...).
- ``pyinfluence.viz``: plotting (optional matplotlib dependency).
"""

from importlib.metadata import PackageNotFoundError, version

from pyinfluence import fairness, functionals
from pyinfluence._banzhaf import BanzhafInfluence
from pyinfluence._base import BaseAttributor
from pyinfluence._bootstrap import BootstrapInfluence
from pyinfluence._functional import (
    Functional,
    FunctionalInfluence,
    RefitFunctionalInfluence,
    SubsampledFunctionalInfluence,
    functional_value,
)
from pyinfluence._influence import InfluenceFunctions
from pyinfluence._loo import LOOInfluence
from pyinfluence._utils import (
    aggregate_influence,
    compare_attributors,
    find_mislabeled,
    influence_by_group,
    influence_summary,
    removal_curve,
    self_influence,
    stability_replicates,
    top_influential,
)
from pyinfluence._validation import supports
from pyinfluence.api import influence

try:
    __version__ = version("pyinfluence")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

__all__ = [
    # High-level API
    "influence",
    "supports",
    # Per-test-point attributors
    "BaseAttributor",
    "InfluenceFunctions",
    "LOOInfluence",
    "BanzhafInfluence",
    "BootstrapInfluence",
    # Scalar-functional engine
    "Functional",
    "FunctionalInfluence",
    "RefitFunctionalInfluence",
    "SubsampledFunctionalInfluence",
    "functional_value",
    # Analysis utilities
    "top_influential",
    "self_influence",
    "influence_summary",
    "find_mislabeled",
    "compare_attributors",
    "aggregate_influence",
    "influence_by_group",
    "removal_curve",
    "stability_replicates",
    # Submodules
    "fairness",
    "functionals",
    "__version__",
]
