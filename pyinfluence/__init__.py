"""
pyinfluence: Training data attribution for scikit-learn models.

This package provides methods to answer: "Which training examples most
influenced this prediction?"
"""

from importlib.metadata import PackageNotFoundError, version

from pyinfluence._base import BaseAttributor
from pyinfluence._influence import InfluenceFunctions
from pyinfluence._loo import LOOInfluence
from pyinfluence._banzhaf import BanzhafInfluence
from pyinfluence._bootstrap import BootstrapInfluence
from pyinfluence._utils import (
    top_influential,
    self_influence,
    influence_summary,
    find_mislabeled,
    compare_attributors,
    aggregate_influence,
    influence_by_group,
    removal_curve,
)
from pyinfluence.api import influence
from pyinfluence import fairness
from pyinfluence.fairness import (
    FairnessInfluenceFunctions,
    RefitFairnessInfluence,
    SubsampledFairnessInfluence,
    disparity_value,
    disparity_value_hard,
    disparity_removal_curve,
    group_removal_effect,
)

try:
    __version__ = version("pyinfluence")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

__all__ = [
    # High-level API
    "influence",
    # Core classes
    "BaseAttributor",
    "InfluenceFunctions",
    "LOOInfluence",
    "BanzhafInfluence",
    "BootstrapInfluence",
    # Utilities
    "top_influential",
    "self_influence",
    "influence_summary",
    "find_mislabeled",
    "compare_attributors",
    "aggregate_influence",
    "influence_by_group",
    "removal_curve",
    # Fairness attribution
    "fairness",
    "FairnessInfluenceFunctions",
    "RefitFairnessInfluence",
    "SubsampledFairnessInfluence",
    "disparity_value",
    "disparity_value_hard",
    "disparity_removal_curve",
    "group_removal_effect",
    "__version__",
]
