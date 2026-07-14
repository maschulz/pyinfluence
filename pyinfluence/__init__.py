"""
pyinfluence: Training data attribution for scikit-learn models.

This package provides methods to answer: "Which training examples most
influenced this prediction?"
"""

from importlib.metadata import PackageNotFoundError, version

from pyinfluence import fairness
from pyinfluence._banzhaf import BanzhafInfluence
from pyinfluence._base import BaseAttributor
from pyinfluence._bootstrap import BootstrapInfluence
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
from pyinfluence.fairness import (
    FairnessInfluenceFunctions,
    RefitFairnessInfluence,
    SubsampledFairnessInfluence,
    cohens_d,
    disparity_removal_curve,
    disparity_value,
    disparity_value_hard,
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
    "stability_replicates",
    "supports",
    # Fairness attribution
    "fairness",
    "FairnessInfluenceFunctions",
    "RefitFairnessInfluence",
    "SubsampledFairnessInfluence",
    "cohens_d",
    "disparity_value",
    "disparity_value_hard",
    "disparity_removal_curve",
    "group_removal_effect",
    "__version__",
]
