"""Switchable EI and MI estimators; optional neural dependencies are loaded lazily."""

from .api import estimate_ei
from .core import EstimatorConfig, InformationEstimate, available_estimators
from .sample import estimate_mi, manuscript_source_features

__all__ = [
    "EstimatorConfig",
    "InformationEstimate",
    "available_estimators",
    "estimate_ei",
    "estimate_mi",
    "manuscript_source_features",
]
