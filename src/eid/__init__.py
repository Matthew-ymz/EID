"""Standalone EID toolbox extracted from EISyn."""

from .analysis import (
    AnalysisResult,
    analyze_dynamics,
    analyze_linear_gaussian,
    analyze_model,
    analyze_samples,
    analyze_tpm,
)
from .discrete import build_deterministic_boolean_tpm, effective_information_from_tpm
from .estimators import EstimatorConfig, InformationEstimate, available_estimators, estimate_ei, estimate_mi
from .interventions import IndependentEmpirical, IndependentGaussian, UniformBox
from .spt import SPTConfig, SPTNonnegativityError, build_spt, build_spt_from_ei_table

__version__ = "0.2.0"

__all__ = [
    "AnalysisResult",
    "EstimatorConfig",
    "IndependentEmpirical",
    "IndependentGaussian",
    "InformationEstimate",
    "SPTConfig",
    "SPTNonnegativityError",
    "UniformBox",
    "analyze_dynamics",
    "analyze_linear_gaussian",
    "analyze_model",
    "analyze_samples",
    "analyze_tpm",
    "available_estimators",
    "build_deterministic_boolean_tpm",
    "build_spt",
    "build_spt_from_ei_table",
    "effective_information_from_tpm",
    "estimate_ei",
    "estimate_mi",
]
