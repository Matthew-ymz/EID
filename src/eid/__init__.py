"""Standalone EID toolbox extracted from EISyn."""

from .analysis import AnalysisResult, analyze_dynamics, analyze_model, analyze_tpm
from .discrete import build_deterministic_boolean_tpm, effective_information_from_tpm
from .interventions import UniformBox
from .spt import SPTConfig, SPTNonnegativityError, build_spt, build_spt_from_ei_table

__version__ = "0.1.1"

__all__ = [
    "AnalysisResult",
    "SPTConfig",
    "SPTNonnegativityError",
    "UniformBox",
    "analyze_dynamics",
    "analyze_model",
    "analyze_tpm",
    "build_deterministic_boolean_tpm",
    "build_spt",
    "build_spt_from_ei_table",
    "effective_information_from_tpm",
]
