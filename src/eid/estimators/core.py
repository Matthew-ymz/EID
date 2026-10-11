"""Declared EI backends, units, and raw numerical results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EstimatorConfig:
    """A backend name and its explicit, backend-specific options."""

    method: str = "tm_polynomial"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "method", canonical_method(self.method))
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class InformationEstimate:
    """An unprojected estimate, with diagnostics in the declared unit."""

    value: float
    unit: str
    estimator: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    quantity: str = "mi"

    def to_dict(self):
        return {
            "value": float(self.value),
            "unit": self.unit,
            "estimator": self.estimator,
            "quantity": self.quantity,
            "diagnostics": json_safe(self.diagnostics),
        }


_METHODS = {
    "discrete": "Exact finite-state TPM; uniform registered inputs",
    "tm_polynomial": "Polynomial autoregressive triangular densities",
    "tm_affine": "Affine Gaussian TM; optional Wishart correction",
    "tm_affine_lifted": "S1.2 scalar/two-scalar feature lifts and corrected affine TM",
    "tm_affine_readout": "Shared full-input affine projection with declared intervention covariance",
    "tm_neural": "One conditional neural TM, with common marginal integration",
    "gaussian": "Gaussian covariance log-determinant MI",
    "kde": "Gaussian kernel density log-ratio MI",
    "ksg": "KSG algorithm 1, Chebyshev k-nearest-neighbor MI",
    "linear_gaussian": "Exact independent Gaussian-input linear channel",
    "local_gaussian": "Local Jacobian Gaussian-channel approximation",
    "nis": "Injective square-map Jacobian entropy approximation",
}
_ALIASES = {
    "tm": "tm_polynomial",
    "transport_map": "tm_polynomial",
    "knn": "ksg",
    "gaussian_logdet": "gaussian",
    "exact-discrete": "discrete",
}


def available_estimators() -> dict[str, str]:
    """Return canonical backend names and their mathematical contracts."""
    return dict(_METHODS)


def canonical_method(method: str) -> str:
    name = _ALIASES.get(method, method)
    if name not in _METHODS:
        raise ValueError(f"Unknown EI estimator {method!r}; choose from {tuple(_METHODS)}.")
    return name


def resolve_config(estimator: str | EstimatorConfig, options=None) -> EstimatorConfig:
    if isinstance(estimator, EstimatorConfig):
        if options is not None:
            raise ValueError("Put options in EstimatorConfig or estimator_options, not both.")
        return estimator
    return EstimatorConfig(estimator, {} if options is None else options)


def unit_scale(unit: str) -> float:
    if unit not in ("nats", "bits"):
        raise ValueError("unit must be 'nats' or 'bits'.")
    return 1.0 if unit == "nats" else 1.0 / np.log(2.0)


def samples(values, name="samples") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite [samples, dimensions] array.")
    return array


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def nonnegative(value, name):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return float(value)


def covariance(values, dimension, name="covariance") -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (dimension, dimension) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite with shape [{dimension}, {dimension}].")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite; specify physical noise explicitly.") from exc
    return matrix


def logdet(matrix) -> float:
    return float(2 * np.log(np.diag(np.linalg.cholesky(matrix))).sum())


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return getattr(value, "__qualname__", type(value).__name__)
    return value
