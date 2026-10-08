"""Explicit independent uniform interventions on bounded source coordinates."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UniformBox:
    """Factorized maximum-entropy input on a specified Cartesian box."""

    bounds: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        values = np.asarray(self.bounds, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not len(values):
            raise ValueError("bounds must contain one [lower, upper] pair per input coordinate.")
        if not np.isfinite(values).all() or np.any(values[:, 0] >= values[:, 1]):
            raise ValueError("Each intervention bound must be finite with lower < upper.")
        object.__setattr__(self, "bounds", tuple(tuple(map(float, row)) for row in values))

    @property
    def dimension(self) -> int:
        return len(self.bounds)

    def sample(self, sample_count: int, rng: np.random.Generator) -> np.ndarray:
        if not isinstance(sample_count, (int, np.integer)) or sample_count < 4:
            raise ValueError("sample_count must be an integer >= 4.")
        bounds = np.asarray(self.bounds)
        return rng.uniform(bounds[:, 0], bounds[:, 1], size=(sample_count, self.dimension))
