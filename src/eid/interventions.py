"""Explicit factorized intervention distributions; never infer these from observations."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

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

    def from_unit(self, values: np.ndarray) -> np.ndarray:
        bounds = np.asarray(self.bounds)
        return bounds[:, 0] + values * (bounds[:, 1] - bounds[:, 0])

    def metadata(self) -> dict:
        return {"intervention": "independent-uniform-box", "bounds": [list(b) for b in self.bounds]}


@dataclass(frozen=True)
class IndependentGaussian:
    """Maximum entropy under separately prescribed source means and variances."""

    mean: Sequence[float]
    std: Sequence[float]

    def __post_init__(self):
        mean, std = np.asarray(self.mean, dtype=float), np.asarray(self.std, dtype=float)
        if (
            mean.ndim != 1
            or not len(mean)
            or std.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ValueError("mean and std must be matching finite vectors with positive std.")
        object.__setattr__(self, "mean", tuple(map(float, mean)))
        object.__setattr__(self, "std", tuple(map(float, std)))

    @property
    def dimension(self):
        return len(self.mean)

    def sample(self, sample_count, rng):
        if not isinstance(sample_count, (int, np.integer)) or sample_count < 4:
            raise ValueError("sample_count must be an integer >= 4.")
        return rng.normal(self.mean, self.std, size=(sample_count, self.dimension))

    def from_unit(self, values):
        from scipy.special import ndtri

        # Sobol points live in [0,1); prevent an infinite quantile at an endpoint.
        return np.asarray(self.mean) + np.asarray(self.std) * ndtri(
            np.clip(values, np.finfo(float).eps, 1 - np.finfo(float).eps)
        )

    def metadata(self):
        return {
            "intervention": "independent-gaussian-moment-constrained",
            "mean": list(self.mean),
            "std": list(self.std),
        }


@dataclass(frozen=True)
class IndependentEmpirical:
    """Independent uniform draws from registered block repertoires.

    Dependence within a block is retained. This is a discrete repertoire
    intervention, not a maximum-entropy continuous density on a box.
    """

    blocks: Sequence[np.ndarray]

    def __post_init__(self):
        from .estimators.core import samples

        blocks = tuple(samples(b).copy() for b in self.blocks)
        if not blocks:
            raise ValueError("Register at least one empirical source block.")
        for b in blocks:
            b.setflags(write=False)
        object.__setattr__(self, "blocks", blocks)

    @property
    def dimension(self):
        return sum(b.shape[1] for b in self.blocks)

    @property
    def source_groups(self):
        edges = np.cumsum([0] + [b.shape[1] for b in self.blocks])
        return tuple(tuple(range(a, b)) for a, b in pairwise(edges))

    def sample(self, sample_count, rng):
        if not isinstance(sample_count, (int, np.integer)) or sample_count < 4:
            raise ValueError("sample_count must be an integer >= 4.")
        return np.column_stack([b[rng.integers(len(b), size=sample_count)] for b in self.blocks])

    def from_unit(self, values):
        return np.column_stack(
            [
                b[np.minimum((values[:, i] * len(b)).astype(int), len(b) - 1)]
                for i, b in enumerate(self.blocks)
            ]
        )

    @property
    def sampling_dimension(self):
        return len(self.blocks)

    def metadata(self):
        return {
            "intervention": "independent-uniform-empirical-block-repertoires",
            "block_repertoire_sizes": [len(b) for b in self.blocks],
            "source_groups": [list(g) for g in self.source_groups],
            "maximum_entropy_scope": "registered discrete block repertoires",
        }
