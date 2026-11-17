"""Scalar marginal transport-map EI with known additive Gaussian response noise."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import ndtr, ndtri

from .core import InformationEstimate, positive_integer, unit_scale

LN2 = math.log(2)
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def _deposit(centers, weights, step):
    """Preserve mass and center mean by distributing to adjacent grid nodes."""
    scaled = centers / step
    if not np.isfinite(scaled).all() or np.ptp(scaled) > 999_998:
        raise ValueError("Marginal TM grid exceeds one million bins; check bandwidth and response scale.")
    lower = math.floor(float(centers.min()) / step)
    positions = centers / step - lower
    left = np.floor(positions).astype(np.int64)
    fraction = positions - left
    size = int(left.max()) + 2
    if size > 1_000_000:
        raise ValueError("Marginal TM grid exceeds one million bins; check bandwidth and response scale.")
    bins = np.bincount(left, weights * (1 - fraction), minlength=size)
    bins += np.bincount(left + 1, weights * fraction, minlength=size)
    return bins


def _tm_entropy(centers, weights, width, bins_per_width):
    """Evaluate the discretized mixture's map and its analytic Jacobian."""
    step = width / bins_per_width
    center_mass = _deposit(centers, weights, step)
    radius = 10 * bins_per_width
    z = np.arange(-radius, radius + 1) / bins_per_width
    density = np.convolve(center_mass, np.exp(-z * z / 2 - HALF_LOG_2PI) / width)

    # Gaussian cell masses integrate the same mixture analytically between
    # evaluation nodes. Integrate separately from either tail for stability.
    edges = (np.arange(-radius - 1, radius + 1) + 0.5) / bins_per_width
    gaussian_cells = np.diff(ndtr(edges))
    right = edges[:-1] >= 0
    gaussian_cells[right] = np.diff(-ndtr(-edges))[right]
    cells = np.convolve(center_mass, gaussian_cells)

    # Evaluate F at actual density grid nodes, using a shifted cell grid.
    half_cells = np.diff(ndtr(z))
    right = z[:-1] >= 0
    half_cells[right] = np.diff(-ndtr(-z))[right]
    intervals = np.convolve(center_mass, half_cells)
    left_cdf = np.r_[0.0, np.cumsum(intervals)]
    right_cdf = np.r_[np.cumsum(intervals[::-1])[::-1], 0.0]
    valid = (density > 0) & (left_cdf > 0) & (right_cdf > 0)
    map_value = np.zeros_like(density)
    map_value[valid] = np.where(
        left_cdf[valid] < right_cdf[valid],
        ndtri(left_cdf[valid]),
        -ndtri(right_cdf[valid]),
    )
    # ndtri of the unselected branch may reach infinity at extreme tails.
    log_reference = -HALF_LOG_2PI - 0.5 * map_value[valid] ** 2
    log_density = np.log(density[valid])
    log_jacobian = log_density - log_reference
    log_mapped_density = log_reference + log_jacobian
    mapped_density = np.zeros_like(density)
    mapped_density[valid] = np.exp(log_mapped_density)
    integrand = np.zeros_like(density)
    integrand[valid] = -mapped_density[valid] * log_mapped_density
    mass = float(trapezoid(mapped_density, dx=step))
    mass_error = max(abs(mass - 1), abs(float(cells.sum()) - 1))
    density_error = float(np.max(np.abs(mapped_density - density)))
    map_mean = float(trapezoid(mapped_density * map_value, dx=step))
    map_second_moment = float(trapezoid(mapped_density * map_value**2, dx=step))
    entropy = float(trapezoid(integrand, dx=step) / LN2)
    return entropy, mass_error, density_error, map_mean, map_second_moment


def marginal_gaussian_ei(
    target, *, response_noise_std, bandwidth=0.05, resolution=25, sample_weight=None, unit="nats"
):
    """Estimate full-input EI as marginal entropy minus known conditional noise entropy."""
    if not np.isfinite(response_noise_std) or response_noise_std <= 0:
        raise ValueError("response_noise_std must be finite and positive.")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive.")
    if positive_integer(resolution, "resolution", 4) > 1000:
        raise ValueError("resolution must be at most 1000 bins per bandwidth.")
    if target.shape[1] != 1:
        raise ValueError("Marginal TM requires a scalar target.")
    centers = target[:, 0]
    if sample_weight is None:
        weight = np.full(len(centers), 1 / len(centers))
    else:
        weight = sample_weight / sample_weight.sum()
    coarse, *_ = _tm_entropy(centers, weight, bandwidth, resolution)
    entropy, mass_error, density_error, mean, second = _tm_entropy(centers, weight, bandwidth, 2 * resolution)
    grid_error = abs(entropy - coarse)
    if (
        not np.isfinite([entropy, mass_error, density_error, mean, second, grid_error]).all()
        or mass_error > 1e-8
        or density_error > 1e-8
        or abs(mean) > 1e-5
        or abs(second - 1) > 1e-5
        or grid_error > 0.002
    ):
        raise ValueError(
            f"Marginal TM validation failed: mass={mass_error}, density={density_error}, "
            f"mean={mean}, second={second}, grid_error_bits={grid_error}"
        )
    scale = LN2 * unit_scale(unit)
    noise_entropy = (HALF_LOG_2PI + math.log(response_noise_std) + 0.5) / LN2
    return InformationEstimate(
        (entropy - noise_entropy) * scale,
        unit,
        "tm_marginal",
        {
            "marginal_entropy": entropy * scale,
            "conditional_entropy": noise_entropy * scale,
            "mass_error": mass_error,
            "map_density_error": density_error,
            "map_mean": mean,
            "map_second_moment": second,
            "grid_error": grid_error * scale,
            "bandwidth": bandwidth,
            "response_noise_std": response_noise_std,
            "resolution": resolution,
            "clipped": False,
        },
        "ei",
    )
