"""Analytic Gaussian channels and the explicitly limited NIS approximation."""

from __future__ import annotations

import numpy as np

from .core import InformationEstimate, covariance, logdet, positive_integer, samples, unit_scale


def linear_gaussian_ei(coupling, noise_covariance, variances, *, sources=None, targets=None, unit="nats"):
    """EI for Y=A X+epsilon with independent Gaussian X and positive definite noise.

    Unselected inputs are marginalized under the same full intervention.
    Rows of A are target coordinates, columns are source coordinates.
    """
    a = np.asarray(coupling, dtype=float)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError("coupling must be a finite nonempty target-by-source matrix.")
    noise = covariance(noise_covariance, a.shape[0], "noise_covariance")
    v = np.asarray(variances, dtype=float)
    if v.shape != (a.shape[1],) or not np.isfinite(v).all() or np.any(v <= 0):
        raise ValueError("variances must be a positive finite vector matching coupling columns.")

    def indices(values, dimension):
        selected = tuple(range(dimension)) if values is None else tuple(values)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(
                isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= dimension
                for i in selected
            )
        ):
            raise ValueError("sources and targets must contain unique, nonempty valid indices.")
        return selected

    s, t = indices(sources, a.shape[1]), indices(targets, a.shape[0])
    b, r = a[list(t)], noise[np.ix_(t, t)]
    omitted = [i for i in range(a.shape[1]) if i not in s]
    total = r + (b * v) @ b.T
    conditional = r + (b[:, omitted] * v[omitted]) @ b[:, omitted].T
    value = 0.5 * (logdet(total) - logdet(conditional))
    return InformationEstimate(
        value * unit_scale(unit),
        unit,
        "linear_gaussian",
        {
            "total_covariance_condition": float(np.linalg.cond(total)),
            "conditional_covariance_condition": float(np.linalg.cond(conditional)),
            "intervention_variances": v,
            "sources": s,
            "targets": t,
            "approximation": "none for the declared linear Gaussian channel",
            "clipped": False,
        },
        "ei",
    )


def dynamics_jacobian(dynamics, points, *, jacobian=None, finite_difference_step=1e-5):
    """Batch central differences, or a supplied batch target-by-source Jacobian."""
    points = samples(points)
    if jacobian is not None:
        result = np.asarray(jacobian(points), dtype=float)
    else:
        if not np.isfinite(finite_difference_step) or finite_difference_step <= 0:
            raise ValueError("finite_difference_step must be finite and positive.")
        columns = []
        for i in range(points.shape[1]):
            step = finite_difference_step * np.maximum(1.0, np.abs(points[:, i]))
            delta = np.zeros_like(points)
            delta[:, i] = step
            plus, minus = samples(dynamics(points + delta)), samples(dynamics(points - delta))
            if len(plus) != len(points) or plus.shape != minus.shape:
                raise ValueError("dynamics must preserve the sample axis and target dimensions.")
            columns.append((plus - minus) / (2 * step[:, None]))
        result = np.stack(columns, axis=2)
    if (
        result.ndim != 3
        or result.shape[0] != len(points)
        or result.shape[2] != points.shape[1]
        or result.shape[1] < 1
        or not np.isfinite(result).all()
    ):
        raise ValueError("jacobian must return finite [samples, targets, sources] values.")
    return result


def nis_ei(
    dynamics,
    *,
    intervention,
    response_noise_cov,
    assume_injective=False,
    sample_count=4096,
    seed=0,
    jacobian=None,
    finite_difference_step=1e-5,
    dimension_average=False,
    unit="nats",
):
    """NIS Appendix B Eq.38: H(X)+E log|det J|-H(Gaussian noise).

    This is a boundary/small-noise approximation, not an exact noisy-channel MI.
    It is unsupported for rectangular, noninjective or singular maps and for
    arbitrary subset EI/SPT. Negative raw approximations are not projected.
    """
    from ..interventions import UniformBox

    if not isinstance(intervention, UniformBox):
        raise TypeError("NIS requires an explicit bounded UniformBox intervention.")
    if assume_injective is not True:
        raise ValueError("Declare assume_injective=True only for a globally injective square map.")
    if not isinstance(dimension_average, (bool, np.bool_)):
        raise TypeError("dimension_average must be a boolean.")
    positive_integer(sample_count, "sample_count", 4)
    x = intervention.sample(sample_count, np.random.default_rng(seed))
    j = dynamics_jacobian(dynamics, x, jacobian=jacobian, finite_difference_step=finite_difference_step)
    d = intervention.dimension
    if j.shape[1] != d:
        raise ValueError(
            "NIS requires a square source-to-target map; rectangular subset channels are unsupported."
        )
    noise = covariance(response_noise_cov, d, "response_noise_cov")
    signs, determinants = np.linalg.slogdet(j)
    bad = (signs == 0) | ~np.isfinite(determinants)
    if bad.any():
        raise ValueError(f"NIS singular Jacobian at {int(bad.sum())} sampled points; no zero-EI fallback.")
    bounds = np.asarray(intervention.bounds)
    input_entropy = float(np.log(bounds[:, 1] - bounds[:, 0]).sum())
    noise_entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet(noise))
    terms = input_entropy + determinants - noise_entropy
    normalization = d if dimension_average else 1
    scale = unit_scale(unit) / normalization
    return InformationEstimate(
        float(terms.mean() * scale),
        unit,
        "nis",
        {
            "input_entropy": input_entropy * unit_scale(unit),
            "noise_entropy": noise_entropy * unit_scale(unit),
            "mean_log_abs_det_jacobian": float(determinants.mean()) * unit_scale(unit),
            "pointwise_ei": terms * scale,
            "sample_count": sample_count,
            "seed": seed,
            **intervention.metadata(),
            "response_noise_cov": noise,
            "dimension_average": bool(dimension_average),
            "normalization_dimension": normalization,
            "information_normalization": "per-source-dimension" if dimension_average else "total",
            "approximation": "injective square-map, boundary/small-noise entropy approximation",
            "minimum_jacobian_abs_det_sign": float(np.abs(signs).min()),
            "clipped": False,
            "negative_pointwise_count": int((terms < 0).sum()),
        },
        "ei",
    )
