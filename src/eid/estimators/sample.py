"""Sample MI estimators. MI becomes EI only under declared interventions."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import gaussian_kde

from .core import (
    EstimatorConfig,
    InformationEstimate,
    logdet,
    nonnegative,
    positive_integer,
    resolve_config,
    samples,
    unit_scale,
)
from .transport_map import fit_affine_transport_map_density, fit_polynomial_triangular_transport_map_density


def manuscript_source_features(x):
    """Exactly S12: scalar [x,x²,x³], pair [x1,x2,x1*x2,x1²,x2²]."""
    x = samples(x)
    if x.shape[1] == 1:
        return np.column_stack([x[:, 0], x[:, 0] ** 2, x[:, 0] ** 3])
    if x.shape[1] == 2:
        a, b = x.T
        return np.column_stack([a, b, a * b, a**2, b**2])
    raise ValueError("S1.2 feature lifting is defined only for one or two scalar source coordinates.")


def wishart_mi_bias(sample_count, source_dim, target_dim):
    """S14, in nats; fail instead of silently dropping an invalid correction."""
    if sample_count <= source_dim + target_dim:
        raise ValueError("Wishart correction requires sample_count > source_dim + target_dim.")

    def c(d):
        return float(
            digamma((sample_count - np.arange(1, d + 1)) / 2).sum()
            + d * np.log(2)
            - d * np.log(sample_count - 1)
        )

    return 0.5 * (c(source_dim) + c(target_dim) - c(source_dim + target_dim))


def _density_mi(
    x,
    y,
    ex,
    ey,
    *,
    method,
    degree=3,
    joint_order="source_first",
    ridge=1e-6,
    min_scale=1e-8,
    jitter=1e-6,
    bias_correction=None,
    sample_weight=None,
):
    positive_integer(degree, "degree")
    nonnegative(ridge, "ridge")
    if not np.isfinite(min_scale) or min_scale <= 0:
        raise ValueError("min_scale must be finite and positive.")
    nonnegative(jitter, "jitter")
    if joint_order not in ("source_first", "target_first"):
        raise ValueError("joint_order must be 'source_first' or 'target_first'.")
    correction_on = (method == "tm_affine_lifted") if bias_correction is None else bias_correction
    if not isinstance(correction_on, (bool, np.bool_)):
        raise TypeError("bias_correction must be a boolean.")
    if method == "tm_polynomial" and correction_on:
        raise ValueError("Wishart correction is specific to affine Gaussian TM.")
    if correction_on and (ex is not x or sample_weight is not None):
        raise ValueError("Wishart correction requires unweighted same-sample fitting and evaluation.")
    if method == "tm_affine_lifted":
        original_x = x
        x = manuscript_source_features(x)
        ex = x if ex is original_x else manuscript_source_features(ex)
    blocks, eval_blocks = (x, y), (ex, ey)
    if joint_order == "target_first":
        blocks, eval_blocks = blocks[::-1], eval_blocks[::-1]
    joint, ejoint = np.concatenate(blocks, axis=1), np.concatenate(eval_blocks, axis=1)
    if method == "tm_polynomial":

        def fit(z):
            return fit_polynomial_triangular_transport_map_density(
                z, degree=degree, ridge=ridge, min_scale=min_scale, sample_weight=sample_weight
            )
    else:
        if sample_weight is not None:
            raise ValueError("Affine TM does not support sample_weight.")

        def fit(z):
            return fit_affine_transport_map_density(z, jitter=jitter)

    joint_model, source_model, target_model = fit(joint), fit(x), fit(y)
    pointwise = joint_model.log_prob(ejoint) - source_model.log_prob(ex) - target_model.log_prob(ey)
    correction = wishart_mi_bias(len(x), x.shape[1], y.shape[1]) if correction_on else 0.0
    if sample_weight is None:
        weight = np.full(len(ex), 1 / len(ex))
    else:
        if ex is not x:
            raise ValueError("sample_weight currently requires same-sample evaluation.")
        weight = np.asarray(sample_weight, dtype=float)
        if (
            weight.shape != (len(x),)
            or not np.isfinite(weight).all()
            or np.any(weight < 0)
            or weight.sum() <= 0
        ):
            raise ValueError("sample_weight must be finite, nonnegative and match sample_count.")
        weight = weight / weight.sum()
    raw = float(weight @ pointwise)
    source_condition = float(np.linalg.cond(np.cov(x, rowvar=False).reshape(x.shape[1], -1)))
    return raw - correction, {
        "uncorrected_value": raw,
        "bias_correction": correction,
        "pointwise_mi": pointwise - correction,
        "joint_order": joint_order,
        "source_feature_dimension": x.shape[1],
        "target_dimension": y.shape[1],
        "effective_sample_size": float(1 / (weight @ weight)),
        "density_family": joint_model.backend,
        "feature_lift": "S12" if method == "tm_affine_lifted" else "none",
        "fit_source_condition": source_condition if np.isfinite(source_condition) else None,
        "singular_source_sample_covariance": not np.isfinite(source_condition),
    }


def _gaussian_mi(x, y, *, jitter=1e-6, bias_correction=False):
    nonnegative(jitter, "jitter")
    if not isinstance(bias_correction, (bool, np.bool_)):
        raise TypeError("bias_correction must be a boolean.")

    def cov(z):
        return np.atleast_2d(np.cov(z, rowvar=False)) + jitter * np.eye(z.shape[1])

    cx, cy, cj = cov(x), cov(y), cov(np.column_stack([x, y]))
    try:
        raw = 0.5 * (logdet(cx) + logdet(cy) - logdet(cj))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Gaussian sample covariance is singular; specify jitter or change estimator."
        ) from exc
    correction = wishart_mi_bias(len(x), x.shape[1], y.shape[1]) if bias_correction else 0.0
    return raw - correction, {
        "uncorrected_value": raw,
        "bias_correction": correction,
        "joint_covariance_condition": float(np.linalg.cond(cj)),
        "jitter": jitter,
    }


def _kde_mi(x, y, ex, ey, *, bandwidth="scott"):
    try:
        models = [gaussian_kde(z.T, bw_method=bandwidth) for z in (x, y, np.column_stack([x, y]))]
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ValueError("KDE requires nonsingular training densities and sufficient samples.") from exc
    terms = models[2].logpdf(np.column_stack([ex, ey]).T) - models[0].logpdf(ex.T) - models[1].logpdf(ey.T)
    return float(terms.mean()), {
        "pointwise_mi": terms,
        "bandwidth_factors": [m.factor for m in models],
        "kernel": "gaussian",
        "bias_correction": 0.0,
    }


def _ksg_mi(x, y, *, k=5, standardize=True, tie_noise=0.0, seed=0):
    k = positive_integer(k, "k")
    if not isinstance(standardize, (bool, np.bool_)):
        raise TypeError("standardize must be a boolean.")
    if k >= len(x):
        raise ValueError("k must be smaller than sample_count.")
    nonnegative(tie_noise, "tie_noise")
    joint = np.column_stack([x, y]).copy()
    duplicate_count = len(joint) - len(np.unique(joint, axis=0))
    if duplicate_count and not tie_noise:
        raise ValueError("KSG encountered duplicate rows; declare positive tie_noise or use discrete EI.")
    if standardize:
        scales = joint.std(axis=0)
        if np.any(scales == 0):
            raise ValueError("KSG needs nonconstant continuous coordinates; remove constants explicitly.")
        joint = (joint - joint.mean(axis=0)) / scales
    if tie_noise:
        joint += np.random.default_rng(seed).normal(scale=tie_noise, size=joint.shape)
    xt, yt, jt = cKDTree(joint[:, : x.shape[1]]), cKDTree(joint[:, x.shape[1] :]), cKDTree(joint)
    radii = jt.query(joint, k=k + 1, p=np.inf)[0][:, k]
    if np.any(radii <= 0):
        raise ValueError(
            "KSG encountered tied zero distances; declare positive tie_noise or use discrete EI."
        )
    # Strict marginal radius: count self, so nx+1 is returned directly.
    radius = np.nextafter(radii, 0)
    nx = xt.query_ball_point(joint[:, : x.shape[1]], radius, p=np.inf, return_length=True)
    ny = yt.query_ball_point(joint[:, x.shape[1] :], radius, p=np.inf, return_length=True)
    terms = digamma(k) + digamma(len(x)) - digamma(nx) - digamma(ny)
    return float(terms.mean()), {
        "pointwise_mi": terms,
        "k": k,
        "algorithm": "KSG-1",
        "metric": "chebyshev",
        "standardize": standardize,
        "tie_noise": tie_noise,
        "duplicate_row_count": duplicate_count,
        "minimum_neighbor_radius": float(radii.min()),
        "bias_correction": 0.0,
    }


def estimate_mi(
    x,
    y,
    *,
    estimator: str | EstimatorConfig = "tm_polynomial",
    estimator_options=None,
    unit="nats",
    evaluation_x=None,
    evaluation_y=None,
) -> InformationEstimate:
    """Estimate MI from paired samples, without assuming causal intervention.

    Explicit evaluation samples enable held-out TM/KDE density evaluation.
    Raw negative values are retained; this function never clips information.
    """
    config = resolve_config(estimator, estimator_options)
    scale = unit_scale(unit)
    x, y = samples(x, "x"), samples(y, "y")
    if len(x) != len(y) or len(x) < 4:
        raise ValueError("x and y must have matching sample counts, at least four.")
    if (evaluation_x is None) != (evaluation_y is None):
        raise ValueError("Supply both evaluation_x and evaluation_y.")
    ex, ey = (x, y) if evaluation_x is None else (samples(evaluation_x), samples(evaluation_y))
    if len(ex) != len(ey) or ex.shape[1] != x.shape[1] or ey.shape[1] != y.shape[1]:
        raise ValueError("Evaluation sample shapes must match paired training dimensions.")
    method, opts = config.method, config.options
    accepted = {
        "tm_polynomial": {"degree", "joint_order", "ridge", "min_scale", "sample_weight"},
        "tm_affine": {"jitter", "bias_correction", "joint_order"},
        "tm_affine_lifted": {"jitter", "bias_correction", "joint_order"},
        "gaussian": {"jitter", "bias_correction"},
        "kde": {"bandwidth"},
        "ksg": {"k", "standardize", "tie_noise", "seed"},
    }
    if method in accepted and (unknown := opts.keys() - accepted[method]):
        raise TypeError(f"Unsupported options for {method}: {sorted(unknown)}.")
    if method in ("tm_polynomial", "tm_affine", "tm_affine_lifted"):
        value, diagnostics = _density_mi(x, y, ex, ey, method=method, **opts)
    elif method == "gaussian":
        if evaluation_x is not None:
            raise ValueError(
                "Gaussian logdet estimates the fitted model; use tm_affine for held-out evaluation."
            )
        value, diagnostics = _gaussian_mi(x, y, **opts)
    elif method == "kde":
        value, diagnostics = _kde_mi(x, y, ex, ey, **opts)
    elif method == "ksg":
        if evaluation_x is not None:
            raise ValueError("KSG uses neighbor counts on one sample ensemble, not held-out densities.")
        value, diagnostics = _ksg_mi(x, y, **opts)
    else:
        raise ValueError(
            f"{method} is not a paired-sample MI backend; use estimate_ei or its analysis entry."
        )
    for key in ("uncorrected_value", "bias_correction", "pointwise_mi"):
        if key in diagnostics:
            diagnostics[key] = diagnostics[key] * scale
    diagnostics.update(
        train_count=len(x),
        evaluation_count=len(ex),
        density_fit_and_evaluation="same-samples" if ex is x else "held-out",
        options=dict(opts),
        clipped=False,
    )
    if not np.isfinite(value):
        raise ValueError(f"Non-finite MI from {method}.")
    return InformationEstimate(float(value * scale), unit, method, diagnostics)
