"""Unified EI dispatch and reusable subset evaluators."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .analytic import dynamics_jacobian, linear_gaussian_ei, nis_ei
from .core import EstimatorConfig, InformationEstimate, json_safe, resolve_config, samples, unit_scale
from .marginal import marginal_gaussian_ei
from .sample import estimate_mi


def source_groups(dimension, groups=None):
    result = tuple((i,) for i in range(dimension)) if groups is None else tuple(tuple(g) for g in groups)
    flat = [i for group in result for i in group]
    if (
        not result
        or any(not g for g in result)
        or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in flat)
        or sorted(flat) != list(range(dimension))
    ):
        raise ValueError("source_groups must partition every raw input coordinate exactly once.")
    return result


def check_intervention(intervention, dimension, groups=None):
    from ..interventions import IndependentEmpirical, IndependentGaussian, UniformBox

    if not isinstance(intervention, (UniformBox, IndependentGaussian, IndependentEmpirical)):
        raise TypeError(
            "Declare the intervention using UniformBox, IndependentGaussian or IndependentEmpirical."
        )
    if intervention.dimension != dimension:
        raise ValueError("intervention dimension must match the full raw source sample dimension.")
    if (
        isinstance(intervention, IndependentEmpirical)
        and groups is not None
        and tuple(groups) != intervention.source_groups
    ):
        raise ValueError("Empirical source_groups must preserve the independently sampled blocks.")


def _reweighted_samples(x, y, intervention, config):
    """Prepare a single weighted empirical joint for all requested marginals."""
    from ..interventions import IndependentGaussian, UniformBox

    if not isinstance(intervention, (UniformBox, IndependentGaussian)):
        raise TypeError("rw_ei requires a continuous UniformBox or IndependentGaussian intervention.")
    options = dict(config.options)
    if options.pop("assume_identified", False) is not True:
        raise ValueError(
            "rw_ei requires assume_identified=True: the target input law must be covered by the "
            "observational input law, and the observed conditional response must identify the "
            "interventional response."
        )
    weight = np.asarray(options.pop("sample_weight", None), dtype=float)
    if weight.shape != (len(x),) or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("rw_ei sample_weight must be one finite nonnegative density ratio per paired row.")
    original_count = int(np.count_nonzero(weight))
    if original_count < 4:
        raise ValueError("rw_ei requires at least four positive-weight rows.")
    if isinstance(intervention, UniformBox):
        low, high = np.asarray(intervention.bounds).T
        outside = np.any((x < low) | (x > high), axis=1)
        if np.any(outside & (weight > 0)):
            raise ValueError("rw_ei requires zero weight for rows outside the UniformBox intervention.")
    # Normalize before summing to avoid overflow for proportionally equivalent weights.
    weight = weight / weight.max()
    weight /= weight.sum()
    positive = weight > 0
    count = int(positive.sum())
    if count < 4:
        raise ValueError("rw_ei requires at least four positive-weight rows after rescaling.")
    weight = weight[positive]
    squared_mass = float(weight @ weight)
    if squared_mass >= 1:
        raise ValueError("rw_ei weights are numerically concentrated on one row.")
    diagnostics = {
        "weight_count": len(x),
        "effective_sample_size": 1 / squared_mass,
        "effective_sample_fraction": 1 / (squared_mass * len(x)),
        "maximum_normalized_weight": float(weight.max()),
        "zero_weight_count": len(x) - original_count,
        "underflow_weight_count": original_count - count,
        "weight_source": "caller-supplied-full-source-density-ratio",
        "identification": "assumed-by-caller",
        "weight_clipping": "none",
    }
    fit_config = EstimatorConfig("tm_polynomial", {**options, "sample_weight": weight})
    return x[positive], y[positive], fit_config, diagnostics


def sample_evaluator(
    x, y, intervention, groups, config, *, unit, train_fraction=1.0, seed=0, source_transforms=None
):
    unit_scale(unit)
    method = config.method
    reweighting = None
    if method == "rw_ei":
        if config.options.get("readout", "polynomial_tm") != "polynomial_tm":
            raise ValueError("Only the polynomial_tm RW-EI reader supports subset EI and SPT.")
        config = EstimatorConfig(method, {k: v for k, v in config.options.items() if k != "readout"})
        if train_fraction != 1.0:
            raise ValueError("rw_ei supports only same-sample fitting and evaluation (train_fraction=1).")
        x, y, config, reweighting = _reweighted_samples(x, y, intervention, config)
    elif config.options.get("sample_weight") is not None:
        raise ValueError(
            "Weighted MI changes the intervention distribution; use estimate_mi for weighted diagnostics."
        )
    if config.method == "tm_affine_readout":
        if source_transforms is not None or train_fraction != 1.0:
            raise ValueError(
                "Affine readout uses raw source coordinates and same-ensemble full-channel fitting."
            )
        from .readout import AffineReadoutEIOracle

        oracle = AffineReadoutEIOracle(x, y, intervention, groups, **config.options)
        records = {}

        def evaluator(coalition):
            value = oracle.ei(coalition) * unit_scale(unit)
            records[tuple(coalition)] = {
                "sources": list(coalition),
                "value": value,
                "unit": unit,
                "clipped": False,
            }
            return value

        return evaluator, {"protocol": oracle.metadata, "estimates": records}
    if config.method == "tm_neural":
        if source_transforms is not None or train_fraction != 1.0:
            raise ValueError("Neural TM takes context_transform and train_fraction in estimator_options.")
        from .neural import NeuralEIOracle

        options = dict(config.options)
        options.setdefault("seed", seed)
        oracle = NeuralEIOracle(x, y, intervention, groups, **options)
        records = {}

        def evaluator(coalition):
            value = oracle.ei(coalition) * unit_scale(unit)
            records[tuple(coalition)] = {
                "sources": list(coalition),
                "value": value,
                "unit": unit,
                **oracle.estimate_diagnostics[tuple(coalition)],
            }
            return value

        return evaluator, {"protocol": oracle.metadata, "estimates": records}
    if config.method not in ("tm_polynomial", "tm_affine", "tm_affine_lifted", "gaussian", "kde", "ksg"):
        raise ValueError(
            f"{config.method} cannot compute general subset EI from samples. "
            "Use estimate_ei for NIS, or analyze_linear_gaussian for analytic channels."
        )
    if not np.isfinite(train_fraction) or not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1].")
    split = len(x) if train_fraction == 1 else int(len(x) * train_fraction)
    if split < 4 or (split < len(x) and len(x) - split < 4):
        raise ValueError("Leave at least four fitting and, if held out, four evaluation samples.")
    if source_transforms is None:
        transforms = (None,) * len(groups)
    else:
        transforms = tuple(source_transforms)
        if len(transforms) != len(groups):
            raise ValueError("source_transforms must contain one callable or None per source group.")
    blocks = []
    for group, transform in zip(groups, transforms):
        block = x[:, group]
        if transform is not None:
            block = samples(transform(block), "source_transform output")
        if len(block) != len(x):
            raise ValueError("source_transforms must preserve the sample axis.")
        blocks.append(block)
    records = {}

    def evaluator(coalition):
        source = np.column_stack([blocks[g] for g in coalition])
        options = {} if split == len(x) else {"evaluation_x": source[split:], "evaluation_y": y[split:]}
        result = estimate_mi(source[:split], y[:split], estimator=config, unit=unit, **options)
        detail = {k: v for k, v in result.diagnostics.items() if k != "pointwise_mi"}
        if reweighting is not None:
            detail["options"] = {k: v for k, v in detail["options"].items() if k != "sample_weight"}
        records[tuple(coalition)] = {
            "sources": list(coalition),
            "value": result.value,
            "unit": unit,
            **detail,
        }
        return result.value

    if reweighting is not None:
        protocol = "same-reweighted-observational-ensemble"
    elif split == len(x):
        protocol = "same-intervention-ensemble"
    else:
        protocol = "held-out-intervention-samples"
    return evaluator, {
        "protocol": {
            "train_count": split,
            "evaluation_count": len(x) if split == len(x) else len(x) - split,
            "density_fit_and_evaluation": protocol,
            **({"reweighting": reweighting} if reweighting is not None else {}),
            "source_transforms": transforms,
        },
        "estimates": records,
    }


def estimate_ei(
    data,
    target=None,
    *,
    estimator="tm_polynomial",
    estimator_options=None,
    intervention=None,
    response_noise_cov=None,
    unit="nats",
) -> InformationEstimate:
    """Select EI method explicitly.

    Sample backends: data=X and target=Y from a declared intervention ensemble.
    rw_ei: paired observations with estimator_options containing sample_weight
    (the full-source q_do / p_obs density ratio, up to a constant) and
    assume_identified=True. The caller establishes overlap and identification.
    With readout="marginal_tm", use a scalar target and known response_noise_std
    in estimator_options. tm_marginal applies the same reader to unweighted
    intervention samples. Both marginal readers estimate full-input EI only.
    discrete: data=TPM. linear_gaussian: data=target-by-source coupling matrix.
    local_gaussian/nis: data=batch response callable. Analytic methods require
    response_noise_cov and their appropriate intervention distribution.
    """
    config = resolve_config(estimator, estimator_options)
    method, options = config.method, dict(config.options)
    unit_scale(unit)
    if method == "discrete":
        from ..discrete import effective_information_from_tpm

        if target is not None or intervention is not None or response_noise_cov is not None:
            raise ValueError(
                "Discrete TPM EI uses its registered uniform input repertoire; no sample/noise arguments."
            )
        value = effective_information_from_tpm(data, log_base=np.e if unit == "nats" else 2.0, **options)
        return InformationEstimate(
            value, unit, method, {"intervention": "uniform-registered-input-states", "clipped": False}, "ei"
        )
    if method == "nis":
        if target is not None:
            raise ValueError("NIS takes a batch dynamics function, not paired target samples.")
        return nis_ei(
            data, intervention=intervention, response_noise_cov=response_noise_cov, unit=unit, **options
        )
    if method in ("linear_gaussian", "local_gaussian"):
        from ..interventions import IndependentGaussian

        if not isinstance(intervention, IndependentGaussian) or target is not None:
            raise ValueError(
                "Analytic Gaussian EI requires IndependentGaussian input and response_noise_cov."
            )
        if method == "local_gaussian":
            center = np.asarray(options.pop("center", intervention.mean), dtype=float)
            if center.shape != (intervention.dimension,) or not np.isfinite(center).all():
                raise ValueError("center must be a finite vector matching the intervention dimension.")
            a = dynamics_jacobian(
                data,
                center[None, :],
                jacobian=options.pop("jacobian", None),
                finite_difference_step=options.pop("finite_difference_step", 1e-5),
            )[0]
        else:
            a = data
        result = linear_gaussian_ei(
            a, response_noise_cov, np.asarray(intervention.std) ** 2, unit=unit, **options
        )
        result = replace(
            result,
            diagnostics={
                **result.diagnostics,
                **intervention.metadata(),
                "response_noise_cov": np.asarray(response_noise_cov, dtype=float),
            },
        )
        if method == "local_gaussian":
            return replace(
                result,
                estimator=method,
                diagnostics={
                    **result.diagnostics,
                    "center": center,
                    "approximation": "Jacobian at center; Gaussian perturbation of the local linear channel",
                },
            )
        return result
    if response_noise_cov is not None:
        raise ValueError(
            "Sample EI receives already generated targets; add physical response noise before estimation."
        )
    if method != "rw_ei" and options.get("sample_weight") is not None:
        raise ValueError("Weighted MI needs a separately verified intervention; use estimate_mi.")
    x, y = samples(data), samples(target)
    check_intervention(intervention, x.shape[1])
    if len(x) != len(y):
        raise ValueError("Source and target sample counts must match.")
    reader = options.pop("readout", "polynomial_tm") if method == "rw_ei" else None
    if method == "rw_ei" and reader not in ("polynomial_tm", "marginal_tm"):
        raise ValueError("rw_ei readout must be 'polynomial_tm' or 'marginal_tm'.")
    if method == "tm_marginal" or reader == "marginal_tm":
        meta = intervention.metadata()
        if method == "rw_ei":
            _, y, fit, reweighting = _reweighted_samples(x, y, intervention, EstimatorConfig(method, options))
            options = dict(fit.options)
            meta["reweighting"] = reweighting
        result = marginal_gaussian_ei(y, unit=unit, **options)
        return replace(result, estimator=method, diagnostics={**result.diagnostics, **meta})
    if method in ("tm_neural", "tm_affine_readout", "rw_ei"):
        groups = source_groups(x.shape[1], getattr(intervention, "source_groups", None))
        evaluator, detail = sample_evaluator(x, y, intervention, groups, config, unit=unit)
        value = evaluator(tuple(range(len(groups))))
        return InformationEstimate(
            value, unit, method, {**json_safe(detail), **intervention.metadata()}, "ei"
        )
    result = estimate_mi(x, y, estimator=config, unit=unit)
    return replace(result, quantity="ei", diagnostics={**result.diagnostics, **intervention.metadata()})
