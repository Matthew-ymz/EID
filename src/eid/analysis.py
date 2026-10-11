"""Small public adapters around the copied EI, TM, and SPT implementations."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from .discrete import _validate_system_tpm, discrete_effective_information
from .estimators.analytic import dynamics_jacobian, linear_gaussian_ei
from .estimators.api import check_intervention, sample_evaluator
from .estimators.api import source_groups as _source_groups
from .estimators.core import EstimatorConfig, covariance, json_safe, resolve_config, samples
from .interventions import IndependentEmpirical, IndependentGaussian, UniformBox
from .spt import (
    SPTAudit,
    SPTConfig,
    SPTNonnegativityError,
    SPTResult,
    build_spt,
    exact_candidate_selector,
    spectral_candidate_selector,
)


def _unit_factor(unit: str) -> float:
    if unit not in ("nats", "bits"):
        raise ValueError("unit must be 'nats' or 'bits'.")
    return math.log(2.0) if unit == "nats" else 1.0


def _analysis_options(unit, syn_tolerance, exact_max_size):
    _unit_factor(unit)
    if not math.isfinite(syn_tolerance) or syn_tolerance < 0:
        raise ValueError("syn_tolerance must be finite and nonnegative in the declared unit.")
    if (
        isinstance(exact_max_size, bool)
        or not isinstance(exact_max_size, (int, np.integer))
        or exact_max_size < 2
    ):
        raise ValueError("exact_max_size must be an integer >= 2.")


def _indices(values: Sequence[int] | None, dimension: int, name: str) -> tuple[int, ...]:
    if values is None:
        return tuple(range(dimension))
    raw = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in raw):
        raise ValueError(f"{name} must contain integer indices.")
    indices = tuple(sorted(map(int, raw)))
    if not indices or len(set(indices)) != len(indices):
        raise ValueError(f"{name} must be nonempty and unique.")
    if indices[0] < 0 or indices[-1] >= dimension:
        raise ValueError(f"{name} contain out-of-range indices.")
    return indices


@dataclass
class AnalysisResult:
    """Information values and tree in one explicitly declared information unit.

    ``ei_table`` contains only queried coalitions, not necessarily all subsets.
    Raw tolerance-scale negative estimates are retained for auditing and closure.
    """

    unit: str
    sources: tuple[int, ...]
    targets: tuple[int, ...]
    ei: float
    xi: float
    singleton_ei: dict[int, float]
    ei_table: dict[tuple[int, ...], float]
    tree: SPTResult
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible results without ambiguous tuple object keys."""
        return {
            "unit": self.unit,
            "sources": list(self.sources),
            "targets": list(self.targets),
            "ei": self.ei,
            "xi": self.xi,
            "singleton_ei": [{"source": key, "ei": value} for key, value in self.singleton_ei.items()],
            "ei_table": [{"sources": list(key), "ei": value} for key, value in self.ei_table.items()],
            "tree": {
                "root": asdict(self.tree.root),
                "audit": self.diagnostics["spt_audit"],
                "closure_error": self.tree.closure_error,
            },
            "diagnostics": json_safe(self.diagnostics),
            "metadata": json_safe(self.metadata),
        }


@dataclass
class _EIOracle:
    evaluator: Callable[[tuple[int, ...]], float]
    sources: tuple[int, ...]
    tolerance: float
    cache: dict[tuple[int, ...], float] = field(default_factory=dict)
    xi_cache: dict[tuple[int, ...], float] = field(default_factory=dict)

    def ei(self, sources: Iterable[int]) -> float:
        key = tuple(sorted(sources))
        if key not in self.cache:
            value = float(self.evaluator(key))
            if not math.isfinite(value):
                raise ValueError(f"Non-finite EI for coalition {key}.")
            if value < -self.tolerance:
                raise ValueError(
                    f"EI nonnegativity violation for {key}: minimum={value:.12g}, "
                    f"threshold={-self.tolerance:.12g}, affected_count=1."
                )
            self.cache[key] = value
        return self.cache[key]

    def xi(self, sources: Iterable[int]) -> float:
        key = tuple(sorted(sources))
        if len(key) <= 1:
            return 0.0
        if key not in self.xi_cache:
            value = self.ei(key) - sum(self.ei((source,)) for source in key)
            if value < -self.tolerance:
                raise SPTNonnegativityError(
                    f"Xi nonnegativity violation for {key}: minimum={value:.12g}, "
                    f"threshold={-self.tolerance:.12g}, affected_count=1."
                )
            self.xi_cache[key] = value
        return self.xi_cache[key]


def _analyze(
    evaluator: Callable[[tuple[int, ...]], float],
    *,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
    unit: str,
    syn_tolerance: float,
    exact_max_size: int,
    metadata: dict[str, Any],
) -> AnalysisResult:
    _analysis_options(unit, syn_tolerance, exact_max_size)
    oracle = _EIOracle(evaluator, sources, syn_tolerance)
    singles = {source: oracle.ei((source,)) for source in sources}
    overall = oracle.ei(sources)
    xi = oracle.xi(sources)
    audit = SPTAudit()
    selector = exact_candidate_selector
    affinity_tolerance_zero_count = 0
    if len(sources) > exact_max_size:
        # SPT's spectral adapter uses consecutive local indices. This mapping
        # also allows a user to analyze a subset of the full source coordinates.
        affinity = np.zeros((len(sources), len(sources)))
        for left, right in combinations(range(len(sources)), 2):
            value = oracle.xi((sources[left], sources[right]))
            # Only tolerance-scale negatives reach here: the oracle fails for
            # significant violations. Retain raw Xi, but use numerical zero in
            # the graph weights so a spectral Laplacian has nonnegative affinity.
            if value < 0:
                affinity_tolerance_zero_count += 1
                value = 0.0
            affinity[left, right] = affinity[right, left] = value
        spectral = spectral_candidate_selector(affinity, exact_max_size=exact_max_size)
        positions = {source: index for index, source in enumerate(sources)}

        def selector(coalition):
            kind, splits = spectral(tuple(positions[source] for source in coalition))
            return kind, [
                (tuple(sources[index] for index in left), tuple(sources[index] for index in right))
                for left, right in splits
            ]

    tree = build_spt(
        sources,
        oracle,
        config=SPTConfig(syn_tolerance=syn_tolerance),
        candidate_selector=selector,
        audit=audit,
    )
    audit_values = asdict(audit)
    if not math.isfinite(audit.minimum_candidate_syn):
        audit_values["minimum_candidate_syn"] = None
    diagnostics = {
        "syn_tolerance": syn_tolerance,
        "tolerance_unit": unit,
        "xi_tolerance_negative_count": sum(value < 0 for value in oracle.xi_cache.values()),
        "minimum_queried_xi": min(oracle.xi_cache.values(), default=0.0),
        "ei_negative_count": sum(value < 0 for value in oracle.cache.values()),
        "affinity_tolerance_zero_count": affinity_tolerance_zero_count,
        "spt_audit": audit_values,
        "ei_query_count": len(oracle.cache),
        "ei_table_complete": len(oracle.cache) == (1 << len(sources)) - 1,
        "search": "exact-at-each-node" if len(sources) <= exact_max_size else "spectral-candidates",
        "exact_max_size": int(exact_max_size),
        "closure_error": tree.closure_error,
    }
    return AnalysisResult(
        unit, sources, targets, overall, xi, singles, dict(oracle.cache), tree, diagnostics, metadata
    )


def analyze_tpm(
    tpm: np.ndarray,
    *,
    n_nodes: int,
    syn_tolerance: float,
    sources: Sequence[int] | None = None,
    targets: Sequence[int] | None = None,
    horizon: int = 1,
    unit: str = "nats",
    exact_max_size: int = 8,
    estimator: str | EstimatorConfig = "discrete",
) -> AnalysisResult:
    """Analyze a full binary-state TPM under independent uniform interventions.

    Rows and columns use lexicographically ordered binary states. The l-step
    kernel is formed before marginalization, so every EI shares one protocol.
    """
    _analysis_options(unit, syn_tolerance, exact_max_size)
    config = resolve_config(estimator)
    if config.method != "discrete" or config.options:
        raise ValueError("analyze_tpm requires estimator='discrete' without estimator options.")
    if not isinstance(n_nodes, (int, np.integer)) or n_nodes < 1:
        raise ValueError("n_nodes must be an integer >= 1.")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be an integer >= 1.")
    matrix = _validate_system_tpm(tpm, n_nodes)
    kernel = np.linalg.matrix_power(matrix, horizon)
    source_indices = _indices(sources, n_nodes, "sources")
    target_indices = _indices(targets, n_nodes, "targets")

    def evaluator(coalition):
        return discrete_effective_information(
            kernel,
            n_nodes=n_nodes,
            source_indices=coalition,
            target_indices=target_indices,
            log_base=math.e if unit == "nats" else 2.0,
        )

    return _analyze(
        evaluator,
        sources=source_indices,
        targets=target_indices,
        unit=unit,
        syn_tolerance=syn_tolerance,
        exact_max_size=exact_max_size,
        metadata={
            "system_kind": "binary-tpm",
            "horizon": int(horizon),
            "intervention": "independent-uniform-binary",
            "estimator": "exact-discrete",
            "estimator_method": "discrete",
            "source_dimension": int(n_nodes),
        },
    )


def analyze_dynamics(
    dynamics: Callable[[np.ndarray], np.ndarray],
    *,
    intervention: UniformBox | IndependentGaussian | IndependentEmpirical,
    response_noise_std: float | Sequence[float] | None = None,
    response_noise_cov: np.ndarray | None = None,
    syn_tolerance: float,
    sources: Sequence[int] | None = None,
    targets: Sequence[int] | None = None,
    sample_count: int = 4096,
    seed: int = 0,
    degree: int = 3,
    joint_order: str = "source_first",
    unit: str = "nats",
    exact_max_size: int = 8,
    estimator: str | EstimatorConfig = "tm_polynomial",
    estimator_options: dict[str, Any] | None = None,
    train_fraction: float = 1.0,
    source_groups: Sequence[Sequence[int]] | None = None,
    source_transforms: Sequence[Callable | None] | None = None,
) -> AnalysisResult:
    """Analyze a batch response function with an explicitly selected EI backend.

    The function supplies the complete requested time-horizon response. This
    adapter does not integrate an ODE or iterate a one-step map automatically.
    Independent or correlated additive Gaussian response noise is explicit.
    All subsets use marginals of the same full-source intervention ensemble.
    """
    _analysis_options(unit, syn_tolerance, exact_max_size)
    config = resolve_config(estimator, estimator_options)
    check_intervention(intervention, getattr(intervention, "dimension", 0))
    if config.method == "nis":
        raise ValueError(
            "NIS cannot provide general subset EI/SPT. Use estimate_ei(dynamics, estimator='nis')."
        )
    if config.method == "local_gaussian":
        if source_transforms is not None or train_fraction != 1:
            raise ValueError("Local Gaussian analysis operates on the raw local linear channel.")
        if not isinstance(intervention, IndependentGaussian):
            raise ValueError("local_gaussian requires IndependentGaussian perturbations.")
        options = dict(config.options)
        center = np.asarray(options.pop("center", intervention.mean), dtype=float)
        if center.shape != (intervention.dimension,) or not np.isfinite(center).all():
            raise ValueError("center must be finite and match intervention dimension.")
        jacobian = dynamics_jacobian(dynamics, center[None], **options)[0]
        noise = _response_covariance(jacobian.shape[0], response_noise_std, response_noise_cov)
        result = analyze_linear_gaussian(
            jacobian,
            response_noise_cov=noise,
            intervention=intervention,
            sources=sources,
            targets=targets,
            source_groups=source_groups,
            syn_tolerance=syn_tolerance,
            unit=unit,
            exact_max_size=exact_max_size,
        )
        result.metadata.update(
            estimator="local_gaussian",
            estimator_method="local_gaussian",
            center=center.tolist(),
            approximation="Jacobian at center, local Gaussian channel",
        )
        return result
    rng = np.random.default_rng(seed)
    x = intervention.sample(sample_count, rng)
    y = np.asarray(dynamics(x), dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != sample_count or y.shape[1] < 1 or not np.isfinite(y).all():
        raise ValueError("dynamics must return finite responses with shape [sample_count, output_dimension].")
    noise = _response_covariance(y.shape[1], response_noise_std, response_noise_cov)
    if response_noise_cov is None:
        y = y + rng.normal(size=y.shape) * np.sqrt(np.diag(noise))
    else:
        y = y + rng.multivariate_normal(np.zeros(y.shape[1]), noise, size=sample_count)
    # Backward-compatible polynomial settings; explicit backend options take precedence.
    if config.method == "tm_polynomial":
        config = EstimatorConfig(
            config.method, {"degree": degree, "joint_order": joint_order, **config.options}
        )
    result = analyze_samples(
        x,
        y,
        intervention=intervention,
        sources=sources,
        targets=targets,
        estimator=config,
        source_groups=source_groups,
        source_transforms=source_transforms,
        train_fraction=train_fraction,
        seed=seed,
        syn_tolerance=syn_tolerance,
        unit=unit,
        exact_max_size=exact_max_size,
    )
    result.metadata.update(
        system_kind="batch-response",
        seed=int(seed),
        response_noise="independent-additive-gaussian"
        if np.array_equal(noise, np.diag(np.diag(noise)))
        else "correlated-additive-gaussian",
        response_noise_cov=noise.tolist(),
        time_horizon="defined-by-response-function",
    )
    if response_noise_std is not None:
        result.metadata["response_noise_std"] = np.sqrt(np.diag(noise)).tolist()
    return result


def _response_covariance(dimension, response_noise_std, response_noise_cov):
    if (response_noise_std is None) == (response_noise_cov is None):
        raise ValueError("Specify exactly one of response_noise_std or response_noise_cov.")
    if response_noise_cov is not None:
        return covariance(response_noise_cov, dimension, "response_noise_cov")
    noise = np.asarray(response_noise_std, dtype=float)
    if noise.ndim > 1 or (noise.ndim == 1 and noise.shape != (dimension,)):
        raise ValueError("response_noise_std must be scalar or contain one value per output coordinate.")
    if not np.isfinite(noise).all() or np.any(noise < 0):
        raise ValueError("response_noise_std must be finite and nonnegative.")
    if np.any(noise == 0):
        raise ValueError(
            "This continuous adapter requires positive response noise. Use an explicit noisy "
            "measurement channel; a deterministic continuous response can have singular density."
        )
    return covariance(np.diag(np.broadcast_to(noise, (dimension,)) ** 2), dimension, "response_noise_cov")


def analyze_samples(
    x,
    y,
    *,
    intervention,
    syn_tolerance: float,
    estimator: str | EstimatorConfig = "tm_polynomial",
    estimator_options=None,
    sources=None,
    targets=None,
    source_groups=None,
    source_transforms=None,
    train_fraction=1.0,
    seed=0,
    unit="nats",
    exact_max_size=8,
) -> AnalysisResult:
    """Analyze a complete paired intervention ensemble (including simulator process noise).

    The caller declares how inputs were intervened on. Observational samples do
    not become causal merely by supplying a distribution object. source_groups
    partition raw input coordinates into independent players; source_transforms
    act on each player separately. Targets, samples and fitting split stay fixed.
    """
    _analysis_options(unit, syn_tolerance, exact_max_size)
    config = resolve_config(estimator, estimator_options)
    x, y = samples(x, "x"), samples(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must have matching sample counts.")
    if source_groups is None:
        source_groups = getattr(intervention, "source_groups", None)
    groups = _source_groups(x.shape[1], source_groups)
    check_intervention(intervention, x.shape[1], groups)
    source_indices = _indices(sources, len(groups), "sources")
    target_indices = _indices(targets, y.shape[1], "targets")
    evaluator, detail = sample_evaluator(
        x,
        y[:, target_indices],
        intervention,
        groups,
        config,
        unit=unit,
        train_fraction=train_fraction,
        seed=seed,
        source_transforms=source_transforms,
    )
    metadata = {
        "system_kind": "intervention-samples",
        **intervention.metadata(),
        "source_dimension": x.shape[1],
        "target_dimension": y.shape[1],
        "sample_count": len(x),
        "source_groups": [list(g) for g in groups],
        "estimator_method": config.method,
        "estimator_options": json_safe(config.options),
        "estimator": config.method,
        **json_safe(detail["protocol"]),
    }
    if config.method == "tm_polynomial":
        degree = config.options.get("degree", 3)
        metadata.update(
            estimator=f"polynomial_triangular_transport_map_degree_{degree}",
            joint_order=config.options.get("joint_order", "source_first"),
            bias_correction="none",
            residual_min_scale=config.options.get("min_scale", 1e-8),
            regression_ridge=config.options.get("ridge", 1e-6),
        )
    result = _analyze(
        evaluator,
        sources=source_indices,
        targets=target_indices,
        unit=unit,
        syn_tolerance=syn_tolerance,
        exact_max_size=exact_max_size,
        metadata=metadata,
    )
    result.diagnostics["estimates"] = json_safe(list(detail["estimates"].values()))
    return result


def analyze_linear_gaussian(
    coupling,
    *,
    intervention: IndependentGaussian,
    response_noise_cov,
    syn_tolerance,
    sources=None,
    targets=None,
    source_groups=None,
    unit="nats",
    exact_max_size=8,
) -> AnalysisResult:
    """Exact rectangular linear Gaussian channel, including omitted-source marginalization."""
    _analysis_options(unit, syn_tolerance, exact_max_size)
    if not isinstance(intervention, IndependentGaussian):
        raise TypeError("Exact linear Gaussian EI requires IndependentGaussian input.")
    a = np.asarray(coupling, dtype=float)
    if a.ndim != 2 or a.shape[1] != intervention.dimension:
        raise ValueError("coupling columns must match intervention dimension.")
    noise = covariance(response_noise_cov, a.shape[0], "response_noise_cov")
    groups = _source_groups(a.shape[1], source_groups)
    si, ti = _indices(sources, len(groups), "sources"), _indices(targets, a.shape[0], "targets")
    records = []

    def evaluator(coalition):
        columns = [i for g in coalition for i in groups[g]]
        result = linear_gaussian_ei(
            a, noise, np.asarray(intervention.std) ** 2, sources=columns, targets=ti, unit=unit
        )
        records.append({"sources": list(coalition), **result.to_dict()})
        return result.value

    result = _analyze(
        evaluator,
        sources=si,
        targets=ti,
        unit=unit,
        syn_tolerance=syn_tolerance,
        exact_max_size=exact_max_size,
        metadata={
            "system_kind": "linear-gaussian-channel",
            **intervention.metadata(),
            "estimator": "linear_gaussian",
            "estimator_method": "linear_gaussian",
            "source_groups": [list(g) for g in groups],
            "coupling": a.tolist(),
            "response_noise_cov": noise.tolist(),
            "source_dimension": a.shape[1],
            "target_dimension": a.shape[0],
            "time_horizon": "encoded-by-coupling",
            "approximation": "none",
        },
    )
    result.diagnostics["estimates"] = records
    return result


def analyze_model(model: Any, **kwargs: Any) -> AnalysisResult:
    """Analyze an already trained model exposing a batch ``predict(X)`` method.

    Training, hold-out validation and intervention-support validation belong to
    the caller. The returned information describes interventions on this model.
    """
    predictor = getattr(model, "predict", None)
    if not callable(predictor):
        raise TypeError("model must expose a callable predict(X) method.")
    result = analyze_dynamics(predictor, **kwargs)
    result.metadata["system_kind"] = "learned-model"
    result.metadata["model_class"] = type(model).__name__
    result.metadata["interpretation"] = "interventions-on-the-fitted-model"
    return result
