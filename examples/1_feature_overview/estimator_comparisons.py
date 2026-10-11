"""Small paired estimator comparisons for the feature overview, outside eid.

References integrate the declared channel, rather than fit a competing density.
No estimator is tuned on these results. Optional neural fits use one fixed budget.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import ndtr, roots_genlaguerre, roots_legendre, xlogy

import eid
from eid import IndependentGaussian, UniformBox, analyze_samples, estimate_ei, estimate_mi
from eid.estimators.core import json_safe

BASE_METHODS = (
    "tm_polynomial",
    "tm_affine",
    "tm_affine_lifted",
    "tm_affine_readout",
    "gaussian",
    "kde",
    "kde_heldout",
    "ksg",
)
LABELS = {
    "tm_polynomial": "Polynomial TM (deg. 3)",
    "tm_affine": "Affine TM",
    "tm_affine_lifted": "Affine TM + S12 lift",
    "tm_affine_readout": "Shared affine readout",
    "gaussian": "Gaussian logdet",
    "kde": "KDE (same samples)",
    "kde_heldout": "KDE (70/30 split)",
    "ksg": "KSG (k=5)",
    "tm_neural": "Neural TM (raw context)",
    "neural_fourier": "Neural TM + Fourier",
    "linear_gaussian": "Analytic Gaussian (known channel)",
}


@dataclass(frozen=True)
class ComparisonConfig:
    seeds: tuple[int, ...] = (7, 19, 31)
    sample_count: int = 1000
    sample_sizes: tuple[int, ...] = (128, 384, 1024)
    dimensions: tuple[int, ...] = (1, 4, 8)
    noise_scales: tuple[float, ...] = (0.05, 0.15, 0.5, 1.5, 4.5)
    input_scales: tuple[float, ...] = (0.1, 0.3, 0.7, 1.3)
    syn_tolerance: float = 0.03
    include_neural: bool = True
    neural_epochs: int = 80


def fourier_context(x):
    """An explicit structural prior for this toy channel, not learned discovery."""
    return np.column_stack([x, np.sin(3 * x[:, 0]), np.cos(3 * x[:, 0])])


def noise_entropy(std):
    return 0.5 * np.log(2 * np.pi * np.e * std**2)


@lru_cache(maxsize=8)
def _legendre_rule(order):
    return roots_legendre(order)


def _entropy(latent, weights, noise_std, bounds, order):
    """Gaussian-mixture output entropy by deterministic integration."""
    nodes, qweights = _legendre_rule(order)
    lo, hi = bounds
    y = (nodes + 1) * (hi - lo) / 2 + lo
    density = (
        np.exp(-0.5 * ((y[:, None] - latent[None]) / noise_std) ** 2)
        @ weights
        / (np.sqrt(2 * np.pi) * noise_std)
    )
    scale = (hi - lo) / 2
    mass = float(scale * (qweights @ density))
    if abs(mass - 1) > 2e-7:
        raise ValueError(f"Reference output density is not normalized: integral={mass}.")
    return float(-scale * (qweights @ xlogy(density, density)))


def uniform_noise_entropy(half_width, noise_std, order=256):
    """Exact uniform/Gaussian convolution, with stable tail differences."""
    nodes, weights = _legendre_rule(order)
    bound = half_width + 9 * noise_std
    y = nodes * bound
    low, high = (y - half_width) / noise_std, (y + half_width) / noise_std
    probability = np.where(low > 0, ndtr(-low) - ndtr(-high), ndtr(high) - ndtr(low))
    density = probability / (2 * half_width)
    if abs(bound * (weights @ density) - 1) > 2e-7:
        raise ValueError("Uniform/noise reference quadrature needs more points.")
    return float(-bound * (weights @ xlogy(density, density)))


def _product_reference(noise_std, order):
    # |X1 X2|=exp(-T), T~Gamma(2,1); integrate t*exp(-t) with Laguerre.
    nodes, weights = roots_genlaguerre(order, 1)
    latent = np.r_[np.exp(-nodes), -np.exp(-nodes)]
    weights = np.r_[weights, weights] / 2
    hy = _entropy(latent, weights, noise_std, (-1 - 9 * noise_std, 1 + 9 * noise_std), 512)
    nodes_a, weights_a = _legendre_rule(order)
    amplitudes = (nodes_a + 1) / 2
    conditional_entropy = sum(
        w * uniform_noise_entropy(a, noise_std) / 2 for a, w in zip(amplitudes, weights_a)
    )
    joint, single = hy - noise_entropy(noise_std), hy - conditional_entropy
    return {"ei": joint, "singleton_ei": single, "syn": joint - 2 * single}


def product_reference(noise_std=0.3):
    fine, coarse = _product_reference(noise_std, 128), _product_reference(noise_std, 64)
    error = max(abs(fine[k] - coarse[k]) for k in fine)
    if error > 5e-5 or fine["syn"] < 0:
        raise ValueError(f"Product reference convergence/nonnegativity check failed: {error}.")
    return {**fine, "refinement_difference": error, "kind": "channel quadrature"}


def periodic_reference(noise_std=0.3):
    values = []
    for order in (128, 256):
        latent = np.cos((np.arange(order) + 0.5) * np.pi / order)
        values.append(
            _entropy(
                latent,
                np.full(order, 1 / order),
                noise_std,
                (-1 - 9 * noise_std, 1 + 9 * noise_std),
                512,
            )
            - noise_entropy(noise_std)
        )
    if abs(values[1] - values[0]) > 5e-5:
        raise ValueError("Periodic reference quadrature did not converge.")
    return {
        "ei": values[1],
        "refinement_difference": abs(values[1] - values[0]),
        "kind": "channel quadrature",
    }


def gaussian_periodic_reference(input_std, noise_std=0.3):
    values = []
    phase_std = 3 * input_std
    image_count = int(np.ceil(9 * phase_std / (2 * np.pi))) + 1
    images = 2 * np.pi * np.arange(-image_count, image_count + 1)
    for order in (128, 256):
        nodes, weights = _legendre_rule(order)
        phase = np.pi * nodes
        density = np.exp(-0.5 * ((phase[:, None] + images) / phase_std) ** 2).sum(axis=1)
        density /= np.sqrt(2 * np.pi) * phase_std
        values.append(
            _entropy(
                np.sin(phase),
                weights * np.pi * density,
                noise_std,
                (-1 - 9 * noise_std, 1 + 9 * noise_std),
                512,
            )
            - noise_entropy(noise_std)
        )
    if abs(values[1] - values[0]) > 5e-5:
        raise ValueError("Gaussian periodic reference quadrature did not converge.")
    return values[1], abs(values[1] - values[0])


def _fingerprint(x, y):
    return hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()


def _options(method, config, seed):
    if method in ("tm_neural", "neural_fourier"):
        return "tm_neural", {
            "epochs": config.neural_epochs,
            "hidden_features": 24,
            "layers": 2,
            "evaluation_count": 256,
            "marginal_samples": 128,
            "seed": seed,
            **({"context_transform": fourier_context} if method == "neural_fourier" else {}),
        }
    return ("kde" if method == "kde_heldout" else method), {}


def _estimate(x, y, intervention, method, config, seed, *, decomposition=False):
    backend, options = _options(method, config, seed)
    fingerprint = _fingerprint(x, y)
    start = time.perf_counter()
    if decomposition:
        result = analyze_samples(
            x,
            y,
            intervention=intervention,
            estimator=backend,
            estimator_options=options,
            train_fraction=0.7 if method == "kde_heldout" else 1.0,
            syn_tolerance=config.syn_tolerance,
        )
        record = {
            "ei": result.ei,
            "syn": result.xi,
            "singleton_ei": list(result.singleton_ei.values()),
            "diagnostics": result.diagnostics,
            "metadata": result.metadata,
        }
    else:
        if method == "linear_gaussian":
            result = estimate_ei(
                [[1.0]],
                estimator=method,
                intervention=intervention,
                response_noise_cov=[[1.0]],
            )
        elif method == "kde_heldout":
            split = int(0.7 * len(x))
            result = estimate_mi(
                x[:split],
                y[:split],
                estimator=backend,
                evaluation_x=x[split:],
                evaluation_y=y[split:],
            )
        else:
            result = estimate_ei(
                x, y, intervention=intervention, estimator=backend, estimator_options=options
            )
        record = {"ei": result.value, "diagnostics": result.diagnostics}
    if fingerprint != _fingerprint(x, y):
        raise AssertionError("An estimator changed the paired intervention ensemble.")
    return {"method": method, "seed": seed, "seconds": time.perf_counter() - start, **record}


def _compact(value):
    if isinstance(value, dict):
        return {str(k): _compact(v) for k, v in value.items() if k not in ("pointwise_mi", "pointwise_ei")}
    if isinstance(value, (list, tuple)):
        return [_compact(v) for v in value]
    return json_safe(value)


def _case_data(name, count, seed):
    rng = np.random.default_rng(seed)
    if name == "gaussian":
        prior = IndependentGaussian([0], [1])
        x = prior.sample(count, rng)
        y, coordinate = x + rng.normal(size=x.shape), x[:, 0]
    elif name == "product":
        prior = UniformBox([[-1, 1], [-1, 1]])
        x = prior.sample(count, rng)
        coordinate = x[:, 0] * x[:, 1]
        y = coordinate[:, None] + 0.3 * rng.normal(size=(count, 1))
    else:
        prior = UniformBox([[-np.pi, np.pi]])
        x = prior.sample(count, rng)
        coordinate = x[:, 0]
        y = np.sin(3 * x) + 0.3 * rng.normal(size=x.shape)
    return x, y, prior, coordinate


def _cases(config, neural):
    references = {
        "gaussian": {"ei": 0.5 * np.log(2), "kind": "analytic Gaussian channel"},
        "product": product_reference(),
        "periodic": periodic_reference(),
    }
    cases = {}
    for index, name in enumerate(references):
        methods = list(BASE_METHODS) + (["tm_neural"] if neural else [])
        if name == "gaussian":
            methods.append("linear_gaussian")
        if name == "periodic" and neural:
            methods.append("neural_fourier")
        records = []
        for seed in config.seeds:
            x, y, prior, coordinate = _case_data(name, config.sample_count, seed + 1000 * index)
            if seed == config.seeds[0]:
                scatter = {"x": x, "coordinate": coordinate, "y": y[:, 0]}
            for method in np.random.default_rng(seed + 9000).permutation(methods):
                records.append(_estimate(x, y, prior, method, config, seed, decomposition=name == "product"))
        cases[name] = {
            "reference": references[name],
            "methods": methods,
            "records": records,
            "scatter": scatter,
        }
    return cases


def _sensitivity(config):
    methods = ("tm_polynomial", "tm_affine", "kde", "kde_heldout", "ksg")
    sizes, dimensions = [], []
    for seed in config.seeds:
        rng = np.random.default_rng(seed + 10000)
        prior = IndependentGaussian([0], [1])
        x = prior.sample(max(config.sample_sizes), rng)
        y = x + rng.normal(size=x.shape)
        for count in config.sample_sizes:
            for method in methods:
                sizes.append({"n": count, **_estimate(x[:count], y[:count], prior, method, config, seed)})
        # Nested columns: common input/noise for all dimensions and estimators.
        dimension_prior = IndependentGaussian([0] * max(config.dimensions), [1] * max(config.dimensions))
        full_x = dimension_prior.sample(config.sample_count, rng)
        full_y = full_x + rng.normal(size=full_x.shape)
        for d in config.dimensions:
            prior = IndependentGaussian([0] * d, [1] * d)
            for method in ("tm_affine", "kde", "kde_heldout", "ksg"):
                dimensions.append(
                    {
                        "dimension": d,
                        **_estimate(full_x[:, :d], full_y[:, :d], prior, method, config, seed),
                    }
                )
    return {"sizes": sizes, "dimensions": dimensions, "reference_per_coordinate": 0.5 * np.log(2)}


def _approximations(config):
    noise, local = [], []
    uniform = UniformBox([[-1, 1]])
    noise_refs = []
    for std in config.noise_scales:
        fine = uniform_noise_entropy(2, std, 1024) - noise_entropy(std)
        coarse = uniform_noise_entropy(2, std, 512) - noise_entropy(std)
        if abs(fine - coarse) > 5e-5:
            raise ValueError("Bounded-channel reference needs finer integration.")
        noise_refs.append({"scale": std, "ei": fine, "refinement_difference": abs(fine - coarse)})
    local_refs = []
    for std in config.input_scales:
        value, error = gaussian_periodic_reference(std)
        local_refs.append({"scale": std, "ei": value, "refinement_difference": error})
    for seed in config.seeds:
        rng = np.random.default_rng(seed + 20000)
        x = uniform.sample(config.sample_count, rng)
        standard_noise = rng.normal(size=x.shape)
        for std in config.noise_scales:
            y = 2 * x + std * standard_noise
            for method in ("tm_polynomial", "ksg"):
                noise.append({"scale": std, **_estimate(x, y, uniform, method, config, seed)})
            result = estimate_ei(
                lambda a: 2 * a,
                estimator="nis",
                intervention=uniform,
                response_noise_cov=[[std**2]],
                estimator_options={
                    "assume_injective": True,
                    "sample_count": config.sample_count,
                    "seed": seed,
                },
            )
            noise.append({"scale": std, "method": "nis", "seed": seed, "ei": result.value})
        standardized_x = rng.normal(size=(config.sample_count, 1))
        standard_noise = rng.normal(size=standardized_x.shape)
        for std in config.input_scales:
            prior = IndependentGaussian([0], [std])
            x = std * standardized_x
            y = np.sin(3 * x) + 0.3 * standard_noise
            for method in ("tm_polynomial", "ksg"):
                local.append({"scale": std, **_estimate(x, y, prior, method, config, seed)})
            result = estimate_ei(
                lambda a: np.sin(3 * a),
                estimator="local_gaussian",
                intervention=prior,
                response_noise_cov=[[0.3**2]],
            )
            local.append({"scale": std, "method": "local_gaussian", "seed": seed, "ei": result.value})
    return {"noise": noise, "noise_references": noise_refs, "local": local, "local_references": local_refs}


def _discrete():
    flips = np.linspace(0, 0.5, 11)
    rows = []
    for flip in flips:
        parity = np.array([0, 1, 1, 0])
        tpm = np.column_stack([np.where(parity == 0, 1 - flip, flip), np.where(parity == 1, 1 - flip, flip)])
        value = estimate_ei(tpm, estimator="discrete").value
        single_tables = (
            np.stack([tpm[:2].mean(axis=0), tpm[2:].mean(axis=0)]),
            np.stack([tpm[[0, 2]].mean(axis=0), tpm[[1, 3]].mean(axis=0)]),
        )
        singles = [estimate_ei(a, estimator="discrete").value for a in single_tables]
        syn = value - sum(singles)
        tolerance = 1e-12
        if syn < -tolerance:
            raise ValueError(
                f"Discrete Syn violation: minimum={syn}, threshold={-tolerance}, affected_count=1."
            )
        binary_entropy = -xlogy(flip, flip) - xlogy(1 - flip, 1 - flip)
        reference = np.log(2) - binary_entropy
        if abs(value - reference) > 1e-12:
            raise AssertionError("Discrete XOR differs from its analytic reference.")
        rows.append(
            {
                "flip_probability": flip,
                "ei": value,
                "syn": syn,
                "reference": reference,
                "syn_tolerance": tolerance,
                "tolerance_negative_count": int(syn < 0),
            }
        )
    return rows


def _identity(config, neural):
    versions = {k: importlib.metadata.version(k) for k in ("numpy", "scipy")}
    if neural:
        versions.update({k: importlib.metadata.version(k) for k in ("torch", "nflows")})
    package_dir = Path(eid.__file__).parent
    code_hash = hashlib.sha256(Path(__file__).read_bytes())
    for p in sorted(package_dir.rglob("*.py")):
        code_hash.update(p.read_bytes())
    return {
        "config": asdict(config),
        "neural_available": neural,
        "eid_version": eid.__version__,
        "dependency_versions": versions,
        "code_sha256": code_hash.hexdigest(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def run_comparisons(config=None, *, cache_path=None, refresh=False):
    """One fixed tutorial comparison; reuse valid cached training results."""
    config = ComparisonConfig() if config is None else config
    if len(config.seeds) < 2 or config.sample_count < 100:
        raise ValueError("Use at least two paired seeds and 100 intervention samples.")
    if config.syn_tolerance < 0 or not np.isfinite(config.syn_tolerance):
        raise ValueError("Declare a finite nonnegative Syn tolerance in nats.")
    neural = config.include_neural and all(importlib.util.find_spec(k) for k in ("torch", "nflows"))
    neural = bool(neural)
    identity = _identity(config, neural)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists() and not refresh:
        with np.load(path, allow_pickle=False) as saved:
            if saved["signature"].item() == signature:
                report = json.loads(saved["payload"].item())
                report["cache_reused"] = True
                print("Reused matching comparison cache; recorded timings describe the original run.")
                return report
    if neural:
        # Exclude optional-library import from fit/evaluation timing.
        import nflows  # noqa: F401
        import torch  # noqa: F401
    start = time.perf_counter()
    report = _compact(
        {
            "identity": identity,
            "unit": "nats",
            "cache_reused": False,
            "neural_status": "included" if neural else "unavailable or explicitly disabled; not a zero score",
            "cases": _cases(config, neural),
            "sensitivity": _sensitivity(config),
            "approximations": _approximations(config),
            "discrete": _discrete(),
        }
    )
    report["computation_seconds"] = time.perf_counter() - start
    json.dumps(report, allow_nan=False)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, signature=signature, payload=json.dumps(report, allow_nan=False))
    print(
        f"Compared fixed protocols in {report['computation_seconds']:.1f}s; neural: {report['neural_status']}."
    )
    return report


def summary(report):
    """Compact audit; full signed values and per-estimate details remain in report."""
    from plotting import show_table

    product = report["cases"]["product"]
    rows = []
    for method in product["methods"]:
        records = [r for r in product["records"] if r["method"] == method]
        rows.append(
            [
                LABELS[method],
                f"{np.mean([r['ei'] for r in records]):.4f}",
                f"{np.mean([r['syn'] for r in records]):.4f}",
                sum(r["diagnostics"]["xi_tolerance_negative_count"] for r in records),
                sum(r["diagnostics"]["spt_audit"]["tolerance_zero_count"] for r in records),
                sum(r["diagnostics"]["ei_negative_count"] for r in records),
            ]
        )
    show_table(
        [
            "Product channel",
            "Mean EI (nats)",
            "Mean Syn (nats)",
            "Tol. Ξ count",
            "Tol. Syn count",
            "Negative EI count",
        ],
        rows,
    )
    ref = product["reference"]
    print(
        f"Channel reference: joint EI={ref['ei']:.4f}, singleton EI={ref['singleton_ei']:.4f}, Syn={ref['syn']:.4f} nats."
    )
    print(f"Syn tolerance={report['identity']['config']['syn_tolerance']} nats; values are never clipped.")
