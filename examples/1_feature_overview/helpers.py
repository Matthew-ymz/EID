"""Example systems and numerical checks for the feature overview.

These helpers prepare the declared channels; EI estimation remains in eid.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm

from eid import AnalysisResult, SPTConfig, SPTNonnegativityError, build_spt_from_ei_table
from eid.compat import flatten_phi_tree, greedy_phi_atoms, greedy_phi_tree, subset_phi_raw
from eid.discrete import (
    build_deterministic_boolean_tpm,
    build_probabilistic_boolean_tpm,
    coarse_grain_output_tpm,
    coarse_grain_tpm_by_state_labels,
    discrete_effective_information,
    enumerate_binary_states,
    source_target_tpm,
)
from eid.estimators.transport_map import (
    AffineTransportMapDensityEstimator,
    PolynomialTriangularTransportMapDensityEstimator,
    QuadraticTriangularTransportMapDensityEstimator,
    multivariate_gaussian_logpdf,
    standard_gaussian_logpdf,
)
from eid.spt import (
    ALL_ORDER_CROSS_DENSITY,
    SIGNED,
    all_nonempty_subsets,
    audit_syn_value,
    canonical_split,
    cross_coalition_count,
    exact_candidate_selector,
    nontrivial_bipartitions,
    split_objective_value,
)


def show_environment():
    """Display versions and keep embedded figures in PNG format."""
    import matplotlib
    import scipy
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    import eid

    set_matplotlib_formats("png")
    print(
        f"EID {eid.__version__} | NumPy {np.__version__} | SciPy {scipy.__version__} | Matplotlib {matplotlib.__version__}"
    )


def noisy_parity_tpm(strength=1.7):
    """Build noisy parity on sources 0 and 2, with two auxiliary outputs."""
    adjacency = np.eye(3)
    node_specs = [
        {"alpha": 0, "beta": 0, "gamma": strength, "parity_sources": [0, 2]},
        {"alpha": 1.5, "beta": 0, "gamma": 0},
        {"alpha": 1.5, "beta": 0, "gamma": 0},
    ]
    return build_probabilistic_boolean_tpm(adjacency, node_specs)


def sum_response(x):
    return x[:, 0] + x[:, 1]


def product_response(x):
    return x[:, 0] * x[:, 1]


def logistic_map(rate=3.7, steps=1):
    """Return a batch response after the requested logistic iterations."""

    def response(x):
        values = x[:, 0].copy()
        for _ in range(steps):
            values = rate * values * (1 - values)
        return values

    return response


def henon_map(a=1.4, b=0.3):
    """Return the one-step, two-output Hénon response."""

    def response(x):
        return np.column_stack([1 - a * x[:, 0] ** 2 + x[:, 1], b * x[:, 0]])

    return response


def oscillator_flow(frequency=1.5, damping_ratio=0.15, horizon=0.8):
    """Return the exact linear flow to a declared forecast interval."""
    matrix = np.array([[0.0, 1.0], [-(frequency**2), -2 * damping_ratio * frequency]])
    flow = expm(matrix * horizon)

    def response(x):
        return x @ flow.T

    response.matrix = matrix
    return response


def lorenz_vector_field(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Evaluate the dimensionless Lorenz benchmark vector field."""
    return np.column_stack(
        [
            sigma * (x[:, 1] - x[:, 0]),
            x[:, 0] * (rho - x[:, 2]) - x[:, 1],
            x[:, 0] * x[:, 1] - beta * x[:, 2],
        ]
    )


def lorenz_flow(dt=0.01, steps=8):
    """Return a fixed-step RK4 future-state response, with no observation noise."""

    def response(x, dt=dt, steps=steps):
        state = x.copy()
        for _ in range(steps):
            k1 = lorenz_vector_field(state)
            k2 = lorenz_vector_field(state + dt * k1 / 2)
            k3 = lorenz_vector_field(state + dt * k2 / 2)
            k4 = lorenz_vector_field(state + dt * k3)
            state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return state

    return response


def check_lorenz_integration(lorenz_response, lorenz_box, dt, steps):
    """Compare RK4 at the same forecast time with half-sized integration steps."""
    lorenz_check = lorenz_box.sample(64, np.random.default_rng(19))
    integration_difference = np.max(
        np.abs(lorenz_response(lorenz_check) - lorenz_response(lorenz_check, dt=dt / 2, steps=2 * steps))
    )
    assert integration_difference < 0.002
    print(f"Lorenz RK4 step-halving maximum state difference: {integration_difference:.2e}")
    return float(integration_difference)


class LinearResponse:
    def __init__(self, x, y):
        self.coefficients = np.linalg.lstsq(self.design(x), y, rcond=None)[0]

    @staticmethod
    def design(x):
        return np.column_stack([np.ones(len(x)), x])

    def predict(self, x):
        return self.design(x) @ self.coefficients


def fit_linear_example(seed=13, sample_count=3000, train_count=2250, noise_std=0.3):
    """Fit a NumPy linear predictor and estimate noise from untouched holdout samples."""
    training_rng = np.random.default_rng(seed)
    training_x = training_rng.uniform(-1, 1, size=(sample_count, 2))
    training_y = sum_response(training_x) + training_rng.normal(0, noise_std, len(training_x))
    model = LinearResponse(training_x[:train_count], training_y[:train_count])
    holdout_predictions = model.predict(training_x[train_count:])
    residuals = training_y[train_count:] - holdout_predictions
    validated_noise_std = float(residuals.std(ddof=1))
    validation_rmse = float(np.sqrt(np.mean(residuals**2)))
    return SimpleNamespace(
        model=model,
        observed=training_y[train_count:],
        predictions=holdout_predictions,
        residuals=residuals,
        noise_std=validated_noise_std,
        rmse=validation_rmse,
    )


def check_model_adapter(learned_result, model_as_dynamics, training):
    """Check model/function adapter agreement and retain validation metadata."""
    assert learned_result.ei_table == model_as_dynamics.ei_table
    learned_result.metadata.update(
        validation_rmse=training.rmse, validation_sample_count=len(training.residuals)
    )


def xor_modules_tpm(pair_count):
    """Build independent two-bit XOR modules with auxiliary copy outputs."""
    return build_deterministic_boolean_tpm(
        2 * pair_count,
        lambda s: tuple(v for i in range(0, 2 * pair_count, 2) for v in (s[i] ^ s[i + 1], s[i + 1])),
    )


def check_discrete_channels(copy_tpm, xor_tpm, source_only, target_only, rectangular, macro_or):
    """Cross-check equivalent marginalization and OR pooling routes."""
    states = enumerate_binary_states(2)
    assert np.allclose(rectangular, coarse_grain_output_tpm(source_only, states[:, 0]))
    assert np.allclose(target_only, source_target_tpm(xor_tpm, 2, source_indices=[0, 1], target_indices=[0]))
    assert np.isclose(
        discrete_effective_information(
            xor_tpm, n_nodes=2, source_indices=[0], target_indices=[0], log_base=2
        ),
        0,
    )
    or_labels = np.any(states, axis=1).astype(int)
    macro_labels = coarse_grain_tpm_by_state_labels(
        copy_tpm,
        input_state_labels=or_labels,
        output_state_labels=or_labels,
    )
    assert np.allclose(macro_or, macro_labels)


def check_hierarchies(pair_result, table_tree, oracle_tree, affinity_tolerance_count, six_affinity_count):
    """Check planted modules, complete tables, and exact candidate conventions."""
    assert np.isclose(pair_result.xi, 2)
    assert {child.sources for child in pair_result.tree.root.children} == {(0, 1), (2, 3)}
    assert set(pair_result.ei_table) == set(all_nonempty_subsets(pair_result.sources))
    assert np.isclose(table_tree.closure_error, 0) and np.isclose(oracle_tree.closure_error, 0)
    assert affinity_tolerance_count == six_affinity_count == 0
    _, splits = exact_candidate_selector((0, 1, 2))
    assert len(list(splits)) == len(nontrivial_bipartitions((0, 1, 2))) == 3
    assert canonical_split((2,), (0, 1), (0, 1, 2)) == ((0, 1), (2,))


def fixed_split_selector(allowed_splits):
    """Create a candidate selector from declared legal partitions."""

    def selector(coalition):
        return "fixed-prior", [allowed_splits[coalition]]

    return selector


def density_samples(bend=0.6, seed=41):
    """Draw the original Gaussian and unit-Jacobian banana train/test samples."""
    rng = np.random.default_rng(seed)
    mean = np.array([0.8, -0.4])
    covariance = np.array([[1.6, 0.45], [0.45, 0.9]])
    gaussian_train = rng.multivariate_normal(mean, covariance, size=4000)
    gaussian_test = rng.multivariate_normal(mean, covariance, size=800)

    def transform(base):
        warped = base.copy()
        warped[:, 1] += bend * (warped[:, 0] ** 2 - 1)
        return warped

    def inverse(values):
        base = values.copy()
        base[:, 1] -= bend * (base[:, 0] ** 2 - 1)
        return base

    banana_train = transform(rng.normal(size=(4000, 2)))
    banana_test = transform(rng.normal(size=(1000, 2)))
    return SimpleNamespace(
        gaussian_train=gaussian_train,
        gaussian_test=gaussian_test,
        mean=mean,
        covariance=covariance,
        banana_train=banana_train,
        banana_test=banana_test,
        true_banana_log_prob=standard_gaussian_logpdf(inverse(banana_test)),
    )


def evaluate_densities(data, affine_gaussian, banana_models):
    """Evaluate held-out log-density errors and PDF/log-probability consistency."""
    assert isinstance(affine_gaussian, AffineTransportMapDensityEstimator)
    assert isinstance(banana_models["Quadratic"], QuadraticTriangularTransportMapDensityEstimator)
    assert isinstance(banana_models["Polynomial, degree 2"], PolynomialTriangularTransportMapDensityEstimator)
    assert np.allclose(
        affine_gaussian.pdf(data.gaussian_test), np.exp(affine_gaussian.log_prob(data.gaussian_test))
    )
    gaussian_log_rmse = np.sqrt(
        np.mean(
            (
                affine_gaussian.log_prob(data.gaussian_test)
                - multivariate_gaussian_logpdf(data.gaussian_test, data.mean, data.covariance)
            )
            ** 2
        )
    )
    density_errors = {
        name: float(np.sqrt(np.mean((estimator.log_prob(data.banana_test) - data.true_banana_log_prob) ** 2)))
        for name, estimator in banana_models.items()
    }
    for estimator in banana_models.values():
        assert np.allclose(estimator.pdf(data.banana_test), np.exp(estimator.log_prob(data.banana_test)))
    return float(gaussian_log_rmse), density_errors


def gaussian_channel(correlation=0.75, sample_count=3000, seed=47):
    """Draw paired samples from the original scalar Gaussian channel."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(sample_count, 1))
    y = correlation * x + np.sqrt(1 - correlation**2) * rng.normal(size=x.shape)
    return x, y


def check_mi(mi, mi_uniform_weight, analytic_mi_bits):
    """Check uniform weighting and the known Gaussian-channel MI reference."""
    assert np.isclose(mi["mi_hat"], mi_uniform_weight["mi_hat"])
    assert abs(mi["mi_hat"] - analytic_mi_bits) < 0.08


def noisy_pairwise_response(noise_std=0.25, seed=23):
    """Return a simulator with explicit measurement noise for legacy pairwise EI."""
    rng = np.random.default_rng(seed)

    def response(x):
        mean = np.column_stack([x[:, 0] + 0.2 * x[:, 1], x[:, 2]])
        return mean + rng.normal(0, noise_std, size=mean.shape)

    return response


def export_result(result):
    """Check finite JSON export and the declared continuous estimator metadata."""
    exported = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert exported["unit"] == "nats"
    assert exported["metadata"]["bias_correction"] == "none"
    return exported


def tolerance_audit(audit_tolerance_bits=0.01):
    """Retain a tolerance-scale negative and require a significant violation to fail."""
    tiny_table = {("a",): 0.1, ("b",): 0.1, ("a", "b"): 0.195}
    tiny_tree = build_spt_from_ei_table(
        ("a", "b"), tiny_table, config=SPTConfig(syn_tolerance=audit_tolerance_bits)
    )
    assert np.isclose(tiny_tree.root.syn_value, -0.005)
    assert tiny_tree.audit.tolerance_zero_count == 1
    assert audit_syn_value(tiny_tree.root.syn_value, tolerance=audit_tolerance_bits, context="tutorial")
    bad_table = {("a",): 0.1, ("b",): 0.1, ("a", "b"): 0.1}
    try:
        build_spt_from_ei_table(("a", "b"), bad_table, config=SPTConfig(syn_tolerance=audit_tolerance_bits))
    except SPTNonnegativityError as error:
        expected_error = str(error)
    else:
        raise AssertionError("A significant Syn nonnegativity violation must fail.")
    return SimpleNamespace(
        tree=tiny_tree, tolerance=audit_tolerance_bits, bad_table=bad_table, expected_error=expected_error
    )


def legacy_examples(pair_result, DISCRETE_TOL, audit_tolerance_bits, bad_table):
    """Exercise historical Phi aliases and explicitly nonstandard SPT settings."""
    pair_table = pair_result.ei_table
    legacy_sources = tuple("abcd")
    legacy_table = {tuple(legacy_sources[i] for i in key): value for key, value in pair_table.items()}
    legacy_root = greedy_phi_tree(legacy_sources, legacy_table, split_tolerance=DISCRETE_TOL)
    assert np.isclose(subset_phi_raw(legacy_sources, legacy_table), pair_result.xi)
    assert np.isclose(sum(a.value for a in flatten_phi_tree(legacy_root)), pair_result.xi)
    assert greedy_phi_atoms(legacy_sources, legacy_table, split_tolerance=DISCRETE_TOL) == flatten_phi_tree(
        legacy_root
    )

    signed_diagnostic = build_spt_from_ei_table(
        ("a", "b"),
        bad_table,
        config=SPTConfig(policy=SIGNED, syn_tolerance=audit_tolerance_bits),
    )
    normalized_tree = build_spt_from_ei_table(
        pair_result.sources,
        pair_table,
        config=SPTConfig(syn_tolerance=DISCRETE_TOL, split_objective=ALL_ORDER_CROSS_DENSITY),
    )
    terminal_tree = build_spt_from_ei_table(
        ("a", "b"),
        {("a",): 0.2, ("b",): 0.3, ("a", "b"): 0.5},
        config=SPTConfig(syn_tolerance=DISCRETE_TOL, eps=1e-8, complete_to_singletons=False),
    )
    assert cross_coalition_count(2, 2) == 9
    assert np.isclose(split_objective_value(1.0, 2, 2, objective=ALL_ORDER_CROSS_DENSITY), 1 / 9)
    return SimpleNamespace(
        legacy_root=legacy_root,
        signed_diagnostic=signed_diagnostic,
        normalized_tree=normalized_tree,
        terminal_tree=terminal_tree,
    )


def validate_tutorial(all_standard_results, trees, xor, xor_nats, lag_results):
    """Run the original tutorial assertions outside the reader-facing cells."""
    assert isinstance(xor, AnalysisResult)
    assert np.isclose(xor.ei, 1) and np.isclose(xor.xi, 1)
    assert xor.singleton_ei == {0: 0.0, 1: 0.0}
    assert np.isclose(xor_nats.xi, xor.xi * math.log(2))
    assert np.allclose([r.ei for r in lag_results.values()], [0, 1, 0, 1, 0, 1])
    for result in all_standard_results:
        assert abs(result.tree.closure_error) < 1e-9
        audit = result.diagnostics["spt_audit"]
        minimum = audit["minimum_candidate_syn"]
        assert minimum is None or minimum >= -result.diagnostics["syn_tolerance"]
        json.dumps(result.to_dict(), allow_nan=False)
    for tree in trees:
        assert abs(tree.closure_error) < 1e-9
    print(
        f"Validated {len(all_standard_results)} public analyses, JSON export, tree closure, and expected failure."
    )
