"""Numerical references and estimator protocol boundaries, not implementation mirrors."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from eid import (
    EstimatorConfig,
    IndependentEmpirical,
    IndependentGaussian,
    UniformBox,
    analyze_dynamics,
    analyze_linear_gaussian,
    analyze_model,
    analyze_samples,
    available_estimators,
    estimate_ei,
    estimate_mi,
)
from eid.estimators.sample import manuscript_source_features
from eid.estimators.transport_map import fit_affine_transport_map_density


@pytest.fixture
def channel():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(2400, 1))
    y = 0.8 * x + 0.6 * rng.normal(size=x.shape)
    return x, y


@pytest.mark.parametrize("method", ["tm", "tm_affine", "gaussian", "kde", "knn"])
def test_estimators_recover_known_gaussian_mi_and_units(channel, method):
    x, y = channel
    result = estimate_mi(x, y, estimator=method)
    reference = -0.5 * math.log(1 - 0.8**2)
    assert result.value == pytest.approx(reference, abs=0.10)
    bits = estimate_mi(x, y, estimator=method, unit="bits")
    assert bits.value * math.log(2) == pytest.approx(result.value)
    assert result.diagnostics["clipped"] is False
    json.dumps(result.to_dict(), allow_nan=False)


def test_affine_protocol_matches_s13_density_ratio_and_s14_correction(channel):
    x, y = channel
    raw = estimate_mi(x, y, estimator="tm_affine")
    corrected = estimate_mi(x, y, estimator=EstimatorConfig("tm_affine", {"bias_correction": True}))
    covs = [np.atleast_2d(np.cov(z, rowvar=False)) for z in (x, y, np.column_stack([x, y]))]
    expected = 0.5 * sum(s * np.linalg.slogdet(c)[1] for s, c in zip([1, 1, -1], covs))
    assert raw.value == pytest.approx(expected, abs=1e-5)
    # Scalar Gaussian MI bias is ~1/(2*N), not bits and not a dimension-free offset.
    assert corrected.diagnostics["bias_correction"] == pytest.approx(1 / (2 * len(x)), rel=0.005)
    assert corrected.value == pytest.approx(raw.value - corrected.diagnostics["bias_correction"])


def test_manuscript_lift_detects_product_and_uses_exact_source_dimensions():
    box = UniformBox([[-1, 1], [-1, 1]])
    result = analyze_dynamics(
        lambda x: x[:, 0] * x[:, 1],
        intervention=box,
        response_noise_std=0.3,
        syn_tolerance=0.01,
        estimator="tm_affine_lifted",
        sample_count=2000,
        seed=7,
    )
    assert result.xi > 0.35
    assert [e["source_feature_dimension"] for e in result.diagnostics["estimates"]] == [3, 3, 5]
    assert all(e["bias_correction"] > 0 for e in result.diagnostics["estimates"])
    assert result.tree.closure_error == pytest.approx(0)
    assert np.array_equal(manuscript_source_features([[2, 3]]), [[2, 3, 6, 4, 9]])
    with pytest.raises(ValueError, match="one or two scalar"):
        estimate_mi(np.ones((20, 3)), np.arange(20), estimator="tm_affine_lifted")


def test_heldout_affine_density_evaluation_and_correction_boundary(channel):
    x, y = channel
    train, test = 1600, 800
    result = estimate_mi(
        x[:train], y[:train], estimator="tm_affine", evaluation_x=x[train:], evaluation_y=y[train:]
    )
    models = [
        fit_affine_transport_map_density(z) for z in (x[:train], y[:train], np.c_[x[:train], y[:train]])
    ]
    expected = (
        models[2].log_prob(np.c_[x[train:], y[train:]])
        - models[0].log_prob(x[train:])
        - models[1].log_prob(y[train:])
    ).mean()
    assert result.value == pytest.approx(expected)
    assert result.diagnostics["evaluation_count"] == test
    assert result.diagnostics["density_fit_and_evaluation"] == "held-out"
    with pytest.raises(ValueError, match="same-sample"):
        estimate_mi(
            x[:train], y[:train], estimator="tm_affine_lifted", evaluation_x=x[train:], evaluation_y=y[train:]
        )


def test_gaussian_rectangular_channel_marginalizes_omitted_sources():
    intervention = IndependentGaussian([0, 0, 0], [1, 2, 3])
    a, r = np.array([[1, 1, 1]]), np.array([[2.0]])
    full = estimate_ei(a, estimator="linear_gaussian", intervention=intervention, response_noise_cov=r)
    part = estimate_ei(
        a,
        estimator="linear_gaussian",
        intervention=intervention,
        response_noise_cov=r,
        estimator_options={"sources": [0, 2]},
    )
    assert full.value == pytest.approx(0.5 * math.log(16 / 2))
    assert part.value == pytest.approx(0.5 * math.log(16 / 6))
    tree = analyze_linear_gaussian(
        a, intervention=intervention, response_noise_cov=r, source_groups=[[0, 2], [1]], syn_tolerance=1e-12
    )
    assert tree.singleton_ei[0] == pytest.approx(part.value)
    assert tree.xi >= 0
    assert abs(tree.tree.closure_error) < 1e-12
    json.dumps(tree.to_dict(), allow_nan=False)


def test_local_gaussian_matches_exact_linear_channel_with_correlated_noise():
    a = np.array([[1, 0.3, 0.2], [0, 0.8, -0.2]])
    noise = np.array([[0.5, 0.1], [0.1, 0.8]])
    intervention = IndependentGaussian([1, -2, 3], [0.8, 0.7, 0.6])
    exact = analyze_linear_gaussian(
        a,
        intervention=intervention,
        response_noise_cov=noise,
        sources=[0, 2],
        targets=[1],
        syn_tolerance=1e-12,
    )
    local = analyze_dynamics(
        lambda x: x @ a.T,
        intervention=intervention,
        response_noise_cov=noise,
        sources=[0, 2],
        targets=[1],
        estimator="local_gaussian",
        syn_tolerance=1e-12,
    )
    assert local.ei == pytest.approx(exact.ei, rel=1e-8)
    assert local.xi == pytest.approx(exact.xi, rel=1e-8)
    assert local.metadata["approximation"] != "none"
    with pytest.raises(ValueError, match="IndependentGaussian"):
        analyze_dynamics(
            lambda x: x,
            intervention=UniformBox([[-1, 1]]),
            response_noise_std=0.3,
            estimator="local_gaussian",
            syn_tolerance=0.01,
        )


def test_nis_formula_retains_total_per_dimension_and_negative_raw_values():
    box = UniformBox([[-2, 2], [-1, 1]])
    a = np.array([[2, 0], [0, 3]])
    noise = np.array([[0.3, 0.1], [0.1, 0.5]])
    kwargs = {
        "estimator": "nis",
        "intervention": box,
        "response_noise_cov": noise,
        "estimator_options": {"assume_injective": True, "sample_count": 20},
    }
    result = estimate_ei(lambda x: x @ a.T, **kwargs)
    expected = math.log(8) + math.log(6) - math.log(2 * math.pi * math.e) - 0.5 * np.linalg.slogdet(noise)[1]
    assert result.value == pytest.approx(expected, abs=1e-9)
    kwargs["estimator_options"]["dimension_average"] = True
    assert estimate_ei(lambda x: x @ a.T, **kwargs).value == pytest.approx(result.value / 2)
    negative = estimate_ei(
        lambda x: x,
        estimator="nis",
        intervention=UniformBox([[-0.1, 0.1]]),
        response_noise_cov=[[100]],
        estimator_options={"assume_injective": True},
    )
    assert negative.value < 0
    assert negative.diagnostics["clipped"] is False


def test_nis_rejects_singular_rectangular_and_undeclared_injective_channels():
    box = UniformBox([[-1, 1], [-1, 1]])
    kwargs = {
        "estimator": "nis",
        "intervention": box,
        "response_noise_cov": np.eye(2),
        "estimator_options": {"assume_injective": True, "sample_count": 10},
    }
    with pytest.raises(ValueError, match="singular"):
        estimate_ei(lambda x: np.column_stack([x[:, 0], x[:, 0]]), **kwargs)
    with pytest.raises(ValueError, match="square"):
        estimate_ei(lambda x: x[:, 0], **kwargs)
    with pytest.raises(ValueError, match="assume_injective"):
        estimate_ei(lambda x: x, estimator="nis", intervention=box, response_noise_cov=np.eye(2))
    with pytest.raises(ValueError, match="subset EI/SPT"):
        analyze_dynamics(
            lambda x: x, intervention=box, response_noise_std=1, estimator="nis", syn_tolerance=0.01
        )


def test_empirical_blocks_preserve_within_block_dependence_and_use_group_players():
    base = np.arange(80, dtype=float)[:, None]
    intervention = IndependentEmpirical([np.column_stack([base, 2 * base]), base + 1])
    x = intervention.sample(1500, np.random.default_rng(27))
    assert np.array_equal(x[:, 1], 2 * x[:, 0])
    assert abs(np.corrcoef(x[:, 0], x[:, 2])[0, 1]) < 0.1
    result = analyze_samples(
        x,
        x[:, [0]] + x[:, [2]] + np.random.default_rng(28).normal(size=(len(x), 1)),
        intervention=intervention,
        estimator="gaussian",
        syn_tolerance=0.01,
    )
    assert result.sources == (0, 1)
    assert result.metadata["source_groups"] == [[0, 1], [2]]
    with pytest.raises(ValueError, match="preserve"):
        analyze_samples(
            x, x[:, [0]], intervention=intervention, source_groups=[[0], [1], [2]], syn_tolerance=0.01
        )


def test_shared_split_transforms_model_adaptation_and_process_noise():
    class Model:
        def predict(self, x):
            return np.sin(x[:, 0]) + x[:, 1]

    kwargs = {
        "intervention": UniformBox([[-1, 1], [-1, 1]]),
        "response_noise_cov": [[0.2]],
        "syn_tolerance": 0.03,
        "estimator": "tm",
        "estimator_options": {"degree": 2},
        "source_transforms": [lambda x: np.column_stack([x, x**2]), None],
        "train_fraction": 0.7,
        "sample_count": 1000,
        "seed": 2,
    }
    model = Model()
    result, known = analyze_model(model, **kwargs), analyze_dynamics(model.predict, **kwargs)
    assert result.ei_table == known.ei_table
    assert result.metadata["train_count"] == 700
    assert result.metadata["evaluation_count"] == 300
    assert all(e["density_fit_and_evaluation"] == "held-out" for e in result.diagnostics["estimates"])
    json.dumps(result.to_dict(), allow_nan=False)


def test_ksg_keeps_negative_estimates_and_rejects_tied_discrete_data():
    rng = np.random.default_rng(2)
    result = estimate_mi(rng.normal(size=(200, 1)), rng.normal(size=(200, 1)), estimator="ksg")
    assert result.value < 0
    repeated = np.zeros((20, 1))
    with pytest.raises(ValueError, match="duplicate"):
        estimate_mi(repeated, repeated, estimator="ksg")
    with pytest.raises(ValueError, match="smaller"):
        estimate_mi(np.arange(20), np.arange(20), estimator="ksg", estimator_options={"k": 20})


def test_discrete_dispatch_and_wrong_protocol_options(channel):
    assert estimate_ei(np.eye(3), estimator="discrete", unit="bits").value == pytest.approx(math.log2(3))
    x, y = channel
    with pytest.raises(TypeError, match="Declare the intervention"):
        estimate_ei(x, y)
    with pytest.raises(ValueError, match="Unknown"):
        EstimatorConfig("unknown")
    with pytest.raises(TypeError):
        estimate_mi(x, y, estimator="ksg", estimator_options={"bandwidth": 0.2})
    with pytest.raises(ValueError, match="not both"):
        estimate_mi(x, y, estimator=EstimatorConfig(), estimator_options={})
    for method in ["gaussian", "ksg"]:
        with pytest.raises(ValueError, match="held-out"):
            estimate_mi(x[:1000], y[:1000], estimator=method, evaluation_x=x[1000:], evaluation_y=y[1000:])
    assert {"tm_neural", "nis", "kde", "ksg", "tm_affine_lifted"} <= available_estimators().keys()


def test_neural_is_optional_and_missing_dependencies_have_actionable_error():
    # A fresh process ensures imports used elsewhere in this test suite cannot mask the check.
    code = """
import sys
import eid
assert 'torch' not in sys.modules and 'nflows' not in sys.modules
sys.modules['torch'] = None
from eid.estimators.neural import _torch
try:
    _torch()
except ImportError as e:
    assert 'eid-toolbox[neural]' in str(e)
else:
    raise AssertionError('missing optional dependency was ignored')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_declared_intervention_and_singular_source_diagnostics_remain_exportable():
    x = np.zeros((30, 1))
    y = np.random.default_rng(8).normal(size=(30, 1))
    result = estimate_mi(x, y)
    assert result.diagnostics["singular_source_sample_covariance"] is True
    assert result.diagnostics["fit_source_condition"] is None
    json.dumps(result.to_dict(), allow_nan=False)
    intervention = UniformBox([[-1, 1]])
    x = intervention.sample(30, np.random.default_rng(9))
    ei = estimate_ei(x, y, intervention=intervention)
    assert ei.quantity == "ei"
    assert ei.diagnostics["bounds"] == [[-1.0, 1.0]]


def test_invalid_analysis_tolerance_fails_before_calling_the_channel():
    def expensive_channel(x):
        raise AssertionError("invalid analysis settings must fail before channel evaluation")

    with pytest.raises(ValueError, match="syn_tolerance"):
        analyze_dynamics(
            expensive_channel,
            intervention=UniformBox([[-1, 1]]),
            response_noise_std=1,
            syn_tolerance=-1,
        )
    with pytest.raises(TypeError, match="Declare the intervention"):
        analyze_dynamics(expensive_channel, intervention=None, response_noise_std=1, syn_tolerance=0.01)


def test_backend_specific_options_are_not_silently_ignored(channel):
    x, y = channel
    for method, options in [
        ("tm_polynomial", {"jitter": 0.1}),
        ("tm_affine", {"degree": 2}),
        ("gaussian", {"ridge": 0.1}),
    ]:
        with pytest.raises(TypeError, match="Unsupported options"):
            estimate_mi(x, y, estimator=method, estimator_options=options)


def test_invalid_neural_component_partitions_fail_before_optional_training():
    x = np.random.default_rng(5).normal(size=(20, 1))
    for components in [
        [],
        [{"sources": [0], "targets": [0], "typo": True}],
        [{"sources": [False], "targets": [0]}],
    ]:
        with pytest.raises(ValueError, match="component"):
            estimate_ei(
                x,
                x,
                intervention=IndependentGaussian([0], [1]),
                estimator="tm_neural",
                estimator_options={"components": components},
            )


def test_shared_affine_readout_uses_known_uniform_covariance_and_marginalizes_groups():
    intervention = UniformBox([[-4, 4], [-4, 4]])
    rng = np.random.default_rng(31)
    x = intervention.sample(3000, rng)
    y = x.sum(axis=1, keepdims=True) + 0.5 * rng.normal(size=(len(x), 1))
    result = analyze_samples(
        x, y, intervention=intervention, estimator="tm_affine_readout", syn_tolerance=1e-12
    )
    assert result.ei == pytest.approx(0.5 * np.log(1 + (2 * 16 / 3) / 0.25), abs=0.04)
    assert np.array_equal(result.metadata["source_covariance"], (16 / 3) * np.eye(2))
    assert result.metadata["applied_residual_ridge"] == 1e-6
    assert result.xi >= 0
    assert abs(result.tree.closure_error) < 1e-12
    single = estimate_ei(x, y, intervention=intervention, estimator="tm_affine_readout", unit="bits")
    assert single.value * np.log(2) == pytest.approx(result.ei)
    grouped = analyze_samples(
        x,
        y,
        intervention=intervention,
        source_groups=[[0, 1]],
        estimator="tm_affine_readout",
        syn_tolerance=1e-12,
    )
    assert grouped.ei == pytest.approx(result.ei)
    assert grouped.xi == 0
    json.dumps(result.to_dict(), allow_nan=False)


def test_affine_readout_standardized_calibration_and_rank_boundaries():
    intervention = UniformBox([[-4, 4], [-4, 4]])
    rng = np.random.default_rng(42)
    x = intervention.sample(400, rng)
    y = x[:, [0]] + x[:, [1]] + rng.normal(size=(len(x), 1))
    result = analyze_samples(
        x,
        y,
        intervention=intervention,
        estimator="tm_affine_readout",
        estimator_options={"standardize": True},
        syn_tolerance=1e-12,
    )
    assert result.metadata["source_covariance"] == [[1, 0], [0, 1]]
    assert result.metadata["source_scale"] == pytest.approx(x.std(axis=0, ddof=1))
    assert result.metadata["source_covariance_rule"] == "identity-in-standardized-coordinates"
    scaled = analyze_samples(
        x,
        10 * y,
        intervention=intervention,
        estimator="tm_affine_readout",
        estimator_options={"ridge_scaling": "trace"},
        syn_tolerance=1e-12,
    )
    assert scaled.metadata["applied_residual_ridge"] == pytest.approx(
        1e-6 * scaled.metadata["raw_residual_covariance"][0][0]
    )
    with pytest.raises(ValueError, match="rank deficient"):
        estimate_ei(
            np.column_stack([x[:, 0], x[:, 0]]),
            y,
            intervention=intervention,
            estimator="tm_affine_readout",
        )
    with pytest.raises(ValueError, match="same-ensemble"):
        analyze_samples(
            x,
            y,
            intervention=intervention,
            estimator="tm_affine_readout",
            train_fraction=0.7,
            syn_tolerance=0.03,
        )
