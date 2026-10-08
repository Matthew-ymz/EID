import json
import math

import numpy as np
import pytest

from eid import (
    UniformBox,
    analyze_dynamics,
    analyze_model,
    analyze_tpm,
    build_deterministic_boolean_tpm,
)
from eid.discrete import effective_information_from_tpm
from eid.spt import SPTConfig, SPTNonnegativityError, build_spt_from_ei_table


def test_exact_xor_units_marginalization_and_closure():
    tpm = build_deterministic_boolean_tpm(3, lambda s: (s[0] ^ s[2], s[1], s[2]))
    bits = analyze_tpm(tpm, n_nodes=3, sources=[0, 2], targets=[0], unit="bits", syn_tolerance=1e-12)
    nats = analyze_tpm(tpm, n_nodes=3, sources=[0, 2], targets=[0], unit="nats", syn_tolerance=1e-12)
    assert bits.ei == pytest.approx(1)
    assert bits.xi == pytest.approx(1)
    assert bits.singleton_ei == {0: 0, 2: 0}
    assert nats.xi == pytest.approx(math.log(2))
    assert bits.tree.closure_error == pytest.approx(0, abs=1e-12)
    json.dumps(bits.to_dict(), allow_nan=False)


def test_singleton_copy_has_zero_xi_and_json_safe_audit():
    result = analyze_tpm(np.eye(2), n_nodes=1, syn_tolerance=1e-12, unit="bits")
    assert result.ei == pytest.approx(1)
    assert result.xi == 0
    assert result.diagnostics["spt_audit"]["minimum_candidate_syn"] is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_horizon_forms_kernel_before_source_marginalization():
    tpm = build_deterministic_boolean_tpm(2, lambda s: (s[1], s[0]))
    lag1 = analyze_tpm(tpm, n_nodes=2, sources=[0], targets=[0], horizon=1, syn_tolerance=1e-12)
    lag2 = analyze_tpm(tpm, n_nodes=2, sources=[0], targets=[0], horizon=2, syn_tolerance=1e-12)
    assert lag1.ei == 0
    assert lag2.ei == pytest.approx(math.log(2))


def test_spectral_route_maps_nonconsecutive_source_coordinates():
    result = analyze_tpm(
        np.eye(32),
        n_nodes=5,
        sources=[1, 3, 4],
        targets=[1, 3, 4],
        syn_tolerance=1e-12,
        exact_max_size=2,
        unit="bits",
    )
    assert result.ei == pytest.approx(3)
    assert result.xi == pytest.approx(0, abs=1e-12)
    assert result.diagnostics["search"] == "spectral-candidates"
    assert result.tree.root.sources == (1, 3, 4)


def test_model_and_dynamics_use_the_same_intervention_ensemble():
    class Model:
        def predict(self, x):
            return x[:, 0] + x[:, 1]

    kwargs = {
        "intervention": UniformBox([[-1, 1], [-1, 1]]),
        "response_noise_std": 0.3,
        "syn_tolerance": 0.05,
        "degree": 1,
        "sample_count": 1000,
        "seed": 8,
    }
    model = Model()
    learned = analyze_model(model, **kwargs)
    known = analyze_dynamics(model.predict, **kwargs)
    assert learned.ei_table == known.ei_table
    assert learned.xi > 0
    assert learned.metadata["system_kind"] == "learned-model"
    assert abs(learned.tree.closure_error) < 1e-12
    json.dumps(learned.to_dict(), allow_nan=False)


@pytest.mark.parametrize("unit", ["bytes", "nat", ""])
def test_unsupported_units_fail(unit):
    with pytest.raises(ValueError, match="unit"):
        analyze_tpm(np.eye(2), n_nodes=1, syn_tolerance=0, unit=unit)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerances_fail(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        analyze_tpm(np.eye(2), n_nodes=1, syn_tolerance=tolerance)
    with pytest.raises(ValueError, match="tolerance"):
        SPTConfig(syn_tolerance=tolerance)


def test_nonfinite_information_cannot_pass_spt_audit():
    with pytest.raises(ValueError, match="Non-finite Xi"):
        build_spt_from_ei_table((0, 1), {(0,): 0, (1,): 0, (0, 1): float("nan")})


def test_syn_below_tolerance_fails_with_threshold_and_count():
    with pytest.raises(SPTNonnegativityError, match="threshold=-0.01, affected_count=1"):
        build_spt_from_ei_table(
            (0, 1), {(0,): 0.1, (1,): 0.1, (0, 1): 0.1}, config=SPTConfig(syn_tolerance=0.01)
        )


@pytest.mark.parametrize("base", [0, -2, 1, float("nan"), float("inf")])
def test_invalid_log_bases_fail(base):
    with pytest.raises(ValueError, match="log_base"):
        effective_information_from_tpm(np.eye(2), log_base=base)


def test_continuous_adapter_requires_explicit_positive_noise():
    with pytest.raises(ValueError, match="positive response noise"):
        analyze_dynamics(
            lambda x: x,
            intervention=UniformBox([[-1, 1]]),
            response_noise_std=0,
            syn_tolerance=0.01,
            sample_count=100,
        )


def test_invalid_intervention_bounds_fail():
    with pytest.raises(ValueError, match="finite"):
        UniformBox([[0, float("inf")]])
