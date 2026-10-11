import json

import numpy as np
import pytest

from eid import IndependentGaussian, analyze_samples


def test_shared_neural_channel_recovers_linear_mi_and_preserves_rng():
    torch = pytest.importorskip("torch")
    pytest.importorskip("nflows")
    intervention = IndependentGaussian([0], [1])
    rng = np.random.default_rng(5)
    x = intervention.sample(1000, rng)
    y = x + rng.normal(size=x.shape)
    state, threads = torch.random.get_rng_state().clone(), torch.get_num_threads()
    result = analyze_samples(
        x,
        y,
        intervention=intervention,
        syn_tolerance=0.03,
        estimator="tm_neural",
        estimator_options={
            "epochs": 60,
            "hidden_features": 16,
            "layers": 1,
            "evaluation_count": 256,
            "marginal_samples": 128,
            "seed": 3,
        },
    )
    assert result.ei == pytest.approx(0.5 * np.log(2), abs=0.15)
    assert len(result.metadata["fits"]) == 1
    assert result.metadata["evaluation_distribution"].endswith("fitted conditional neural channel")
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    json.dumps(result.to_dict(), allow_nan=False)


def test_independent_neural_components_close_exactly_without_projection():
    pytest.importorskip("torch")
    pytest.importorskip("nflows")
    intervention = IndependentGaussian([0, 0], [1, 1])
    rng = np.random.default_rng(21)
    x = intervention.sample(600, rng)
    y = x + rng.normal(size=x.shape)
    result = analyze_samples(
        x,
        y,
        intervention=intervention,
        estimator="tm_neural",
        syn_tolerance=1e-12,
        estimator_options={
            "epochs": 30,
            "hidden_features": 12,
            "layers": 1,
            "evaluation_count": 64,
            "marginal_samples": 32,
            "context_selection": False,
            "components": [{"sources": [0], "targets": [0]}, {"sources": [1], "targets": [1]}],
        },
    )
    assert result.ei == pytest.approx(sum(result.singleton_ei.values()), abs=1e-12)
    assert result.xi == pytest.approx(0, abs=1e-12)
    assert result.tree.closure_error == pytest.approx(0, abs=1e-12)
    assert result.metadata["component_factorization"] is True
    assert len(result.metadata["fits"]) == 2


def test_one_shared_two_source_neural_channel_marginalizes_both_singletons():
    pytest.importorskip("torch")
    pytest.importorskip("nflows")
    intervention = IndependentGaussian([0, 0], [1, 1])
    rng = np.random.default_rng(14)
    x = intervention.sample(1200, rng)
    y = x[:, [0]] + x[:, [1]] + rng.normal(size=(len(x), 1))
    result = analyze_samples(
        x,
        y,
        intervention=intervention,
        estimator="tm_neural",
        syn_tolerance=0.03,
        estimator_options={
            "epochs": 60,
            "hidden_features": 16,
            "layers": 1,
            "seed": 3,
            "evaluation_count": 256,
            "marginal_samples": 128,
            "context_selection": False,
        },
    )
    assert result.ei == pytest.approx(0.5 * np.log(3), abs=0.15)
    assert result.xi == pytest.approx(0.5 * np.log(3) - np.log(1.5), abs=0.12)
    assert set(result.ei_table) == {(0,), (1,), (0, 1)}
    assert len(result.metadata["fits"]) == 1
    assert abs(result.tree.closure_error) < 1e-12
