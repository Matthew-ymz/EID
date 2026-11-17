import json

import numpy as np
import pytest

from eid import (
    IndependentGaussian,
    SPTNonnegativityError,
    UniformBox,
    analyze_dynamics,
    analyze_samples,
    available_estimators,
    estimate_ei,
    estimate_mi,
)


@pytest.fixture
def observational_channel():
    rng = np.random.default_rng(24)
    covariance = np.array([[1.0, 0.55], [0.55, 1.0]])
    x = rng.multivariate_normal(np.zeros(2), covariance, size=6000)
    y = x.sum(axis=1) + rng.normal(size=len(x))
    log_ratio = 0.5 * (
        np.linalg.slogdet(covariance)[1]
        + np.einsum("ni,ij,nj->n", x, np.linalg.inv(covariance) - np.eye(2), x)
    )
    return x, y, IndependentGaussian([0, 0], [1, 1]), np.exp(log_ratio)


def test_known_density_ratio_recovers_joint_singleton_and_syn(observational_channel):
    x, y, prior, weights = observational_channel
    options = {"sample_weight": weights, "assume_identified": True, "degree": 1, "ridge": 0}
    result = analyze_samples(
        x, y, intervention=prior, estimator="rw_ei", estimator_options=options, syn_tolerance=1e-12
    )
    # Independent unit Gaussian sources, unit Gaussian noise, Y = X0 + X1 + noise.
    joint, single = 0.5 * np.log(3), 0.5 * np.log(1.5)
    assert result.ei == pytest.approx(joint, abs=0.025)
    assert list(result.singleton_ei.values()) == pytest.approx([single, single], abs=0.025)
    assert result.tree.root.syn_value == pytest.approx(joint - 2 * single, abs=0.025)
    assert result.tree.closure_error == pytest.approx(0, abs=1e-12)
    assert result.diagnostics["spt_audit"]["tolerance_zero_count"] == 0
    ess = weights.sum() ** 2 / (weights @ weights)
    assert result.metadata["reweighting"]["effective_sample_size"] == pytest.approx(ess)
    # Subset weights are the joint ratio, not ratios recomputed on each marginal.
    for indices, value in result.ei_table.items():
        ref = estimate_mi(
            x[:, indices], y, estimator_options={"sample_weight": weights, "degree": 1, "ridge": 0}
        )
        assert value == pytest.approx(ref.value, abs=1e-12)
    assert '"sample_weight"' not in json.dumps(result.to_dict(), allow_nan=False)
    scaled = estimate_ei(
        x,
        y,
        intervention=prior,
        estimator="rw-ei",
        estimator_options={**options, "sample_weight": weights * 1e300},
        unit="bits",
    )
    assert "rw_ei" in available_estimators()
    assert (scaled.quantity, scaled.estimator, scaled.unit) == ("ei", "rw_ei", "bits")
    assert scaled.value == pytest.approx(result.ei / np.log(2), abs=1e-12)


def test_misspecified_weights_do_not_project_negative_xi(observational_channel):
    x, y, prior, _ = observational_channel
    with pytest.raises(SPTNonnegativityError, match=r"minimum=-.*threshold=-1e-12, affected_count=1"):
        analyze_samples(
            x,
            y,
            intervention=prior,
            estimator="rw_ei",
            syn_tolerance=1e-12,
            estimator_options={"sample_weight": np.ones(len(x)), "assume_identified": True, "degree": 1},
        )


def test_box_support_and_zero_weight_rows():
    prior = UniformBox([[-1, 1]])
    x = np.array([-0.8, -0.5, 0, 0.4, 0.9, 1000.0])
    y = np.array([-0.7, -0.2, 0.2, 0.25, 0.7, -1000.0])
    options = {"sample_weight": [1, 2, 1, 3, 2, 0], "assume_identified": True, "degree": 1}
    weighted = estimate_ei(x, y, intervention=prior, estimator="rw_ei", estimator_options=options)
    ref = estimate_mi(x[:-1], y[:-1], estimator_options={"sample_weight": [1, 2, 1, 3, 2], "degree": 1})
    assert weighted.value == pytest.approx(ref.value, abs=1e-12)
    assert weighted.diagnostics["protocol"]["reweighting"]["zero_weight_count"] == 1
    with pytest.raises(ValueError, match="outside the UniformBox"):
        estimate_ei(
            x,
            y,
            intervention=prior,
            estimator="rw_ei",
            estimator_options={**options, "sample_weight": [1, 2, 1, 3, 2, 1]},
        )


def test_reweighting_requires_deliberate_identification_and_input():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(20, 2)).T
    prior = IndependentGaussian([0], [1])
    options = {"sample_weight": np.ones(len(x)), "assume_identified": True}
    with pytest.raises(ValueError, match="assume_identified=True"):
        estimate_ei(
            x,
            y,
            estimator="rw_ei",
            intervention=prior,
            estimator_options={"sample_weight": options["sample_weight"]},
        )
    with pytest.raises(ValueError, match="finite nonnegative density ratio"):
        estimate_ei(
            x,
            y,
            estimator="rw_ei",
            intervention=prior,
            estimator_options={**options, "sample_weight": [-1] * len(x)},
        )
    with pytest.raises(ValueError, match="positive-weight rows"):
        estimate_ei(
            x,
            y,
            estimator="rw_ei",
            intervention=prior,
            estimator_options={**options, "sample_weight": [0] * len(x)},
        )
    with pytest.raises(ValueError, match="separately verified intervention"):
        estimate_ei(x, y, intervention=prior, estimator_options={"sample_weight": options["sample_weight"]})
    with pytest.raises(ValueError, match="train_fraction=1"):
        analyze_samples(
            x,
            y,
            estimator="rw_ei",
            estimator_options=options,
            intervention=prior,
            syn_tolerance=1e-12,
            train_fraction=0.5,
        )
    with pytest.raises(ValueError, match="takes weighted observations"):
        analyze_dynamics(lambda x: x, estimator="rw_ei", intervention=prior, syn_tolerance=1e-12)
