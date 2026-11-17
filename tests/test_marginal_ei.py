import numpy as np
import pytest

from eid import UniformBox, analyze_samples, estimate_ei


def test_gaussian_marginal_reader_matches_analytic_entropy():
    prior = UniformBox([[-1, 1]])
    # One kernel with width equal to the noise reproduces the Gaussian exactly;
    # a constant response conveys zero EI.
    result = estimate_ei(
        [[0]],
        [0],
        estimator="tm_marginal",
        intervention=prior,
        estimator_options={"response_noise_std": 0.3, "bandwidth": 0.3},
    )
    assert result.value == pytest.approx(0, abs=1e-8)
    assert result.diagnostics["mass_error"] < 1e-8
    assert result.diagnostics["map_second_moment"] == pytest.approx(1, abs=1e-5)


def test_marginal_rw_full_input_only_and_retains_signed_estimates():
    prior = UniformBox([[-1, 1]])
    x = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
    opts = {"response_noise_std": 0.3, "bandwidth": 0.15}
    plain = estimate_ei(
        x, np.zeros(len(x)), estimator="tm_marginal", intervention=prior, estimator_options=opts, unit="bits"
    )
    assert plain.value == pytest.approx(-1, abs=1e-8)
    weighted_opts = {
        **opts,
        "sample_weight": np.arange(1, 6),
        "assume_identified": True,
        "readout": "marginal_tm",
    }
    weighted = estimate_ei(
        x,
        np.zeros(len(x)),
        estimator="rw_ei",
        intervention=prior,
        estimator_options=weighted_opts,
        unit="bits",
    )
    assert weighted.value == pytest.approx(plain.value)
    assert weighted.diagnostics["reweighting"]["effective_sample_size"] == pytest.approx(225 / 55)
    with pytest.raises(ValueError, match="supports subset EI and SPT"):
        analyze_samples(
            x,
            np.zeros(len(x)),
            estimator="rw_ei",
            intervention=prior,
            estimator_options=weighted_opts,
            syn_tolerance=1e-12,
        )
