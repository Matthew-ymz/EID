"""Small fixed-ensemble backend tour; no external data or expensive experiments."""

import argparse
import json

import numpy as np

from eid import (
    IndependentGaussian,
    UniformBox,
    analyze_dynamics,
    analyze_linear_gaussian,
    analyze_samples,
    estimate_ei,
)


def run(neural=False):
    box = UniformBox([[-1, 1], [-1, 1]])
    rng = np.random.default_rng(7)
    x = box.sample(1200, rng)
    y = x[:, [0]] * x[:, [1]] + 0.3 * rng.normal(size=(len(x), 1))
    print("Fixed product channel; nats; Syn tolerance = 0.03 nats")
    print("estimator                 EI         Xi    negative-Xi-count")
    results = {}
    for method in [
        "tm_polynomial",
        "tm_affine",
        "tm_affine_lifted",
        "tm_affine_readout",
        "gaussian",
        "kde",
        "ksg",
    ]:
        result = analyze_samples(x, y, intervention=box, estimator=method, syn_tolerance=0.03)
        print(
            f"{method:24} {result.ei:8.4f} {result.xi:10.4f} "
            f"{result.diagnostics['xi_tolerance_negative_count']:4}"
        )
        json.dumps(result.to_dict(), allow_nan=False)
        results[method] = result
    heldout = analyze_samples(
        x,
        y,
        intervention=box,
        estimator="tm_polynomial",
        train_fraction=0.7,
        estimator_options={"degree": 2},
        syn_tolerance=0.03,
    )
    print(f"degree-2 TM, held out      {heldout.ei:8.4f} {heldout.xi:10.4f}")
    gaussian = IndependentGaussian([0, 0], [1, 1])
    analytic = analyze_linear_gaussian(
        [[1, 1]], intervention=gaussian, response_noise_cov=[[0.25]], syn_tolerance=1e-12
    )
    local = analyze_dynamics(
        lambda x: x[:, 0] + x[:, 1],
        intervention=gaussian,
        response_noise_cov=[[0.25]],
        estimator="local_gaussian",
        syn_tolerance=1e-12,
    )
    assert np.isclose(analytic.ei, local.ei)
    print(f"Linear Gaussian sum       {analytic.ei:8.4f} {analytic.xi:10.4f}")
    discrete = estimate_ei(np.eye(2), estimator="discrete", unit="bits")
    nis = estimate_ei(
        lambda x: 2 * x,
        estimator="nis",
        intervention=UniformBox([[-2, 2]]),
        response_noise_cov=[[0.01]],
        estimator_options={"assume_injective": True},
    )
    print(f"Discrete COPY: {discrete.value:.4f} bits; NIS square-map approximation: {nis.value:.4f} nats")
    if neural:
        # A one-source check avoids disguising a small training budget as a converged SPT experiment.
        rng = np.random.default_rng(5)
        intervention = IndependentGaussian([0], [1])
        source = intervention.sample(1000, rng)
        target = source + rng.normal(size=source.shape)
        result = analyze_samples(
            source,
            target,
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
            },
        )
        print(f"Shared neural TM: {result.ei:.4f} nats; analytic Gaussian reference: {0.5 * np.log(2):.4f}")
        json.dumps(result.to_dict(), allow_nan=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", action="store_true", help="also fit the optional small neural channel")
    run(parser.parse_args().neural)
