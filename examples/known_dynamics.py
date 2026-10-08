"""A nonlinear joint response with explicitly specified measurement noise."""

import json

from eid import UniformBox, analyze_dynamics


def response(x):
    return x[:, 0] * x[:, 1]


result = analyze_dynamics(
    response,
    intervention=UniformBox([[-1, 1], [-1, 1]]),
    response_noise_std=0.3,
    syn_tolerance=0.03,
    degree=2,
    sample_count=4096,
    seed=7,
)
print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
