"""Small, declared response mechanisms for the SPT gallery.

Cases are fixed in advance for structural variety, with no parameter search.
All discrete inputs use the full independent-uniform binary intervention.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from eid import build_deterministic_boolean_tpm


@dataclass(frozen=True)
class DiscreteCase:
    tpm: np.ndarray
    n_nodes: int
    targets: tuple[int, ...]
    mechanism: str
    horizon: int = 1


def _scalar_channel(n_nodes, response):
    """Register a scalar response at output 0 and copy auxiliary outputs."""
    return build_deterministic_boolean_tpm(n_nodes, lambda s: (int(response(s)), *s[1:]))


def structure_cases():
    """Six distinct gates: separate, global, nested, overlapping and switched."""
    pair_triad = build_deterministic_boolean_tpm(
        5, lambda s: (s[0] ^ s[1], s[1], s[2] ^ s[3] ^ s[4], s[3], s[4])
    )
    overlapping = build_deterministic_boolean_tpm(
        5, lambda s: (s[0] ^ s[1] ^ s[2], s[2] ^ s[3] ^ s[4], *s[2:])
    )
    return {
        "Pair + triad": DiscreteCase(pair_triad, 5, (0, 2), "y0=x0 XOR x1; y2=x2 XOR x3 XOR x4"),
        "Global parity": DiscreteCase(_scalar_channel(6, lambda s: sum(s) % 2), 6, (0,), "y0=XOR(x0,...,x5)"),
        "Gated parity": DiscreteCase(
            _scalar_channel(6, lambda s: (s[0] ^ s[1]) & (s[2] ^ s[3]) & (s[4] ^ s[5])),
            6,
            (0,),
            "y0=(x0 XOR x1) AND (x2 XOR x3) AND (x4 XOR x5)",
        ),
        "Overlapping parity": DiscreteCase(
            overlapping, 5, (0, 1), "y0=x0 XOR x1 XOR x2; y1=x2 XOR x3 XOR x4"
        ),
        "Majority vote": DiscreteCase(
            _scalar_channel(5, lambda s: sum(s) >= 3), 5, (0,), "y0=1 when sum(x0,...,x4)>=3"
        ),
        "Switched logic": DiscreteCase(
            _scalar_channel(4, lambda s: (s[1] ^ s[2]) if s[0] else s[3]),
            4,
            (0,),
            "y0=(x1 XOR x2) if x0=1, otherwise x3",
        ),
    }


def relay_cases(horizons=(1, 2, 3)):
    """Vary only the full-kernel horizon in a six-node periodic Boolean ring."""
    n_nodes = 6
    tpm = build_deterministic_boolean_tpm(
        n_nodes,
        lambda s: tuple(s[(i - 1) % n_nodes] ^ (s[i] & s[(i + 1) % n_nodes]) for i in range(n_nodes)),
    )
    return {
        f"Boolean relay · h={horizon}": DiscreteCase(
            tpm, n_nodes, (0,), "x_i'=x_(i-1) XOR (x_i AND x_(i+1)); periodic indices", horizon
        )
        for horizon in horizons
    }


def annotate_discrete_results(results, cases):
    """Keep the declared model equation alongside the raw analysis metadata."""
    for name, result in results.items():
        result.metadata.update(tutorial_model=name, response_mechanism=cases[name].mechanism)


def coupled_oscillator_flow(horizon):
    """Exact flow for q'=v, v'=-Kq-Dv, with state [q0,q1,q2,v0,v1,v2].

    Coordinates and time are dimensionless. Only horizon varies across cases;
    the same independent input box and observation-noise channel are used.
    """
    stiffness = np.array([[2.0, -0.8, 0.0], [-0.8, 2.4, -0.5], [0.0, -0.5, 1.7]])
    damping = np.diag([0.2, 0.3, 0.15])
    matrix = np.block([[np.zeros((3, 3)), np.eye(3)], [-stiffness, -damping]])
    flow = expm(matrix * horizon)

    def response(x):
        return x @ flow.T

    response.metadata = {
        "tutorial_model": "three coupled damped oscillators",
        "response_mechanism": "q'=v; v'=-Kq-Dv",
        "state_order": ["q0", "q1", "q2", "v0", "v1", "v2"],
        "stiffness": stiffness.tolist(),
        "damping": damping.tolist(),
        "forecast_interval": horizon,
        "integrator": "matrix-exponential",
    }
    return response


def annotate_flow_results(results, responses):
    """Retain the flow parameters and its forecast interval with each result."""
    for name, result in results.items():
        result.metadata.update(responses[name].metadata)
