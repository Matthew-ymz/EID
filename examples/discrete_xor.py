"""Exact XOR under a common factorized binary intervention: Xi = 1 bit."""

import json

from eid import analyze_tpm, build_deterministic_boolean_tpm

tpm = build_deterministic_boolean_tpm(2, lambda state: (state[0] ^ state[1], state[1]))
result = analyze_tpm(tpm, n_nodes=2, targets=[0], unit="bits", syn_tolerance=1e-12)
print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
