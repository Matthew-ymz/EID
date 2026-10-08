"""Fit and validate a small data-driven response, then analyze the fitted model.

This self-contained NumPy example demonstrates the model adapter. Real temporal
data requires a time-ordered split and an explicitly chosen history/forecast lag.
"""

import json

import numpy as np

from eid import UniformBox, analyze_model


class LinearResponse:
    def __init__(self, x, y):
        self.coefficients = np.linalg.lstsq(self.design(x), y, rcond=None)[0]

    @staticmethod
    def design(x):
        return np.column_stack([np.ones(len(x)), x])

    def predict(self, x):
        return self.design(x) @ self.coefficients


rng = np.random.default_rng(13)
x = rng.uniform(-1, 1, size=(3000, 2))
y = x[:, 0] + x[:, 1] + rng.normal(0, 0.3, len(x))
model = LinearResponse(x[:2250], y[:2250])
residuals = y[2250:] - model.predict(x[2250:])
noise_std = float(residuals.std(ddof=1))
result = analyze_model(
    model,
    intervention=UniformBox([[-1, 1], [-1, 1]]),
    response_noise_std=noise_std,
    syn_tolerance=0.03,
    degree=1,
    seed=7,
)
result.metadata["validation_rmse"] = float(np.sqrt(np.mean(residuals**2)))
result.metadata["validation_sample_count"] = len(residuals)
print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
