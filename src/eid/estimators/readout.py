"""Shared first-order affine readout with a declared intervention covariance."""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag

from .core import covariance, logdet, nonnegative


class AffineReadoutEIOracle:
    """Fit one full-input channel, then marginalize every subset of that channel.

    This implements the UniCM S12.1.2 readout. Gaussian log-determinants are
    a moment approximation for uniform/empirical interventions. Source prior
    moments are never mixed with an independently fitted target covariance.
    """

    def __init__(
        self,
        x,
        y,
        intervention,
        groups,
        *,
        standardize=False,
        residual_ridge=1e-6,
        ridge_scaling="absolute",
    ):
        from ..interventions import IndependentEmpirical, IndependentGaussian, UniformBox

        if not isinstance(standardize, (bool, np.bool_)):
            raise TypeError("standardize must be a boolean.")
        nonnegative(residual_ridge, "residual_ridge")
        if ridge_scaling not in ("absolute", "trace"):
            raise ValueError("ridge_scaling must be 'absolute' or 'trace'.")
        if len(x) < 4 or len(x) <= x.shape[1]:
            raise ValueError("Affine readout requires sample_count > full source dimension.")
        source_mean, target_mean = x.mean(axis=0), y.mean(axis=0)
        source, target = x - source_mean, y - target_mean
        if isinstance(intervention, UniformBox):
            widths = np.diff(np.asarray(intervention.bounds), axis=1).ravel()
            prior = np.diag(widths**2 / 12)
        elif isinstance(intervention, IndependentGaussian):
            prior = np.diag(np.asarray(intervention.std) ** 2)
        elif isinstance(intervention, IndependentEmpirical):
            prior = block_diag(
                *[(b - b.mean(axis=0)).T @ (b - b.mean(axis=0)) / len(b) for b in intervention.blocks]
            )
        else:
            raise TypeError("Affine readout requires a declared factorized intervention.")
        if not np.isfinite(prior).all():
            raise ValueError("Declared source moments must be finite.")
        source_scale, target_scale = np.ones(x.shape[1]), np.ones(y.shape[1])
        if standardize:
            if isinstance(intervention, IndependentEmpirical):
                raise ValueError("Unit-covariance standardized readout requires independent raw coordinates.")
            source_scale, target_scale = x.std(axis=0, ddof=1), y.std(axis=0, ddof=1)
            if np.any(source_scale == 0) or np.any(target_scale == 0):
                raise ValueError("Standardized readout cannot contain constant coordinates.")
            source, target = source / source_scale, target / target_scale
            # The calibration protocol explicitly replaces standardized prior
            # moments with identity, rather than reusing empirical correlations.
            prior = np.eye(x.shape[1])
        weights, _, rank, singular_values = np.linalg.lstsq(source, target, rcond=None)
        if rank != source.shape[1]:
            raise ValueError(
                f"Affine readout source is rank deficient: rank={rank}, dimension={source.shape[1]}."
            )
        residual = target - source @ weights
        raw_noise = residual.T @ residual / (len(x) - 1)
        ridge = residual_ridge * (
            max(float(np.trace(raw_noise) / len(raw_noise)), 1.0) if ridge_scaling == "trace" else 1.0
        )
        self.noise = covariance(
            raw_noise + ridge * np.eye(y.shape[1]), y.shape[1], "regularized readout residual covariance"
        )
        self.contributions = [weights[list(g)].T @ prior[np.ix_(g, g)] @ weights[list(g)] for g in groups]
        self.full_logdet = logdet(self.noise + sum(self.contributions))
        self.metadata = {
            "protocol": "one shared full-input affine readout with declared source moments",
            "train_count": len(x),
            "evaluation_count": 0,
            "density_fit_and_evaluation": "analytic-Gaussian-readout-after-projection",
            "source_covariance": prior,
            "source_covariance_rule": "identity-in-standardized-coordinates"
            if standardize
            else "declared-intervention-moments",
            "standardize": bool(standardize),
            "source_mean": source_mean,
            "target_mean": target_mean,
            "source_scale": source_scale,
            "target_scale": target_scale,
            "coupling": weights.T,
            "raw_residual_covariance": raw_noise,
            "residual_covariance": self.noise,
            "residual_ridge": residual_ridge,
            "applied_residual_ridge": ridge,
            "ridge_scaling": ridge_scaling,
            "source_fit_condition": float(singular_values[0] / singular_values[-1]),
            "approximation": "Gaussian first-order affine readout; not exact nonlinear or uniform-input MI",
            "clipped": False,
        }

    def ei(self, coalition):
        conditional = self.noise + sum(
            (c for g, c in enumerate(self.contributions) if g not in coalition),
            np.zeros_like(self.noise),
        )
        return 0.5 * (self.full_logdet - logdet(conditional))
