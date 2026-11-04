"""Numerical reference for the synthetic Gaussian product channel."""

import numpy as np


def product_interaction_ei(scale=1.0, noise_sd=0.5, *, n_grid=2 ** 17, step=0.005):
    """EI (nats) for y = scale * x1 * x2 + eps under independent N(0,1) sources.

    For independent standard normals the product has characteristic function
    (1 + t^2)^(-1/2), so the response CF is (1 + scale^2 t^2)^(-1/2) times the
    Gaussian noise factor. The response density is recovered by inverse FFT of
    that CF, which avoids the log singularity of the product density at zero.
    EI = h(y) - h(eps) holds exactly because the noise is additive and
    independent of the sources.
    """
    if not scale > 0 or not noise_sd > 0:
        raise ValueError("Positive scale and noise_sd required")
    t = 2 * np.pi * np.fft.fftfreq(n_grid, d=step)
    cf = np.exp(-0.5 * (noise_sd * t) ** 2) / np.sqrt(1.0 + (scale * t) ** 2)
    f = np.fft.fftshift(np.real(np.fft.ifft(cf))) / step
    y = (np.arange(n_grid) - n_grid // 2) * step
    keep = np.abs(y) <= 12.0
    f, y = f[keep], y[keep]
    if np.min(f) <= 0:
        raise ValueError("Response density not positive on the retained grid")
    h_y = float(-np.sum(f * np.log(f)) * step)
    h_eps = float(.5 * np.log(2 * np.pi * np.e * noise_sd ** 2))
    return {"ei_nats": h_y - h_eps, "h_response": h_y, "h_noise": h_eps,
            "step": step, "n_grid": n_grid, "scale": scale, "noise_sd": noise_sd,
            "mass_check": float(np.sum(f) * step),
            "estimator": "inverse FFT of the product-channel characteristic function"}
