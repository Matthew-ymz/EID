"""Known-truth Gaussian reader benchmark. All information is in nats.

Structure is supplied as a benchmark assumption, never learned from the truth.
No negative MI is clipped and no empirical null value is subtracted.
"""
from functools import lru_cache
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from eid.estimators.sample import wishart_mi_bias


DIMENSIONS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
SAMPLE_SIZES = [64, 128, 256, 512, 1024, 2048, 4096]


@lru_cache(None)
def correction(n, p, q):
    return wishart_mi_bias(n, p, q)


@lru_cache(None)
def rotation(d):
    rng = np.random.default_rng(817 + d)
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))[None, :]


def source_pair(n, d, seed):
    rng = np.random.default_rng(np.random.SeedSequence([20261029, n, d, seed]))
    x = rng.normal(size=(n, d))
    common = rng.normal(size=(n, (d + 4) // 5))
    obs = np.sqrt(.4) * x + np.sqrt(.6) * common[:, np.arange(d) // 5]
    noise = rng.normal(size=(n, d))
    return x, obs, noise


def truth(d, mode, family):
    if family == 'null':
        return 0.
    if mode == 'EI':
        return .5 * d * np.log(2.)
    value = 0.
    for lo in range(0, d, 5):
        b = min(5, d - lo)
        value += .5 * (np.log(1 + .4 + .6 * b) + (b - 1) * np.log(1.4))
    return float(value)


def response(x, noise, family):
    if family == 'null':
        return noise
    if family == 'diagonal':
        return x + noise
    if family == 'dense':
        # Rotate signal and noise together: identical sampled full information,
        # while the registered coordinate/block decomposition is destroyed.
        return (x + noise) @ rotation(x.shape[1]).T
    raise ValueError(f'Unknown channel {family}')


def logdet_positive(a):
    factor, _ = cho_factor(a, lower=True, check_finite=False)
    return float(2 * np.log(np.diag(factor)).sum())


def full_gaussian_readers(x, y, ridge=.01):
    n, d = x.shape
    a, b = x - x.mean(0), y - y.mean(0)
    if n <= 2 * d:
        ga, gb = a @ a.T / (n - 1), b @ b.T / (n - 1)
        eye = ridge * np.eye(n)
        # Equal-dimensional blocks make the dimensional constants cancel.
        value = .5 * (logdet_positive(ga + eye) + logdet_positive(gb + eye)
                      - logdet_positive(ga + gb + eye) - n * np.log(ridge))
        return dict(full_ridge=float(value), full_raw=None, full_corrected=None)
    z = np.c_[a, b]
    c = z.T @ z / (n - 1)
    cx, cy = c[:d, :d], c[d:, d:]
    raw = .5 * (logdet_positive(cx) + logdet_positive(cy) - logdet_positive(c))
    reg = .5 * (logdet_positive(cx + ridge * np.eye(d))
                + logdet_positive(cy + ridge * np.eye(d))
                - logdet_positive(c + ridge * np.eye(2 * d)))
    return dict(full_ridge=float(reg), full_raw=float(raw),
                full_corrected=float(raw - correction(n, d, d)))


def block_gaussian_reader(x, y, block_size=1, corrected=True):
    n, d = x.shape
    if n <= 2 * min(block_size, d):
        raise ValueError('Insufficient samples for block Gaussian correction')
    x, y = x - x.mean(0), y - y.mean(0)
    if block_size == 1:
        vx, vy = np.sum(x ** 2, axis=0), np.sum(y ** 2, axis=0)
        rho2 = np.sum(x * y, axis=0) ** 2 / (vx * vy)
        if np.any(rho2 >= 1):
            raise ValueError('Singular coordinate-pair covariance')
        value = -.5 * np.log1p(-rho2).sum()
        return float(value - (d * correction(n, 1, 1) if corrected else 0))
    blocks = d // block_size
    total = 0.
    if blocks:
        width = blocks * block_size
        xb = x[:, :width].reshape(n, blocks, block_size)
        yb = y[:, :width].reshape(n, blocks, block_size)
        z = np.concatenate([xb, yb], axis=2)
        cov = np.einsum('nbi,nbj->bij', z, z, optimize=True) / (n - 1)
        signs_x, lx = np.linalg.slogdet(cov[:, :block_size, :block_size])
        signs_y, ly = np.linalg.slogdet(cov[:, block_size:, block_size:])
        signs_j, lj = np.linalg.slogdet(cov)
        if np.any(signs_x <= 0) or np.any(signs_y <= 0) or np.any(signs_j <= 0):
            raise ValueError('Nonpositive block determinant')
        total += .5 * (lx + ly - lj).sum()
        if corrected:
            total -= blocks * correction(n, block_size, block_size)
    if d % block_size:
        total += block_gaussian_reader(x[:, blocks * block_size:], y[:, blocks * block_size:],
                                       d % block_size, corrected)
    return float(total)


def structure_is_valid(mode, family, d):
    return family != 'dense' or (mode == 'MI' and d <= 5)


def errors(value, reference, d):
    absolute = abs(value - reference)
    strict_tolerance = .05 if reference == 0 else max(.05, .1 * reference)
    broad_tolerance = max(.05, .01 * d) if reference == 0 else strict_tolerance
    return dict(error_nats=float(value - reference), absolute_error_nats=float(absolute),
                strict_pass=bool(absolute <= strict_tolerance),
                normalized_pass=bool(absolute <= broad_tolerance),
                tolerance_nats=float(strict_tolerance))
