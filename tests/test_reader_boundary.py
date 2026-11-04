"""Known truths and optimized-reader equivalence, independent of scan results."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/reader_boundary'))
from methods import (block_gaussian_reader, full_gaussian_readers, response, rotation,
                     source_pair, truth)
from eid import estimate_mi
from analyze import boundaries, interval, paired_results


def test_gaussian_reader_matches_existing_public_estimator():
    x, _, noise = source_pair(80, 5, 0)
    y = x + noise
    row = full_gaussian_readers(x, y)
    for method, opts in [('full_raw', dict(jitter=0, bias_correction=False)),
                         ('full_corrected', dict(jitter=0, bias_correction=True)),
                         ('full_ridge', dict(jitter=.01, bias_correction=False))]:
        value = estimate_mi(x, y, estimator='gaussian', estimator_options=opts).value
        assert abs(row[method] - value) < 1e-10


def test_dual_ridge_matches_direct_covariance_when_n_is_small():
    x, _, noise = source_pair(12, 15, 3)
    y = x + noise
    c = np.cov(np.c_[x, y], rowvar=False) + .01 * np.eye(30)
    expected = .5 * (np.linalg.slogdet(c[:15, :15])[1]
                     + np.linalg.slogdet(c[15:, 15:])[1] - np.linalg.slogdet(c)[1])
    row = full_gaussian_readers(x, y)
    assert abs(row['full_ridge'] - expected) < 1e-9
    assert row['full_corrected'] is None


def test_block_sum_matches_explicit_block_estimates():
    _, x, noise = source_pair(128, 13, 8)
    y = x + noise
    for b in [1, 5]:
        value = 0
        for lo in range(0, 13, b):
            value += estimate_mi(x[:, lo:lo+b], y[:, lo:lo+b], estimator='gaussian',
                                 estimator_options=dict(jitter=0, bias_correction=True)).value
        assert abs(block_gaussian_reader(x, y, b) - value) < 1e-10


def test_analytic_truth_and_orthogonal_invariance():
    for d in [2, 5, 13]:
        c = .4 * np.eye(d)
        for lo in range(0, d, 5):
            c[lo:lo+5, lo:lo+5] += .6
        q = rotation(d)
        assert np.allclose(q.T @ q, np.eye(d), atol=1e-12)
        expected = .5 * np.linalg.slogdet(np.eye(d) + q @ c @ q.T)[1]
        assert abs(truth(d, 'MI', 'dense') - expected) < 1e-12
        assert abs(truth(d, 'EI', 'diagonal') - .5 * d * np.log(2)) < 1e-12
        x, _, noise = source_pair(80, d, 2)
        a = full_gaussian_readers(x, response(x, noise, 'diagonal'))
        b = full_gaussian_readers(x, response(x, noise, 'dense'))
        for method in a:
            assert abs(a[method] - b[method]) < 1e-10


def test_boundary_requires_null_and_signal_and_sustained_larger_n():
    rows = []
    for n in [64, 128, 256]:
        for seed in range(20):
            for family in ['null', 'diagonal']:
                ok = not (n == 128 and family == 'null')
                rows.append(dict(kind='gaussian', mode='EI', method='structured',
                                 dimension=2, sample_count=n, seed=seed, family=family,
                                 status='ok', assumption_valid=True,
                                 strict_pass=ok, normalized_pass=ok))
    pairs = paired_results(rows)
    result = boundaries(pairs, [2], [64, 128, 256])
    assert all(r['candidate_minimum_n'] == 256 for r in result)
    assert interval(20, 20)[0] < .9
    assert interval(100, 100)[0] > .9
