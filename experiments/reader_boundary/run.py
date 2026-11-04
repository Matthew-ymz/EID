"""Reproducible fixed-weight, known-truth reader calibration (nats).

Run from repository root. Results resume only with matching scientific code and
configuration. This does not estimate observational reweighting weights.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from methods import (DIMENSIONS, SAMPLE_SIZES, ROOT, block_gaussian_reader,
                     errors, full_gaussian_readers, response, source_pair,
                     structure_is_valid, truth)
from eid import estimate_mi

VERSION = 'fixed-weight-reader-v1'
METHODS = ['full_ridge', 'full_raw', 'full_corrected', 'structured', 'ksg',
           'diagonal_mechanism_ei']


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def observation(seed, d, n, mode, family, method, value=None, *, valid=True,
                status='ok', reference=None, **extra):
    reference = truth(d, mode, family) if reference is None else reference
    row = dict(seed=seed, dimension=d, sample_count=n, mode=mode,
               family=family, method=method, reference_nats=reference,
               value_nats=value, assumption_valid=valid, status=status, **extra)
    if value is not None:
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite estimate: {row}')
        row.update(errors(value, reference, d))
    return row


def gaussian_cell(d, n, seeds, include_ksg=True):
    rows = []
    for seed in seeds:
        intervention, observed, noise = source_pair(n, d, seed)
        for mode, x in [('EI', intervention), ('MI', observed)]:
            full_diagonal = None
            for family in ['null', 'diagonal', 'dense']:
                y = response(x, noise, family)
                # Full Gaussian logdet and isotropic ridge are exactly invariant
                # to this target rotation. Reuse verified algebra, not a fitted null.
                full = full_diagonal if family == 'dense' else full_gaussian_readers(x, y)
                if family == 'diagonal':
                    full_diagonal = full
                for method, value in full.items():
                    rows.append(observation(seed, d, n, mode, family, method, value,
                                            status='ok' if value is not None else 'inapplicable_n_le_2d',
                                            reused_rotation_invariance=family == 'dense'))
                structured = block_gaussian_reader(x, y, 1 if mode == 'EI' else 5)
                rows.append(observation(seed, d, n, mode, family, 'structured', structured,
                                        valid=structure_is_valid(mode, family, d)))
                if mode == 'MI':
                    # Observed coordinate marginals equal the intervention ones;
                    # additive independent diagonal mechanisms allow this sum to
                    # target EI, even though observational joint MI is different.
                    value = block_gaussian_reader(x, y, 1)
                    rows.append(observation(seed, d, n, 'EI_from_observed', family,
                                            'diagonal_mechanism_ei', value,
                                            reference=truth(d, 'EI', family),
                                            valid=family != 'dense'))
                if include_ksg and d <= 20:
                    value = estimate_mi(x, y, estimator='ksg',
                                        estimator_options=dict(k=3, standardize=True),
                                        unit='nats').value
                    rows.append(observation(seed, d, n, mode, family, 'ksg', value))
    return rows


def product_truth():
    from product_reference import product_interaction_ei
    coarse = product_interaction_ei(n_grid=2**17, step=.005)
    fine = product_interaction_ei(n_grid=2**18, step=.0025)
    # Same retained support; this checks discretization, not the shared tail error.
    if abs(coarse['ei_nats'] - fine['ei_nats']) > 1e-5 or fine['mass_check'] < .99999:
        raise ValueError('Nonlinear numerical reference failed resolution/mass check')
    return dict(per_pair_nats=fine['ei_nats'], coarse=coarse, fine=fine,
                resolution_difference_nats=abs(coarse['ei_nats'] - fine['ei_nats']),
                support=[-12., 12.], caveat='Tiny omitted tail remains; reference is numerical.')


def nonlinear_cell(d, n, seeds, reference):
    rows = []
    for seed in seeds:
        x, _, noise = source_pair(n, d, seed + 20000)
        for family in ['product_null', 'product']:
            y = .5 * noise.copy()
            if family == 'product':
                y[:, ::2] += x[:, ::2] * x[:, 1::2]
            target = 0. if family == 'product_null' else d // 2 * reference
            full = full_gaussian_readers(x, y)
            for method, value in full.items():
                rows.append(observation(seed, d, n, 'common', family, method, value,
                                        reference=target, valid=family == 'product_null',
                                        status='ok' if value is not None else 'inapplicable_n_le_2d'))
            value = block_gaussian_reader(x, y, 1)
            rows.append(observation(seed, d, n, 'common', family, 'structured', value,
                                    reference=target, valid=family == 'product_null'))
            # Known independent two-source blocks, full polynomial reader per block.
            value = 0.
            for j in range(0, d, 2):
                value += estimate_mi(x[:, j:j+2], y[:, j:j+1], estimator='tm_polynomial',
                                     estimator_options=dict(degree=3, ridge=1e-6,
                                                            joint_order='source_first'),
                                     unit='nats').value
            rows.append(observation(seed, d, n, 'common', family, 'structured_polynomial', value,
                                    reference=target, valid=False,
                                    model_note='Correct block/conditional mean; Gaussian residual marginal is approximate.'))
            if d <= 10:
                value = estimate_mi(x, y, estimator='ksg',
                                    estimator_options=dict(k=3, standardize=True), unit='nats').value
                rows.append(observation(seed, d, n, 'common', family, 'ksg', value,
                                        reference=target))
    return rows


def fingerprint(config):
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for path in [Path(__file__), Path(__file__).with_name('methods.py'),
                 ROOT / 'src/eid/estimators/sample.py',
                 Path(__file__).with_name('product_reference.py')]:
        h.update(path.read_bytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/reader_boundary/results/reproduction')
    parser.add_argument('--dimensions', type=int, nargs='+', default=DIMENSIONS)
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=SAMPLE_SIZES)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--budget-seconds', type=float, default=1800)
    parser.add_argument('--skip-nonlinear', action='store_true')
    parser.add_argument('--skip-ksg', action='store_true')
    args = parser.parse_args()
    config = dict(version=VERSION, dimensions=args.dimensions, sample_sizes=args.sample_sizes,
                  repeats=args.repeats, seeds=list(range(args.repeats)),
                  modes=['EI', 'MI'], families=['null', 'diagonal', 'dense'],
                  weights='fixed equal', ridge=.01, source_block_size=5,
                  source_block_correlation=.6, ksg_k=3,
                  ksg_dimensions=[d for d in args.dimensions if d <= 20] if not args.skip_ksg else [],
                  ksg_sample_sizes=args.sample_sizes,
                  nonlinear=not args.skip_nonlinear,
                  nonlinear_dimensions=[2, 10, 50], nonlinear_sample_sizes=[256, 1024, 4096],
                  strict_null_tolerance_nats=.05, signal_relative_tolerance=.1,
                  normalized_null_tolerance_per_dimension_nats=.01,
                  empirical_cell_pass_fraction=.9, budget_seconds=args.budget_seconds,
                  unit='nats', null_subtraction=False, clipping=False)
    run_hash = fingerprint(config)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['fingerprint'] != run_hash:
            raise RuntimeError('Existing cache provenance differs; use another output directory.')
    manifest = dict(config=config, fingerprint=run_hash,
                    started_utc=datetime.now(timezone.utc).isoformat(), status='running',
                    runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                 platform=platform.platform()), completed_cells=0)
    dump(manifest_path, manifest)
    start = time.monotonic()
    planned = [('gaussian', d, n) for d in args.dimensions for n in args.sample_sizes]
    if not args.skip_nonlinear:
        manifest['nonlinear_reference'] = product_truth()
        planned += [('nonlinear', d, n) for d in [2, 10, 50] for n in [256, 1024, 4096]]
    manifest['planned_cells'] = len(planned)
    for kind, d, n in planned:
        path = output / f'{kind}_d{d}_n{n}.json'
        if path.exists():
            cached = json.loads(path.read_text())
            if cached['fingerprint'] != run_hash or len(cached['seeds']) != args.repeats:
                raise RuntimeError(f'Mismatched cell cache: {path}')
            manifest['completed_cells'] += 1
            continue
        if time.monotonic() - start >= args.budget_seconds:
            manifest.update(status='budget_reached', elapsed_seconds=time.monotonic()-start)
            dump(manifest_path, manifest)
            print(json.dumps(dict(event='budget_reached', **manifest)), flush=True)
            return
        cell_start = time.monotonic()
        try:
            if kind == 'gaussian':
                rows = gaussian_cell(d, n, config['seeds'], include_ksg=not args.skip_ksg)
            else:
                rows = nonlinear_cell(d, n, config['seeds'],
                                      manifest['nonlinear_reference']['per_pair_nats'])
        except Exception as exc:
            manifest.update(status='failed', failed_cell=[kind, d, n], exception=repr(exc),
                            elapsed_seconds=time.monotonic()-start)
            dump(manifest_path, manifest)
            raise
        elapsed = time.monotonic()-cell_start
        dump(path, dict(fingerprint=run_hash, kind=kind, dimension=d, sample_count=n,
                        seeds=config['seeds'], elapsed_seconds=elapsed, rows=rows))
        manifest['completed_cells'] += 1
        manifest['elapsed_seconds'] = time.monotonic()-start
        dump(manifest_path, manifest)
        print(json.dumps(dict(event='cell_complete', kind=kind, d=d, n=n,
                              seconds=round(elapsed, 2),
                              completed=manifest['completed_cells'], total=len(planned))), flush=True)
    manifest.update(status='complete', elapsed_seconds=time.monotonic()-start,
                    finished_utc=datetime.now(timezone.utc).isoformat())
    dump(manifest_path, manifest)
    print(json.dumps(dict(event='complete', seconds=manifest['elapsed_seconds'],
                          cells=manifest['completed_cells'])), flush=True)


if __name__ == '__main__':
    main()
