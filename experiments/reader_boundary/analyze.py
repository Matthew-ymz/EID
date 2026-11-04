"""Summarize fixed-threshold known-truth calibration without reselecting seeds."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from methods import DIMENSIONS, SAMPLE_SIZES, ROOT

LABELS = dict(full_ridge='Full Gaussian: ridge 0.01', full_raw='Full Gaussian: raw',
              full_corrected='Full Gaussian: analytic correction',
              structured='Gaussian: registered blocks', ksg='KSG (k=3)',
              structured_polynomial='Polynomial TM: registered pairs',
              diagonal_mechanism_ei='EI: diagonal mechanism')
COLORS = dict(full_ridge='#CC6677', full_raw='#AA4499', full_corrected='#4477AA',
              structured='#228833', ksg='#EE7733', structured_polynomial='#66AACC')


def interval(successes, trials):
    return [float(beta.ppf(.025, successes, trials-successes+1)) if successes else 0.,
            float(beta.ppf(.975, successes+1, trials-successes)) if successes < trials else 1.]


def load_results(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows, cells = [], []
    paths = [p for pattern in ['gaussian_d*_n*.json', 'nonlinear_d*_n*.json']
             for p in directory.glob(pattern)]
    for path in sorted(paths):
        cell = json.loads(path.read_text())
        if cell['fingerprint'] != manifest['fingerprint']:
            raise RuntimeError(f'Cache provenance mismatch: {path}')
        if cell['seeds'] != manifest['config']['seeds']:
            raise RuntimeError(f'Seed mismatch: {path}')
        if set(r['seed'] for r in cell['rows']) != set(cell['seeds']):
            raise RuntimeError(f'Missing seed rows: {path}')
        cells.append({k: cell[k] for k in ['kind', 'dimension', 'sample_count', 'elapsed_seconds']})
        rows.extend(dict(kind=cell['kind'], **row) for row in cell['rows'])
    keys = [tuple(r[k] for k in ['kind', 'mode', 'method', 'family', 'dimension', 'sample_count', 'seed'])
            for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError('Duplicate raw result rows')
    if any(r['value_nats'] is not None and not np.isfinite(r['value_nats']) for r in rows):
        raise RuntimeError('Nonfinite raw result')
    if manifest['status'] == 'complete':
        config = manifest['config']
        expected = {('gaussian', d, n) for d in config['dimensions'] for n in config['sample_sizes']}
        if config['nonlinear']:
            expected |= {('nonlinear', d, n) for d in config['nonlinear_dimensions']
                         for n in config['nonlinear_sample_sizes']}
        actual = {(r['kind'], r['dimension'], r['sample_count']) for r in cells}
        if expected != actual:
            raise RuntimeError('Completed manifest does not match registered grid')
    return manifest, rows, cells


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ['kind', 'mode', 'method', 'family', 'dimension', 'sample_count'])
        groups[key].append(row)
    summary = []
    for key, group in sorted(groups.items()):
        values = np.array([r['value_nats'] for r in group if r['status'] == 'ok'])
        entry = dict(zip(['kind', 'mode', 'method', 'family', 'dimension', 'sample_count'], key))
        entry.update(repetitions=len(group), returned=len(values),
                     status='ok' if len(values) == len(group) else group[0]['status'],
                     assumption_valid=all(r['assumption_valid'] for r in group),
                     reference_nats=group[0]['reference_nats'])
        if len(values):
            error = values-entry['reference_nats']
            entry.update(mean_nats=float(values.mean()), sd_nats=float(values.std(ddof=1)),
                         min_nats=float(values.min()), max_nats=float(values.max()),
                         negative_count=int((values < 0).sum()), bias_nats=float(error.mean()),
                         rmse_nats=float(np.sqrt(np.mean(error**2))),
                         absolute_error_q90_nats=float(np.quantile(abs(error), .9)))
            for criterion in ['strict', 'normalized']:
                successes = sum(r.get(criterion+'_pass', False) for r in group)
                entry[criterion+'_successes'] = successes
                entry[criterion+'_rate'] = successes / len(group)
                entry[criterion+'_interval95'] = interval(successes, len(group))
        summary.append(entry)
    return summary


def paired_results(rows):
    groups = defaultdict(dict)
    for r in rows:
        key = tuple(r[k] for k in ['kind', 'mode', 'method', 'dimension', 'sample_count'])
        groups[key][(r['seed'], r['family'])] = r
    results = []
    for key, group in sorted(groups.items()):
        kind, mode, method, d, n = key
        families = ['null', 'diagonal'] if kind == 'gaussian' else ['product_null', 'product']
        seeds = sorted({s for s, _ in group})
        selected = [[group.get((s, f)) for f in families] for s in seeds]
        if any(any(r is None for r in pair) for pair in selected):
            raise RuntimeError(f'Incomplete paired cell {key}')
        returned = all(all(r['status'] == 'ok' for r in pair) for pair in selected)
        valid = all(all(r['assumption_valid'] for r in pair) for pair in selected)
        entry = dict(kind=kind, mode=mode, method=method, dimension=d, sample_count=n,
                     trials=len(seeds), returned=returned, assumption_valid=valid,
                     status='ok' if returned else selected[0][0]['status'])
        for criterion in ['strict', 'normalized']:
            k = sum(all(r.get(criterion+'_pass', False) for r in pair) for pair in selected)
            entry[criterion+'_successes'] = k
            entry[criterion+'_rate'] = k / len(seeds)
            entry[criterion+'_interval95'] = interval(k, len(seeds))
        results.append(entry)
    return results


def boundaries(pairs, dimensions, sample_sizes):
    lookup = {(r['mode'], r['method'], r['dimension'], r['sample_count']): r
              for r in pairs if r['kind'] == 'gaussian'}
    methods = sorted({(mode, method) for mode, method, _, _ in lookup})
    results = []
    for mode, method in methods:
        for d in dimensions:
            for criterion in ['strict', 'normalized']:
                candidate = None
                tested = []
                for n in sample_sizes:
                    r = lookup.get((mode, method, d, n))
                    if r is not None:
                        tested.append(n)
                    remaining = [lookup.get((mode, method, d, larger))
                                 for larger in sample_sizes if larger >= n]
                    if candidate is None and all(x is not None and x['returned']
                            and x['assumption_valid'] and x[criterion+'_rate'] >= .9
                            for x in remaining):
                        candidate = n
                results.append(dict(mode=mode, method=method, dimension=d,
                                    criterion=criterion, candidate_minimum_n=candidate,
                                    tested_sample_sizes=tested))
    return results


def draw_maps(pairs, output, dimensions, sample_sizes, criterion='strict'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    methods = ['full_ridge', 'full_corrected', 'structured', 'ksg']
    lookup = {(r['mode'], r['method'], r['dimension'], r['sample_count']): r
              for r in pairs if r['kind'] == 'gaussian'}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'xtick.major.size': 0, 'ytick.major.size': 0}):
        fig = plt.figure(figsize=(12, 7.3), layout='constrained')
        grid = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, .04],
                               height_ratios=[1, 1, .08])
        cmap = plt.get_cmap('Blues')
        for j, mode in enumerate(['EI', 'MI']):
            for k, method in enumerate(methods):
                ax = fig.add_subplot(grid[j, k])
                for iy, d in enumerate(dimensions):
                    for ix, n in enumerate(sample_sizes):
                        r = lookup.get((mode, method, d, n))
                        face, hatch = '#e6e6e6', None
                        if r and not r['returned']:
                            face, hatch = '#fafafa', '///'
                        elif r and r['assumption_valid']:
                            face = cmap(r[criterion+'_rate'])
                        patch = Rectangle((ix-.5, iy-.5), 1, 1, facecolor=face,
                                          edgecolor='#cccccc', lw=.3, hatch=hatch)
                        ax.add_patch(patch)
                        if r and r['returned'] and r['assumption_valid'] and r[criterion+'_rate'] >= .9:
                            ax.text(ix, iy, '•', ha='center', va='center', color='white', fontsize=11)
                ax.set(xlim=(-.5, len(sample_sizes)-.5), ylim=(len(dimensions)-.5, -.5),
                       xticks=range(len(sample_sizes)), yticks=range(len(dimensions)))
                ax.set_xticklabels(sample_sizes, rotation=45, ha='right')
                ax.set_yticklabels(dimensions if k == 0 else [])
                if k == 0:
                    ax.set_ylabel(('EI: independent Gaussian input' if mode == 'EI'
                                   else 'MI: correlated Gaussian input') + '\nDimension per side, d')
                if j == 1:
                    ax.set_xlabel('Independent samples, N')
                if j == 0:
                    short = {'full_ridge': 'Full Gaussian\nridge 0.01',
                             'full_corrected': 'Full Gaussian\nanalytic correction',
                             'structured': 'Gaussian\nregistered blocks', 'ksg': 'KSG\nk = 3'}
                    ax.set_title(short[method], fontsize=10, pad=8)
                ax.text(-.03, 1.04, chr(97+j*4+k), transform=ax.transAxes,
                        fontweight='bold', fontsize=11)
        cax = fig.add_subplot(grid[:2, -1])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax,
                          ticks=[0, .25, .5, .75, 1])
        cb.set_label('Fraction passing null AND signal (20 seeds)')
        lax = fig.add_subplot(grid[2, :])
        lax.axis('off')
        handles = [Patch(facecolor='#08306b', label='White dot: at least 18/20 pass'),
                   Patch(facecolor='#fafafa', edgecolor='#cccccc', hatch='///', label='Inapplicable: N ≤ 2d'),
                   Patch(facecolor='#e6e6e6', label='Not measured')]
        lax.legend(handles=handles, loc='center', ncols=3, frameon=False, fontsize=8)
        fig.savefig(output / f'{criterion}_reliability.png', dpi=220, bbox_inches='tight')
        plt.close(fig)


def draw_null(rows, output):
    import matplotlib.pyplot as plt
    methods = ['full_ridge', 'full_raw', 'full_corrected', 'structured']
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), layout='constrained')
        for ax, d in zip(axes, [500, 1000]):
            for method in methods:
                ns, medians, lo, hi = [], [], [], []
                for n in SAMPLE_SIZES:
                    vals = [abs(r['value_nats']) for r in rows if r['kind'] == 'gaussian'
                            and r['mode'] == 'EI' and r['family'] == 'null' and r['dimension'] == d
                            and r['sample_count'] == n and r['method'] == method and r['status'] == 'ok']
                    if vals:
                        ns.append(n)
                        medians.append(np.median(vals)); lo.append(np.quantile(vals, .1)); hi.append(np.quantile(vals, .9))
                if ns:
                    ax.plot(ns, medians, color=COLORS[method], marker='o', ms=3, lw=1.2, label=LABELS[method])
                    ax.fill_between(ns, lo, hi, color=COLORS[method], alpha=.12, lw=0)
            ax.axhline(.05, color='#555555', ls='--', lw=.9)
            ax.set(xscale='log', yscale='log', xlabel='Independent samples, N',
                   ylabel='Absolute null estimate (nats)', title=f'd = {d}')
            ax.set_xticks([64, 256, 1024, 4096], ['64', '256', '1024', '4096'])
        axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, .5), frameon=False, fontsize=8)
        fig.savefig(output / 'high_dimension_null.png', dpi=220, bbox_inches='tight')
        plt.close(fig)


def draw_misspecification(rows, output):
    import matplotlib.pyplot as plt
    settings = [('gaussian', 'EI', 'dense', 1000, 4096,
                 ['full_ridge', 'full_corrected', 'structured'], 'Dense rotation; d = 1000'),
                ('nonlinear', 'common', 'product', 50, 4096,
                 ['full_corrected', 'structured', 'structured_polynomial'], 'Nonlinear products; d = 50')]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), layout='constrained')
        rng = np.random.default_rng(17)
        for ax, (kind, mode, family, d, n, methods, title) in zip(axes, settings):
            plotted = []
            for i, method in enumerate(methods):
                subset = [r for r in rows if r['kind'] == kind and r['mode'] == mode
                          and r['family'] == family and r['dimension'] == d
                          and r['sample_count'] == n and r['method'] == method and r['status'] == 'ok']
                if not subset:
                    continue
                values = np.array([r['value_nats']/r['reference_nats'] for r in subset])
                plotted.extend(values.tolist())
                ax.scatter(i+rng.uniform(-.08, .08, len(values)), values, s=12, alpha=.5,
                           color=COLORS[method], linewidth=0)
                ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt='o', color='black',
                            ms=4, lw=1, capsize=3)
            ax.axhline(1, color='#555555', ls='--', lw=.9)
            ax.axhspan(.9, 1.1, facecolor='#888888', alpha=.08)
            names = dict(full_ridge='Full Gaussian\nridge 0.01', full_corrected='Full Gaussian\ncorrected',
                         structured='Registered\nGaussian blocks', structured_polynomial='Registered pairs\npolynomial TM')
            ax.set(xticks=range(len(methods)), xticklabels=[names[m] for m in methods],
                   ylabel='Estimate / true information', title=title,
                   ylim=(min(-.08, min(plotted, default=0)-.05),
                         max(1.65, max(plotted, default=1)+.08)))
        fig.savefig(output / 'assumption_failure.png', dpi=220, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'experiments/reader_boundary/results/full_20261029')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    manifest, rows, cells = load_results(args.directory)
    pairs = paired_results(rows)
    summary = dict(manifest_status=manifest['status'], fingerprint=manifest['fingerprint'],
                   cells=len(cells), raw_rows=len(rows), cell_times=cells,
                   status_counts=dict(Counter(r['status'] for r in rows)),
                   negative_values_retained=sum(r['value_nats'] is not None and r['value_nats'] < 0 for r in rows),
                   groups=aggregate(rows), paired_cells=pairs,
                   boundaries=boundaries(pairs, manifest['config']['dimensions'], manifest['config']['sample_sizes']))
    (args.directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    if not args.no_plots:
        figures = ROOT / 'fig/reader_boundary_20261029'
        figures.mkdir(parents=True, exist_ok=True)
        for criterion in ['strict', 'normalized']:
            draw_maps(pairs, figures, manifest['config']['dimensions'], manifest['config']['sample_sizes'], criterion)
        draw_null(rows, figures)
        draw_misspecification(rows, figures)
    print(json.dumps(dict(status=manifest['status'], cells=len(cells), rows=len(rows),
                          output=str(args.directory / 'summary.json'))))


if __name__ == '__main__':
    main()
