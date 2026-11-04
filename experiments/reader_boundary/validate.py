"""Independent frontier checks registered before seeing the main scan results."""
import hashlib
import json
import time
from pathlib import Path

from analyze import aggregate, interval, paired_results
from methods import ROOT, block_gaussian_reader, full_gaussian_readers, source_pair
from run import dump, observation


def main():
    directory = ROOT / 'experiments/reader_boundary/results/full_20261029'
    manifest = json.loads((directory / 'manifest.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    if manifest['status'] != 'complete' or summary['fingerprint'] != manifest['fingerprint']:
        raise RuntimeError('Finish and summarize the main scan before selecting frontier checks')
    ns = manifest['config']['sample_sizes']
    selected = []
    for boundary in summary['boundaries']:
        if (boundary['mode'] in ['EI', 'MI']
                and boundary['method'] in ['full_corrected', 'structured']
                and boundary['dimension'] in [100, 500, 1000]
                and boundary['criterion'] == 'strict'
                and boundary['candidate_minimum_n'] is not None):
            frontier = boundary['candidate_minimum_n']
            i = ns.index(frontier)
            for n in ns[i:i+2]:
                selected.append(dict(mode=boundary['mode'], method=boundary['method'],
                                     dimension=boundary['dimension'], sample_count=n,
                                     is_selected_frontier=n == frontier))
    plan = dict(main_fingerprint=manifest['fingerprint'], selected=selected,
                seeds=list(range(1000, 1100)), criterion='strict',
                scope='Selection on 20 seeds; independent validation on 100 new seeds per point.',
                confidence_rule='Individual 95% two-sided binomial lower bound >= 0.90; no simultaneous claim.')
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()
    plan['fingerprint'] = fingerprint
    dump(directory / 'validation_plan.json', plan)
    start = time.monotonic()
    remaining = 1800-manifest['elapsed_seconds']
    rows, checks = [], []
    for condition in selected:
        mode, method, d, n = (condition[k] for k in ['mode', 'method', 'dimension', 'sample_count'])
        path = directory / f'validation_{mode}_{method}_d{d}_n{n}.json'
        if path.exists():
            cache = json.loads(path.read_text())
            if cache['fingerprint'] != fingerprint:
                raise RuntimeError(f'Validation cache mismatch: {path}')
            rows.extend(cache['rows']); checks.append(condition)
            continue
        if time.monotonic()-start >= remaining:
            break
        point_rows = []
        for seed in plan['seeds']:
            intervention, observed, noise = source_pair(n, d, seed)
            x = intervention if mode == 'EI' else observed
            for family in ['null', 'diagonal']:
                y = noise if family == 'null' else x+noise
                if method == 'full_corrected':
                    value = full_gaussian_readers(x, y)[method]
                else:
                    value = block_gaussian_reader(x, y, 1 if mode == 'EI' else 5)
                point_rows.append(dict(kind='gaussian', **observation(seed, d, n, mode, family, method, value)))
        dump(path, dict(fingerprint=fingerprint, condition=condition, rows=point_rows))
        rows.extend(point_rows); checks.append(condition)
        print(json.dumps(dict(event='validation_point', **condition, completed=len(checks), total=len(selected))), flush=True)
    pairs = paired_results(rows)
    for point in pairs:
        point['individual_lower_bound_at_least_90pct'] = point['strict_interval95'][0] >= .9
    result = dict(plan=plan, completed_conditions=checks, elapsed_seconds=time.monotonic()-start,
                  total_experiment_seconds=manifest['elapsed_seconds']+time.monotonic()-start,
                  status='complete' if len(checks) == len(selected) else 'budget_reached',
                  paired_cells=pairs, groups=aggregate(rows))
    dump(directory / 'validation_summary.json', result)
    print(json.dumps(dict(event='validation_complete', status=result['status'],
                          points=len(checks), seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
