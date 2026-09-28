#!/usr/bin/env python3
"""Aggregate independent launches without treating timed updates as repeats."""

import argparse
import json
from pathlib import Path
import statistics


def aggregate(paths, required_repeats=3):
    reports = [json.loads(Path(path).read_text()) for path in paths]
    if not reports or len({r['launch_id'] for r in reports}) != len(reports):
        raise ValueError('distinct nonempty launch identities required')
    if any(r.get('kind') != 'configuration' or r.get('status') != 'complete' for r in reports):
        raise ValueError('only complete configuration measurements can be combined')
    first = reports[0]
    ignore = {'output', 'repeat_id'}
    recipe = {k: v for k, v in first['config'].items() if k not in ignore}
    for report in reports:
        if (report['sources'] != first['sources'] or report['world_size'] != first['world_size']
                or report['GPU_executed'] != first['GPU_executed']
                or {k: v for k, v in report['config'].items() if k not in ignore} != recipe):
            raise ValueError('source, world, device or experiment recipe changed across launches')
        signature = lambda r: sorted((x['device'], x['torch'], x['cuda']) for x in r['hardware'])
        if signature(report) != signature(first):
            raise ValueError('hardware/software type changed across launches')
    groups = {}
    for report in reports:
        seen = set()
        for row in report['records']:
            key = json.dumps(row['case'], sort_keys=True)
            if key in seen:
                raise ValueError('duplicate configuration inside a launch')
            seen.add(key)
            groups.setdefault(key, []).append((report['launch_id'], row))
    rows, costs = [], []
    for key, samples in sorted(groups.items()):
        case = json.loads(key)
        times = [r['launch_mean_seconds'] for _, r in samples if r['status'] == 'measured']
        entry = dict(case=case, independent_launches=len(times),
                     launch_ids=[uid for uid, r in samples if r['status'] == 'measured'],
                     rejected_launches=[uid for uid, r in samples if r['status'] != 'measured'],
                     complete=len(times) >= required_repeats,
                     median_seconds=statistics.median(times) if times else None,
                     min_seconds=min(times) if times else None, max_seconds=max(times) if times else None,
                     stdev_seconds=statistics.stdev(times) if len(times) > 1 else None,
                     raw_launch_means=times)
        rows.append(entry)
        if entry['complete']:
            costs.append({**case, 'seconds': entry['median_seconds']})
    return dict(status='complete' if all(r['complete'] or not r['independent_launches'] for r in rows) else 'incomplete',
                required_independent_launches=required_repeats, GPU_executed=first['GPU_executed'],
                world_size=first['world_size'],
                updates_treated_as_independent_runs=False, recipe=recipe, sources=first['sources'],
                configurations=rows, costs=costs,
                cost_scope='synthetic flat-master timings; validate planner choices in Qwen')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--required-repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.required_repeats < 1:
        parser.error('positive repeat requirement and fresh output required')
    result = aggregate(args.reports, args.required_repeats)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
