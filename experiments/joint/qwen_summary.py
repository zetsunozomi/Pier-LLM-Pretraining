#!/usr/bin/env python3
"""Paper ablation aggregation with per-boundary budgets, traffic and timings.

One completed training launch is one statistical sample. Static controls that
violate the announced next-phase budget are retained, labeled infeasible, and
excluded from the best feasible static comparison. Every original rank report
remains authoritative; this file is a derived, read-only aggregation.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qwen.n2_summary import collect as collect_n2


def boundaries(directory, expected_world, announcement=None):
    reports = [json.loads((directory / f'cycles-rank-{rank}.json').read_text()) for rank in range(expected_world)]
    rows = []
    for index, first in enumerate(reports[0]['cycles']):
        cycles = [report['cycles'][index] for report in reports]
        payloads = [cycle.get('payload') for cycle in cycles]
        joint = bool(payloads[0] and 'plan' in payloads[0])
        row = dict(round=first['outer_boundary_index'], warmup=first['warmup'], eligible=first['eligible'],
                   outer_seconds=max(c['local_outer_and_commit_seconds'] for c in cycles),
                   cycle_seconds=max(c['local_cycle_seconds'] for c in cycles),
                   peak_allocated_bytes=max(c['torch_peak_allocated_bytes'] for c in cycles),
                   peak_reserved_bytes=max(c['torch_peak_reserved_bytes'] for c in cycles))
        if joint:
            phases = payloads[0]['sent_tensor_bytes_by_phase']
            row.update(old_cohort=payloads[0]['plan']['old_cohort'], cohort=payloads[0]['plan']['cohort'],
                       configs_by_rank=[p['plan'] for p in payloads],
                       configuration_hashes=sorted({p['configuration_sha256'] for p in payloads}),
                       announcement=payloads[0]['announcement'],
                       accounted_peak_bytes=max(p['explicit_peak_bytes'] for p in payloads),
                       aggregate_sent_bytes={key: sum(p['sent_tensor_bytes_by_phase'][key] for p in payloads)
                                             for key in phases},
                       training_peak_before_outer_bytes=max(p['training_peak_before_outer_bytes'] or 0 for p in payloads),
                       trace_available=all(bool(p['trace']) for p in payloads))
            if len(row['configuration_hashes']) != 1:
                raise ValueError('joint configurations disagree across ranks')
            row['is_transition'] = any(p['plan']['old_cohort'] != p['plan']['cohort'] for p in payloads)
            row['budget_feasible'] = all(
                p['explicit_peak_bytes'] <= p['announcement']['transition_mib'] * 2**20
                and max(p['plan']['final_bytes']) <= p['announcement']['next_phase_mib'] * 2**20
                and (p['training_budget_bytes'] is None or p['training_peak_before_outer_bytes'] <= p['training_budget_bytes'])
                for p in payloads)
        else:
            current = next((b for b in reversed(announcement or []) if b['round'] <= row['round']), None)
            row['announcement'] = current
            # Conservative fixed-layout check: its entire observed cycle must
            # fit the phase budget. Never infer fit from R/M formulas alone.
            row['budget_feasible'] = current is None or row['peak_allocated_bytes'] <= current['next_phase_mib'] * 2**20
        rows.append(row)
    return rows


def campaign(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get('plan_only'):
        raise ValueError('a launch plan is not experimental evidence')
    announcements = manifest.get('budget_announcements')
    if announcements is None:
        recipe = next((c['config'] for c in manifest['planned_cases'] if c['config'] is not None), None)
        if recipe is None:
            raise ValueError('control-only campaign is missing its budget announcements')
        announcements = recipe['budgets']
    rows = []
    completed = {case['name'] for case in manifest['cases']}
    for case in manifest['planned_cases']:
        root = path.parent / case['name']
        if case['name'] not in completed or not (root / 'summary.json').exists():
            rows.append(dict(case=case['name'], status='unfinished', repeat_id=manifest['repeat_id']))
            continue
        data = collect_n2(root)
        executed = json.loads((root / 'manifest.json').read_text())
        if executed['sources'] != manifest['sources']:
            raise ValueError('case sources differ from the paired campaign')
        if executed['config'].get('joint_recipe') != case['config']:
            raise ValueError('case runtime configuration differs from the paired campaign')
        for arm in data['cases']:
            row = dict(case=case['name'], arm=arm['arm'], status=arm['status'], repeat_id=manifest['repeat_id'],
                       directory=str(root / arm['id']), GPU_executed=arm.get('GPU_executed', False))
            if arm['status'] == 'measured':
                evidence = boundaries(root / arm['id'], data['config']['world_size'], announcements)
                kept = [b for b in evidence if b['eligible']]
                row.update(run_id=arm['run_id'], boundaries=evidence,
                           tokens_per_second=arm['useful_tokens_per_second'],
                           mean_outer_seconds=arm['outer_and_commit_seconds_mean'],
                           total_outer_seconds=sum(b['outer_seconds'] for b in kept),
                           total_cycle_seconds=sum(b['cycle_seconds'] for b in kept),
                           budget_feasible=all(b['budget_feasible'] for b in evidence),
                           recipe=arm['recipe'], hardware=arm['hardware'])
            else:
                errors = '\n'.join(p.read_text() for p in (root / arm['id']).glob('failure-rank-*.json'))
                row['error'] = arm.get('error')
                if 'AdmissionError' in errors:
                    row['status'] = 'infeasible'
                # Allocation failures or exceeded measured budgets are not
                # planner infeasibility and must remain failures.
            rows.append(row)
    return manifest, rows


def merge_allocations(loaded):
    """Group explicitly split case launches without inventing co-run pairing."""
    recipes, groups, seen = {}, {}, set()
    for index, (manifest, rows) in enumerate(loaded):
        repeat = manifest['repeat_id']
        allocation = manifest.get('slurm_job_id') or f'input-{index}'
        for case in manifest['planned_cases']:
            name = case['name']
            if name in recipes and recipes[name] != case:
                raise ValueError('case configuration changed across split allocations')
            recipes[name] = case
            if (repeat, name) in seen:
                raise ValueError('duplicate case/repeat across split allocations')
            seen.add((repeat, name))
        group = groups.setdefault(repeat, [])
        group.extend({**row, 'allocation_id': str(allocation)} for row in rows)
    merged = []
    for repeat, rows in sorted(groups.items()):
        present = {row['case'] for row in rows}
        rows += [dict(case=name, repeat_id=repeat, status='unfinished')
                 for name in recipes if name not in present]
        merged.append(({**loaded[0][0], 'repeat_id': repeat,
                        'planned_cases': list(recipes.values())}, rows))
    return merged


def aggregate(paths, required_repeats=3, *, split_allocations=False):
    loaded = [campaign(path) for path in paths]
    if not loaded:
        raise ValueError('campaigns required')
    first = loaded[0][0]
    if not split_allocations and len({m['repeat_id'] for m, _ in loaded}) != len(loaded):
        raise ValueError('duplicate repeat identity; do not pool reruns as extra independent repeats')
    for manifest, _ in loaded:
        if (manifest['sources'] != first['sources'] or manifest['scenario'] != first['scenario']
                or manifest.get('budget_announcements') != first.get('budget_announcements')
                or (not split_allocations and sorted(manifest['planned_cases'], key=lambda c: c['name']) !=
                    sorted(first['planned_cases'], key=lambda c: c['name']))):
            raise ValueError('source, scenario or planned configurations changed across repeats')
    software, model_recipe = None, None
    for _, rows in loaded:
        measured = [r for r in rows if r['status'] == 'measured']
        for row in measured:
            if row['recipe'] != measured[0]['recipe'] or row['hardware'] != measured[0]['hardware']:
                raise ValueError('paired cases used different model/data identities or physical allocations')
            if model_recipe is None:
                model_recipe = row['recipe']
            elif row['recipe'] != model_recipe:
                raise ValueError('model/data recipe changed across independent launches')
            signature = sorted((h['gpu'], h['torch'], h['cuda']) for h in row['hardware'])
            if software is None:
                software = signature
            elif signature != software:
                raise ValueError('GPU/software type changed across independent launches')
    if split_allocations:
        loaded = merge_allocations(loaded)
    runs = [r for _, rows in loaded for r in rows]
    ids = [r['run_id'] for r in runs if r['status'] == 'measured']
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate training launch evidence')
    groups = {}
    for row in runs:
        groups.setdefault((row['case'], row.get('arm')), []).append(row)
    summary = []
    for (name, arm), rows in sorted(groups.items(), key=str):
        measured = [r for r in rows if r['status'] == 'measured' and r['budget_feasible']]
        times = [r['mean_outer_seconds'] for r in measured]
        entry = dict(case=name, arm=arm, independent_launches=len(times),
                     excluded=[dict(repeat_id=r['repeat_id'], status=r['status'],
                                    budget_feasible=r.get('budget_feasible')) for r in rows if r not in measured],
                     median_outer_seconds=statistics.median(times) if times else None,
                     min_outer_seconds=min(times) if times else None, max_outer_seconds=max(times) if times else None,
                     stdev_outer_seconds=statistics.stdev(times) if len(times) > 1 else None,
                     median_tokens_per_second=statistics.median(r['tokens_per_second'] for r in measured) if times else None)
        for metric in ('allocated', 'reserved'):
            key = f'peak_{metric}_bytes'
            peaks = [max(b[key] for b in r['boundaries'] if b['eligible']) / 2**30
                     for r in measured if all(key in b for b in r['boundaries'])]
            entry[f'max_peak_{metric}_gib'] = max(peaks) if peaks else None
            entry[f'median_peak_{metric}_gib'] = statistics.median(peaks) if peaks else None
        summary.append(entry)
    comparisons = []
    for manifest, rows in loaded:
        valid = [r for r in rows if r['status'] == 'measured' and r['budget_feasible']]
        joint = next((r for r in valid if r['case'] == 'joint-s1'), None)
        separate = next((r for r in valid if r['case'] == 'separate-s1'), None)
        static = [r for r in valid if r['case'].startswith(('single-', 'pipeline-', 'reference-'))]
        w = next((r for r in valid if r['arm'] == 'W'), None)
        if joint:
            for label, baseline in (('separate', separate), ('W', w),
                                    ('best_feasible_static', min(static, key=lambda r: r['total_outer_seconds']) if static else None)):
                if baseline:
                    cumulative, series = 0., []
                    if len(joint['boundaries']) != len(baseline['boundaries']):
                        raise ValueError('paired boundary sequences have different lengths')
                    for a, b in zip(joint['boundaries'], baseline['boundaries']):
                        if (a['round'] != b['round'] or a['eligible'] != b['eligible']
                                or a['warmup'] != b['warmup']):
                            raise ValueError('paired boundary sequences differ')
                        if a['eligible']:
                            cumulative += b['outer_seconds'] - a['outer_seconds']
                            series.append(dict(round=a['round'], cumulative_outer_seconds_saved=cumulative,
                                               joint_is_transition=a.get('is_transition', False)))
                    comparisons.append(dict(repeat_id=manifest['repeat_id'], baseline=label,
                                             baseline_case=baseline['case'],
                                             comparison_basis=('same-allocation paired' if
                                                 joint.get('allocation_id') == baseline.get('allocation_id')
                                                 and joint['hardware'] == baseline['hardware'] else
                                                 'different-allocation unpaired; matched recipe and boundary sequence'),
                                             outer_latency_reduction_pct=100 * (1 - joint['total_outer_seconds'] / baseline['total_outer_seconds']),
                                             cycle_time_reduction_pct=100 * (1 - joint['total_cycle_seconds'] / baseline['total_cycle_seconds']),
                                             cumulative_savings=series))
    incomplete = any(r['status'] not in ('measured', 'infeasible') for r in runs)
    for entry in summary:
        if entry['case'].startswith(('joint-', 'separate-')) and entry['independent_launches'] < required_repeats:
            incomplete = True
    for prefix in ('single-', 'pipeline-', 'reference-'):
        matching = [r for r in summary if r['case'].startswith(prefix)]
        if matching and not any(r['independent_launches'] >= required_repeats for r in matching):
            incomplete = True
    missing_rows = [variant for variant in ('single', 'pipeline', 'separate', 'joint')
                    if not any(r['case'].startswith(variant + '-') and r['independent_launches'] >= required_repeats
                               for r in summary)]
    if split_allocations and missing_rows:
        incomplete = True
    return dict(status='incomplete' if incomplete or len(loaded) < required_repeats else 'complete',
                scenario=first['scenario'], required_independent_launches=required_repeats,
                split_allocations=split_allocations, missing_table_vi_rows=missing_rows,
                cycles_treated_as_independent_runs=False, runs=runs, by_case=summary, comparisons=comparisons,
                limitations=['synthetic-token execution, not model quality', 'logical API payload, not physical link counters',
                             'PyTorch allocated/reserved memory, not NVML total device memory'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaigns', nargs='+', type=Path)
    parser.add_argument('--required-repeats', type=int, default=3)
    parser.add_argument('--split-allocations', action='store_true',
                        help='Combine per-case jobs; label cross-allocation comparisons as unpaired')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.required_repeats < 1:
        parser.error('positive repeat count and fresh output required')
    result = aggregate(args.campaigns, args.required_repeats, split_allocations=args.split_allocations)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
