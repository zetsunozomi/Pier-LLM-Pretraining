#!/usr/bin/env python3
"""Summarize paired strict/relaxed outer timings; raw samples remain untouched."""

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(directory):
    directory = Path(directory)
    gate = json.loads((directory / 'correctness.json').read_text())
    report = json.loads((directory / 'benchmark.json').read_text())
    if gate['status'] != 'passed' or gate['schedule'] != 'contiguous':
        raise ValueError('contiguous correctness gate did not pass')
    if gate['GPU_executed'] != report['GPU_executed']:
        raise ValueError('correctness and timing devices differ')
    if gate['ranks'] != report['world_size'] // report['config']['tp']:
        raise ValueError('correctness gate did not cover the benchmark learner count')
    source = report['sources']['megatron/core/outer_sync/executor.py']
    if gate['production_executor_sha256'] != source:
        raise ValueError('executor changed between correctness gate and benchmark')
    expected = {'reference', 'contiguous', 'recenter'}
    grouped = {arm: [] for arm in expected}
    repeats = {}
    for record in report['records']:
        arm, repeat = record['arm'], record['repeat']
        if arm not in expected or arm in repeats.setdefault(repeat, {}):
            raise ValueError('unexpected or duplicate arm in paired benchmark')
        ranks = record['rank_records']
        if len(ranks) != report['world_size'] or len({r['rank'] for r in ranks}) != len(ranks):
            raise ValueError('incomplete rank coverage')
        samples = report['config']['samples']
        if any(len(r['seconds']) != samples or not r['finite_state'] for r in ranks):
            raise ValueError('missing samples or nonfinite outer state')
        maxima = [max(r['seconds'][i] for r in ranks) for i in range(samples)]
        if maxima != record['max_rank_seconds'] or not all(math.isfinite(x) and x > 0 for x in maxima):
            raise ValueError('invalid maximum-rank timing samples')
        median = statistics.median(maxima)
        repeats[repeat][arm] = median
        grouped[arm].append(record)
    if len(repeats) != report['config']['repeats'] or any(set(r) != expected for r in repeats.values()):
        raise ValueError('incomplete paired benchmark')
    arms = {}
    for arm, records in grouped.items():
        peaks = [r['peak_allocated_bytes'] for c in records for r in c['rank_records']]
        if report['GPU_executed'] and any(x is None for x in peaks):
            raise ValueError('missing CUDA memory peak')
        arms[arm] = {
            'outer_plus_commit_seconds': statistics.median(r[arm] for r in repeats.values()),
            'peak_allocated_gib': max(peaks) / 2**30 if all(x is not None for x in peaks) else None,
        }
    comparisons = []
    for repeat, times in sorted(repeats.items()):
        old, new, relaxed = (times[a] for a in ('reference', 'contiguous', 'recenter'))
        comparisons.append({
            'repeat': repeat,
            'new_vs_old_latency_reduction_pct': 100 * (1 - new / old),
            'old_vs_relaxed_latency_overhead_pct': 100 * (old / relaxed - 1),
            'new_vs_relaxed_latency_overhead_pct': 100 * (new / relaxed - 1),
        })
    return {
        'status': 'measured' if report['GPU_executed'] else 'cpu_smoke_only',
        'GPU_executed': report['GPU_executed'],
        'scope': report['scope'], 'config': report['config'],
        'statistic': 'median of per-repeat medians of maximum-rank outer+commit samples',
        'repeat_scope': report['repeat_scope'], 'by_arm': arms, 'comparisons': comparisons,
        'production_executor_sha256': source,
        'not_measured': ['Qwen parameter fragmentation and full training', 'convergence'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summary = summarize(args.directory)
    (args.directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = [f"Status: {summary['status']}", summary['scope'],
             'Arm                      Outer+commit (s)   Peak allocated (GiB)']
    for arm in ('reference', 'contiguous', 'recenter'):
        row = summary['by_arm'][arm]
        peak = row['peak_allocated_gib']
        lines.append(f"{arm:<25}{row['outer_plus_commit_seconds']:>14.6f}   "
                     + (f'{peak:.3f}' if peak is not None else 'N/A'))
    for row in summary['comparisons']:
        lines.append(f"Repeat {row['repeat']}: new Pier latency reduction vs old "
                     f"{row['new_vs_old_latency_reduction_pct']:+.2f}%; "
                     f"overhead vs W: old {row['old_vs_relaxed_latency_overhead_pct']:+.2f}%, "
                     f"new {row['new_vs_relaxed_latency_overhead_pct']:+.2f}%")
    lines.append('Flat-master experiment; use Qwen runs before replacing paper results.')
    text = '\n'.join(lines) + '\n'
    (args.directory / 'results.txt').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
