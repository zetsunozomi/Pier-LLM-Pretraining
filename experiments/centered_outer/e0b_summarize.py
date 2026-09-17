#!/usr/bin/env python3
"""Verify complete real-training traces, across cohorts and checkpoint restart."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from e0b_config import CASES, training_args
from manifest import source_hashes


def comparable(events):
    return [{k: v for k, v in row.items() if k != 'payload'} for row in events]


def summarize(output, launcher_exit=0):
    errors, reports = [], {}

    def read(path):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError('expected a JSON object')
            return value
        except (OSError, ValueError) as exc:
            errors.append(f'{path.relative_to(output)}: {exc}')
            return {}

    manifest = read(output / 'manifest.json')
    if manifest.get('nodes') != 1 or manifest.get('gpus_per_node') != 4:
        errors.append('expected one node with four GPUs')
    if manifest.get('source_sha256') != source_hashes():
        errors.append('run sources differ from summary sources')
    node = read(output / 'node-0.json')
    if not node.get('cuda_available') or node.get('visible_devices') != 4:
        errors.append('missing four-GPU node preflight')
    if launcher_exit:
        errors.append(f'launcher failed with exit code {launcher_exit}')
    # Use the cluster path recorded in argv when revalidating a pulled archive
    # on a machine with a different repository path.
    run_output = output
    recorded = read(output / 'case-tp1-s1/launch-rank-0.json').get('argv', [])
    try:
        run_output = Path(recorded[recorded.index('--outer-trace-dir') + 1]).parent
    except (ValueError, IndexError, TypeError):
        errors.append('missing original output path in launch record')
    for case, config in CASES.items():
        directory = output / f'case-{case}'
        reports[case] = []
        if list(directory.glob('failure-rank-*.json')):
            errors.append(f'{case}: worker failure report present')
        for rank in range(4):
            row = read(directory / f'rank-{rank}.json')
            launch = read(directory / f'launch-rank-{rank}.json')
            reports[case].append(row)
            end = config.get('stop', 11)
            success = end - 1
            prefix = f'{case} rank {rank}'
            if (launch.get('entrypoint') != 'pretrain_gpt.py' or launch.get('case') != case
                    or launch.get('rank') != rank or launch.get('argv') != training_args(case, run_output)
                    or not launch.get('GPU_executed')):
                errors.append(f'{prefix}: missing real-entrypoint launch record')
            if (row.get('status') != ('partial' if case == 'split' else 'passed')
                    or row.get('rank') != rank or row.get('world_size') != 4
                    or not row.get('GPU_executed') or row.get('backend') != 'nccl'
                    or row.get('clock') != dict(interval=3, attempted=end, successful=success, boundaries=success // 3)
                    or row.get('pending_consumer') or row.get('consumer_checks', 0) < success // 3
                    or row.get('cohort') != config['cohort']
                    or row.get('state_tier') != ('host' if config.get('host') else 'device')
                    or row.get('performance_result') is not False or row.get('error') is not None
                    or row.get('restored') != bool(config.get('resume'))):
                errors.append(f'{prefix}: incomplete training/clock/consumer/restore checks')
            if config.get('resume'):
                restored = row.get('restore_evidence') or {}
                if (restored.get('iteration') != 5 or restored.get('consumed_train_samples') != 40
                        or restored.get('clock') != dict(interval=3, attempted=5, successful=4, boundaries=1)
                        or len(restored.get('rank_file_sha256', '')) != 64
                        or restored.get('rank_file_bytes', 0) <= 0):
                    errors.append(f'{prefix}: missing verified midcycle checkpoint evidence')
            dp_ranks = list(range(rank % config['tp'], 4, config['tp']))
            dp_index = dp_ranks.index(rank)
            inner_size = config['inner']
            expected_inner = dp_ranks[(dp_index // inner_size) * inner_size:(dp_index // inner_size + 1) * inner_size]
            expected_outer = dp_ranks[dp_index % inner_size::inner_size]
            if row.get('inner_ranks') != expected_inner or row.get('outer_ranks') != expected_outer:
                errors.append(f'{prefix}: incorrect coordinate-group membership')
            events = row.get('events', [])
            if len(events) != end:
                errors.append(f'{prefix}: missing attempted-step traces')
                continue
            successful = 0
            for attempted, event in enumerate(events, 1):
                skipped = attempted == 3
                successful += int(not skipped)
                boundary = not skipped and successful % 3 == 0
                if (event.get('attempted') != attempted or event.get('successful') != successful
                        or event.get('skipped') != skipped or event.get('outer_boundary') != boundary
                        or event.get('boundaries') != successful // 3
                        or not all(isinstance(event.get(key), str) and len(event[key]) == 64
                                   for key in ('master_sha256', 'model_sha256', 'inner_sha256'))
                        or ('lm loss' not in event.get('loss', {}))
                        or not all(isinstance(v, (int, float)) and math.isfinite(v)
                                   for v in event.get('loss', {}).values())
                        or event.get('oracle_states_bitwise') != (True if boundary else None)
                        or event.get('outer_retained_inner_state') != (True if boundary else None)
                        or event.get('skip_retained_optimizer') != (True if skipped else None)
                        or (event.get('payload') is not None) != boundary):
                    errors.append(f'{prefix}: malformed trajectory at attempt {attempted}')
    comparisons = []
    for case in ('tp1-s2', 'tp1-s4', 'tp1-host', 'resume'):
        for rank in range(4):
            baseline, actual = reports['tp1-s1'][rank], reports[case][rank]
            good = bool(baseline.get('events')) and comparable(baseline.get('events', [])) == comparable(actual.get('events', []))
            if not good:
                errors.append(f'{case} rank {rank}: loss/master/model/inner trajectory differs from s1')
            comparisons.append({'case': case, 'rank': rank, 'bitwise_trajectory_equal': good})
    for rank in range(4):
        if reports['split'][rank].get('events') != reports['resume'][rank].get('events', [])[:5]:
            errors.append(f'resume rank {rank}: checkpoint prefix differs')
    return {'status': 'passed' if not errors else 'failed', 'stage': 'E0b',
            'GPU_executed': any(r.get('GPU_executed') for rows in reports.values() for r in rows),
            'performance_result': False, 'entrypoint': 'pretrain_gpt.py',
            'world_size': 4, 'cases': list(CASES), 'comparisons': comparisons, 'errors': errors,
            'finished_utc': datetime.now(timezone.utc).isoformat(),
            'not_validated': ['Qwen conversion and real-data recipe', 'distributed optimizer',
                              'multi-slot CUDA pipeline', 'PP/CP/MoE', 'topology-changing recovery',
                              'strong baseline performance', 'GPU peak memory or physical wire traffic']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--launcher-exit', type=int, default=0)
    args = parser.parse_args()
    report = summarize(args.output_dir.resolve(), args.launcher_exit)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
