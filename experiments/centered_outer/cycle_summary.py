#!/usr/bin/env python3
"""Validate raw all-rank cycle reports and produce one throughput sample per run.

This validates measurement bookkeeping, not the workload, job exit, numerical
gate, tuning fairness, hardware identity or a paper-ready paired comparison.
Input reports are read-only; neither CPU fixtures nor raw GPU timings become
performance evidence just by passing this collector.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from megatron.core.outer_sync.cycle_metrics import (
    EXCLUDED_FROM_CYCLE, FORMAT, TIMING_SCOPE, TOKEN_SEMANTICS,
)


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f'{label}: expected integer >= {minimum}')
    return value


def number(value, label, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f'{label}: invalid finite duration/value')
    return value


def equal_number(actual, expected, label):
    number(actual, label)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f'{label}: recomputed value differs')


def clock(value, label):
    require(isinstance(value, dict) and set(value) == {'interval', 'attempted', 'successful', 'boundaries'},
            f'{label}: missing clock schema')
    for field in value:
        integer(value[field], f'{label}.{field}', minimum=1 if field == 'interval' else 0)
    require(value['successful'] <= value['attempted']
            and value['boundaries'] == value['successful'] // value['interval'], f'{label}: inconsistent clock')
    return value


def validate_reports(reports, *, allow_cpu=False):
    require(isinstance(reports, list) and reports, 'missing rank reports')
    require(all(isinstance(report, dict) for report in reports), 'rank reports must be objects')
    first = reports[0]
    world = integer(first.get('world_size'), 'world_size', 1)
    require(len(reports) == world, 'missing or extra rank reports')
    ranks = [integer(report.get('rank'), 'rank') for report in reports]
    require(sorted(ranks) == list(range(world)), 'duplicate or missing rank identity')
    reports = sorted(reports, key=lambda report: report['rank'])
    first = reports[0]
    try:
        require(str(uuid.UUID(first['run_id'])) == first['run_id'], 'invalid run UUID')
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise EvidenceError('missing/invalid run UUID') from exc
    gpu = first.get('GPU_executed')
    require(type(gpu) is bool and (gpu or allow_cpu), 'GPU reports required; CPU fixtures need explicit --allow-cpu')
    initial = clock(first.get('initial_clock'), 'initial_clock')
    final = clock(first.get('final_clock'), 'final_clock')
    interval = initial['interval']
    require(final['interval'] == interval, 'outer interval changed during run')
    budget = integer(first.get('planned_attempts'), 'planned_attempts', 1)
    require(budget <= 500 and final['attempted'] == budget, 'run did not finish its declared <=500 attempted-step budget')
    require(initial['attempted'] < final['attempted'] and initial['successful'] <= final['successful'],
            'invalid/empty measured step window')
    warmup = integer(first.get('warmup_cycles'), 'warmup_cycles')
    require(first.get('format') == FORMAT and first.get('token_semantics') == TOKEN_SEMANTICS
            and first.get('timing_scope') == TIMING_SCOPE
            and first.get('excluded_from_cycle') == EXCLUDED_FROM_CYCLE, 'unsupported measurement contract')
    shared = ('format', 'run_id', 'world_size', 'GPU_executed', 'planned_attempts', 'initial_clock',
              'final_clock', 'warmup_cycles', 'token_semantics', 'timing_scope', 'excluded_from_cycle')
    for report in reports:
        require(report.get('status') == 'complete' and report.get('performance_result') is False,
                f"rank {report['rank']}: incomplete or mislabelled raw report")
        require(all(type(report.get(key)) is type(first[key]) and report[key] == first[key] for key in shared),
                'mixed run/clock/measurement identities')
        clock(report['initial_clock'], 'rank initial_clock')
        clock(report['final_clock'], 'rank final_clock')
        require(isinstance(report.get('metadata'), dict), 'missing rank metadata')
        require(isinstance(report.get('cycles'), list), 'missing cycle list')
    rows = first['cycles']
    require(rows and all(len(report['cycles']) == len(rows) for report in reports), 'incomplete per-rank cycle coverage')
    # Arm/config fields must agree whenever the training adapter supplies them.
    # Coordinate fingerprints/group membership may differ legitimately with TP.
    for key in ('arm', 'cohort', 'tile_elements'):
        require(all(report['metadata'].get(key) == first['metadata'].get(key) for report in reports),
                f'mixed rank configuration: {key}')
    end_a, end_s, complete_count, eligible_count = initial['attempted'], initial['successful'], 0, 0
    validated = []
    common_fields = ('outer_boundary_index', 'complete_cycle', 'warmup', 'eligible', 'attempted_start',
                     'attempted_end', 'successful_start', 'successful_end', 'attempts', 'successful_steps',
                     'starts_midcycle', 'ends_at_outer_boundary', 'max_rank_cycle_seconds',
                     'max_rank_outer_and_commit_seconds', 'processed_loss_tokens_global',
                     'successful_loss_tokens_global', 'skipped_loss_tokens_global', 'useful_tokens_per_second')
    for index, row in enumerate(rows):
        prefix = f'cycle {index}'
        require(isinstance(row, dict), f'{prefix}: expected object')
        for key in ('outer_boundary_index', 'attempted_start', 'attempted_end', 'successful_start',
                    'successful_end', 'attempts', 'successful_steps', 'processed_loss_tokens_global',
                    'successful_loss_tokens_global', 'skipped_loss_tokens_global'):
            integer(row.get(key), f'{prefix}.{key}')
        for key in ('complete_cycle', 'warmup', 'eligible', 'starts_midcycle', 'ends_at_outer_boundary'):
            require(type(row.get(key)) is bool, f'{prefix}.{key}: expected bool')
        require((row['attempted_start'], row['successful_start']) == (end_a, end_s),
                f'{prefix}: noncontiguous/duplicated step window')
        attempts, successes = row['attempted_end'] - end_a, row['successful_end'] - end_s
        require(attempts == row['attempts'] and attempts > 0 and successes == row['successful_steps']
                and 0 <= successes <= attempts, f'{prefix}: inconsistent attempted/successful steps')
        midcycle = end_s % interval != 0
        boundary = row['ends_at_outer_boundary']
        remaining = interval - end_s % interval
        require(successes == remaining if boundary else successes < remaining,
                f'{prefix}: missing or spurious outer boundary')
        require(boundary or index == len(rows) - 1, f'{prefix}: nonterminal partial cycle')
        require(row['outer_boundary_index'] == row['successful_end'] // interval,
                f'{prefix}: outer boundary index differs')
        complete = boundary and not midcycle
        complete_count += int(complete)
        is_warmup = complete and complete_count <= warmup
        processed, useful = row['processed_loss_tokens_global'], row['successful_loss_tokens_global']
        require(0 <= useful <= processed and row['skipped_loss_tokens_global'] == processed - useful,
                f'{prefix}: inconsistent token counts')
        require(successes > 0 or useful == 0, f'{prefix}: skipped updates contributed useful tokens')
        require(attempts != successes or processed == useful, f'{prefix}: token loss without skipped steps')
        eligible = complete and not is_warmup and useful > 0
        require((row['starts_midcycle'], row['complete_cycle'], row['warmup'], row['eligible'])
                == (midcycle, complete, is_warmup, eligible), f'{prefix}: incorrect warmup/partial/eligibility labels')
        local_times, local_outer, allocations, reservations = [], [], [], []
        for report in reports:
            other = report['cycles'][index]
            require(isinstance(other, dict) and all(key in other for key in common_fields),
                    f'{prefix}: incomplete rank cycle schema')
            require(all(type(other[key]) is type(row[key]) and other[key] == row[key] for key in common_fields),
                    f'{prefix}: ranks disagree on cycle evidence')
            seconds = number(other.get('local_cycle_seconds'), f'{prefix} local time', positive=True)
            outer = number(other.get('local_outer_and_commit_seconds'), f'{prefix} local outer time')
            require(outer <= seconds and (boundary or outer == 0), f'{prefix}: invalid outer/commit duration')
            local_times.append(seconds)
            local_outer.append(outer)
            require(other.get('device_level_peak_bytes') is None and other.get('physical_wire_bytes') is None,
                    'v1 cycle meter does not measure device-level memory or physical wire bytes')
            allocated, reserved = other.get('torch_peak_allocated_bytes'), other.get('torch_peak_reserved_bytes')
            if gpu:
                integer(allocated, 'torch allocated bytes')
                integer(reserved, 'torch reserved bytes')
                require(reserved >= allocated, 'reserved peak below allocated peak')
                allocations.append(allocated)
                reservations.append(reserved)
            else:
                require(allocated is None and reserved is None, 'CPU report claims CUDA allocator peaks')
        seconds, outer = max(local_times), max(local_outer)
        equal_number(row['max_rank_cycle_seconds'], seconds, f'{prefix} slowest rank duration')
        equal_number(row['max_rank_outer_and_commit_seconds'], outer, f'{prefix} outer rank maximum')
        if eligible:
            equal_number(row['useful_tokens_per_second'], useful / seconds, f'{prefix} throughput')
        else:
            require(row['useful_tokens_per_second'] is None, f'{prefix}: excluded cycle has throughput')
        eligible_count += int(eligible)
        exclusion = (None if eligible else 'warmup' if is_warmup else 'resumed_prefix' if midcycle
                     else 'trailing_partial' if not boundary else 'zero_useful_tokens')
        validated.append({'index': index, 'eligible': eligible, 'exclusion': exclusion,
                          'attempted_start': row['attempted_start'], 'attempted_end': row['attempted_end'],
                          'processed_loss_tokens_global': processed, 'successful_loss_tokens_global': useful,
                          'max_rank_cycle_seconds': seconds, 'max_rank_outer_and_commit_seconds': outer,
                          'useful_tokens_per_second': useful / seconds if eligible else None,
                          'max_rank_torch_peak_allocated_bytes': max(allocations) if allocations else None,
                          'max_rank_torch_peak_reserved_bytes': max(reservations) if reservations else None})
        end_a, end_s = row['attempted_end'], row['successful_end']
    require((end_a, end_s) == (final['attempted'], final['successful']), 'cycle coverage does not reach final clock')
    for report in reports:
        require(type(report.get('complete_cycles')) is int and report['complete_cycles'] == complete_count
                and type(report.get('eligible_cycles')) is int and report['eligible_cycles'] == eligible_count,
                'cycle count summary differs from records')
    kept = [row for row in validated if row['eligible']]
    require(kept, 'no eligible complete post-warmup cycle')
    useful = sum(row['successful_loss_tokens_global'] for row in kept)
    seconds = sum(row['max_rank_cycle_seconds'] for row in kept)
    outer = [row['max_rank_outer_and_commit_seconds'] for row in kept]
    return {'status': 'raw_cycles_validated', 'run_id': first['run_id'], 'GPU_executed': gpu,
            'world_size': world, 'planned_attempts': budget, 'initial_clock': initial, 'final_clock': final,
            'measurement_contract': {'token_semantics': TOKEN_SEMANTICS, 'timing_scope': TIMING_SCOPE,
                                     'excluded_from_cycle': EXCLUDED_FROM_CYCLE},
            'complete_cycles': complete_count, 'eligible_cycles': eligible_count, 'cycles': validated,
            'rank_metadata': [report['metadata'] for report in reports],
            'run_sample': {'successful_loss_tokens_global': useful, 'sum_slowest_rank_cycle_seconds': seconds,
                           'useful_tokens_per_second': useful / seconds,
                           'outer_and_commit_seconds_mean': statistics.mean(outer),
                           'outer_and_commit_seconds_median': statistics.median(outer),
                           'outer_and_commit_seconds_min': min(outer), 'outer_and_commit_seconds_max': max(outer)},
            'run_samples': 1, 'cycles_treated_as_independent_runs': False,
            'independence_validated': False, 'performance_result': False,
            'not_validated': ['model/data/source and hardware identity', 'launcher success and total GPU-hours',
                              'GPU numerical training gate', 'equal tuning and independent paired runs',
                              'device-level/node host/pinned peaks and physical link traffic']}


def parse_json(data):
    def constant(value):
        raise EvidenceError(f'nonfinite JSON value: {value}')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, f'duplicate JSON key: {key}')
            value[key] = item
        return value
    return json.loads(data, parse_constant=constant, object_pairs_hook=unique)


def summarize(directory, *, allow_cpu=False):
    directory = Path(directory)
    files = sorted(directory.glob('cycles-rank-*.json'))
    require(files, f'{directory}: no cycle reports')
    require(not list(directory.glob('failure-rank-*.json')), 'worker failure report present')
    reports, inputs = [], []
    for path in files:
        match = re.fullmatch(r'cycles-rank-(0|[1-9][0-9]*)\.json', path.name)
        require(match is not None, f'invalid rank filename: {path.name}')
        data = path.read_bytes()
        report = parse_json(data)
        require(isinstance(report, dict) and type(report.get('rank')) is int
                and report['rank'] == int(match[1]), 'rank filename disagrees with contents')
        reports.append(report)
        inputs.append({'name': path.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    result = validate_reports(reports, allow_cpu=allow_cpu)
    result['input_reports'] = inputs
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-cpu', action='store_true', help='explicit fixture review, never GPU evidence')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite an existing output file')
    results, errors, identities = [], [], set()
    for directory in args.directories:
        try:
            result = summarize(directory, allow_cpu=args.allow_cpu)
            require(result['run_id'] not in identities, 'same run UUID supplied more than once')
            identities.add(result['run_id'])
            results.append({'directory': str(directory), **result})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({'directory': str(directory), 'error': str(exc)})
    output = {'format': 'pier-cycle-collection-v1', 'status': 'failed' if errors else 'raw_cycles_validated',
              'runs': results, 'errors': errors, 'performance_result': False,
              'run_samples': len(results), 'independence_validated': False,
              'paired_comparison': None, 'confidence_interval': None}
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
