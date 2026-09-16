#!/usr/bin/env python3
"""Fail closed when any rank, phase, or expected evidence is missing."""

import argparse
import json
from pathlib import Path

from manifest import source_hashes


def summarize(output, launcher_exit):
    errors = []

    def read(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            errors.append(f'{path.name}: {exc}')
            return {}

    manifest = read(output / 'manifest.json')
    world = manifest.get('nodes', 0) * manifest.get('gpus_per_node', 0)
    if launcher_exit:
        errors.append(f'launcher failed with exit code {launcher_exit}; inspect phase logs')
    if world < 4 or world & (world - 1):
        errors.append('expected a power-of-two GPU allocation of at least four GPUs')
    if manifest.get('source_sha256') != source_hashes():
        errors.append('source hashes changed or are missing; do not mix results across revisions')
    for node in range(manifest.get('nodes', 0)):
        info = read(output / f'node-{node}.json')
        if not info.get('cuda_available') or info.get('visible_devices', 0) < manifest['gpus_per_node']:
            errors.append(f'node {node}: expected CUDA allocation unavailable')
    for inner in (1, 2):
        for rank in range(world):
            report = read(output / f'megatron-dp{inner}' / f'rank-{rank}.json')
            if (report.get('status') != 'passed' or not report.get('GPU_executed')
                    or report.get('rank') != rank or report.get('world_size') != world
                    or report.get('inner_dp_size') != inner):
                errors.append(f'Megatron inner-DP{inner} rank {rank}: missing/failed/mismatched evidence')
            norms, updates = report.get('normalization', []), report.get('outer_update', [])
            combinations = {(False, False), (False, True), (True, False), (True, True)}
            if (len(norms) != 4 or not all(x.get('inner_and_warmup_gradient_bits_match') for x in norms)
                    or {(x.get('inner_average'), x.get('collective_average')) for x in norms} != combinations):
                errors.append(f'Megatron inner-DP{inner} rank {rank}: incomplete normalization checks')
            required = ('missing_copy_negative_control_observed', 'master_model_next_forward_bits_match',
                        'inner_moments_retained', 'local_optimizer_restore')
            if (len(updates) != 4 or not all(all(x.get(k) for k in required) for x in updates)
                    or {(x.get('outer_shard'), x.get('outer_cpu_offload')) for x in updates} != combinations):
                errors.append(f'Megatron inner-DP{inner} rank {rank}: incomplete outer/consumer checks')
    protocol = read(output / 'protocol.json')
    if (protocol.get('status') != 'passed' or protocol.get('ranks') != world
            or not protocol.get('GPU_executed') or protocol.get('backend') != 'nccl'):
        errors.append('ordered protocol: no complete CUDA/NCCL evidence')
    expected_per_rank = world.bit_length() * 4 * 2 * 2
    operators = protocol.get('records', {}).get('operator', [])
    training = protocol.get('records', {}).get('training', [])
    if len(operators) != world * expected_per_rank or len(training) != world * world.bit_length():
        errors.append('ordered protocol: incomplete operator/training matrix')
    expected_operators = {(rank, 1 << level, n, tile, tier)
                          for rank in range(world) for level in range(world.bit_length())
                          for n in (1, 5, 17, 64) for tile in (1, 7) for tier in ('device', 'host')}
    observed_operators = {(r.get('rank'), r.get('s'), r.get('N'),
                           r.get('tile_elements_per_momentum_owner'), r.get('state_tier'))
                          for r in operators}
    if (observed_operators != expected_operators
            or not all(r.get('bitwise_states_and_model') and r.get('fixed_workspace_storages')
                       for r in operators)):
        errors.append('ordered protocol: missing cases or state/storage verification')
    if ({(r.get('rank'), r.get('s')) for r in training}
            != {(rank, 1 << level) for rank in range(world) for level in range(world.bit_length())}
            or not all(r.get('next_forward_and_inner_moments_bitwise')
                       and r.get('drained_boundary_executor_restore') for r in training)):
        errors.append('ordered protocol: missing training/consumer/restore verification')
    for name, digest in protocol.get('source_sha256', {}).items():
        relative = 'experiments/centered_outer/reference/' + name
        if manifest.get('source_sha256', {}).get(relative) != digest:
            errors.append(f'ordered protocol source mismatch: {name}')
    if set(protocol.get('source_sha256', {})) != {'executor.py', 'verify_executor.py'}:
        errors.append('ordered protocol source hashes missing')
    return {'status': 'passed' if not errors else 'failed', 'stage': 'E0a',
            'GPU_executed': not errors, 'performance_result': False,
            'world_size': world, 'errors': errors,
            'next_step': 'Review evidence before production coordinate adapter and performance work.',
            'not_validated': manifest.get('not_validated', [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--launcher-exit', type=int, default=0)
    args = parser.parse_args()
    result = summarize(args.output_dir, args.launcher_exit)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
