"""Combine completed N2 runs, retaining failures and one sample per launch."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.centered_outer.cycle_summary import summarize, require
from experiments.qwen.n2_config import training_args


def expected_argv(output, manifest, case):
    """New runs carry a frozen launch plan; support the two original N2 logs."""
    if 'training_argv' in manifest:
        return manifest['training_argv'][case['id']]
    cfg = dict(manifest['config'])
    if 'log_interval' not in cfg:
        # The original v1 manifests predate the explicit log_interval field.
        # Recover only this known 10 -> 1 change; all other arguments still match.
        first = json.loads((output / case['id'] / 'worker-rank-0.json').read_text())['argv']
        interval = first[first.index('--log-interval') + 1]
        require(interval in ('1', '10'), 'unsupported historical logging interval')
        cfg['log_interval'] = int(interval)
    directory = Path(manifest.get('output_directory', output)) / case['id']
    return training_args(cfg, case, directory)


def collect_case(output, manifest, case):
    directory = output / case['id']
    row = {**case, 'status': 'pending'}
    launch = directory / 'exit.json'
    if not launch.exists():
        if directory.exists():
            row.update(status='unfinished', error='no launcher exit receipt; running or interrupted')
        return row
    row['launcher'] = json.loads(launch.read_text())
    if row['launcher']['exit_code'] != 0:
        failures = [p.read_text() for p in directory.glob('failure-rank-*.json')]
        kind = 'CUDA OOM' if any('OutOfMemoryError' in text or 'CUDA out of memory' in text
                                for text in failures) else 'worker/launcher error'
        row.update(status='failed', failure_kind=kind,
                   error=f"{kind}, exit={row['launcher']['exit_code']}; see {case['id']}.log")
        return row
    try:
        raw = summarize(directory)
        cfg = manifest['config']
        require(raw['world_size'] == cfg['world_size'] and raw['planned_attempts'] == cfg['attempts'],
                'world/attempt budget differs from N2 recipe')
        require(raw['initial_clock'] == dict(interval=cfg['interval'], attempted=0, successful=0, boundaries=0),
                'N2 must start from the shared pretrained snapshot')
        require(raw['final_clock']['successful'] == cfg['attempts'], 'skipped updates; not a clean paired window')
        require(raw['eligible_cycles'] == cfg['measured_cycles'], 'incomplete measurement window')
        fingerprint = hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest()
        identities, hardware, losses = [], [], []
        argv = expected_argv(output, manifest, case)
        for rank, meta in enumerate(raw['rank_metadata']):
            require(meta['arm'] == case['backend'] and meta['cohort'] == case['cohort'], 'wrong backend/cohort')
            if case['backend'] == 'pier':
                require(meta['allocation'].get('schedule', 'reference') == cfg.get('pier_schedule', 'reference'),
                        'Pier execution schedule differs from the launch configuration')
            offloaded = case['arm'] in ('O', 'OS')
            naive = case['backend'] == 'cpu_offload'
            if offloaded or 'state_storage' in meta:
                storage = meta['state_storage']
                expected_device = 'cpu' if offloaded else 'cuda'
                for name in ('reference', 'momentum'):
                    require(storage[name]['device'] == expected_device, 'outer state is on the wrong device')
                    require(storage[name]['bytes'] == meta['allocation'][name + '_bytes'],
                            'outer state storage size differs from allocation receipt')
                    if offloaded:
                        require(storage[name]['pinned'] is (not naive), 'offload pinning differs from the selected implementation')
                    if naive:
                        require(storage[name]['bytes'] == meta['coordinate_numel'] * 4,
                                'naive offload must retain a full R and M replica per learner')
                if naive:
                    require(meta['allocation']['state_layout'] == 'replicated', 'naive offload state is sharded')
                    require(meta['allocation']['workspace_cap_applies'] is False,
                            'naive whole-parameter scratch must not claim a tiled workspace cap')
            health = meta['final_health']
            require(health['model_matches_master'] and health['finite_model'] and health['finite_loss'],
                    'training health check failed')
            losses.append(health['losses']['lm loss'])
            require(math.isfinite(losses[-1]), 'nonfinite final loss')
            worker = json.loads((directory / f'worker-rank-{rank}.json').read_text())
            require(worker['rank'] == rank and worker['case'] == case and worker['GPU_executed']
                    and worker['manifest_sha256'] == fingerprint
                    and worker['argv'] == argv, 'worker launch differs from recipe')
            if cfg.get('expected_gpu'):
                require(worker['gpu'] == cfg['expected_gpu'], 'GPU model differs from requested hardware')
            init = json.loads((directory / f'initialization-rank-{rank}.json').read_text())
            require(init['loaded_before_optimizer_construction'], 'missing pretrained initialization')
            identities.append(init['qwen_recipe'])
            hardware.append({key: worker[key] for key in ('hostname', 'local_rank', 'gpu', 'gpu_total_bytes',
                'python', 'torch', 'cuda', 'transformers', 'safetensors', 'tf32',
                'bf16_reduced_precision_reduction', 'nccl_algo')})
        require(all(item == identities[0] for item in identities), 'rank model/data identities disagree')
        kept = [cycle for cycle in raw['cycles'] if cycle['eligible']]
        row.update(status='measured', GPU_executed=True, run_id=raw['run_id'],
                   **raw['run_sample'], eligible_cycles=raw['eligible_cycles'],
                   peak_allocated_gib=max(c['max_rank_torch_peak_allocated_bytes'] for c in kept) / 2**30,
                   peak_reserved_gib=max(c['max_rank_torch_peak_reserved_bytes'] for c in kept) / 2**30,
                   final_loss_by_rank=losses, recipe=identities[0], hardware=hardware,
                   allocations=[m['allocation'] for m in raw['rank_metadata']],
                   tile_elements=[m['tile_elements'] for m in raw['rank_metadata']],
                   measurement_contract=raw['measurement_contract'], input_reports=raw['input_reports'])
        if case['backend'] == 'pier' and 'pier_schedule' in cfg:
            row['pier_schedule'] = cfg.get('pier_schedule', 'reference')
        if all('state_storage' in m for m in raw['rank_metadata']):
            row['state_storage_by_rank'] = [m['state_storage'] for m in raw['rank_metadata']]
            row['host_outer_state_gib_max_rank'] = max(
                sum(t['bytes'] for t in m['state_storage'].values() if t['device'] == 'cpu')
                for m in raw['rank_metadata']) / 2**30
    except (ValueError, KeyError, TypeError, OSError) as exc:
        row.update(status='invalid', error=str(exc))
    return row


def collect(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    rows = [collect_case(output, manifest, case) for case in manifest['cases']]
    sweep = manifest['config'].get('suite') == 'cohorts'
    measured = [row for row in rows if row['status'] == 'measured']
    # Paired comparisons require the same starting weights, tokens and hardware.
    if measured:
        for row in measured:
            if row['recipe'] != measured[0]['recipe'] or row['hardware'] != measured[0]['hardware']:
                row.update(status='invalid', error='model/data/hardware differs between arms')
    for row in rows:
        if row['status'] == 'measured':
            base = next((g for g in rows if g['arm'] == 'G' and g['repeat'] == row['repeat']
                         and g['status'] == 'measured'), None)
            row['speedup_vs_G'] = row['useful_tokens_per_second'] / base['useful_tokens_per_second'] if base else None
            if 'O' in manifest['config']['arms']:
                offload = next((o for o in rows if o['arm'] == 'O' and o['repeat'] == row['repeat']
                                and o['status'] == 'measured'), None)
                row['speedup_vs_O'] = (row['useful_tokens_per_second'] / offload['useful_tokens_per_second']
                                       if offload else None)
            if sweep:
                anchor = next((p for p in rows if p['arm'] == 'P' and p['cohort'] == 2
                               and p['repeat'] == row['repeat'] and p['status'] == 'measured'), None)
                row['speedup_vs_P_s2'] = (row['useful_tokens_per_second'] / anchor['useful_tokens_per_second']
                                         if anchor else None)
    groups = []
    keys = list(dict.fromkeys((case['arm'], case['cohort']) for case in manifest['cases']))
    for arm, cohort in keys:
        samples = [r for r in rows if r['arm'] == arm and r['cohort'] == cohort and r['status'] == 'measured']
        if samples:
            groups.append({'arm': arm, **({'cohort': cohort} if sweep else {}), 'independent_launches': len(samples),
                           'tokens_per_second_median': statistics.median(r['useful_tokens_per_second'] for r in samples)})
    complete = all(row['status'] == 'measured' for row in rows)
    result = {'stage': manifest.get('stage', 'N3' if sweep else 'N2'), 'status': 'complete' if complete else 'incomplete',
              'config': manifest['config'], 'cases': rows, 'by_arm': groups,
              'performance_result': any(row['status'] == 'measured' for row in rows),
              'paper_ready': False, 'cycles_treated_as_independent_runs': False,
              'remaining_for_main_table': ['32-GPU paired repeats, equal 64/256 MiB tuning and independent baseline S',
                                          'short-window numerical comparison for the native reduction arms'],
              'memory_scope': 'maximum rank PyTorch allocated/reserved peak in measured cycles; not device/NVML',
              'data_note': 'Synthetic-token runs measure Qwen execution, not training quality on a corpus.'}
    if sweep:
        result['historical_reference'] = ('historical-n2.json' if (output / 'historical-n2.json').is_file() else None)
        result['comparison_note'] = 'Cohort ratios use this allocation only; historical N2 is a separate context table.'
        result['remaining_for_main_table'] = ['independent paired repeats and cohort sweep at the 32-GPU main point']
    elif 'O' in manifest['config']['arms']:
        result['comparison_note'] = 'Main O/R/P: report both time and GPU memory for every arm; G/W remain auxiliary controls.'
        result['offload_implementation'] = 'Author implementation: pinned CPU R/M shards, blocking tiled transfers, GPU gather/RS/update/AG.'
        result['host_memory_scope'] = 'Persistent outer R/M tensor bytes only; not process/node host peak or all pinned allocations.'
        result['remaining_for_main_table'] = ['selected configurations with equal tuning and three independent paired launches',
                                              'offload transfer/scheduling characterization; no optimized-offload claim from this path alone']
        if any(case['backend'] == 'cpu_offload' for case in manifest['cases']):
            result['offload_implementation'] = ('Naive unsharded CPU offload v1: full pageable R/M replicas, '
                'blocking per-parameter copies/AllReduce, CPU Nesterov, no overlap or bucketing.')
            result['workspace_comparison'] = ('O uses dynamic whole-parameter buffers; the 64/256 MiB tile cap '
                'applies to tiled R/P/G/W/OS only. Compare actual measured GPU peaks.')
            result['remaining_for_main_table'] = ['independent paired repeats for the explicitly naive placement comparison',
                'keep sharded offload OS and G/W as separate supporting controls']
    return result


def render(result):
    sweep = result['config'].get('suite') == 'cohorts'
    offload = not sweep and 'O' in result['config']['arms']
    ratio = 'P(s=2)' if sweep else ('O' if offload else 'G')
    lines = [result['stage'] + ': ' + result['status'],
             f'case           status       tokens/s    outer+commit(s)   alloc/reserved GiB   speedup/{ratio}']
    if result['config'].get('pier_schedule', 'reference') != 'reference':
        lines.insert(1, 'Pier schedule: ' + result['config']['pier_schedule'])
    for row in result['cases']:
        if row['status'] == 'measured':
            value = row.get('speedup_vs_P_s2' if sweep else ('speedup_vs_O' if offload else 'speedup_vs_G'))
            speedup = '-' if value is None else f'{value:.4f}x'
            lines.append(f"{row['id']:<14} measured  {row['useful_tokens_per_second']:11.2f}"
                         f" {row['outer_and_commit_seconds_mean']:18.4f}"
                         f" {row['peak_allocated_gib']:8.2f}/{row['peak_reserved_gib']:.2f}   {speedup}")
        else:
            lines.append(f"{row['id']:<14} {row['status']:<12} {row.get('error', '')}")
    if offload:
        if any(row['backend'] == 'cpu_offload' for row in result['cases']):
            lines.append('O-naive: full pageable CPU R/M replicas, serial parameter copies/AllReduce and CPU update; no tiled workspace cap.')
        else:
            lines.append('O: pinned CPU R/M shards with blocking tiled transfers; not a tuned asynchronous offload implementation.')
        for row in result['cases']:
            if row['arm'] == 'O' and row['status'] == 'measured':
                lines.append(f"{row['id']} persistent host R/M: {row['host_outer_state_gib_max_rank']:.3f} GiB/rank (maximum; not host peak).")
    return '\n'.join(lines) + '\n'


def copy_historical_reference(reference, output):
    """Optional context, never a speedup denominator for the new allocation."""
    try:
        result = collect(reference)
        if not result['performance_result'] or result['stage'] != 'N2':
            raise ValueError('reference has no valid N2 performance results')
        record = {'source_directory': str(reference.resolve()), 'comparison_scope': 'historical context only',
                  'manifest_sha256': hashlib.sha256((reference / 'manifest.json').read_bytes()).hexdigest(),
                  'result': result}
        (output / 'historical-n2.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        text = 'Historical N2 allocation; do not pool with the new cohort measurements.\n' + render(result)
        (output / 'historical-n2.txt').write_text(text)
        print(f'[N3] Historical context saved from {reference}; new cohort ratios use the current allocation.', flush=True)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f'[N3] Historical context unavailable ({exc}); the cohort job will still run.', flush=True)


def save_summary(output):
    output = Path(output)
    result = collect(output)
    temporary = output / 'summary.json.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(output / 'summary.json')
    table = render(result)
    (output / 'results.txt').write_text(table)
    print(table, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    save_summary(args.directory)
