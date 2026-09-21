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
        launch_directory = Path(manifest.get('output_directory', output)) / case['id']
        for rank, meta in enumerate(raw['rank_metadata']):
            require(meta['arm'] == case['backend'] and meta['cohort'] == case['cohort'], 'wrong backend/cohort')
            health = meta['final_health']
            require(health['model_matches_master'] and health['finite_model'] and health['finite_loss'],
                    'training health check failed')
            losses.append(health['losses']['lm loss'])
            require(math.isfinite(losses[-1]), 'nonfinite final loss')
            worker = json.loads((directory / f'worker-rank-{rank}.json').read_text())
            require(worker['rank'] == rank and worker['case'] == case and worker['GPU_executed']
                    and worker['manifest_sha256'] == fingerprint
                    and worker['argv'] == training_args(cfg, case, launch_directory), 'worker launch differs from recipe')
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
    except (ValueError, KeyError, TypeError, OSError) as exc:
        row.update(status='invalid', error=str(exc))
    return row


def collect(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    rows = [collect_case(output, manifest, case) for case in manifest['cases']]
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
    groups = []
    for arm in manifest['config']['arms']:
        samples = [r for r in rows if r['arm'] == arm and r['status'] == 'measured']
        if samples:
            groups.append({'arm': arm, 'independent_launches': len(samples),
                           'tokens_per_second_median': statistics.median(r['useful_tokens_per_second'] for r in samples)})
    complete = all(row['status'] == 'measured' for row in rows)
    result = {'stage': 'N2', 'status': 'complete' if complete else 'incomplete',
              'config': manifest['config'], 'cases': rows, 'by_arm': groups,
              'performance_result': any(row['status'] == 'measured' for row in rows),
              'paper_ready': False, 'cycles_treated_as_independent_runs': False,
              'remaining_for_main_table': ['32-GPU paired repeats, equal 64/256 MiB tuning and independent baseline S',
                                          'short-window numerical comparison for the native reduction arms'],
              'memory_scope': 'maximum rank PyTorch allocated/reserved peak in measured cycles; not device/NVML',
              'data_note': 'Synthetic-token runs measure Qwen execution, not training quality on a corpus.'}
    return result


def render(result):
    lines = ['N2: ' + result['status'],
             'case           status       tokens/s    outer+commit(s)   alloc/reserved GiB   speedup/G']
    for row in result['cases']:
        if row['status'] == 'measured':
            speedup = '-' if row['speedup_vs_G'] is None else f"{row['speedup_vs_G']:.4f}x"
            lines.append(f"{row['id']:<14} measured  {row['useful_tokens_per_second']:11.2f}"
                         f" {row['outer_and_commit_seconds_mean']:18.4f}"
                         f" {row['peak_allocated_gib']:8.2f}/{row['peak_reserved_gib']:.2f}   {speedup}")
        else:
            lines.append(f"{row['id']:<14} {row['status']:<12} {row.get('error', '')}")
    return '\n'.join(lines) + '\n'


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
