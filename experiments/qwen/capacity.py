#!/usr/bin/env python3
"""Find one maximum feasible depth per arm; resume after short Slurm allocations."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qwen.capacity_config import (
    ARMS, next_trial, parameters, recipe, tokenizer_identity, training_args,
)
from experiments.qwen.n2 import source_identity as n2_source_identity
from experiments.qwen.n2_config import ARMS as BACKENDS


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sources():
    result = n2_source_identity()
    config = ROOT / 'experiments/qwen/qwen2.5-3B-config.json'
    result[str(config.relative_to(ROOT))] = hashlib.sha256(config.read_bytes()).hexdigest()
    for path in (ROOT / 'experiments/qwen').glob('capacity*'):
        if path.is_file():
            result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def classify(directory, manifest, code, interrupted=False):
    """Only explicit CUDA OOM evidence can move the failing bound."""
    if interrupted:
        return {'status': 'interrupted', 'reason': 'allocation deadline or signal'}
    failures = [json.loads(p.read_text()) for p in directory.glob('failure-rank-*.json')]
    if any(f.get('cuda_oom') is True for f in failures):
        return {'status': 'oom', 'oom_ranks': [f['rank'] for f in failures if f.get('cuda_oom')],
                'reason': 'explicit worker CUDA OutOfMemoryError; see failure rank traceback'}
    if code or failures:
        return {'status': 'error', 'reason': 'non-OOM worker/Slurm failure; inspect out.txt'}
    try:
        manifest_sha = hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest()
        steps, world = manifest['steps'], manifest['config']['world_size']
        run_ids = set()
        for rank in range(world):
            def check(condition, message):
                if not condition:
                    raise ValueError(f'rank {rank}: {message}')

            worker = json.loads((directory / f'worker-rank-{rank}.json').read_text())
            init = json.loads((directory / f'initialization-rank-{rank}.json').read_text())
            success = json.loads((directory / f'success-rank-{rank}.json').read_text())
            report = json.loads((directory / f'cycles-rank-{rank}.json').read_text())
            check(worker['rank'] == init['rank'] == success['rank'] == report['rank'] == rank, 'rank identity differs')
            check(worker['manifest_sha256'] == manifest_sha and worker['GPU_executed'] is True, 'worker identity differs')
            check(worker['gpu'] == manifest['config']['expected_gpu'], 'GPU model differs')
            check(init['parameters'] == manifest['parameters'] and init['model_size_count_checked'] is True, 'parameter count differs')
            check(init['initialization'] == 'random' and init['pretrained_weights_loaded'] is False, 'initialization differs')
            check(success['status'] == 'passed' and report['status'] == 'complete', 'run did not complete')
            check(report['world_size'] == world and report['GPU_executed'] is True, 'GPU world differs')
            check(report['final_clock'] == dict(interval=50, attempted=steps, successful=steps, boundaries=steps//50),
                  'missing successful steps or outer boundaries')
            check(report['initial_clock'] == dict(interval=50, attempted=0, successful=0, boundaries=0), 'not a fresh run')
            check(report['planned_attempts'] == steps and report['complete_cycles'] == steps//50, 'cycle count differs')
            check(len(report['cycles']) == steps//50 + 1 and report['cycles'][-1]['successful_steps'] == 1,
                  'missing final post-sync step')
            label, cohort = ARMS[manifest['arm']]
            check(report['metadata']['arm'] == BACKENDS[label] and report['metadata']['cohort'] == cohort, 'backend differs')
            expected_tier = 'cpu' if label in ('OS', 'O') else 'cuda'
            check(all(report['metadata']['state_storage'][name]['device'] == expected_tier
                      for name in ('reference', 'momentum')), 'state placement differs')
            health = report['metadata']['final_health']
            check(all(health[k] is True for k in ('model_matches_master', 'finite_model', 'finite_loss')), 'health check failed')
            run_ids.add(report['run_id'])
        if len(run_ids) != 1:
            raise ValueError('mixed run identities')
    except (KeyError, ValueError, TypeError, OSError) as exc:
        return {'status': 'error', 'reason': f'incomplete/invalid all-rank success evidence: {type(exc).__name__}: {exc}'}
    return {'status': 'passed', 'successful_steps': steps, 'outer_cycles': steps//50}


def history(directory):
    trials = []
    for path in sorted(directory.glob('trial-*/result.json')):
        record = json.loads(path.read_text())
        trials.append(record)
    return trials


def result(arm, trials, terminal):
    failed = min((t['layers'] for t in trials if t['status'] == 'oom'), default=None)
    good = [t for t in trials if t['status'] == 'passed'
            and (failed is None or t['layers'] < failed)]
    confirmed = [t for t in good if t['phase'] == 'confirm']
    low = max((t['layers'] for t in good), default=None)
    maximum = max((t['layers'] for t in confirmed), default=None)
    return {'arm': arm, 'status': terminal,
            'max_confirmed_layers': maximum,
            'max_confirmed_parameters': parameters(maximum) if maximum else None,
            'largest_probe_pass_layers': low, 'smallest_oom_layers': failed,
            'smallest_oom_parameters': parameters(failed) if failed else None,
            'exact_layer_boundary': terminal == 'complete' and maximum is not None and failed == maximum + 1,
            'capacity_result': terminal == 'complete', 'performance_result': False,
            'trial_count': len(trials)}


def print_summary(root):
    rows = []
    for arm in ARMS:
        path = root / arm / 'summary.json'
        if path.exists():
            rows.append(json.loads(path.read_text()))
    print('arm  status                   max_layers  parameters(B)  next_OOM_layers')
    for r in rows:
        count = r['max_confirmed_parameters']
        formatted = f'{count/1e9:.6f}' if count else 'pending'
        print(f"{r['arm']:4} {r['status']:24} {str(r['max_confirmed_layers']):>10} "
              f"{formatted:>14} {str(r['smallest_oom_layers']):>16}")
    # No shared writable summary: concurrent array tasks own separate arm dirs.
    return rows


def run(args):
    if int(os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES', '0'))) != 8:
        raise ValueError('use an eight-node A100-40GB allocation (TP2, K16)')
    root = args.output_dir.resolve()
    directory = root / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'another job is scanning {args.arm} in {root}') from None
        return scan(args, root, directory)


def scan(args, root, directory):
    cfg = recipe(args.snapshot)
    plan = dict(format='pier-capacity-v1', config=cfg, arm=args.arm,
                start_layers=args.start_layers, ceiling_layers=args.ceiling_layers,
                tokenizer=tokenizer_identity(cfg['snapshot']), sources=sources())
    plan_path = directory / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('recipe/source/tokenizer changed; choose a fresh PIER_CAPACITY_DIR')
    else:
        write(plan_path, plan)
    end = time.monotonic() + args.budget_seconds
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    terminal = 'needs_resume'
    while not interrupted:
        trials = history(directory)
        layers, phase = next_trial(trials, start=args.start_layers, ceiling=args.ceiling_layers)
        if layers is None:
            terminal = phase
            break
        steps = cfg['confirm_steps'] if phase == 'confirm' else cfg['probe_steps']
        # Reserve enough time for a whole fresh run. Final confirmation is longer.
        passed = [t for t in trials if t['status'] == 'passed']
        estimate = 180 + 4 * steps
        if passed:
            last = passed[-1]
            estimate = max(180, last['wall_seconds'] * steps/last['steps'] * layers/last['layers'] * 1.4)
        if end - time.monotonic() < estimate + 60:
            print(f'[{args.arm}] Pausing before {layers} layers/{steps} steps; rerun the same submission to resume.', flush=True)
            break
        if sources() != plan['sources']:
            raise RuntimeError('source changed during capacity scan')
        # An abruptly killed previous trial has no result; retain it and use a new directory.
        serial = max([int(p.name.split('-')[1]) for p in directory.glob('trial-*')], default=0) + 1
        trial = directory / f'trial-{serial:03d}-L{layers}-{phase}'
        trial.mkdir()
        manifest = dict(config=cfg, arm=args.arm, layers=layers, parameters=parameters(layers),
                        steps=steps, phase=phase, created_utc=datetime.now(timezone.utc).isoformat(),
                        training_argv=training_args(cfg, args.arm, layers, steps, trial))
        write(trial / 'manifest.json', manifest)
        command = ['srun', '--nodes=8', '--ntasks=8', '--ntasks-per-node=1',
                   '--kill-on-bad-exit=1', '--gpu-bind=none',
                   'bash', str(ROOT / 'experiments/qwen/capacity_node.sh'), str(trial)]
        print(f'[{args.arm}] {phase}: L={layers}, {parameters(layers)/1e9:.6f}B, {steps} steps; {trial}/out.txt', flush=True)
        start = time.monotonic()
        with (trial / 'out.txt').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None and not interrupted and time.monotonic() < end:
                time.sleep(1)
            if process.poll() is None:
                interrupted = True
                # TERM reaches srun, which cancels the remote step and its torchrun workers.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
        outcome = classify(trial, manifest, code, interrupted)
        outcome.update(layers=layers, parameters=parameters(layers), steps=steps, phase=phase,
                       wall_seconds=time.monotonic()-start, exit_code=code,
                       command=command, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        write(trial / 'result.json', outcome)
        write(directory / 'summary.json', result(args.arm, history(directory), 'searching'))
        print(f"[{args.arm}] {outcome['status']}: L={layers} ({outcome['wall_seconds']:.1f}s)", flush=True)
        if outcome['status'] == 'error':
            terminal = 'error'
            break
    write(directory / 'summary.json', result(args.arm, history(directory), terminal))
    print_summary(root)
    return 1 if terminal in ('error', 'no_feasible_model', 'search_ceiling_reached') else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(ARMS))
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--start-layers', type=int, default=36)
    parser.add_argument('--ceiling-layers', type=int, default=256)
    parser.add_argument('--budget-seconds', type=int, default=3300)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.summary:
        print_summary(args.output_dir)
        return 0
    if not args.arm or not args.snapshot:
        parser.error('--arm and --snapshot are required to run')
    if not 1 <= args.start_layers <= args.ceiling_layers or args.budget_seconds < 60:
        parser.error('require 1 <= start <= ceiling and budget >= 60 seconds')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
