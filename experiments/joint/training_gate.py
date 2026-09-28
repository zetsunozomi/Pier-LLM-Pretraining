#!/usr/bin/env python3
"""Four-GPU actual pretrain_gpt gate, including dynamic full-state restart."""

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.centered_outer.e0b_config import training_args as e0b_args
from experiments.joint.common import sources
from megatron.core.outer_sync.joint_config import JointController

GATE_ENV = dict(NCCL_ALGO='Ring', CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_MAX_CONNECTIONS='1')


def replace(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv.extend((flag, str(value)))


def phase_plan(output):
    output = Path(output)
    rows = []
    for name, variant, topology, stop, resume in (
            ('reference-tp1', None, 'tp1', None, None),
            ('single-tp1', 'single', 'tp1', None, None),
            ('pipeline-tp1', 'pipeline', 'tp1', None, None),
            ('separate-tp1', 'separate', 'tp1', None, None),
            ('joint-tp1', 'joint', 'tp1', None, None),
            ('split-step', 'joint', 'tp1', 5, None),
            ('resume-step', 'joint', 'tp1', None, 'split-step'),
            ('split-boundary', 'joint', 'tp1', 4, None),
            ('resume-boundary', 'joint', 'tp1', None, 'split-boundary'),
            ('reference-tp2', None, 'tp2', None, None),
            ('joint-tp2', 'joint', 'tp2', None, None),
            ('reference-dp2', None, 'dp2', None, None),
            ('joint-dp2', 'joint', 'dp2', None, None)):
        template = {'tp1': 'tp1-s1', 'tp2': 'tp2-s2', 'dp2': 'dp2-s2'}[topology]
        argv = e0b_args(template, output)
        initial = 2 if variant in ('single', 'pipeline') else 1
        replace(argv, '--outer-cohort-size', initial)
        replace(argv, '--outer-trace-dir', output / name)
        config = None
        if variant:
            k = 4 if topology == 'tp1' else 2
            config = dict(version=1, variant=variant, page_elements=257, capacities=[257, 1028],
                          slot_counts=[1, 2, 4], workspace_mib=64, headroom_mib=128, trace=True,
                          budgets=[dict(round=i + 1, transition_mib=2048, next_phase_mib=2048, target=s)
                                   for i, s in enumerate((2, k, 1))])
            replace(argv, '--outer-pier-schedule', 'joint')
            replace(argv, '--outer-joint-config', output / f'{name}.json')
        if stop or resume:
            checkpoint = output / 'checkpoints' / (resume or name)
            replace(argv, '--save', checkpoint)
            replace(argv, '--save-interval', 4 if 'boundary' in name else 5)
        if stop:
            replace(argv, '--exit-interval', stop)
        if resume:
            replace(argv, '--load', output / 'checkpoints' / resume)
        rows.append(dict(name=name, variant=variant, topology=topology, stop=stop, resume=resume,
                         config=config, argv=argv))
    return rows


def summarize(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    errors, reports, comparisons, pending = [], {}, [], []
    if manifest['sources'] != sources():
        errors.append('source changed during the training gate')
    for phase in manifest['phases']:
        name = phase['name']
        if not (output / name).exists():
            pending.append(name)
            continue
        reports[name] = []
        expected_end = phase['stop'] or 11
        for rank in range(4):
            try:
                row = json.loads((output / name / f'rank-{rank}.json').read_text())
                launch = json.loads((output / name / f'launch-rank-{rank}.json').read_text())
                exit_code = json.loads((output / name / 'exit.json').read_text())['exit_code']
                if (exit_code or not row['GPU_executed'] or row['backend'] != 'nccl' or row['world_size'] != 4
                        or row['status'] != ('partial' if phase['stop'] else 'passed')
                        or row['clock'] != dict(interval=3, attempted=expected_end, successful=expected_end - 1,
                                               boundaries=(expected_end - 1) // 3)
                        or launch['argv'] != phase['argv'] or not launch['GPU_executed']
                        or launch['environment'] != manifest['environment']
                        or row['restored'] != bool(phase['resume']) or row['error'] is not None):
                    raise ValueError('incomplete launch/clock/device/restore contract')
                events = row['events']
                if len(events) != expected_end:
                    raise ValueError('missing attempted-step history')
                for event in events:
                    if event['outer_boundary'] and (not event['oracle_states_bitwise'] or not event['outer_retained_inner_state']):
                        raise ValueError('outer oracle or inner-state check failed')
                    if event['skipped'] and not event['skip_retained_optimizer']:
                        raise ValueError('skipped attempt modified optimizer')
                    if event['outer_boundary'] and phase['config']:
                        controller = JointController(phase['config'])
                        payload = event['payload']
                        target = (2 if phase['variant'] in ('single', 'pipeline') else
                                  controller.announcement(event['boundaries'])['target'])
                        if (payload['configuration_sha256'] != controller.sha256
                                or payload['plan']['cohort'] != target or payload['variant'] != phase['variant']):
                            raise ValueError('runtime did not execute the planned layout/variant')
                if phase['resume']:
                    expected_checkpoint = 4 if 'boundary' in name else 5
                    if row['restore_evidence']['iteration'] != expected_checkpoint:
                        raise ValueError('wrong resume checkpoint')
                if not phase['stop'] and (row['pending_consumer'] or row['consumer_checks'] != 3):
                    raise ValueError('post-sync forward was not checked')
                reports[name].append(row)
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f'{name}/rank{rank}: {exc}')
    if not errors:
        comparable = lambda row: [{k: v for k, v in event.items() if k != 'payload'} for event in row['events']]
        for phase in manifest['phases']:
            if phase['variant'] is None or phase['name'] in pending:
                continue
            if f"reference-{phase['topology']}" in pending:
                continue
            for rank in range(4):
                base = comparable(reports[f"reference-{phase['topology']}"][rank])
                actual = comparable(reports[phase['name']][rank])
                equal = actual == base[:len(actual)]
                comparisons.append(dict(phase=phase['name'], rank=rank, bitwise_trajectory_equal=equal))
                if not equal:
                    errors.append(f"{phase['name']}/rank{rank}: training trajectory differs")
    return dict(format='pier-joint-training-gate-v1', kind='training_correctness',
                status='failed' if errors else ('incomplete' if pending else 'passed'),
                GPU_executed=any(r.get('GPU_executed') for rows in reports.values() for r in rows), world_size=4,
                torch_versions=sorted({r['torch_version'] for rows in reports.values() for r in rows}),
                sources=manifest['sources'], comparisons=comparisons, errors=errors, pending_phases=pending,
                not_tested=['full-size Qwen training', 'convergence', 'performance'])


def worker(output, phase_name):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    phase = next(p for p in manifest['phases'] if p['name'] == phase_name)
    directory = output / phase_name
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['RANK'])
    try:
        import torch
        if any(os.environ.get(key) != value for key, value in manifest['environment'].items()):
            raise RuntimeError('deterministic training gate environment differs from the frozen plan')
        if not torch.cuda.is_available() or int(os.environ['WORLD_SIZE']) != 4:
            raise RuntimeError('training gate requires four CUDA workers')
        if sources() != manifest['sources']:
            raise RuntimeError('source changed after training gate plan')
        if phase['config'] is not None:
            if json.loads((output / f'{phase_name}.json').read_text()) != phase['config']:
                raise RuntimeError('joint configuration changed after the gate plan')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        (directory / f'launch-rank-{rank}.json').write_text(json.dumps(dict(
            argv=phase['argv'], GPU_executed=True, rank=rank, gpu=torch.cuda.get_device_name(),
            torch=torch.__version__, cuda=torch.version.cuda, entrypoint='pretrain_gpt.py',
            environment={key: os.environ[key] for key in GATE_ENV}), indent=2))
        sys.argv = [str(ROOT / 'pretrain_gpt.py'), *phase['argv']]
        runpy.run_path(str(ROOT / 'pretrain_gpt.py'), run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    except BaseException as exc:
        (directory / f'failure-rank-{rank}.json').write_text(json.dumps(dict(
            error=repr(exc), traceback=traceback.format_exc()), indent=2))
        raise


def write_summary(output):
    summary = summarize(output)
    temporary = output / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(output / 'summary.json')
    return summary


def run(output, plan_only=False, *, resume=False, max_phases=13, budget_seconds=1500):
    if max_phases < 1 or budget_seconds < 120 or (resume and plan_only):
        raise ValueError('positive phase count, >=120 second budget, and no resume/plan-only combination required')
    started = time.monotonic()
    output = Path(output).resolve()
    plan = phase_plan(output)
    manifest = dict(phases=plan, sources=sources(), plan_only=plan_only, environment=GATE_ENV)
    if resume:
        previous = json.loads((output / 'manifest.json').read_text())
        if previous != manifest:
            raise ValueError('cannot resume: source, environment, output path or phase plan changed')
        for phase in plan:
            if phase['config'] is not None and json.loads((output / f"{phase['name']}.json").read_text()) != phase['config']:
                raise ValueError('cannot resume: frozen phase configuration changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        for phase in plan:
            if phase['config'] is not None:
                (output / f"{phase['name']}.json").write_text(json.dumps(phase['config'], indent=2))
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if plan_only:
        return
    summary = write_summary(output)
    if summary['status'] == 'failed':
        raise RuntimeError('existing phase failed or was interrupted; inspect evidence before retrying')
    completed_here, durations = 0, []
    for phase in plan:
        if phase['name'] not in summary['pending_phases']:
            continue
        remaining = budget_seconds - (time.monotonic() - started)
        # Leave room to close the Slurm step and write the partial summary.
        if completed_here >= max_phases or remaining < max(120, 2 * max(durations, default=0) + 60):
            break
        directory = output / phase['name']
        directory.mkdir()
        step_minutes = max(1, int((remaining - 60) // 60))
        command = ['srun', '--nodes=1', '--ntasks=1', '--ntasks-per-node=1', '--kill-on-bad-exit=1',
                   '--gpus-per-node=4', f'--time={step_minutes}',
                   '--gpu-bind=none', 'bash', 'experiments/joint/node.sh',
                   'experiments/joint/training_gate.py', '--worker', phase['name'], '--output-dir', str(output)]
        print(f"START {phase['name']} (step limit {step_minutes} min)", flush=True)
        phase_started = time.monotonic()
        with (directory / 'out.txt').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, **GATE_ENV})
        elapsed = time.monotonic() - phase_started
        (directory / 'exit.json').write_text(json.dumps(dict(
            exit_code=result.returncode, command=command, elapsed_seconds=elapsed,
            slurm_job_id=os.environ.get('SLURM_JOB_ID')), indent=2))
        durations.append(elapsed)
        completed_here += 1
        print(f"{phase['name']}: exit={result.returncode}, elapsed={elapsed:.1f}s", flush=True)
        summary = write_summary(output)
        if summary['status'] == 'failed':
            break
    print(f"Gate {summary['status']}; {len(summary['pending_phases'])} phases pending. Evidence: {output}", flush=True)
    if summary['status'] == 'failed':
        raise RuntimeError('training gate failed; inspect summary.json and per-phase out.txt')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Continue a clean partial gate with the identical frozen source')
    parser.add_argument('--max-phases', type=int, default=13, help='Maximum new phases in this allocation')
    parser.add_argument('--budget-seconds', type=int, default=1500, help='Driver time budget; reserve 5 min in a fresh 30 min allocation')
    args = parser.parse_args()
    if args.worker:
        worker(args.output_dir, args.worker)
    else:
        run(args.output_dir, args.plan_only, resume=args.resume,
            max_phases=args.max_phases, budget_seconds=args.budget_seconds)
