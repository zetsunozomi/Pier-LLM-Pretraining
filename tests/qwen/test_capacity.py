"""Capacity search, failure classification, resume, and actual shell launch wiring.

All evidence here is synthetic test data; these tests never assert GPU validation.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.qwen import capacity
from experiments.qwen.capacity_config import ARMS, ROOT, architecture, next_trial, parameters, recipe, training_args
from experiments.qwen.n2_config import ARMS as BACKENDS


def simulate(limit, confirm_limit=None, history=None):
    trials = list(history or [])
    for _ in range(100):
        layer, phase = next_trial(trials)
        if layer is None:
            return capacity.result('R', trials, phase), trials
        threshold = confirm_limit if phase == 'confirm' and confirm_limit is not None else limit
        trials.append(dict(layers=layer, phase=phase, status='passed' if layer <= threshold else 'oom'))
    raise AssertionError('search did not converge')


def evidence(path, manifest):
    capacity.write(path / 'manifest.json', manifest)
    fingerprint = hashlib.sha256((path / 'manifest.json').read_bytes()).hexdigest()
    steps, world = manifest['steps'], manifest['config']['world_size']
    label, cohort = ARMS[manifest['arm']]
    for rank in range(world):
        capacity.write(path / f'worker-rank-{rank}.json', dict(rank=rank, GPU_executed=True,
            gpu=manifest['config']['expected_gpu'], manifest_sha256=fingerprint))
        capacity.write(path / f'initialization-rank-{rank}.json', dict(rank=rank,
            parameters=manifest['parameters'], initialization='random',
            model_size_count_checked=True, pretrained_weights_loaded=False))
        capacity.write(path / f'success-rank-{rank}.json', dict(rank=rank, status='passed'))
        capacity.write(path / f'cycles-rank-{rank}.json', dict(rank=rank, status='complete',
            world_size=world, GPU_executed=True, run_id='fixture', planned_attempts=steps,
            initial_clock=dict(interval=50, attempted=0, successful=0, boundaries=0),
            final_clock=dict(interval=50, attempted=steps, successful=steps, boundaries=steps//50),
            complete_cycles=steps//50, cycles=[{'successful_steps': 50}]*(steps//50)+[{'successful_steps': 1}],
            metadata=dict(arm=BACKENDS[label], cohort=cohort,
                state_storage={name: {'device': 'cpu' if label in ('OS', 'O') else 'cuda'}
                               for name in ('reference', 'momentum')},
                final_health=dict(model_matches_master=True, finite_model=True, finite_loss=True))))


class CapacityTests(unittest.TestCase):
    def test_search_every_threshold_and_ceiling(self):
        for limit in range(1, 256):
            with self.subTest(limit=limit):
                summary, trials = simulate(limit)
                self.assertTrue(summary['exact_layer_boundary'])
                self.assertEqual(summary['max_confirmed_layers'], limit)
                self.assertEqual(summary['smallest_oom_layers'], limit+1)
                self.assertLessEqual(len(trials), 18)
        self.assertEqual(simulate(0)[0]['status'], 'no_feasible_model')
        self.assertEqual(simulate(256)[0]['status'], 'search_ceiling_reached')
        self.assertFalse(simulate(256)[0]['capacity_result'])

    def test_confirmation_failure_and_interruption_do_not_create_false_limit(self):
        summary, trials = simulate(43, confirm_limit=40)
        self.assertEqual(summary['max_confirmed_layers'], 40)
        self.assertTrue(summary['exact_layer_boundary'])
        partial = trials[:4]
        candidate = next_trial(partial)
        for status in ('interrupted', 'error'):
            self.assertEqual(next_trial(partial + [dict(layers=candidate[0], phase=candidate[1], status=status)]), candidate)
        self.assertEqual(simulate(43, confirm_limit=40, history=partial)[0], summary)

    def test_model_count_matches_independent_qwen_shape_schema(self):
        # Load this pure config module without importing the torch-dependent package.
        spec = importlib.util.spec_from_file_location('capacity_schema_fixture', ROOT/'megatron/core/models/qwen/config.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            for layers in (1, 17, 36, 48, 256):
                self.assertEqual(parameters(layers), module.QwenArchitecture.from_config(architecture(layers)).unique_parameters())
        finally:
            del sys.modules[spec.name]
        pins = json.loads((ROOT/'experiments/qwen/pins.json').read_text())
        self.assertEqual(parameters(36), pins['models']['3B']['unique_parameters'])

    def test_arguments_reuse_backends_and_keep_recipe_fixed(self):
        cfg = recipe('/tmp/tokenizer')
        for arm, (label, cohort) in ARMS.items():
            argv = training_args(cfg, arm, 45, 151, '/tmp/trial')
            self.assertEqual(argv[argv.index('--outer-arm')+1], BACKENDS[label])
            self.assertEqual(argv[argv.index('--outer-cohort-size')+1], str(cohort))
            self.assertEqual(argv[argv.index('--capacity-layers')+1], '45')
            self.assertEqual(argv[argv.index('--train-iters')+1], '151')
            self.assertNotIn('--qwen-snapshot', argv)
            self.assertEqual('--outer-cpu-offload' in argv, label in ('O', 'OS'))
            self.assertEqual('--outer-workspace-mib' in argv, label != 'O')

    def test_missing_rank_skips_nonfinite_and_non_oom_failures_reject(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            manifest = dict(config=recipe('/tmp/tokenizer'), arm='OS', layers=40, parameters=parameters(40), steps=51)
            evidence(path, manifest)
            self.assertEqual(capacity.classify(path, manifest, 0)['status'], 'passed')
            reportpath = path/'cycles-rank-31.json'
            good = reportpath.read_text()
            report = json.loads(good)
            report['final_clock']['successful'] = 50
            capacity.write(reportpath, report)
            self.assertEqual(capacity.classify(path, manifest, 0)['status'], 'error')
            reportpath.write_text(good)
            (path/'success-rank-31.json').unlink()
            self.assertEqual(capacity.classify(path, manifest, 0)['status'], 'error')
            self.assertEqual(capacity.classify(path, manifest, 137)['status'], 'error')
            failure = path/'failure-rank-0.json'
            capacity.write(failure, dict(rank=0, cuda_oom=False, error='NCCL timeout'))
            self.assertEqual(capacity.classify(path, manifest, 1)['status'], 'error')
            capacity.write(failure, dict(rank=0, cuda_oom=True, error='CUDA out of memory'))
            self.assertEqual(capacity.classify(path, manifest, 1)['status'], 'oom')
            self.assertEqual(capacity.classify(path, manifest, 1, interrupted=True)['status'], 'interrupted')

    def test_driver_orchestrates_resume_from_persisted_trials(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            args = SimpleNamespace(output_dir=directory, arm='R', snapshot=directory/'tokenizer',
                                   start_layers=36, ceiling_layers=256, budget_seconds=3300)
            calls = []

            class FakeProcess:
                def __init__(self, command, **kwargs):
                    trial = Path(command[-1])
                    manifest = json.loads((trial/'manifest.json').read_text())
                    calls.append(manifest['layers'])
                    if manifest['layers'] <= 39:
                        evidence(trial, manifest)
                        self.code = 0
                    else:
                        capacity.write(trial/'failure-rank-0.json', {'rank': 0, 'cuda_oom': True})
                        self.code = 1
                def poll(self): return self.code
                def wait(self, **kwargs): return self.code

            with patch.dict(os.environ, SLURM_JOB_NUM_NODES='8'), \
                 patch.object(capacity, 'sources', return_value={'fixture': 'fixed'}), \
                 patch.object(capacity, 'tokenizer_identity', return_value={}), \
                 patch.object(capacity.signal, 'signal'), \
                 patch.object(capacity.subprocess, 'Popen', FakeProcess), \
                 contextlib.redirect_stdout(io.StringIO()):
                args.budget_seconds = 60
                self.assertEqual(capacity.run(args), 0)
                self.assertFalse(calls)
                self.assertEqual(json.loads((directory/'R/summary.json').read_text())['status'], 'needs_resume')
                args.budget_seconds = 3300
                self.assertEqual(capacity.run(args), 0)
                count = len(calls)
                self.assertEqual(capacity.run(args), 0)
                self.assertEqual(len(calls), count)
            report = json.loads((directory/'R/summary.json').read_text())
            self.assertEqual(report['max_confirmed_layers'], 39)
            self.assertTrue(report['exact_layer_boundary'])

    def test_submit_selects_unfinished_arms_and_exports_environment(self):
        with tempfile.TemporaryDirectory(prefix='capacity launch ') as name:
            path = Path(name)
            campaign = path/'campaign'
            (campaign/'R').mkdir(parents=True)
            capacity.write(campaign/'R/summary.json', {'status': 'complete'})
            sbatch = path/'sbatch'
            sbatch.write_text(f'#!{sys.executable}\nimport json,os,sys\nprint(json.dumps(dict(argv=sys.argv[1:],budget=os.environ["PIER_CAPACITY_BUDGET_SECONDS"],root=os.environ["PIER_ROOT"])))\n')
            sbatch.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith(('PIER_', 'SLURM_'))}
            env.update(PATH=f'{path}:{os.environ["PATH"]}', PIER_ROOT=str(ROOT), PIER_PYTHON=sys.executable,
                       PIER_CAPACITY_DIR=str(campaign), PIER_CAPACITY_MINUTES='30')
            run = subprocess.run(['bash', str(ROOT/'experiments/qwen/capacity_submit.sh'), 'R', 'OS', 'P16'],
                                 env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            call = json.loads(run.stdout.splitlines()[-1])
            self.assertIn('--array=2,4%1', call['argv'])
            self.assertIn('--time=00:30:00', call['argv'])
            self.assertIn('--export=ALL', call['argv'])
            self.assertEqual(call['budget'], '1500')

    def test_batch_spool_and_interactive_launcher_preserve_recipe(self):
        for spool in (False, True):
            with self.subTest(spool=spool), tempfile.TemporaryDirectory(prefix='capacity batch ') as name:
                path = Path(name)
                fake_python = path/'fake-python'
                fake_python.write_text(f'#!{sys.executable}\nimport json,sys,os\n'
                    'print(json.dumps(dict(argv=sys.argv[1:],master=os.environ["PIER_CAPACITY_MASTER_ADDR"],'
                    'export=os.environ["SLURM_EXPORT_ENV"])))\n')
                fake_python.chmod(0o755)
                scontrol = path/'scontrol'
                scontrol.write_text('#!/bin/bash\nprintf "nid001\\nnid002\\n"\n')
                scontrol.chmod(0o755)
                script = ROOT/'experiments/qwen/capacity.sbatch'
                if spool:
                    copied = path/'slurm_script'
                    shutil.copyfile(script, copied)
                    script = copied
                env = {k: v for k, v in os.environ.items() if not k.startswith(('PIER_', 'SLURM_'))}
                env.update(PATH=f'{path}:{os.environ["PATH"]}', PIER_PYTHON=str(fake_python),
                           PIER_CAPACITY_DIR=str(path/'campaign'), SLURM_SUBMIT_DIR=str(ROOT),
                           SLURM_JOB_ID='123', SLURM_JOB_NUM_NODES='8', SLURM_JOB_NODELIST='nid[001-008]',
                           SLURM_ARRAY_TASK_ID='4')
                run = subprocess.run(['bash', str(script)], env=env, cwd=path,
                                     capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr)
                logfile = next((path/'campaign/P16').glob('job-*.out.txt'))
                call = json.loads(logfile.read_text())
                self.assertEqual(call['master'], 'nid001')
                self.assertEqual(call['export'], 'ALL')
                self.assertEqual(call['argv'][call['argv'].index('--arm')+1], 'P16')
                self.assertEqual(call['argv'][call['argv'].index('--budget-seconds')+1], '3300')
                self.assertIn(str(logfile), run.stdout)
                env['SLURM_JOB_NUM_NODES'] = '1'
                run = subprocess.run(['bash', str(script), 'R'], env=env, cwd=path,
                                     capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, 2)
                self.assertIn('eight nodes', run.stderr)


if __name__ == '__main__':
    unittest.main()
