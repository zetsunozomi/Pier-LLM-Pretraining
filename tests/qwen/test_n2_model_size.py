"""Model selection and the actual N4 shell chain; no GPU or model downloads."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments.qwen import n2
from experiments.qwen.n2_config import ROOT, cases, configuration, training_args


class ModelSizeTests(unittest.TestCase):
    def test_model_pin_snapshot_and_training_args_change_together(self):
        pins = json.loads((ROOT / 'experiments/qwen/pins.json').read_text())['models']
        configs = {}
        for size in ('1.5B', '3B'):
            cfg = configuration({'PIER_QWEN_MODEL_SIZE': size, 'SLURM_NNODES': '8',
                                 'PIER_N2_ARMS': 'O,R,P', 'PIER_ROOT': '/scratch/pier'})
            configs[size] = cfg
            self.assertEqual(cfg['model'], pins[size]['model'])
            self.assertEqual(cfg['revision'], pins[size]['revision'])
            self.assertEqual(cfg['snapshot'], f'/scratch/pier/local/qwen/models/Qwen2.5-{size}/{pins[size]["revision"]}')
            self.assertEqual((cfg['attempts'], cfg['warmup_cycles'], cfg['measured_cycles']), (100, 1, 1))
            self.assertEqual((cfg['world_size'], cfg['tp'], cfg['learners']), (32, 2, 16))
            for case in cases(cfg):
                argv = training_args(cfg, case, '/tmp/model-size-case')
                self.assertEqual(argv[argv.index('--qwen-model-size') + 1], size)
                self.assertEqual(argv[argv.index('--qwen-snapshot') + 1], cfg['snapshot'])
                self.assertEqual(argv[argv.index('--tokenizer-model') + 1], cfg['snapshot'])
                self.assertEqual(argv[argv.index('--recompute-granularity') + 1], 'full')
                self.assertIn('--bf16', argv)
                self.assertEqual('--outer-workspace-mib' in argv, case['arm'] != 'O')
        different = {key for key in configs['3B'] if configs['3B'][key] != configs['1.5B'][key]}
        self.assertEqual(different, {'model_size', 'model', 'revision', 'snapshot'})

    def test_default_and_historical_manifests_keep_3b_arguments(self):
        cfg = configuration({})
        self.assertEqual(cfg['model_size'], '3B')
        legacy = dict(cfg)
        del legacy['model_size']
        for case in cases(cfg):
            self.assertEqual(training_args(cfg, case, '/tmp/legacy'),
                             training_args(legacy, case, '/tmp/legacy'))
        for value in ('', '3B'):
            self.assertEqual(configuration({'PIER_QWEN_MODEL_SIZE': value}), cfg)
        with self.assertRaisesRegex(ValueError, 'PIER_QWEN_MODEL_SIZE'):
            configuration({'PIER_QWEN_MODEL_SIZE': '1.5b'})

    def test_snapshot_override_keeps_selected_model_identity(self):
        cfg = configuration({'PIER_QWEN_MODEL_SIZE': '1.5B', 'PIER_QWEN_SNAPSHOT': '/scratch/custom'})
        self.assertEqual(cfg['snapshot'], '/scratch/custom')
        self.assertEqual(cfg['model'], 'Qwen/Qwen2.5-1.5B')

    def test_launcher_checks_selected_snapshot_files_before_real_subprocesses(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'experiments/qwen').mkdir(parents=True)
            (root / 'experiments/qwen/pins.json').write_text(json.dumps({'models': {
                '1.5B': {'files': {'model.safetensors': {'bytes': 1}}},
                '3B': {'files': {'model-00001-of-00002.safetensors': {'bytes': 2}}}}}))
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            (snapshot / 'model.safetensors').write_text('x')
            cfg = configuration({'PIER_QWEN_MODEL_SIZE': '1.5B', 'PIER_QWEN_SNAPSHOT': str(snapshot),
                                 'PIER_N2_ARMS': 'O,R,P', 'SLURM_NNODES': '8'})
            summary = types.ModuleType('experiments.qwen.n2_summary')
            # This test exercises the launcher only, not GPU evidence acceptance.
            summary.save_summary = lambda output: {'status': 'complete'}
            with patch.object(n2, 'ROOT', root), patch.object(n2, 'configuration', return_value=cfg), \
                 patch.object(n2, 'source_identity', return_value={}), \
                 patch.object(n2, 'launch_command', return_value=[sys.executable, '-c', 'print("CPU launcher fixture")']), \
                 patch.object(n2.subprocess, 'check_output', return_value='fixture-revision'), \
                 patch.dict(sys.modules, {'experiments.qwen.n2_summary': summary}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(n2.run(root / 'out'), 0)
                manifest = json.loads((root / 'out/manifest.json').read_text())
                self.assertEqual(manifest['config']['model_size'], '1.5B')
                for case in manifest['cases']:
                    argv = manifest['training_argv'][case['id']]
                    self.assertEqual(argv[argv.index('--qwen-model-size') + 1], '1.5B')
                    self.assertEqual(json.loads((root / 'out' / case['id'] / 'exit.json').read_text())['exit_code'], 0)
                (snapshot / 'model.safetensors').unlink()
                with self.assertRaisesRegex(ValueError, 'model.safetensors'):
                    n2.run(root / 'missing')

    def test_n4_batch_and_interactive_shells_freeze_recipe_and_capture_logs(self):
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory(prefix='pier n4 launch ') as name:
                root = Path(name)
                repo = root / 'repo with spaces'
                scripts = repo / 'experiments/qwen'
                scripts.mkdir(parents=True)
                (repo / 'pretrain_qwen.py').touch()
                (scripts / 'n2.py').touch()
                for filename in ('n4_15b.sbatch', 'n2.sbatch', 'log_output.sh', 'n2_config.py', 'pins.json'):
                    shutil.copyfile(ROOT / 'experiments/qwen' / filename, scripts / filename)
                script = scripts / 'n4_15b.sbatch'
                if batch:
                    spool = root / 'spool/job'
                    spool.mkdir(parents=True)
                    shutil.copyfile(script, spool / 'slurm_script')
                    script = spool / 'slurm_script'
                fake_python = root / 'python'
                fake_python.write_text(f'#!{sys.executable}\nimport json,sys\n'
                    'from experiments.qwen.n2_config import configuration, cases\n'
                    'cfg=configuration()\nprint(json.dumps(cfg))\n'
                    'print("worker-stderr",file=sys.stderr)\nraise SystemExit(23)\n')
                fake_python.chmod(0o755)
                scontrol = root / 'scontrol'
                scontrol.write_text('#!/bin/bash\nprintf "nid-test\\n"\n')
                scontrol.chmod(0o755)
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(('PIER_', 'SLURM_'))}
                env.update(PIER_PYTHON=str(fake_python), PATH=f'{root}:{os.environ["PATH"]}',
                    SLURM_JOB_ID='42', SLURM_JOB_NUM_NODES='8', SLURM_JOB_NODELIST='nid-test',
                    SLURM_SUBMIT_DIR=str(repo if batch else root), SLURM_ARRAY_TASK_ID='3',
                    PIER_QWEN_MODEL_SIZE='3B', PIER_QWEN_SNAPSHOT='/stale/3B', PIER_QWEN_DATA_PREFIX='/stale/data',
                    PIER_N2_PROFILE='main', PIER_N2_SUITE='cohorts', PIER_N2_ARMS='G,W',
                    PIER_N2_REPEATS='3', PIER_N2_REPEAT_START='2', PIER_N2_COHORT='16', PIER_N2_WORKSPACE_MIB='256')
                result = subprocess.run(['bash', str(script)], cwd=root, env=env,
                                        text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 23, result.stderr)
                logfile = next((repo / 'out').glob('n2-*/out.txt'))
                self.assertEqual(result.stdout, f'[n2] Full stdout/stderr: {logfile}\n')
                self.assertEqual(result.stderr, '')
                self.assertIn('worker-stderr', logfile.read_text())
                cfg = json.loads(next(line for line in logfile.read_text().splitlines() if line.startswith('{')))
                self.assertEqual(cfg, configuration({'PIER_ROOT': str(repo), 'SLURM_JOB_NUM_NODES': '8',
                    'PIER_QWEN_MODEL_SIZE': '1.5B', 'PIER_N2_ARMS': 'O,R,P',
                    'PIER_N2_EXPECTED_GPU': 'NVIDIA A100-SXM4-40GB'}))
                env.update(SLURM_JOB_NUM_NODES='1')
                rejected = subprocess.run(['bash', str(script)], cwd=root, env=env,
                                          text=True, capture_output=True, timeout=10)
                self.assertEqual(rejected.returncode, 2)
                self.assertIn('eight-node allocation', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
