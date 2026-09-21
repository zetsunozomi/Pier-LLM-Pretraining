"""Run actual shell launchers with fake workers, including Slurm spool copies.

These tests perform no GPU work and produce no experimental evidence.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(os.environ.get('PIER_LAUNCH_TEST_SOURCE', ROOT / 'experiments/qwen'))
REVISION = '3aab1f1954e9cc14eb9509a215f9e5ca08227a9b'


class LaunchPathTests(unittest.TestCase):
    def run_launcher(self, directory, stage, mode, *, missing_snapshot=False, invalid_root=False):
        directory = directory.resolve()
        repository = directory / 'Pier with spaces'
        scripts = repository / 'experiments/qwen'
        scripts.mkdir(parents=True)
        (repository / 'pretrain_qwen.py').touch()
        (scripts / f'{stage}_evidence.py').touch()
        script = scripts / f'{stage}.sbatch'
        shutil.copyfile(SCRIPTS / script.name, script)
        shutil.copyfile(SCRIPTS / 'log_output.sh', scripts / 'log_output.sh')
        prior_log = repository / 'out/previous-attempt/out.txt'
        prior_log.parent.mkdir(parents=True)
        prior_log.write_text('previous output\n')
        snapshot = repository / f'local/qwen/models/Qwen2.5-3B/{REVISION}'
        if not missing_snapshot:
            snapshot.mkdir(parents=True)
        unrelated = directory / 'unrelated'
        unrelated.mkdir()
        dispatcher = directory / 'srun'
        dispatcher.write_text('#!/usr/bin/env bash\nwhile [[ "$1" == --* ]]; do shift; done\nexec "$@"\n')
        dispatcher.chmod(0o755)
        interpreter = directory / 'fake-python'
        interpreter.write_text('''#!/usr/bin/env bash
printf '%s|%s|%s|%s\\n' "$PIER_ROOT" "$PIER_QWEN_SNAPSHOT" "$SLURM_OVERLAP" "$*" >> "$CALLS"
echo "worker stdout: $*"
echo "worker stderr: $*" >&2
if [[ "$*" == *"--phase reference"* || "$*" == *"--case s2-device"* ]]; then exit 23; fi
exit 0
''')
        interpreter.chmod(0o755)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PIER_', 'SLURM_'))}
        env.update(PIER_PYTHON=str(interpreter), SLURM_JOB_ID='42', SLURM_NNODES='1',
                   SLURM_SUBMIT_DIR=str(unrelated), CALLS=str(directory / 'calls'),
                   PATH=str(directory) + os.pathsep + os.environ['PATH'])
        if stage == 'e0d':
            prerequisite = directory / 'e0c'
            prerequisite.mkdir()
            env['PIER_E0C_EVIDENCE'] = str(prerequisite)
            env['PIER_QWEN_DATA_PREFIX'] = str(directory / 'train')
            for suffix in ('.bin', '.idx', '.manifest.json'):
                (directory / ('train' + suffix)).touch()
        cwd = unrelated
        if mode in ('batch', 'cwd', 'override'):
            spool = directory / 'spool/job42'
            spool.mkdir(parents=True)
            shutil.copyfile(script, spool / 'slurm_script')
            script = spool / 'slurm_script'
            if mode == 'batch':
                env['SLURM_SUBMIT_DIR'] = str(repository)
            elif mode == 'cwd':
                cwd = repository
            else:
                env['PIER_ROOT'] = str(repository)
        if invalid_root:
            env['PIER_ROOT'] = str(unrelated)
        result = subprocess.run(['bash', str(script)], cwd=cwd, env=env, capture_output=True,
                                text=True, timeout=15)
        calls = (directory / 'calls').read_text() if (directory / 'calls').exists() else ''
        return result, calls, repository, snapshot

    def test_batch_spool_uses_submission_repository(self):
        self.check_modes('batch')

    def test_interactive_script_ignores_unrelated_allocation_directory(self):
        self.check_modes('interactive')

    def test_explicit_repository_and_working_directory_fallback(self):
        self.check_modes('override', 'cwd')

    def check_modes(self, *modes):
        for mode in modes:
            for stage in ('e0c', 'e0d'):
                with self.subTest(mode=mode, stage=stage), tempfile.TemporaryDirectory(prefix='pier-launch-') as name:
                    result, calls, repository, snapshot = self.run_launcher(Path(name), stage, mode)
                    self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                    logs = list((repository / 'out').glob(f'{stage}-*/out.txt'))
                    self.assertEqual(len(logs), 1)
                    log = logs[0].read_text()
                    self.assertEqual(result.stdout, f'[{stage}] Full stdout/stderr: {logs[0]}\n')
                    self.assertEqual(result.stderr, '')
                    self.assertIn(f'Repository: {repository}', log)
                    self.assertIn(f'Snapshot: {snapshot}', log)
                    self.assertIn('worker stdout:', log)
                    self.assertIn('worker stderr:', log)
                    self.assertIn('--launcher-exit 23', log)
                    self.assertIn('FAILED', log)
                    self.assertIn('artifacts:', log)
                    self.assertEqual((repository / 'out/previous-attempt/out.txt').read_text(), 'previous output\n')
                    self.assertIn(f'{repository}|{snapshot}|1|', calls)
                    self.assertIn('--launcher-exit 23', calls)
                    self.assertNotIn('--phase native', calls)
                    self.assertNotIn('--case resume', calls)

    def test_missing_snapshot_prints_exact_directory_without_running_workers(self):
        for stage in ('e0c', 'e0d'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory(prefix='pier-launch-') as name:
                result, calls, repository, snapshot = self.run_launcher(Path(name), stage, 'batch', missing_snapshot=True)
                self.assertEqual(result.returncode, 2)
                log = next((repository / 'out').glob(f'{stage}-*/out.txt')).read_text()
                self.assertIn(str(snapshot), log)
                self.assertIn('Missing', log)
                self.assertEqual(result.stderr, '')
                self.assertEqual(calls, '')

    def test_invalid_explicit_root_is_not_silently_replaced(self):
        for stage in ('e0c', 'e0d'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory(prefix='pier-launch-') as name:
                result, calls, _, _ = self.run_launcher(Path(name), stage, 'interactive', invalid_root=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('Cannot locate Pier repository', result.stderr)
                self.assertEqual(calls, '')


if __name__ == '__main__':
    unittest.main()
