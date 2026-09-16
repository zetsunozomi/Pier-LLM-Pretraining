"""Exercise shell exit/log handling with fake Slurm commands, never fake GPU success."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import json

ROOT = Path(__file__).resolve().parents[2]


class LauncherTest(unittest.TestCase):
    def test_remote_worker_failure_preserves_exit_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            def executable(name, code):
                path = tmp / name
                path.write_text('#!' + sys.executable + '\n' + code)
                path.chmod(0o700)
                return path
            executable('scontrol', 'print("127.0.0.1")\n')
            executable('srun', '''import subprocess, sys
args = sys.argv[1:]
while args and args[0].startswith('--'):
    args.pop(0)
sys.exit(subprocess.call(args))
''')
            fake_python = executable('worker-python', '''import subprocess, sys
if sys.argv[1:3] == ['-u', '-m']:
    print('injected worker failure: no GPU work was executed', flush=True)
    sys.exit(23)
sys.exit(subprocess.call([sys.executable, *sys.argv[1:]]))
''')
            env = dict(os.environ, PATH=str(tmp) + os.pathsep + os.environ['PATH'],
                       PIER_ROOT=str(ROOT), PIER_PYTHON=str(fake_python),
                       PIER_GPUS_PER_NODE='4', PIER_E0_RUN_DIR=str(tmp / 'output'),
                       SLURM_SUBMIT_DIR=str(ROOT), SLURM_JOB_NUM_NODES='1',
                       SLURM_NODEID='0', SLURM_JOB_ID='123456', SLURM_JOB_NODELIST='fake')
            result = subprocess.run(['bash', 'experiments/centered_outer/e0.sbatch'],
                                    cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
            summary = json.loads((tmp / 'output/summary.json').read_text())
            self.assertEqual(summary['status'], 'failed')
            self.assertTrue(any('exit code 23' in e for e in summary['errors']))
            self.assertIn('injected worker failure', (tmp / 'output/megatron-dp1.log').read_text())
            self.assertFalse((tmp / 'output/protocol.json').exists())


if __name__ == '__main__':
    unittest.main()
