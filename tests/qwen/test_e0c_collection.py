"""Collection may advance after finite mismatch; strict verdicts stay failed."""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiments.qwen.e0c import finish_native_comparison
from experiments.qwen.e0c_metrics import compare_tensor
from test_e0c import cpu_tensor_factory

ROOT = Path(__file__).resolve().parents[2]


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        baseline = compare_tensor(torch.ones(2), torch.ones(2), 'fp32', 'gradient')
        outcomes = {}
        for name, collect in [('passed', False), ('finite-default', False), ('finite-collect', True),
                              ('nonfinite', True), ('overflow', True), ('empty', True)]:
            stats = dict(baseline)
            passed = rank != 0 or name == 'passed'
            if rank == 0 and not passed:
                stats['outside_tolerance'] = 1
                stats['passed'] = False
                if name == 'nonfinite':
                    stats['nonfinite'] = 1
                if name == 'overflow':
                    stats['error_sq'] = float('inf')
            comparisons = [] if rank == 0 and name == 'empty' else [stats]
            with patch('torch.tensor', side_effect=cpu_tensor_factory(torch.tensor)):
                try:
                    finish_native_comparison(passed, comparisons, collect)
                    outcomes[name] = 'returned'
                except ValueError:
                    outcomes[name] = 'stopped'
        Path(directory, f'rank-{rank}.json').write_text(json.dumps(outcomes))
    finally:
        dist.destroy_process_group()


class CollectionTests(unittest.TestCase):
    def test_one_rank_failure_propagates_and_only_finite_collection_returns(self):
        with tempfile.TemporaryDirectory(prefix='e0c-collect-cpu-') as directory:
            mp.spawn(worker, args=(f'file://{directory}/rendezvous', directory), nprocs=4, join=True)
            for rank in range(4):
                result = json.loads(Path(directory, f'rank-{rank}.json').read_text())
                self.assertEqual(result, {'passed': 'returned', 'finite-default': 'stopped',
                    'finite-collect': 'returned', 'nonfinite': 'stopped',
                    'overflow': 'stopped', 'empty': 'stopped'})

    def test_launcher_collects_six_phases_but_stops_on_execution_error(self):
        for runtime_failure in (False, True):
            with self.subTest(runtime_failure=runtime_failure), tempfile.TemporaryDirectory(prefix='e0c-collect-launch-') as name:
                directory = Path(name)
                (directory / 'snapshot').mkdir()
                srun = directory / 'srun'
                srun.write_text('#!/usr/bin/env bash\nwhile [[ "$1" == --* ]]; do shift; done\nexec "$@"\n')
                interpreter = directory / 'fake-python'
                interpreter.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CALLS"
if [[ "$*" == *"--summarize"* ]]; then exit 1; fi
if [[ "$SIMULATE_RUNTIME_FAILURE" == 1 && "$*" == *"--phase native"* ]]; then exit 23; fi
exit 0
''')
                srun.chmod(0o755)
                interpreter.chmod(0o755)
                env = dict(os.environ, PIER_ROOT=str(ROOT), PIER_PYTHON=str(interpreter),
                    PIER_OUT_ROOT=str(directory / 'out'),
                    PIER_QWEN_SNAPSHOT=str(directory / 'snapshot'), PIER_E0C_RUN_DIR=str(directory / 'run'),
                    PIER_E0C_COLLECT_ALL='1', SLURM_JOB_NUM_NODES='1', SLURM_JOB_ID='fixture',
                    CALLS=str(directory / 'calls'), SIMULATE_RUNTIME_FAILURE=str(int(runtime_failure)),
                    PATH=str(directory) + os.pathsep + os.environ['PATH'])
                result = subprocess.run(['bash', 'experiments/qwen/e0c.sbatch'], cwd=ROOT, env=env,
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 23 if runtime_failure else 1, result.stdout + result.stderr)
                calls = (directory / 'calls').read_text().splitlines()
                natives = [x for x in calls if '--phase native' in x]
                references = [x for x in calls if '--phase reference' in x]
                self.assertEqual(len(natives), 1 if runtime_failure else 4)
                self.assertEqual(len(references), 1 if runtime_failure else 2)
                self.assertTrue(all('--collect-numerical-mismatches' in x for x in natives))
                self.assertTrue(all('--collect-numerical-mismatches' not in x for x in references))
                self.assertIn('--collect-numerical-mismatches', calls[0])
                self.assertIn('--launcher-exit 23' if runtime_failure else '--launcher-exit 0', calls[-1])


if __name__ == '__main__':
    unittest.main()
