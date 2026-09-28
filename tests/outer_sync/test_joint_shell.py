"""Run the real submission shells with recording Slurm/Python stand-ins."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class JointShellTests(unittest.TestCase):
    def invoke(self, script, nodes, extra=(), missing_training=False, resume=None):
        with tempfile.TemporaryDirectory(prefix='pier-joint-shell-') as folder:
            root = Path(folder)
            capture = root / 'args.txt'
            tools = root / 'bin'
            tools.mkdir()
            for name, content in {
                'scontrol': '#!/bin/bash\necho fixture-node\n',
                'srun': '#!/bin/bash\nprintf "srun\\n%s\\n" "$*" >> "$CAPTURE"\n',
                'python-recorder': '#!/bin/bash\nprintf "python\\n%s\\n" "$*" >> "$CAPTURE"\n',
            }.items():
                path = tools / name
                path.write_text(content)
                path.chmod(0o755)
            # Slurm runs a spool copy; BASH_SOURCE cannot locate the repository.
            spool = root / 'slurm_script'
            shutil.copyfile(ROOT / 'experiments/joint' / script, spool)
            env = {**os.environ, 'PATH': str(tools) + os.pathsep + os.environ['PATH'],
                   'CAPTURE': str(capture), 'SLURM_SUBMIT_DIR': str(ROOT),
                   'SLURM_JOB_NUM_NODES': str(nodes), 'SLURM_JOB_NODELIST': 'fixture',
                   'SLURM_JOB_ID': '901', 'SLURM_ARRAY_TASK_ID': '2',
                   'PIER_PYTHON': str(tools / 'python-recorder'), 'PIER_OUT_ROOT': str(root / 'out'),
                   'PIER_JOINT_GATE': '/fixture/operator.json'}
            env.pop('PIER_ROOT', None)
            env.pop('PIER_JOINT_TRAINING_GATE', None)
            env.pop('PIER_JOINT_TRAINING_RESUME', None)
            if resume:
                env['PIER_JOINT_TRAINING_RESUME'] = resume
            if not missing_training:
                env['PIER_JOINT_TRAINING_GATE'] = '/fixture/training.json'
            result = subprocess.run(['bash', str(spool), *extra], cwd=root, env=env,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return result, capture.read_text() if capture.exists() else ''

    def test_outer_array_uses_real_stage_repeat_and_gate(self):
        result, captured = self.invoke('outer.sbatch', 8, ('transitions', '--slots', '2'))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('require_gate.py /fixture/operator.json --world 32 --tp 2', captured)
        self.assertIn('--scenario transitions --repeat-id 2 --slots 2', captured)
        self.assertIn('--nodes=8 --ntasks=8', captured)

    def test_qwen_routes_repeat_without_separate_gate_jobs(self):
        result, captured = self.invoke('qwen.sbatch', 8, ('budgets', '--only', 'joint-s1'))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('--repeat-id 2', captured)
        self.assertNotIn('--gate', captured)
        self.assertNotIn('--training-gate', captured)
        self.assertIn('--only joint-s1', captured)
        result, _ = self.invoke('qwen.sbatch', 8, missing_training=True)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_paper_job_runs_one_full_case_with_explicit_repeat(self):
        result, captured = self.invoke('paper.sbatch', 8, ('joint-s1', '1'), missing_training=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('--scenario fixed --only joint-s1 --repeat-id 1', captured)
        self.assertNotIn('--gate', captured)
        self.assertNotIn('--training-gate', captured)
        for nodes, extra in ((1, ()), (8, ('native-controls',)), (8, ('joint-s1', '4'))):
            result, _ = self.invoke('paper.sbatch', nodes, extra)
            self.assertNotEqual(result.returncode, 0)

    def test_training_gate_rejects_wrong_allocation(self):
        result, captured = self.invoke('training_gate.sbatch', 1, ('--max-phases', '2'))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('training_gate.py --output-dir', captured)
        self.assertIn('--max-phases 2', captured)
        result, captured = self.invoke('training_gate.sbatch', 1, ('--max-phases', '3'), resume='/fixture/gate')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('--output-dir /fixture/gate --resume --max-phases 3', captured)
        result, _ = self.invoke('training_gate.sbatch', 8)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
