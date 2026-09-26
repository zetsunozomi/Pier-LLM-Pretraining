"""Verify launch sequencing and report interpretation without a Slurm cluster."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.centered_outer.summarize_strict import summarize

ROOT = Path(__file__).resolve().parents[2]


def fixtures(world=4, tp=1, samples=3):
    gate = {'status': 'passed', 'schedule': 'contiguous', 'GPU_executed': True,
            'ranks': world // tp, 'production_executor_sha256': 'fixture-hash'}
    report = {'GPU_executed': True, 'world_size': world, 'config': {'tp': tp, 'samples': samples, 'repeats': 1},
              'scope': 'synthetic fixture', 'repeat_scope': 'one launch',
              'sources': {'megatron/core/outer_sync/executor.py': 'fixture-hash'}, 'records': []}
    for arm, seconds in [('reference', 10.), ('contiguous', 8.), ('recenter', 5.)]:
        report['records'].append({
            'arm': arm, 'repeat': 1, 'max_rank_seconds': [seconds] * samples,
            'rank_records': [{'rank': rank, 'seconds': [seconds] * samples, 'finite_state': True,
                             'peak_allocated_bytes': (rank + 1) * 2**30} for rank in range(world)]})
    return gate, report


def write_reports(directory, gate, report):
    (directory / 'correctness.json').write_text(json.dumps(gate))
    (directory / 'benchmark.json').write_text(json.dumps(report))


class SummaryTests(unittest.TestCase):
    def test_ratios_use_outer_time_and_maximum_rank_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_reports(directory, *fixtures())
            result = summarize(directory)
        row = result['comparisons'][0]
        self.assertAlmostEqual(row['new_vs_old_latency_reduction_pct'], 20.)
        self.assertAlmostEqual(row['old_vs_relaxed_latency_overhead_pct'], 100.)
        self.assertAlmostEqual(row['new_vs_relaxed_latency_overhead_pct'], 60.)
        self.assertEqual(result['by_arm']['contiguous']['peak_allocated_gib'], 4.)

    def test_slower_new_schedule_remains_visible(self):
        gate, report = fixtures()
        row = report['records'][1]
        row['max_rank_seconds'] = [12.] * 3
        for rank in row['rank_records']:
            rank['seconds'] = [12.] * 3
        with tempfile.TemporaryDirectory() as tmp:
            write_reports(Path(tmp), gate, report)
            result = summarize(tmp)
        self.assertAlmostEqual(result['comparisons'][0]['new_vs_old_latency_reduction_pct'], -20.)

    def test_rejects_mismatched_gate_or_incomplete_measurement(self):
        gate, report = fixtures()
        bad_hash = copy.deepcopy(gate); bad_hash['production_executor_sha256'] = 'other'
        bad_ranks = copy.deepcopy(report); bad_ranks['records'][0]['rank_records'].pop()
        bad_pairs = copy.deepcopy(report); bad_pairs['records'].pop()
        bad_state = copy.deepcopy(report); bad_state['records'][0]['rank_records'][0]['finite_state'] = False
        for g, r in ((bad_hash, report), (gate, bad_ranks), (gate, bad_pairs), (gate, bad_state)):
            with self.subTest(), tempfile.TemporaryDirectory() as tmp:
                write_reports(Path(tmp), g, r)
                with self.assertRaises(ValueError):
                    summarize(tmp)


class LauncherTests(unittest.TestCase):
    def launch(self, directory, profile, nodes, fail_gate=False):
        binpath = directory / 'bin'; binpath.mkdir()
        gate, report = fixtures(nodes * 4, 2 if profile == 'flat32' else 1, samples=10)
        write_reports(directory, gate, report)
        srun = binpath / 'srun'
        srun.write_text(f'#!{sys.executable}\nimport sys,subprocess\n'
                        'sys.exit(subprocess.call(sys.argv[sys.argv.index("bash"):]))\n')
        control = binpath / 'scontrol'
        control.write_text('#!/bin/sh\nprintf "test-node-0\\n"\n')
        python = binpath / 'mock-python'
        python.write_text(f'''#!{sys.executable}
import json, os, pathlib, subprocess, sys
args=sys.argv[1:]
base=pathlib.Path(os.environ['STRICT_TEST_DIR'])
if args[:2] != ['-m', 'torch.distributed.run']:
    sys.exit(subprocess.call([{sys.executable!r}, *args]))
with (base/'calls.jsonl').open('a') as stream:
    stream.write(json.dumps(args)+'\\n')
gate=any(a.endswith('verify_executor.py') for a in args)
if gate and os.environ.get('STRICT_FAIL_GATE')=='1':
    sys.exit(17)
kind='correctness.json' if gate else 'benchmark.json'
pathlib.Path(args[args.index('--output')+1]).write_text((base/kind).read_text())
''')
        for p in (srun, control, python): p.chmod(0o755)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'PIER_'))}
        env.update(PATH=f'{binpath}:{env["PATH"]}', PIER_ROOT=str(ROOT), PIER_PYTHON=str(python),
                   PIER_OUT_ROOT=str(directory / 'out'), PIER_STRICT_PROFILE=profile,
                   SLURM_JOB_NUM_NODES=str(nodes), SLURM_JOB_ID='12345', SLURM_JOB_NODELIST='test-node-0',
                   STRICT_TEST_DIR=str(directory), STRICT_FAIL_GATE='1' if fail_gate else '0')
        return subprocess.run(['bash', str(ROOT / 'experiments/centered_outer/strict_bench.sbatch')],
                              env=env, text=True, capture_output=True, timeout=20)

    def test_smoke_and_flat32_route_workers_and_capture_output(self):
        for profile, nodes, gate_workers, tp, elements in [
                ('smoke', 1, 4, '1', '16777217'), ('flat32', 8, 2, '2', '1543044096')]:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                completed = self.launch(directory, profile, nodes)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                calls = [json.loads(line) for line in (directory / 'calls.jsonl').read_text().splitlines()]
                self.assertEqual(len(calls), 2)
                self.assertIn(f'--nproc_per_node={gate_workers}', calls[0])
                self.assertIn('--nproc_per_node=4', calls[1])
                self.assertIn(f'--nnodes={nodes}', calls[1])
                self.assertEqual(calls[1][calls[1].index('--tp') + 1], tp)
                self.assertEqual(calls[1][calls[1].index('--elements') + 1], elements)
                result_dir = next((directory / 'out').iterdir())
                self.assertTrue((result_dir / 'results.txt').is_file())
                self.assertIn('phase=complete exit=0', (result_dir / 'out.txt').read_text())
                self.assertIn(str(result_dir / 'out.txt'), completed.stdout)

    def test_correctness_failure_stops_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            completed = self.launch(directory, 'smoke', 1, fail_gate=True)
            self.assertEqual(completed.returncode, 17)
            self.assertEqual(len((directory / 'calls.jsonl').read_text().splitlines()), 1)
            result_dir = next((directory / 'out').iterdir())
            self.assertFalse((result_dir / 'benchmark.json').exists())
            self.assertIn('phase=correctness exit=17', (result_dir / 'out.txt').read_text())

    def test_reject_wrong_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            completed = self.launch(directory, 'flat32', 1)
            self.assertEqual(completed.returncode, 2)
            self.assertFalse((directory / 'calls.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
