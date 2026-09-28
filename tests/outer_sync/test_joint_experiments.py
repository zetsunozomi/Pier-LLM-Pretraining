"""Experiment contracts: independent repeats, frozen recipes and budget checks."""

import copy
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.joint.benchmark import cases
from experiments.joint.qwen import configurations
from experiments.joint.summarize import aggregate
from experiments.joint.qwen_summary import boundaries
from experiments.joint.training_gate import GATE_ENV, phase_plan
from experiments.qwen.n2_config import configuration, cases as n2_cases, training_args
from megatron.core.outer_sync.joint_config import validate_config
from types import SimpleNamespace


class JointExperimentTests(unittest.TestCase):
    def test_real_training_gate_has_both_restart_boundaries_and_topologies(self):
        rows = {r['name']: r for r in phase_plan('/tmp/test-joint-gate')}
        self.assertEqual(rows['split-step']['stop'], 5)
        self.assertEqual(rows['split-boundary']['stop'], 4)
        self.assertEqual(rows['resume-step']['resume'], 'split-step')
        self.assertEqual(rows['resume-boundary']['resume'], 'split-boundary')
        for split, resume in (('split-step', 'resume-step'), ('split-boundary', 'resume-boundary')):
            self.assertEqual(rows[split]['config'], rows[resume]['config'])
        self.assertEqual({r['topology'] for r in rows.values()}, {'tp1', 'tp2', 'dp2'})
        for row in rows.values():
            self.assertIn('--outer-verify', row['argv'])
            if row['config']:
                validate_config(row['config'])

    def test_training_gate_uses_real_megatron_parser(self):
        from megatron.training.arguments import parse_args, validate_args
        with tempfile.TemporaryDirectory() as folder:
            for phase in phase_plan(folder):
                if phase['config']:
                    (Path(folder) / f"{phase['name']}.json").write_text(json.dumps(phase['config']))
                with patch.dict(os.environ, {**GATE_ENV, 'WORLD_SIZE': '4', 'RANK': '0'}), \
                     patch.object(sys, 'argv', ['pretrain_gpt.py', *phase['argv']]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    args = validate_args(parse_args())
                self.assertTrue(args.outer_verify)
                self.assertEqual(args.train_iters, 11)
                self.assertEqual(args.tensor_model_parallel_size, 2 if phase['topology'] == 'tp2' else 1)
                if phase['config']:
                    self.assertEqual(args.outer_pier_schedule, 'joint')

    def test_training_recipe_preserves_baselines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            value = configurations('budgets')[-1]['config']
            path.write_text(json.dumps(value))
            baseline = configuration({'PIER_N2_ARMS': 'G,OS,R,W,P', 'SLURM_NNODES': '8'})
            joint = configuration({'PIER_N2_ARMS': 'G,OS,R,W,P', 'SLURM_NNODES': '8',
                                   'PIER_N2_PIER_SCHEDULE': 'joint', 'PIER_N2_JOINT_CONFIG': str(path)})
            for case in n2_cases(baseline):
                original = training_args(baseline, case, folder)
                updated = training_args(joint, case, folder)
                if case['arm'] == 'P':
                    for flag in ('--outer-pier-schedule', '--outer-joint-config'):
                        index = updated.index(flag)
                        del updated[index:index + 2]
                self.assertEqual(original, updated)
            extended = configuration({'PIER_N2_PROFILE': 'main', 'PIER_N2_MEASURED_CYCLES': '8'})
            self.assertEqual(extended['attempts'], 500)
            with self.assertRaises(ValueError):
                configuration({'PIER_N2_PROFILE': 'main', 'PIER_N2_MEASURED_CYCLES': '9'})

    def test_four_variants_fixed_budget_and_forced_edges(self):
        for scenario in ('fixed', 'budgets', 'forced'):
            rows = configurations(scenario)
            self.assertEqual({r['config']['variant'] for r in rows}, {'single', 'pipeline', 'separate', 'joint'})
            for row in rows:
                validate_config(row['config'])
                self.assertFalse(row['config']['trace'])
            if scenario == 'forced':
                self.assertEqual([b['target'] for b in rows[-1]['config']['budgets']], [1, 2, 16, 2, 1])
        for scenario in ('pipeline', 'transitions'):
            args = SimpleNamespace(cohorts=None, scenario=scenario, capacities=[3, 6], slots=[1, 2], repeat_id=1)
            rows = cases(args, 4)
            self.assertEqual(len({json.dumps(r, sort_keys=True) for r in rows}), len(rows))
            self.assertEqual({r['mode'] for r in rows}, {'static', 'joint', 'separate'})
            if scenario == 'pipeline':
                self.assertTrue(all(r['old_cohort'] == r['cohort'] for r in rows))

    def test_independence_and_recipe_mismatch(self):
        case = dict(old_cohort=1, cohort=2, capacity=3, slots=2, mode='joint')
        template = dict(kind='configuration', status='complete', sources={'executor': 'hash'},
                        world_size=4, GPU_executed=False, config={'tp': 1, 'output': 'unused', 'repeat_id': 1},
                        hardware=[dict(device='CPU', torch='test', cuda=None)],
                        records=[dict(case=case, status='measured', launch_mean_seconds=1.)])
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for i in range(3):
                row = copy.deepcopy(template)
                row.update(launch_id=f'launch-{i}', repeat_id=i + 1)
                row['records'][0]['launch_mean_seconds'] = i + 1.
                path = Path(folder) / f'{i}.json'
                path.write_text(json.dumps(row))
                paths.append(path)
            summary = aggregate(paths)
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['configurations'][0]['independent_launches'], 3)
            self.assertEqual(summary['configurations'][0]['median_seconds'], 2.)
            self.assertEqual(aggregate(paths[:1])['status'], 'incomplete')
            with self.assertRaises(ValueError):
                aggregate([paths[0], paths[0], paths[1]])
            row['sources']['executor'] = 'changed'
            paths[-1].write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                aggregate(paths)

    def test_static_budget_checks_use_actual_peak(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for rank in range(2):
                row = dict(outer_boundary_index=1, warmup=False, eligible=True,
                           local_outer_and_commit_seconds=2. + rank, local_cycle_seconds=10. + rank,
                           torch_peak_allocated_bytes=(30 + rank) * 2**20,
                           torch_peak_reserved_bytes=32 * 2**20, payload=None)
                (root / f'cycles-rank-{rank}.json').write_text(json.dumps({'cycles': [row]}))
            report = boundaries(root, 2, [dict(round=1, next_phase_mib=30)])
            self.assertFalse(report[0]['budget_feasible'])
            self.assertEqual(report[0]['outer_seconds'], 3.)
            self.assertEqual(report[0]['cycle_seconds'], 11.)


if __name__ == '__main__':
    unittest.main()
