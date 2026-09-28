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
from experiments.joint.training_gate import GATE_ENV, phase_plan, run as run_training_gate, summarize as summarize_training_gate
from experiments.joint.require_gate import check_training
from experiments.qwen.n2_config import configuration, cases as n2_cases, training_args
from megatron.core.outer_sync.joint_config import validate_config
from types import SimpleNamespace


class JointExperimentTests(unittest.TestCase):
    @staticmethod
    def completed_gate_phase(command, **kwargs):
        """Synthetic complete rank receipts exercise the real partial-gate validator."""
        output = Path(command[command.index('--output-dir') + 1])
        name = command[command.index('--worker') + 1]
        manifest = json.loads((output / 'manifest.json').read_text())
        phase = next(p for p in manifest['phases'] if p['name'] == name)
        events = []
        from megatron.core.outer_sync.joint_config import JointController
        for attempt in range(1, 12):
            event = dict(outer_boundary=attempt in (4, 7, 10), skipped=attempt == 3,
                         oracle_states_bitwise=True, outer_retained_inner_state=True,
                         skip_retained_optimizer=True, boundaries=(attempt - (attempt >= 3)) // 3)
            if phase['config'] and event['outer_boundary']:
                controller = JointController(phase['config'])
                event['payload'] = dict(configuration_sha256=controller.sha256,
                                        plan=dict(cohort=2), variant=phase['variant'])
            events.append(event)
        for rank in range(4):
            row = dict(GPU_executed=True, backend='nccl', world_size=4, status='passed',
                       clock=dict(interval=3, attempted=11, successful=10, boundaries=3),
                       restored=False, error=None, events=events, pending_consumer=False,
                       consumer_checks=3, torch_version='fixture')
            (output / name / f'rank-{rank}.json').write_text(json.dumps(row))
            launch = dict(argv=phase['argv'], GPU_executed=True, environment=manifest['environment'])
            (output / name / f'launch-rank-{rank}.json').write_text(json.dumps(launch))
        return SimpleNamespace(returncode=0)

    def test_training_gate_chunks_resume_and_do_not_pass_early(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('experiments.joint.training_gate.sources', return_value={'fixture': 'v1'}), \
                patch('experiments.joint.training_gate.subprocess.run', side_effect=self.completed_gate_phase) as launch:
            output = Path(folder) / 'gate'
            summary = run_training_gate(output, max_phases=2)
            self.assertEqual(summary['status'], 'incomplete')
            self.assertEqual(len(summary['pending_phases']), 11)
            self.assertEqual(len(summary['comparisons']), 4)
            with self.assertRaises(ValueError):
                check_training(output / 'summary.json')
            original = (output / 'reference-tp1' / 'exit.json').read_bytes()
            summary = run_training_gate(output, resume=True, max_phases=1)
            self.assertEqual(len(summary['pending_phases']), 10)
            self.assertEqual(len(summary['comparisons']), 8)
            self.assertEqual(launch.call_count, 3)
            self.assertEqual((output / 'reference-tp1' / 'exit.json').read_bytes(), original)
            for args in launch.call_args_list:
                self.assertIn('--gpus-per-node=4', args.args[0])
                limit = next(arg for arg in args.args[0] if arg.startswith('--time='))
                self.assertLessEqual(int(limit.split('=')[1]), 24)
            with patch('experiments.joint.training_gate.sources', return_value={'fixture': 'v2'}):
                with self.assertRaisesRegex(ValueError, 'source'):
                    run_training_gate(output, resume=True)
            self.assertEqual(launch.call_count, 3)
            (output / 'single-tp1' / 'rank-0.json').unlink()
            self.assertEqual(summarize_training_gate(output)['status'], 'failed')
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                run_training_gate(output, resume=True)
            self.assertEqual(launch.call_count, 3)

    def test_training_gate_stops_between_phases_before_time_budget(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('experiments.joint.training_gate.sources', return_value={'fixture': 'v1'}), \
                patch('experiments.joint.training_gate.subprocess.run', side_effect=self.completed_gate_phase) as launch, \
                patch('experiments.joint.training_gate.time.monotonic', side_effect=[0, 0, 0, 500, 500]):
            summary = run_training_gate(Path(folder) / 'gate', budget_seconds=600)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(summary['status'], 'incomplete')
            self.assertEqual(len(summary['pending_phases']), 12)

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
