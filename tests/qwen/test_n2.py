"""N2 recipe, launch routing and arithmetic tests; no GPU performance evidence."""

import contextlib
import hashlib
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
import uuid

from experiments.qwen.n2_config import ROOT, configuration, cases, launch_command, training_args
from experiments.qwen.n2_summary import collect, copy_historical_reference, render
from experiments.qwen import n2
from experiments.qwen.preflight import pinned_architecture
from megatron.core.models.qwen.training import training_defaults, validate_training_contract
from megatron.core.outer_sync.cycle_metrics import FORMAT, TOKEN_SEMANTICS, TIMING_SCOPE, EXCLUDED_FROM_CYCLE
from megatron.training.arguments import parse_args, validate_args
from pretrain_qwen import data_identity


class N2Tests(unittest.TestCase):
    def test_real_parser_preserves_recipe_on_four_and_32_gpus(self):
        _, arch = pinned_architecture('3B')
        for nodes in (1, 8):
            for profile in ('pilot', 'main'):
                cfg = configuration({'SLURM_NNODES': str(nodes), 'PIER_N2_PROFILE': profile,
                                     'PIER_N2_ARMS': 'G,O,R,W,P'})
                self.assertEqual(cfg['global_batch'] // cfg['learners'], 8)
                self.assertEqual(cfg['attempts'], cfg['interval'] * (cfg['warmup_cycles'] + cfg['measured_cycles']))
                for case in cases(cfg)[:5]:
                    argv = training_args(cfg, case, '/tmp/n2-fixture')
                    def extra(parser):
                        parser.add_argument('--qwen-model-size')
                        parser.add_argument('--qwen-snapshot')
                        parser.add_argument('--qwen-trace-dir')
                        parser.add_argument('--qwen-synthetic-benchmark', action='store_true')
                        parser.set_defaults(**training_defaults(arch), tokenizer_type='HuggingFaceTokenizer')
                        return parser
                    with patch.dict(os.environ, WORLD_SIZE=str(nodes * 4), RANK='0', CUDA_DEVICE_MAX_CONNECTIONS='1'), \
                         patch.object(sys, 'argv', ['pretrain_qwen.py', *argv]), contextlib.redirect_stdout(io.StringIO()):
                        args = validate_args(parse_args(extra))
                        validate_training_contract(args, arch)
                    self.assertFalse(args.outer_verify)
                    self.assertIsNone(args.save)
                    self.assertEqual(args.num_subgroup, nodes * 2)
                    self.assertEqual(args.outer_sync_interval, 50)
                    self.assertEqual(args.outer_cohort_size, 2 if case['arm'] == 'P' else 1)
                    self.assertEqual(args.outer_cpu_offload, case['arm'] == 'O')
        cfg = configuration({'PIER_N2_PROFILE': 'main'})
        self.assertEqual(cases(cfg), cases(cfg))
        self.assertEqual(len(cases(cfg)), 12)
        for env in ({'PIER_N2_COHORT': '3'}, {'SLURM_NNODES': '3'}, {'PIER_N2_ARMS': 'G,G'}):
            with self.assertRaises(ValueError):
                configuration(env)

    def test_synthetic_input_is_explicit_and_real_data_path_is_preserved(self):
        args = SimpleNamespace(mock_data=True, qwen_synthetic_benchmark=True,
                               outer_measure_dir='/tmp/n2', seed=1234)
        self.assertEqual(data_identity(args, {})['kind'], 'synthetic_tokens')
        args.qwen_synthetic_benchmark = False
        with self.assertRaises(ValueError):
            data_identity(args, {})
        cfg = configuration({'PIER_QWEN_DATA_PREFIX': '/scratch/train'})
        argv = training_args(cfg, cases(cfg)[0], '/tmp/n2')
        self.assertNotIn('--mock-data', argv)
        self.assertIn('/scratch/train', argv)

    def test_torchrun_single_and_multinode_routes_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='pier n2 ') as name:
            root = Path(name)
            fake = root / 'fake-python'
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            for nodes in (1, 8):
                env = dict(os.environ, PIER_ROOT=str(ROOT), PIER_PYTHON=str(fake), PIER_N2_NODES=str(nodes),
                           SLURM_PROCID='3', PIER_N2_MASTER_ADDR='nid-test', PIER_N2_MASTER_PORT='29541')
                result = subprocess.run(['bash', str(ROOT / 'experiments/qwen/n2_node.sh'), name, 'run-1-G'],
                                        env=env, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                argv = json.loads(result.stdout)
                self.assertEqual(argv[-3:], [name, '--case', 'run-1-G'])
                self.assertIn('--nproc_per_node=4', argv)
                self.assertIn('--standalone' if nodes == 1 else '--node_rank=3', argv)
                cfg = configuration({'SLURM_NNODES': str(nodes)})
                command = launch_command(cfg, cases(cfg)[0], name)
                self.assertIn(f'--ntasks={nodes}', command)
                self.assertIn('--gpu-bind=none', command)

    def test_entrypoint_interactive_and_spooled_batch_capture_output(self):
        for stage, batch in ((stage, batch) for stage in ('n2', 'n3', 'n2_main') for batch in (False, True)):
            with self.subTest(stage=stage, batch=batch), tempfile.TemporaryDirectory(prefix='pier n2 launch ') as name:
                root = Path(name)
                repo = root / 'repo with spaces'
                scripts = repo / 'experiments/qwen'
                scripts.mkdir(parents=True)
                (repo / 'pretrain_qwen.py').touch()
                (scripts / 'n2.py').touch()
                for file in ('n2.sbatch', 'n3.sbatch', 'n2_main.sbatch', 'log_output.sh'):
                    shutil.copyfile(ROOT / 'experiments/qwen' / file, scripts / file)
                script = scripts / f'{stage}.sbatch'
                if batch:
                    spool = root / 'spool/job'
                    spool.mkdir(parents=True)
                    shutil.copyfile(script, spool / 'slurm_script')
                    script = spool / 'slurm_script'
                fake = root / 'python'
                fake.write_text('#!/bin/bash\necho "root=$PIER_ROOT overlap=$SLURM_OVERLAP suite=${PIER_N2_SUITE:-baselines}"\n'
                                'echo "arms=${PIER_N2_ARMS:-} repeat=${PIER_N2_REPEAT_START:-} repeats=${PIER_N2_REPEATS:-}"\n'
                                'echo worker-stderr >&2\nexit 23\n')
                fake.chmod(0o755)
                env = {k: v for k, v in os.environ.items() if not k.startswith(('PIER_', 'SLURM_'))}
                env.update(PIER_PYTHON=str(fake), SLURM_JOB_ID='42', SLURM_NNODES='1',
                           SLURM_SUBMIT_DIR=str(repo if batch else root))
                if stage == 'n2_main':
                    env.update(SLURM_ARRAY_TASK_ID='2', PIER_N2_REPEATS='3', PIER_N2_ARMS='G,W',
                               PIER_N2_REPEAT_START='1', PIER_N2_SUITE='cohorts')
                result = subprocess.run(['bash', str(script)], cwd=root, env=env, text=True,
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 23, result.stderr)
                prefix = 'n2' if stage == 'n2_main' else stage
                logfile = next((repo / 'out').glob(f'{prefix}-*/out.txt'))
                self.assertEqual(result.stdout, f'[{prefix}] Full stdout/stderr: {logfile}\n')
                self.assertEqual(result.stderr, '')
                self.assertIn(f'root={repo} overlap=1', logfile.read_text())
                self.assertIn('worker-stderr', logfile.read_text())
                self.assertIn('suite=cohorts' if stage == 'n3' else 'suite=baselines', logfile.read_text())
                if stage == 'n2_main':
                    self.assertIn('arms=O,R,P repeat=2 repeats=1', logfile.read_text())

    def test_split_main_jobs_preserve_global_repeats_and_original_method_order(self):
        env = {'PIER_N2_PROFILE': 'main', 'PIER_N2_ARMS': 'O,R,P', 'SLURM_NNODES': '8'}
        full = cases(configuration(env))
        split = []
        for repeat in (1, 2, 3):
            cfg = configuration({**env, 'PIER_N2_REPEATS': '1', 'PIER_N2_REPEAT_START': str(repeat)})
            split.extend(cases(cfg))
            self.assertEqual(cfg['attempts'], 250)
        self.assertEqual(split, full)
        self.assertEqual({c['repeat'] for c in split}, {1, 2, 3})
        for start, repeats in (('0', '1'), ('4', '1'), ('2', '3')):
            with self.assertRaises(ValueError):
                configuration({**env, 'PIER_N2_REPEAT_START': start, 'PIER_N2_REPEATS': repeats})

    def test_real_subprocess_failure_does_not_stop_other_arms(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'experiments/qwen').mkdir(parents=True)
            (root / 'experiments/qwen/pins.json').write_text(json.dumps({
                'models': {'3B': {'files': {'test-config': {'bytes': 1}}}}}))
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            (snapshot / 'test-config').write_text('x')
            cfg = configuration({'PIER_QWEN_SNAPSHOT': str(snapshot)})
            def command(config, case, output, **kwargs):
                code = 23 if case['arm'] == 'G' else 0
                return [sys.executable, '-c', f'print("fake CPU launcher only"); raise SystemExit({code})']
            with patch.object(n2, 'ROOT', root), patch.object(n2, 'configuration', return_value=cfg), \
                 patch.object(n2, 'source_identity', return_value={}), patch.object(n2, 'launch_command', command), \
                 patch.object(n2.subprocess, 'check_output', return_value='test-revision'), \
                 contextlib.redirect_stdout(io.StringIO()):
                status = n2.run(root / 'out')
            self.assertEqual(status, 1)
            for case in cases(cfg):
                receipt = json.loads((root / 'out' / case['id'] / 'exit.json').read_text())
                self.assertEqual(receipt['exit_code'], 23 if case['arm'] == 'G' else 0)
            result = json.loads((root / 'out/summary.json').read_text())
            self.assertEqual([c['status'] for c in result['cases']], ['failed', 'invalid', 'invalid', 'invalid'])
            self.assertTrue((root / 'out/results.txt').exists())

    def make_reports(self, root, cfg=None):
        # Controlled numbers solely exercise the collector's GPU JSON schema.
        cfg = cfg or configuration({'PIER_N2_ARMS': 'G,P,R,W'})
        manifest = {'config': cfg, 'cases': cases(cfg), 'output_directory': str(root)}
        path = root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        selected = manifest['cases'] if cfg.get('suite') == 'cohorts' else manifest['cases'][:2]
        tokens = cfg['global_batch'] * cfg['sequence'] * cfg['interval']
        for case in selected:
            directory = root / case['id']
            directory.mkdir()
            (directory / 'exit.json').write_text('{"exit_code":0}')
            run_id = str(uuid.uuid4())
            seconds = 20. if case['arm'] in ('G', 'O') else 10.
            if cfg.get('suite') == 'cohorts':
                seconds = 10. * case['cohort']
            for rank in range(cfg['world_size']):
                rows = []
                for cycle in range(2):
                    rows.append(dict(outer_boundary_index=cycle + 1, complete_cycle=True, warmup=cycle == 0, eligible=cycle == 1,
                        attempted_start=cycle * 50, attempted_end=(cycle + 1) * 50,
                        successful_start=cycle * 50, successful_end=(cycle + 1) * 50,
                        attempts=50, successful_steps=50, starts_midcycle=False, ends_at_outer_boundary=True,
                        local_cycle_seconds=seconds, max_rank_cycle_seconds=seconds,
                        local_outer_and_commit_seconds=1., max_rank_outer_and_commit_seconds=1.,
                        processed_loss_tokens_global=tokens, successful_loss_tokens_global=tokens,
                        skipped_loss_tokens_global=0, useful_tokens_per_second=tokens / seconds if cycle else None,
                        payload=None, torch_peak_allocated_bytes=(rank + 1) * 2**30,
                        torch_peak_reserved_bytes=(rank + 2) * 2**30,
                        device_level_peak_bytes=None, physical_wire_bytes=None))
                report = dict(format=FORMAT, token_semantics=TOKEN_SEMANTICS, timing_scope=TIMING_SCOPE,
                    excluded_from_cycle=EXCLUDED_FROM_CYCLE, run_id=run_id, rank=rank, world_size=cfg['world_size'],
                    status='complete', GPU_executed=True, performance_result=False, planned_attempts=100,
                    initial_clock=dict(interval=50, attempted=0, successful=0, boundaries=0),
                    final_clock=dict(interval=50, attempted=100, successful=100, boundaries=2),
                    warmup_cycles=1, complete_cycles=2, eligible_cycles=1, cycles=rows,
                    metadata=dict(arm=case['backend'], cohort=case['cohort'], tile_elements=1024, allocation={},
                                  final_health=dict(model_matches_master=True, finite_model=True,
                                                    finite_loss=True, losses={'lm loss': 2.})))
                if case['arm'] == 'O':
                    report['metadata']['allocation'] = dict(reference_bytes=2 * 2**30, momentum_bytes=2**30)
                    report['metadata']['state_storage'] = {
                        'reference': dict(device='cpu', bytes=2 * 2**30, pinned=True),
                        'momentum': dict(device='cpu', bytes=2**30, pinned=True)}
                (directory / f'cycles-rank-{rank}.json').write_text(json.dumps(report))
                worker = dict(rank=rank, case=case, GPU_executed=True, manifest_sha256=fingerprint,
                              argv=training_args(cfg, case, directory), hostname='fixture-only', local_rank=rank,
                              gpu='FAKE: test fixture', gpu_total_bytes=40 * 2**30, python='fixture', torch='fixture',
                              cuda='fixture', transformers='fixture', safetensors='fixture', tf32=False,
                              bf16_reduced_precision_reduction=True, nccl_algo='automatic')
                (directory / f'worker-rank-{rank}.json').write_text(json.dumps(worker))
                (directory / f'initialization-rank-{rank}.json').write_text(json.dumps({
                    'loaded_before_optimizer_construction': True, 'qwen_recipe': {'fixture': 'not real measurements'}}))
        return manifest

    def test_offload_summary_requires_actual_pinned_host_storage_and_same_repeat(self):
        cfg = configuration({'PIER_N2_ARMS': 'O,P', 'PIER_N2_REPEAT_START': '2'})
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.make_reports(root, cfg)
            result = collect(root)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual([r['speedup_vs_O'] for r in result['cases']], [1., 2.])
            self.assertEqual(result['cases'][0]['host_outer_state_gib_max_rank'], 3.)
            self.assertIn('speedup/O', render(result))
            path = root / 'run-2-O/cycles-rank-3.json'
            original = json.loads(path.read_text())
            for wrong in ({'device': 'cuda'}, {'pinned': False}, {'bytes': 1}):
                report = json.loads(json.dumps(original))
                report['metadata']['state_storage']['reference'].update(wrong)
                path.write_text(json.dumps(report))
                invalid = collect(root)
                self.assertEqual(invalid['cases'][0]['status'], 'invalid')
                self.assertIsNone(invalid['cases'][1]['speedup_vs_O'])

    def test_cohort_sweep_keeps_each_configuration_separate_and_uses_current_anchor(self):
        cfg = configuration({'SLURM_NNODES': '2', 'PIER_N2_SUITE': 'cohorts',
                             'PIER_N2_EXPECTED_GPU': 'FAKE: test fixture'})
        self.assertEqual([c['cohort'] for c in cases(cfg)], [1, 2, 4])
        self.assertEqual([c['id'] for c in cases(cfg)], ['run-1-P-s1', 'run-1-P-s2', 'run-1-P-s4'])
        for case in cases(cfg):
            argv = training_args(cfg, case, '/tmp/n3')
            self.assertEqual(argv[argv.index('--outer-cohort-size') + 1], str(case['cohort']))
            self.assertEqual(argv[argv.index('--log-interval') + 1], '1')
        formal = configuration({'SLURM_NNODES': '8', 'PIER_N2_SUITE': 'cohorts', 'PIER_N2_PROFILE': 'main'})
        self.assertEqual({c['cohort'] for c in cases(formal)}, {1, 2, 16})
        self.assertEqual(len(cases(formal)), 9)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.make_reports(root, cfg)
            result = collect(root)
            self.assertEqual(result['stage'], 'N3')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual([r['speedup_vs_P_s2'] for r in result['cases']], [2., 1., .5])
            self.assertEqual([r['independent_launches'] for r in result['by_arm']], [1, 1, 1])
            self.assertEqual([r['cohort'] for r in result['by_arm']], [1, 2, 4])
            self.assertIn('speedup/P(s=2)', render(result))
            self.assertTrue(all(r['speedup_vs_G'] is None for r in result['cases']))
            worker_path = root / 'run-1-P-s2/worker-rank-7.json'
            worker = json.loads(worker_path.read_text())
            worker['gpu'] = 'WRONG GPU'
            worker_path.write_text(json.dumps(worker))
            result = collect(root)
            self.assertEqual(result['cases'][1]['status'], 'invalid')
            self.assertTrue(all(c.get('speedup_vs_P_s2') is None for c in result['cases']))

    def test_frozen_launch_plan_survives_recipe_edits_but_rejects_worker_drift(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = self.make_reports(root)
            manifest['training_argv'] = {c['id']: training_args(manifest['config'], c, root / c['id'])
                                         for c in manifest['cases']}
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest))
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            for worker_path in root.glob('run-*/worker-rank-*.json'):
                w = json.loads(worker_path.read_text())
                w['manifest_sha256'] = fingerprint
                worker_path.write_text(json.dumps(w))
            with patch('experiments.qwen.n2_summary.training_args', side_effect=AssertionError('new recipe')):
                result = collect(root)
            self.assertEqual([r['status'] for r in result['cases'][:2]], ['measured', 'measured'])
            worker_path = root / 'run-1-P/worker-rank-1.json'
            w = json.loads(worker_path.read_text())
            w['argv'][w['argv'].index('--outer-cohort-size') + 1] = '1'
            worker_path.write_text(json.dumps(w))
            self.assertEqual(collect(root)['cases'][1]['status'], 'invalid')

    def test_archived_four_and_eight_gpu_results_are_read_only_and_still_recollect(self):
        archives = ['n2-58675806-20260920-222500.BFBhNF', 'n2-58678984-20260921-000738.rUk4TS',
                    'n2-58728454-20260922-075100.gEXpVH']
        for name, states in zip(archives, (['measured', 'measured', 'failed', 'measured'], ['measured'] * 4,
                                           ['measured'] * 12)):
            root = ROOT / 'out' / name
            if not root.is_dir():
                self.skipTest('archived pilot evidence is not present in this checkout')
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            result = collect(root)
            self.assertEqual([c['status'] for c in result['cases']], states)
            self.assertEqual(result, json.loads((root / 'summary.json').read_text()))
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                target = Path(directory)
                copy_historical_reference(root, target)
                self.assertEqual(json.loads((target / 'historical-n2.json').read_text())['result'], result)
                copy_historical_reference(target / 'missing', target)
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in root.rglob('*') if p.is_file()})

    def test_summary_partial_failure_rank_completeness_and_paired_arithmetic(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.make_reports(root)
            result = collect(root)
            g, p, r, w = result['cases']
            self.assertEqual([c['status'] for c in result['cases']], ['measured', 'measured', 'pending', 'pending'])
            self.assertEqual(g['useful_tokens_per_second'], 81920.)
            self.assertEqual(p['speedup_vs_G'], 2.)
            self.assertEqual(p['peak_allocated_gib'], 4.)
            self.assertEqual(p['peak_reserved_gib'], 5.)
            self.assertFalse(result['paper_ready'])
            self.assertTrue(result['performance_result'])
            relocated = root / 'copied-from-cluster'
            relocated.mkdir()
            shutil.copyfile(root / 'manifest.json', relocated / 'manifest.json')
            for arm in ('G', 'P'):
                shutil.copytree(root / f'run-1-{arm}', relocated / f'run-1-{arm}')
            self.assertEqual(collect(relocated)['cases'][1]['speedup_vs_G'], 2.)
            directory = root / 'run-1-R'
            directory.mkdir()
            (directory / 'exit.json').write_text('{"exit_code":137}')
            (directory / 'failure-rank-0.json').write_text('{"error":"OutOfMemoryError: test fixture"}')
            (root / 'run-1-G/cycles-rank-3.json').unlink()
            result = collect(root)
            self.assertEqual([c['status'] for c in result['cases']], ['invalid', 'measured', 'failed', 'pending'])
            self.assertIsNone(result['cases'][1]['speedup_vs_G'])
            self.assertEqual(result['cases'][2]['failure_kind'], 'CUDA OOM')


if __name__ == '__main__':
    unittest.main()
