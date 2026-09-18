"""E0d parser, dispatch and evidence tests; no full-size model or CUDA run.

Synthetic JSON fixtures below only exercise acceptance/rejection and are kept
in temporary directories. They are not experimental evidence.
"""

import contextlib
import copy
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.qwen import e0d_evidence as evidence, e0d_train as train
from experiments.qwen.e0c import write_json
from experiments.qwen.e0c_evidence import CONTRACT, summarize as summarize_e0c
from experiments.qwen.e0c_metrics import compare_tensor
from experiments.qwen.e0d_config import ATTEMPTS, CASES, RECOMPUTE, training_args
from experiments.qwen.preflight import pinned_architecture
from megatron.core.models.qwen.training import training_defaults, validate_training_contract
from megatron.core.models.qwen.weights import parameter_mappings, sha256_file
from megatron.core.outer_sync.checkpoint import recipe
from megatron.core.outer_sync.runtime import validate_args as validate_outer
from megatron.training.arguments import parse_args, validate_args
from pretrain_qwen import snapshot_identity

ROOT = Path(__file__).resolve().parents[2]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def e0c_fixture(directory):
    """Full-size metadata only, with synthetic comparison statistics."""
    directory.mkdir()
    pin, arch = pinned_architecture('3B')
    counts = [sum(math.prod(p.shape) for p in parameter_mappings(arch, 2, r)) for r in range(2)]
    snapshot = {'model': pin['model'], 'revision': pin['revision'], 'architecture': asdict(arch),
                'tp_size': 2, 'local_parameter_elements': counts,
                'config_sha256': pin['config_sha256'], 'checkpoint_verified': True,
                'tokenizer_files_verified': True, 'unique_parameters': arch.unique_parameters(),
                'files': {n: {'bytes': f['bytes'], 'sha256': f.get('lfs_sha256') or digest(n)}
                          for n, f in pin['files'].items()}}
    write_json(directory / 'inputs.json', {'synthetic': True})
    input_sha = sha256_file(directory / 'inputs.json')
    manifest = {'stage': 'E0c', 'contract': CONTRACT, 'inputs_sha256': input_sha,
                'snapshot': snapshot, 'source_sha256': {'synthetic-source': digest('source')}}
    write_json(directory / 'manifest.json', manifest)
    manifest_sha = sha256_file(directory / 'manifest.json')
    stats = compare_tensor(torch.ones(1), torch.ones(1), 'fp32', 'gradient')
    for dtype in ('fp32', 'bf16'):
        reference = {'status': 'reference_written', 'dtype': dtype, 'manifest_sha256': manifest_sha,
                     'inputs_sha256': input_sha, 'environment': {'GPU_executed': True},
                     'optimizer_steps': 0, 'performance_result': False, 'loading_info': {}, 'loss': 1.,
                     'gradient_elements': arch.unique_parameters(),
                     'gradients': {n: {'shape': list(s), 'elements': math.prod(s)} for n, s in arch.hf_shapes().items()},
                     'weights': {n: {'bitwise_equal': True, 'elements': math.prod(s)} for n, s in arch.hf_shapes().items()}}
        ref = directory / f'hf-{dtype}.json'
        write_json(ref, reference)
        for tp in (1, 2):
            for rank in range(4):
                shapes = {p.target: math.prod(p.shape) for p in parameter_mappings(arch, tp, rank % tp)}
                record = dict(phase='native', status='passed', dtype=dtype, tp=tp, rank=rank,
                              tp_rank=rank % tp, world_size=4, optimizer_steps=0, performance_result=False,
                              manifest_sha256=manifest_sha, inputs_sha256=input_sha,
                              reference_report_sha256=sha256_file(ref), environment={'GPU_executed': True},
                              gradient_elements=sum(shapes.values()),
                              weights={n: dict(elements=c, bitwise_equal=True) for n, c in shapes.items()},
                              gradients={n: dict(stats, elements=c) for n, c in shapes.items()},
                              logits=dict(stats, elements=2 * 64 * arch.vocab), loss=stats)
                write_json(directory / f'native-{dtype}-tp{tp}-rank{rank}.json', record)
    summary = summarize_e0c(directory, check_current_source=False)
    assert summary['status'] == 'passed', summary
    write_json(directory / 'summary.json', summary)
    return snapshot, manifest['source_sha256']


def case_fixture(case, rank, manifest, manifest_sha):
    config, end = CASES[case], CASES[case].get('stop', ATTEMPTS)
    _, arch = pinned_architecture('3B')
    n = manifest['snapshot']['local_parameter_elements'][rank % 2]
    events = []
    for attempted in range(1, end + 1):
        success = attempted - int(attempted >= 3)
        boundary = attempted != 3 and success % 3 == 0
        events.append({'attempted': attempted, 'successful': success, 'boundaries': success // 3,
                       'skipped': attempted == 3, 'outer_boundary': boundary,
                       'oracle_states_bitwise': True if boundary else None,
                       'outer_retained_inner_state': True if boundary else None,
                       'skip_retained_optimizer': True if attempted == 3 else None,
                       'master_sha256': digest(f'master-{rank % 2}-{success}'),
                       'model_sha256': digest(f'model-{rank % 2}-{success}'),
                       'inner_sha256': digest(f'inner-{rank}-{success}'),
                       'loss': {'lm loss': 3. / attempted}})
    record = {'status': 'partial' if case == 'split' else 'passed', 'rank': rank, 'world_size': 4,
              'GPU_executed': True, 'backend': 'nccl', 'arm': 'pier', 'cohort': config['cohort'],
              'state_tier': 'host' if config['host'] else 'device', 'oracle_storage': 'streamed',
              'coordinate_numel': n, 'outer_ranks': list(range(rank % 2, 4, 2)), 'inner_ranks': [rank],
              'workspace_budget_mib': 64., 'restored': bool(config.get('resume')), 'pending_consumer': False,
              'consumed_train_samples': end * 16, 'consumed_valid_samples': 0, 'skipped_train_samples': 0,
              'clock': {'interval': 3, 'attempted': end, 'successful': end - 1, 'boundaries': (end - 1) // 3},
              'performance_result': False, 'error': None, 'consumer_checks': (end - 1) // 3,
              'coordinate_fingerprint': digest(f'coordinates-{rank % 2}'), 'events': events,
              'oracle_allocation': {'kind': 'streamed-full-coordinate-fixed-tree', 'coordinates_per_vector': n,
                                    'last_master_coordinates_checked': n, 'boundaries_checked': (end - 1) // 3,
                                    'file_logical_bytes': 8 * n, 'tile_elements': 65536,
                                    'persistent_device_scratch_bytes': 3 * 65536 * 4, 'performance_result': False}}
    if config.get('resume'):
        record['restore_evidence'] = {'iteration': 5, 'consumed_train_samples': 80,
                                      'clock': {'interval': 3, 'attempted': 5, 'successful': 4, 'boundaries': 1},
                                      'rank_file_sha256': digest(f'checkpoint-{rank}'), 'rank_file_bytes': 99}
    launch = {'case': case, 'rank': rank, 'entrypoint': 'pretrain_qwen.py',
              'argv': training_args(case, manifest['run_output'], manifest['snapshot_path'], manifest['data_prefix']),
              'manifest_sha256': manifest_sha, 'GPU_executed': True, 'environment': {'GPU_executed': True}}
    initialization = {'tp_size': 2, 'tp_rank': rank % 2, 'local_parameter_elements': n,
                      'loaded_parameters': len(parameter_mappings(arch, 2, rank % 2)),
                      'unique_model_elements': arch.unique_parameters(), 'optimizer_state_restored': False,
                      'loaded_before_optimizer_construction': True, 'exact_output_vocabulary': arch.vocab,
                      'architecture': asdict(arch), 'activation_recompute': RECOMPUTE,
                      'qwen_recipe': manifest['qwen_recipe'], 'initialization': 'weights-only warm start'}
    return record, launch, initialization


def archive_fixture(directory):
    snapshot, sources = e0c_fixture(directory / 'prerequisite-e0c')
    pin, _ = pinned_architecture('3B')
    eos = json.loads((ROOT / 'experiments/qwen' / pin['config_filename']).read_text())['eos_token_id']
    data = {'manifest_sha256': digest('data-manifest'), 'tokens': 400000, 'documents': 10,
            'input': {'bytes': 99, 'sha256': digest('source-text')}, 'tokenization': {'append_eod': eos},
            'files': {'.bin': {'bytes': 1600000, 'sha256': digest('bin')},
                      '.idx': {'bytes': 99, 'sha256': digest('idx')}}}
    manifest = {'stage': 'E0d', 'world_size': 4, 'tp_size': 2, 'inner_dp': 1, 'outer_learners': 2,
                'cases': CASES, 'attempts': ATTEMPTS, 'sequence_length': 2048, 'global_batch': 16,
                'micro_batch': 1, 'microbatches_per_learner': 8, 'recompute': RECOMPUTE,
                'performance_result': False, 'environment': {'GPU_executed': True}, 'data': data,
                'run_output': str(directory), 'snapshot_path': '/synthetic/model', 'data_prefix': '/synthetic/data',
                'source_sha256': sources, 'snapshot': snapshot,
                'scratch_estimate': evidence.scratch_estimate(snapshot['local_parameter_elements']),
                'e0c_files': evidence.prerequisite(directory / 'prerequisite-e0c', check_current_source=False),
                'qwen_recipe': {'snapshot_sha256': snapshot_identity(snapshot), 'architecture': snapshot['architecture'],
                                'data': data, 'eod': eos, 'initialization': 'weights-only warm start'}}
    write_json(directory / 'manifest.json', manifest)
    write_json(directory / 'node-0.json', {'cuda_available': True, 'visible_devices': 4})
    write_json(directory / 'checkpoint-manifest.json', {
        'format': 'pier-centered-v1', 'iteration': 5, 'world': 4,
        'files': [{'rank': r, 'name': f'rank-{r}.pt', 'bytes': 99, 'sha256': digest(f'checkpoint-{r}')}
                  for r in range(4)]})
    manifest_sha = sha256_file(directory / 'manifest.json')
    for case in CASES:
        folder = directory / f'case-{case}'
        folder.mkdir()
        for rank in range(4):
            record, launch, init = case_fixture(case, rank, manifest, manifest_sha)
            for prefix, value in (('', record), ('launch-', launch), ('initialization-', init)):
                write_json(folder / f'{prefix}rank-{rank}.json', value)
    return manifest


class E0dTests(unittest.TestCase):
    def test_actual_parser_and_checkpoint_recipe_for_every_phase(self):
        _, arch = pinned_architecture('3B')
        def extra(parser):
            parser.add_argument('--qwen-model-size')
            parser.add_argument('--qwen-snapshot')
            parser.add_argument('--qwen-trace-dir')
            parser.set_defaults(**training_defaults(arch), tokenizer_type='HuggingFaceTokenizer')
            return parser
        parsed = {}
        with patch.dict(os.environ, WORLD_SIZE='4', RANK='0', CUDA_DEVICE_MAX_CONNECTIONS='1', NCCL_ALGO='Ring'):
            for case in CASES:
                argv = ['pretrain_qwen.py', *training_args(case, '/run', '/snapshot', '/data/train')]
                with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    args = validate_args(parse_args(extra))
                validate_training_contract(args, arch)
                validate_outer(args)
                self.assertEqual(args.global_batch_size // args.data_parallel_size // args.micro_batch_size, 8)
                self.assertEqual(args.padded_vocab_size, arch.vocab)
                self.assertEqual(args.recompute_num_layers, 1)
                self.assertEqual(args.outer_verify_storage, 'streamed')
                self.assertEqual(args.save, '/run/checkpoints' if case == 'split' else None)
                self.assertEqual(args.load, '/run/checkpoints' if case == 'resume' else None)
                parsed[case] = args
        self.assertEqual(recipe(parsed['split']), recipe(parsed['resume']))
        self.assertEqual(recipe(parsed['split']), recipe(parsed['s2-device']))
        self.assertNotEqual(recipe(parsed['s1-host']), recipe(parsed['s2-device']))

    def test_complete_archive_and_negative_controls(self):
        with tempfile.TemporaryDirectory(prefix='e0d-synthetic-validator-') as name:
            directory = Path(name)
            manifest = archive_fixture(directory)
            check = lambda: evidence.summarize(directory, check_current_source=False)
            report = check()
            self.assertEqual(report['errors'], [])
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(len(report['comparisons']), 16)
            self.assertEqual(sum(c['prefix_only'] for c in report['comparisons']), 4)
            self.assertFalse(report['performance_result'])
            self.assertEqual(evidence.summarize(directory)['status'], 'failed')
            self.assertEqual(evidence.summarize(directory, 23, check_current_source=False)['status'], 'failed')
            target = directory / 'case-resume/rank-3.json'
            original = target.read_bytes()
            record = json.loads(original)
            variants = [dict(record, consumed_train_samples=0), dict(record, GPU_executed=False),
                        dict(record, pending_consumer=True), dict(record, restore_evidence={}),
                        dict(record, coordinate_fingerprint=digest('wrong-layout'))]
            changed = copy.deepcopy(record)
            changed['events'][2]['master_sha256'] = digest('updated-during-skip')
            variants.append(changed)
            changed = copy.deepcopy(record)
            changed['events'][-1]['loss']['lm loss'] = 22.
            variants.append(changed)
            changed = copy.deepcopy(record)
            changed['restore_evidence']['rank_file_sha256'] = digest('wrong-checkpoint')
            variants.append(changed)
            for changed in variants:
                target.write_text(json.dumps(changed))
                self.assertEqual(check()['status'], 'failed')
            target.unlink()
            self.assertEqual(check()['status'], 'failed')
            target.write_bytes(original)
            self.assertEqual(check()['status'], 'passed')
            manifest_path = directory / 'manifest.json'
            original_manifest = manifest_path.read_bytes()
            changed_manifest = json.loads(original_manifest)
            changed_manifest['data']['tokens'] = 1
            manifest_path.write_text(json.dumps(changed_manifest))
            self.assertTrue(any('indexed data identity' in e for e in check()['errors']))
            manifest_path.write_bytes(original_manifest)
            # The prerequisite is revalidated and its exact portable bytes bound.
            prerequisite = directory / 'prerequisite-e0c/native-bf16-tp2-rank3.json'
            prerequisite.write_bytes(prerequisite.read_bytes() + b'\n')
            self.assertTrue(any('E0c evidence bytes changed' in e for e in check()['errors']))
            self.assertEqual(manifest['snapshot']['unique_parameters'], 3085938688)

    def test_case_validator_checks_whole_state_not_just_pass_label(self):
        with tempfile.TemporaryDirectory(prefix='e0d-synthetic-validator-') as name:
            directory = Path(name)
            manifest = archive_fixture(directory)
            sha = sha256_file(directory / 'manifest.json')
            record, launch, init = case_fixture('s2-device', 0, manifest, sha)
            check = lambda: evidence.validate_case('s2-device', 0, record, launch, init, manifest, sha)
            self.assertEqual(check(), [])
            record['oracle_allocation']['last_master_coordinates_checked'] -= 1
            self.assertIn('incomplete full-coordinate oracle coverage', check())
            record['events'][3]['oracle_states_bitwise'] = False
            self.assertTrue(any('attempt 4' in e for e in check()))
            launch['argv'].append('--mock-data')
            self.assertIn('missing manifest-bound real training launch', check())
            init['loaded_parameters'] -= 1
            self.assertTrue(any('coverage differs' in e for e in check()))

    def test_wrapper_dispatches_real_entrypoint_and_records_failure(self):
        for outcome in (None, SystemExit(0), RuntimeError('synthetic entrypoint failure')):
            with tempfile.TemporaryDirectory(prefix='e0d-wrapper-') as name:
                directory = Path(name)
                write_json(directory / 'manifest.json', {'stage': 'E0d', 'source_sha256': {'fixture': 'hash'},
                                                         'snapshot_path': '/model', 'data_prefix': '/data'})
                (directory / 'checkpoints/iter_0000005').mkdir(parents=True)
                write_json(directory / 'checkpoints/iter_0000005/complete.json', {'synthetic': True})
                with patch.dict(os.environ, RANK='0', WORLD_SIZE='4'), \
                     patch.object(sys, 'argv', ['e0d_train', '--case', 'split', '--output-dir', name]), \
                     patch.object(train, 'source_hashes', return_value={'fixture': 'hash'}), \
                     patch.object(train, 'configure_gpu', return_value={'GPU_executed': False}), \
                     patch.object(train.runpy, 'run_path', side_effect=outcome) as run, \
                     patch('torch.distributed.is_initialized', return_value=True), \
                     patch('torch.distributed.destroy_process_group') as destroy:
                    if isinstance(outcome, RuntimeError):
                        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                            train.main()
                        self.assertTrue((directory / 'case-split/failure-rank-0.json').exists())
                    else:
                        train.main()
                        self.assertFalse((directory / 'case-split/failure-rank-0.json').exists())
                        self.assertEqual(json.loads((directory / 'checkpoint-manifest.json').read_text()),
                                         {'synthetic': True})
                    run.assert_called_once_with(str(ROOT / 'pretrain_qwen.py'), run_name='__main__')
                    destroy.assert_called_once()
                    launch = json.loads((directory / 'case-split/launch-rank-0.json').read_text())
                    self.assertFalse(launch['environment']['GPU_executed'])
                    self.assertEqual(launch['argv'], training_args('split', directory.resolve(), '/model', '/data'))

    def test_shell_stops_on_first_failure_and_preserves_exit(self):
        for failed in ('preflight', 's2-device', 'none'):
            with tempfile.TemporaryDirectory(prefix='e0d-shell-') as name:
                directory = Path(name)
                for folder in ('snapshot', 'e0c'):
                    (directory / folder).mkdir()
                for suffix in ('.bin', '.idx', '.manifest.json'):
                    (directory / ('data' + suffix)).touch()
                (directory / 'srun').write_text('#!/usr/bin/env bash\nwhile [[ "$1" == --* ]]; do shift; done\nexec "$@"\n')
                (directory / 'fake-python').write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CALLS"
if [[ "$FAIL_PHASE" == preflight && "$*" == *e0d_evidence.py* && "$*" != *--summarize* ]]; then exit 23; fi
if [[ "$FAIL_PHASE" == s2-device && "$*" == *"--case s2-device"* ]]; then exit 23; fi
exit 0
''')
                for path in ('srun', 'fake-python'):
                    (directory / path).chmod(0o755)
                env = dict(os.environ, PIER_ROOT=str(ROOT), PIER_PYTHON=str(directory / 'fake-python'),
                           PIER_QWEN_SNAPSHOT=str(directory / 'snapshot'), PIER_QWEN_DATA_PREFIX=str(directory / 'data'),
                           PIER_E0C_EVIDENCE=str(directory / 'e0c'), PIER_E0D_RUN_DIR=str(directory / 'run'),
                           SLURM_JOB_NUM_NODES='1', SLURM_JOB_ID='42', CALLS=str(directory / 'calls'), FAIL_PHASE=failed,
                           PATH=str(directory) + os.pathsep + os.environ['PATH'])
                result = subprocess.run(['bash', 'experiments/qwen/e0d.sbatch'], cwd=ROOT, env=env,
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0 if failed == 'none' else 23, result.stdout + result.stderr)
                calls = (directory / 'calls').read_text()
                self.assertIn('--summarize', calls)
                self.assertIn('--launcher-exit ' + ('0' if failed == 'none' else '23'), calls)
                self.assertEqual(calls.count('--case '), {'preflight': 0, 's2-device': 1, 'none': 5}[failed])
                if failed != 'none':
                    self.assertNotIn('--case resume', calls)
                    self.assertIn('FAILED ' + failed, result.stderr)

    def test_scratch_bound_and_git_return_scope(self):
        estimate = evidence.scratch_estimate([10, 12])
        self.assertEqual(estimate['oracle_files_bytes'], 5 * 8 * 44)
        self.assertEqual(estimate['checkpoint_conservative_bytes'], 24 * 44)
        self.assertEqual(estimate['required_free_bytes'], 64 * 44 + (32 << 30))
        for path, ignored in [('local/qwen/e0d-fixture/summary.json', False),
                              ('local/qwen/e0d-fixture/s2-device.log', False),
                              ('local/qwen/e0d-fixture/case-resume/launch-rank-0.json', False),
                              ('local/qwen/e0d-fixture/prerequisite-e0c/manifest.json', False),
                              ('local/qwen/e0d-fixture/case-s2-device/.oracle/reference.fp32', True),
                              ('local/qwen/e0d-fixture/checkpoints/iter_0000005/rank_0.pt', True),
                              ('local/qwen/e0d-fixture/dataset-cache/sample.npy', True)]:
            result = subprocess.run(['git', 'check-ignore', '--no-index', '-q', path], cwd=ROOT)
            self.assertEqual(result.returncode == 0, ignored, path)

    def test_preflight_rejects_prerequisite_data_and_space_before_manifest(self):
        # Heavy snapshot/GPU/tokenizer loading is mocked only for gate ordering;
        # no files from this test are accepted as CUDA evidence.
        with tempfile.TemporaryDirectory(prefix='e0d-preflight-') as name:
            directory = Path(name)
            snapshot, sources = e0c_fixture(directory / 'e0c')
            accepted = evidence.prerequisite(directory / 'e0c', check_current_source=False)
            dataset = type('DatasetFixture', (), {'__len__': lambda self: 10,
                                                  'sequence_lengths': torch.tensor([400000])})()
            tokenizer = SimpleNamespace(eos_token_id=151643)
            data = {'documents': 10, 'tokens': 400000}
            def make(output):
                return evidence.make_manifest(output, directory / 'snapshot', directory / 'data', directory / 'e0c')
            with patch.object(evidence, 'configure_gpu', return_value={'GPU_executed': False}), \
                 patch.object(evidence, 'prerequisite', return_value=accepted) as prerequisite, \
                 patch.object(evidence, 'preflight', return_value=snapshot) as model, \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                 patch.object(evidence, 'data_identity', return_value=data), \
                 patch('megatron.core.datasets.indexed_dataset.IndexedDataset', return_value=dataset), \
                 patch.object(evidence, 'source_hashes', return_value=sources), \
                 patch.object(evidence.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)) as disk:
                with self.assertRaisesRegex(RuntimeError, 'free scratch'):
                    make(directory / 'no-space')
                self.assertFalse((directory / 'no-space/manifest.json').exists())
                data['tokens'] = 2
                with self.assertRaisesRegex(ValueError, 'indexed data count'):
                    make(directory / 'short-data')
                data['tokens'] = 400000
                prerequisite.side_effect = ValueError('E0c prerequisite failed')
                model.reset_mock()
                with self.assertRaisesRegex(ValueError, 'E0c prerequisite failed'):
                    make(directory / 'bad-prerequisite')
                model.assert_not_called()
                prerequisite.side_effect = None
                disk.return_value = SimpleNamespace(free=10**12)
                output = directory / 'accepted'
                manifest = make(output)
                self.assertFalse(manifest['environment']['GPU_executed'])
                self.assertEqual(len(list((output / 'prerequisite-e0c').iterdir())), 21)
                self.assertEqual(manifest['data'], data)
                with self.assertRaises(FileExistsError):
                    make(output)


if __name__ == '__main__':
    unittest.main()
