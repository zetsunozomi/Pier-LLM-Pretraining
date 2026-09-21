"""E0c checker/I/O/launcher regression tests; every model here is tiny and CPU.

The CPU adapters below exist only in this test. Temporary reports explicitly
say GPU_executed=False and cannot pass the production evidence validator.
"""

import contextlib
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Qwen2ForCausalLM

from experiments.qwen import e0c
from experiments.qwen.e0c_evidence import CONTRACT, source_hashes, summarize, validate_native
from experiments.qwen.e0c_metrics import accepts, compare_tensor, exact_tensor
from experiments.qwen.preflight import pinned_architecture
from experiments.qwen.prepare_snapshot import prepare
from megatron.core.models.qwen.config import QwenArchitecture
from megatron.core.models.qwen.model import build_model
from megatron.core.models.qwen.weights import parameter_mappings, sha256_file
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from test_weights import tiny_config

ROOT = Path(__file__).resolve().parents[2]


def cpu_tensor_factory(original):
    def call(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return original(*args, **kwargs)
    return call


def cpu_native(rank, directory, dtype, tp, rendezvous):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='4')
    directory = Path(directory)
    initialize = dist.init_process_group
    def group(*args, **kwargs):
        return initialize('gloo', init_method=rendezvous, rank=rank, world_size=4,
                          timeout=kwargs['timeout'])
    def builder(arch, **kwargs):
        return build_model(arch, use_cpu_initialization=True, **kwargs)
    with patch.object(e0c, 'configure_gpu', return_value={'GPU_executed': False}), \
         patch.object(dist, 'init_process_group', side_effect=group), \
         patch('megatron.core.models.qwen.model.build_model', side_effect=builder), \
         patch('megatron.core.tensor_parallel.random.model_parallel_cuda_manual_seed'), \
         patch('torch.cuda.current_device', return_value='cpu'), \
         patch.object(get_cuda_rng_tracker(), 'fork', side_effect=lambda *a, **kw: contextlib.nullcontext()), \
         patch('torch.tensor', side_effect=cpu_tensor_factory(torch.tensor)), \
         patch('torch.ones', side_effect=cpu_tensor_factory(torch.ones)):
        args = type('Args', (), dict(output_dir=directory, dtype=dtype, tp=tp))()
        e0c.native(args, json.loads((directory / 'manifest.json').read_text()))


class E0cTests(unittest.TestCase):
    def test_whole_tensor_metrics_and_negative_controls(self):
        torch.manual_seed(101)
        expected = torch.randn(31, 47)
        exact = compare_tensor(expected, expected.clone(), 'fp32', 'gradient', chunk_elements=73)
        self.assertTrue(exact['passed'])
        self.assertEqual(exact['elements'], 1457)
        self.assertTrue(exact_tensor(expected, expected.clone(), chunk_elements=71)['bitwise_equal'])
        perturbed = expected.clone()
        perturbed.view(-1)[-1] += .05
        self.assertFalse(compare_tensor(perturbed, expected, 'fp32', 'gradient', 73)['passed'])
        self.assertFalse(exact_tensor(perturbed, expected, 71)['bitwise_equal'])
        self.assertFalse(compare_tensor(-expected, expected, 'bf16', 'gradient')['passed'])
        self.assertFalse(compare_tensor(expected * 1.16, expected, 'bf16', 'gradient')['passed'])
        self.assertTrue(compare_tensor(expected * 1.01, expected, 'bf16', 'gradient')['passed'])
        self.assertFalse(compare_tensor(torch.ones(4), torch.zeros(4), 'bf16', 'gradient')['passed'])
        self.assertTrue(compare_tensor(torch.zeros(4), torch.zeros(4), 'bf16', 'gradient')['passed'])
        invalid = expected.clone()
        invalid[0, 0] = float('nan')
        record = compare_tensor(invalid, expected, 'fp32', 'gradient')
        self.assertFalse(record['passed'])
        json.dumps(record, allow_nan=False)
        with self.assertRaises(ValueError):
            compare_tensor(expected[:1], expected, 'fp32', 'logits')

    def test_real_hf_reference_and_native_io_cpu_tp1_tp2(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='pier-e0c-cpu-') as name:
            directory = Path(name)
            snapshot = directory / 'snapshot'
            torch.manual_seed(901)
            hf = Qwen2ForCausalLM(tiny_config(heads=16, kv_heads=2))
            hf.save_pretrained(snapshot, safe_serialization=True)
            arch = QwenArchitecture.from_config(hf.config.to_dict())
            del hf
            manifest = {'snapshot_path': str(snapshot), 'snapshot': {'architecture': asdict(arch)}}
            e0c.write_json(directory / 'manifest.json', manifest)
            e0c.write_json(directory / 'inputs.json', {'tokens': [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]],
                                                      'positions': [list(range(6))] * 2})
            for dtype in ('fp32', 'bf16'):
                args = type('Args', (), dict(output_dir=directory, dtype=dtype))()
                with patch.object(e0c, 'configure_gpu', return_value={'GPU_executed': False}), \
                     patch.object(torch.nn.Module, 'cuda', lambda self: self), \
                     patch('torch.tensor', side_effect=cpu_tensor_factory(torch.tensor)):
                    e0c.reference(args, manifest)
                reference = json.loads((directory / f'hf-{dtype}.json').read_text())
                self.assertFalse(reference['environment']['GPU_executed'])
                self.assertEqual(reference['gradient_elements'], arch.unique_parameters())
                e0c.check_files(directory / f'reference-{dtype}', reference['files'])
                if dtype == 'fp32':
                    self.assertEqual(reference['linear_probe']['layer'], arch.layers - 1)
                    self.assertIn('linear-trace.safetensors', reference['files'])
                else:
                    self.assertNotIn('linear_probe', reference)
                for tp in (1, 2):
                    mp.spawn(cpu_native, args=(name, dtype, tp, f'file://{name}/rendezvous-{dtype}-{tp}'),
                             nprocs=4, join=True)
                    for rank in range(4):
                        record = json.loads((directory / f'native-{dtype}-tp{tp}-rank{rank}.json').read_text())
                        self.assertEqual(record['status'], 'passed')
                        self.assertFalse(record['environment']['GPU_executed'])
                        self.assertTrue(all(x['passed'] for x in record['gradients'].values()))
                        if dtype == 'fp32' and tp == 1:
                            diagnostic = json.loads((directory / f'linear-probe-fp32-tp1-rank{rank}.json').read_text())
                            self.assertFalse(diagnostic['environment']['GPU_executed'])
                            self.assertFalse(diagnostic['acceptance_override'])
                            self.assertEqual(diagnostic['native_report_sha256'],
                                             sha256_file(directory / f'native-{dtype}-tp{tp}-rank{rank}.json'))
                            self.assertEqual(diagnostic['reference_report_sha256'], sha256_file(directory / f'hf-{dtype}.json'))
                            self.assertEqual(diagnostic['manifest_sha256'], record['manifest_sha256'])
                            self.assertEqual(diagnostic['inputs_sha256'], record['inputs_sha256'])
                            self.assertEqual(diagnostic['rank'], rank)
                            for path, metadata in diagnostic['trace_files'].items():
                                self.assertEqual(metadata, e0c.file_record(directory / path))
                            self.assertEqual(len(diagnostic['points']), 1)
                            point = diagnostic['points'][0]
                            self.assertLess(abs(point['decomposition_residual']), 1e-12)
                            self.assertLess(abs(point['operand_split_residual']), 1e-12)
            changed = directory / 'reference-fp32/outputs.safetensors'
            changed.write_bytes(b'corrupt fixture')
            reference = json.loads((directory / 'hf-fp32.json').read_text())
            with self.assertRaises(ValueError):
                e0c.check_files(directory / 'reference-fp32', reference['files'])

    def test_evidence_rejects_incomplete_and_cpu_records(self):
        with tempfile.TemporaryDirectory() as name:
            report = summarize(Path(name))
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['GPU_conversion_validated'])
        # In-memory validator fixture: real 3B parameter counts, no model allocation.
        _, arch = pinned_architecture('3B')
        mapping = parameter_mappings(arch, 2, 0)
        stats = compare_tensor(torch.ones(1), torch.ones(1), 'fp32', 'gradient')
        counts = {p.target: math.prod(p.shape) for p in mapping}
        record = dict(phase='native', status='passed', dtype='fp32', tp=2, rank=0, tp_rank=0,
                      world_size=4, performance_result=False, optimizer_steps=0,
                      manifest_sha256='manifest', inputs_sha256='inputs', reference_report_sha256='ref',
                      environment={'GPU_executed': False}, gradient_elements=sum(counts.values()),
                      weights={name: dict(elements=count, bitwise_equal=True) for name, count in counts.items()},
                      gradients={name: dict(stats, elements=count) for name, count in counts.items()},
                      logits=dict(stats, elements=2 * 64 * arch.vocab), loss=dict(stats))
        check = lambda: validate_native(record, {}, 'fp32', 2, 0, 'manifest', 'inputs', 'ref')
        self.assertIn('missing actual GPU execution', check())
        record['environment']['GPU_executed'] = True  # synthetic validator input only
        self.assertEqual(check(), [])
        key = next(iter(record['gradients']))
        record['gradients'][key]['outside_tolerance'] = 1
        self.assertTrue(any('gradients mismatch' in e for e in check()))
        record['gradients'].pop(key)
        self.assertIn('gradients parameter coverage differs', check())
        record['logits']['elements'] -= 1
        self.assertIn('logits mismatch/incomplete coverage', check())

    def test_summary_requires_every_rank_and_preserves_historical_identity(self):
        # Synthetic JSON fixture exercises the verifier only, never saved as real evidence.
        pin, arch = pinned_architecture('3B')
        stats = compare_tensor(torch.ones(1), torch.ones(1), 'fp32', 'gradient')
        with tempfile.TemporaryDirectory(prefix='e0c-synthetic-validator-') as name:
            directory = Path(name)
            e0c.write_json(directory / 'inputs.json', {'synthetic': True})
            input_sha = sha256_file(directory / 'inputs.json')
            snapshot = {'revision': pin['revision'], 'architecture': asdict(arch),
                        'config_sha256': pin['config_sha256'], 'checkpoint_verified': True,
                        'tokenizer_files_verified': True, 'unique_parameters': arch.unique_parameters(),
                        'files': {n: {'bytes': x['bytes'], 'sha256': x.get('lfs_sha256') or 'fixture'}
                                  for n, x in pin['files'].items()}}
            manifest = {'stage': 'E0c', 'contract': CONTRACT, 'inputs_sha256': input_sha,
                        'collect_numerical_mismatches': True,
                        'snapshot': snapshot, 'source_sha256': {'synthetic-source': 'fixture'}}
            e0c.write_json(directory / 'manifest.json', manifest)
            manifest_sha = sha256_file(directory / 'manifest.json')
            for dtype in ('fp32', 'bf16'):
                ref_path = directory / f'hf-{dtype}.json'
                shapes = arch.hf_shapes()
                reference = {'status': 'reference_written', 'dtype': dtype,
                             'manifest_sha256': manifest_sha, 'inputs_sha256': input_sha,
                             'environment': {'GPU_executed': True}, 'optimizer_steps': 0,
                             'performance_result': False, 'loading_info': {}, 'loss': 1.,
                             'gradient_elements': arch.unique_parameters(),
                             'gradients': {n: dict(shape=list(s), elements=math.prod(s)) for n, s in shapes.items()},
                             'weights': {n: dict(bitwise_equal=True, elements=math.prod(s)) for n, s in shapes.items()}}
                e0c.write_json(ref_path, reference)
                for tp in (1, 2):
                    for rank in range(4):
                        counts = {p.target: math.prod(p.shape) for p in parameter_mappings(arch, tp, rank % tp)}
                        record = dict(phase='native', status='passed', dtype=dtype, tp=tp, rank=rank,
                                      collect_numerical_mismatches=True,
                                      tp_rank=rank % tp, world_size=4, optimizer_steps=0, performance_result=False,
                                      manifest_sha256=manifest_sha, inputs_sha256=input_sha,
                                      reference_report_sha256=sha256_file(ref_path), environment={'GPU_executed': True},
                                      gradient_elements=sum(counts.values()),
                                      weights={n: dict(elements=c, bitwise_equal=True) for n, c in counts.items()},
                                      gradients={n: dict(stats, elements=c) for n, c in counts.items()},
                                      logits=dict(stats, elements=2 * 64 * arch.vocab), loss=copy.deepcopy(stats))
                        e0c.write_json(directory / f'native-{dtype}-tp{tp}-rank{rank}.json', record)
            self.assertEqual(summarize(directory, check_current_source=False)['status'], 'passed')
            mismatch_path = directory / 'native-fp32-tp1-rank0.json'
            unchanged = mismatch_path.read_bytes()
            mismatch = json.loads(unchanged)
            mismatch['status'] = 'failed'
            next(iter(mismatch['gradients'].values()))['outside_tolerance'] = 1
            mismatch_path.write_text(json.dumps(mismatch))
            collected = summarize(directory, check_current_source=False)
            self.assertEqual(collected['status'], 'failed')
            self.assertFalse(collected['GPU_conversion_validated'])
            self.assertEqual(collected['native_reports_collected'], 16)
            self.assertTrue(collected['collect_numerical_mismatches'])
            mismatch_path.write_bytes(unchanged)
            self.assertEqual(summarize(directory)['status'], 'failed')
            self.assertEqual(summarize(directory, launcher_exit=23, check_current_source=False)['status'], 'failed')
            original = (directory / 'native-bf16-tp2-rank3.json').read_bytes()
            (directory / 'native-bf16-tp2-rank3.json').unlink()
            failed = summarize(directory, check_current_source=False)
            self.assertEqual(failed['status'], 'failed')
            self.assertTrue(any('rank3' in e for e in failed['errors']))
            (directory / 'native-bf16-tp2-rank3.json').write_bytes(original)
            e0c.write_json(directory / 'failure-native.json', {'error': 'synthetic failure'})
            self.assertEqual(summarize(directory, check_current_source=False)['status'], 'failed')

    def test_snapshot_requires_download_opt_in_and_preserves_bad_files(self):
        with tempfile.TemporaryDirectory() as name:
            with patch('huggingface_hub.snapshot_download') as download:
                with self.assertRaises(FileNotFoundError):
                    prepare('3B', name)
                download.assert_not_called()
                Path(name, 'config.json').write_text('{}')
                with self.assertRaises(ValueError):
                    prepare('3B', name, download=True)
                download.assert_not_called()

    def test_sources_and_git_evidence_scope(self):
        files = source_hashes()
        for path in ('pretrain_qwen.py', 'experiments/qwen/e0c.py', 'experiments/qwen/e0c.sbatch',
                     'experiments/qwen/e0c_linear_probe.py',
                     'experiments/qwen/pins.json', 'megatron/core/models/qwen/model.py', 'requirements.txt'):
            self.assertEqual(files[path], sha256_file(ROOT / path))
        for path, ignored in [('local/qwen/e0c-fixture/summary.json', False),
                              ('local/qwen/e0c-fixture/hf-fp32.log', False),
                              ('local/qwen/e0c-fixture/linear-probe-fp32-tp1-rank0.json', False),
                              ('local/qwen/e0c-fixture/linear-traces/fp32-tp1-rank0.safetensors', True),
                              ('local/qwen/e0c-fixture/reference-fp32/linear-trace.safetensors', True),
                              ('local/qwen/e0c-fixture/reference-fp32/gradient-0000.safetensors', True),
                              ('local/qwen/models/Qwen2.5-3B/weights.safetensors', True)]:
            result = subprocess.run(['git', 'check-ignore', '--no-index', '-q', path], cwd=ROOT)
            self.assertEqual(result.returncode == 0, ignored, path)

    def test_launcher_stops_on_first_failure_in_existing_allocation(self):
        with tempfile.TemporaryDirectory(prefix='pier-e0c-launcher-') as name:
            directory = Path(name)
            (directory / 'snapshot').mkdir()
            srun = directory / 'srun'
            srun.write_text('#!/usr/bin/env bash\nwhile [[ "$1" == --* ]]; do shift; done\nexec "$@"\n')
            interpreter = directory / 'fake-python'
            interpreter.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CALLS"
if [[ "$*" == *"--phase reference"* ]]; then exit 23; fi
exit 0
''')
            srun.chmod(0o755)
            interpreter.chmod(0o755)
            env = dict(os.environ, PIER_ROOT=str(ROOT), PIER_PYTHON=str(interpreter),
                       PIER_QWEN_SNAPSHOT=str(directory / 'snapshot'),
                       PIER_E0C_RUN_DIR=str(directory / 'run'), SLURM_JOB_NUM_NODES='1',
                       SLURM_JOB_ID='42', CALLS=str(directory / 'calls'),
                       PATH=str(directory) + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', 'experiments/qwen/e0c.sbatch'], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
            calls = (directory / 'calls').read_text()
            self.assertIn('--summarize', calls)
            self.assertIn('--launcher-exit 23', calls)
            self.assertEqual(calls.count('--phase reference'), 1)
            self.assertNotIn('--phase native', calls)
            self.assertNotIn('DONE hf-fp32', result.stdout)
            self.assertIn('FAILED hf-fp32', result.stderr)
            self.assertEqual(CONTRACT['version'], 1)


if __name__ == '__main__':
    unittest.main()
