"""CPU-only FP64 diagnostic verification; cannot produce GPU acceptance."""

import contextlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import Qwen2ForCausalLM

from experiments.qwen import e0c_fp64
from experiments.qwen.e0c import file_record, write_json
from experiments.qwen.e0c_evidence import CONTRACT, source_hashes
from experiments.qwen.fp64_math import double_softmax, loss64, promote
from megatron.core.models.qwen.config import QwenArchitecture
from megatron.core.models.qwen.model import build_model
from megatron.core.models.qwen.weights import sha256_file
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from test_e0c import cpu_tensor_factory
from test_weights import tiny_config

ROOT = Path(__file__).resolve().parents[2]


class FP64Tests(unittest.TestCase):
    def test_prepare_binds_original_inputs_and_rejects_changed_identity(self):
        # Synthetic metadata tests validation only; it is never real GPU evidence.
        arch = QwenArchitecture.from_config(tiny_config().to_dict())
        architecture = dict(asdict(arch), layers=4)
        checked = {'revision': 'synthetic', 'files': {}, 'architecture': architecture}
        with tempfile.TemporaryDirectory(prefix='pier-fp64-identity-') as name:
            root = Path(name)
            previous = root / 'previous'
            previous.mkdir()
            write_json(previous / 'inputs.json', {'synthetic': True})
            inputs_sha = sha256_file(previous / 'inputs.json')
            write_json(previous / 'manifest.json', dict(stage='E0c', contract=CONTRACT,
                inputs_sha256=inputs_sha, snapshot_path=str(root / 'snapshot'), snapshot=checked))
            identity = dict(manifest_sha256=sha256_file(previous / 'manifest.json'),
                            inputs_sha256=inputs_sha, environment={'GPU_executed': True})
            write_json(previous / 'hf-fp32.json', dict(identity, status='reference_written', dtype='fp32'))
            point = dict(index=[1, 2], actual=.1, reference=.2, tolerance=2e-5, ratio=5000.)
            native = dict(identity, dtype='fp32', tp=1, rank=0,
                reference_report_sha256=sha256_file(previous / 'hf-fp32.json'),
                gradients={'decoder.layers.2.mlp.linear_fc2.weight': dict(shape=[arch.hidden, arch.intermediate],
                            outside_tolerance_samples=[point], worst_element=point)})
            write_json(previous / 'native-fp32-tp1-rank0.json', native)
            with patch.object(e0c_fp64, 'configure_gpu', return_value={'GPU_executed': False}), \
                 patch.object(e0c_fp64, 'preflight', return_value=checked):
                e0c_fp64.prepare(root / 'output', previous)
                manifest = json.loads((root / 'output/manifest.json').read_text())
                self.assertEqual(manifest['fp32_points'], [point])
                self.assertFalse(manifest['acceptance_override'])
                self.assertEqual(manifest['fp32_native_report_sha256'], sha256_file(previous / 'native-fp32-tp1-rank0.json'))
                with self.assertRaisesRegex(ValueError, 'reuse'):
                    e0c_fp64.prepare(root / 'output', previous)
                (previous / 'inputs.json').write_text('{"changed": true}')
                with self.assertRaisesRegex(ValueError, 'identity'):
                    e0c_fp64.prepare(root / 'bad', previous)

    def test_freezing_other_parameters_preserves_target_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(819)
        config = tiny_config()
        config.num_hidden_layers = 4
        config.layer_types = ['full_attention'] * 4
        config._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(config)
        arch = QwenArchitecture.from_config(config.to_dict())
        name, parameter, receipt = promote(model, 'hf', arch, 2)
        probe = torch.randn(2, 3, arch.hidden, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(model.model.norm, (probe,), fast_mode=True,
                                                eps=1e-6, atol=1e-8, rtol=1e-5))
        tokens = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
        positions = torch.arange(4).expand_as(tokens)
        original = torch.nn.functional.softmax
        with double_softmax() as softmax:
            expected = model(tokens, position_ids=positions, use_cache=False).logits
            loss = loss64(expected, tokens)
            loss.backward()
        self.assertIs(torch.nn.functional.softmax, original)
        self.assertEqual(softmax['fp64_softmax_calls'], 4)
        self.assertEqual(receipt['norm_modules'], 9)
        self.assertEqual([n for n, p in model.named_parameters() if p.grad is not None], [name])
        gradient = parameter.grad.clone()
        model.zero_grad(set_to_none=True)
        model.requires_grad_(True)
        with double_softmax():
            actual = model(tokens, position_ids=positions, use_cache=False).logits
            loss64(actual, tokens).backward()
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        torch.testing.assert_close(parameter.grad, gradient, rtol=1e-12, atol=1e-12)

    def test_real_diagnostic_hf_native_io_and_cpu_rejection(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='pier-fp64-cpu-') as name:
            directory = Path(name)
            snapshot = directory / 'snapshot'
            config = tiny_config(heads=16, kv_heads=2)
            config.num_hidden_layers = 4
            config.layer_types = ['full_attention'] * 4
            torch.manual_seed(98)
            hf = Qwen2ForCausalLM(config)
            arch = QwenArchitecture.from_config(config.to_dict())
            hf.save_pretrained(snapshot, safe_serialization=True)
            del hf
            write_json(directory / 'inputs.json', {'tokens': [[1, 3, 5, 7], [2, 4, 6, 8]],
                                                 'positions': [list(range(4))] * 2})
            write_json(directory / 'manifest.json', dict(layer=2, snapshot_path=str(snapshot),
                snapshot={'architecture': asdict(arch)}, source_sha256=source_hashes(),
                inputs_sha256=sha256_file(directory / 'inputs.json'),
                fp32_points=[{'index': [1, 2], 'actual': .1, 'reference': .2, 'tolerance': 2e-5, 'ratio': 5000.}]))
            initialize = dist.init_process_group
            def group(*args, **kwargs):
                return initialize('gloo', init_method=f'file://{name}/rendezvous',
                                  rank=0, world_size=1, timeout=kwargs['timeout'])
            def builder(arch, **kwargs):
                return build_model(arch, use_cpu_initialization=True, **kwargs)
            with patch.object(e0c_fp64, 'configure_gpu', return_value={'GPU_executed': False}), \
                 patch.object(torch.nn.Module, 'cuda', lambda self: self), \
                 patch.object(dist, 'init_process_group', side_effect=group), \
                 patch('megatron.core.models.qwen.model.build_model', side_effect=builder), \
                 patch('megatron.core.tensor_parallel.random.model_parallel_cuda_manual_seed'), \
                 patch('torch.cuda.current_device', return_value='cpu'), \
                 patch.object(get_cuda_rng_tracker(), 'fork', side_effect=lambda: contextlib.nullcontext()), \
                 patch('torch.tensor', side_effect=cpu_tensor_factory(torch.tensor)), \
                 patch('torch.ones', side_effect=cpu_tensor_factory(torch.ones)):
                e0c_fp64.run_model(directory, 'hf')
                e0c_fp64.run_model(directory, 'native')
            result = json.loads((directory / 'native-fp64.json').read_text())
            self.assertTrue(result['fp64_implementations_agree'])
            self.assertFalse(result['acceptance_override'])
            self.assertFalse(result['GPU_conversion_validated'])
            self.assertFalse(result['environment']['GPU_executed'])
            self.assertEqual(result['comparisons']['gradient']['elements'], arch.hidden * arch.intermediate)
            for stats in result['comparisons'].values():
                self.assertLess(stats['max_abs'], 1e-12)
            for backend in ('hf', 'native'):
                report = json.loads((directory / f'{backend}-fp64.json').read_text())
                self.assertEqual(report['tensors'], file_record(directory / 'reference-fp64' / f'{backend}.safetensors'))
            self.assertTrue(e0c_fp64.summarize(directory, 0))  # CPU evidence cannot be accepted.

    def test_fp64_comparison_rejects_nonfinite_and_incorrect_gradients(self):
        expected = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        self.assertTrue(e0c_fp64.compare64(expected, expected)['agrees'])
        actual = expected.clone()
        actual[0, 1] += 1e-5
        self.assertFalse(e0c_fp64.compare64(actual, expected)['agrees'])
        actual[0, 0] = float('nan')
        result = e0c_fp64.compare64(actual, expected)
        self.assertFalse(result['agrees'])
        self.assertEqual(result['nonfinite'], 1)
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            e0c_fp64.compare64(expected.float(), expected)

    def test_launcher_failure_stops_native_and_git_scope(self):
        with tempfile.TemporaryDirectory(prefix='pier-fp64-launch-') as name:
            directory = Path(name)
            old = directory / 'old'
            old.mkdir()
            (old / 'native-fp32-tp1-rank0.json').write_text('{}')
            fake = directory / 'python'
            fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CALLS"\nif [[ "$*" == *"--phase hf"* ]]; then exit 23; fi\nexit 0\n')
            srun = directory / 'srun'
            srun.write_text('#!/usr/bin/env bash\nwhile [[ "$1" == --* ]]; do shift; done\nexec "$@"\n')
            fake.chmod(0o755)
            srun.chmod(0o755)
            env = dict(os.environ, PIER_ROOT=str(ROOT), PIER_PYTHON=str(fake),
                PIER_E0C_FP32_RUN=str(old), PIER_E0C_FP64_RUN_DIR=str(directory / 'new'),
                SLURM_JOB_NUM_NODES='1', SLURM_JOB_ID='fixture', CALLS=str(directory / 'calls'),
                PATH=str(directory) + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', 'experiments/qwen/e0c_fp64.sbatch'], cwd=ROOT,
                                    env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
            calls = (directory / 'calls').read_text()
            self.assertIn('--phase summary', calls)
            self.assertIn('--launcher-exit 23', calls)
            self.assertNotIn('--phase native', calls)
        for path, ignored in [('local/qwen/e0c-fp64-fixture/summary.json', False),
                              ('local/qwen/e0c-fp64-fixture/hf.log', False),
                              ('local/qwen/e0c-fp64-fixture/reference-fp64/hf.safetensors', True)]:
            self.assertEqual(subprocess.run(['git', 'check-ignore', '--no-index', '-q', path], cwd=ROOT).returncode == 0, ignored)


if __name__ == '__main__':
    unittest.main()
