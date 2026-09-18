"""Actual native Qwen recomputation/math on CPU with real Gloo TP groups.

Only CUDA device/RNG access is adapted for this zero-dropout CPU check. The
Megatron checkpoint autograd function, recomputed model and TP collectives run
unchanged. This does not validate CUDA RNG, memory savings or GPU performance.
"""

import contextlib
from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Qwen2ForCausalLM

from megatron.core import parallel_state as ps
from megatron.core.models.qwen.config import QwenArchitecture
from megatron.core.models.qwen.model import build_model, recompute_config
from megatron.core.models.qwen.training import initialize_training_model, training_defaults
from megatron.core.models.qwen.weights import TensorSource, load_weights
from megatron.core.outer_sync.verification import same
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from test_weights import tiny_config


def modes(tp):
    result = [('selective', dict(recompute_granularity='selective'))]
    for method in ('uniform', 'block'):
        for count in (1, 2):
            result.append((f'{method}-{count}', dict(recompute_granularity='full',
                           recompute_method=method, recompute_num_layers=count)))
        if tp > 1:
            result.append((f'{method}-distributed', dict(recompute_granularity='full',
                           recompute_method=method, recompute_num_layers=1,
                           distribute_saved_activations=True)))
    return result


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=120))
    records = []
    try:
        for tp, heads, kv_heads, tied in ((1, 16, 2, True), (2, 16, 2, True), (4, 28, 4, False)):
            ps.initialize_model_parallel(tensor_model_parallel_size=tp, num_subgroups=4 // tp)
            torch.manual_seed(77)
            hf = Qwen2ForCausalLM(tiny_config(heads, kv_heads, tied))
            source = TensorSource(hf.state_dict())
            arch = QwenArchitecture.from_config(hf.config.to_dict())
            tokens = torch.tensor([[1, 3, 5, 7, 9, 11], [2, 8, 4, 16, 10, 12]])
            positions = torch.arange(tokens.shape[1]).expand_as(tokens)
            causal = torch.ones(1, 1, tokens.shape[1], tokens.shape[1], dtype=torch.bool).triu(1)
            tp_rank = ps.get_tensor_model_parallel_rank()
            with patch('torch.cuda.current_device', return_value='cpu'), \
                    patch.object(get_cuda_rng_tracker(), 'fork',
                                 side_effect=lambda *a, **k: contextlib.nullcontext()), \
                    patch('megatron.core.tensor_parallel.random._get_cuda_rng_state',
                          side_effect=lambda: torch.get_rng_state().clone()), \
                    patch('megatron.core.tensor_parallel.random._set_cuda_rng_state'):
                for dtype in (torch.float32, torch.bfloat16):
                    baseline = build_model(arch, dtype=dtype, use_cpu_initialization=True).train()
                    load_weights(baseline, source, arch, tp, tp_rank)
                    expected = baseline(tokens, positions, causal, runtime_gather_output=True)
                    expected_loss = torch.nn.functional.cross_entropy(
                        expected[:, :-1].float().reshape(-1, arch.vocab), tokens[:, 1:].reshape(-1))
                    expected_loss.backward()
                    gradients = {name: p.grad.clone() for name, p in baseline.named_parameters()}
                    for label, options in modes(tp):
                        if dtype == torch.bfloat16:
                            args = SimpleNamespace(**training_defaults(arch), **options,
                                tensor_model_parallel_size=tp, pipeline_model_parallel_size=1,
                                context_parallel_size=1, expert_model_parallel_size=1, num_experts=None,
                                seq_length=6, train_iters=11, outer_runtime='centered',
                                local_sgd_inner_average=True, tokenizer_type='HuggingFaceTokenizer',
                                use_cpu_initialization=True, deterministic_mode=False)
                            model, receipt = initialize_training_model(args, arch, source)
                            assert receipt['activation_recompute'] == recompute_config(arch, tp, **options)
                        else:
                            model = build_model(arch, dtype=dtype, use_cpu_initialization=True, **options)
                            load_weights(model, source, arch, tp, tp_rank)
                        model.train()
                        for field, value in recompute_config(arch, tp, **options).items():
                            assert getattr(model.config, field) == value
                        visits = [0] * arch.layers
                        def hook(index):
                            def count(*unused):
                                visits[index] += 1
                            return count
                        handles = [layer.self_attention.core_attention.register_forward_pre_hook(hook(index))
                                   for index, layer in enumerate(model.decoder.layers)]
                        output = model(tokens, positions, causal, runtime_gather_output=True)
                        loss = torch.nn.functional.cross_entropy(
                            output[:, :-1].float().reshape(-1, arch.vocab), tokens[:, 1:].reshape(-1))
                        after_forward_rng = torch.get_rng_state().clone()
                        loss.backward()
                        assert torch.equal(torch.get_rng_state(), after_forward_rng)
                        torch.testing.assert_close(output, expected, rtol=0., atol=0.)
                        torch.testing.assert_close(loss, expected_loss, rtol=0., atol=0.)
                        same(output, expected, f'{label}: logits')
                        same(loss, expected_loss, f'{label}: loss')
                        for name, parameter in model.named_parameters():
                            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0., atol=0.,
                                                       msg=lambda message: f'{label}: {name}: {message}')
                            same(parameter.grad, gradients[name], f'{label}: {name}: gradient')
                        expected_visits = ([2, 1] if options.get('recompute_method') == 'block'
                                           and options['recompute_num_layers'] == 1 else [2, 2])
                        assert visits == expected_visits, (label, visits)
                        # Eval remains an ordinary forward even when recompute is configured.
                        visits[:] = [0] * arch.layers
                        model.eval()
                        with torch.no_grad():
                            torch.testing.assert_close(model(tokens, positions, causal, runtime_gather_output=True),
                                                       expected, rtol=0., atol=0.)
                        assert visits == [1, 1]
                        for handle in handles:
                            handle.remove()
                        records.append(dict(tp=tp, dtype=str(dtype), mode=label,
                                            logits_bitwise_equal=True, gradients_bitwise_equal=True,
                                            loss_bitwise_equal=True, training_attention_visits=expected_visits,
                                            GPU_executed=False))
            ps.destroy_model_parallel()
        Path(directory, f'rank-{rank}.json').write_text(json.dumps(records, indent=2))
        dist.barrier()
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


class QwenRecomputeTests(unittest.TestCase):
    def test_training_recompute_outputs_gradients_and_actual_reexecution(self):
        with tempfile.TemporaryDirectory(prefix='pier-qwen-recompute-') as name:
            mp.spawn(worker, args=(f'file://{name}/rendezvous', name), nprocs=4, join=True)
            records = [record for rank in range(4)
                       for record in json.loads(Path(name, f'rank-{rank}.json').read_text())]
            self.assertEqual(len(records), 152)
            print(json.dumps({'CPU_Qwen_recompute_records': len(records), 'GPU_executed': False,
                              'logits_loss_gradients_bitwise_equal': True}))

    def test_recompute_rejects_ignored_and_unsafe_configurations(self):
        arch = QwenArchitecture.from_config(tiny_config().to_dict())
        invalid = [dict(recompute_method='uniform'), dict(recompute_num_layers=1),
                   dict(distribute_saved_activations=True), dict(recompute_granularity='typo'),
                   dict(recompute_granularity='full'),
                   dict(recompute_granularity='full', recompute_method='uniform', recompute_num_layers=0),
                   dict(recompute_granularity='full', recompute_method='block', recompute_num_layers=3),
                   dict(recompute_granularity='full', recompute_method='block', recompute_num_layers=True),
                   dict(recompute_granularity='selective', recompute_method='block'),
                   dict(recompute_granularity='selective', recompute_num_layers=1),
                   dict(recompute_granularity='selective', distribute_saved_activations=True)]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                recompute_config(arch, 2, **options)
        with self.assertRaises(ValueError):
            recompute_config(arch, 1, recompute_granularity='full', recompute_method='uniform',
                             recompute_num_layers=1, distribute_saved_activations=True)
        config = tiny_config().to_dict()
        config['num_hidden_layers'] = 3
        with self.assertRaisesRegex(ValueError, 'divide'):
            recompute_config(QwenArchitecture.from_config(config), 2, recompute_granularity='full',
                             recompute_method='uniform', recompute_num_layers=2)


if __name__ == '__main__':
    unittest.main()
