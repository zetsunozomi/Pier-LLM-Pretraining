"""Native Megatron/HF forward and gradient parity on real CPU Gloo groups.

No pretrained weights or GPU results: tiny random models retain the three
checkpoint families' Q:KV ratios and tied/untied embedding choices.
Only CPU device selection and the unused CUDA RNG context are adapted for
Megatron's unfused, zero-dropout CPU test. The actual model/math/TP code runs.
"""

import contextlib
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Qwen2Config, Qwen2ForCausalLM

from megatron.core import parallel_state as ps
from megatron.core.models.qwen.config import QwenArchitecture
from megatron.core.models.qwen.model import QwenRMSNorm, build_model
from megatron.core.models.qwen.training import initialize_training_model, training_defaults
from megatron.core.models.qwen.weights import (
    SafeTensorSource, TensorSource, load_weights, parameter_mappings, validate_source,
)
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker


def tiny_config(heads=12, kv_heads=2, tied=True):
    return Qwen2Config(vocab_size=64, hidden_size=heads * 4, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=heads, num_key_value_heads=kv_heads,
        max_position_embeddings=64, rms_norm_eps=1e-6, rope_theta=1000000.,
        tie_word_embeddings=tied, attention_dropout=0., use_cache=False,
        bos_token_id=1, eos_token_id=2, pad_token_id=None)


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    records = []
    try:
        for heads, kv_heads, tied, sizes in ((12, 2, True, (1, 2)),
                                           (16, 2, True, (1, 2)),
                                           (28, 4, False, (1, 2, 4))):
            for tp in sizes:
                ps.initialize_model_parallel(tensor_model_parallel_size=tp, num_subgroups=4 // tp)
                torch.manual_seed(77)
                config = tiny_config(heads, kv_heads, tied)
                config._attn_implementation = 'eager'
                hf = Qwen2ForCausalLM(config).eval()
                # Nonzero QKV biases make omitted/permuted bias conversions detectable.
                with torch.no_grad():
                    for name, p in hf.named_parameters():
                        if name.endswith('_proj.bias'):
                            p.copy_(torch.linspace(-.07, .08, p.numel()))
                arch = QwenArchitecture.from_config(config.to_dict())
                native = build_model(arch, dtype=torch.float32, use_cpu_initialization=True,
                                     parallel_output=False).eval()
                tp_rank = ps.get_tensor_model_parallel_rank()
                if tp > 1:
                    try:
                        load_weights(native, TensorSource(hf.state_dict()), arch, tp, (tp_rank + 1) % tp)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('loader accepted the wrong TP coordinate')
                load_weights(native, TensorSource(hf.state_dict()), arch, tp, tp_rank)
                tokens = torch.tensor([[1, 3, 5, 7, 9, 11], [2, 8, 4, 16, 10, 12]])
                positions = torch.arange(tokens.shape[1]).expand_as(tokens)
                causal = torch.ones(1, 1, tokens.shape[1], tokens.shape[1], dtype=torch.bool).triu(1)
                with patch('torch.cuda.current_device', return_value='cpu'), \
                        patch.object(get_cuda_rng_tracker(), 'fork',
                                     side_effect=lambda *a, **k: contextlib.nullcontext()):
                    output = native(tokens, positions, causal)
                    expected = hf(tokens, position_ids=positions, use_cache=False).logits
                    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
                    actual_loss = torch.nn.functional.cross_entropy(output[:, :-1].reshape(-1, 64),
                                                                   tokens[:, 1:].reshape(-1))
                    reference_loss = torch.nn.functional.cross_entropy(expected[:, :-1].reshape(-1, 64),
                                                                      tokens[:, 1:].reshape(-1))
                    actual_loss.backward()
                    reference_loss.backward()
                gradients = TensorSource({name: p.grad for name, p in hf.named_parameters(remove_duplicate=False)})
                params = dict(native.named_parameters())
                grad_error = 0.
                for mapping in parameter_mappings(arch, tp, tp_rank):
                    expected_grad = mapping.materialize(gradients)
                    actual_grad = params[mapping.target].grad
                    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)
                    grad_error = max(grad_error, float((actual_grad - expected_grad).abs().max()))
                records.append(dict(heads=heads, kv_heads=kv_heads, tied=tied, tp=tp,
                                    logits_max_abs=float((output - expected).detach().abs().max()),
                                    gradients_max_abs=grad_error,
                                    loss_abs=abs(float(actual_loss.detach() - reference_loss.detach()))))
                if heads == 16 and tp == 2:
                    args = SimpleNamespace(**training_defaults(arch),
                        tensor_model_parallel_size=tp, pipeline_model_parallel_size=1,
                        context_parallel_size=1, expert_model_parallel_size=1, num_experts=None,
                        seq_length=6, train_iters=11, outer_runtime='centered', local_sgd_inner_average=True,
                        tokenizer_type='HuggingFaceTokenizer', use_cpu_initialization=True,
                        deterministic_mode=False)
                    initialized, receipt = initialize_training_model(args, arch, TensorSource(hf.state_dict()))
                    assert receipt['loaded_before_optimizer_construction']
                    assert not receipt['optimizer_state_restored']
                    parameters = dict(initialized.named_parameters())
                    for mapping in parameter_mappings(arch, tp, tp_rank):
                        expected_parameter = mapping.materialize(TensorSource(hf.state_dict())).bfloat16()
                        assert torch.equal(parameters[mapping.target], expected_parameter)
                ps.destroy_model_parallel()
        Path(directory, f'rank-{rank}.json').write_text(json.dumps(records, indent=2))
        dist.barrier()
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


class QwenWeightTests(unittest.TestCase):
    def test_native_forward_and_gradients_tp_1_2_4(self):
        with tempfile.TemporaryDirectory(prefix='pier-qwen-parity-') as name:
            mp.spawn(worker, args=(f'file://{name}/rendezvous', name), nprocs=4, join=True)
            records = [row for rank in range(4) for row in json.loads(Path(name, f'rank-{rank}.json').read_text())]
            print(json.dumps({'CPU_Qwen_parity_records': len(records), 'GPU_executed': False,
                              'max_logits_abs': max(r['logits_max_abs'] for r in records),
                              'max_gradients_abs': max(r['gradients_max_abs'] for r in records),
                              'max_loss_abs': max(r['loss_abs'] for r in records)}))

    def test_safetensors_shards_missing_keys_and_tied_weights(self):
        from safetensors.torch import save_file
        hf = Qwen2ForCausalLM(tiny_config())
        arch = QwenArchitecture.from_config(hf.config.to_dict())
        original = hf.state_dict()
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            # Omit the shared lm_head duplicate, as HF safe serialization does.
            items = [(key, value) for key, value in original.items() if key != 'lm_head.weight']
            weight_map = {}
            for index in range(2):
                shard_name = f'model-{index}.safetensors'
                shard = dict(items[index::2])
                save_file(shard, str(directory / shard_name))
                weight_map.update({key: shard_name for key in shard})
            (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
            source = SafeTensorSource(directory)
            validate_source(arch, source)
            for rank in (0, 1):
                for mapping in parameter_mappings(arch, 2, rank):
                    self.assertTrue(torch.equal(mapping.materialize(source), mapping.materialize(TensorSource(original))))
            source.shapes.pop('model.layers.0.self_attn.q_proj.bias')
            with self.assertRaises(ValueError):
                validate_source(arch, source)
            bad = dict(original)
            bad['lm_head.weight'] = bad['lm_head.weight'] + 1
            with self.assertRaises(ValueError):
                validate_source(arch, TensorSource(bad))
            with self.assertRaises(ValueError):
                arch.validate_tp(4)

    def test_bf16_rmsnorm_cast_order(self):
        from types import SimpleNamespace
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        config = SimpleNamespace(layernorm_zero_centered_gamma=False, sequence_parallel=False,
                                 use_cpu_initialization=True, params_dtype=torch.bfloat16)
        actual = QwenRMSNorm(config, 48, eps=1e-6)
        reference = Qwen2RMSNorm(48, eps=1e-6).bfloat16()
        torch.manual_seed(19)
        with torch.no_grad():
            actual.weight.copy_(torch.randn(48, dtype=torch.bfloat16))
            reference.weight.copy_(actual.weight)
        a = torch.randn(2, 7, 48, dtype=torch.bfloat16, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        x, y = actual(a), reference(b)
        self.assertTrue(torch.equal(x, y))
        x.float().square().sum().backward()
        y.float().square().sum().backward()
        self.assertTrue(torch.equal(a.grad, b.grad))
        self.assertTrue(torch.equal(actual.weight.grad, reference.weight.grad))


if __name__ == '__main__':
    unittest.main()
