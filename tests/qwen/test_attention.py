"""Qwen attention forward and Q/K/V gradient checks against pinned HF eager."""

import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

from megatron.core import parallel_state
from megatron.core.models.qwen.attention import QwenDotProductAttention
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


def attention(dtype, tp=1):
    config = TransformerConfig(num_layers=2, hidden_size=2048, num_attention_heads=16,
        num_query_groups=2, kv_channels=128, tensor_model_parallel_size=tp,
        params_dtype=dtype, bf16=dtype == torch.bfloat16, attention_dropout=0.,
        masked_softmax_fusion=False, attention_softmax_in_fp32=True,
        apply_query_key_layer_scaling=False, use_cpu_initialization=True)
    with patch.object(parallel_state, 'get_tensor_model_parallel_world_size', return_value=tp):
        return QwenDotProductAttention(config, 1, AttnMaskType.causal, 'self')


class QwenAttentionTests(unittest.TestCase):
    def test_hf_forward_and_qkv_gradients_on_tp_local_heads(self):
        torch.set_num_threads(1)
        for dtype in (torch.float32, torch.bfloat16):
            for tp in (1, 2):
                for implicit_mask in (True, False):
                    with self.subTest(dtype=dtype, tp=tp, implicit_mask=implicit_mask):
                        torch.manual_seed(512)
                        core = attention(dtype, tp)
                        heads, kv_heads = 16 // tp, 2 // tp
                        # Noncontiguous views represent native packed QKV projections.
                        packed = torch.randn(7, 2, heads + 2 * kv_heads, 128, dtype=dtype)
                        native = [x.detach().requires_grad_() for x in
                                  (packed[:, :, :heads], packed[:, :, heads:heads + kv_heads],
                                   packed[:, :, heads + kv_heads:])]
                        reference = [x.detach().permute(1, 2, 0, 3).contiguous().requires_grad_()
                                     for x in native]
                        mask = torch.ones(2, 1, 7, 7, dtype=torch.bool).triu(1)
                        if not implicit_mask:
                            mask[1, :, :, 2] = True  # batch-specific padding in addition to causality
                        hf_mask = torch.zeros_like(mask, dtype=dtype).masked_fill(mask, torch.finfo(dtype).min)
                        expected, _ = eager_attention_forward(
                            SimpleNamespace(num_key_value_groups=8, training=True), *reference,
                            hf_mask, scaling=128 ** -0.5, dropout=0.)
                        with patch.object(get_cuda_rng_tracker(), 'fork',
                                          side_effect=lambda: contextlib.nullcontext()):
                            actual = core(*native, None if implicit_mask else mask)
                        expected = expected.reshape(2, 7, heads * 128)
                        torch.testing.assert_close(actual.transpose(0, 1), expected, rtol=0., atol=0.)
                        upstream = torch.randn_like(expected)
                        actual.backward(upstream.transpose(0, 1).contiguous())
                        expected.backward(upstream)
                        for name, a, b in zip(('Q', 'K', 'V'), native, reference):
                            torch.testing.assert_close(a.grad.permute(1, 2, 0, 3), b.grad,
                                                       rtol=0., atol=0., msg=name)

    def test_mask_blocks_future_values_and_supports_single_query(self):
        torch.set_num_threads(1)
        torch.manual_seed(812)
        core = attention(torch.float32)
        q = torch.randn(4, 1, 16, 128)
        k = torch.randn(4, 1, 2, 128)
        v = torch.randn(4, 1, 2, 128, requires_grad=True)
        with patch.object(get_cuda_rng_tracker(), 'fork', side_effect=lambda: contextlib.nullcontext()):
            full = core(q, k, v, None)
            full[0].sum().backward()
            self.assertEqual(int(torch.count_nonzero(v.grad[1:])), 0)
            # A single cached query may attend to every preceding key.
            cached = core(q[-1:], k, v, None, attn_mask_type=AttnMaskType.no_mask)
            torch.testing.assert_close(cached[0], full[-1], rtol=1e-6, atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'explicit mask'):
            core(q[:2], k, v, None)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            core(q, k, v, torch.zeros(4, 4))
        with self.assertRaisesRegex(ValueError, 'packed'):
            core(q, k, v, None, packed_seq_params=object())


if __name__ == '__main__':
    unittest.main()
