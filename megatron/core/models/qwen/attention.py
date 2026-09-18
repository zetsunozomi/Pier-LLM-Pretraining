"""Unfused Qwen attention with the pinned HF eager operation order.

Reference: Transformers v4.57.3 modeling_qwen2.eager_attention_forward
(Apache-2.0). Megatron supplies local TP Q/K/V heads in sequence-first order.
Scaling is a separate operation after QK matmul, including during backward.
"""

import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType


def _repeat_kv(value, groups):
    # [batch, kv_heads, sequence, dim] -> [batch, query_heads, sequence, dim].
    if groups == 1:
        return value
    batch, heads, sequence, dim = value.shape
    return value[:, :, None].expand(batch, heads, groups, sequence, dim).reshape(
        batch, heads * groups, sequence, dim)


class QwenDotProductAttention(DotProductAttention):
    """Preserve TP interfaces while matching HF's scale, mask and softmax order."""

    def __init__(self, config, layer_number, attn_mask_type, attention_type,
                 attention_dropout=None, softmax_scale=None, cp_comm_type=None):
        if (config.apply_query_key_layer_scaling or config.sequence_parallel
                or config.masked_softmax_fusion):
            raise ValueError('Qwen eager attention requires unfused, non-layer-scaled, non-SP attention')
        super().__init__(config, layer_number, attn_mask_type, attention_type,
                         attention_dropout, softmax_scale, cp_comm_type)
        if softmax_scale is None:
            self.softmax_scale = self.hidden_size_per_attention_head ** -0.5

    def forward(self, query, key, value, attention_mask, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        if packed_seq_params is not None or attention_bias is not None:
            raise ValueError('Qwen eager attention does not support packed sequences or attention bias')
        query = query.permute(1, 2, 0, 3)
        groups = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        key = _repeat_kv(key.permute(1, 2, 0, 3), groups)
        value = _repeat_kv(value.permute(1, 2, 0, 3), groups)
        # Do not merge scaling into baddbmm alpha: that changes rounding and
        # moves the scale within the Q/K gradient computation.
        scores = torch.matmul(query, key.transpose(2, 3)) * self.softmax_scale
        mask_type = self.attn_mask_type if attn_mask_type is None else attn_mask_type
        if attention_mask is None and mask_type == AttnMaskType.causal and query.size(2) > 1:
            if query.size(2) != key.size(2):
                raise ValueError('non-square causal attention requires an explicit mask')
            attention_mask = torch.ones(query.size(2), key.size(2), dtype=torch.bool,
                                        device=query.device).triu(1)
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise ValueError('Qwen native attention expects a boolean mask (True = blocked)')
            # HF adds a mask in the score dtype, before the FP32 softmax cast.
            additive_mask = torch.zeros_like(attention_mask, dtype=scores.dtype).masked_fill(
                attention_mask, torch.finfo(scores.dtype).min)
            scores = scores + additive_mask
        probabilities = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        with tensor_parallel.get_cuda_rng_tracker().fork():
            probabilities = self.attention_dropout(probabilities)
        context = torch.matmul(probabilities, value)
        # [batch, heads, sequence, dim] -> Megatron's [sequence, batch, hidden].
        context = context.permute(2, 0, 1, 3).contiguous()
        return context.view(context.size(0), context.size(1), self.hidden_size_per_partition)
