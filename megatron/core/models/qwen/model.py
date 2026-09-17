"""Native dense Qwen2.5 model with an explicit unfused numerical recipe.

The RMSNorm cast order follows the checkpoint model's FP32 normalization and
cast-before-gain contract (Transformers v4.57.3 Qwen2RMSNorm, Apache-2.0).
This builder does not enable the legacy Hugging Face model wrapper or FSDP.
"""

import torch

from .config import QwenArchitecture


class QwenRMSNorm(torch.nn.Module):
    def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        if config.layernorm_zero_centered_gamma or config.sequence_parallel:
            raise ValueError('Qwen norm recipe currently requires ordinary gain and no sequence parallelism')
        device = 'cpu' if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype, device=device))
        self.eps = eps

    def forward(self, value):
        working = value.float()
        inverse_rms = torch.rsqrt(working.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (working * inverse_rms).to(dtype=value.dtype)
        return normalized * self.weight


def build_model(arch, *, dtype=torch.bfloat16, use_cpu_initialization=False, parallel_output=True):
    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    tp = ps.get_tensor_model_parallel_world_size()
    arch.validate_tp(tp, ps.get_tensor_model_parallel_rank())
    if (ps.get_pipeline_model_parallel_world_size() != 1 or ps.get_context_parallel_world_size() != 1
            or ps.get_expert_model_parallel_world_size() != 1):
        raise ValueError('Qwen builder currently requires dense PP1/CP1/EP1')
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Qwen builder supports FP32 or BF16')
    config = TransformerConfig(num_layers=arch.layers, hidden_size=arch.hidden,
        ffn_hidden_size=arch.intermediate, num_attention_heads=arch.heads,
        num_query_groups=arch.kv_heads, kv_channels=arch.head_dim, normalization='RMSNorm',
        layernorm_epsilon=arch.rms_epsilon, gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        add_bias_linear=False, add_qkv_bias=True, hidden_dropout=0., attention_dropout=arch.attention_dropout,
        use_cpu_initialization=use_cpu_initialization, params_dtype=dtype, bf16=dtype == torch.bfloat16,
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=1, init_method_std=arch.init_std,
        persist_layer_norm=False, masked_softmax_fusion=False, bias_activation_fusion=False,
        bias_dropout_fusion=False, apply_rope_fusion=False, gradient_accumulation_fusion=False,
        attention_softmax_in_fp32=True, apply_query_key_layer_scaling=False)
    layer = get_gpt_layer_local_spec()
    layer.submodules.input_layernorm = QwenRMSNorm
    layer.submodules.pre_mlp_layernorm = QwenRMSNorm
    block = TransformerBlockSubmodules(layer_specs=[layer] * arch.layers, layer_norm=QwenRMSNorm)
    model = GPTModel(config, block, arch.vocab, arch.max_positions,
                     position_embedding_type='rope', rotary_percent=1., rotary_base=arch.rope_base,
                     share_embeddings_and_output_weights=arch.tied_embeddings,
                     parallel_output=parallel_output)
    model.qwen_architecture = arch
    model.qwen_tp_size = tp
    model.qwen_tp_rank = ps.get_tensor_model_parallel_rank()
    return model
