"""Architecture and initialization contract for native Qwen training."""

from dataclasses import asdict

import torch

from .model import build_model, recompute_config
from .weights import load_weights


def training_defaults(arch):
    """Defaults must be installed before Megatron validation/tokenizer creation."""
    return dict(num_layers=arch.layers, hidden_size=arch.hidden,
                ffn_hidden_size=arch.intermediate, num_attention_heads=arch.heads,
                group_query_attention=True, num_query_groups=arch.kv_heads,
                kv_channels=arch.head_dim, max_position_embeddings=arch.max_positions,
                normalization='RMSNorm', norm_epsilon=arch.rms_epsilon,
                position_embedding_type='rope', rotary_base=arch.rope_base, rotary_percent=1.,
                swiglu=True, add_bias_linear=False, add_qkv_bias=True,
                untie_embeddings_and_output_weights=not arch.tied_embeddings,
                hidden_dropout=0., attention_dropout=arch.attention_dropout,
                init_method_std=arch.init_std, padded_vocab_size=arch.vocab,
                bf16=True, transformer_impl='local', no_persist_layer_norm=True,
                masked_softmax_fusion=False, bias_gelu_fusion=False,
                bias_swiglu_fusion=False, bias_dropout_fusion=False,
                apply_rope_fusion=False, gradient_accumulation_fusion=False,
                attention_softmax_in_fp32=True, apply_query_key_layer_scaling=False)


def training_recompute_config(args, arch):
    return recompute_config(
        arch, args.tensor_model_parallel_size,
        recompute_granularity=getattr(args, 'recompute_granularity', None),
        recompute_method=getattr(args, 'recompute_method', None),
        recompute_num_layers=getattr(args, 'recompute_num_layers', None),
        distribute_saved_activations=getattr(args, 'distribute_saved_activations', False))


def validate_training_contract(args, arch):
    differences = {key: (getattr(args, key, None), value)
                   for key, value in training_defaults(arch).items()
                   if getattr(args, key, None) != value}
    if differences:
        raise ValueError(f'Qwen CLI differs from native model recipe: {differences}')
    arch.validate_tp(args.tensor_model_parallel_size)
    if (args.pipeline_model_parallel_size != 1 or args.context_parallel_size != 1
            or args.expert_model_parallel_size != 1 or args.num_experts is not None):
        raise ValueError('Qwen training currently requires dense PP1/CP1/EP1')
    for key in ('use_legacy_models', 'use_distributed_optimizer', 'sequence_parallel',
                'fp16', 'fp8', 'use_rope_scaling', 'apply_layernorm_1p', 'squared_relu',
                'apply_residual_connection_post_layernorm', 'qk_layernorm', 'multi_latent_attention',
                'fp32_residual_connection', 'rotary_interleaved', 'rotary_seq_len_interpolation_factor',
                'fp16_lm_cross_entropy', 'defer_embedding_wgrad_compute', 'cross_entropy_loss_fusion',
                'enable_cuda_graph', 'use_custom_fsdp', 'use_torch_fsdp2', 'yaml_cfg',
                'spec', 'use_checkpoint_args',
                'virtual_pipeline_model_parallel_size', 'mtp_num_layers'):
        if getattr(args, key, None):
            raise ValueError(f'{key} is outside the initial unfused Qwen training recipe')
    training_recompute_config(args, arch)
    if not 1 <= args.seq_length <= arch.max_positions:
        raise ValueError('sequence length exceeds the checkpoint position range')
    if args.train_iters is None or not 1 <= args.train_iters <= 500:
        raise ValueError('Qwen experiment runs require 1..500 attempted steps')
    if args.outer_runtime != 'centered' or not args.local_sgd_inner_average:
        raise ValueError('Qwen experiment entrypoint requires centered Local-SGD')
    if args.tokenizer_type != 'HuggingFaceTokenizer':
        raise ValueError('Qwen training requires the pinned real tokenizer')


def initialize_training_model(args, arch, source):
    """Return the raw model already loaded, before DDP/master/optimizer creation."""
    from megatron.core import parallel_state as ps
    validate_training_contract(args, arch)
    recompute = training_recompute_config(args, arch)
    model = build_model(arch, dtype=torch.bfloat16,
                        use_cpu_initialization=args.use_cpu_initialization, **recompute)
    model.config.deterministic_mode = args.deterministic_mode
    receipt = load_weights(model, source, arch,
                           ps.get_tensor_model_parallel_world_size(), ps.get_tensor_model_parallel_rank())
    receipt['architecture'] = asdict(arch)
    receipt['exact_output_vocabulary'] = arch.vocab
    receipt['loaded_before_optimizer_construction'] = True
    receipt['activation_recompute'] = recompute
    return model, receipt
