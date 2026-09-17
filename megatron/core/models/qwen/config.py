"""Validated architecture contract for the pinned, dense Qwen2.5 base models."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class QwenArchitecture:
    layers: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    vocab: int
    max_positions: int
    rms_epsilon: float
    rope_base: float
    tied_embeddings: bool
    attention_dropout: float
    init_std: float

    @classmethod
    def from_config(cls, config):
        if config.get('model_type') != 'qwen2' or config.get('hidden_act') != 'silu':
            raise ValueError('requires dense Qwen2/Qwen2.5 with SiLU')
        for name in ('use_sliding_window', 'use_mrope', 'rope_scaling'):
            if config.get(name):
                raise ValueError(f'{name} is outside this Qwen2.5 contract')
        if any(kind != 'full_attention' for kind in (config.get('layer_types') or [])):
            raise ValueError('requires full attention in every layer')
        arch = cls(*(int(config[key]) for key in ('num_hidden_layers', 'hidden_size',
                     'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
                     'vocab_size', 'max_position_embeddings')),
                   rms_epsilon=float(config['rms_norm_eps']), rope_base=float(config['rope_theta']),
                   tied_embeddings=bool(config['tie_word_embeddings']),
                   attention_dropout=float(config.get('attention_dropout', 0.)),
                   init_std=float(config.get('initializer_range', .02)))
        if min(arch.layers, arch.hidden, arch.intermediate, arch.heads, arch.kv_heads,
               arch.vocab, arch.max_positions) < 1:
            raise ValueError('architecture dimensions must be positive')
        if arch.hidden % arch.heads or arch.heads % arch.kv_heads or arch.head_dim % 2:
            raise ValueError('invalid attention head/group dimensions')
        if config.get('head_dim', arch.head_dim) != arch.head_dim:
            raise ValueError('nonstandard head dimension is unsupported')
        if not (0 <= arch.attention_dropout < 1 and arch.rms_epsilon > 0 and arch.rope_base > 0
                and arch.init_std > 0 and all(math.isfinite(v) for v in
                    (arch.rms_epsilon, arch.rope_base, arch.init_std))):
            raise ValueError('invalid normalization/RoPE/dropout/initialization parameters')
        return arch

    @property
    def head_dim(self):
        return self.hidden // self.heads

    def validate_tp(self, size, rank=0):
        if size < 1 or not 0 <= rank < size:
            raise ValueError('invalid tensor parallel size/rank')
        if any(value % size for value in (self.heads, self.kv_heads, self.intermediate, self.vocab)):
            raise ValueError('TP must divide Q heads, KV groups, FFN and exact checkpoint vocabulary')

    def hf_shapes(self):
        shapes = {'model.embed_tokens.weight': (self.vocab, self.hidden),
                  'model.norm.weight': (self.hidden,)}
        if not self.tied_embeddings:
            shapes['lm_head.weight'] = (self.vocab, self.hidden)
        for layer in range(self.layers):
            root = f'model.layers.{layer}.'
            shapes[root + 'input_layernorm.weight'] = (self.hidden,)
            shapes[root + 'post_attention_layernorm.weight'] = (self.hidden,)
            for kind, rows in (('q', self.hidden), ('k', self.kv_heads * self.head_dim),
                               ('v', self.kv_heads * self.head_dim)):
                shapes[root + f'self_attn.{kind}_proj.weight'] = (rows, self.hidden)
                shapes[root + f'self_attn.{kind}_proj.bias'] = (rows,)
            shapes[root + 'self_attn.o_proj.weight'] = (self.hidden, self.hidden)
            for kind in ('gate', 'up'):
                shapes[root + f'mlp.{kind}_proj.weight'] = (self.intermediate, self.hidden)
            shapes[root + 'mlp.down_proj.weight'] = (self.hidden, self.intermediate)
        return shapes

    def unique_parameters(self):
        return sum(math.prod(shape) for shape in self.hf_shapes().values())
