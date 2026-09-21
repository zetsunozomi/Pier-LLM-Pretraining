"""Diagnostic-only FP64 adapters; never used by the production Qwen builder.

Promote RMSNorm, softmax, RoPE and CE as well as weights/activations. Merely
calling model.double() leaves explicit FP32 casts in both model implementations.
Only one weight requires a gradient, bounding memory while retaining its exact
partial derivative through every downstream layer.
"""

from contextlib import contextmanager
from types import MethodType
from unittest.mock import patch

import torch


def _hf_norm(self, x):
    if x.dtype != torch.float64:
        raise ValueError('FP64 diagnostic received a lower-precision norm input')
    variance = x.pow(2).mean(-1, keepdim=True)
    return self.weight * (x * torch.rsqrt(variance + self.variance_epsilon))


def _native_norm(self, x):
    if x.dtype != torch.float64:
        raise ValueError('FP64 diagnostic received a lower-precision norm input')
    inverse = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
    return (x * inverse) * self.weight


def _hf_rope(self, x, position_ids):
    frequency = position_ids.to(device=x.device, dtype=torch.float64).unsqueeze(-1) * self.inv_freq
    angle = torch.cat((frequency, frequency), dim=-1)
    return angle.cos(), angle.sin()


def promote(model, backend, arch, layer):
    """Adapt a fresh, loaded FP32 model; no parameter names or values change."""
    from megatron.core.models.qwen.model import QwenRMSNorm
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    if backend not in ('hf', 'native'):
        raise ValueError('unknown FP64 backend')
    if arch.attention_dropout != 0.:
        raise ValueError('FP64 diagnostic requires zero dropout')
    name = (f'model.layers.{layer}.mlp.down_proj.weight' if backend == 'hf'
            else f'decoder.layers.{layer}.mlp.linear_fc2.weight')
    model.requires_grad_(False)
    model.double()
    parameters = dict(model.named_parameters())
    parameters[name].requires_grad_(True)
    if any(p.dtype != torch.float64 for p in parameters.values()):
        raise ValueError('FP64 parameter promotion incomplete')
    count = 0
    for module in model.modules():
        if isinstance(module, Qwen2RMSNorm):
            module.forward = MethodType(_hf_norm, module)
            count += 1
        elif isinstance(module, QwenRMSNorm):
            module.forward = MethodType(_native_norm, module)
            count += 1
    if count != 2 * arch.layers + 1:
        raise ValueError('FP64 norm coverage differs')
    parameter = parameters[name]
    inv_freq = 1. / (arch.rope_base ** (torch.arange(
        0, arch.head_dim, 2, dtype=torch.float64, device=parameter.device) / arch.head_dim))
    if backend == 'hf':
        model.model.rotary_emb.inv_freq = inv_freq
        model.model.rotary_emb.forward = MethodType(_hf_rope, model.model.rotary_emb)
    else:
        # Native unfused RoPE follows inv_freq dtype through outer/cos/sin.
        model.rotary_pos_emb.inv_freq = inv_freq
        model.rotary_pos_emb.forward.cache_clear()
    return name, parameter, {'norm_modules': count, 'rope_dtype': 'float64',
                            'trainable_parameters': [name], 'trainable_elements': parameter.numel()}


@contextmanager
def double_softmax():
    """Override only explicit FP32 softmax casts on FP64 inputs in this process."""
    original = torch.nn.functional.softmax
    receipt = {'fp64_softmax_calls': 0}
    def softmax(input, dim=None, _stacklevel=3, dtype=None):
        if input.dtype == torch.float64:
            if dtype not in (None, torch.float32, torch.float64):
                raise ValueError('unexpected softmax cast in FP64 diagnostic')
            dtype = torch.float64
            receipt['fp64_softmax_calls'] += 1
        return original(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    with patch('torch.nn.functional.softmax', side_effect=softmax):
        yield receipt


def loss64(logits, tokens):
    if logits.dtype != torch.float64:
        raise ValueError('FP64 diagnostic produced lower-precision logits')
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                              tokens[:, 1:].reshape(-1))
