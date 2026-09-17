"""Stream a local HF safetensors checkpoint into native Megatron TP parameters.

Q/K/V are packed per query group. SwiGLU packs each rank's gate then up rows.
No checkpoint download, pickle load, full-model flatten or dtype guessing occurs.
Call before optimizer construction: this is a weights-only warm start.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from .config import QwenArchitecture


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class TensorSource:
    """In-memory source for small HF models and conversion tests."""
    def __init__(self, tensors):
        self.tensors = tensors
        self.shapes = {name: tuple(value.shape) for name, value in tensors.items()}
        self.dtypes = {name: value.dtype for name, value in tensors.items()}

    def read(self, name, axis=None, start=None, stop=None):
        value = self.tensors[name].detach()
        return value if axis is None else value.narrow(axis, start, stop - start)


class SafeTensorSource:
    """Inspect headers once and mmap only a requested parameter/slice at a time."""
    def __init__(self, directory):
        from safetensors import safe_open
        self.directory = Path(directory)
        index = self.directory / 'model.safetensors.index.json'
        if index.exists():
            self.weight_map = json.loads(index.read_text())['weight_map']
        else:
            with safe_open(str(self.directory / 'model.safetensors'), framework='pt', device='cpu') as f:
                self.weight_map = {name: 'model.safetensors' for name in f.keys()}
        self.shapes, self.dtypes = {}, {}
        dtype_map = {'F32': torch.float32, 'BF16': torch.bfloat16}
        for filename in sorted(set(self.weight_map.values())):
            if Path(filename).name != filename or not filename.endswith('.safetensors'):
                raise ValueError('checkpoint index must reference local safetensors filenames')
            with safe_open(str(self.directory / filename), framework='pt', device='cpu') as f:
                for key in f.keys():
                    if key in self.shapes or self.weight_map.get(key) != filename:
                        raise ValueError(f'duplicate or unindexed checkpoint tensor: {key}')
                    view = f.get_slice(key)
                    self.shapes[key] = tuple(view.get_shape())
                    self.dtypes[key] = dtype_map.get(view.get_dtype())
        if set(self.shapes) != set(self.weight_map):
            raise ValueError('checkpoint index contains missing tensors')

    def read(self, name, axis=None, start=None, stop=None):
        from safetensors import safe_open
        with safe_open(str(self.directory / self.weight_map[name]), framework='pt', device='cpu') as f:
            if axis is None:
                return f.get_tensor(name)
            indices = [slice(None)] * len(self.shapes[name])
            indices[axis] = slice(start, stop)
            return f.get_slice(name)[tuple(indices)]

    def file_manifest(self):
        files = sorted(set(self.weight_map.values()))
        files += [name for name in ('config.json', 'model.safetensors.index.json')
                  if (self.directory / name).exists()]
        return {name: {'bytes': (self.directory / name).stat().st_size,
                       'sha256': sha256_file(self.directory / name)} for name in files}


@dataclass(frozen=True)
class ParameterMapping:
    target: str
    shape: tuple
    sources: tuple  # (name, slice axis, start, stop)
    operation: str = 'copy'
    query_groups: int = 1

    def materialize(self, source):
        tensors = [source.read(*selection) for selection in self.sources]
        if self.operation == 'qkv':
            # HF head-major Q rows become Megatron [group, Q..., K, V].
            tail = tensors[0].shape[1:]
            tensors = [x.reshape(self.query_groups, -1, *tail) for x in tensors]
            return torch.cat(tensors, dim=1).reshape(self.shape).contiguous()
        if self.operation == 'concat':
            return torch.cat(tensors, dim=0).contiguous()
        return tensors[0].contiguous()


def parameter_mappings(arch, tp_size=1, tp_rank=0):
    arch.validate_tp(tp_size, tp_rank)
    h, d, groups = arch.hidden, arch.head_dim, arch.kv_heads // tp_size
    v, f = arch.vocab // tp_size, arch.intermediate // tp_size
    q, kv = h // tp_size, groups * d
    mappings = []
    def part(name, axis, width):
        return (name, axis, tp_rank * width, (tp_rank + 1) * width)
    def full(name):
        return (name, None, None, None)
    def add(target, shape, *selections, operation='copy'):
        mappings.append(ParameterMapping(target, shape, selections, operation, groups))
    add('embedding.word_embeddings.weight', (v, h), part('model.embed_tokens.weight', 0, v))
    if not arch.tied_embeddings:
        add('output_layer.weight', (v, h), part('lm_head.weight', 0, v))
    add('decoder.final_layernorm.weight', (h,), full('model.norm.weight'))
    for layer in range(arch.layers):
        hf, mg = f'model.layers.{layer}.', f'decoder.layers.{layer}.'
        for before, after in (('input_layernorm', 'input_layernorm'),
                               ('post_attention_layernorm', 'pre_mlp_layernorm')):
            add(mg + after + '.weight', (h,), full(hf + before + '.weight'))
        for suffix, shape in (('weight', (q + 2 * kv, h)), ('bias', (q + 2 * kv,))):
            selections = [part(hf + f'self_attn.{kind}_proj.{suffix}', 0, rows)
                          for kind, rows in (('q', q), ('k', kv), ('v', kv))]
            add(mg + 'self_attention.linear_qkv.' + suffix, shape, *selections, operation='qkv')
        add(mg + 'self_attention.linear_proj.weight', (h, q), part(hf + 'self_attn.o_proj.weight', 1, q))
        add(mg + 'mlp.linear_fc1.weight', (2 * f, h),
            part(hf + 'mlp.gate_proj.weight', 0, f), part(hf + 'mlp.up_proj.weight', 0, f), operation='concat')
        add(mg + 'mlp.linear_fc2.weight', (h, f), part(hf + 'mlp.down_proj.weight', 1, f))
    return mappings


def validate_source(arch, source):
    expected = arch.hf_shapes()
    if arch.tied_embeddings and 'lm_head.weight' in source.shapes:
        expected['lm_head.weight'] = (arch.vocab, arch.hidden)
    if source.shapes != expected:
        differences = sorted(name for name in set(expected) | set(source.shapes)
                             if expected.get(name) != source.shapes.get(name))
        raise ValueError(f'checkpoint parameter coverage/shape differs: {differences}')
    if any(dtype not in (torch.float32, torch.bfloat16) for dtype in source.dtypes.values()):
        raise ValueError('checkpoint tensors must be FP32 or BF16')
    if len(set(source.dtypes.values())) != 1:
        raise ValueError('mixed checkpoint tensor dtypes require a separate initialization contract')
    if arch.tied_embeddings and 'lm_head.weight' in source.shapes:
        for start in range(0, arch.vocab, 256):
            end = min(start + 256, arch.vocab)
            embedding = source.read('model.embed_tokens.weight', 0, start, end)
            output = source.read('lm_head.weight', 0, start, end)
            if not torch.equal(embedding.contiguous().view(torch.uint8), output.contiguous().view(torch.uint8)):
                raise ValueError('tied embedding and lm_head weights differ')


@torch.no_grad()
def load_weights(model, source, arch, tp_size=1, tp_rank=0):
    """Preflight every key/shape before modifying raw native GPT parameters."""
    if getattr(model, 'qwen_architecture', None) != arch:
        raise ValueError('use the validated native Qwen builder for this architecture')
    if (model.qwen_tp_size, model.qwen_tp_rank) != (tp_size, tp_rank):
        raise ValueError('loader TP coordinates differ from the native model process')
    validate_source(arch, source)
    mapping = parameter_mappings(arch, tp_size, tp_rank)
    parameters = dict(model.named_parameters())
    expected = {entry.target: entry.shape for entry in mapping}
    actual = {name: tuple(value.shape) for name, value in parameters.items()}
    if actual != expected:
        raise ValueError('native model parameters do not match the Qwen/TP layout')
    if any(p.dtype not in (torch.float32, torch.bfloat16) for p in parameters.values()):
        raise ValueError('native model must use FP32 or BF16 parameters')
    for entry in mapping:
        value = entry.materialize(source)
        parameters[entry.target].copy_(value)
        del value
    return {'initialization': 'weights-only warm start', 'tp_size': tp_size, 'tp_rank': tp_rank,
            'loaded_parameters': len(mapping), 'local_parameter_elements': sum(p.numel() for p in parameters.values()),
            'unique_model_elements': arch.unique_parameters(), 'optimizer_state_restored': False,
            'GPU_conversion_validated': False}
