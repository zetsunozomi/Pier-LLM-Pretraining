"""Checkpoint I/O through the DDP/BF16 wrapper stack used by pretrain_gpt."""

from collections import OrderedDict
import io
from types import SimpleNamespace
import unittest

import torch

# Match the entrypoint's import order; the legacy package imports training too.
import megatron.training
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.outer_sync.runtime import digest
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.legacy.model.module import Float16Module


def wrapped_model():
    model = torch.nn.Linear(3, 2)
    model.register_buffer('running_value', torch.tensor([1.25, -2.5]))
    model.register_buffer('count', torch.tensor(7))
    model.register_buffer('scratch', torch.tensor(0.), persistent=False)
    config = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1,
                               bf16=True, params_dtype=torch.bfloat16)
    # Production DDP inherits these exact state_dict/load_state_dict methods.
    # Its CUDA gradient buffers are unnecessary for this CPU serialization check.
    return _BaseDataParallel(config, Float16Module(model, SimpleNamespace(fp16=False, bf16=True)))


class CheckpointWrapperTests(unittest.TestCase):
    def test_ddp_bf16_checkpoint_roundtrip(self):
        model = wrapped_model()
        state = model.state_dict()
        self.assertEqual(set(state), {'weight', 'bias', 'running_value', 'count'})
        self.assertEqual(state['weight'].dtype, torch.bfloat16)
        self.assertEqual(state['running_value'].dtype, torch.bfloat16)
        self.assertEqual(state['count'].dtype, torch.int64)
        expected = digest(state)
        archive = io.BytesIO()
        torch.save(state, archive)
        with torch.no_grad():
            for value in model.parameters():
                value.zero_()
            for value in model.buffers():
                value.zero_()
        self.assertNotEqual(digest(model.state_dict()), expected)
        archive.seek(0)
        model.load_state_dict(torch.load(archive, map_location='cpu', weights_only=True), strict=True)
        self.assertEqual(digest(model.state_dict()), expected)

    def test_destination_prefix_keep_vars_and_parent_traversal(self):
        model = wrapped_model()
        destination = OrderedDict(existing=torch.tensor(1.))
        result = model.state_dict(destination=destination, prefix='learner.', keep_vars=True)
        self.assertIs(result, destination)
        self.assertIn('existing', result)
        self.assertIs(result['learner.weight'], model.module.module.weight)
        self.assertIn('learner.running_value', result)
        # torch.nn.Module also passes destination when traversing child modules.
        parent = torch.nn.ModuleDict({'learner': model})
        self.assertEqual(set(parent.state_dict()), {
            'learner.weight', 'learner.bias', 'learner.running_value', 'learner.count'})
        # Preserve the legacy wrapper's existing positional prefix/keep_vars API.
        legacy = model.module.state_dict('legacy.', True)
        self.assertIs(legacy['legacy.weight'], model.module.module.weight)


if __name__ == '__main__':
    unittest.main()
