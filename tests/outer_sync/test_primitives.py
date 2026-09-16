"""CPU tests for shared code; these do not certify Megatron's CUDA runtime."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from megatron.core.optimizer.optimizer import (
    ChainedOptimizer, Float16OptimizerWithFloat16Params, FP32Optimizer, MegatronOptimizer,
)
from megatron.core.outer_sync.legacy import centered_allreduce_update
from megatron.core.outer_sync.normalization import inner_gradient_scale, warmup_rescale


def cpu_copy_fixture():
    # Only the storage/copy methods are exercised on CPU; the CUDA constructor
    # and optimizer step are exercised by megatron_gate.py on the cluster.
    opt = object.__new__(Float16OptimizerWithFloat16Params)
    opt.config = SimpleNamespace(use_precision_aware_optimizer=False)
    opt.is_stub_optimizer = False
    opt._dummy_overflow_buf = None
    model = torch.tensor([1., -2., 3.], dtype=torch.bfloat16)
    master = model.float().clone()
    opt.float16_groups = [[model]]
    opt.fp32_from_float16_groups = [[master]]
    opt.fp32_from_fp32_groups = [[]]
    opt.optimizer = torch.optim.AdamW([master], lr=.01)
    return opt, master, model


class PrimitivesTest(unittest.TestCase):
    def test_normalization_and_warmup(self):
        for full in (1, 4, 8, 32):
            for inner in (1, 2, 4, 8, 32):
                if inner > full or full % inner:
                    continue
                for normalized in (False, True):
                    for avg in (False, True):
                        # Sum of each rank's equal gradient = inner * 3.
                        collective = 3. if avg else inner * 3.
                        scale = inner_gradient_scale(full, inner, inner_average=normalized,
                                                     collective_average=avg)
                        local = collective * scale
                        self.assertEqual(local, 3. if normalized else 3. * inner / full)
                        self.assertEqual(local * warmup_rescale(
                            full, inner, inner_average=normalized), 3.)
        for full, inner in ((0, 1), (4, 0), (4, 3), (2, 4)):
            with self.assertRaises(ValueError):
                inner_gradient_scale(full, inner, inner_average=True, collective_average=False)

    def test_commit_copies_without_changing_master(self):
        opt, master, model = cpu_copy_fixture()
        master.add_(.25)
        self.assertFalse(torch.equal(model, master.bfloat16()))
        expected = master.clone()
        opt.commit_outer_update()
        self.assertTrue(torch.equal(model, expected.bfloat16()))
        self.assertTrue(torch.equal(master, expected))

    def test_shared_update_commits_and_keeps_inner_state(self):
        opt, master, model = cpu_copy_fixture()
        reference = master.clone()
        master.add_(.5)
        moments = [torch.full_like(master, .25)]
        inner_state = {'step': torch.tensor(7.), 'exp_avg': torch.ones_like(master)}
        opt.optimizer.state[master] = inner_state
        before = {key: value.clone() for key, value in inner_state.items()}
        with patch('torch.distributed.all_reduce') as reduce:
            centered_allreduce_update(
                opt, moments, reference_for_param=lambda p: reference,
                load_state=lambda m, p: m, store_state=lambda m, p: m,
                group=None, momentum=.5, learning_rate=.25,
            )
        reduce.assert_called_once()
        expected_m = torch.full_like(master, -.375)
        expected_r = reference - .25 * (-.5 + .5 * expected_m)
        self.assertTrue(torch.equal(moments[0], expected_m))
        self.assertTrue(torch.equal(master, expected_r))
        self.assertTrue(torch.equal(model, expected_r.bfloat16()))
        for key in before:
            self.assertTrue(torch.equal(inner_state[key], before[key]))

    def test_reject_before_mutating(self):
        opt, master, model = cpu_copy_fixture()
        before = master.clone()
        with self.assertRaises(ValueError):
            centered_allreduce_update(opt, [], reference_for_param=None, load_state=None,
                                     store_state=None, group=None, momentum=.5, learning_rate=1.)
        self.assertTrue(torch.equal(master, before))
        opt.config.use_precision_aware_optimizer = True
        with self.assertRaises(NotImplementedError):
            opt.commit_outer_update()

    def test_chain_and_fp32_alias(self):
        opt, master, model = cpu_copy_fixture()
        fp32 = object.__new__(FP32Optimizer)
        fp32.config = SimpleNamespace(use_precision_aware_optimizer=False)
        chain = object.__new__(ChainedOptimizer)
        chain.chained_optimizers = [opt, fp32]
        master.add_(.5)
        chain.commit_outer_update()
        self.assertTrue(torch.equal(model, master.bfloat16()))

    def test_unimplemented_adapter_fails_closed(self):
        # DistributedOptimizer inherits this method until its coordinate/gather
        # adapter is implemented. Never silently treat a shard as a full model.
        with self.assertRaises(NotImplementedError):
            MegatronOptimizer.validate_outer_update_support(SimpleNamespace())

    def test_chain_validates_all_children_before_copying(self):
        opt, master, model = cpu_copy_fixture()
        unsupported = object.__new__(FP32Optimizer)
        unsupported.config = SimpleNamespace(use_precision_aware_optimizer=True)
        chain = object.__new__(ChainedOptimizer)
        chain.chained_optimizers = [opt, unsupported]
        before = model.clone()
        master.add_(.5)
        with self.assertRaises(NotImplementedError):
            chain.commit_outer_update()
        self.assertTrue(torch.equal(model, before))


if __name__ == '__main__':
    unittest.main()
