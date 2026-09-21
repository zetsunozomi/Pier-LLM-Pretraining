"""Four-process production adapter/clock/checkpoint regression on real CPU Gloo."""

import copy
from datetime import timedelta
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.optimizer.optimizer import FP32Optimizer, MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.outer_sync.checkpoint import load, restore_rng, save
from megatron.core.outer_sync.runtime import CenteredRuntime, StepClock, digest


def fixture(rank, cohort, directory, arm=None, oracle_storage='memory', trace_dir=None):
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 2))
    inner = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False)
    # FP32Optimizer's constructor only adds a CUDA unity-scale tensor; the real
    # prepare/step/commit/state methods below are used unchanged on CPU.
    optimizer = object.__new__(FP32Optimizer)
    MegatronOptimizer.__init__(optimizer, inner, OptimizerConfig(clip_grad=0.), None)
    optimizer.is_stub_optimizer = False
    optimizer.grad_stats_parallel_group = dist.group.WORLD
    scheduler = torch.optim.lr_scheduler.StepLR(inner, step_size=3, gamma=.9)
    args = SimpleNamespace(outer_cohort_size=cohort, outer_sync_interval=3,
                           outer_tile_elements=2, outer_cpu_offload=False,
                           outer_verify=arm is None, outer_trace_dir=str(trace_dir) if trace_dir else None,
                           outer_arm=arm, outer_verify_storage=oracle_storage, outer_verify_tile_elements=7,
                           outer_momentum=.9, outer_learning_rate=.7,
                           outer_inject_skip_at=[3], outer_inject_skip_rank=1,
                           group_query_attention=False, num_attention_heads=4, num_query_groups=1,
                           qwen_recipe={'snapshot_sha256': 'test-snapshot', 'data_sha256': 'test-data'},
                           train_iters=11, save=str(directory), consumed_train_samples=0,
                           consumed_valid_samples=0, skipped_train_samples=0)
    runtime = CenteredRuntime(args, [model], optimizer, group=dist.group.WORLD)
    torch.manual_seed(100 + rank)
    random.seed(200 + rank)
    np.random.seed(300 + rank)
    return runtime, model, optimizer, scheduler


def advance(runtime, model, optimizer, scheduler, end):
    for attempt in range(runtime.clock.attempted + 1, end + 1):
        runtime.before_attempt()
        optimizer.optimizer.zero_grad(set_to_none=True)
        x = torch.randn(2, 3) + random.random() + float(np.random.random())
        loss = model(x).square().mean()
        loss.backward()
        for parameter in model.parameters():
            parameter.main_grad = parameter.grad
        success, _, _ = optimizer.step()
        if success:
            scheduler.step()
        if runtime.meter is not None:
            # Synthetic measurement fixture: eight global loss tokens per attempt.
            runtime.meter.record_tokens(torch.tensor(8.))
        runtime.after_attempt(success, attempt, {'loss': loss.detach()})
        runtime.args.consumed_train_samples += 8


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=60))
    try:
        baseline = None
        for cohort in (1, 2, 4):
            runtime, model, optimizer, scheduler = fixture(rank, cohort, Path(directory) / f's{cohort}')
            advance(runtime, model, optimizer, scheduler, 11)
            assert runtime.clock.state_dict() == dict(interval=3, attempted=11, successful=10, boundaries=3)
            assert runtime.consumer_checks == 3 and not runtime.pending_consumer
            comparable = [{key: value for key, value in row.items() if key != 'payload'} for row in runtime.events]
            if baseline is None:
                baseline = comparable
            else:
                assert comparable == baseline, f'cohort {cohort} changes the training trajectory'
            if cohort == 2:
                expected = dict(events=copy.deepcopy(runtime.events), optimizer=digest(optimizer.state_dict()),
                                scheduler=digest(scheduler.state_dict()), model=digest(model.state_dict()),
                                r=digest(runtime.executor.reference), m=digest(runtime.executor.momentum))
        runtime, model, optimizer, scheduler = fixture(rank, 2, Path(directory) / 'resume')
        advance(runtime, model, optimizer, scheduler, 5)
        assert runtime.clock.successful == 4  # Checkpoint inside an outer cycle.
        # Match the CLI-versus-reporting values seen in the real GPU split run.
        runtime.args.num_query_groups = runtime.args.num_attention_heads
        save(runtime, 5, scheduler, 123.)
        runtime, model, optimizer, scheduler = fixture(rank, 2, Path(directory) / 'resume')
        runtime.args.group_query_attention = True
        runtime.args.num_query_groups = 2
        try:
            load(runtime, str(Path(directory) / 'resume'), scheduler)
        except ValueError as error:
            assert 'num_query_groups' in str(error)
        else:
            raise AssertionError('a changed GQA architecture must not restore')
        runtime.args.group_query_attention = False
        runtime.args.num_query_groups = 1
        runtime.args.qwen_recipe['data_sha256'] = 'different-test-data'
        try:
            load(runtime, str(Path(directory) / 'resume'), scheduler)
        except ValueError as error:
            assert 'qwen_recipe' in str(error)
        else:
            raise AssertionError('changed Qwen data must not restore the old training state')
        runtime.args.qwen_recipe['data_sha256'] = 'test-data'
        runtime.args.recompute_granularity = 'full'
        runtime.args.recompute_method = 'uniform'
        runtime.args.recompute_num_layers = 1
        try:
            load(runtime, str(Path(directory) / 'resume'), scheduler)
        except ValueError as error:
            assert 'recompute_granularity' in str(error)
        else:
            raise AssertionError('changed activation recompute schedule must not restore silently')
        runtime.args.recompute_granularity = None
        runtime.args.recompute_method = None
        runtime.args.recompute_num_layers = None
        runtime.args.distribute_saved_activations = False
        # This getter has no effect on tensor copies; initialize only its scalar
        # return for Megatron's legacy-compatible optimizer loading method.
        with patch('megatron.core.optimizer.optimizer.parallel_state.get_pipeline_model_parallel_world_size', return_value=1):
            iteration, flops = load(runtime, str(Path(directory) / 'resume'), scheduler)
        assert iteration == 5 and flops == 123. and runtime.args.consumed_train_samples == 40
        restore_rng(runtime.pending_rng)
        runtime.pending_rng = None
        advance(runtime, model, optimizer, scheduler, 11)
        assert runtime.events == expected['events']
        assert digest(optimizer.state_dict()) == expected['optimizer']
        assert digest(scheduler.state_dict()) == expected['scheduler']
        assert digest(model.state_dict()) == expected['model']
        assert digest(runtime.executor.reference) == expected['r']
        assert digest(runtime.executor.momentum) == expected['m']
        # A real nonfinite gradient on only rank 2 must stop all learners before
        # any model or inner moments change, without relying on the test vote.
        runtime, model, optimizer, scheduler = fixture(rank, 1, Path(directory) / 'nonfinite')
        runtime.args.outer_inject_skip_at = []
        before = digest(optimizer.state_dict())
        for p in model.parameters():
            p.main_grad = torch.ones_like(p)
        if rank == 2:
            next(model.parameters()).main_grad.view(-1)[0] = float('inf')
        success, _, _ = optimizer.step()
        assert not success and digest(optimizer.state_dict()) == before
        runtime.after_attempt(False, 1, {})
        assert runtime.clock.successful == 0 and runtime.clock.boundaries == 0
        dist.barrier()
    finally:
        dist.destroy_process_group()


def native_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        for arm in ('gather', 'resident', 'recenter'):
            path = Path(directory) / arm
            runtime, model, optimizer, scheduler = fixture(rank, 1, path, arm=arm)
            advance(runtime, model, optimizer, scheduler, 11)
            runtime.coordinates.assert_model_committed()
            expected = digest([optimizer.state_dict(), model.state_dict(), scheduler.state_dict(),
                               runtime.executor.reference, runtime.executor.momentum, runtime.clock.state_dict()])
            runtime, model, optimizer, scheduler = fixture(rank, 1, path, arm=arm)
            advance(runtime, model, optimizer, scheduler, 5)
            save(runtime, 5, scheduler, 123.)
            runtime, model, optimizer, scheduler = fixture(rank, 1, path, arm=arm)
            original = runtime.args.outer_arm
            runtime.args.outer_arm = 'recenter' if arm != 'recenter' else 'gather'
            try:
                load(runtime, str(path), scheduler)
            except ValueError as exc:
                assert 'outer_arm' in str(exc)
            else:
                raise AssertionError('cross-arm checkpoint restore accepted')
            runtime.args.outer_arm = original
            with patch('megatron.core.optimizer.optimizer.parallel_state.get_pipeline_model_parallel_world_size', return_value=1):
                load(runtime, str(path), scheduler)
            restore_rng(runtime.pending_rng)
            runtime.pending_rng = None
            advance(runtime, model, optimizer, scheduler, 11)
            runtime.coordinates.assert_model_committed()
            assert digest([optimizer.state_dict(), model.state_dict(), scheduler.state_dict(),
                           runtime.executor.reference, runtime.executor.momentum, runtime.clock.state_dict()]) == expected
            assert runtime.clock.state_dict() == dict(interval=3, attempted=11, successful=10, boundaries=3)
            from megatron.core.outer_sync.cycle_metrics import CycleMeter
            runtime, model, optimizer, scheduler = fixture(rank, 1, path, arm=arm)
            runtime.meter = CycleMeter('cpu', path / 'cycles', warmup_cycles=1,
                                      metadata={'fixture': 'CPU test only', 'arm': arm})
            advance(runtime, model, optimizer, scheduler, 11)
            runtime.finish(11)
            assert runtime.meter.complete_count == 3
            assert [r['eligible'] for r in runtime.meter.records] == [False, True, True, False]
            assert runtime.meter.records[0]['processed_loss_tokens_global'] == 32
            assert runtime.meter.records[0]['successful_loss_tokens_global'] == 24
            assert runtime.meter.records[0]['skipped_loss_tokens_global'] == 8
            assert runtime.meter.metadata['final_health']['model_matches_master']
            assert runtime.meter.metadata['final_health']['finite_model']
            # Matching nonfinite master/model values still invalidate a measured run.
            _, master, target = runtime.coordinates.pairs[0]
            with torch.no_grad():
                master.view(-1)[0] = float('inf')
                target.view(-1)[0] = float('inf')
            try:
                runtime.coordinates.assert_model_committed(finite=True)
            except AssertionError as exc:
                assert 'nonfinite' in str(exc)
            else:
                raise AssertionError('nonfinite committed model accepted')
        dist.barrier()
    finally:
        dist.destroy_process_group()


class RuntimeTests(unittest.TestCase):
    def test_success_clock(self):
        clock = StepClock(2)
        self.assertFalse(clock.advance(True))
        self.assertFalse(clock.advance(False))
        self.assertTrue(clock.advance(True))
        with self.assertRaises(ValueError):
            clock.load_state_dict(dict(interval=2, attempted=3, successful=2, boundaries=0))

    def test_production_coordinates_skip_and_exact_midcycle_resume(self):
        with tempfile.TemporaryDirectory(prefix='pier-runtime-') as name:
            mp.spawn(worker, args=(f'file://{name}/rendezvous', name), nprocs=4, join=True)

    def test_native_arms_use_same_optimizer_skip_and_checkpoint_path(self):
        with tempfile.TemporaryDirectory(prefix='pier-native-runtime-') as name:
            mp.spawn(native_worker, args=(f'file://{name}/rendezvous', name), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
