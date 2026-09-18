"""Full-coordinate file oracle tests: real Gloo, restart, corruption and >1M N."""

import copy
from datetime import timedelta
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.optimizer.optimizer import FP32Optimizer, MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.outer_sync.checkpoint import load, restore_rng, save
from megatron.core.outer_sync.coordinates import ParameterCoordinates
from megatron.core.outer_sync.runtime import CenteredRuntime, digest
from megatron.core.outer_sync.streamed_oracle import FileVector
from test_runtime import advance, fixture


class VectorModel(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-.5, .5, n))

    def forward(self):
        return self.weight[0] + self.weight[-1]


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, world_size=4, rank=rank,
                           timeout=timedelta(seconds=180))
    try:
        directory = Path(directory)
        expected = None
        for cohort in (1, 2, 4):
            memory, model, optimizer, scheduler = fixture(rank, cohort, directory / 'memory')
            advance(memory, model, optimizer, scheduler, 11)
            expected_events = copy.deepcopy(memory.events)
            expected_r, expected_m = memory.oracle_reference.copy(), memory.oracle_momentum.copy()
            runtime, model, optimizer, scheduler = fixture(
                rank, cohort, directory / 'streamed', oracle_storage='streamed', trace_dir=directory / f's{cohort}')
            with patch.object(ParameterCoordinates, 'cpu_flat', side_effect=AssertionError('full flat copy forbidden')):
                advance(runtime, model, optimizer, scheduler, 11)
            assert runtime.events == expected_events
            oracle = runtime.streamed_oracle
            assert oracle.checkpoint_receipt()['reference_sha256'] == hashlib.sha256(expected_r.tobytes()).hexdigest()
            assert oracle.checkpoint_receipt()['momentum_sha256'] == hashlib.sha256(expected_m.tobytes()).hexdigest()
            assert oracle.last_master_coordinates == runtime.coordinates.numel
            assert oracle.boundaries_checked == 3
            assert oracle.allocation_bytes()['persistent_device_scratch_bytes'] == 5 * 7 * 4
            if cohort == 2:
                expected = digest([optimizer.state_dict(), model.state_dict(), scheduler.state_dict(),
                                   runtime.executor.reference, runtime.executor.momentum, runtime.clock.state_dict()])
                expected_events_resume = copy.deepcopy(runtime.events)
            oracle.close()

        path = directory / 'checkpoint'
        runtime, model, optimizer, scheduler = fixture(
            rank, 2, path, oracle_storage='streamed', trace_dir=directory / 'split')
        with patch.object(ParameterCoordinates, 'cpu_flat', side_effect=AssertionError('full flat copy forbidden')):
            advance(runtime, model, optimizer, scheduler, 5)
            save(runtime, 5, scheduler, 123.)
        state = torch.load(path / 'iter_0000005' / f'rank-{rank}.pt', weights_only=False)
        assert state['oracle_reference'] is None and state['oracle_momentum'] is None
        assert state['streamed_oracle']['elements'] == runtime.coordinates.numel
        receipt = state['streamed_oracle']
        runtime.streamed_oracle.close()
        runtime, model, optimizer, scheduler = fixture(
            rank, 2, path, oracle_storage='streamed', trace_dir=directory / 'resume')
        with patch.object(ParameterCoordinates, 'cpu_flat', side_effect=AssertionError('full flat copy forbidden')), \
             patch('megatron.core.optimizer.optimizer.parallel_state.get_pipeline_model_parallel_world_size', return_value=1):
            iteration, flops = load(runtime, str(path), scheduler)
        assert iteration == 5 and flops == 123.
        assert runtime.streamed_oracle.checkpoint_receipt() == receipt
        assert runtime.streamed_oracle.boundaries_checked == 1
        # R replicas other than the reconstruction source must also be checked.
        original = runtime.executor.reference.view(-1)[0].clone()
        if rank == 2:
            runtime.executor.reference.view(-1)[0].add_(1.)
        caught = False
        try:
            runtime.streamed_oracle.restore_from_owners(receipt)
        except AssertionError as exc:
            assert 'restored owned R' in str(exc)
            caught = True
        outcomes = [None] * 4
        dist.all_gather_object(outcomes, caught)
        assert outcomes == [False, False, True, False]
        runtime.executor.reference.view(-1)[0].copy_(original)
        runtime.streamed_oracle.restore_from_owners(receipt)
        bad_receipt = dict(receipt, momentum_sha256='0' * 64)
        try:
            runtime.streamed_oracle.restore_from_owners(bad_receipt)
        except AssertionError as exc:
            assert 'momentum_sha256' in str(exc)
        else:
            raise AssertionError('changed oracle digest accepted')
        runtime.streamed_oracle.restore_from_owners(receipt)
        restore_rng(runtime.pending_rng)
        runtime.pending_rng = None
        with patch.object(ParameterCoordinates, 'cpu_flat', side_effect=AssertionError('full flat copy forbidden')):
            advance(runtime, model, optimizer, scheduler, 11)
        assert runtime.events == expected_events_resume
        assert digest([optimizer.state_dict(), model.state_dict(), scheduler.state_dict(),
                       runtime.executor.reference, runtime.executor.momentum, runtime.clock.state_dict()]) == expected
        runtime.streamed_oracle.close()

        # More than the old cap, with no full-flat accessor and a short final tile.
        n = 1_000_017
        large = VectorModel(n)
        inner = torch.optim.SGD(large.parameters(), lr=.01)
        optimizer = object.__new__(FP32Optimizer)
        MegatronOptimizer.__init__(optimizer, inner, OptimizerConfig(clip_grad=0.), None)
        optimizer.is_stub_optimizer = False
        optimizer.grad_stats_parallel_group = dist.group.WORLD
        args = copy.copy(runtime.args)
        args.outer_sync_interval = 1
        args.outer_tile_elements = 16384
        args.outer_verify_tile_elements = 65536
        args.outer_trace_dir = str(directory / 'large')
        args.outer_verify_storage = 'memory'
        try:
            CenteredRuntime(args, [large], optimizer, group=dist.group.WORLD)
        except ValueError as exc:
            assert '1M' in str(exc)
        else:
            raise AssertionError('old memory-mode cap was bypassed')
        args.outer_verify_storage = 'streamed'
        with patch.object(ParameterCoordinates, 'cpu_flat', side_effect=AssertionError('full flat copy forbidden')):
            large_runtime = CenteredRuntime(args, [large], optimizer, group=dist.group.WORLD)
            with torch.no_grad():
                large.weight.add_(rank / 128.)
            large_runtime.after_attempt(True, 1, {'loss': torch.tensor(0.)})
        oracle = large_runtime.streamed_oracle
        assert oracle.last_master_coordinates == n
        assert oracle.allocation_bytes()['persistent_device_scratch_bytes'] == 5 * 65536 * 4
        large()  # The real module pre-hook checks the next consumer, including the tail.
        assert large_runtime.consumer_checks == 1 and not large_runtime.pending_consumer
        with torch.no_grad():
            large.weight[-1].add_(1.)
        try:
            oracle.check()
        except AssertionError as exc:
            assert 'master vs streamed oracle' in str(exc)
        else:
            raise AssertionError('corrupted tail beyond 1M coordinates was not detected')
        oracle.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


class StreamedOracleTests(unittest.TestCase):
    def test_file_vector_bounds_sign_bits_and_truncation(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'state.f32'
            vector = FileVector(path, 7)
            values = np.array([-0., 1., -2.], np.float32)
            vector.write(2, values)
            self.assertTrue(np.array_equal(vector.read(2, 3).view(np.uint32), values.view(np.uint32)))
            self.assertEqual(vector.read(0, 2).tolist(), [0., 0.])
            with self.assertRaises(FileExistsError):
                FileVector(path, 7)
            with self.assertRaises(ValueError):
                vector.read(6, 2)
            with self.assertRaises(ValueError):
                vector.write(0, np.zeros(1, np.float64))
            os.ftruncate(vector.fd, 16)
            with self.assertRaises(IOError):
                vector.read(2, 3)
            vector.close()

    def test_real_runtime_streamed_trajectory_resume_and_large_tail(self):
        with tempfile.TemporaryDirectory(prefix='pier-streamed-oracle-') as name:
            mp.spawn(worker, args=(f'file://{name}/rendezvous', name), nprocs=4, join=True)

    def test_chunked_bf16_model_commit_checks_the_final_coordinate(self):
        master = torch.linspace(-.5, .5, 1_000_017)
        model = master.bfloat16()
        coordinates = ParameterCoordinates([('large', master, model)])
        coordinates.assert_model_committed()
        model[-1] = -4.
        with self.assertRaisesRegex(AssertionError, 'large at'):
            coordinates.assert_model_committed()


if __name__ == '__main__':
    unittest.main()
