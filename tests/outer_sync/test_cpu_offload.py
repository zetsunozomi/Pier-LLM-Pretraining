"""Full-state CPU offload against the independent NumPy update oracle."""

from datetime import timedelta
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.outer_sync.coordinates import ParameterCoordinates
from megatron.core.outer_sync.cpu_offload import NaiveCPUOffloadExecutor
from megatron.core.outer_sync.spec import resident_reference


def run_case(group):
    rank, k = dist.get_rank(group), dist.get_world_size(group)
    expected_r = np.arange(23, dtype=np.float32) / 16
    expected_m = np.arange(23, dtype=np.float32) / 128
    flat = torch.from_numpy(expected_r.copy())
    model = flat.bfloat16()
    coordinates = ParameterCoordinates([
        ('c', flat[7:], model[7:]), ('a', flat[:7], model[:7]),
        ('b-empty', flat[:0], model[:0])])
    reference = torch.from_numpy(expected_r.copy())
    momentum = torch.from_numpy(expected_m.copy())
    executor = NaiveCPUOffloadExecutor(reference, momentum, coordinates=coordinates, group=group)
    for step in range(3):
        weights = np.stack([expected_r + np.float32((r + 1) * (step + 1) / 32)
                            for r in range(k)])
        flat.copy_(torch.from_numpy(weights[rank]))
        before_model = model.clone()
        expected = resident_reference(expected_r, expected_m, weights, .5, .25)
        expected_r, expected_m = expected['reference'], expected['momentum']
        payload = executor.step(mu=.5, eta=.25)
        # Dyadic values make the intermediate sums exact. This does not claim
        # general bitwise equality between CPU/GPU or native SUM/tree execution.
        assert torch.equal(flat, torch.from_numpy(expected_r))
        assert torch.equal(reference, torch.from_numpy(expected_r))
        assert torch.equal(momentum, torch.from_numpy(expected_m))
        assert torch.equal(model, before_model)  # The shared runtime owns commit.
        model.copy_(flat)
        coordinates.assert_model_committed()
        assert payload['allreduce_calls'] == (2 if k > 1 else 0)
        assert payload['physical_wire_bytes'] is None
    for bad_r, bad_m in ((reference[:7], momentum), (reference, momentum[:7]),
                         (reference, reference), (flat, momentum)):
        try:
            NaiveCPUOffloadExecutor(bad_r, bad_m, coordinates=coordinates, group=group)
        except ValueError:
            pass
        else:
            raise AssertionError('sharded or aliased outer state accepted')


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        run_case(dist.group.WORLD)
        for peers in ((0, 2), (1, 3)):
            group = dist.new_group(list(peers))
            if rank in peers:
                own = group
        run_case(own)
        for peer in range(4):
            group = dist.new_group([peer])
            if rank == peer:
                own = group
        run_case(own)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class CPUOffloadTests(unittest.TestCase):
    def test_nesterov_full_replicas_and_noncontiguous_groups(self):
        with tempfile.TemporaryDirectory(prefix='pier-cpu-offload-') as directory:
            mp.spawn(worker, args=(f'file://{directory}/rendezvous',), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
