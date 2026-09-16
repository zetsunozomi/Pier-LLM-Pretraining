"""Real Gloo regression for default groups and noncontiguous subgroup peers."""

from datetime import timedelta
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REFERENCE = Path(__file__).resolve().parents[2] / 'experiments/centered_outer/reference'
sys.path.insert(0, str(REFERENCE))
from executor import CenteredExecutor


def exercise_group(group, peers):
    k = len(peers)
    group_rank = peers.index(dist.get_rank())
    n = 7  # Also exercise padding and multiple tiles.
    width = (n + k - 1) // k
    for cohort in (1 << level for level in range(k.bit_length())):
        g = k // cohort
        master = torch.arange(n, dtype=torch.float32) + dist.get_rank()
        reference = torch.zeros(g, width)
        momentum = torch.zeros(width)
        model = torch.empty(n, dtype=torch.bfloat16)
        engine = CenteredExecutor(master, reference, momentum, cohort=cohort,
                                  tile_elements=1, model=model, group=group)
        assert engine.rank == group_rank and engine.peers == peers
        # With zero initial state, mu=0 and eta=1, one step is the group mean.
        engine.step(mu=0., eta=1.)
        expected = torch.arange(n, dtype=torch.float32) + sum(peers) / k
        torch.testing.assert_close(master, expected, rtol=0, atol=0)
        torch.testing.assert_close(model, expected.bfloat16(), rtol=0, atol=0)
        padded = torch.zeros(k * width)
        padded[:n] = expected
        a, b = divmod(group_rank, cohort)
        torch.testing.assert_close(reference, padded[b * g * width:(b + 1) * g * width]
                                   .view(g, width), rtol=0, atol=0)
        start = (b * g + a) * width
        torch.testing.assert_close(momentum, -padded[start:start + width], rtol=0, atol=0)


def worker(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=init_method, rank=rank, world_size=4,
                            timeout=timedelta(seconds=30))
    get_ranks = dist.get_process_group_ranks

    def require_explicit_group(group):
        # Reproduce the cluster API's KeyError on newer local PyTorch as well.
        if group is None:
            raise KeyError(None)
        return get_ranks(group)

    try:
        with patch.object(dist, 'get_process_group_ranks', require_explicit_group):
            exercise_group(None, [0, 1, 2, 3])
            exercise_group(dist.group.WORLD, [0, 1, 2, 3])
            groups = [dist.new_group(peers, timeout=timedelta(seconds=30))
                      for peers in ([0, 2], [1, 3])]
            peers = [0, 2] if rank % 2 == 0 else [1, 3]
            exercise_group(groups[rank % 2], peers)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class ProcessGroupTests(unittest.TestCase):
    def test_default_world_and_noncontiguous_subgroups(self):
        with tempfile.TemporaryDirectory(prefix='pier-groups-') as name:
            mp.spawn(worker, args=(f'file://{name}/rendezvous',), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
