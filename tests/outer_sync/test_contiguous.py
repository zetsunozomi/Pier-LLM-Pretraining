"""Real Gloo checks for direct tree slices; no CUDA performance claim."""

from collections import Counter
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from megatron.core.outer_sync.checkpoint import load, restore_rng, save
from megatron.core.outer_sync.coordinates import ParameterCoordinates
from megatron.core.outer_sync.executor import CenteredExecutor
from megatron.core.outer_sync.runtime import digest
from test_runtime import advance, fixture


class OperationCounter(TorchDispatchMode):
    def __init__(self, engine):
        super().__init__()
        self.counts = Counter()
        self.storages = {t.untyped_storage().data_ptr() for t in engine.storage_tensors()}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        self.counts[str(func)] += 1
        for value in tree_leaves(result):
            if isinstance(value, torch.Tensor) and value.numel():
                assert value.untyped_storage().data_ptr() in self.storages
        return result


def check_coordinates(group):
    k, rank = dist.get_world_size(group), dist.get_rank(group)
    for s in (1 << i for i in range(k.bit_length())):
        for n, tile in ((1, 1), (5, 1), (29, 3), (64, 7)):
            width, g = (n + k - 1) // k, k // s
            engines = []
            for schedule in ('reference', 'contiguous'):
                pairs = [(str(i).zfill(4), torch.tensor([.25 + i / 7]),
                          torch.empty(1, dtype=torch.bfloat16)) for i in range(n)]
                pairs.append(('empty', torch.empty(0), torch.empty(0, dtype=torch.bfloat16)))
                coordinates = ParameterCoordinates(pairs)
                engine = CenteredExecutor(None, torch.zeros(g, width), torch.zeros(width),
                                          cohort=s, tile_elements=tile, coordinates=coordinates,
                                          group=group, schedule=schedule)
                engines.append(engine)
            for turn in range(3):
                counts, payloads = [], []
                for engine in engines:
                    with torch.no_grad():
                        for i, (_, master, _) in enumerate(engine.coordinates.pairs):
                            master.add_((rank + 1) * .00013 * (turn + 1))
                            if master.numel() and i == 0 and turn == 2:
                                master.fill_((2**24, 1., -2**24, 1.)[rank % 4])
                    with OperationCounter(engine) as counter:
                        payloads.append(engine.step())
                    counts.append(counter.counts)
                assert payloads[0] == payloads[1]
                assert digest(engines[0].reference) == digest(engines[1].reference)
                assert digest(engines[0].momentum) == digest(engines[1].momentum)
                assert digest(engines[0].coordinates.cpu_flat()) == digest(engines[1].coordinates.cpu_flat())
                assert engines[0].allocation_bytes()['workspace_tensor_bytes'] == engines[1].allocation_bytes()['workspace_tensor_bytes']
                if g >= 4:
                    assert counts[1]['aten.add.out'] < counts[0]['aten.add.out']
                    assert counts[1]['aten.copy_.default'] < counts[0]['aten.copy_.default']


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        check_coordinates(dist.group.WORLD)
        groups = [dist.new_group(peers) for peers in ([0, 2], [1, 3])]
        check_coordinates(groups[rank % 2])
        for s in (1, 2, 4):
            expected = None
            for schedule in ('reference', 'contiguous'):
                runtime, model, optimizer, scheduler = fixture(
                    rank, s, Path(directory) / f'{schedule}-{s}', schedule=schedule)
                advance(runtime, model, optimizer, scheduler, 11)
                actual = digest([runtime.events, model.state_dict(), optimizer.state_dict(),
                                 scheduler.state_dict(), runtime.executor.reference, runtime.executor.momentum])
                if expected is None:
                    expected = actual
                else:
                    assert actual == expected
            path = Path(directory) / f'restart-{s}'
            runtime, model, optimizer, scheduler = fixture(rank, s, path, schedule='contiguous')
            advance(runtime, model, optimizer, scheduler, 5)
            save(runtime, 5, scheduler, 0.)
            runtime, model, optimizer, scheduler = fixture(rank, s, path, schedule='contiguous')
            with patch('megatron.core.optimizer.optimizer.parallel_state.get_pipeline_model_parallel_world_size', return_value=1):
                load(runtime, str(path), scheduler)
            restore_rng(runtime.pending_rng)
            runtime.pending_rng = None
            advance(runtime, model, optimizer, scheduler, 11)
            assert digest([runtime.events, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                           runtime.executor.reference, runtime.executor.momentum]) == expected
        dist.barrier()
    finally:
        dist.destroy_process_group()


class ContiguousTests(unittest.TestCase):
    def test_coordinates_noncontiguous_groups_training_and_restart(self):
        with tempfile.TemporaryDirectory(prefix='pier-contiguous-') as directory:
            mp.spawn(worker, args=(f'file://{directory}/rendezvous', directory), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
