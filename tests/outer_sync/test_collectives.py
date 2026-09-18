"""Real CPU/Gloo native-collective arms, padding, state, alias and byte checks."""

from datetime import timedelta
import math
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.outer_sync.collectives import ARMS, CollectiveExecutor, tile_for_budget, workspace_elements_per_tile
from megatron.core.outer_sync.coordinates import ParameterCoordinates
from megatron.core.outer_sync.spec import add_tree, resident_reference, update


def run_case(group, n, tile, *, witness=False, coordinates=False):
    rank, k = dist.get_rank(group), dist.get_world_size(group)
    width = (n + k - 1) // k
    initial_r = np.ones(n, np.float32) if witness else np.arange(n, dtype=np.float32) / 16
    initial_m = np.zeros(n, np.float32) if witness else np.arange(n, dtype=np.float32) / 128
    outputs = {}
    for arm in ARMS:
        expected_r, expected_m = initial_r.copy(), initial_m.copy()
        flat = torch.from_numpy(initial_r.copy())
        model = flat.bfloat16()
        coordinate_map = None
        if coordinates:
            split = n // 2
            coordinate_map = ParameterCoordinates([('a', flat[:split], model[:split]),
                                                   ('b', flat[split:], model[split:])])
        padded_r = torch.from_numpy(np.pad(initial_r, (0, width * k - n)))
        padded_m = torch.from_numpy(np.pad(initial_m, (0, width * k - n)))
        reference = (padded_r if arm == 'resident' else padded_r[rank * width:(rank + 1) * width]).clone()
        momentum = padded_m[rank * width:(rank + 1) * width].clone()
        executor = CollectiveExecutor(None if coordinates else flat, reference, momentum,
                                      arm=arm, tile_elements=tile, group=group,
                                      model=None if coordinates else model, coordinates=coordinate_map)
        addresses = [p.data_ptr() for p in executor.storage_tensors()]
        for step in range(1 if witness else 3):
            if witness:
                weights = np.stack([initial_r + np.float32((r % 2) * 2**-23) for r in range(k)])
                mu, eta = .9, .7
            else:
                weights = np.stack([expected_r + np.float32((r + 1) * (step + 1) / 32) for r in range(k)])
                mu, eta = .5, .25
            flat.copy_(torch.from_numpy(weights[rank]))
            if arm == 'recenter':
                average = np.subtract(expected_r, np.divide(add_tree(weights), np.float32(k), dtype=np.float32), dtype=np.float32)
                expected_r, expected_m = update(expected_r, expected_m, average, mu, eta)
            else:
                expected = resident_reference(expected_r, expected_m, weights, mu, eta)
                expected_r, expected_m = expected['reference'], expected['momentum']
            payload = executor.step(mu=mu, eta=eta)
            # These dyadic fixtures have exactly representable intermediate sums;
            # equality here is not a claim about opaque SUM ordering in general.
            assert torch.equal(flat, torch.from_numpy(expected_r)), (arm, n, tile, step, flat, expected_r)
            if coordinates:
                for _, master, target in coordinate_map.pairs:
                    target.copy_(master)
                coordinate_map.assert_model_committed()
            else:
                assert torch.equal(model, flat.bfloat16())
            r_pad = torch.from_numpy(np.pad(expected_r, (0, width * k - n)))
            m_pad = torch.from_numpy(np.pad(expected_m, (0, width * k - n)))
            assert torch.equal(momentum, m_pad[rank * width:(rank + 1) * width])
            assert torch.equal(reference, r_pad if arm == 'resident' else r_pad[rank * width:(rank + 1) * width])
            assert payload['modeled_payload_bytes'] == (3 if arm == 'gather' else 2) * (k - 1) * width * 4
            assert payload['physical_wire_bytes'] is None
            tiles = math.ceil(width / tile) if k > 1 else 0
            assert payload['collective_calls']['reduce_scatter'] == tiles
            assert payload['collective_calls']['reference_allgather'] == (tiles if arm == 'gather' else 0)
            assert payload['api_input_bytes']['reduce_scatter'] == (k * width * 4 if k > 1 else 0)
            assert payload['api_output_bytes']['updated_reference_allgather'] == (k * width * 4 if k > 1 else 0)
            assert addresses == [p.data_ptr() for p in executor.storage_tensors()]
        outputs[arm] = flat.clone()
        ledger = executor.allocation_bytes()
        assert ledger['workspace_tensor_bytes'] == 4 * tile * workspace_elements_per_tile(k, arm)
        assert ledger['reference_bytes'] == (k if arm == 'resident' else 1) * width * 4
    if witness and k > 1:
        assert torch.equal(outputs['gather'], outputs['resident'])
        assert not torch.equal(outputs['gather'], outputs['recenter'])


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, world_size=4, rank=rank,
                           timeout=timedelta(seconds=120))
    try:
        for n in (1, 5, 23, 65):
            for tile in (1, 3, 17):
                run_case(dist.group.WORLD, n, tile)
        run_case(dist.group.WORLD, 3, 1, witness=True)
        run_case(dist.group.WORLD, 23, 2, coordinates=True)
        own = None
        for peers in ((0, 2), (1, 3)):
            group = dist.new_group(list(peers))
            if rank in peers:
                own = group
        run_case(own, 9, 2)
        run_case(own, 3, 2, witness=True)
        for peer in range(4):
            group = dist.new_group([peer])
            if rank == peer:
                own = group
        run_case(own, 7, 3)
        master = torch.zeros(8)
        try:
            CollectiveExecutor(master, master[:8], torch.zeros(8), arm='resident', tile_elements=1, group=own)
        except ValueError as exc:
            assert 'alias' in str(exc)
        else:
            raise AssertionError('aliased master/reference accepted')
        dist.barrier()
    finally:
        dist.destroy_process_group()


class CollectiveTests(unittest.TestCase):
    def test_real_collectives_state_padding_and_recenter_witness(self):
        with tempfile.TemporaryDirectory(prefix='pier-native-collectives-') as name:
            mp.spawn(worker, args=(f'file://{name}/rendezvous',), nprocs=4, join=True)

    def test_equal_workspace_caps(self):
        for k in (1, 2, 4, 16, 32):
            for budget in (64 << 20, 256 << 20):
                for arm, s in [(a, 1) for a in ARMS] + [('pier', 2**j) for j in range(k.bit_length())]:
                    capacity = tile_for_budget(k, arm, s, 10**10, budget)
                    per_element = 4 * workspace_elements_per_tile(k, arm, s)
                    self.assertLessEqual(capacity * per_element, budget)
                    self.assertGreater((capacity + 1) * per_element, budget)
        with self.assertRaises(ValueError):
            tile_for_budget(4, 'gather', 1, 100, 1)


if __name__ == '__main__':
    unittest.main()
