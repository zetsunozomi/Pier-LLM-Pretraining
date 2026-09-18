"""Real four-process Gloo DTensor layouts, padding and shared-runtime restore.

CPU collectives can use fallback implementations; this is not a GPU result or
evidence of physical communication volume, tuned throughput or total memory.
"""

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.outer_sync.checkpoint import load, restore_rng, save
from megatron.core.outer_sync.collectives import tile_for_budget
from megatron.core.outer_sync.coordinates import ParameterCoordinates
from megatron.core.outer_sync.dtensor import DTensorExecutor, make_mesh, workspace_elements
from megatron.core.outer_sync.runtime import digest
from megatron.core.outer_sync.spec import resident_reference
from test_runtime import advance, fixture


def check_case(group, mesh, s, n, tile, ordinary=False):
    rank, k = dist.get_rank(group), dist.get_world_size(group)
    g, (a, b), width = k // s, divmod(rank, s), (n + k - 1) // k
    rng = np.random.default_rng(728)
    r = (rng.normal(0, .2, n).astype(np.float32) if ordinary else np.arange(n, dtype=np.float32) / 16)
    m = (rng.normal(0, .01, n).astype(np.float32) if ordinary else np.arange(n, dtype=np.float32) / 128)
    masters = torch.from_numpy(r.copy())
    model = masters.bfloat16()
    split = n // 2
    coordinates = ParameterCoordinates([('a', masters[:split], model[:split]),
                                       ('b', masters[split:], model[split:])])
    r_pad = torch.from_numpy(np.pad(r, (0, k * width - n)))
    m_pad = torch.from_numpy(np.pad(m, (0, k * width - n)))
    reference = r_pad[b * g * width:(b + 1) * g * width].clone()
    momentum = m_pad[(b * g + a) * width:(b * g + a + 1) * width].clone()
    executor = DTensorExecutor(None, reference, momentum, cohort=s, tile_elements=tile,
                               coordinates=coordinates, group=group, mesh=mesh)
    addresses = [t.data_ptr() for t in executor.storage_tensors()]
    for step in range(3):
        delta = (rng.normal(0, .003, (k, n)).astype(np.float32) if ordinary
                 else np.asarray([np.full(n, (peer + 1) * (step + 1) / 32, np.float32) for peer in range(k)]))
        weights = np.add(r[None], delta, dtype=np.float32)
        masters.copy_(torch.from_numpy(weights[rank]))
        expected = resident_reference(r, m, weights, mu=.5, eta=.25)
        r, m = expected['reference'], expected['momentum']
        payload = executor.step(mu=.5, eta=.25)
        if ordinary:
            torch.testing.assert_close(masters, torch.from_numpy(r), rtol=1e-6, atol=1e-8)
        else:
            assert torch.equal(masters, torch.from_numpy(r)), (s, n, tile, step)
        for _, source, target in coordinates.pairs:
            target.copy_(source)
        coordinates.assert_model_committed()
        r_pad = torch.from_numpy(np.pad(r, (0, k * width - n)))
        m_pad = torch.from_numpy(np.pad(m, (0, k * width - n)))
        expected_r = r_pad[b * g * width:(b + 1) * g * width]
        expected_m = m_pad[(b * g + a) * width:(b * g + a + 1) * width]
        if ordinary:
            torch.testing.assert_close(reference, expected_r, rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(momentum, expected_m, rtol=1e-6, atol=1e-8)
        else:
            assert torch.equal(reference, expected_r) and torch.equal(momentum, expected_m)
        assert addresses == [t.data_ptr() for t in executor.storage_tensors()]
        assert payload['modeled_payload_bytes'] == 2 * (k - 1) * width * 4
        assert payload['physical_wire_bytes'] is None
    allocation = executor.allocation_bytes()
    assert allocation['workspace_tensor_bytes'] == 4 * tile * (k + 2 * g + 2)
    assert allocation['explicit_live_upper_bound_bytes'] == 4 * tile * workspace_elements(k, s)
    assert allocation['reference_bytes'] == 4 * g * width
    assert allocation['momentum_bytes'] == 4 * width


def operator_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=120))
    try:
        for s in (1, 2, 4):
            mesh = make_mesh(dist.group.WORLD, s, 'cpu')
            for n, tile in ((1, 1), (5, 3), (23, 1), (23, 3), (65, 17)):
                check_case(dist.group.WORLD, mesh, s, n, tile)
            check_case(dist.group.WORLD, mesh, s, 65, 7, ordinary=True)
        own = None
        for peers in ((0, 2), (1, 3)):
            group = dist.new_group(list(peers))
            if rank in peers:
                own = group
        for s in (1, 2):
            mesh = make_mesh(own, s, 'cpu')
            check_case(own, mesh, s, 9, 2)
        for peer in range(4):
            group = dist.new_group([peer])
            if rank == peer:
                own = group
        mesh = make_mesh(own, 1, 'cpu')
        check_case(own, mesh, 1, 7, 3)
        source, target = torch.arange(8, dtype=torch.float32), torch.empty(8, dtype=torch.bfloat16)
        executor = DTensorExecutor(source, source.clone(), torch.zeros(8), cohort=1, tile_elements=3,
                                   model=target, group=own, mesh=mesh)
        executor.step()
        assert torch.equal(target, source.bfloat16())
        try:
            DTensorExecutor(source, source, torch.zeros(8), cohort=1, tile_elements=3, group=own, mesh=mesh)
        except ValueError as exc:
            assert 'alias' in str(exc)
        else:
            raise AssertionError('aliased state accepted')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def training_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=120))
    try:
        for s in (1, 2, 4):
            path = Path(directory) / f's{s}'
            runtime, model, optimizer, scheduler = fixture(rank, s, path, arm='dtensor')
            advance(runtime, model, optimizer, scheduler, 11)
            runtime.coordinates.assert_model_committed()
            expected = digest([optimizer.state_dict(), model.state_dict(), scheduler.state_dict(),
                               runtime.executor.reference, runtime.executor.momentum, runtime.clock.state_dict()])
            runtime, model, optimizer, scheduler = fixture(rank, s, path, arm='dtensor')
            advance(runtime, model, optimizer, scheduler, 5)
            save(runtime, 5, scheduler, 0.)
            runtime, model, optimizer, scheduler = fixture(rank, s, path, arm='dtensor')
            runtime.args.outer_tile_elements = 3
            try:
                load(runtime, str(path), scheduler)
            except ValueError as exc:
                assert 'native_collective_tile' in str(exc)
            else:
                raise AssertionError('changed native reduction tile accepted')
            runtime.args.outer_tile_elements = 2
            with patch('megatron.core.optimizer.optimizer.parallel_state.get_pipeline_model_parallel_world_size', return_value=1):
                iteration, _ = load(runtime, str(path), scheduler)
            assert iteration == 5 and runtime.args.consumed_train_samples == 40
            restore_rng(runtime.pending_rng)
            runtime.pending_rng = None
            advance(runtime, model, optimizer, scheduler, 11)
            runtime.finish(11)
            runtime.coordinates.assert_model_committed()
            assert digest([optimizer.state_dict(), model.state_dict(), scheduler.state_dict(),
                           runtime.executor.reference, runtime.executor.momentum, runtime.clock.state_dict()]) == expected
            assert runtime.clock.successful == 10 and runtime.clock.boundaries == 3
            from megatron.core.outer_sync.cycle_metrics import CycleMeter
            from experiments.centered_outer.cycle_summary import summarize
            runtime, model, optimizer, scheduler = fixture(rank, s, path, arm='dtensor')
            runtime.meter = CycleMeter('cpu', path / 'cycles', warmup_cycles=1, planned_attempts=11,
                                      metadata={'fixture': 'CPU only', 'arm': 'dtensor', 'cohort': s})
            advance(runtime, model, optimizer, scheduler, 11)
            runtime.finish(11)
            dist.barrier()
            if rank == 0:
                collected = summarize(path / 'cycles', allow_cpu=True)
                assert collected['status'] == 'raw_cycles_validated' and not collected['GPU_executed']
                assert collected['eligible_cycles'] == 2
                assert collected['run_sample']['successful_loss_tokens_global'] == 48
        dist.barrier()
    finally:
        dist.destroy_process_group()


class DTensorTests(unittest.TestCase):
    def test_real_dtensor_cohorts_padding_coordinate_owners_and_subgroups(self):
        with tempfile.TemporaryDirectory(prefix='pier-dtensor-operators-') as name:
            mp.spawn(operator_worker, args=(f'file://{name}/rendezvous',), nprocs=4, join=True)

    def test_shared_adamw_skip_and_exact_checkpoint_resume(self):
        with tempfile.TemporaryDirectory(prefix='pier-dtensor-training-') as name:
            mp.spawn(training_worker, args=(f'file://{name}/rendezvous', name), nprocs=4, join=True)

    def test_tile_cap_covers_exposed_dynamic_outputs(self):
        for k in (1, 2, 4, 16, 32):
            for s in (2**j for j in range(k.bit_length())):
                for budget in (64 << 20, 256 << 20):
                    capacity = tile_for_budget(k, 'dtensor', s, 10**10, budget)
                    coefficient = 4 * workspace_elements(k, s)
                    self.assertLessEqual(capacity * coefficient, budget)
                    self.assertGreater((capacity + 1) * coefficient, budget)
        with self.assertRaises(ValueError):
            workspace_elements(4, 8)


if __name__ == '__main__':
    unittest.main()
