"""Distributed tests of actual reference refresh, payload and page admission."""

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.outer_sync.coordinates import ParameterCoordinates
from megatron.core.outer_sync.joint import JointExecutor
from megatron.core.outer_sync.joint_plan import Layout, Planner, memory_schedule, slot_bytes
from megatron.core.outer_sync.spec import resident_reference


def same(actual, expected):
    a = actual.detach().cpu().numpy().view(np.uint32)
    b = np.asarray(expected, dtype=np.float32).view(np.uint32)
    np.testing.assert_array_equal(a, b)


def exercise(group, directory):
    k, rank = dist.get_world_size(group), dist.get_rank(group)
    for n in (1, 7, 65):
        pairs = [(str(i).zfill(4), torch.tensor([.125 + i / 31]), torch.zeros(1, dtype=torch.bfloat16))
                 for i in range(n)]
        coordinates = ParameterCoordinates(pairs)
        engine = JointExecutor(coordinates=coordinates, group=group, cohort=1,
                               page_elements=3, capacities=(3, 6), slot_counts=(1, 2, 4), trace=True)
        reference = coordinates.cpu_flat().numpy().copy()
        momentum = np.zeros(n, dtype=np.float32)
        cohorts = sorted({1, min(2, k), k})
        # Every directed edge, plus consecutive self-layout boundaries.
        targets = []
        for old in cohorts:
            for target in cohorts:
                targets.extend((old, target))
        for turn, target in enumerate(targets):
            q, capacity = (1, 2, 4)[turn % 3], (3, 6)[turn % 2]
            for i, (_, master, _) in enumerate(pairs):
                master.add_((rank + 1) * (turn + 1) * 1e-5)
                if turn == 4 and i == 0:
                    master.fill_((2**24, 1., -2**24, 1.)[rank % 4])
            local = coordinates.cpu_flat()
            leaves = [torch.empty_like(local) for _ in range(k)]
            dist.all_gather(leaves, local, group=group)
            expected = resident_reference(reference, momentum, np.stack([v.numpy() for v in leaves]))
            old = engine.s
            mode = 'separate' if turn % 4 == 1 else 'joint'
            candidates = engine.planner.candidates(old, [10**9] * k, base=[engine._base_bytes()] * k,
                                                   target=target, mode=mode, fixed_slots=q)
            plan = next(p for p in candidates if p.capacity == capacity)
            payload = engine.step(plan=plan)
            same(coordinates.cpu_flat(), expected['reference'])
            coordinates.assert_model_committed()
            padded_r = np.pad(expected['reference'], (0, k * engine.width - n))
            padded_m = np.pad(expected['momentum'], (0, k * engine.width - n))
            same(engine.momentum, padded_m[rank * engine.width:(rank + 1) * engine.width])
            assert engine.s == target
            for (j, offset), value in engine.reference.pages.items():
                start = j * engine.width + offset
                same(value, padded_r[start:start + value.numel()])
            assert engine.reference.nbytes == 4 * engine.width * (k // target)
            assert engine.reference.nbytes == sum(value.numel() * 4 for value in engine.reference.pages.values())
            assert payload['explicit_peak_bytes'] <= plan.peaks[rank]
            assert payload['reference_migration_bytes'] == 0
            assert all(row['accounted_total_bytes'] <= plan.peaks[rank] for row in payload['trace'])
            rows = [None] * k
            dist.all_gather_object(rows, payload['sent_tensor_bytes_by_phase'], group=group)
            totals = {key: sum(row[key] for row in rows) for key in rows[0]}
            assert totals['learner_input'] + totals['partial_reduce'] == 4 * engine.width * k * (k - 1)
            assert totals['parameter_return'] == 4 * engine.width * k * (k - 1)
            remotes = sum(j != Layout(k, old).executor(j) for j in range(k)) * engine.width * 4
            assert totals['momentum_stage'] == totals['momentum_writeback'] == remotes
            assert totals['reference_migration'] == 0
            if turn == len(targets) // 2:
                checkpoint = Path(directory) / f'joint-k{k}-n{n}-r{rank}.pt'
                torch.save(engine.state_dict(), checkpoint)
                # Restore into the same configured executor after corrupting
                # storage: a real deserialization, not a dictionary alias.
                for value in engine.reference.pages.values():
                    value.fill_(99)
                engine.momentum.fill_(99)
                engine.load_state_dict(torch.load(checkpoint, weights_only=False))
                same(engine.momentum, padded_m[rank * engine.width:(rank + 1) * engine.width])
                for (j, offset), value in engine.reference.pages.items():
                    start = j * engine.width + offset
                    same(value, padded_r[start:start + value.numel()])
            reference, momentum = expected['reference'], expected['momentum']
        # Rejected admission must not mutate any persistent state.
        before = engine.state_dict()
        try:
            engine.plan(1, target=1)
        except MemoryError:
            pass
        else:
            raise AssertionError('infeasible budget accepted')
        assert engine.round == before['round'] and not engine.busy
        engine.close()


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        exercise(dist.group.WORLD, directory)
        groups = [dist.new_group(peers) for peers in ([0, 2], [1, 3])]
        exercise(groups[rank % 2], Path(directory) / f'group-{rank % 2}')
        dist.barrier()
    finally:
        dist.destroy_process_group()


class JointTests(unittest.TestCase):
    def test_peak_formula_matches_all_tile_prefixes(self):
        # Independent literal lifetime simulation, including uneven tails,
        # ownership changes and rank-specific training allocations.
        for k in (1, 2, 4, 8, 16):
            for width in (1, 3, 7, 17):
                for capacity in (1, 3, 6, 32):
                    for old in sorted({1, min(2, k), k}):
                        for new in sorted({1, min(2, k), k}):
                            for q in (1, 2, 4):
                                before, after = Layout(k, old), Layout(k, new)
                                base = tuple(7 * i for i in range(k))
                                fixed = [b + 13 + 4 * width + q * slot_bytes(old, capacity) for b in base]
                                references = [4 * width * before.groups] * k
                                expected = [fixed[i] + references[i] for i in range(k)]
                                for offset in range(0, width, capacity):
                                    size = 4 * min(capacity, width - offset)
                                    for shard in range(k):
                                        for rank in before.owners(shard):
                                            references[rank] -= size
                                        for rank in after.owners(shard):
                                            references[rank] += size
                                        expected = [max(expected[i], fixed[i] + references[i]) for i in range(k)]
                                actual = memory_schedule(k, width, old, new, capacity, q, base=base, headroom=13)
                                self.assertEqual(actual[0], tuple(expected))

    def test_planner(self):
        planner = Planner(4, 17, 3, capacities=(3, 6), slots=(1, 2))
        for mode in ('static', 'separate', 'joint'):
            for old in (1, 2, 4):
                for new in ((old,) if mode == 'static' else (1, 2, 4)):
                    plans = planner.candidates(old, [10000] * 4, target=new, mode=mode)
                    self.assertEqual(len(plans), 4)
                    for plan in plans:
                        self.assertEqual(plan.workspace_bytes, plan.slots * slot_bytes(old, plan.capacity))
                        # Exact peak is accepted; one byte below at any limiting
                        # rank must reject this specific configuration.
                        fitted = planner.candidates(old, plan.peaks, target=new, mode=mode)
                        self.assertIn(plan, fitted)
                        tight = tuple(x - 1 for x in plan.peaks)
                        self.assertNotIn(plan, planner.candidates(old, tight, target=new, mode=mode))
        with self.assertRaises(MemoryError):
            planner.choose(1, (1, 1, 1, 1))
        with self.assertRaises(ValueError):
            Planner(4, 17, 3, capacities=(4,))

    def test_contraction_reduces_workspace_and_expansion_needs_transition_headroom(self):
        planner = Planner(4, 17, 3, capacities=(3, 6), slots=(1, 2))
        plan = planner.choose(1, [400] * 4, final_budgets=[160] * 4, target=4)
        self.assertEqual((plan.cohort, plan.capacity, plan.slots), (4, 3, 1))
        # Both endpoints fit 400 bytes (final state is 340), but the
        # minimum live return workspace makes this expansion need 412.
        with self.assertRaises(MemoryError):
            planner.choose(4, [400] * 4, target=1)
        plan = planner.choose(4, [412] * 4, target=1)
        self.assertEqual(plan.peaks, (412, 412, 412, 412))
        with self.assertRaises(ValueError):
            planner.choose(1, [float('nan')] * 4)

    def test_real_gloo_all_transitions_payloads_pages_and_restore(self):
        with tempfile.TemporaryDirectory(prefix='pier-joint-') as directory:
            for suffix in ('group-0', 'group-1'):
                (Path(directory) / suffix).mkdir()
            mp.spawn(worker, args=(f'file://{directory}/rendezvous', directory), nprocs=4, join=True)

    def test_switch_requires_amortizing_its_measured_cost(self):
        costs = [dict(old_cohort=old, cohort=new, capacity=3, slots=1, mode='joint', seconds=seconds)
                 for old, new, seconds in ((1, 1, 10.), (1, 2, 20.), (2, 2, 1.))]
        planner = Planner(4, 17, 3, cohorts=(1, 2), capacities=(3,), slots=(1,), costs=costs)
        self.assertEqual(planner.choose(1, [1000] * 4, remaining_rounds=1).cohort, 1)
        self.assertEqual(planner.choose(1, [1000] * 4, remaining_rounds=4).cohort, 2)


if __name__ == '__main__':
    unittest.main()
