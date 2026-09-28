#!/usr/bin/env python3
"""CUDA/Gloo gate for all directed layout edges, slots, pages and durable state.

Uses the production executor and an independent NumPy FP32 resident oracle.
This is operator/commit correctness evidence, not full Qwen performance.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist

from experiments.joint.common import gather, initialize, receipt, write_report
from megatron.core.outer_sync.coordinates import ParameterCoordinates
from megatron.core.outer_sync.joint import JointExecutor
from megatron.core.outer_sync.joint_plan import Layout
from megatron.core.outer_sync.spec import resident_reference


def same(actual, expected):
    np.testing.assert_array_equal(actual.detach().cpu().numpy().view(np.uint32),
                                  np.asarray(expected, dtype=np.float32).view(np.uint32))


def verify(group, device, scratch):
    k, rank = dist.get_world_size(group), dist.get_rank(group)
    records = []
    scratch.mkdir(parents=True, exist_ok=True)
    for n in (1, 7, 257):
        pairs = [(f'p{i:04}', torch.tensor([.125 + i / 31], device=device),
                  torch.zeros(1, device=device, dtype=torch.bfloat16)) for i in range(n)]
        coords = ParameterCoordinates(pairs)
        engine = JointExecutor(coordinates=coords, group=group, cohort=1, page_elements=3,
                               capacities=(3, 6), slot_counts=(1, 2, 4), trace=True)
        r, m = coords.cpu_flat().numpy().copy(), np.zeros(n, dtype=np.float32)
        choices = sorted({1, min(2, k), k})
        targets = [(target, q, mode) for q in (1, 2, 4) for mode in ('joint', 'separate')
                   for old in choices for new in choices for target in (old, new)]
        for round_index, (target, q, mode) in enumerate(targets):
            for i, (_, master, _) in enumerate(pairs):
                master.add_((rank + 1) * (round_index + 1) * 1e-5)
                if round_index == 4 and i == 0:
                    master.fill_((2**24, 1., -2**24, 1.)[rank % 4])
            local = coords.cpu_flat().to(device)
            leaves = [torch.empty_like(local) for _ in range(k)]
            dist.all_gather(leaves, local, group=group)
            expected = resident_reference(r, m, np.stack([value.cpu().numpy() for value in leaves]))
            old, capacity = engine.s, (3, 6)[round_index % 2]
            bases = gather(engine._base_bytes(), group)
            candidates = engine.planner.candidates(old, [2**40] * k, base=bases, headroom=64 * 2**20,
                                                   mode=mode,
                                                   target=target, fixed_slots=q)
            plan = next(p for p in candidates if p.capacity == capacity)
            result = engine.step(plan=plan, headroom_bytes=64 * 2**20)
            same(coords.cpu_flat(), expected['reference'])
            coords.assert_model_committed(finite=True)
            padded_r = np.pad(expected['reference'], (0, k * engine.width - n))
            padded_m = np.pad(expected['momentum'], (0, k * engine.width - n))
            same(engine.momentum, padded_m[rank * engine.width:(rank + 1) * engine.width])
            for (j, offset), value in engine.reference.pages.items():
                start = j * engine.width + offset
                same(value, padded_r[start:start + value.numel()])
            sent = gather(result['sent_tensor_bytes_by_phase'], group)
            totals = {key: sum(row[key] for row in sent) for key in sent[0]}
            expected_oneway = 4 * engine.width * k * (k - 1)
            assert totals['learner_input'] + totals['partial_reduce'] == expected_oneway
            assert totals['parameter_return'] == expected_oneway
            remote = sum(j != Layout(k, old).executor(j) for j in range(k)) * engine.width * 4
            assert totals['momentum_stage'] == totals['momentum_writeback'] == remote
            assert totals['reference_migration'] == 0
            if round_index == len(targets) // 2:
                path = scratch / f'n{n}.pt'
                torch.save(engine.state_dict(), path)
                for value in engine.reference.pages.values():
                    value.fill_(99)
                engine.momentum.fill_(99)
                engine.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
                same(engine.momentum, padded_m[rank * engine.width:(rank + 1) * engine.width])
                path.unlink()
            records.append(dict(elements=n, old_cohort=old, target=target, slots=q, capacity=capacity,
                                mode=plan.mode, bitwise_states_and_commit=True,
                                peak_within_admitted_ledger=result['explicit_peak_bytes'] <= plan.peaks[rank],
                                aggregate_sent_bytes=totals))
            r, m = expected['reference'], expected['momentum']
        engine.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a fresh output path')
    device, group = initialize(args.device, args.tp)
    try:
        report = receipt(args, device)
        records = verify(group, device, args.output.parent / 'checkpoint-gate' / f'rank{dist.get_rank()}')
        report.update(kind='correctness', status='passed', rank_records=gather(records),
                      not_tested=['full Qwen CUDA training', 'convergence', 'performance'])
        write_report(args.output, report)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
