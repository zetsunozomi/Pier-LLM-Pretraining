#!/usr/bin/env python3
"""Short operator trajectories comparing centered Pier and native raw-W (W).

Identical per-round learner perturbations are applied to each arm's own model.
Report numerical differences; do not equate non-bitwise equality with training
failure or claim downstream quality from these synthetic trajectories.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from experiments.joint.common import gather, initialize, receipt, write_report
from megatron.core.outer_sync.collectives import CollectiveExecutor, tile_for_budget
from megatron.core.outer_sync.joint import JointExecutor
from megatron.core.outer_sync.spec import counterexample


def compare(args, device, group):
    k, rank = dist.get_world_size(group), dist.get_rank(group)
    n, width = args.elements, (args.elements + k - 1) // k
    a = torch.full((n,), 1., device=device)
    b = a.clone()
    ref = torch.zeros(width, device=device)
    valid = max(0, min(width, n - rank * width))
    ref[:valid].fill_(1.)
    momentum = torch.zeros(width, device=device)
    pier = JointExecutor(a, group=group, cohort=min(2, k), page_elements=args.page_elements,
                         capacities=(args.page_elements,), slot_counts=(1, 2))
    capacity = tile_for_budget(k, 'recenter', 1, width, 64 * 2**20)
    w = CollectiveExecutor(b, ref, momentum, group=group, arm='recenter', tile_elements=capacity)
    generator = torch.Generator(device=device).manual_seed(2901 + rank)
    rows = []
    for step in range(args.rounds):
        perturbation = torch.randn(n, device=device, generator=generator) * (1e-3 if step < args.rounds // 2 else 1e-7)
        a.add_(perturbation)
        b.add_(perturbation)
        plan = pier.plan(2**40, mode='static', slots=2, headroom_bytes=64 * 2**20)
        pier.step(plan=plan, headroom_bytes=64 * 2**20)
        w.step()
        error = (a.double() - b.double()).abs()
        local = dict(round=step + 1, max_abs_master=error.max().item(),
                     rms_master=error.square().mean().sqrt().item(),
                     max_abs_momentum=(pier.momentum.double() - momentum.double()).abs().max().item(),
                     fp32_disagreements=int((a.view(torch.int32) != b.view(torch.int32)).sum().item()),
                     bf16_disagreements=int((a.bfloat16().view(torch.int16) != b.bfloat16().view(torch.int16)).sum().item()),
                     finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
        rows.append(local)
    pier.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--elements', type=int, default=65537)
    parser.add_argument('--page-elements', type=int, default=4096)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.elements, args.page_elements, args.rounds) < 1:
        parser.error('positive sizes and a fresh output required')
    device, group = initialize(args.device, args.tp)
    try:
        report = receipt(args, device)
        report.update(kind='W_numerics', status='complete',
                      scope='synthetic operator trajectories, not Qwen convergence',
                      constructed_fp32_witness=counterexample(), rank_records=gather(compare(args, device, group)),
                      interpretation='W implements the same real-arithmetic update with different FP32 ordering; '
                                     'both performance and numerical differences must be reported.')
        write_report(args.output, report)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
