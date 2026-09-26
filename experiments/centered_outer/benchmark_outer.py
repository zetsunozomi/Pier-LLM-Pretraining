#!/usr/bin/env python3
"""Isolated outer-update + BF16 commit benchmark, launched with torchrun.

Synthetic flat masters isolate outer execution. This is not Qwen training,
convergence evidence, or end-to-end token throughput. All raw samples remain
in the report; each repeat uses a recorded randomized arm order.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from megatron.core.outer_sync.collectives import CollectiveExecutor, tile_for_budget
from megatron.core.outer_sync.executor import CenteredExecutor, power2


def drain(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_case(args, arm, group, device):
    k, rank = dist.get_world_size(group), dist.get_rank(group)
    n, width = args.elements, (args.elements + k - 1) // k
    pier = arm in ('reference', 'contiguous')
    g, b = k // args.cohort, rank % args.cohort
    # Flat tensors deliberately exclude the parameter-fragmentation overhead
    # of a full Megatron model. Confirm final gains in the N2 training path.
    master = torch.full((n,), .125, dtype=torch.float32, device=device)
    model = torch.empty(n, dtype=torch.bfloat16, device=device)
    host = arm == 'offload'
    state_device = torch.device('cpu') if host else device
    reference = torch.zeros(g * width if pier else width, dtype=torch.float32,
                            device=state_device, pin_memory=host and device.type == 'cuda')
    start = b * g * width if pier else rank * width
    reference[:max(0, min(reference.numel(), n - start))].fill_(.125)
    momentum = torch.zeros(width, dtype=torch.float32, device=state_device,
                           pin_memory=host and device.type == 'cuda')
    native_arm = 'recenter' if arm == 'recenter' else 'gather'
    capacity = tile_for_budget(k, 'pier' if pier else native_arm, args.cohort if pier else 1,
                               width, args.workspace_mib * 2**20)
    shared = dict(group=group, tile_elements=capacity)
    engine = (CenteredExecutor(master, reference, momentum, cohort=args.cohort, schedule=arm, **shared)
              if pier else CollectiveExecutor(master, reference, momentum, arm=native_arm, **shared))
    samples, payload = [], None
    for step in range(args.warmup + args.samples):
        # Stand-in learner displacement is outside the timed window.
        master.add_((rank + 1) * 1e-5)
        drain(device)
        dist.barrier()
        if step == args.warmup and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        payload = engine.step(mu=.9, eta=.7)
        model.copy_(master)
        drain(device)
        duration = time.perf_counter() - started
        if step >= args.warmup:
            samples.append(duration)
    # Snapshot peaks before validation allocates its independent cast buffer.
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
    if not all(torch.isfinite(value).all().item() for value in (master, reference, momentum)):
        raise AssertionError('nonfinite master or outer state')
    if not torch.equal(model.view(torch.uint8), master.bfloat16().view(torch.uint8)):
        raise AssertionError('stale BF16 model after outer update')
    return {'rank': dist.get_rank(), 'outer_rank': rank, 'outer_ranks': dist.get_process_group_ranks(group),
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            'seconds': samples, 'allocation': engine.allocation_bytes(), 'payload': payload,
            'finite_state': True,
            'tile_elements': capacity, 'reference_and_momentum_on': state_device.type,
            'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--elements', type=int, default=16_777_217,
                        help='FP32 coordinates per corresponding TP group participant, not full model size')
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--cohort', type=int, default=2)
    parser.add_argument('--workspace-mib', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--arms', nargs='+', choices=('reference', 'contiguous', 'gather', 'offload', 'recenter'),
                        default=['reference', 'contiguous', 'gather', 'offload'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    world = int(os.environ['WORLD_SIZE'])
    if (min(args.elements, args.tp, args.workspace_mib, args.samples, args.repeats) < 1
            or args.warmup < 1 or world % args.tp or not power2(world // args.tp)
            or not power2(args.cohort) or (world // args.tp) % args.cohort
            or len(set(args.arms)) != len(args.arms)):
        parser.error('positive sizes, warmup, unique arms, and nested power-of-two learner/cohort counts required')
    if args.output.exists():
        parser.error('output already exists; choose a fresh report path')
    torch.set_num_threads(1)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    try:
        group = None
        for offset in range(args.tp):
            peers = list(range(offset, world, args.tp))
            candidate = dist.new_group(peers)
            if dist.get_rank() in peers:
                group = candidate
        dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
        records, orders = [], []
        for repeat in range(args.repeats):
            order = list(args.arms)
            random.Random(421 + repeat).shuffle(order)
            orders.append(order)
            for arm in order:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                local = run_case(args, arm, group, device)
                rows = [None] * world
                dist.all_gather_object(rows, local)
                slowest = [max(row['seconds'][i] for row in rows) for i in range(args.samples)]
                record = {'repeat': repeat + 1, 'arm': arm, 'rank_records': rows,
                          'max_rank_seconds': slowest, 'mean_seconds': statistics.mean(slowest),
                          'median_seconds': statistics.median(slowest)}
                records.append(record)
                if dist.get_rank() == 0:
                    print(json.dumps({key: record[key] for key in ('repeat', 'arm', 'mean_seconds', 'median_seconds')}), flush=True)
        if dist.get_rank() == 0:
            sources = [Path(__file__), *(ROOT / 'megatron/core/outer_sync').glob('*.py')]
            report = {'scope': 'synthetic flat-master outer update plus BF16 commit; no inner training',
                      'offload_definition': 'sharded pinned CPU R/M with GPU reference gather and GPU owner update',
                      'arm_definitions': {
                          'reference': 'strict Pier, original ordered schedule',
                          'contiguous': 'strict Pier, contiguous ordered schedule',
                          'gather': 'G, sharded GPU reference reconstruction',
                          'offload': 'OS, sharded pinned CPU reference reconstruction',
                          'recenter': 'W, fully sharded GPU R/M, native raw-weight averaging'},
                      'GPU_executed': device.type == 'cuda', 'end_to_end_training_result': False,
                      'torch': torch.__version__, 'cuda': torch.version.cuda, 'world_size': world,
                      'config': {**vars(args), 'output': str(args.output)}, 'orders': orders, 'records': records,
                      'repeat_scope': 'repeated blocks in one launch, not independent job launches',
                      'timing': 'per-step maximum rank wall time; warmup, input perturbation and barriers excluded',
                      'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
