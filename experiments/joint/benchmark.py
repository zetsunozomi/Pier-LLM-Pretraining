#!/usr/bin/env python3
"""Configuration and transition measurements: one independent torchrun per file.

Inputs are synthetic flat masters; complete outer update includes BF16 commit.
Transition repetitions rebuild the specified old layout outside the timer from
the preceding committed model, then perturb learner values. This isolates each
directed transition rather than mislabeling steady rounds as transition samples.
"""

import argparse
import gc
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from experiments.joint.common import drain, gather, initialize, receipt, write_report
from megatron.core.outer_sync.joint import JointExecutor, ReferencePages
from megatron.core.outer_sync.joint_plan import AdmissionError, Layout


def cases(args, k):
    cohorts = args.cohorts or sorted({1, min(2, k), k})
    for s in cohorts:
        Layout(k, s)
    result = []
    for old in cohorts:
        for target in (cohorts if args.scenario == 'transitions' else (old,)):
            for capacity in args.capacities:
                for slots in args.slots:
                    for mode in (('static', 'joint', 'separate') if old == target else ('joint', 'separate')):
                        result.append(dict(old_cohort=old, cohort=target, capacity=capacity, slots=slots, mode=mode))
    random.Random(421 + args.repeat_id).shuffle(result)
    return result


def run_case(args, case, device, group):
    k, rank = dist.get_world_size(group), dist.get_rank(group)
    master = torch.full((args.elements,), .125, device=device)
    model = torch.empty(args.elements, device=device, dtype=torch.bfloat16)
    engine = JointExecutor(master, model=model, group=group, cohort=case['old_cohort'],
                           page_elements=args.page_elements, capacities=(case['capacity'],),
                           slot_counts=(case['slots'],), trace=args.trace)
    try:
        return measure_case(args, case, device, group, engine, master, model)
    finally:
        if not engine.busy:
            engine.close()


def measure_case(args, case, device, group, engine, master, model):
    k, rank = dist.get_world_size(group), dist.get_rank(group)
    records = []
    for i in range(args.warmup + args.samples):
        if engine.s != case['old_cohort']:
            engine.reference.clear()
            engine.reference = ReferencePages(learners=k, width=engine.width, page_elements=args.page_elements,
                                              rank=rank, cohort=case['old_cohort'], device=device, read=engine._read)
        master.add_((rank + 1) * 1e-5)
        drain(device)
        dist.barrier()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        plan = engine.plan(int(args.budget_mib * 2**20), target=case['cohort'], mode=case['mode'],
                           slots=case['slots'], headroom_bytes=int(args.headroom_mib * 2**20),
                           workspace_limit=int(args.workspace_mib * 2**20))
        payload = engine.step(plan=plan, headroom_bytes=int(args.headroom_mib * 2**20))
        drain(device)
        duration = time.perf_counter() - started
        row = dict(seconds=duration, warmup=i < args.warmup, payload=payload,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
        records.append(row)
    if not torch.isfinite(master).all() or not torch.equal(model, master.to(model.dtype)):
        raise AssertionError('nonfinite/stale model after benchmark')
    return dict(case=case, samples=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--elements', type=int, default=1_543_044_096)
    parser.add_argument('--page-elements', type=int, default=65536)
    parser.add_argument('--capacities', type=int, nargs='+', default=[65536, 262144, 1048576])
    parser.add_argument('--slots', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--cohorts', type=int, nargs='+')
    parser.add_argument('--scenario', choices=('pipeline', 'transitions'), default='pipeline')
    parser.add_argument('--budget-mib', type=float, default=39 * 1024)
    parser.add_argument('--headroom-mib', type=float, default=1024)
    parser.add_argument('--workspace-mib', type=float, default=64)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--repeat-id', type=int, default=1)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.elements, args.page_elements, args.warmup, args.samples, args.repeat_id) < 1:
        parser.error('positive experiment sizes and a fresh output path required')
    device, group = initialize(args.device, args.tp)
    try:
        report = receipt(args, device)
        report.update(kind='configuration', status='running', records=[],
                      scope='synthetic flat-master complete outer update and BF16 commit; no inner training')
        write_report(args.output, report)
        for case in cases(args, dist.get_world_size(group)):
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            try:
                local = run_case(args, case, device, group)
            except AdmissionError as exc:
                # Planned infeasibility is a result, never a fabricated time.
                local = dict(case=case, infeasible=str(exc))
            ranks = gather(local)
            if any('infeasible' in row for row in ranks):
                record = dict(case=case, status='infeasible', ranks=ranks)
            else:
                maxima = [max(row['samples'][i]['seconds'] for row in ranks)
                          for i in range(args.warmup, args.warmup + args.samples)]
                record = dict(case=case, status='measured', ranks=ranks,
                              maximum_rank_seconds=maxima, launch_mean_seconds=statistics.mean(maxima),
                              launch_median_seconds=statistics.median(maxima))
            report['records'].append(record)
            write_report(args.output, report, update=True)
            if dist.get_rank() == 0:
                print({**case, 'status': record['status'], 'seconds': record.get('launch_mean_seconds')}, flush=True)
        report['status'] = 'complete'
        write_report(args.output, report, update=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
