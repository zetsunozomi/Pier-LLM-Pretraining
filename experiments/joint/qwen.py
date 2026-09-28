#!/usr/bin/env python3
"""One paired Qwen campaign per allocation; repeat via independent array jobs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.joint.common import sources
from experiments.joint.require_gate import check, check_training
from experiments.qwen.n2 import run as run_n2


def configurations(scenario, *, low=29.5 * 1024, middle=32 * 1024, high=35 * 1024, costs=(), trace=False):
    result = []
    for variant in ('single', 'pipeline', 'separate', 'joint'):
        for initial in ((1, 2, 16) if variant in ('single', 'pipeline') else (1,)):
            budgets = [dict(round=1, transition_mib=39 * 1024, next_phase_mib=high, remaining_rounds=3)]
            if scenario in ('budgets', 'forced'):
                budgets += [dict(round=r, transition_mib=39 * 1024, next_phase_mib=b, remaining_rounds=3)
                            for r, b in ((4, middle), (5, low), (7, middle), (8, high))]
            if scenario == 'forced' and variant in ('separate', 'joint'):
                for row, target in zip(budgets, (1, 2, 16, 2, 1)):
                    row['target'] = target
            config = dict(version=1, variant=variant, page_elements=65536,
                          capacities=[65536, 262144, 1048576], slot_counts=[1, 2, 4],
                          workspace_mib=64, headroom_mib=1536, trace=trace, budgets=budgets, costs=list(costs))
            result.append(dict(name=f'{variant}-s{initial}', initial_cohort=initial, config=config))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('fixed', 'budgets', 'forced'), default='fixed')
    parser.add_argument('--repeat-id', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--training-gate', type=Path, help='Passed real pretrain_gpt restart gate; required for GPU runs')
    parser.add_argument('--cost-table', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--only', nargs='+', help='Explicit subset, e.g. joint-s1 separate-s1; recorded in campaign')
    parser.add_argument('--low-mib', type=float, default=29.5 * 1024)
    parser.add_argument('--middle-mib', type=float, default=32 * 1024)
    parser.add_argument('--high-mib', type=float, default=35 * 1024)
    parser.add_argument('--plan-only', action='store_true', help='Write exact configs/manifest; do not launch GPUs')
    parser.add_argument('--trace', action='store_true', help='Diagnostic memory ledger and CUDA event timeline; keep separate from primary timings')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('fresh output directory required')
    if not 0 < args.low_mib < args.middle_mib < args.high_mib <= 39 * 1024:
        parser.error('strictly increasing positive next-phase budgets up to transition capacity required')
    costs = []
    if args.cost_table:
        table = json.loads(args.cost_table.read_text())
        if table.get('status') != 'complete' or not table.get('GPU_executed') or not table.get('costs'):
            parser.error('cost table must contain completed CUDA configuration measurements')
        if (table.get('world_size') != 32 or table['recipe']['tp'] != 2
                or table['recipe']['elements'] != 1_543_044_096 or table['recipe']['page_elements'] != 65536
                or table['recipe']['workspace_mib'] != 64 or table['sources'] != sources()):
            parser.error('cost table does not match this source, topology, coordinate count or workspace')
        costs = table['costs']
    planned = configurations(args.scenario, low=args.low_mib, middle=args.middle_mib, high=args.high_mib,
                             costs=costs, trace=args.trace)
    # Keep the common budget trace even for a control-only --only subset.
    budget_announcements = planned[0]['config']['budgets']
    # Re-run the original colocated executor as a strong static control. The
    # paged variants hold the same canonical momentum homes across all layouts;
    # their s=2 single-slot arm alone is not the best colocated static baseline.
    planned += [dict(name=f'reference-s{s}', initial_cohort=s, config=None) for s in (1, 2, 16)]
    planned.append(dict(name='native-controls', initial_cohort=2, config=None))
    if args.only:
        if set(args.only) - {case['name'] for case in planned}:
            parser.error('unknown case in --only')
        planned = [case for case in planned if case['name'] in args.only]
    random.Random(511 + args.repeat_id).shuffle(planned)
    if not args.plan_only:
        if int(os.environ.get('PIER_N2_NODES', 0)) != 8:
            parser.error('this paper campaign requires 8 nodes / 32 GPUs / TP2')
        check(args.gate, 32, 2)
        if args.training_gate is None:
            parser.error('--training-gate is required for GPU training campaigns')
        check_training(args.training_gate)
    args.output_dir.mkdir(parents=True)
    manifest = dict(format='pier-joint-qwen-campaign-v1', scenario=args.scenario, repeat_id=args.repeat_id,
                    created_utc=datetime.now(timezone.utc).isoformat(), sources=sources(),
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'), scope='32 A100-40GB, Qwen3B/TP2, synthetic tokens',
                    planned_cases=planned, measured_cycles=3 if args.scenario == 'fixed' else 8,
                    budget_announcements=budget_announcements,
                    cost_source=str(args.cost_table) if args.cost_table else 'uncalibrated analytic estimate',
                    plan_only=args.plan_only, cases=[])
    for case in planned:
        if case['config'] is not None:
            (args.output_dir / f"{case['name']}.json").write_text(json.dumps(case['config'], indent=2) + '\n')
    path = args.output_dir / 'campaign.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.plan_only:
        print(path)
        return
    for case in planned:
        if sources() != manifest['sources']:
            raise RuntimeError('source changed during paired campaign')
        env = dict(PIER_QWEN_MODEL_SIZE='3B', PIER_QWEN_DATA_PREFIX='', PIER_N2_PROFILE='main',
                   PIER_N2_SUITE='baselines', PIER_N2_REPEATS='1', PIER_N2_REPEAT_START=str(args.repeat_id),
                   PIER_N2_MEASURED_CYCLES=str(manifest['measured_cycles']), PIER_N2_WORKSPACE_MIB='64',
                   PIER_N2_EXPECTED_GPU='NVIDIA A100-SXM4-40GB', PIER_N2_COHORT=str(case['initial_cohort']))
        if case['config'] is not None:
            env.update(PIER_N2_ARMS='P', PIER_N2_PIER_SCHEDULE='joint',
                       PIER_N2_JOINT_CONFIG=str((args.output_dir / f"{case['name']}.json").resolve()))
        else:
            env.update(PIER_N2_ARMS='G,OS,R,W' if case['name'] == 'native-controls' else 'P',
                       PIER_N2_PIER_SCHEDULE='reference', PIER_N2_JOINT_CONFIG='')
        os.environ.update(env)
        directory = args.output_dir / case['name']
        exit_code = run_n2(directory)
        manifest['cases'].append(dict(name=case['name'], directory=str(directory), exit_code=exit_code))
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('All campaign attempts recorded; use qwen_summary.py to retain infeasible and failed controls separately.')


if __name__ == '__main__':
    main()
