"""Explicit, checkpointed configuration for the four paper runtime variants."""

import hashlib
import json
import math
from pathlib import Path

from .joint_plan import AdmissionError, Layout


def read_config(path):
    value = json.loads(Path(path).read_text())
    validate_config(value)
    return value


def validate_config(value):
    allowed = {'version', 'variant', 'page_elements', 'capacities', 'slot_counts',
               'workspace_mib', 'headroom_mib', 'trace', 'budgets', 'costs'}
    if not isinstance(value, dict) or set(value) - allowed or value.get('version') != 1:
        raise ValueError('expected joint config version 1 with known fields')
    if value.get('variant') not in ('single', 'pipeline', 'separate', 'joint'):
        raise ValueError('unknown joint-runtime ablation variant')
    page = value.get('page_elements', 65536)
    if not isinstance(page, int) or page < 1:
        raise ValueError('positive allocation page size required')
    capacities = value.get('capacities', [page, 2 * page, 4 * page])
    slots = value.get('slot_counts', [1, 2, 4])
    if (not capacities or any(not isinstance(c, int) or c < page or c % page for c in capacities)
            or not slots or any(not isinstance(q, int) or q < 1 for q in slots) or 1 not in slots):
        raise ValueError('page-aligned capacities and positive slot candidates including one required')
    for name in ('workspace_mib', 'headroom_mib'):
        if name in value and (not math.isfinite(value[name]) or value[name] < (1e-12 if name == 'workspace_mib' else 0)):
            raise ValueError('invalid joint memory allowance')
    rows = value.get('budgets')
    if not isinstance(rows, list) or not rows or rows[0].get('round') != 1:
        raise ValueError('budget announcements must start at round 1')
    prior = 0
    for row in rows:
        if set(row) - {'round', 'transition_mib', 'next_phase_mib', 'target', 'remaining_rounds'}:
            raise ValueError('unknown budget announcement field')
        if not isinstance(row.get('round'), int) or row['round'] <= prior:
            raise ValueError('budget announcement rounds must strictly increase')
        prior = row['round']
        for key in ('transition_mib', 'next_phase_mib'):
            if key not in row or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError('positive transition and next-phase budgets are required')
        if row['next_phase_mib'] > row['transition_mib']:
            raise ValueError('next-phase budget cannot exceed the transition capacity')
        if not isinstance(row.get('remaining_rounds', 1), int) or row.get('remaining_rounds', 1) < 1:
            raise ValueError('positive planning horizon required')
        if 'target' in row and (not isinstance(row['target'], int) or row['target'] < 1):
            raise ValueError('explicit target must be a positive cohort size')
    return value


class JointController:
    def __init__(self, config):
        self.config = validate_config(config)
        self.sha256 = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def executor_options(self):
        return dict(page_elements=self.config.get('page_elements', 65536),
                    capacities=self.config.get('capacities'), slot_counts=self.config.get('slot_counts', (1, 2, 4)),
                    costs=self.config.get('costs', ()), trace=self.config.get('trace', False))

    def announcement(self, round_number):
        return next(row for row in reversed(self.config['budgets']) if row['round'] <= round_number)

    def step(self, executor, *, mu, eta):
        row = self.announcement(executor.round + 1)
        variant = self.config['variant']
        target = row.get('target')
        if target is not None:
            Layout(executor.k, target)
        mode = 'static' if variant in ('single', 'pipeline') else variant
        if mode == 'static':
            target = executor.s
        common = dict(budget_bytes=int(row['transition_mib'] * 2**20),
                      final_budget_bytes=int(row['next_phase_mib'] * 2**20),
                      headroom_bytes=int(self.config.get('headroom_mib', 0) * 2**20),
                      workspace_limit=(int(self.config['workspace_mib'] * 2**20)
                                       if 'workspace_mib' in self.config else None))
        if mode == 'separate' and target is None:
            # Independently select a final layout from its best steady-state
            # execution. Only then tune the current-layout pipeline. The joint
            # planner instead scores the transition and continuation together.
            choices = []
            import torch.distributed as dist
            bases = [None] * executor.k
            dist.all_gather_object(bases, executor._base_bytes(), group=executor.group)
            for cohort in executor.planner.cohorts:
                # Plan the hypothetical steady layout without mutating state or
                # querying an allocator as if that layout were already resident.
                candidates = executor.planner.candidates(
                    cohort, [common['budget_bytes']] * executor.k, base=bases,
                    final_budgets=[common['final_budget_bytes']] * executor.k,
                    headroom=common['headroom_bytes'], mode='static', target=cohort,
                    workspace_limit=common['workspace_limit'])
                if candidates:
                    choices.append(candidates[0])
            if not choices:
                raise AdmissionError('no independently selected final layout fits the next-phase budget')
            target = min(choices, key=lambda p: (p.estimate_seconds, max(p.peaks), p.cohort)).cohort
        round_number = executor.round + 1
        future = [r['round'] for r in self.config['budgets'] if r['round'] > round_number]
        horizon = min(row.get('remaining_rounds', 1), min(future) - round_number + 1) if future else row.get('remaining_rounds', 1)
        plan = executor.plan(**common, target=target, mode=mode,
                             slots=1 if variant == 'single' else None,
                             remaining_rounds=horizon)
        result = executor.step(mu=mu, eta=eta, plan=plan, headroom_bytes=common['headroom_bytes'])
        result['announcement'] = dict(row)
        result['variant'] = variant
        result['configuration_sha256'] = self.sha256
        return result
