"""Synthetic reporting fixtures; these are never GPU measurement evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.joint import qwen_summary


def row(name, arm, repeat, times):
    boundaries = [dict(round=i + 1, warmup=i == 0, eligible=i > 0,
                       outer_seconds=t, cycle_seconds=10 * t, is_transition=i == 1)
                  for i, t in enumerate(times)]
    return dict(case=name, arm=arm, repeat_id=repeat, status='measured',
                budget_feasible=True, run_id=f'{repeat}-{name}', boundaries=boundaries,
                total_outer_seconds=sum(times[1:]), total_cycle_seconds=10 * sum(times[1:]),
                mean_outer_seconds=sum(times[1:]) / 2, tokens_per_second=100 / sum(times[1:]),
                recipe={'data': 'synthetic reporting fixture'},
                hardware=[dict(gpu='fixture', torch='fixture', cuda='fixture', hostname=f'node-{repeat}')])


def campaigns():
    loaded = []
    for repeat in (1, 2, 3):
        rows = [row('joint-s1', 'P', repeat, [100, 4, 3]),
                row('separate-s1', 'P', repeat, [1, 3, 6]),
                row('reference-s16', 'P', repeat, [10, 5, 5]),
                row('native-controls', 'W', repeat, [10, 2, 2])]
        manifest = dict(repeat_id=repeat, sources={'executor': 'fixture'}, scenario='budgets',
                        planned_cases=[dict(name=r['case'], config=None) for r in rows],
                        budget_announcements=[dict(round=1, transition_mib=39, next_phase_mib=35)])
        loaded.append((manifest, rows))
    return loaded


class JointReportingTests(unittest.TestCase):
    def aggregate(self, loaded, **kwargs):
        with patch.object(qwen_summary, 'campaign', side_effect=loaded):
            return qwen_summary.aggregate(list(range(len(loaded))), **kwargs)

    def split_campaigns(self):
        loaded = []
        for repeat in (1, 2, 3):
            for name in ('single-s2', 'pipeline-s2', 'separate-s1', 'joint-s1', 'reference-s2'):
                r = row(name, 'P', repeat, [5, 3, 4])
                r['hardware'][0]['hostname'] = f'node-{repeat}-{name}'
                for boundary in r['boundaries']:
                    boundary.update(peak_allocated_bytes=30 * 2**30, peak_reserved_bytes=31 * 2**30)
                m = dict(repeat_id=repeat, sources={'executor': 'fixture'}, scenario='fixed',
                         slurm_job_id=f'{repeat}-{name}', planned_cases=[dict(name=name, config=None)],
                         budget_announcements=[dict(round=1, transition_mib=39, next_phase_mib=35)])
                loaded.append((m, [r]))
        return loaded

    def test_split_allocations_keep_repeat_counts_memory_and_unpaired_labels(self):
        loaded = self.split_campaigns()
        summary = self.aggregate(loaded, split_allocations=True)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['missing_table_vi_rows'], [])
        self.assertEqual(len(summary['by_case']), 5)
        self.assertTrue(all(r['independent_launches'] == 3 for r in summary['by_case']))
        self.assertTrue(all(r['max_peak_allocated_gib'] == 30 for r in summary['by_case']))
        self.assertTrue(all(r['median_peak_reserved_gib'] == 31 for r in summary['by_case']))
        self.assertTrue(all('unpaired' in r['comparison_basis'] for r in summary['comparisons']))
        partial = self.aggregate(loaded[:1], split_allocations=True)
        self.assertEqual(partial['status'], 'incomplete')
        self.assertEqual(partial['by_case'][0]['independent_launches'], 1)
        only_joint = self.aggregate([x for x in loaded if x[1][0]['case'] == 'joint-s1'], split_allocations=True)
        self.assertEqual(only_joint['status'], 'incomplete')
        self.assertEqual(only_joint['missing_table_vi_rows'], ['single', 'pipeline', 'separate'])
        self.assertEqual(self.aggregate(loaded[:-1], split_allocations=True)['status'], 'incomplete')

    def test_split_allocations_reject_duplicate_or_changed_case_and_recipe(self):
        for mutation in ('duplicate', 'config', 'source', 'model', 'software'):
            loaded = self.split_campaigns()
            if mutation == 'duplicate':
                loaded.append(copy.deepcopy(loaded[0]))
            elif mutation == 'config':
                loaded[5][0]['planned_cases'][0]['config'] = {'changed': True}
            elif mutation == 'source':
                loaded[1][0]['sources']['executor'] = 'changed'
            elif mutation == 'model':
                loaded[1][1][0]['recipe']['data'] = 'changed'
            else:
                loaded[1][1][0]['hardware'][0]['torch'] = 'changed'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.aggregate(loaded, split_allocations=True)

    def test_payback_and_negative_w_result_preserve_independent_launches(self):
        summary = self.aggregate(campaigns())
        self.assertEqual(summary['status'], 'complete')
        self.assertTrue(all(r['independent_launches'] == 3 for r in summary['by_case']))
        separate = next(r for r in summary['comparisons'] if r['baseline'] == 'separate')
        self.assertEqual([r['cumulative_outer_seconds_saved'] for r in separate['cumulative_savings']], [-1., 2.])
        static = next(r for r in summary['comparisons'] if r['baseline'] == 'best_feasible_static')
        self.assertEqual(static['baseline_case'], 'reference-s16')
        self.assertAlmostEqual(static['cycle_time_reduction_pct'], 30.)
        w = next(r for r in summary['comparisons'] if r['baseline'] == 'W')
        self.assertAlmostEqual(w['outer_latency_reduction_pct'], -75.)
        self.assertEqual(self.aggregate(campaigns()[:2])['status'], 'incomplete')

    def test_incomplete_pairs_and_changed_allocation_are_rejected(self):
        for mutation in ('length', 'eligible', 'hardware', 'budget', 'duplicate'):
            loaded = campaigns()
            if mutation == 'length':
                loaded[0][1][1]['boundaries'].pop()
            elif mutation == 'eligible':
                loaded[0][1][1]['boundaries'][1]['eligible'] = False
            elif mutation == 'hardware':
                loaded[0][1][1]['hardware'][0]['hostname'] = 'different-node'
            elif mutation == 'budget':
                loaded[1][0]['budget_announcements'][0]['next_phase_mib'] = 30
            else:
                loaded[1][1][0]['run_id'] = loaded[0][1][0]['run_id']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.aggregate(loaded)

    def test_control_only_campaign_preserves_budget_and_checks_executed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / 'reference-s16'
            directory.mkdir()
            manifest = dict(plan_only=False, repeat_id=1, sources={'executor': 'fixture'},
                            planned_cases=[dict(name='reference-s16', config=None)], cases=[],
                            budget_announcements=[dict(round=1, next_phase_mib=30, transition_mib=39)])
            path = root / 'campaign.json'
            path.write_text(json.dumps(manifest))
            self.assertEqual(qwen_summary.campaign(path)[1][0]['status'], 'unfinished')
            manifest['cases'] = [dict(name='reference-s16')]
            path.write_text(json.dumps(manifest))
            (directory / 'summary.json').write_text('{}')
            executed = dict(sources=manifest['sources'], config={'joint_recipe': None})
            (directory / 'manifest.json').write_text(json.dumps(executed))
            arm = dict(arm='P', status='measured', id='run-1-P', run_id='fixture',
                       useful_tokens_per_second=1, outer_and_commit_seconds_mean=1,
                       recipe={}, hardware=[])
            captured = []
            def boundary_fixture(output, world, announcements):
                captured.append(copy.deepcopy(announcements))
                return [dict(eligible=True, outer_seconds=1, cycle_seconds=2, budget_feasible=False)]
            with patch.object(qwen_summary, 'collect_n2', return_value={'config': {'world_size': 2}, 'cases': [arm]}), \
                 patch.object(qwen_summary, 'boundaries', side_effect=boundary_fixture):
                rows = qwen_summary.campaign(path)[1]
                self.assertFalse(rows[0]['budget_feasible'])
                self.assertEqual(captured, [manifest['budget_announcements']])
                executed['sources'] = {'executor': 'changed'}
                (directory / 'manifest.json').write_text(json.dumps(executed))
                with self.assertRaises(ValueError):
                    qwen_summary.campaign(path)


if __name__ == '__main__':
    unittest.main()
