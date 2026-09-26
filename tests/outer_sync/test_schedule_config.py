"""Opt-in scheduling must not alter native baselines or historical recipes."""

from types import SimpleNamespace
import unittest

from experiments.qwen.n2_config import cases, configuration, training_args
from megatron.core.outer_sync.checkpoint import recipe
from megatron.core.outer_sync.runtime import validate_args


class ScheduleConfigTests(unittest.TestCase):
    def test_n2_only_changes_pier_schedule(self):
        original = configuration({'PIER_N2_ARMS': 'G,O,OS,R,W,P'})
        updated = {**original, 'pier_schedule': 'contiguous'}
        for case in cases(original):
            before = training_args(original, case, '/tmp/fixture')
            after = training_args(updated, case, '/tmp/fixture')
            if case['arm'] == 'P':
                index = after.index('--outer-pier-schedule')
                self.assertEqual(after[index + 1], 'contiguous')
                del after[index:index + 2]
            self.assertEqual(before, after)
        historical = {key: value for key, value in original.items() if key != 'pier_schedule'}
        self.assertEqual(training_args(original, cases(original)[-1], '/tmp/fixture'),
                         training_args(historical, cases(historical)[-1], '/tmp/fixture'))

    def test_invalid_schedule_and_wrong_arm_fail(self):
        with self.assertRaises(ValueError):
            configuration({'PIER_N2_PIER_SCHEDULE': 'typo'})
        for runtime, arm in (('legacy', 'pier'), ('centered', 'gather'), ('centered', 'cpu_offload')):
            with self.assertRaises(ValueError):
                validate_args(SimpleNamespace(outer_runtime=runtime, outer_arm=arm,
                                              outer_pier_schedule='contiguous'))

    def test_historical_checkpoint_default_and_explicit_new_schedule(self):
        old = recipe(SimpleNamespace())
        self.assertEqual(old, recipe(SimpleNamespace(outer_pier_schedule='reference')))
        self.assertIsNone(old['outer_pier_schedule'])
        self.assertEqual(recipe(SimpleNamespace(outer_pier_schedule='contiguous'))['outer_pier_schedule'],
                         'contiguous')


if __name__ == '__main__':
    unittest.main()
