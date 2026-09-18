"""Cycle bookkeeping with deterministic clocks and real Gloo agreement checks."""

from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.outer_sync.cycle_metrics import CycleMeter
from megatron.core.outer_sync.runtime import StepClock


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, world_size=4, rank=rank,
                           timeout=timedelta(seconds=60))
    try:
        current = [0.]
        meter = CycleMeter('cpu', Path(directory) / 'whole', warmup_cycles=1,
                           metadata={'fixture': 'synthetic clock; not timing evidence'}, now=lambda: current[0])
        clock = StepClock(2)
        for success in (True, False, True, True, True, True):
            meter.before_attempt(clock)
            # Already global DP token count: should remain 100, not 4 * 100.
            meter.record_tokens(torch.tensor(100.))
            current[0] += 1. + rank * .25
            boundary = clock.advance(success)
            if boundary:
                meter.before_outer()
                current[0] += .5
            meter.after_attempt(success, boundary, clock)
        meter.finish(clock)
        first, second, last = meter.records
        assert first['max_rank_cycle_seconds'] == 5.75
        assert first['processed_loss_tokens_global'] == 300
        assert first['successful_loss_tokens_global'] == 200
        assert first['skipped_loss_tokens_global'] == 100
        assert first['warmup'] and not first['eligible']
        assert second['max_rank_cycle_seconds'] == 4.
        assert second['max_rank_outer_and_commit_seconds'] == .5
        assert second['useful_tokens_per_second'] == 50.
        assert not last['complete_cycle'] and not last['eligible']
        assert last['successful_steps'] == 1 and last['useful_tokens_per_second'] is None
        report = json.loads(meter.path.read_text())
        assert not report['GPU_executed'] and not report['performance_result']
        assert report['eligible_cycles'] == 1 and report['complete_cycles'] == 2

        # Resuming in the middle of a cycle must not count a short fragment as
        # a complete sample or consume a complete warmup cycle.
        resumed = CycleMeter('cpu', Path(directory) / 'resumed', warmup_cycles=1, now=lambda: current[0])
        clock = StepClock(2, attempted=1, successful=1)
        for _ in range(3):
            resumed.before_attempt(clock)
            resumed.record_tokens(torch.tensor(100.))
            current[0] += 1.
            boundary = clock.advance(True)
            if boundary:
                resumed.before_outer()
                current[0] += .5
            resumed.after_attempt(True, boundary, clock)
        resumed.finish(clock)
        assert resumed.complete_count == 1
        assert resumed.records[0]['starts_midcycle'] and not resumed.records[0]['complete_cycle']
        assert resumed.records[1]['warmup']

        for case, count in (('different-ranks', 100. + rank), ('fractional', 1.5), ('nonfinite', float('nan'))):
            bad = CycleMeter('cpu', Path(directory) / case, now=lambda: current[0])
            clock = StepClock(1)
            bad.before_attempt(clock)
            bad.record_tokens(torch.tensor(count))
            current[0] += 1.
            bad.before_outer()
            current[0] += .5
            clock.advance(True)
            try:
                bad.after_attempt(True, True, clock)
            except ValueError as exc:
                assert 'token counts' in str(exc)
            else:
                raise AssertionError(f'invalid token count accepted: {case}')
        dist.barrier()
    finally:
        dist.destroy_process_group()


class CycleMetricsTests(unittest.TestCase):
    def test_skip_warmup_partial_and_global_token_accounting(self):
        with tempfile.TemporaryDirectory(prefix='pier-cycle-metrics-') as name:
            mp.spawn(worker, args=(f'file://{name}/rendezvous', name), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
