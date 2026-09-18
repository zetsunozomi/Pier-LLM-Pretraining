"""Read actual CPU/Gloo recorder output; reject damaged/mixed measurement files.

All timing inputs are controlled clocks and explicitly CPU fixtures. There is
no GPU, workload, tuning or independent-run performance evidence in this test.
"""

import copy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiments.centered_outer.cycle_summary import EvidenceError, parse_json, summarize, validate_reports
from megatron.core.outer_sync.cycle_metrics import CycleMeter
from megatron.core.outer_sync.runtime import StepClock


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=60))
    try:
        current = [0.]
        meter = CycleMeter('cpu', directory, planned_attempts=8, warmup_cycles=1,
                           metadata={'fixture': 'controlled CPU clocks; no training/performance result'},
                           now=lambda: current[0])
        clock = StepClock(2)
        for success, inner_duration in zip((True, False, True, True, True, True, True, True),
                                          (1., 1., 1., 1.5, 1.5, 7.5, 7.5, 1.)):
            meter.before_attempt(clock)
            meter.record_tokens(torch.tensor(100.))
            current[0] += inner_duration * (rank + 1) / 4
            boundary = clock.advance(success)
            if boundary:
                meter.before_outer()
                current[0] += (rank + 1) / 4
            meter.after_attempt(success, boundary, clock)
        meter.finish(clock)
        resumed = CycleMeter('cpu', Path(directory).parent / 'resumed', planned_attempts=6, warmup_cycles=1,
                             metadata={'fixture': 'controlled resumed CPU clocks'}, now=lambda: current[0])
        clock = StepClock(2, attempted=1, successful=1)
        for _ in range(5):
            resumed.before_attempt(clock)
            resumed.record_tokens(torch.tensor(100.))
            current[0] += (rank + 1) / 4
            boundary = clock.advance(True)
            if boundary:
                resumed.before_outer()
                current[0] += (rank + 1) / 8
            resumed.after_attempt(True, boundary, clock)
        resumed.finish(clock)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class CycleSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='pier-cycle-summary-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.directory = cls.root / 'run'
        mp.spawn(worker, args=(f'file://{cls.root}/rendezvous', str(cls.directory)), nprocs=4, join=True)
        cls.reports = [json.loads((cls.directory / f'cycles-rank-{rank}.json').read_text()) for rank in range(4)]

    def test_actual_recorder_global_tokens_rank_maxima_and_one_run_sample(self):
        result = summarize(self.directory, allow_cpu=True)
        self.assertEqual(result['status'], 'raw_cycles_validated')
        self.assertFalse(result['GPU_executed'])
        self.assertFalse(result['performance_result'])
        self.assertFalse(result['independence_validated'])
        self.assertEqual(result['run_samples'], 1)
        self.assertEqual((result['complete_cycles'], result['eligible_cycles']), (3, 2))
        self.assertEqual(result['run_sample']['successful_loss_tokens_global'], 400)
        self.assertEqual(result['run_sample']['sum_slowest_rank_cycle_seconds'], 20.)
        self.assertEqual(result['run_sample']['useful_tokens_per_second'], 20.)
        # mean(50, 12.5) = 31.25 is NOT the full-window throughput 400/(4+16).
        self.assertEqual([row['useful_tokens_per_second'] for row in result['cycles']],
                         [None, 50., 12.5, None])
        self.assertEqual([row['exclusion'] for row in result['cycles']],
                         ['warmup', None, None, 'trailing_partial'])
        self.assertEqual(result['cycles'][0]['processed_loss_tokens_global'], 300)
        self.assertEqual(result['cycles'][0]['successful_loss_tokens_global'], 200)
        for record in result['input_reports']:
            data = (self.directory / record['name']).read_bytes()
            self.assertEqual(record['sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(record['bytes'], len(data))
        with self.assertRaisesRegex(EvidenceError, 'GPU reports'):
            summarize(self.directory)

    def test_rejects_missing_mixed_truncated_and_tampered_cycle_evidence(self):
        mutations = []
        missing = copy.deepcopy(self.reports[:-1])
        mutations.append(('missing rank', missing))
        duplicate = copy.deepcopy(self.reports)
        duplicate[3] = copy.deepcopy(duplicate[2])
        mutations.append(('duplicate rank', duplicate))
        mixed = copy.deepcopy(self.reports)
        mixed[1]['run_id'] = str(uuid.uuid4())
        mutations.append(('mixed run', mixed))
        running = copy.deepcopy(self.reports)
        running[2]['status'] = 'running'
        mutations.append(('unfinished rank', running))
        early = copy.deepcopy(self.reports)
        for report in early:
            report['planned_attempts'] = 9
        mutations.append(('early exit', early))
        truncated = copy.deepcopy(self.reports)
        for report in truncated:
            report['cycles'].pop()
        mutations.append(('lost final partial', truncated))
        gap = copy.deepcopy(self.reports)
        for report in gap:
            report['cycles'].pop(1)
        mutations.append(('lost middle cycle', gap))
        for field, value in (('max_rank_cycle_seconds', 8.), ('useful_tokens_per_second', 25.),
                             ('warmup', True), ('successful_steps', 1), ('attempted_end', 7),
                             ('skipped_loss_tokens_global', 100)):
            damaged = copy.deepcopy(self.reports)
            for report in damaged:
                report['cycles'][1][field] = value
            mutations.append((field, damaged))
        disagreement = copy.deepcopy(self.reports)
        disagreement[1]['cycles'][1]['successful_loss_tokens_global'] *= 4
        mutations.append(('summed replicas', disagreement))
        fakewire = copy.deepcopy(self.reports)
        fakewire[1]['cycles'][1]['physical_wire_bytes'] = 123
        mutations.append(('unmeasured wire bytes', fakewire))
        nan = copy.deepcopy(self.reports)
        nan[3]['cycles'][1]['local_cycle_seconds'] = float('nan')
        mutations.append(('nonfinite time', nan))
        for label, reports in mutations:
            with self.subTest(label=label), self.assertRaises(EvidenceError):
                validate_reports(reports, allow_cpu=True)

    def test_resumed_prefix_does_not_count_as_complete_warmup_or_sample(self):
        result = summarize(self.root / 'resumed', allow_cpu=True)
        self.assertEqual(result['initial_clock']['attempted'], 1)
        self.assertEqual(result['final_clock']['boundaries'], 3)
        self.assertEqual((result['complete_cycles'], result['eligible_cycles']), (2, 1))
        self.assertEqual([row['exclusion'] for row in result['cycles']], ['resumed_prefix', 'warmup', None])
        self.assertEqual(result['run_sample']['useful_tokens_per_second'], 80.)

    def test_allocator_schema_only_does_not_claim_device_level_memory(self):
        # Deliberately synthetic schema fixture; no CUDA execution occurs here.
        reports = copy.deepcopy(self.reports)
        for rank, report in enumerate(reports):
            report['GPU_executed'] = True
            for row in report['cycles']:
                row['torch_peak_allocated_bytes'] = 1024 * (rank + 1)
                row['torch_peak_reserved_bytes'] = 2048 * (rank + 1)
        result = validate_reports(reports)
        self.assertFalse(result['performance_result'])
        self.assertEqual(result['cycles'][1]['max_rank_torch_peak_allocated_bytes'], 4096)
        self.assertEqual(result['cycles'][1]['max_rank_torch_peak_reserved_bytes'], 8192)
        reports[1]['cycles'][1]['torch_peak_reserved_bytes'] = 1
        with self.assertRaisesRegex(EvidenceError, 'reserved peak'):
            validate_reports(reports)

    def test_cli_preserves_inputs_rejects_duplicate_runs_and_does_not_overwrite(self):
        script = Path(__file__).resolve().parents[2] / 'experiments/centered_outer/cycle_summary.py'
        before = {p.name: p.read_bytes() for p in self.directory.glob('*.json')}
        output = self.root / 'collection.json'
        command = [sys.executable, str(script), str(self.directory), '--allow-cpu', '--output', str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = output.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), saved)
        duplicate_output = self.root / 'duplicate.json'
        result = subprocess.run([sys.executable, str(script), str(self.directory), str(self.directory),
                                 '--allow-cpu', '--output', str(duplicate_output)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stderr)
        duplicate = json.loads(duplicate_output.read_text())
        self.assertEqual(duplicate['status'], 'failed')
        self.assertEqual(duplicate['run_samples'], 1)
        self.assertTrue(duplicate['errors'])
        self.assertEqual({p.name: p.read_bytes() for p in self.directory.glob('*.json')}, before)

    def test_strict_json_refuses_duplicate_keys_and_nonfinite_values(self):
        for text in ('{"rank": 0, "rank": 1}', '{"time": NaN}', '{"time": Infinity}'):
            with self.subTest(text=text), self.assertRaises(EvidenceError):
                parse_json(text)


if __name__ == '__main__':
    unittest.main()
