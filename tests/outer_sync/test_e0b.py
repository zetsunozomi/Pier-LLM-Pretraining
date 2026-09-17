"""E0b argument/data bounds and report rejection checks; no GPU evidence."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/centered_outer'))
from e0b_config import CASES, training_args
from e0b_summarize import summarize
from manifest import source_hashes


class E0bTests(unittest.TestCase):
    def test_real_parser_and_mock_token_bounds(self):
        from megatron.training.arguments import parse_args, validate_args
        from megatron.core.outer_sync.runtime import validate_args as validate_outer
        from megatron.core.datasets.gpt_dataset import MockGPTLowLevelDataset
        from megatron.training.tokenizer.tokenizer import _NullTokenizer
        with patch.dict(os.environ, WORLD_SIZE='4', RANK='0', NCCL_ALGO='Ring',
                        CUDA_DEVICE_MAX_CONNECTIONS='1'):
            for case in CASES:
                with patch.object(sys, 'argv', ['pretrain_gpt.py', *training_args(case, Path('/tmp/e0b'))]):
                    with contextlib.redirect_stdout(io.StringIO()):
                        args = validate_args(parse_args())
                        validate_outer(args)
                    tokenizer = _NullTokenizer(args.vocab_size)
                    data = MockGPTLowLevelDataset(tokenizer)
                    longest = int(data.sequence_lengths.argmax())
                    self.assertLess(int(data[longest].max()), tokenizer.vocab_size)
                    self.assertEqual(args.train_iters, 11)
                    self.assertFalse(args.apply_rope_fusion)
                    bad = copy.copy(args)
                    bad.rerun_mode = 'validate_results'
                    with self.assertRaises(ValueError):
                        validate_outer(bad)
                    bad = copy.copy(args)
                    bad.use_distributed_optimizer = True
                    with self.assertRaises(ValueError):
                        validate_outer(bad)

    def test_complete_archive_then_corrupt_evidence(self):
        # Only schema fixtures, under a temporary directory removed at exit.
        # Different archive/run paths exercise auditing a cluster Git pull.
        with tempfile.TemporaryDirectory() as name:
            output, run_output = Path(name), Path('/cluster/local/centered_outer/e0b-fixture')
            def write(relative, value):
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            write('manifest.json', dict(nodes=1, gpus_per_node=4, source_sha256=source_hashes()))
            write('node-0.json', dict(cuda_available=True, visible_devices=4))
            for case, config in CASES.items():
                end = config.get('stop', 11)
                events, success = [], 0
                for attempted in range(1, end + 1):
                    skip = attempted == 3
                    success += int(not skip)
                    boundary = not skip and success % 3 == 0
                    events.append(dict(attempted=attempted, successful=success,
                                       boundaries=success // 3, skipped=skip, outer_boundary=boundary,
                                       payload={} if boundary else None,
                                       oracle_states_bitwise=True if boundary else None,
                                       outer_retained_inner_state=True if boundary else None,
                                       skip_retained_optimizer=True if skip else None,
                                       master_sha256='a' * 64, model_sha256='b' * 64,
                                       inner_sha256='c' * 64, loss={'lm loss': 1.0}))
                for rank in range(4):
                    dp = list(range(rank % config['tp'], 4, config['tp']))
                    index, inner = dp.index(rank), config['inner']
                    row = dict(status='partial' if case == 'split' else 'passed', rank=rank,
                               world_size=4, GPU_executed=True, backend='nccl',
                               clock=dict(interval=3, attempted=end, successful=success, boundaries=success // 3),
                               pending_consumer=False, consumer_checks=success // 3,
                               cohort=config['cohort'], state_tier='host' if config.get('host') else 'device',
                               performance_result=False, error=None, restored=bool(config.get('resume')),
                               inner_ranks=dp[index // inner * inner:(index // inner + 1) * inner],
                               outer_ranks=dp[index % inner::inner], events=events)
                    if config.get('resume'):
                        row['restore_evidence'] = dict(iteration=5, consumed_train_samples=40,
                            rank_file_sha256='d' * 64, rank_file_bytes=1000,
                            clock=dict(interval=3, attempted=5, successful=4, boundaries=1))
                    write(f'case-{case}/rank-{rank}.json', row)
                    write(f'case-{case}/launch-rank-{rank}.json', dict(entrypoint='pretrain_gpt.py',
                        case=case, rank=rank, argv=training_args(case, run_output), GPU_executed=True))
            self.assertEqual(summarize(output)['errors'], [])
            path = 'case-resume/rank-2.json'
            original = json.loads((output / path).read_text())
            for field, value in [('restored', False), ('pending_consumer', True), ('inner_ranks', [0]),
                                 ('restore_evidence', {}), ('GPU_executed', False), ('events', [])]:
                with self.subTest(field=field):
                    changed = copy.deepcopy(original)
                    changed[field] = value
                    write(path, changed)
                    self.assertEqual(summarize(output)['status'], 'failed')
            changed = copy.deepcopy(original)
            changed['events'][6]['oracle_states_bitwise'] = False
            write(path, changed)
            self.assertEqual(summarize(output)['status'], 'failed')
            write(path, original)
            self.assertEqual(summarize(output, 23)['status'], 'failed')
            (output / 'case-tp1-s4/rank-3.json').unlink()
            self.assertEqual(summarize(output)['status'], 'failed')

    def test_empty_archive_is_failure(self):
        with tempfile.TemporaryDirectory() as name:
            result = summarize(Path(name))
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['GPU_executed'])


if __name__ == '__main__':
    unittest.main()
