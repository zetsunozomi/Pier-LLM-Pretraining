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
    def test_checkpoint_recipe_uses_effective_attention_config(self):
        from megatron.training.arguments import parse_args, validate_args, core_transformer_config_from_args
        from megatron.training.theoretical_memory_usage import compute_weight_and_optimizer_memory
        from megatron.training.training import num_floating_point_operations
        from megatron.core.outer_sync.checkpoint import recipe
        with patch.dict(os.environ, WORLD_SIZE='4', RANK='0', NCCL_ALGO='Ring',
                        CUDA_DEVICE_MAX_CONNECTIONS='1'):
            with patch.object(sys, 'argv', ['pretrain_gpt.py', *training_args('split', Path('/tmp/e0b'))]):
                with contextlib.redirect_stdout(io.StringIO()):
                    args = validate_args(parse_args())
            args.padded_vocab_size = 4224
            original = recipe(args)
            config = core_transformer_config_from_args(args)
            self.assertEqual(original['num_query_groups'], config.num_query_groups)
            # The real reporting helpers mutate the ignored CLI default 1 to 4.
            for estimator in (compute_weight_and_optimizer_memory,
                              lambda a: num_floating_point_operations(a, a.global_batch_size)):
                measured = copy.deepcopy(args)
                self.assertGreater(estimator(measured), 0)
                self.assertEqual(recipe(measured), original)
            # Explicit GQA groups are architecture, so they must remain distinct.
            gqa = copy.deepcopy(args)
            gqa.group_query_attention = True
            for groups in (1, 2):
                gqa.num_query_groups = groups
                self.assertEqual(recipe(gqa)['num_query_groups'],
                                 core_transformer_config_from_args(gqa).num_query_groups)
                self.assertNotEqual(recipe(gqa), original)

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

    def test_native_arm_and_cycle_measurement_argument_contract(self):
        from megatron.training.arguments import parse_args, validate_args
        from megatron.core.outer_sync.runtime import validate_args as validate_outer
        from megatron.core.outer_sync.checkpoint import recipe
        argv = training_args('tp1-s1', Path('/tmp/e0b'))
        index = argv.index('--outer-tile-elements')
        del argv[index:index + 2]
        argv += ['--outer-workspace-mib', '64']
        with patch.dict(os.environ, WORLD_SIZE='4', RANK='0', NCCL_ALGO='Ring', CUDA_DEVICE_MAX_CONNECTIONS='1'):
            with patch.object(sys, 'argv', ['pretrain_gpt.py', *argv]), contextlib.redirect_stdout(io.StringIO()):
                args = validate_args(parse_args())
        args.outer_verify = False
        args.outer_inject_skip_at = []
        args.outer_measure_dir = '/tmp/synthetic-cycle-test'
        args.eval_iters = 0
        args.save = None
        original = recipe(args)
        args.outer_arm = 'pier'
        self.assertEqual(recipe(args), original)
        streamed = copy.copy(args)
        streamed.outer_verify_storage = 'streamed'
        with self.assertRaises(ValueError):
            validate_outer(streamed)
        streamed.outer_verify = True
        streamed.outer_measure_dir = None
        validate_outer(streamed)
        self.assertEqual(recipe(streamed)['outer_verify_storage'], 'streamed')
        streamed.outer_verify_tile_elements = 0
        with self.assertRaises(ValueError):
            validate_outer(streamed)
        for arm in ('gather', 'resident', 'recenter'):
            args.outer_arm = arm
            validate_outer(args)
            self.assertEqual(recipe(args)['outer_arm'], arm)
            self.assertEqual(recipe(args)['native_collective_tile']['workspace_mib'], 64)
            for key, value in (('outer_verify', True), ('outer_cohort_size', 2),
                               ('outer_runtime', 'legacy'),
                               ('eval_iters', 1), ('save', '/tmp/save'), ('profile', True),
                               ('outer_warmup_cycles', -1), ('train_iters', 501),
                               ('outer_workspace_mib', float('nan'))):
                changed = copy.copy(args)
                setattr(changed, key, value)
                with self.assertRaises(ValueError, msg=f'{arm}: {key}'):
                    validate_outer(changed)
        # T uses the whole cohort family, but retains the native-SUM contract.
        with patch.dict(os.environ, WORLD_SIZE='4', RANK='0', NCCL_ALGO='Ring', CUDA_DEVICE_MAX_CONNECTIONS='1'):
            with patch.object(sys, 'argv', ['pretrain_gpt.py', *argv, '--outer-arm', 'dtensor']), \
                 contextlib.redirect_stdout(io.StringIO()):
                dtensor_args = validate_args(parse_args())
        dtensor_args.outer_verify = False
        dtensor_args.outer_inject_skip_at = []
        for cohort in (1, 2, 4):
            dtensor_args.outer_cohort_size = cohort
            validate_outer(dtensor_args)
        dtensor_args.outer_verify = True
        with self.assertRaisesRegex(ValueError, 'Pier fixed-tree'):
            validate_outer(dtensor_args)

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
