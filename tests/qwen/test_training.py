"""Real parser and tokenizer/indexed-data checks for the Qwen training entrypoint."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from experiments.qwen.preflight import pinned_architecture
from experiments.qwen.prepare_data import prepare
from megatron.core.models.qwen.training import (
    training_defaults, training_recompute_config, validate_training_contract,
)
from megatron.training.arguments import parse_args, validate_args
from megatron.core.datasets.indexed_dataset import IndexedDataset
from pretrain_qwen import data_identity


class QwenTrainingTests(unittest.TestCase):
    def test_real_parser_preserves_exact_architecture_and_vocabulary(self):
        with patch.dict(os.environ, WORLD_SIZE='4', RANK='0', CUDA_DEVICE_MAX_CONNECTIONS='1'):
            for size, layouts in (('1.5B', (1, 2)), ('3B', (1, 2)), ('7B', (1, 2, 4))):
                _, arch = pinned_architecture(size)
                for tp in layouts:
                    def extra(parser):
                        parser.set_defaults(**training_defaults(arch), tokenizer_type='HuggingFaceTokenizer')
                        return parser
                    argv = ['pretrain_qwen.py', '--tensor-model-parallel-size', str(tp),
                            '--num-subgroup', str(4 // tp), '--seq-length', '64',
                            '--micro-batch-size', '1', '--train-iters', '11', '--lr', '.0001',
                            '--outer-runtime', 'centered', '--local-sgd-inner-average']
                    with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                        args = validate_args(parse_args(extra))
                    validate_training_contract(args, arch)
                    self.assertEqual(args.padded_vocab_size, arch.vocab)
                    variants = [['--recompute-activations'],
                                ['--recompute-granularity', 'full', '--recompute-method', 'uniform',
                                 '--recompute-num-layers', '1'],
                                ['--recompute-granularity', 'full', '--recompute-method', 'block',
                                 '--recompute-num-layers', str(arch.layers)]]
                    if tp > 1:
                        variants.append(variants[1] + ['--distribute-saved-activations'])
                    for flags in variants:
                        with patch.object(sys, 'argv', argv + flags), contextlib.redirect_stdout(io.StringIO()):
                            recomputed = validate_args(parse_args(extra))
                        validate_training_contract(recomputed, arch)
                        recipe = training_recompute_config(recomputed, arch)
                        self.assertIn(recipe['recompute_granularity'], ('full', 'selective'))
                        self.assertEqual(recipe['distribute_saved_activations'],
                                         '--distribute-saved-activations' in flags)
                    for field, value in (('padded_vocab_size', arch.vocab + 128),
                                         ('rotary_base', 10000.), ('num_query_groups', arch.kv_heads * 2),
                                         ('train_iters', 501), ('sequence_parallel', True)):
                        bad = copy.copy(args)
                        setattr(bad, field, value)
                        with self.assertRaises(ValueError):
                            validate_training_contract(bad, arch)

    def test_real_tokenizer_to_indexed_data_and_receipt_rejection(self):
        raw = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1, 'alpha': 2, 'beta': 3, 'gamma': 4}, unk_token='[UNK]'))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', eos_token='[EOS]')
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, prefix = root / 'source.jsonl', root / 'corpus'
            source.write_text('\n'.join(json.dumps({'text': text}) for text in ('alpha beta', '', 'gamma', 'alpha')) + '\n')
            files = {'tokenizer.json': {'bytes': 10, 'sha256': 'synthetic-tokenizer-identity'}}
            report = prepare(source, prefix, tokenizer, vocab_size=8, revision='test-revision',
                             tokenizer_files=files, max_documents=2, max_tokens=10)
            dataset = IndexedDataset(str(prefix))
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0].tolist(), [2, 3, 1])
            self.assertEqual(dataset[1].tolist(), [4, 1])
            self.assertEqual(report['tokens'], 5)
            args = SimpleNamespace(mock_data=False, data_path=[str(prefix)], data_args_path=None,
                                   per_split_data_args_path=None, seq_length=3)
            snapshot = {'files': files, 'revision': 'test-revision'}
            with patch('pretrain_qwen.get_tokenizer', return_value=SimpleNamespace(_tokenizer=tokenizer)):
                identity = data_identity(args, snapshot)
                self.assertEqual(identity['tokens'], 5)
                wrong = dict(snapshot, revision='different-revision')
                with self.assertRaises(ValueError):
                    data_identity(args, wrong)
                path = Path(str(prefix) + '.bin')
                original = path.read_bytes()
                path.write_bytes(b'X' + original[1:])
                with self.assertRaises(ValueError):
                    data_identity(args, snapshot)
                path.write_bytes(original)
            with self.assertRaises(FileExistsError):
                prepare(source, prefix, tokenizer, vocab_size=8, revision='test-revision', tokenizer_files=files)
            bounded = prepare(source, root / 'bounded', tokenizer, vocab_size=8,
                              revision='test-revision', tokenizer_files=files, max_tokens=4)
            self.assertEqual((bounded['documents'], bounded['tokens']), (1, 3))


if __name__ == '__main__':
    unittest.main()
