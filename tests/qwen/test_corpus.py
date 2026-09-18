"""Real local Parquet/tokenizer/indexed-data checks; no public corpus download."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from experiments.qwen.prepare_corpus import fetch_source, prepare_or_check
from experiments.qwen.prepare_data import prepare
from megatron.core.datasets.indexed_dataset import IndexedDataset
from pretrain_qwen import data_identity


def tokenizer():
    raw = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1, 'alpha': 2, 'beta': 3, '中文': 4}, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', eos_token='[EOS]')


def pin_for(path):
    return {'dataset': 'local-synthetic-fixture', 'revision': 'f' * 40,
            'path': 'part.parquet', 'file': {'bytes': path.stat().st_size,
                                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}


class CorpusTests(unittest.TestCase):
    def test_parquet_and_jsonl_produce_identical_tokens_and_reusable_receipts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'part.parquet'
            texts = ['alpha beta', '', '中文\nalpha', ' beta ']
            pq.write_table(pa.table({'text': texts}), source, row_group_size=2)
            jsonl = root / 'source.jsonl'
            jsonl.write_text(''.join(json.dumps({'text': text}, ensure_ascii=False) + '\n' for text in texts))
            tok, pin = tokenizer(), pin_for(source)
            args = dict(pin=pin, vocab_size=8, revision='test-model', tokenizer_files={'tokenizer.json': 'fixture'},
                        max_documents=100, max_tokens=20)
            prefix = root / 'indexed'
            report = prepare_or_check(source, prefix, tok, **args)
            matching = prepare(jsonl, root / 'jsonl', tok, vocab_size=8, revision='test-model',
                               tokenizer_files=args['tokenizer_files'], max_documents=100, max_tokens=20)
            self.assertEqual(report['files'], matching['files'])
            self.assertEqual((report['documents'], report['tokens']), (3, 8))
            data = IndexedDataset(str(prefix))
            self.assertEqual([row.tolist() for row in data], [[2, 3, 1], [4, 2, 1], [3, 1]])
            before = {suffix: Path(str(prefix) + suffix).read_bytes() for suffix in ('.bin', '.idx', '.manifest.json')}
            self.assertEqual(prepare_or_check(source, prefix, tok, **args), report)
            self.assertEqual(before, {suffix: Path(str(prefix) + suffix).read_bytes() for suffix in before})
            changed = dict(args, max_tokens=21)
            with self.assertRaisesRegex(ValueError, 'recipe differs'):
                prepare_or_check(source, prefix, tok, **changed)
            path = Path(str(prefix) + '.bin')
            path.write_bytes(b'X' + path.read_bytes()[1:])
            with self.assertRaisesRegex(ValueError, 'contents differ'):
                prepare_or_check(source, prefix, tok, **args)

    def test_budget_preserves_document_prefix_and_rejects_bad_rows_and_provenance(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'part.parquet'
            pq.write_table(pa.table({'text': ['alpha', 'alpha beta beta beta', 'beta']}), source)
            tok, pin = tokenizer(), pin_for(source)
            common = dict(vocab_size=8, revision='model', tokenizer_files={}, input_format='parquet')
            report = prepare(source, root / 'budget', tok, max_tokens=4, provenance=pin, **common)
            self.assertEqual((report['documents'], report['tokens'], report['lines_read']), (1, 2, 2))
            self.assertEqual(report['corpus_provenance'], pin)
            wrong = copy.deepcopy(pin)
            wrong['file']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'provenance'):
                prepare(source, root / 'wrong', tok, provenance=wrong, **common)
            pq.write_table(pa.table({'text': ['alpha', None]}), source)
            with self.assertRaisesRegex(ValueError, 'text string'):
                prepare(source, root / 'bad-row', tok, **common)
            self.assertFalse(Path(str(root / 'bad-row') + '.bin').exists())
            pq.write_table(pa.table({'wrong_column': ['alpha']}), source)
            with self.assertRaisesRegex(ValueError, 'lacks text column'):
                prepare(source, root / 'bad-schema', tok, **common)

    def test_download_is_explicit_revision_pinned_and_never_replaces_wrong_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            original = root / 'source.parquet'
            pq.write_table(pa.table({'text': ['alpha']}), original)
            pin, output = pin_for(original), root / 'download'
            def fake_download(**kwargs):
                self.assertEqual(kwargs, {'repo_id': pin['dataset'], 'repo_type': 'dataset',
                                         'revision': pin['revision'], 'filename': pin['path'],
                                         'local_dir': str(output), 'token': False})
                output.mkdir()
                (output / pin['path']).write_bytes(original.read_bytes())
            with patch('huggingface_hub.hf_hub_download', side_effect=fake_download) as download:
                with self.assertRaises(FileNotFoundError):
                    fetch_source(pin, output)
                download.assert_not_called()
                source = fetch_source(pin, output, download=True)
                self.assertEqual(source.read_bytes(), original.read_bytes())
                self.assertEqual(fetch_source(pin, output, download=True), source)
                download.assert_called_once()
                source.write_bytes(b'X' + source.read_bytes()[1:])
                with self.assertRaisesRegex(ValueError, 'contents differ'):
                    fetch_source(pin, output, download=True)
                download.assert_called_once()

    def test_training_accepts_v2_provenance_and_keeps_legacy_v1_support(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'part.parquet'
            pq.write_table(pa.table({'text': ['alpha beta', '中文']}), source)
            prefix, tok = root / 'data', tokenizer()
            files = {'tokenizer.json': {'bytes': 10, 'sha256': 'synthetic-tokenizer'}}
            report = prepare(source, prefix, tok, vocab_size=8, revision='model', tokenizer_files=files,
                             input_format='parquet', provenance=pin_for(source))
            args = SimpleNamespace(mock_data=False, data_path=[str(prefix)], data_args_path=None,
                                   per_split_data_args_path=None, seq_length=3)
            snapshot = {'files': files, 'revision': 'model'}
            manifest = Path(str(prefix) + '.manifest.json')
            with patch('pretrain_qwen.get_tokenizer', return_value=SimpleNamespace(_tokenizer=tok)):
                identity = data_identity(args, snapshot)
                self.assertEqual(identity['corpus_provenance'], report['corpus_provenance'])
                broken = copy.deepcopy(report)
                broken['corpus_provenance']['file']['bytes'] += 1
                manifest.write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    data_identity(args, snapshot)
                # Use an actual JSONL receipt to check the historical v1 path.
                jsonl = root / 'old.jsonl'
                jsonl.write_text('{"text":"alpha beta"}\n')
                old_prefix = root / 'old'
                legacy = prepare(jsonl, old_prefix, tok, vocab_size=8, revision='model', tokenizer_files=files)
                legacy['format'] = 'pier-qwen-jsonl-v1'
                legacy.pop('corpus_provenance')
                legacy['tokenization'].pop('input_format')
                legacy['tokenization'].pop('pyarrow')
                Path(str(old_prefix) + '.manifest.json').write_text(json.dumps(legacy))
                args.data_path = [str(old_prefix)]
                args.seq_length = 2
                self.assertEqual(data_identity(args, snapshot)['tokens'], 3)


if __name__ == '__main__':
    unittest.main()
