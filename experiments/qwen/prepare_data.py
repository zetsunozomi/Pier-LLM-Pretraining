#!/usr/bin/env python3
"""Convert local JSONL/Parquet text to a hashed, bounded Qwen indexed dataset."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qwen.preflight import pinned_architecture, verify_file
from megatron.core.models.qwen.weights import sha256_file

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')
PROBES = ('A small training test.', '中文与 English mixed.', '  spaces\nnew line\t12345', '<|endoftext|>')


def tokenizer_probe(tokenizer):
    return [tokenizer.encode(text, add_special_tokens=False) for text in PROBES]


def text_rows(input_path, input_format, text_key):
    if input_format == 'jsonl':
        with Path(input_path).open(encoding='utf-8') as stream:
            for index, line in enumerate(stream, 1):
                yield index, json.loads(line)[text_key]
    elif input_format == 'parquet':
        import pyarrow.parquet as pq
        # Row batches bound Python text materialization. Arrow's own decoding
        # buffers are additional; this is not a total-RSS bound.
        with pq.ParquetFile(input_path) as source:
            if text_key not in source.schema_arrow.names:
                raise ValueError(f'Parquet lacks text column {text_key!r}')
            index = 0
            for batch in source.iter_batches(batch_size=256, columns=[text_key], use_threads=False):
                for text in batch.column(0).to_pylist():
                    index += 1
                    yield index, text
    else:
        raise ValueError('input format must be jsonl or parquet')


def prepare(input_path, output_prefix, tokenizer, *, vocab_size, revision, tokenizer_files,
            text_key='text', max_documents=100000, max_tokens=20000000,
            input_format='jsonl', provenance=None):
    import numpy as np
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    input_path, output_prefix = Path(input_path), Path(output_prefix)
    destinations = {suffix: Path(str(output_prefix) + suffix) for suffix in ('.bin', '.idx', '.manifest.json')}
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError('refusing to overwrite an existing dataset or receipt')
    if max_documents < 1 or max_tokens < 2:
        raise ValueError('positive document count and at least two tokens required')
    if input_format not in ('jsonl', 'parquet'):
        raise ValueError('input format must be jsonl or parquet')
    eod = tokenizer.eos_token_id
    if not isinstance(eod, int) or not 0 <= eod < vocab_size:
        raise ValueError('invalid tokenizer end-of-document ID')
    source = {'bytes': input_path.stat().st_size, 'sha256': sha256_file(input_path)}
    if provenance is not None and provenance.get('file') != source:
        raise ValueError('source bytes differ from declared corpus provenance')
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.qwen-data-', dir=output_prefix.parent) as temporary:
        temporary = Path(temporary)
        builder = IndexedDatasetBuilder(str(temporary / 'data.bin'), dtype=np.int32)
        documents = tokens = lines_read = 0
        try:
            rows = text_rows(input_path, input_format, text_key)
            try:
                for row_number, text in rows:
                    lines_read = row_number
                    if not isinstance(text, str):
                        raise ValueError(f'row {row_number}: expected a text string')
                    if not text.strip():
                        continue
                    ids = tokenizer.encode(text, add_special_tokens=False) + [eod]
                    if min(ids) < 0 or max(ids) >= vocab_size:
                        raise ValueError(f'row {row_number}: token outside exact model vocabulary')
                    if tokens + len(ids) > max_tokens:
                        break  # Keep a source-order prefix of whole documents.
                    builder.add_document(ids, [len(ids)])
                    documents += 1
                    tokens += len(ids)
                    if documents >= max_documents:
                        break
            finally:
                rows.close()
            if not documents:
                raise ValueError('no documents fit the requested data budget')
            builder.finalize(str(temporary / 'data.idx'))
        finally:
            builder.data_file.close()
        if {'bytes': input_path.stat().st_size, 'sha256': sha256_file(input_path)} != source:
            raise ValueError('source text file changed during preprocessing')
        files = {suffix: {'bytes': (temporary / ('data' + suffix)).stat().st_size,
                          'sha256': sha256_file(temporary / ('data' + suffix))}
                 for suffix in ('.bin', '.idx')}
        report = {'format': 'pier-qwen-indexed-v2', 'input': source, 'model_revision': revision,
                  'corpus_provenance': provenance,
                  'tokenizer_files': tokenizer_files, 'tokenizer_probes': tokenizer_probe(tokenizer),
                  'documents': documents, 'tokens': tokens, 'lines_read': lines_read,
                  'tokenization': {'text_key': text_key, 'add_special_tokens': False,
                                   'input_format': input_format,
                                   'append_eod': eod, 'order': 'source-order whole-document prefix',
                                   'max_documents': max_documents, 'max_tokens': max_tokens,
                                   'transformers': importlib.metadata.version('transformers'),
                                   'tokenizers': importlib.metadata.version('tokenizers'),
                                   'pyarrow': importlib.metadata.version('pyarrow') if input_format == 'parquet' else None},
                  'files': files, 'GPU_executed': False, 'performance_result': False}
        (temporary / 'data.manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        for suffix, destination in destinations.items():
            os.replace(temporary / ('data' + suffix), destination)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('1.5B', '3B', '7B'), required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-jsonl', type=Path)
    inputs.add_argument('--input-parquet', type=Path)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--text-key', default='text')
    parser.add_argument('--max-documents', type=int, default=100000)
    parser.add_argument('--max-tokens', type=int, default=20000000)
    args = parser.parse_args()
    pin, arch = pinned_architecture(args.model)
    verify_file(args.snapshot / 'config.json', pin['files']['config.json'])
    files = {name: verify_file(args.snapshot / name, pin['files'][name]) for name in TOKENIZER_FILES}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    report = prepare(args.input_jsonl or args.input_parquet, args.output_prefix, tokenizer, vocab_size=arch.vocab,
                     revision=pin['revision'], tokenizer_files=files, text_key=args.text_key,
                     max_documents=args.max_documents, max_tokens=args.max_tokens,
                     input_format='jsonl' if args.input_jsonl else 'parquet')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
