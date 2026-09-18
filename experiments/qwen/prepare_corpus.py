#!/usr/bin/env python3
"""Prepare one pinned public text shard for offline native Qwen training.

Only --download fetches missing source data. Tokenization uses the already
pinned local Qwen tokenizer; model weights are neither needed nor downloaded.
Existing complete outputs are verified and reused; partial/wrong outputs fail.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qwen.preflight import pinned_architecture, verify_file
from experiments.qwen.prepare_data import TOKENIZER_FILES, prepare, tokenizer_probe


def corpus_pin():
    return json.loads(Path(__file__).with_name('corpus_pin.json').read_text())


def fetch_source(pin, directory, *, download=False):
    path = Path(directory) / pin['path']
    if not path.exists():
        if not download:
            raise FileNotFoundError('missing pinned corpus shard; use --download on a networked login node')
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=pin['dataset'], repo_type='dataset', revision=pin['revision'],
                        filename=pin['path'], local_dir=str(directory), token=False)
    verify_file(path, pin['file'])
    return path


def prepare_or_check(source, output_prefix, tokenizer, *, pin, vocab_size, revision,
                     tokenizer_files, max_documents, max_tokens):
    verify_file(source, pin['file'])
    prefix = Path(output_prefix)
    destinations = {suffix: Path(str(prefix) + suffix) for suffix in ('.bin', '.idx', '.manifest.json')}
    if not any(path.exists() for path in destinations.values()):
        return prepare(source, prefix, tokenizer, vocab_size=vocab_size, revision=revision,
                       tokenizer_files=tokenizer_files, max_documents=max_documents, max_tokens=max_tokens,
                       input_format='parquet', provenance=pin)
    if not all(path.is_file() for path in destinations.values()):
        raise ValueError('partial indexed dataset; choose a fresh output prefix rather than overwriting evidence')
    report = json.loads(destinations['.manifest.json'].read_text())
    expected = {'format': 'pier-qwen-indexed-v2', 'input': pin['file'], 'corpus_provenance': pin,
                'model_revision': revision, 'tokenizer_files': tokenizer_files,
                'tokenizer_probes': tokenizer_probe(tokenizer), 'GPU_executed': False, 'performance_result': False}
    expected_tokenization = {'text_key': 'text', 'input_format': 'parquet', 'add_special_tokens': False,
                             'append_eod': tokenizer.eos_token_id, 'order': 'source-order whole-document prefix',
                             'max_documents': max_documents, 'max_tokens': max_tokens,
                             'transformers': importlib.metadata.version('transformers'),
                             'tokenizers': importlib.metadata.version('tokenizers'),
                             'pyarrow': importlib.metadata.version('pyarrow')}
    if any(report.get(key) != value for key, value in expected.items()) or report.get('tokenization') != expected_tokenization:
        raise ValueError('existing dataset recipe differs; choose a fresh output prefix')
    for key, limit in (('documents', max_documents), ('tokens', max_tokens)):
        if type(report.get(key)) is not int or not 1 <= report[key] <= limit:
            raise ValueError('invalid existing dataset document/token count')
    for suffix in ('.bin', '.idx'):
        verify_file(destinations[suffix], report['files'][suffix])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('1.5B', '3B', '7B'), default='3B')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--output-prefix', type=Path)
    parser.add_argument('--max-documents', type=int, default=1000000)
    parser.add_argument('--max-tokens', type=int, default=300000000)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.max_documents < 1 or args.max_tokens < 2:
        parser.error('positive document count and at least two tokens required')
    versions = {'pyarrow': '22.0.0', 'transformers': '4.57.3', 'tokenizers': '0.22.2'}
    if args.download:
        versions['huggingface-hub'] = '0.36.0'
    for package, version in versions.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'corpus preparation requires the repository-pinned {package}=={version}')
    pin = corpus_pin()
    model_pin, arch = pinned_architecture(args.model)
    snapshot = args.snapshot or ROOT / 'local/qwen/models' / f'Qwen2.5-{args.model}' / model_pin['revision']
    verify_file(snapshot / 'config.json', model_pin['files']['config.json'])
    files = {name: verify_file(snapshot / name, model_pin['files'][name]) for name in TOKENIZER_FILES}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    directory = args.source_dir or ROOT / 'local/qwen/data/fineweb-edu' / pin['revision']
    source = fetch_source(pin, directory, download=args.download)
    output = args.output_prefix or (directory / 'indexed' / model_pin['revision']
                                   / f'docs-{args.max_documents}-tokens-{args.max_tokens}' / 'train')
    report = prepare_or_check(source, output, tokenizer, pin=pin, vocab_size=arch.vocab,
                              revision=model_pin['revision'], tokenizer_files=files,
                              max_documents=args.max_documents, max_tokens=args.max_tokens)
    print(json.dumps({'status': 'prepared_and_verified', 'output_prefix': str(output.resolve()),
                      'source_path': str(source.resolve()), 'receipt': report,
                      'GPU_executed': False, 'performance_result': False}, indent=2))


if __name__ == '__main__':
    main()
