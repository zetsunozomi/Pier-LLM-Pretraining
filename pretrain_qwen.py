"""Native Qwen weights-only initialization, then the real Megatron GPT loop.

This entrypoint needs a pinned local snapshot and matching pretokenized data.
It does not download models, pad the output vocabulary, or claim GPU validation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch.distributed as dist

from megatron.training import get_args, get_tokenizer, pretrain
from megatron.core.enums import ModelType
from megatron.core.models.qwen.training import training_defaults, initialize_training_model
from megatron.core.models.qwen.weights import SafeTensorSource, sha256_file
from experiments.qwen.preflight import pinned_architecture, preflight, verify_file
from experiments.qwen.prepare_data import tokenizer_probe
from pretrain_gpt import train_valid_test_datasets_provider, forward_step


def snapshot_identity(report):
    identity = {key: report[key] for key in ('model', 'revision', 'config_sha256', 'architecture', 'files')}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def data_identity(args, report, *, tokenizer=None):
    if args.mock_data:
        raise ValueError('pretrain_qwen requires real text preprocessed with the pinned tokenizer')
    if not args.data_path or len(args.data_path) != 1 or args.data_args_path or args.per_split_data_args_path:
        raise ValueError('initial Qwen recipe requires a single indexed dataset prefix')
    prefix = Path(args.data_path[0])
    manifest = Path(str(prefix) + '.manifest.json')
    data = json.loads(manifest.read_text())
    if data.get('format') not in ('pier-qwen-jsonl-v1', 'pier-qwen-indexed-v2'):
        raise ValueError('run experiments/qwen/prepare_data.py for an auditable tokenizer/data recipe')
    if data.get('format') == 'pier-qwen-indexed-v2':
        if data.get('tokenization', {}).get('input_format') not in ('jsonl', 'parquet'):
            raise ValueError('unknown indexed-data source format')
        provenance = data.get('corpus_provenance')
        if provenance is not None and provenance.get('file') != data.get('input'):
            raise ValueError('indexed-data source identity differs from corpus provenance')
    expected = {name: value for name, value in report['files'].items()
                if name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')}
    if data.get('tokenizer_files') != expected or data.get('model_revision') != report['revision']:
        raise ValueError('data tokenizer differs from the pinned model snapshot')
    if tokenizer is None:
        tokenizer = get_tokenizer()._tokenizer
    if data.get('tokenizer_probes') != tokenizer_probe(tokenizer):
        raise ValueError('tokenizer behavior differs from preprocessing probes')
    for suffix in ('.bin', '.idx'):
        path = Path(str(prefix) + suffix)
        if data['files'][suffix] != {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}:
            raise ValueError(f'dataset bytes differ from preprocessing receipt: {suffix}')
    if data.get('documents', 0) < 1 or data.get('tokens', 0) < args.seq_length + 1:
        raise ValueError('dataset is empty or shorter than one training sequence')
    identity = {'manifest_sha256': sha256_file(manifest), 'files': data['files'],
                'tokens': data['tokens'], 'documents': data['documents'],
                'input': data['input'], 'tokenization': data['tokenization']}
    if data.get('corpus_provenance') is not None:
        identity['corpus_provenance'] = data['corpus_provenance']
    return identity


def make_model_provider(size, snapshot, arch):
    def provider(pre_process=True, post_process=True):
        if not pre_process or not post_process:
            raise ValueError('Qwen provider requires a complete PP1 model')
        args = get_args()
        if Path(args.tokenizer_model).resolve() != snapshot:
            raise ValueError('tokenizer must come from the same pinned snapshot as model weights')
        records = [None]
        if dist.get_rank() == 0:
            try:
                report = preflight(size, args.tensor_model_parallel_size, snapshot)
                records[0] = {'snapshot': report, 'data': data_identity(args, report)}
            except Exception as exc:
                records[0] = {'error': f'{type(exc).__name__}: {exc}'}
        dist.broadcast_object_list(records, src=0)
        record = records[0]
        if 'error' in record:
            raise ValueError(record['error'])
        tokenizer = get_tokenizer()
        ids = list(tokenizer.vocab.values())
        if not ids or min(ids) < 0 or max(ids) >= arch.vocab or not 0 <= tokenizer.eod < arch.vocab:
            raise ValueError('tokenizer IDs exceed the checkpoint embedding vocabulary')
        args.qwen_recipe = {'snapshot_sha256': snapshot_identity(record['snapshot']),
                            'architecture': record['snapshot']['architecture'],
                            'data': record['data'], 'eod': tokenizer.eod,
                            'initialization': 'weights-only warm start'}
        model, receipt = initialize_training_model(args, arch, SafeTensorSource(snapshot))
        receipt['qwen_recipe'] = args.qwen_recipe
        if args.qwen_trace_dir:
            output = Path(args.qwen_trace_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / f'initialization-rank-{dist.get_rank()}.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return model
    return provider


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--qwen-model-size', choices=('1.5B', '3B', '7B'), required=True)
    parser.add_argument('--qwen-snapshot', type=Path, required=True)
    parsed, _ = parser.parse_known_args()
    snapshot = parsed.qwen_snapshot.resolve()
    pin, arch = pinned_architecture(parsed.qwen_model_size)
    verify_file(snapshot / 'config.json', pin['files']['config.json'])

    def extra_args(parser):
        parser.add_argument('--qwen-model-size', choices=('1.5B', '3B', '7B'), required=True)
        parser.add_argument('--qwen-snapshot', type=Path, required=True)
        parser.add_argument('--qwen-trace-dir', type=Path)
        parser.set_defaults(**training_defaults(arch),
                            tokenizer_type='HuggingFaceTokenizer', tokenizer_model=str(snapshot))
        return parser

    train_valid_test_datasets_provider.is_distributed = True
    pretrain(train_valid_test_datasets_provider,
             make_model_provider(parsed.qwen_model_size, snapshot, arch),
             ModelType.encoder_or_decoder, forward_step, extra_args_provider=extra_args)


if __name__ == '__main__':
    main()
