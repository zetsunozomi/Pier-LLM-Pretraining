#!/usr/bin/env python3
"""One fresh-process random-init Qwen depth trial using the existing outer backends."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qwen.capacity_config import architecture, training_args


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def train(manifest, directory, rank):
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.enums import ModelType
    from megatron.core.models.qwen.config import QwenArchitecture
    from megatron.core.models.qwen.model import build_model
    from megatron.core.models.qwen.training import (
        training_defaults, training_recompute_config, validate_training_contract,
    )
    from megatron.training import get_args, get_tokenizer, pretrain
    from pretrain_gpt import forward_step, train_valid_test_datasets_provider

    cfg = manifest['config']
    arch = QwenArchitecture.from_config(architecture(manifest['layers']))
    if arch.unique_parameters() != manifest['parameters']:
        raise ValueError('capacity parameter count differs from Qwen tensor schema')

    def provider(pre_process=True, post_process=True):
        if not pre_process or not post_process:
            raise ValueError('capacity recipe requires PP1')
        args = get_args()
        validate_training_contract(args, arch)
        if args.data_parallel_random_init or not args.mock_data or args.load or args.save:
            raise ValueError('capacity recipe requires identical random initialization and mock data without checkpoints')
        tokenizer = get_tokenizer()
        ids = list(tokenizer.vocab.values())
        if not ids or min(ids) < 0 or max(ids) >= arch.vocab or not 0 <= tokenizer.eod < arch.vocab:
            raise ValueError('tokenizer IDs exceed model vocabulary')
        model = build_model(arch, dtype=torch.bfloat16,
                            use_cpu_initialization=args.use_cpu_initialization,
                            **training_recompute_config(args, arch))
        model.config.deterministic_mode = args.deterministic_mode
        # Count actual model tensors, excluding TP-replicated parameters on TP1.
        local = sum(p.numel() for p in model.parameters()
                    if getattr(p, 'tensor_model_parallel', False)
                    or ps.get_tensor_model_parallel_rank() == 0)
        total = torch.tensor(local, dtype=torch.int64, device='cuda')
        dist.all_reduce(total, group=ps.get_tensor_model_parallel_group())
        if total.item() != arch.unique_parameters():
            raise ValueError(f'actual logical model size {total.item()} differs from schema {arch.unique_parameters()}')
        write(directory / f'initialization-rank-{rank}.json', {
            'rank': rank, 'architecture': asdict(arch), 'parameters': arch.unique_parameters(),
            'initialization': 'random', 'pretrained_weights_loaded': False,
            'seed': args.seed, 'data_kind': 'synthetic_tokens', 'model_size_count_checked': True})
        return model

    def extra_args(parser):
        parser.add_argument('--capacity-layers', type=int, required=True)
        parser.add_argument('--qwen-trace-dir')
        parser.add_argument('--qwen-synthetic-benchmark', action='store_true')
        parser.set_defaults(**training_defaults(arch), tokenizer_type='HuggingFaceTokenizer',
                            tokenizer_model=cfg['snapshot'])
        return parser

    argv = training_args(cfg, manifest['arm'], manifest['layers'], manifest['steps'], directory)
    if argv != manifest['training_argv']:
        raise ValueError('training argv changed since trial was recorded')
    sys.argv = [__file__, *argv]
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(train_valid_test_datasets_provider, provider, ModelType.encoder_or_decoder,
             forward_step, extra_args_provider=extra_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    args = parser.parse_args()
    directory = args.trial.resolve()
    content = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(content)
    rank = int(os.environ['RANK'])
    import torch
    import torch.distributed as dist
    try:
        if int(os.environ['WORLD_SIZE']) != 32 or torch.cuda.device_count() != 4:
            raise ValueError('capacity recipe requires 8 nodes with 4 visible GPUs each')
        device = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(device)
        if torch.cuda.get_device_name(device) != manifest['config']['expected_gpu']:
            raise ValueError('capacity recipe requires NVIDIA A100-SXM4-40GB on every rank')
        if not torch.cuda.is_bf16_supported():
            raise ValueError('native BF16 support required')
        torch.set_num_threads(1)
        write(directory / f'worker-rank-{rank}.json', {
            'rank': rank, 'host': socket.gethostname(), 'world_size': 32,
            'gpu': torch.cuda.get_device_name(device), 'GPU_executed': True,
            'total_memory': torch.cuda.get_device_properties(device).total_memory,
            'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'manifest_sha256': hashlib.sha256(content).hexdigest()})
        train(manifest, directory, rank)
        torch.cuda.synchronize()
        write(directory / f'success-rank-{rank}.json', {'rank': rank, 'status': 'passed'})
    except BaseException as exc:
        # A CPU allocation failure, timeout or generic NCCL error is not GPU OOM.
        cuda_oom = isinstance(exc, torch.cuda.OutOfMemoryError) and 'cuda' in str(exc).lower()
        write(directory / f'failure-rank-{rank}.json', {
            'rank': rank, 'cuda_oom': cuda_oom, 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
