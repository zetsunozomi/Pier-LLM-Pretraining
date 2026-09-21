#!/usr/bin/env python3
"""Run the shared Qwen training loop once, without a correctness oracle."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qwen.n2_config import training_args


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    manifest_bytes = (args.output_dir / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    case = next(c for c in manifest['cases'] if c['id'] == args.case)
    directory = args.output_dir / case['id']
    rank = int(os.environ['RANK'])
    import torch
    import torch.distributed as dist
    try:
        if int(os.environ['WORLD_SIZE']) != manifest['config']['world_size']:
            raise ValueError('torchrun world differs from N2 configuration')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
            raise RuntimeError('N2 needs four visible CUDA GPUs per node')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('N2 requires native BF16 support')
        torch.set_num_threads(1)
        import safetensors
        import transformers
        argv = training_args(manifest['config'], case, directory)
        write(directory / f'worker-rank-{rank}.json', {
            'rank': rank, 'case': case, 'argv': argv,
            'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'hostname': socket.gethostname(), 'local_rank': local_rank,
            'GPU_executed': True, 'gpu': torch.cuda.get_device_name(local_rank),
            'gpu_total_bytes': torch.cuda.get_device_properties(local_rank).total_memory,
            'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'safetensors': safetensors.__version__,
            'transformers': transformers.__version__,
            'tf32': torch.backends.cuda.matmul.allow_tf32,
            'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'nccl_algo': os.environ.get('NCCL_ALGO', 'automatic')})
        sys.argv = [str(ROOT / 'pretrain_qwen.py'), *argv]
        try:
            runpy.run_path(str(ROOT / 'pretrain_qwen.py'), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        torch.cuda.synchronize()
    except BaseException as exc:
        write(directory / f'failure-rank-{rank}.json', {
            'rank': rank, 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
