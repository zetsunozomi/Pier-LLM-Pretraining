#!/usr/bin/env python3
"""Execute the real pretrain_qwen entrypoint with a manifest-bound fixed case."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qwen.e0c import configure_gpu, write_json
from experiments.qwen.e0c_evidence import source_hashes
from experiments.qwen.e0d_config import CASES, training_args
from megatron.core.models.qwen.weights import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    directory = output / f'case-{args.case}'
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['RANK'])
    import torch.distributed as dist
    try:
        if int(os.environ['WORLD_SIZE']) != 4:
            raise ValueError('E0d requires exactly four CUDA workers')
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['stage'] != 'E0d' or manifest['source_sha256'] != source_hashes():
            raise ValueError('E0d run sources differ from the preflight manifest')
        environment = configure_gpu('E0d')
        argv = training_args(args.case, output, manifest['snapshot_path'], manifest['data_prefix'])
        write_json(directory / f'launch-rank-{rank}.json', {
            'case': args.case, 'rank': rank, 'entrypoint': 'pretrain_qwen.py', 'argv': argv,
            'manifest_sha256': sha256_file(output / 'manifest.json'), 'environment': environment,
            'GPU_executed': True, 'performance_result': False})
        sys.argv = [str(ROOT / 'pretrain_qwen.py'), *argv]
        try:
            runpy.run_path(str(ROOT / 'pretrain_qwen.py'), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        if args.case == 'split' and rank == 0:
            # Preserve the small publication receipt; binary rank states stay
            # on scratch. The collector binds each restore hash to this save.
            receipt = json.loads((output / 'checkpoints/iter_0000005/complete.json').read_text())
            write_json(output / 'checkpoint-manifest.json', receipt)
    except BaseException as exc:
        write_json(directory / f'failure-rank-{rank}.json', {
            'case': args.case, 'rank': rank, 'status': 'failed',
            'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
