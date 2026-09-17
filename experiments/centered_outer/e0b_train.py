#!/usr/bin/env python3
"""Launch the actual pretrain_gpt entry point and retain early startup failures."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

from e0b_config import CASES, training_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    directory = output / f'case-{args.case}'
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['RANK'])
    sys.path.insert(0, str(root))
    try:
        import torch
        if not torch.cuda.is_available() or int(os.environ['WORLD_SIZE']) != 4:
            raise RuntimeError('E0b requires exactly four real CUDA workers')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        argv = training_args(args.case, output)
        (directory / f'launch-rank-{rank}.json').write_text(json.dumps({
            'case': args.case, 'rank': rank, 'entrypoint': 'pretrain_gpt.py', 'argv': argv,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(), 'GPU_executed': True,
        }, indent=2) + '\n')
        sys.argv = [str(root / 'pretrain_gpt.py'), *argv]
        runpy.run_path(str(root / 'pretrain_gpt.py'), run_name='__main__')
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            return
        (directory / f'failure-rank-{rank}.json').write_text(json.dumps({
            'case': args.case, 'rank': rank, 'status': 'failed',
            'error': repr(exc), 'traceback': traceback.format_exc(),
        }, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
