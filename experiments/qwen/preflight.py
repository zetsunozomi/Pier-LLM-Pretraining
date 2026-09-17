#!/usr/bin/env python3
"""Read-only Qwen architecture / local snapshot preflight; never launch training."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from megatron.core.models.qwen.config import QwenArchitecture
from megatron.core.models.qwen.weights import SafeTensorSource, parameter_mappings, validate_source

DIRECTORY = Path(__file__).resolve().parent


def pinned_architecture(size):
    pin = json.loads((DIRECTORY / 'pins.json').read_text())['models'][size]
    content = (DIRECTORY / pin['config_filename']).read_bytes()
    if hashlib.sha256(content).hexdigest() != pin['config_sha256']:
        raise ValueError('pinned config hash mismatch')
    arch = QwenArchitecture.from_config(json.loads(content))
    if arch.unique_parameters() != pin['unique_parameters']:
        raise ValueError('parameter schema differs from published tensor metadata')
    return pin, arch


def verify_file(path, expected):
    """Check LFS content SHA256 or regular Git blob identity and retain SHA256."""
    size = path.stat().st_size
    if size != expected['bytes']:
        raise ValueError(f'file size mismatch: {path.name}')
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha256.update(block)
            if not expected.get('lfs_sha256'):
                blob.update(block)
    matches = (sha256.hexdigest() == expected['lfs_sha256'] if expected.get('lfs_sha256')
               else blob.hexdigest() == expected['git_blob_id'])
    if not matches:
        raise ValueError(f'file contents differ from pinned revision: {path.name}')
    return {'bytes': size, 'sha256': sha256.hexdigest()}


def preflight(size, tp, checkpoint=None):
    pin, arch = pinned_architecture(size)
    arch.validate_tp(tp)
    # Norms are replicated within TP, so aggregate local elements need not equal
    # the unique model count. Keep both quantities explicit.
    layouts = [parameter_mappings(arch, tp, rank) for rank in range(tp)]
    report = {'status': 'architecture_checked', 'model': pin['model'], 'revision': pin['revision'],
              'config_sha256': pin['config_sha256'], 'architecture': asdict(arch), 'tp_size': tp,
              'unique_parameters': arch.unique_parameters(),
              'local_parameter_elements': [sum(math.prod(p.shape) for p in layout) for layout in layouts],
              'checkpoint_weight_bytes': sum(value['bytes'] for name, value in pin['files'].items()
                                             if name.endswith('.safetensors')),
              'checkpoint_verified': False, 'tokenizer_files_verified': False,
              'initialization': 'weights-only warm start', 'GPU_executed': False,
              'GPU_conversion_validated': False, 'performance_result': False,
              'ready_for_training': False,
              'remaining': ['GPU conversion/logits/gradient check', 'tokenizer behavior and real-data recipe',
                            'training entry-point integration', 'strong baselines and measured cycles']}
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        # Pin the real bytes, not just a folder name or a matching config.json.
        files = {name: verify_file(checkpoint / name, expected) for name, expected in pin['files'].items()}
        source = SafeTensorSource(checkpoint)
        validate_source(arch, source)
        report.update(status='local_snapshot_checked', checkpoint_verified=True,
                      tokenizer_files_verified=True, files=files)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('1.5B', '3B', '7B'), required=True)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = preflight(args.model, args.tp, args.checkpoint)
    except Exception as exc:
        report = {'status': 'failed', 'error': str(exc), 'GPU_executed': False,
                  'ready_for_training': False, 'performance_result': False}
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
    raise SystemExit(1 if report['status'] == 'failed' else 0)


if __name__ == '__main__':
    main()
