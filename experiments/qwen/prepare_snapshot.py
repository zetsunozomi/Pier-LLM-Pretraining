#!/usr/bin/env python3
"""Fetch only pinned public model/tokenizer files; run on a login/transfer node."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qwen.preflight import pinned_architecture, preflight, verify_file


def prepare(size, output, *, download=False):
    pin, _ = pinned_architecture(size)
    output = Path(output)
    missing = []
    # Existing wrong bytes are not silently replaced, including in an HF cache.
    for name, identity in pin['files'].items():
        if (output / name).exists():
            verify_file(output / name, identity)
        else:
            missing.append(name)
    if missing and not download:
        raise FileNotFoundError(f'missing pinned snapshot files: {missing}; use --download on a networked login node')
    if missing:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=pin['model'], revision=pin['revision'],
                          allow_patterns=missing, local_dir=str(output), max_workers=2,
                          token=False)
    return preflight(size, 1, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('1.5B', '3B', '7B'), default='3B')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--download', action='store_true', help='explicitly fetch missing public files')
    args = parser.parse_args()
    pin, _ = pinned_architecture(args.model)
    output = args.output_dir or ROOT / 'local/qwen/models' / f'Qwen2.5-{args.model}' / pin['revision']
    report = prepare(args.model, output, download=args.download)
    report['snapshot_path'] = str(output.resolve())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
