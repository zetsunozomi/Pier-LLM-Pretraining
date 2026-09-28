#!/usr/bin/env python3
"""Reject stale, CPU-only, or different-topology correctness receipts."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.joint.common import sources
import torch


def check(path, world, tp):
    gate = json.loads(Path(path).read_text())
    if (gate.get('kind') != 'correctness' or gate.get('status') != 'passed'
            or not gate.get('GPU_executed') or gate.get('world_size') != world
            or gate['config']['tp'] != tp):
        raise ValueError('a passed CUDA gate on this world/TP topology is required')
    current = sources()
    if gate['sources'] != current:
        raise ValueError('sources changed since the CUDA gate; rerun the gate on this checkout')
    if any(row['torch'] != torch.__version__ or row['cuda'] != torch.version.cuda for row in gate['hardware']):
        raise ValueError('PyTorch/CUDA changed since the CUDA gate')
    return gate


def check_training(path):
    gate = json.loads(Path(path).read_text())
    if (gate.get('kind') != 'training_correctness' or gate.get('status') != 'passed'
            or not gate.get('GPU_executed') or gate.get('world_size') != 4
            or not gate.get('comparisons') or any(not row['bitwise_trajectory_equal'] for row in gate['comparisons'])
            or gate.get('torch_versions') != [torch.__version__]
            or gate.get('sources') != sources()):
        raise ValueError('a passed four-GPU real-training/restart gate on this source is required')
    return gate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gate', type=Path)
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--tp', type=int, required=True)
    args = parser.parse_args()
    check(args.gate, args.world, args.tp)
    print('CUDA gate matches this source and topology.')
