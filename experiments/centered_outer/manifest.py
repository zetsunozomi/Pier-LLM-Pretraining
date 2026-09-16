#!/usr/bin/env python3
"""Capture source identity and read-only node preflight without model downloads."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def command(argv):
    try:
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=20)
        return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'unavailable': str(exc)}


def source_hashes():
    # Include the Megatron dependency tree, not only the files modified in this round.
    paths = [*ROOT.glob('megatron/**/*.py'), *ROOT.glob('experiments/centered_outer/**/*.py'),
             *ROOT.glob('experiments/centered_outer/*.sh'),
             *ROOT.glob('experiments/centered_outer/*.sbatch')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--node-only', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.node_only:
        manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'purpose': 'E0a correctness gate; no performance measurements',
                    'repo': str(ROOT), 'python': sys.executable,
                    'git_head': command(['git', 'rev-parse', 'HEAD']),
                    'git_status': command(['git', 'status', '--porcelain']),
                    'source_sha256': source_hashes(),
                    'nodes': int(os.environ.get('SLURM_JOB_NUM_NODES', '1')),
                    'gpus_per_node': int(os.environ.get('PIER_GPUS_PER_NODE', '4')),
                    'job_id': os.environ.get('SLURM_JOB_ID'),
                    'not_validated': ['Qwen conversion/training', 'full pretrain_gpt loop',
                                      'successful-step outer schedule and skip/restore integration',
                                      'production per-learner checkpoint', 'distributed optimizer',
                                      'multi-slot CUDA pipeline', 'GPU performance or wire traffic']}
        (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    info = {'hostname': socket.gethostname(), 'platform': platform.platform(),
            'python': sys.version, 'node_id': os.environ.get('SLURM_NODEID', '0'),
            'nvidia_smi': command(['nvidia-smi', '-q']),
            'gpu_topology': command(['nvidia-smi', 'topo', '-m']),
            'environment': {key: os.environ[key] for key in (
                'SLURM_JOB_ID', 'SLURM_NODELIST', 'CUDA_VISIBLE_DEVICES',
                'NCCL_SOCKET_IFNAME', 'NCCL_IB_HCA', 'NCCL_NET', 'CUBLAS_WORKSPACE_CONFIG'
            ) if key in os.environ}}
    try:
        import torch
        info.update({'torch': torch.__version__, 'cuda': torch.version.cuda,
                     'cuda_available': torch.cuda.is_available(),
                     'visible_devices': torch.cuda.device_count()})
        if torch.cuda.is_available():
            info['nccl'] = torch.cuda.nccl.version()
    except Exception as exc:
        info['torch_import_error'] = repr(exc)
    filename = f"node-{info['node_id']}.json"
    (args.output_dir / filename).write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    main()
