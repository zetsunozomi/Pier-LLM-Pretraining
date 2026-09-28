"""Shared launch identity and rank aggregation for the joint-runtime experiments."""

from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import uuid

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]


def sources():
    from experiments.qwen.n2 import source_identity
    return source_identity()


def initialize(device_name, tp):
    torch.set_num_threads(1)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if device_name == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(seconds=300))
    world = dist.get_world_size()
    if tp < 1 or world % tp:
        raise ValueError('TP must divide world size')
    group = None
    for offset in range(tp):
        peers = list(range(offset, world, tp))
        created = dist.new_group(peers)
        if dist.get_rank() in peers:
            group = created
    dist.barrier()
    return device, group


def gather(value, group=None):
    result = [None] * dist.get_world_size(group)
    dist.all_gather_object(result, value, group=group)
    return result


def drain(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def identity(device):
    return dict(rank=dist.get_rank(), hostname=socket.gethostname(), python=platform.python_version(),
                torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                total_device_bytes=torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None)


def write_report(path, report, *, update=False):
    if dist.get_rank() != 0:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_suffix(path.suffix + '.tmp') if update else path
    with target.open('w' if update else 'x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if update:
        target.replace(path)


def receipt(args, device):
    uid = [str(uuid.uuid4()) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    return dict(format='pier-joint-experiment-v1', launch_id=uid[0],
                independent_launch=True, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                repeat_id=getattr(args, 'repeat_id', 1), GPU_executed=device.type == 'cuda',
                config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                hardware=gather(identity(device)), world_size=dist.get_world_size(), sources=sources())
