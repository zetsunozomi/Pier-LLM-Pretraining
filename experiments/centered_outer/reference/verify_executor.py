"""Distributed correctness checks. This script reports no performance result.

Local: GLOO_SOCKET_IFNAME=lo0 python verify_executor.py --world-size 4
External GPU validation: torchrun ... verify_executor.py --device cuda
The harness allocates full oracles outside the executor memory accounting.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from executor import CenteredExecutor
from spec import resident_reference, byte_model


def storage_key(tensor):
    return str(tensor.device), tensor.untyped_storage().data_ptr()


class StorageGuard(TorchDispatchMode):
    """Reject new Python-visible tensor storage during executor.step().

    Communication-library and allocator-internal buffers are outside this
    guard; they still need independent device-level accounting on GPUs.
    """
    def __init__(self, tensors):
        super().__init__()
        self.allowed = {storage_key(t) for t in tensors}
        self.operations = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        for item in tree_leaves(output):
            if isinstance(item, torch.Tensor) and item.numel() and storage_key(item) not in self.allowed:
                raise AssertionError(f'new visible tensor storage in {func}')
        self.operations += 1
        return output


def same(actual, expected, label):
    a = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    b = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    if not torch.equal(a, b):
        raise AssertionError(f'{label}: bit mismatch')


def from_numpy(array, device, *, pin=False):
    tensor = torch.from_numpy(array.copy())
    if pin:
        tensor = tensor.pin_memory()
    return tensor.to(device)


def initial_states(r, m, k, rank, s, device, state_tier):
    g = k // s
    a, b = divmod(rank, s)
    width = (len(r) + k - 1) // k
    rp = np.pad(r, (0, width * k - len(r)))
    mpad = np.pad(m, (0, width * k - len(m)))
    state_device = torch.device('cpu') if state_tier == 'host' else device
    pin = state_tier == 'host' and device.type == 'cuda'
    ref = from_numpy(rp[b * g * width:(b + 1) * g * width].reshape(g, width), state_device, pin=pin)
    mom = from_numpy(mpad[(b * g + a) * width:(b * g + a + 1) * width], state_device, pin=pin)
    return ref, mom


def state_check(engine, expected):
    width = engine.width
    rp = np.pad(expected['reference'], (0, width * engine.k - engine.n))
    mpad = np.pad(expected['momentum'], (0, width * engine.k - engine.n))
    ref = rp[engine.b * engine.g * width:(engine.b + 1) * engine.g * width].reshape(engine.g, width)
    lo = (engine.b * engine.g + engine.a) * width
    same(engine.reference, torch.from_numpy(ref), 'reference shard and replicas')
    same(engine.momentum, torch.from_numpy(mpad[lo:lo + width]), 'momentum shard')
    same(engine.master, torch.from_numpy(expected['reference']), 'master')
    if engine.model is not None:
        same(engine.model, torch.from_numpy(expected['reference']).to(engine.model.dtype), 'model commit')


def payload_check(engine, record):
    cost = byte_model(engine.k, engine.s, engine.k * engine.width * 4)
    expected = [cost[key] for key in ('raw_a2a_sent_per_rank', 'upper_rs_sent_per_rank',
                                    'upper_ag_sent_per_rank', 'local_ag_sent_per_rank')]
    assert record['sent_tensor_bytes_by_phase'] == expected


def operator_checks(rank, k, device, schedule='reference'):
    records = []
    tiers = ('device', 'host') if device.type == 'cuda' else ('device',)
    for s in [1 << i for i in range(k.bit_length())]:
        for n in (1, 5, 17, 64):
            for capacity in (1, 7):
                for tier in tiers:
                    rng = np.random.default_rng(110 + n)
                    r = rng.normal(0, .2, n).astype(np.float32)
                    m = rng.normal(0, .01, n).astype(np.float32)
                    master = from_numpy(r, device)
                    model = master.to(torch.bfloat16)
                    ref, mom = initial_states(r, m, k, rank, s, device, tier)
                    engine = CenteredExecutor(master, ref, mom, cohort=s, tile_elements=capacity,
                                              model=model, schedule=schedule)
                    allocation = engine.allocation_bytes()
                    pointers = [storage_key(x) for x in engine.buffers.values()]
                    total_guard_ops = 0
                    for round_id, scale in enumerate((1e-2, 1e-4, 1e-7)):
                        w = np.add(r[None, :], rng.normal(0, scale, (k, n)).astype(np.float32), dtype=np.float32)
                        if n == 5 and round_id == 2:
                            # Alternating large/small leaves expose reassociation;
                            # a signed-zero column covers ordered copy/arithmetic.
                            pattern = np.array([2**24, 1, -2**24, 1], dtype=np.float32)
                            for learner in range(k):
                                w[learner, 0] = pattern[learner % 4]
                                w[learner, 1] = np.float32(-0.0 if learner % 2 == 0 else 0.0)
                        master.copy_(from_numpy(w[rank], device))
                        expected = resident_reference(r, m, w)
                        # Stress nonuniform entry, without pretending to test
                        # CUDA in-flight copy races on this CPU host.
                        if rank == k - 1 and n == 17 and round_id == 1:
                            time.sleep(.01)
                        with StorageGuard(engine.storage_tensors()) as guard:
                            message_record = engine.step()
                        total_guard_ops += guard.operations
                        assert pointers == [storage_key(x) for x in engine.buffers.values()]
                        state_check(engine, expected)
                        payload_check(engine, message_record)
                        r, m = expected['reference'], expected['momentum']
                    records.append({'rank': rank, 'K': k, 's': s, 'N': n,
                                    'tile_elements_per_momentum_owner': capacity, 'state_tier': tier,
                                    'outer_boundaries': 3, 'bitwise_states_and_model': True,
                                    'fixed_workspace_storages': True, 'guarded_operations': total_guard_ops,
                                    'allocation': allocation, 'payload': message_record})
    return records


def model_loss(model, inputs, target):
    hidden = torch.tanh(F.linear(inputs, model[:12].view(4, 3), model[12:16]))
    output = F.linear(hidden, model[16:].view(1, 4)).view(-1).float()
    return F.mse_loss(output, target)


def training_checks(rank, k, device, schedule='reference'):
    records = []
    for s in [1 << i for i in range(k.bit_length())]:
        rng = np.random.default_rng(775)
        r = rng.normal(0, .02, 20).astype(np.float32)
        m = rng.normal(0, .001, 20).astype(np.float32)
        master = torch.nn.Parameter(from_numpy(r, device))
        oracle_master = torch.nn.Parameter(master.detach().clone())
        model = torch.nn.Parameter(master.detach().to(torch.bfloat16))
        oracle_model = torch.nn.Parameter(model.detach().clone())
        opt = torch.optim.AdamW([master], lr=1e-3, betas=(.9, .95), weight_decay=.1,
                               foreach=False, fused=False)
        oracle_opt = torch.optim.AdamW([oracle_master], lr=1e-3, betas=(.9, .95), weight_decay=.1,
                                      foreach=False, fused=False)
        ref, mom = initial_states(r, m, k, rank, s, device, 'device')
        engine = CenteredExecutor(master, ref, mom, cohort=s, tile_elements=2, model=model, schedule=schedule)
        generator = torch.Generator().manual_seed(304 + rank)
        restored = False
        for step in range(7):
            x = torch.randn(5, 3, generator=generator).to(device=device, dtype=torch.bfloat16)
            target = torch.randn(5, generator=generator).to(device)
            losses = []
            for p, net, optimizer in ((master, model, opt), (oracle_master, oracle_model, oracle_opt)):
                optimizer.zero_grad(set_to_none=True)
                net.grad = None
                loss = model_loss(net, x, target)
                losses.append(loss.detach())
                loss.backward()
                p.grad = net.grad.detach().float()
                optimizer.step()
                with torch.no_grad():
                    net.copy_(p)
            same(losses[0], losses[1], 'next-forward loss')
            same(master, oracle_master, 'inner update')
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                same(opt.state[master][key], oracle_opt.state[oracle_master][key], 'retained inner ' + key)
            if (step + 1) % 2 == 0:
                leaves = [torch.empty_like(oracle_master) for _ in range(k)]
                dist.all_gather(leaves, oracle_master.detach())
                w = np.stack([x.detach().cpu().numpy() for x in leaves])
                expected = resident_reference(r, m, w)
                with StorageGuard(engine.storage_tensors()):
                    record = engine.step()
                state_check(engine, expected)
                payload_check(engine, record)
                with torch.no_grad():
                    oracle_master.copy_(from_numpy(expected['reference'], device))
                    oracle_model.copy_(oracle_master)
                r, m = expected['reference'], expected['momentum']
                if step == 3:
                    # Reconstruct the executor at a drained boundary with only
                    # owned outer state; keep the actual local inner optimizer.
                    engine = CenteredExecutor(master, engine.reference.clone(), engine.momentum.clone(),
                                              cohort=s, tile_elements=2, model=model, schedule=schedule)
                    restored = True
        records.append({'rank': rank, 'K': k, 's': s, 'inner_steps': 7, 'sync_period': 2,
                        'outer_boundaries': 3, 'next_forward_and_inner_moments_bitwise': True,
                        'drained_boundary_executor_restore': restored,
                        'model': '20-parameter BF16 two-layer toy network, FP32 master and AdamW'})
    return records


def rejection_checks(rank, k, device):
    r = np.zeros(20, dtype=np.float32)
    master = from_numpy(r, device)
    ref, mom = initial_states(r, r, k, rank, k, device, 'device')
    rejected = 0
    fixtures = [dict(cohort=3, reference=ref, momentum=mom),
                dict(cohort=k, reference=master[:ref.numel()], momentum=mom),
                dict(cohort=k, reference=ref, momentum=mom, model=master.to(torch.int32))]
    for kwargs in fixtures:
        try:
            CenteredExecutor(master, tile_elements=2, **kwargs)
        except ValueError:
            rejected += 1
    assert rejected == len(fixtures)
    return rejected


def worker(rank, k, init_method, device_kind, output, schedule='reference'):
    torch.set_num_threads(1)
    if device_kind == 'cuda':
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        device = torch.device('cpu')
    torch.use_deterministic_algorithms(True)
    backend = 'nccl' if device.type == 'cuda' else 'gloo'
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=k,
                            timeout=timedelta(seconds=120))
    try:
        # Bind the first NCCL collective to the device selected by LOCAL_RANK.
        dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
        local = {'operator': operator_checks(rank, k, device, schedule),
                 'training': training_checks(rank, k, device, schedule),
                 'rejected_unsafe_configs': rejection_checks(rank, k, device)}
        gathered = [None] * k if rank == 0 else None
        dist.gather_object(local, gathered, dst=0)
        if rank == 0:
            records = {key: [r for rank_record in gathered for r in rank_record[key]]
                       for key in ('operator', 'training')}
            report = {'status': 'passed', 'backend': backend, 'device': device_kind,
                      'schedule': schedule,
                      'production_executor_sha256': hashlib.sha256(
                          (Path(__file__).resolve().parents[3] / 'megatron/core/outer_sync/executor.py').read_bytes()).hexdigest(),
                      'torch_version': torch.__version__, 'ranks': k, 'records': records,
                      'actual_distributed_transport': k > 1,
                      'rejected_unsafe_configs_per_rank': [x['rejected_unsafe_configs'] for x in gathered],
                      'GPU_executed': device_kind == 'cuda', 'performance_result': False,
                      'memory_claim': 'explicit executor tensor storages only; library/allocator excluded',
                      'not_tested': ['production Megatron integration', 'multi-slot pipeline',
                                     'interrupted-round recovery', 'network/device fault recovery'],
                      'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                        for name in ('executor.py', 'verify_executor.py')}}
            Path(output).write_text(json.dumps(report, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--schedule', choices=('reference', 'contiguous'), default='reference')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    k = int(os.environ.get('WORLD_SIZE', args.world_size))
    output = (args.output or Path(__file__).with_name(f'executor_{args.device}_{k}.json')).resolve()
    if 'RANK' in os.environ:
        worker(int(os.environ['RANK']), k, 'env://', args.device, str(output), args.schedule)
    else:
        with tempfile.TemporaryDirectory(prefix='pier-executor-') as task_tmp:
            init_method = 'file://' + str(Path(task_tmp) / 'rendezvous')
            mp.spawn(worker, args=(k, init_method, args.device, str(output), args.schedule), nprocs=k, join=True)
    if int(os.environ.get('RANK', '0')) == 0:
        report = json.loads(output.read_text())
        print(json.dumps({'status': report['status'], 'device': report['device'], 'ranks': k,
                          'operator_configurations': len(report['records']['operator']) // k,
                          'training_configurations': len(report['records']['training']) // k,
                          'performance_result': False}, indent=2))
