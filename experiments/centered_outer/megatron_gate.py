#!/usr/bin/env python3
"""GPU gate using the repository's real MyDDP and mixed-precision optimizer.

Run under torchrun. This is a correctness fixture, not a throughput benchmark,
Qwen run, or validation of the full pretrain_gpt training/checkpoint pipeline.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import socket
import sys
import traceback
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist


def same(actual, expected, name):
    a = actual.detach().contiguous().reshape(-1).view(torch.uint8)
    b = expected.detach().to(actual.device).contiguous().reshape(-1).view(torch.uint8)
    if not torch.equal(a, b):
        raise AssertionError(f'{name}: bit mismatch')


def wrapper(inner_average, average_in_collective=False):
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from megatron.core.distributed.my_distributed_data_parallel import MyDistributedDataParallel
    from megatron.core.transformer.transformer_config import TransformerConfig
    config = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1,
                               bf16=True, params_dtype=torch.bfloat16)
    model = torch.nn.Linear(4, 4, bias=False, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(.125)
    model.config = config
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True, overlap_grad_reduce=False,
        local_sgd_inner_average=inner_average, average_in_collective=average_in_collective,
    )
    ddp = MyDistributedDataParallel(config, ddp_config, model)
    return model, ddp


def gradient_checks(inner_size):
    from megatron.core import parallel_state as ps
    from megatron.core.outer_sync.normalization import warmup_rescale
    rank, world = dist.get_rank(), dist.get_world_size()
    peers = dist.get_process_group_ranks(ps.get_data_parallel_sub_group())
    assert len(peers) == inner_size
    records = []
    # Test both SUM-with-prescale and native AVG on real MyDDP. Also retain the
    # old divisor as a measured negative control, explicitly not the new recipe.
    for inner_average in (False, True):
        for collective_average in (False, True):
            model, ddp = wrapper(inner_average, collective_average)
            ddp.zero_grad_buffer()
            x = torch.full((1, 4), rank + 1., device='cuda', dtype=torch.bfloat16)
            with ddp.no_sync():
                ddp(x).float().sum().backward()
            ddp(x).float().sum().backward()
            ddp.finish_grad_sync()
            expected_average = 2. * sum(p + 1 for p in peers) / len(peers)
            expected = expected_average if inner_average else expected_average * inner_size / world
            same(model.weight.main_grad, torch.full_like(model.weight.main_grad, expected),
                 'inner gradient scaling')
            grad = model.weight.main_grad.clone()
            grad.mul_(warmup_rescale(world, inner_size, inner_average=inner_average))
            dist.all_reduce(grad, op=dist.ReduceOp.AVG)
            same(grad, torch.full_like(grad, world + 1.), 'lazy-start global average')
            records.append({'inner_average': inner_average, 'collective_average': collective_average,
                            'inner_ranks': peers, 'expected_gradient': expected,
                            'inner_and_warmup_gradient_bits_match': True})
    return records


class StateFixture:
    """Small reproduction of the current sharded/offloaded state callbacks."""
    def __init__(self, master, group, shard, host):
        self.group, self.shard, self.host = group, shard, host
        self.rank, self.k = dist.get_rank(group), dist.get_world_size(group)
        self.reference = self.store(master.clone(), master)
        self.momentum = [self.store(torch.full_like(master, .125), master)]

    def store(self, state, param):
        if self.shard:
            flat = state.view(-1)
            width = (flat.numel() + self.k - 1) // self.k
            start = self.rank * width
            out = torch.zeros(width, dtype=flat.dtype, device=flat.device)
            count = max(0, min(width, flat.numel() - start))
            if count:
                out[:count].copy_(flat[start:start + count])
        else:
            out = state.clone()
        return out.cpu() if self.host else out

    def load(self, state, param):
        state = state.to(param.device)
        if self.shard:
            chunks = [torch.empty_like(state) for _ in range(self.k)]
            dist.all_gather(chunks, state, group=self.group)
            return torch.cat(chunks)[:param.numel()].view_as(param)
        return state

    def reference_for_param(self, param):
        if self.shard:
            return self.load(self.reference, param)
        value = self.load(self.reference, param) if self.rank == 0 else torch.empty_like(param)
        dist.broadcast(value, src=dist.get_process_group_ranks(self.group)[0], group=self.group)
        return value


def outer_checks(outer_group, output):
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from megatron.core.outer_sync.legacy import centered_allreduce_update
    records = []
    for shard in (False, True):
        for host in (False, True):
            model, ddp = wrapper(True)
            inner = torch.optim.AdamW(model.parameters(), lr=1/64, weight_decay=.01)
            opt = Float16OptimizerWithFloat16Params(
                inner, OptimizerConfig(bf16=True, clip_grad=0., params_dtype=torch.bfloat16),
                grad_scaler=None, init_state_fn=lambda _: None,
            )
            master = opt.get_parameters()[0]
            for _ in range(2):
                ddp.zero_grad_buffer()
                opt.zero_grad()
                x = torch.full((1, 4), dist.get_rank() + 1., device='cuda', dtype=torch.bfloat16)
                ddp(x).float().sum().backward()
                ddp.finish_grad_sync()
                success, _, _ = opt.step()
                assert success
                same(model.weight, master.bfloat16(), 'inner optimizer normal commit')
            # Reproduce the missing-copy hazard using a real optimizer/master.
            previous = model.weight.detach().clone()
            with torch.no_grad():
                master.add_(.25)
            assert not torch.equal(model.weight, master.bfloat16()), 'negative control did not diverge'
            probe = torch.ones((1, 4), device='cuda', dtype=torch.bfloat16)
            same(model(probe), torch.nn.functional.linear(probe, previous), 'stale next forward control')
            opt.commit_outer_update()
            same(model.weight, master.bfloat16(), 'explicit corrected commit')

            # Dyadic outer inputs make native reduction exactly representable,
            # isolating commit/state placement from different-tree roundoff.
            with torch.no_grad():
                master.fill_(.125)
            state = StateFixture(master, outer_group, shard, host)
            with torch.no_grad():
                master.add_((dist.get_rank(outer_group) + 1) / 32.)
            opt.commit_outer_update()
            moments_before = copy.deepcopy(inner.state[master])
            centered_allreduce_update(
                opt, state.momentum, reference_for_param=state.reference_for_param,
                load_state=state.load, store_state=state.store, group=outer_group,
                momentum=.5, learning_rate=.25,
            )
            avg_delta = -(dist.get_world_size(outer_group) + 1) / 64.
            expected_m = .5 * .125 + avg_delta
            expected_r = .125 - .25 * (avg_delta + .5 * expected_m)
            same(master, torch.full_like(master, expected_r), 'outer master')
            same(state.load(state.momentum[0], master), torch.full_like(master, expected_m), 'outer M')
            same(model.weight, torch.full_like(model.weight, expected_r), 'outer model bits')
            same(model(probe), torch.full((1, 4), 4 * expected_r, device='cuda', dtype=torch.bfloat16),
                 'actual next model consumer')
            for key, before in moments_before.items():
                same(inner.state[master][key], before, f'inner state preserved: {key}')

            # Local per-learner optimizer serialization, not a claim about the
            # production distributed checkpoint pipeline or outer-round recovery.
            checkpoint = output / f'fixture-rank{dist.get_rank()}-s{int(shard)}-h{int(host)}.pt'
            torch.save({'optimizer': opt.state_dict(), 'model': model.state_dict()}, checkpoint)
            saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
            with torch.no_grad():
                master.add_(1.)
            opt.load_state_dict(saved['optimizer'])
            opt.commit_outer_update()
            same(model.weight, saved['model']['weight'], 'optimizer snapshot restoration')
            same(model(probe), torch.full((1, 4), 4 * expected_r, device='cuda', dtype=torch.bfloat16),
                 'restored next consumer')
            records.append({'outer_shard': shard, 'outer_cpu_offload': host,
                            'missing_copy_negative_control_observed': True,
                            'master_model_next_forward_bits_match': True,
                            'inner_moments_retained': True, 'local_optimizer_restore': True})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inner-dp-size', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', '0'))
    result = {'status': 'failed', 'rank': rank, 'hostname': socket.gethostname(),
              'inner_dp_size': args.inner_dp_size, 'GPU_executed': False,
              'performance_result': False}
    result_path = args.output_dir / f'rank-{rank}.json'
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('this gate requires CUDA; CPU success is not a substitute')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        dist.init_process_group('nccl', timeout=timedelta(seconds=180))
        world = dist.get_world_size()
        if world < 2 or world % args.inner_dp_size:
            raise ValueError('world size must be >=2 and divisible by inner DP size')
        from megatron.core import parallel_state as ps
        ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                                     num_subgroups=world // args.inner_dp_size,
                                     distributed_timeout_minutes=3, create_gloo_process_groups=False)
        outer_group = None
        for offset in range(args.inner_dp_size):
            peers = list(range(offset, world, args.inner_dp_size))
            candidate = dist.new_group(peers, timeout=timedelta(seconds=180))
            if rank in peers:
                outer_group = candidate
        result.update({'GPU_executed': True, 'torch_version': torch.__version__,
                       'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                       'world_size': world, 'outer_ranks': dist.get_process_group_ranks(outer_group),
                       'normalization': gradient_checks(args.inner_dp_size),
                       'outer_update': outer_checks(outer_group, args.output_dir)})
        torch.cuda.synchronize()
        dist.barrier()
        result['status'] = 'passed'
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        # Avoid an error-path collective: torchrun/srun terminate other ranks.
        if result['status'] == 'passed' and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
