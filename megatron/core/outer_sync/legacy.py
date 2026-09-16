"""The existing centered Nesterov update with an explicit model commit.

This is the development AllReduce baseline, not the paper's tuned G baseline.
State placement remains supplied by the caller, so all four existing combinations
of outer sharding and CPU storage use the same update and commit.
"""

import torch
import torch.distributed as dist


@torch.no_grad()
def centered_allreduce_update(
    optimizer, momentum_buffer, *, reference_for_param, load_state, store_state,
    group, momentum, learning_rate,
):
    """Update masters and commit before any following model consumer.

    The callback order and separate FP32 operations preserve the original
    development path. NCCL's reduction tree is not the prescribed exact tree.
    Inner optimizer moments and step counters are intentionally untouched.
    """
    optimizer.validate_outer_update_support()
    params = [p for pg in optimizer.param_groups for p in pg['params']]
    if len(params) != len(momentum_buffer):
        raise ValueError('outer momentum count must match optimizer parameters')
    if any(p.dtype != torch.float32 for p in params):
        raise ValueError('outer arithmetic requires owned FP32 master parameters')
    for index, param in enumerate(params):
        reference = reference_for_param(param)
        delta = reference - param.detach()
        dist.all_reduce(delta, op=dist.ReduceOp.AVG, group=group)
        state = load_state(momentum_buffer[index], param)
        state.mul_(momentum).add_(delta)
        momentum_buffer[index] = store_state(state, param)
        param.copy_(reference - learning_rate * (delta + momentum * state))
    optimizer.commit_outer_update()
