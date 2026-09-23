"""Naive unsharded CPU offload, with one blocking AllReduce per parameter.

This is an explicit unoptimized placement control, not the sharded/tiled host
executor. R/M are full pageable CPU replicas per learner. Every parameter loads
R onto the compute device, centers and AllReduces there, copies the displacement
to the CPU, updates CPU Nesterov state, then copies the result back to its master.
No prefetch, bucketing, overlap, pinned state, or owner-state sharding is used.
"""

import math

import torch
import torch.distributed as dist

from .executor import overlaps


class NaiveCPUOffloadExecutor:
    def __init__(self, reference, momentum, *, coordinates, group=None):
        if not dist.is_initialized():
            raise ValueError('initialized outer group required')
        self.group = dist.group.WORLD if group is None else group
        self.k = dist.get_world_size(self.group)
        self.coordinates, self.device = coordinates, coordinates.device
        self.n = coordinates.numel
        for state in (reference, momentum):
            if (state.device.type != 'cpu' or state.dtype != torch.float32
                    or state.numel() != self.n or not state.is_contiguous() or state.is_pinned()):
                raise ValueError('naive offload requires full, contiguous, pageable CPU FP32 R/M replicas')
        if overlaps(reference, momentum) or any(
                overlaps(state, tensor) for state in (reference, momentum)
                for tensor in coordinates.storage_tensors()):
            raise ValueError('outer state must not alias itself or model/master storage')
        self.reference, self.momentum = reference.view(-1), momentum.view(-1)
        self.capacity = max(master.numel() for _, master, _ in coordinates.pairs)

    def allocation_bytes(self):
        largest = self.capacity * 4
        return {
            'reference_bytes': self.n * 4, 'momentum_bytes': self.n * 4,
            'state_layout': 'replicated', 'state_memory': 'pageable_cpu',
            'workspace_policy': 'whole_parameter_dynamic_allocations',
            'workspace_cap_applies': False,
            'cpu_update_threads': torch.get_num_threads(),
            'max_explicit_gpu_temporary_bytes': 2 * largest if self.device.type == 'cuda' else 0,
            'max_explicit_host_temporary_bytes': (2 if self.device.type == 'cuda' else 4) * largest,
            'parameter_count': sum(master.numel() > 0 for _, master, _ in self.coordinates.pairs),
            'excluded': ['master/model', 'inner optimizer', 'allocator/library storage', 'process memory'],
        }

    def storage_tensors(self):
        return [*self.coordinates.storage_tensors(), self.reference, self.momentum]

    @torch.no_grad()
    def step(self, *, mu=.9, eta=.7):
        if not math.isfinite(mu) or not math.isfinite(eta) or not 0 <= mu < 1 or eta <= 0:
            raise ValueError('finite Nesterov coefficients required')
        calls = 0
        for index, (_, master, _) in enumerate(self.coordinates.pairs):
            start, stop = self.coordinates.offsets[index:index + 2]
            if start == stop:
                continue
            reference = self.reference[start:stop]
            momentum = self.momentum[start:stop]
            # Explicit copies also keep the CPU/Gloo test path free of aliases.
            device_reference = reference.to(self.device, non_blocking=False, copy=True)
            delta = device_reference - master.view(-1)
            if self.k > 1:
                dist.all_reduce(delta, op=dist.ReduceOp.SUM, group=self.group)
                calls += 1
            delta.div_(self.k)
            host_delta = delta.to('cpu', non_blocking=False, copy=True)
            momentum.mul_(mu).add_(host_delta)
            direction = momentum * mu
            direction.add_(host_delta).mul_(eta)
            reference.sub_(direction)
            master.view(-1).copy_(reference, non_blocking=False)
            # Keep the naive parameter-at-a-time lifetime, not a full GPU model copy.
            del device_reference, delta, host_delta, direction
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return {
            'arm': 'cpu_offload', 'implementation': 'naive_unsharded_cpu_v1',
            'numerical_contract': 'native_fp32_centered_allreduce_cpu_nesterov',
            'allreduce_calls': calls,
            'host_to_device_tensor_bytes': self.n * 8 if self.device.type == 'cuda' else 0,
            'device_to_host_tensor_bytes': self.n * 4 if self.device.type == 'cuda' else 0,
            'transfer_accounting': 'API tensor bytes per rank; not measured physical link traffic',
            'physical_wire_bytes': None,
        }
