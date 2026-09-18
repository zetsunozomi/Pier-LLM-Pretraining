"""Tiled native-collective G/R/W arms for the shared Local-SGD training loop.

G gathers sharded R before centering; R keeps R replicated; W reduces raw W
then recenters at the owner. All use sharded M and one native RS/update/AG.
Native SUM ordering is opaque. W also changes the centering graph. None of
these arms claims the ordered executor's bitwise FP32 tree contract.
"""

import math

import torch
import torch.distributed as dist

from .executor import overlaps, power2

ARMS = ('gather', 'resident', 'recenter')


def workspace_elements_per_tile(k, arm, cohort=1):
    if not power2(k) or not power2(cohort) or k % cohort:
        raise ValueError('nested power-of-two group/cohort required')
    if arm == 'pier':
        return (2 * cohort + 3) * (k // cohort) + 1
    if arm == 'dtensor':
        from .dtensor import workspace_elements
        return workspace_elements(k, cohort)
    if arm not in ARMS:
        raise ValueError(f'unknown outer arm: {arm}')
    return 2 * k + 3 if arm in ('gather', 'resident') else k + 4


def tile_for_budget(k, arm, cohort, width, budget_bytes):
    capacity = int(budget_bytes) // (4 * workspace_elements_per_tile(k, arm, cohort))
    if width < 1 or capacity < 1:
        raise ValueError('workspace budget cannot hold one tile coordinate per owner')
    return min(width, capacity)


class CollectiveExecutor:
    def __init__(self, master, reference, momentum, *, arm, tile_elements,
                 model=None, group=None, coordinates=None):
        if not dist.is_initialized() or arm not in ARMS:
            raise ValueError('initialized distributed group and a G/R/W arm required')
        self.group = dist.group.WORLD if group is None else group
        self.k, self.rank = dist.get_world_size(self.group), dist.get_rank(self.group)
        self.arm = arm
        if not power2(self.k):
            raise ValueError('power-of-two learner count required by the experiment recipe')
        if (coordinates is not None and (master is not None or model is not None)
                or coordinates is None and master is None):
            raise ValueError('use either parameter coordinates or a master/optional model vector')
        self.coordinates = coordinates
        self.device = coordinates.device if coordinates is not None else master.device
        self.n = coordinates.numel if coordinates is not None else master.numel()
        self.width = (self.n + self.k - 1) // self.k
        self.capacity = int(tile_elements)
        if self.n < 1 or self.capacity < 1:
            raise ValueError('nonempty model and positive tile required')
        masters = [p for _, p, _ in coordinates.pairs] if coordinates is not None else [master]
        inputs = coordinates.storage_tensors() if coordinates is not None else masters
        r_size = self.k * self.width if arm == 'resident' else self.width
        if reference.numel() != r_size or momentum.numel() != self.width:
            raise ValueError('R/M sizes differ from native owner layout')
        if any(x.dtype != torch.float32 or not x.is_contiguous()
               for x in (*masters, reference, momentum)):
            raise ValueError('contiguous FP32 masters and R/M required')
        if any(x.device != self.device and x.device.type != 'cpu' for x in (reference, momentum)):
            raise ValueError('state must reside on the master device or CPU')
        if overlaps(reference, momentum) or any(overlaps(p, state) for p in inputs for state in (reference, momentum)):
            raise ValueError('R/M and master/model storage must not alias')
        if model is not None:
            if (model.numel() != self.n or not model.is_contiguous() or model.device != self.device
                    or model.dtype not in (torch.float32, torch.bfloat16)):
                raise ValueError('matching contiguous FP32/BF16 model required')
            if (overlaps(model, reference) or overlaps(model, momentum)
                    or overlaps(model, master) and not (model.data_ptr() == master.data_ptr() and model.dtype == master.dtype)):
                raise ValueError('only exact FP32 master/model aliasing is supported')
        self.master = master.view(-1) if master is not None else None
        self.model = model.view(-1) if model is not None else None
        self.reference = reference.view(-1)
        self.momentum = momentum.view(-1)
        sizes = {'pack': self.k * self.capacity, 'reduced': self.capacity,
                 'momentum_stage': self.capacity, 'direction': self.capacity,
                 'reference_stage': (self.k if arm != 'recenter' else 1) * self.capacity}
        self.buffers = {name: torch.empty(size, dtype=torch.float32, device=self.device)
                        for name, size in sizes.items()}

    def allocation_bytes(self):
        size = sum(x.numel() * x.element_size() for x in self.buffers.values())
        expected = 4 * self.capacity * workspace_elements_per_tile(self.k, self.arm)
        assert size == expected
        return {'workspace_tensor_bytes': size, 'formula_bytes': expected,
                'reference_bytes': self.reference.numel() * 4, 'momentum_bytes': self.momentum.numel() * 4,
                'slot_count': 1, 'arm': self.arm,
                'excluded': ['master/model', 'inner optimizer', 'allocator/library storage',
                             'Python and process metadata']}

    def storage_tensors(self):
        inputs = self.coordinates.storage_tensors() if self.coordinates is not None else [self.master]
        return [*inputs, self.reference, self.momentum, *self.buffers.values()] + ([] if self.model is None else [self.model])

    def _gather(self, output, value):
        if self.k == 1:
            output.copy_(value)
        else:
            dist.all_gather_into_tensor(output, value, group=self.group)

    @torch.no_grad()
    def step(self, *, mu=.9, eta=.7):
        if not math.isfinite(mu) or not math.isfinite(eta) or not 0 <= mu < 1 or eta <= 0:
            raise ValueError('finite outer momentum in [0,1) and positive learning rate required')
        calls = {'reference_allgather': 0, 'reduce_scatter': 0, 'updated_reference_allgather': 0}
        api_input, api_output = dict.fromkeys(calls, 0), dict.fromkeys(calls, 0)
        for offset in range(0, self.width, self.capacity):
            t = min(self.capacity, self.width - offset)
            pack = self.buffers['pack'][:self.k * t].view(self.k, t)
            average = self.buffers['reduced'][:t]
            momentum = self.buffers['momentum_stage'][:t]
            direction = self.buffers['direction'][:t]
            pack.zero_()
            # Read all old local masters before this tile's updated values are installed.
            for owner in range(self.k):
                start = owner * self.width + offset
                count = max(0, min(t, self.n - start))
                if count:
                    if self.coordinates is None:
                        pack[owner, :count].copy_(self.master[start:start + count])
                    else:
                        self.coordinates.read_into(start, pack[owner, :count])
            if self.arm == 'recenter':
                own_reference = self.buffers['reference_stage'][:t]
                own_reference.copy_(self.reference[offset:offset + t])
            else:
                reference = self.buffers['reference_stage'][:self.k * t].view(self.k, t)
                if self.arm == 'gather':
                    average.copy_(self.reference[offset:offset + t])
                    self._gather(reference.view(-1), average)
                    calls['reference_allgather'] += int(self.k > 1)
                    api_input['reference_allgather'] += t * 4 if self.k > 1 else 0
                    api_output['reference_allgather'] += self.k * t * 4 if self.k > 1 else 0
                else:
                    reference.copy_(self.reference.view(self.k, self.width)[:, offset:offset + t])
                own_reference = reference[self.rank]
                torch.sub(reference, pack, out=pack)
            if self.k == 1:
                average.copy_(pack.view(-1))
            else:
                dist.reduce_scatter_tensor(average, pack.view(-1), op=dist.ReduceOp.SUM, group=self.group)
                calls['reduce_scatter'] += 1
                api_input['reduce_scatter'] += self.k * t * 4
                api_output['reduce_scatter'] += t * 4
            average.div_(self.k)
            if self.arm == 'recenter':
                torch.sub(own_reference, average, out=average)
            momentum.copy_(self.momentum[offset:offset + t])
            momentum.mul_(mu).add_(average)
            torch.mul(momentum, mu, out=direction)
            direction.add_(average).mul_(eta)
            torch.sub(own_reference, direction, out=average)
            self.momentum[offset:offset + t].copy_(momentum)
            if self.arm != 'resident':
                self.reference[offset:offset + t].copy_(average)
            self._gather(pack.view(-1), average)
            calls['updated_reference_allgather'] += int(self.k > 1)
            api_input['updated_reference_allgather'] += t * 4 if self.k > 1 else 0
            api_output['updated_reference_allgather'] += self.k * t * 4 if self.k > 1 else 0
            if self.arm == 'resident':
                self.reference.view(self.k, self.width)[:, offset:offset + t].copy_(pack)
            for owner in range(self.k):
                start = owner * self.width + offset
                count = max(0, min(t, self.n - start))
                if count:
                    if self.coordinates is None:
                        self.master[start:start + count].copy_(pack[owner, :count])
                    else:
                        self.coordinates.write_from(start, pack[owner, :count])
                    if self.model is not None:
                        self.model[start:start + count].copy_(self.master[start:start + count])
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        b = (self.k - 1) * self.width * 4
        return {'arm': self.arm, 'numerical_contract': ('native_fp32_raw_weight_recenter' if self.arm == 'recenter'
                                                       else 'native_fp32_centered_sum'),
                'collective_calls': calls, 'api_input_bytes': api_input, 'api_output_bytes': api_output,
                'modeled_payload_bytes': (3 if self.arm == 'gather' else 2) * b,
                'padded_master_bytes': self.k * self.width * 4,
                'physical_wire_bytes': None}
