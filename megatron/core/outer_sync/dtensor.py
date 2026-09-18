"""Tiled explicit-layout DTensor backend for the reference-owner family.

The DTensor layout is supplied by the experiment, not found by a planner.
Native SUM order is opaque; this backend does not promise Pier's FP32 tree.
All default-world ranks construct their corresponding meshes in the same order.
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from .coordinates import ParameterCoordinates
from .executor import overlaps, power2


def workspace_elements(k, cohort):
    """Bound caller-held tensor storage; DTensor/collective internals excluded.

    Persistent: pack K, R stage g, partial bank g, M stage 1, direction 1.
    Maximum additional exposed redistribution outputs: K + g + 1.
    Each coefficient counts FP32 elements per momentum-owner tile coordinate.
    """
    if not power2(k) or not power2(cohort) or k % cohort:
        raise ValueError('nested power-of-two learner/cohort sizes required')
    return 2 * k + 3 * (k // cohort) + 3


def make_mesh(group, cohort, device_type):
    """Construct disjoint outer meshes without conflicting global new_group order."""
    peers = dist.get_process_group_ranks(group)
    descriptions = [None] * dist.get_world_size()
    dist.all_gather_object(descriptions, (peers, cohort, device_type))
    groups = sorted({tuple(item[0]) for item in descriptions})
    if sorted(rank for ranks in groups for rank in ranks) != list(range(dist.get_world_size())):
        raise ValueError('outer meshes must partition the default world')
    selected = None
    for ranks in groups:
        _, s, kind = descriptions[ranks[0]]
        if (not power2(len(ranks)) or not power2(s) or len(ranks) % s
                or any(descriptions[r] != (list(ranks), s, kind) for r in ranks)):
            raise ValueError('ranks disagree on outer DTensor mesh')
        # Global rank = peers[a*s+b]; mesh axes are (b,a), so coordinate
        # shards split by b first, then by momentum-owner a.
        rank_map = torch.tensor(ranks, dtype=torch.int64).reshape(-1, s).T.contiguous()
        mesh = DeviceMesh(kind, rank_map, mesh_dim_names=('within', 'across'))
        if dist.get_rank() in ranks:
            selected = mesh
    if selected is None:
        raise ValueError('rank has no outer DTensor mesh')
    return selected


def local_dtensor(tensor, mesh, placements, shape):
    stride = torch.empty(shape, device='meta').stride()
    return DTensor.from_local(tensor, mesh, placements, run_check=False,
                              shape=torch.Size(shape), stride=stride)


class DTensorExecutor:
    def __init__(self, master, reference, momentum, *, cohort, tile_elements,
                 model=None, group=None, coordinates=None, mesh=None):
        if not dist.is_initialized():
            raise ValueError('initialized distributed group required')
        self.group = dist.group.WORLD if group is None else group
        self.peers = dist.get_process_group_ranks(self.group)
        self.k, self.rank = len(self.peers), dist.get_rank(self.group)
        self.s = int(cohort)
        workspace_elements(self.k, self.s)
        self.g = self.k // self.s
        self.a, self.b = divmod(self.rank, self.s)
        if coordinates is not None and (master is not None or model is not None):
            raise ValueError('use either parameter coordinates or master/model tensors')
        if coordinates is None:
            if master is None:
                raise ValueError('master or parameter coordinates required')
            coordinates = ParameterCoordinates([('flat', master, master if model is None else model)])
        self.coordinates = coordinates
        self.device, self.n = coordinates.device, coordinates.numel
        self.width = (self.n + self.k - 1) // self.k
        self.capacity = int(tile_elements)
        if self.capacity < 1:
            raise ValueError('positive tile capacity required')
        if (reference.numel() != self.g * self.width or momentum.numel() != self.width
                or any(x.dtype != torch.float32 or not x.is_contiguous() for x in (reference, momentum))):
            raise ValueError('contiguous FP32 R/M with matching coordinate ownership required')
        if any(x.device != self.device and x.device.type != 'cpu' for x in (reference, momentum)):
            raise ValueError('outer state must be on the master device or CPU')
        if (overlaps(reference, momentum)
                or any(overlaps(state, p) for state in (reference, momentum) for p in coordinates.storage_tensors())):
            raise ValueError('outer state must not alias master/model storage')
        for _, source, target in coordinates.pairs:
            if overlaps(source, target) and not (source.data_ptr() == target.data_ptr() and source.dtype == target.dtype):
                raise ValueError('only exact FP32 master/model aliasing is supported')
        self.reference, self.momentum = reference.view(self.g, self.width), momentum.view(-1)
        self.commit_model = master is not None and model is not None
        self.mesh = mesh if mesh is not None else make_mesh(self.group, self.s, self.device.type)
        expected_mesh = torch.tensor(self.peers).reshape(self.g, self.s).T
        if (not torch.equal(self.mesh.mesh, expected_mesh) or self.mesh.device_type != self.device.type
                or tuple(self.mesh.get_coordinate() or ()) != (self.b, self.a)):
            raise ValueError('DTensor mesh differs from outer coordinate ownership')
        sizes = {'pack': self.k, 'reference_stage': self.g, 'partial_bank': self.g,
                 'momentum_stage': 1, 'direction': 1}
        self.buffers = {name: torch.empty(count * self.capacity, dtype=torch.float32, device=self.device)
                        for name, count in sizes.items()}

    def allocation_bytes(self):
        persistent = sum(t.numel() * t.element_size() for t in self.buffers.values())
        bound = 4 * self.capacity * workspace_elements(self.k, self.s)
        return {'workspace_tensor_bytes': persistent, 'explicit_live_upper_bound_bytes': bound,
                'formula_bytes': bound, 'reference_bytes': self.reference.numel() * 4,
                'momentum_bytes': self.momentum.numel() * 4, 'slot_count': 1, 'arm': 'dtensor',
                'dynamic_redistribution_outputs': True, 'total_peak_memory_measured': False,
                'excluded': ['DTensor redistribution intermediates', 'collective/library/allocator storage',
                             'master/model', 'inner optimizer', 'Python and process metadata']}

    def storage_tensors(self):
        return [*self.coordinates.storage_tensors(), self.reference, self.momentum, *self.buffers.values()]

    @torch.no_grad()
    def step(self, *, mu=.9, eta=.7):
        for offset in range(0, self.width, self.capacity):
            t = min(self.capacity, self.width - offset)
            pack = self.buffers['pack'][:self.k * t].view(self.k, t)
            pack.zero_()
            for owner in range(self.k):
                start = owner * self.width + offset
                count = max(0, min(t, self.n - start))
                if count:
                    self.coordinates.read_into(start, pack[owner, :count])
            r = self.buffers['reference_stage'][:self.g * t].view(self.g, t)
            r.copy_(self.reference[:, offset:offset + t])
            w = local_dtensor(pack.view(1, 1, self.k * t), self.mesh,
                              (Shard(1), Shard(0)), (self.g, self.s, self.k * t))
            routed_dt = w.redistribute(placements=(Shard(2), Shard(0)))
            routed = routed_dt.to_local().view(self.s, self.g * t)
            # Every old W for this tile has been routed before centering/writeback.
            torch.sub(r.view(1, -1), routed, out=routed)
            bank = self.buffers['partial_bank'][:self.g * t]
            torch.sum(routed, dim=0, out=bank)
            del routed, routed_dt, w
            partial = local_dtensor(bank, self.mesh, (Shard(0), Partial()), (self.k * t,))
            owned = partial.redistribute(placements=(Shard(0), Shard(0)))
            average = owned.to_local()
            average.div_(self.k)
            m = self.buffers['momentum_stage'][:t]
            m.copy_(self.momentum[offset:offset + t])
            m.mul_(mu).add_(average)
            direction = self.buffers['direction'][:t]
            torch.mul(m, mu, out=direction)
            direction.add_(average).mul_(eta)
            torch.sub(r[self.a], direction, out=average)
            self.momentum[offset:offset + t].copy_(m)
            refreshed = owned.redistribute(placements=(Shard(0), Replicate()))
            self.reference[:, offset:offset + t].copy_(refreshed.to_local().view(self.g, t))
            returned = refreshed.redistribute(placements=(Replicate(), Replicate()))
            values = returned.to_local().view(self.k, t)
            for owner in range(self.k):
                start = owner * self.width + offset
                count = max(0, min(t, self.n - start))
                if count:
                    self.coordinates.write_from(start, values[owner, :count])
            del values, returned, refreshed, average, owned, partial
        if self.commit_model:
            for _, source, target in self.coordinates.pairs:
                target.copy_(source)
        return {'arm': 'dtensor', 'backend': 'explicit DTensor layouts',
                'mesh_global_ranks': self.mesh.mesh.tolist(), 'cohort': self.s,
                'modeled_payload_bytes': 2 * (self.k - 1) * self.width * 4,
                'physical_wire_bytes': None, 'native_sum_order': 'opaque',
                'payload_scope': 'logical GPU layout model; CPU redistribution may use fallbacks',
                'explicit_workspace': self.allocation_bytes()}
