"""Paged reference refresh with a bounded, event-ordered CUDA tile pipeline.

One independent communicator and CUDA stream per live slot permit inter-tile
overlap without relying on timing-dependent collective issue order. CPU/Gloo
executes the same dataflow synchronously for numerical/lifetime testing.
Momentum shard j always lives at group rank j; update execution follows R.
"""

from contextlib import nullcontext
import math
import time

import torch
import torch.distributed as dist

from .joint_plan import Layout, Planner, slot_bytes, tiles


class ReferencePages:
    def __init__(self, *, learners, width, page_elements, rank, cohort, device, read):
        self.k, self.width, self.page = learners, width, page_elements
        self.rank, self.layout, self.device = rank, Layout(learners, cohort), device
        self.pages = {}
        self._nbytes = 0
        for shard in range(learners):
            if self.layout.owns(rank, shard):
                for offset in range(0, width, page_elements):
                    value = torch.zeros(min(page_elements, width - offset), device=device, dtype=torch.float32)
                    read(shard * width + offset, value)
                    self.put((shard, offset), value)

    def put(self, key, value):
        if key in self.pages:
            raise ValueError('reference allocation already exists')
        self.pages[key] = value
        self._nbytes += value.numel() * value.element_size()

    def retire(self, key):
        value = self.pages.pop(key, None)
        if value is not None:
            self._nbytes -= value.numel() * value.element_size()

    def clear(self):
        self.pages.clear()
        self._nbytes = 0

    def keys(self, tile):
        if tile.offset % self.page or (tile.count % self.page and tile.offset + tile.count != self.width):
            raise ValueError('execution tiles must align with allocation pages')
        return tuple((tile.shard, offset) for offset in range(tile.offset, tile.offset + tile.count, self.page))

    def read_into(self, tile, target):
        for key in self.keys(tile):
            value = self.pages[key]
            offset = key[1] - tile.offset
            target[offset:offset + value.numel()].copy_(value)

    def write_from(self, tile, source):
        for key in self.keys(tile):
            value = self.pages[key]
            offset = key[1] - tile.offset
            value.copy_(source[offset:offset + value.numel()])

    def allocate(self, tile):
        for key in self.keys(tile):
            if key not in self.pages:
                self.put(key, torch.empty(min(self.page, self.width - key[1]), device=self.device, dtype=torch.float32))

    @property
    def nbytes(self):
        return self._nbytes


class Slot:
    def __init__(self, capacity, cohort, device, group, *, trace=False):
        self.group, self.device = group, device
        self.stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        with self.context():
            self.weight = torch.empty(capacity, device=device, dtype=torch.float32)
            self.value = torch.empty(capacity, device=device, dtype=torch.float32)
            self.reference = torch.empty(capacity, device=device, dtype=torch.float32)
            self.momentum = torch.empty(capacity, device=device, dtype=torch.float32)
            self.stack = torch.empty(cohort.bit_length() - 1, capacity, device=device, dtype=torch.float32)
        self.consumed = torch.cuda.Event(enable_timing=trace) if self.stream is not None else None
        self.returned = torch.cuda.Event(enable_timing=trace) if self.stream is not None else None
        self.input_started = torch.cuda.Event(enable_timing=True) if self.stream is not None and trace else None
        self.return_started = torch.cuda.Event(enable_timing=True) if self.stream is not None and trace else None

    def context(self):
        return torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()

    def tensors(self):
        return (self.weight, self.value, self.reference, self.momentum, self.stack)


class JointExecutor:
    commits_model = True

    def __init__(self, master=None, *, coordinates=None, model=None, group=None,
                 cohort=2, page_elements=65536, capacities=None, slot_counts=(1, 2, 4),
                 max_slots=None, costs=(), trace=False):
        if not dist.is_initialized():
            raise ValueError('distributed process group required')
        if (master is None) == (coordinates is None) or (coordinates is not None and model is not None):
            raise ValueError('supply tensor masters or parameter coordinates, exclusively')
        self.group = dist.group.WORLD if group is None else group
        self.peers = dist.get_process_group_ranks(self.group)
        self.k, self.rank = len(self.peers), dist.get_rank(self.group)
        self.coordinates, self.master, self.model = coordinates, master, model
        self.device = coordinates.device if coordinates is not None else master.device
        self.n = coordinates.numel if coordinates is not None else master.numel()
        if self.n < 1 or (master is not None and (master.dtype != torch.float32 or not master.is_contiguous())):
            raise ValueError('nonempty contiguous FP32 masters required')
        if model is not None and (model.numel() != self.n or not model.is_contiguous()
                                  or model.device != self.device or model.dtype not in (torch.float32, torch.bfloat16)):
            raise ValueError('matching contiguous FP32/BF16 model required')
        self.width = (self.n + self.k - 1) // self.k
        self.planner = Planner(self.k, self.width, page_elements, capacities=capacities,
                               slots=slot_counts, costs=costs)
        self.max_slots = max_slots or max(slot_counts)
        if self.max_slots < max(slot_counts):
            raise ValueError('communicator count must cover every candidate slot count')
        self.reference = ReferencePages(learners=self.k, width=self.width, page_elements=page_elements,
                                        rank=self.rank, cohort=cohort, device=self.device, read=self._read)
        self.momentum = torch.zeros(self.width, dtype=torch.float32, device=self.device)
        self.slot_groups = []
        # Only members of this corresponding-coordinate group participate. These
        # groups are disjoint across TP positions, and creation order is fixed.
        for _ in range(self.max_slots):
            communication = dist.new_group(self.peers, use_local_synchronization=True)
            dist.barrier(group=communication)
            self.slot_groups.append(communication)
        self.capacity, self.active_slots = page_elements, 0
        self.trace_enabled, self.busy, self.round = trace, False, 0
        self.workspace_bytes = 0
        self.last_receipt = None

    @property
    def s(self):
        return self.reference.layout.cohort

    @torch.no_grad()
    def _read(self, start, target):
        target.zero_()
        count = max(0, min(target.numel(), self.n - start))
        if count:
            if self.coordinates is None:
                target[:count].copy_(self.master[start:start + count])
            else:
                self.coordinates.read_into(start, target[:count])

    @torch.no_grad()
    def _commit(self, start, value):
        count = max(0, min(value.numel(), self.n - start))
        if not count:
            return
        if self.coordinates is None:
            self.master[start:start + count].copy_(value[:count])
            if self.model is not None:
                self.model[start:start + count].copy_(value[:count])
        else:
            self.coordinates.commit_from(start, value[:count])

    def allocation_bytes(self):
        return dict(reference_bytes=self.reference.nbytes, momentum_bytes=self.momentum.numel() * 4,
                    workspace_tensor_bytes=self.workspace_bytes, slot_count=self.active_slots,
                    cohort=self.s, tile_elements=self.capacity, page_elements=self.reference.page,
                    schedule='joint',
                    momentum_home='canonical shard j -> outer rank j',
                    excluded=['other training allocations', 'allocator/library storage'])

    def _send(self, value, peer, slot, phase):
        dist.isend(value, dst=self.peers[peer], group=slot.group).wait()
        self.sent[phase] += value.numel() * value.element_size()

    def _recv(self, value, peer, slot):
        dist.irecv(value, src=self.peers[peer], group=slot.group).wait()

    def _consume(self, tile, slot, layout, mu, eta):
        """Route raw leaves, preserve the adjacent FP32 tree, then update at R."""
        if slot.input_started is not None:
            slot.input_started.record(slot.stream)
        t, shard = tile.count, tile.shard
        weight, value, r, m = (v[:t] for v in (slot.weight, slot.value, slot.reference, slot.momentum))
        a, b = divmod(self.rank, layout.cohort)
        owner_b = shard // layout.groups
        owner = a * layout.cohort + owner_b
        self._read(shard * self.width + tile.offset, weight)
        if b == owner_b:
            self.reference.read_into(tile, r)
        # At most one received leaf and log2(s) accumulated partials are live.
        for leaf in range(layout.cohort):
            sender = a * layout.cohort + leaf
            if self.rank == sender and sender != owner:
                self._send(weight, owner, slot, 'learner_input')
            if self.rank == owner:
                if sender == owner:
                    value.copy_(weight)
                else:
                    self._recv(value, sender, slot)
                torch.sub(r, value, out=value)
                level, prefix = 0, leaf
                while prefix & 1:
                    torch.add(slot.stack[level, :t], value, out=value)
                    level, prefix = level + 1, prefix >> 1
                if leaf + 1 < layout.cohort:
                    slot.stack[level, :t].copy_(value)
        root_a, root = shard % layout.groups, layout.executor(shard)
        if b == owner_b:
            for bit in range(layout.groups.bit_length() - 1):
                hop = 1 << bit
                partner = (a ^ hop) * layout.cohort + owner_b
                if (a & hop) != (root_a & hop):
                    self._send(value, partner, slot, 'partial_reduce')
                    break
                self._recv(weight, partner, slot)
                if a & hop:
                    torch.add(weight, value, out=value)
                else:
                    torch.add(value, weight, out=value)
        home = shard
        persistent_m = self.momentum[tile.offset:tile.offset + t]
        if self.rank == home and home != root:
            self._send(persistent_m, root, slot, 'momentum_stage')
            self._recv(persistent_m, root, slot)
        if self.rank == root:
            if home == root:
                m.copy_(persistent_m)
            else:
                self._recv(m, home, slot)
            value.div_(self.k)
            m.mul_(mu).add_(value)
            torch.mul(m, mu, out=weight)
            weight.add_(value).mul_(eta)
            torch.sub(r, weight, out=value)
            if home == root:
                persistent_m.copy_(m)
            else:
                self._send(m, home, slot, 'momentum_writeback')
        if slot.consumed is not None:
            slot.consumed.record(slot.stream)

    def _return(self, tile, slot, old, destination):
        if slot.return_started is not None:
            slot.return_started.record(slot.stream)
        value = slot.value[:tile.count]
        root = old.executor(tile.shard)
        virtual = self.rank ^ root
        for bit in range(self.k.bit_length() - 1):
            hop = 1 << bit
            if virtual < hop:
                self._send(value, (virtual + hop) ^ root, slot, 'parameter_return')
            elif virtual < 2 * hop:
                self._recv(value, (virtual - hop) ^ root, slot)
        self._commit(tile.shard * self.width + tile.offset, value)
        if destination.owns(self.rank, tile.shard):
            self.reference.write_from(tile, value)
        if slot.returned is not None:
            slot.returned.record(slot.stream)

    def _base_bytes(self):
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) - self.reference.nbytes - self.width * 4
        # CPU accounting is explicit; no CPU allocator or library-peak claim.
        values = (self.coordinates.storage_tensors() if self.coordinates is not None
                  else [x for x in (self.master, self.model) if x is not None])
        storages = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in values}
        return sum(storages.values())

    def plan(self, budget_bytes, *, target=None, mode='joint', slots=None, headroom_bytes=0,
             remaining_rounds=1, workspace_limit=None, final_budget_bytes=None):
        local = dict(base=self._base_bytes(), budget=int(budget_bytes),
                     final_budget=int(budget_bytes if final_budget_bytes is None else final_budget_bytes))
        ranks = [None] * self.k
        dist.all_gather_object(ranks, local, group=self.group)
        return self.planner.choose(self.s, [r['budget'] for r in ranks], base=[r['base'] for r in ranks],
                                   headroom=headroom_bytes, mode=mode, target=target, fixed_slots=slots,
                                   remaining_rounds=remaining_rounds, workspace_limit=workspace_limit,
                                   final_budgets=[r['final_budget'] for r in ranks])

    @torch.no_grad()
    def step(self, *, mu=.9, eta=.7, plan=None, budget_bytes=None, target=None,
             mode='joint', slots=None, headroom_bytes=0, remaining_rounds=1, workspace_limit=None):
        if self.busy:
            raise RuntimeError('overlapping outer boundaries are forbidden')
        if not 0 <= mu < 1 or not math.isfinite(eta) or eta <= 0:
            raise ValueError('invalid Nesterov parameters')
        if plan is None:
            if budget_bytes is None:
                raise ValueError('an explicit budget or admitted plan is required')
            plan = self.plan(budget_bytes, target=target, mode=mode, slots=slots,
                             headroom_bytes=headroom_bytes, remaining_rounds=remaining_rounds,
                             workspace_limit=workspace_limit)
        if (plan.old_cohort != self.s or not 1 <= plan.slots <= self.max_slots
                or plan.capacity not in self.planner.capacities or plan.mode not in ('joint', 'static', 'separate')
                or (plan.mode == 'static' and plan.cohort != self.s)
                or plan.workspace_bytes != plan.slots * slot_bytes(self.s, plan.capacity)
                or len(plan.peaks) != self.k or headroom_bytes < 0):
            raise ValueError('plan does not match current layout or configured resources')
        old, new = self.reference.layout, Layout(self.k, plan.cohort)
        destination = old if plan.mode == 'separate' else new
        self.busy = True
        self.sent = dict.fromkeys(('learner_input', 'partial_reduce', 'parameter_return',
                                  'momentum_stage', 'momentum_writeback', 'reference_migration'), 0)
        self.capacity, self.active_slots = plan.capacity, plan.slots
        self.workspace_bytes = plan.workspace_bytes
        trace, timeline, explicit_peak = [], [], 0
        old_keys = set(self.reference.pages)
        old_bytes = self.reference.nbytes
        base = self._base_bytes()
        started = time.perf_counter()

        def observe(stage, wave):
            nonlocal explicit_peak
            explicit = base + self.reference.nbytes + self.width * 4 + self.workspace_bytes + headroom_bytes
            explicit_peak = max(explicit_peak, explicit)
            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(self.device)
                if allocated > plan.peaks[self.rank]:
                    raise MemoryError('observed CUDA allocation exceeded admitted peak including headroom')
            if explicit > plan.peaks[self.rank]:
                raise AssertionError('actual explicit storage exceeded admitted schedule')
            if self.trace_enabled:
                row = dict(stage=stage, wave=wave, seconds=time.perf_counter() - started,
                           reference_old_bytes=old_bytes, reference_new_bytes=self.reference.nbytes - old_bytes,
                           momentum_home_bytes=self.width * 4, workspace_bytes=self.workspace_bytes,
                           accounted_total_bytes=explicit)
                if self.device.type == 'cuda':
                    row.update(cuda_allocated_bytes=allocated, cuda_reserved_bytes=torch.cuda.memory_reserved(self.device))
                trace.append(row)

        origin = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' and self.trace_enabled else None
        if origin is not None:
            origin.record(torch.cuda.current_stream(self.device))
        live = [Slot(plan.capacity, old.cohort, self.device, self.slot_groups[i], trace=self.trace_enabled)
                for i in range(plan.slots)]
        if self.device.type == 'cuda':
            for slot in live:
                slot.stream.wait_stream(torch.cuda.current_stream(self.device))
        sequence = tiles(self.k, self.width, plan.capacity)
        observe('admit_workspace', -1)
        # Deterministic bounded waves: all ranks assign the same tile to the same
        # slot. Launch all independent consumers before waiting on completion.
        for start in range(0, len(sequence), plan.slots):
            wave = list(zip(sequence[start:start + plan.slots], live))
            for tile, slot in wave:
                with slot.context():
                    self._consume(tile, slot, old, mu, eta)
            # Release only after actual completed reads, including remote M
            # writeback. Page views never outlive their owning allocation here.
            for tile, slot in wave:
                if slot.consumed is not None:
                    slot.consumed.synchronize()
                for key in self.reference.keys(tile):
                    if key in old_keys:
                        old_keys.remove(key)
                        old_bytes -= self.reference.pages[key].numel() * 4
                    if not destination.owns(self.rank, tile.shard):
                        self.reference.retire(key)
                observe('old_consumers_complete', start)
                with slot.context():
                    if destination.owns(self.rank, tile.shard):
                        self.reference.allocate(tile)
                observe('new_destinations_reserved', start)
                with slot.context():
                    self._return(tile, slot, old, destination)
            for slot_index, (tile, slot) in enumerate(wave):
                if slot.returned is not None:
                    slot.returned.synchronize()
                if origin is not None:
                    timeline.append(dict(shard=tile.shard, offset=tile.offset, count=tile.count, slot=slot_index,
                                         input_started_ms=origin.elapsed_time(slot.input_started),
                                         old_consumers_completed_ms=origin.elapsed_time(slot.consumed),
                                         return_started_ms=origin.elapsed_time(slot.return_started),
                                         commit_completed_ms=origin.elapsed_time(slot.returned)))
            observe('commit_and_reference_writes_complete', start)
        # All streams have drained. Drop every reference to scratch before
        # separate conversion, so its measured memory matches the planner.
        wave.clear()
        slot = None
        live.clear()
        self.workspace_bytes, self.active_slots = 0, 0
        if plan.mode == 'separate' and old != new:
            self.reference.clear()
            old_keys.clear()
            old_bytes = 0
            observe('separate_old_reference_released', len(sequence))
            for tile in tiles(self.k, self.width, self.reference.page):
                if new.owns(self.rank, tile.shard):
                    self.reference.allocate(tile)
                    self._read(tile.shard * self.width + tile.offset,
                               self.reference.pages[tile.shard, tile.offset])
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
        self.reference.layout = new
        observe('publish_layout', len(sequence))
        self.busy, self.round = False, self.round + 1
        self.last_receipt = dict(round=self.round, plan=plan.receipt(), trace=trace, cuda_timeline=timeline,
                                 explicit_peak_bytes=explicit_peak, sent_tensor_bytes_by_phase=dict(self.sent),
                                 sent_tensor_bytes=sum(self.sent.values()),
                                 reference_migration_bytes=0,
                                 momentum_home='canonical shard j -> outer rank j',
                                 padded_master_bytes=self.width * self.k * 4,
                                 local_reference_construction_read_bytes=(self.reference.nbytes
                                     if plan.mode == 'separate' and old != new else 0),
                                 physical_wire_bytes=None,
                                 elapsed_seconds=time.perf_counter() - started)
        return self.last_receipt

    def state_dict(self):
        if self.busy:
            raise RuntimeError('checkpoint requires a drained completed boundary')
        return dict(format='pier-joint-pages-v1', learners=self.k, rank=self.rank, numel=self.n,
                    page_elements=self.reference.page, cohort=self.s, round=self.round,
                    momentum_home=self.rank, momentum=self.momentum,
                    reference_pages=self.reference.pages)

    @torch.no_grad()
    def load_state_dict(self, state):
        if self.busy:
            raise RuntimeError('cannot restore an active executor')
        expected = dict(format='pier-joint-pages-v1', learners=self.k, rank=self.rank,
                        numel=self.n, page_elements=self.reference.page, momentum_home=self.rank)
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError('joint checkpoint coordinate/page/momentum identity differs')
        layout = Layout(self.k, state['cohort'])
        keys = {(j, o) for j in range(self.k) if layout.owns(self.rank, j)
                for o in range(0, self.width, self.reference.page)}
        if set(state['reference_pages']) != keys:
            raise ValueError('incomplete reference page ownership in checkpoint')
        for key, value in state['reference_pages'].items():
            if value.dtype != torch.float32 or value.shape != (min(self.reference.page, self.width - key[1]),):
                raise ValueError('invalid reference page shape/dtype')
        if state['momentum'].dtype != torch.float32 or state['momentum'].shape != self.momentum.shape:
            raise ValueError('invalid momentum checkpoint')
        if not isinstance(state['round'], int) or state['round'] < 0:
            raise ValueError('invalid completed round')
        saved_pages = dict(state['reference_pages'])
        self.reference.clear()
        for key, value in saved_pages.items():
            self.reference.put(key, value.to(self.device).clone())
        self.reference.layout, self.round = layout, state['round']
        self.momentum.copy_(state['momentum'])

    def close(self):
        if self.busy:
            raise RuntimeError('cannot close during an outer boundary')
        for group in self.slot_groups:
            dist.destroy_process_group(group)
        self.slot_groups.clear()
