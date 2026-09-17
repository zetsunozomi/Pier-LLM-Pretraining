"""One-slot, ordered reference-owner execution using real point-to-point copies.

CUDA paths are provided for external validation; local evidence is CPU/Gloo.
This is a correctness-oriented executor, not a tuned collective library.
All model/state tensors are caller-owned. No full reference is reconstructed.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def power2(n):
    return n > 0 and not (n & (n - 1))


def overlaps(x, y):
    if x.device != y.device:
        return False
    return max(x.data_ptr(), y.data_ptr()) < min(
        x.data_ptr() + x.numel() * x.element_size(),
        y.data_ptr() + y.numel() * y.element_size())


class CenteredExecutor:
    """Caller initializes the group and, for NCCL, the current CUDA device.

    rank = a*s+b. R has shape [g, ceil(N/K)] on rank (a,b); M has
    ceil(N/K) entries. The reference range b is replicated over a.
    Tiles take t entries from EVERY momentum-owner subrange, so C=4*g*t.
    R/M can reside on the master device or CPU. Host copies are blocking.
    Finite inputs and completion of the final local update are preconditions.
    The caller controls successful-step counting and invokes all group members.
    """

    def __init__(self, master, reference, momentum, *, cohort, tile_elements,
                 model=None, group=None, coordinates=None):
        if not dist.is_initialized():
            raise ValueError('initialize a distributed group first')
        # Older PyTorch accepts None for get_rank/get_world_size, but not for
        # get_process_group_ranks. Resolve the default once for all operations.
        self.group = dist.group.WORLD if group is None else group
        self.k = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.peers = dist.get_process_group_ranks(self.group)
        self.s = int(cohort)
        if not power2(self.k) or not power2(self.s) or self.k % self.s:
            raise ValueError('nested power-of-two K and s required')
        if coordinates is not None and (master is not None or model is not None):
            raise ValueError('use either tensor master/model or parameter coordinates')
        if coordinates is None and master is None:
            raise ValueError('master or parameter coordinates required')
        self.coordinates = coordinates
        self.device = coordinates.device if coordinates is not None else master.device
        self.n = coordinates.numel if coordinates is not None else master.numel()
        masters = ([p for _, p, _ in coordinates.pairs] if coordinates is not None else [master])
        if tile_elements < 1 or self.n < 1:
            raise ValueError('positive tile capacity and nonempty master required')
        self.g = self.k // self.s
        self.a, self.b = divmod(self.rank, self.s)
        self.width = (self.n + self.k - 1) // self.k
        self.capacity = int(tile_elements)
        if any(x.dtype != torch.float32 or not x.is_contiguous()
               for x in (*masters, reference, momentum)):
            raise ValueError('master, reference and momentum must be contiguous FP32')
        if reference.numel() != self.g * self.width or momentum.numel() != self.width:
            raise ValueError('R/M sizes do not match coordinate ownership')
        if any(x.device != self.device and x.device.type != 'cpu'
               for x in (reference, momentum)):
            raise ValueError('state must be on the master device or CPU')
        if (overlaps(reference, momentum) or any(overlaps(p, state)
                for p in (coordinates.storage_tensors() if coordinates is not None else masters)
                for state in (reference, momentum))):
            raise ValueError('master, reference and momentum storage must not overlap')
        if model is not None:
            if model.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError('model commit supports FP32 or BF16')
            if model.numel() != self.n or not model.is_contiguous() or model.device != master.device:
                raise ValueError('model must be a contiguous matching vector on the master device')
            if overlaps(model, reference) or overlaps(model, momentum):
                raise ValueError('model must not alias outer state')
            if overlaps(model, master) and not (
                    model.data_ptr() == master.data_ptr() and model.dtype == master.dtype):
                raise ValueError('only an exact master/model alias is supported')
        self.master = None if master is None else master.view(-1)
        self.reference = reference.view(self.g, self.width)
        self.momentum = momentum.view(-1)
        self.model = None if model is None else model.view(-1)
        c = self.g * self.capacity
        sizes = {'pack': self.s * c, 'received': self.s * c, 'reference_stage': c,
                 'send_stage': c, 'receive_stage': c, 'momentum_stage': self.capacity}
        self.buffers = {name: torch.empty(size, dtype=torch.float32, device=self.device)
                        for name, size in sizes.items()}
        self.sent = [0, 0, 0, 0]

    def allocation_bytes(self):
        c = self.g * self.capacity * 4
        actual = sum(x.numel() * x.element_size() for x in self.buffers.values())
        exact = (2 * self.s + 3) * c + c // self.g
        assert actual == exact
        return {'workspace_tensor_bytes': actual, 'reference_bytes': self.reference.numel() * 4,
                'momentum_bytes': self.momentum.numel() * 4, 'C_bytes': c,
                'slot_count': 1, 'formula_bytes': exact,
                'excluded': ['master/model', 'inner optimizer', 'allocator/library storage',
                             'Python and process metadata']}

    def storage_tensors(self):
        inputs = self.coordinates.storage_tensors() if self.coordinates is not None else [self.master]
        return [*inputs, self.reference, self.momentum, *self.buffers.values()] + (
            [] if self.model is None else [self.model])

    def _exchange(self, send, recv, destination, source, phase):
        if destination == self.rank or source == self.rank:
            raise ValueError('self-copy must be handled locally')
        operations = [dist.P2POp(dist.isend, send, self.peers[destination], self.group),
                      dist.P2POp(dist.irecv, recv, self.peers[source], self.group)]
        requests = dist.batch_isend_irecv(operations)
        for request in requests:
            request.wait()
        self.sent[phase] += send.numel() * send.element_size()

    def _upper(self, bank, t, mu, eta, offset):
        """Ordered recursive halving, update at owner a, then inverse gather."""
        send = self.buffers['send_stage']
        recv = self.buffers['receive_stage']
        active = list(range(self.g))
        for bit in range(self.g.bit_length() - 1):
            hop = 1 << bit
            keep = [j for j in active if bool(j & hop) == bool(self.a & hop)]
            outgoing = [j for j in active if j not in keep]
            for i, j in enumerate(outgoing):
                send[i * t:(i + 1) * t].copy_(bank[j])
            peer = (self.a ^ hop) * self.s + self.b
            self._exchange(send[:len(outgoing) * t], recv[:len(keep) * t], peer, peer, 1)
            for i, j in enumerate(keep):
                incoming = recv[i * t:(i + 1) * t]
                if self.a & hop:
                    torch.add(incoming, bank[j], out=bank[j])
                else:
                    torch.add(bank[j], incoming, out=bank[j])
            active = keep
        assert active == [self.a]
        average = bank[self.a]
        average.div_(self.k)
        m = self.buffers['momentum_stage'][:t]
        m.copy_(self.momentum[offset:offset + t])
        m.mul_(mu).add_(average)
        direction = send[:t]
        torch.mul(m, mu, out=direction)
        torch.add(average, direction, out=direction)
        direction.mul_(eta)
        r = self.buffers['reference_stage'][:self.g * t].view(self.g, t)[self.a]
        torch.sub(r, direction, out=average)
        # The owned momentum is durable before distributing the new reference.
        self.momentum[offset:offset + t].copy_(m)
        known = [self.a]
        for bit in reversed(range(self.g.bit_length() - 1)):
            hop = 1 << bit
            incoming_indices = sorted(j ^ hop for j in known)
            for i, j in enumerate(known):
                send[i * t:(i + 1) * t].copy_(bank[j])
            peer = (self.a ^ hop) * self.s + self.b
            self._exchange(send[:len(known) * t], recv[:len(known) * t], peer, peer, 2)
            for i, j in enumerate(incoming_indices):
                bank[j].copy_(recv[i * t:(i + 1) * t])
            known = sorted(known + incoming_indices)

    @torch.no_grad()
    def step(self, *, mu=.9, eta=.7):
        self.sent = [0, 0, 0, 0]
        for offset in range(0, self.width, self.capacity):
            t = min(self.capacity, self.width - offset)
            c = self.g * t
            pack = self.buffers['pack'][:self.s * c].view(self.s, self.g, t)
            received = self.buffers['received'][:self.s * c].view(self.s, c)
            reference = self.buffers['reference_stage'][:c].view(self.g, t)
            pack.zero_()
            # Read all local old-master inputs before any writeback in this tile.
            for target_b in range(self.s):
                for owner_a in range(self.g):
                    start = (target_b * self.g + owner_a) * self.width + offset
                    count = max(0, min(t, self.n - start))
                    if count:
                        if self.coordinates is None:
                            pack[target_b, owner_a, :count].copy_(self.master[start:start + count])
                        else:
                            self.coordinates.read_into(start, pack[target_b, owner_a, :count])
            reference.copy_(self.reference[:, offset:offset + t])
            received[self.b].copy_(pack[self.b].view(-1))
            # Post the cohort AllToAll as one batch. Each peer pair has one
            # message per direction, avoiding per-peer host serialization.
            operations = []
            for peer_b in range(self.s):
                if peer_b == self.b:
                    continue
                peer = self.peers[self.a * self.s + peer_b]
                operations.append(dist.P2POp(dist.isend, pack[peer_b].view(-1), peer, self.group))
                operations.append(dist.P2POp(dist.irecv, received[peer_b], peer, self.group))
            if operations:
                for request in dist.batch_isend_irecv(operations):
                    request.wait()
                self.sent[0] += (self.s - 1) * c * 4
            torch.sub(reference.view(1, c), received, out=received)
            active = self.s
            while active > 1:
                for pair in range(active // 2):
                    torch.add(received[2 * pair], received[2 * pair + 1], out=received[pair])
                active //= 2
            bank = received[0].view(self.g, t)
            self._upper(bank, t, mu, eta, offset)
            self.reference[:, offset:offset + t].copy_(bank)
            pack[self.b].copy_(bank)
            # All consumers of received/centering data have finished. Reuse it
            # as disjoint outgoing and incoming areas for cohort AllGather.
            scratch = received.view(-1)
            split = (self.s // 2) * c
            known = [self.b]
            for bit in reversed(range(self.s.bit_length() - 1)):
                hop = 1 << bit
                incoming_indices = sorted(j ^ hop for j in known)
                count = len(known) * c
                for i, j in enumerate(known):
                    scratch[i * c:(i + 1) * c].copy_(pack[j].view(-1))
                peer = self.a * self.s + (self.b ^ hop)
                self._exchange(scratch[:count], scratch[split:split + count], peer, peer, 3)
                for i, j in enumerate(incoming_indices):
                    pack[j].view(-1).copy_(scratch[split + i * c:split + (i + 1) * c])
                known = sorted(known + incoming_indices)
            for target_b in range(self.s):
                for owner_a in range(self.g):
                    start = (target_b * self.g + owner_a) * self.width + offset
                    count = max(0, min(t, self.n - start))
                    if count:
                        if self.coordinates is None:
                            self.master[start:start + count].copy_(pack[target_b, owner_a, :count])
                        else:
                            self.coordinates.write_from(start, pack[target_b, owner_a, :count])
                        if self.model is not None:
                            self.model[start:start + count].copy_(self.master[start:start + count])
        # q=1 reference executor returns only after its local device work drains.
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return {'sent_tensor_bytes_by_phase': list(self.sent),
                'sent_tensor_bytes': sum(self.sent),
                'padded_master_bytes': self.k * self.width * 4}
