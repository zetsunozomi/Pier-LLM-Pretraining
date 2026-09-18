"""Full-coordinate ordered FP32 oracle with file-backed R/M and tiled gathers.

This is deliberately expensive validation, never a performance execution path.
It visits every master and owned R/M coordinate; no sampling and no full model
gather on the device. OS file-cache and collective-library storage remain outside
the explicit scratch ledger. Private oracle files are disposable, not checkpoints.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .verification import same


class FileVector:
    """A private FP32 vector with checked, bounded pread/pwrite operations."""
    def __init__(self, path, elements):
        if elements < 1:
            raise ValueError('nonempty oracle vector required')
        self.path, self.elements = Path(path), elements
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(self.fd, elements * 4)  # Fresh sparse suffix reads as exact +0.

    def _bounds(self, start, count):
        if start < 0 or count < 1 or start + count > self.elements:
            raise ValueError('oracle vector range outside its coordinates')

    def read(self, start, count):
        self._bounds(start, count)
        data = bytearray()
        while len(data) < count * 4:
            block = os.pread(self.fd, count * 4 - len(data), start * 4 + len(data))
            if not block:
                raise IOError('truncated oracle vector')
            data.extend(block)
        return np.frombuffer(data, dtype='<f4')

    def write(self, start, values):
        values = np.asarray(values)
        if values.dtype != np.float32 or values.ndim != 1:
            raise ValueError('oracle writes require a flat FP32 array')
        self._bounds(start, values.size)
        view = memoryview(np.ascontiguousarray(values, dtype='<f4')).cast('B')
        offset = 0
        while offset < len(view):
            count = os.pwrite(self.fd, view[offset:], start * 4 + offset)
            if count <= 0:
                raise IOError('short oracle write')
            offset += count

    def sha256(self):
        result = hashlib.sha256()
        for start in range(0, self.elements, 262144):
            result.update(self.read(start, min(262144, self.elements - start)).tobytes())
        return result.hexdigest()

    def close(self):
        if getattr(self, 'fd', None) is not None:
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        self.close()


class StreamedOracle:
    def __init__(self, coordinates, executor, directory, tile_elements=65536):
        if tile_elements < 1:
            raise ValueError('positive oracle tile required')
        self.coordinates, self.executor = coordinates, executor
        self.n, self.k = coordinates.numel, executor.k
        self.capacity = min(tile_elements, self.n)
        self.group, self.device = executor.group, coordinates.device
        if not hasattr(executor, 's'):
            raise ValueError('streamed fixed-tree oracle applies only to the ordered Pier executor')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.reference = FileVector(directory / 'reference.f32', self.n)
        self.momentum = FileVector(directory / 'momentum.f32', self.n)
        self.local = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self.gathered = torch.empty(self.k * self.capacity, dtype=torch.float32, device=self.device)
        self.host = torch.empty(self.k * self.capacity, dtype=torch.float32, device='cpu')
        self.boundaries_checked = 0
        self.last_master_coordinates = 0
        # R starts at the already-agreed initial FP32 masters; M is all +0.
        for start in range(0, self.n, self.capacity):
            count = min(self.capacity, self.n - start)
            coordinates.read_into(start, self.host[:count])
            if not np.isfinite(self.host[:count].numpy()).all():
                raise ValueError('streamed oracle requires finite initial masters')
            self.reference.write(start, self.host[:count].numpy())

    def allocation_bytes(self):
        return {'kind': 'streamed-full-coordinate-fixed-tree', 'tile_elements': self.capacity,
                'persistent_device_scratch_bytes': (self.k + 1) * self.capacity * 4,
                'persistent_host_gather_bytes': self.k * self.capacity * 4,
                'file_logical_bytes': self.n * 8, 'coordinates_per_vector': self.n,
                'boundaries_checked': self.boundaries_checked,
                'last_master_coordinates_checked': self.last_master_coordinates,
                'additional_tile_temporaries': 'FP32 R/M/direction, bounded file-I/O and bit-comparison copies',
                'excluded': ['OS file cache', 'filesystem metadata', 'collective-library storage',
                             'model/inner optimizer/executor state', 'checkpoint loading'],
                'performance_result': False}

    @torch.no_grad()
    def prepare(self, mu, eta):
        """Advance the independent oracle using every learner's old masters."""
        mu, eta = np.float32(mu), np.float32(eta)
        for start in range(0, self.n, self.capacity):
            count = min(self.capacity, self.n - start)
            local, gathered = self.local[:count], self.gathered[:self.k * count]
            self.coordinates.read_into(start, local)
            if self.k == 1:
                gathered.copy_(local)
            else:
                dist.all_gather_into_tensor(gathered, local, group=self.group)
            self.host[:self.k * count].copy_(gathered)
            bank = self.host[:self.k * count].numpy().reshape(self.k, count)
            reference, momentum = self.reference.read(start, count), self.momentum.read(start, count)
            if not all(np.isfinite(value).all() for value in (bank, reference, momentum)):
                raise ValueError(f'nonfinite oracle input at coordinate {start}')
            np.subtract(reference[None, :], bank, out=bank)
            active = self.k
            while active > 1:
                for pair in range(active // 2):
                    np.add(bank[2 * pair], bank[2 * pair + 1], out=bank[pair])
                active //= 2
            average = bank[0]
            np.divide(average, np.float32(self.k), out=average)
            np.multiply(momentum, mu, out=momentum)
            np.add(momentum, average, out=momentum)
            direction = np.multiply(momentum, mu, dtype=np.float32)
            np.add(average, direction, out=direction)
            np.multiply(direction, eta, out=direction)
            np.subtract(reference, direction, out=reference)
            if not np.isfinite(reference).all() or not np.isfinite(momentum).all():
                raise ValueError(f'nonfinite outer result at coordinate {start}')
            self.reference.write(start, reference)
            self.momentum.write(start, momentum)

    def _check_owned(self, value, vector, global_start, label):
        value = value.view(-1)
        for offset in range(0, value.numel(), self.capacity):
            count = min(self.capacity, value.numel() - offset)
            start = global_start + offset
            valid = max(0, min(count, self.n - start))
            expected = torch.zeros(count, dtype=torch.float32)
            if valid:
                expected[:valid].copy_(torch.from_numpy(vector.read(start, valid)))
            same(value[offset:offset + count], expected, f'{label} at {start}')

    @torch.no_grad()
    def check(self):
        checked = 0
        for start in range(0, self.n, self.capacity):
            count = min(self.capacity, self.n - start)
            actual = self.host[:count]
            self.coordinates.read_into(start, actual)
            same(actual, torch.from_numpy(self.reference.read(start, count)), f'master vs streamed oracle at {start}')
            checked += count
        e = self.executor
        self._check_owned(e.reference, self.reference, e.b * e.g * e.width, 'owned R')
        self._check_owned(e.momentum, self.momentum, (e.b * e.g + e.a) * e.width, 'owned M')
        if checked != self.n:
            raise AssertionError('incomplete master coordinate verification')
        self.last_master_coordinates = checked
        self.boundaries_checked += 1

    def checkpoint_receipt(self):
        return {'format': 'pier-streamed-oracle-v1', 'elements': self.n,
                'reference_sha256': self.reference.sha256(), 'momentum_sha256': self.momentum.sha256(),
                'boundaries_checked': self.boundaries_checked}

    @torch.no_grad()
    def restore_from_owners(self, receipt):
        """Rebuild disposable files from restored R/M, then check the saved hashes.

        Masters may already include unsynchronized inner updates at a midcycle
        checkpoint, so they are deliberately not used to reconstruct R.
        """
        if (receipt['format'] != 'pier-streamed-oracle-v1' or receipt['elements'] != self.n
                or not isinstance(receipt['boundaries_checked'], int) or receipt['boundaries_checked'] < 0):
            raise ValueError('streamed oracle checkpoint schema differs')
        e = self.executor
        for kind, vector, owners, span in (
                ('reference', self.reference, e.s, e.g * e.width),
                ('momentum', self.momentum, e.k, e.width)):
            for owner in range(owners):
                # R has g identical copies: pick cohort a=0. M has one owner.
                source_rank = owner if kind == 'reference' else (owner % e.g) * e.s + owner // e.g
                start, end = owner * span, min((owner + 1) * span, self.n)
                for offset in range(start, end, self.capacity):
                    count = min(self.capacity, end - offset)
                    transfer = self.local[:count]
                    if e.rank == source_rank:
                        stored = getattr(e, kind).view(-1)
                        transfer.copy_(stored[offset - start:offset - start + count])
                    if self.k > 1:
                        dist.broadcast(transfer, src=e.peers[source_rank], group=self.group)
                    self.host[:count].copy_(transfer)
                    vector.write(offset, self.host[:count].numpy())
        actual = self.checkpoint_receipt()
        for key in ('reference_sha256', 'momentum_sha256'):
            if actual[key] != receipt[key]:
                raise AssertionError(f'restored streamed oracle differs: {key}')
        # Check all replicated R copies, not just the a=0 reconstruction source.
        self._check_owned(e.reference, self.reference, e.b * e.g * e.width, 'restored owned R')
        self._check_owned(e.momentum, self.momentum, (e.b * e.g + e.a) * e.width, 'restored owned M')
        self.boundaries_checked = receipt['boundaries_checked']

    def close(self):
        self.reference.close()
        self.momentum.close()
