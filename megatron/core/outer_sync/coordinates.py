"""A canonical, tiled view of existing master parameters; no full flat copy."""

from bisect import bisect_right
import hashlib
import json

import torch


class ParameterCoordinates:
    def __init__(self, named_pairs):
        self.pairs = sorted(named_pairs, key=lambda item: item[0])
        if not self.pairs or len({name for name, _, _ in self.pairs}) != len(self.pairs):
            raise ValueError('nonempty, uniquely named master/model pairs required')
        self.device = self.pairs[0][1].device
        self.offsets = [0]
        self.schema = []
        for name, master, model in self.pairs:
            if (master.dtype != torch.float32 or not master.is_contiguous()
                    or master.device != self.device or model.device != self.device
                    or master.shape != model.shape or not model.is_contiguous()
                    or model.dtype not in (torch.float32, torch.bfloat16)):
                raise ValueError(f'unsupported master/model storage for {name}')
            self.schema.append({'name': name, 'shape': list(master.shape),
                                'model_dtype': str(model.dtype), 'offset': self.offsets[-1],
                                'partition_dim': getattr(model, 'partition_dim', None),
                                'partition_stride': getattr(model, 'partition_stride', None)})
            self.offsets.append(self.offsets[-1] + master.numel())
        if len({id(p) for _, p, _ in self.pairs}) != len(self.pairs):
            raise ValueError('duplicate optimizer masters in coordinate map')
        intervals = sorted((p.data_ptr(), p.data_ptr() + p.numel() * p.element_size())
                           for _, p, _ in self.pairs if p.numel())
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError('overlapping master coordinates')
        self.numel = self.offsets[-1]
        if not self.numel:
            raise ValueError('empty optimizer coordinate map')
        self.fingerprint = hashlib.sha256(json.dumps(self.schema, sort_keys=True).encode()).hexdigest()

    def _segments(self, start, count):
        if start < 0 or count < 0 or start + count > self.numel:
            raise ValueError('coordinate range outside master vector')
        cursor = 0
        while cursor < count:
            index = bisect_right(self.offsets, start + cursor) - 1
            local = start + cursor - self.offsets[index]
            size = min(count - cursor, self.offsets[index + 1] - start - cursor)
            yield self.pairs[index][1].view(-1)[local:local + size], cursor, size
            cursor += size

    @torch.no_grad()
    def read_into(self, start, target):
        for master, offset, size in self._segments(start, target.numel()):
            target[offset:offset + size].copy_(master)

    @torch.no_grad()
    def write_from(self, start, source):
        for master, offset, size in self._segments(start, source.numel()):
            master.copy_(source[offset:offset + size])

    def storage_tensors(self):
        return [p for _, master, model in self.pairs for p in (master, model)]

    def assert_model_committed(self, *, finite=False):
        for name, master, model in self.pairs:
            source, target = master.detach().view(-1), model.detach().view(-1)
            for start in range(0, source.numel(), 262144):
                expected = source[start:start + 262144].to(model.dtype)
                if finite and not bool(torch.isfinite(expected).all()):
                    raise AssertionError(f'nonfinite final model parameter: {name} at {start}')
                if not torch.equal(target[start:start + 262144].view(torch.uint8), expected.view(torch.uint8)):
                    raise AssertionError(f'next forward sees stale model parameter: {name} at {start}')

    def cpu_flat(self):
        # Validation only: explicitly outside the executor workspace accounting.
        return torch.cat([p.detach().cpu().view(-1) for _, p, _ in self.pairs])


def optimizer_coordinates(model, optimizer):
    names = {}
    for chunk, module in enumerate(model):
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                names.setdefault(id(parameter), f'chunk{chunk}.{name}')
    pairs = optimizer.outer_parameter_pairs()
    if {id(model_param) for _, model_param in pairs} != set(names):
        raise ValueError('optimizer/model coordinate coverage differs')
    return ParameterCoordinates([(names[id(model_param)], master, model_param)
                                 for master, model_param in pairs])
