"""Localize E0c FP32/TP1 MLP gradient discrepancies without changing arithmetic.

FP64 sums here use the captured FP32 operands. They are not a full-model FP64
reference and never override the E0c numerical acceptance contract.
"""

import math

import torch


def probe_layer(architecture):
    # Layer 2 failed on real 3B; the last layer supports smaller CPU fixtures.
    return min(2, architecture.layers - 1)


class LinearTrace:
    """Capture one linear call in canonical [batch, sequence, channel] order."""

    def __init__(self, module, *, sequence_first=False):
        self.module = module
        self.sequence_first = sequence_first
        self.tensors = {}
        self.handles = []

    def _copy(self, value):
        if value.ndim != 3 or value.dtype != torch.float32:
            raise ValueError('linear probe requires a three-dimensional FP32 tensor')
        value = value.detach()
        if self.sequence_first:
            value = value.transpose(0, 1)
        return value.to(device='cpu', copy=True).contiguous()

    def _forward(self, module, args, output):
        if self.tensors:
            raise ValueError('linear probe expects exactly one forward/backward call')
        self.tensors['input'] = self._copy(args[0])
        value = output[0] if isinstance(output, tuple) else output
        self.handles.append(value.register_hook(self._backward))
        # Returning None leaves the real forward result unchanged.

    def _backward(self, grad_output):
        if 'grad_output' in self.tensors:
            raise ValueError('linear probe expects exactly one backward call')
        self.tensors['grad_output'] = self._copy(grad_output)
        # Returning None leaves the real backward gradient unchanged.

    def __enter__(self):
        self.handles.append(self.module.register_forward_hook(self._forward))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()

    def complete(self):
        if set(self.tensors) != {'input', 'grad_output'}:
            raise ValueError('linear probe is missing forward/backward operands')
        return self.tensors


def _sum(value):
    return math.fsum(value.tolist())


def _vector_difference(actual, reference):
    error_sq = _sum((actual - reference).square())
    reference_sq = _sum(reference.square())
    return {'max_abs': float((actual - reference).abs().max()),
            'relative_l2': math.sqrt(error_sq / reference_sq) if reference_sq else None}


def analyze_linear(reference, actual, comparison):
    """Decompose selected dW differences into local residuals and operand changes."""
    for name in ('input', 'grad_output'):
        a, b = actual[name], reference[name]
        if (a.shape != b.shape or a.ndim != 3 or a.dtype != torch.float32
                or b.dtype != torch.float32 or not bool(torch.isfinite(a).all())
                or not bool(torch.isfinite(b).all())):
            raise ValueError(f'linear probe requires finite matching FP32 {name} tensors')
    batch, sequence, channels = reference['input'].shape
    if reference['grad_output'].shape[:2] != (batch, sequence):
        raise ValueError('linear probe token dimensions differ')
    outputs = reference['grad_output'].shape[-1]
    if comparison['shape'] != [outputs, channels]:
        raise ValueError('linear probe weight shape differs from comparison')
    selected = list(comparison['outside_tolerance_samples'])
    worst = comparison['worst_element']
    if worst is not None and all(p['index'] != worst['index'] for p in selected):
        selected.append(worst)
    points = []
    for point in selected:
        row, column = point['index']
        if not (0 <= row < outputs and 0 <= column < channels):
            raise ValueError('linear probe coordinate is outside the weight')
        hx = reference['input'][..., column].reshape(-1).double()
        nx = actual['input'][..., column].reshape(-1).double()
        hg = reference['grad_output'][..., row].reshape(-1).double()
        ng = actual['grad_output'][..., row].reshape(-1).double()
        hp, np = hx * hg, nx * ng
        hd, nd = _sum(hp), _sum(np)
        observed = point['actual'] - point['reference']
        native_residual, hf_residual = point['actual'] - nd, point['reference'] - hd
        operand_delta = nd - hd
        activation_term = _sum((nx - hx) * hg)
        incoming_term = _sum(nx * (ng - hg))
        # Large signed terms that nearly cancel make the dot product sensitive.
        hsum, nsum = _sum(hp.abs()), _sum(np.abs())
        indices = (np - hp).abs().argsort(descending=True, stable=True)[:4].tolist()
        tokens = [{'batch': k // sequence, 'position': k % sequence,
                   'hf_activation': float(hx[k]), 'native_activation': float(nx[k]),
                   'hf_incoming_gradient': float(hg[k]), 'native_incoming_gradient': float(ng[k]),
                   'operand_product_delta_fp64': float(np[k] - hp[k])} for k in indices]
        points.append({**point, 'hf_dot_fp64': hd, 'native_dot_fp64': nd,
                       'observed_native_minus_hf': observed,
                       'native_actual_minus_dot_fp64': native_residual,
                       'hf_actual_minus_dot_fp64': hf_residual,
                       'operands_delta_fp64': operand_delta,
                       'activation_contribution_fp64': activation_term,
                       'incoming_gradient_contribution_fp64': incoming_term,
                       'decomposition_residual': observed - (native_residual + operand_delta - hf_residual),
                       'operand_split_residual': operand_delta - (activation_term + incoming_term),
                       'hf_sum_abs_products': hsum, 'native_sum_abs_products': nsum,
                       'hf_cancellation_ratio': hsum / abs(hd) if hd else None,
                       'native_cancellation_ratio': nsum / abs(nd) if nd else None,
                       'activation_difference': _vector_difference(nx, hx),
                       'incoming_gradient_difference': _vector_difference(ng, hg),
                       'largest_operand_delta_tokens': tokens})
    return {'status': 'diagnostic_written', 'diagnostic_only': True,
            'performance_result': False, 'acceptance_override': False,
            'method': 'FP64 products and math.fsum over captured FP32 operands; not full-model FP64',
            'token_order': 'batch,sequence', 'batch': batch, 'sequence_length': sequence,
            'weight_shape': [outputs, channels], 'outside_tolerance': comparison['outside_tolerance'],
            'points': points}
