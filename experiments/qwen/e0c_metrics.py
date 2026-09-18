"""Fixed E0c conversion tolerances and bounded-memory whole-tensor comparisons.

BF16 permits rounding differences between HF and Megatron/TP operation orders.
It is a numerical conversion check, never evidence of bitwise training parity.
Thresholds are versioned here, not adjusted by a launcher or after observing a run.
"""

import math

CONTRACT = {
    'version': 1,
    'fp32': {'logits': {'atol': 2e-4, 'rtol': 2e-4},
             'gradient': {'atol': 2e-5, 'rtol': 2e-4},
             'loss': {'atol': 2e-5, 'rtol': 2e-5}},
    'bf16': {'logits': {'relative_l2_max': .02, 'cosine_min': .999},
             'gradient': {'relative_l2_max': .15, 'cosine_min': .99},
             'loss': {'atol': .05, 'rtol': 0.}},
}


def accepts(stats, dtype, kind):
    """Recompute acceptance from raw statistics, including per-parameter checks."""
    keys = ('error_sq', 'reference_sq', 'actual_sq', 'dot', 'max_abs')
    if stats['elements'] < 1 or stats['nonfinite'] or not all(math.isfinite(stats[k]) for k in keys):
        return False
    limits = CONTRACT[dtype][kind]
    if 'atol' in limits:
        return stats['outside_tolerance'] == 0
    if stats['reference_sq'] == 0.:
        return stats['actual_sq'] == 0.
    relative_l2 = math.sqrt(stats['error_sq'] / stats['reference_sq'])
    denominator = math.sqrt(stats['actual_sq'] * stats['reference_sq'])
    cosine = stats['dot'] / denominator if denominator else 0.
    return relative_l2 <= limits['relative_l2_max'] and cosine >= limits['cosine_min']


def compare_tensor(actual, expected, dtype, kind, chunk_elements=1 << 20):
    """Inspect every element; transfer only bounded chunks from GPU to CPU."""
    import torch
    if actual.shape != expected.shape:
        raise ValueError(f'shape mismatch: {actual.shape} != {expected.shape}')
    actual, expected = actual.detach().reshape(-1), expected.detach().reshape(-1)
    result = dict(elements=actual.numel(), nonfinite=0, changed=0, outside_tolerance=0,
                  error_sq=0., reference_sq=0., actual_sq=0., dot=0., max_abs=0.)
    limits = CONTRACT[dtype][kind]
    for start in range(0, actual.numel(), chunk_elements):
        a = actual[start:start + chunk_elements].to(device='cpu', dtype=torch.float64)
        b = expected[start:start + chunk_elements].to(device='cpu', dtype=torch.float64)
        finite = torch.isfinite(a) & torch.isfinite(b)
        result['nonfinite'] += int((~finite).sum())
        result['changed'] += int((a != b).sum())
        if not bool(finite.all()):
            # Preserve valid JSON rather than NaN/Infinity on a failed run.
            a, b = a[finite], b[finite]
        if a.numel() == 0:
            continue
        delta = a - b
        result['error_sq'] += float(delta.dot(delta))
        result['actual_sq'] += float(a.dot(a))
        result['reference_sq'] += float(b.dot(b))
        result['dot'] += float(a.dot(b))
        result['max_abs'] = max(result['max_abs'], float(delta.abs().max()))
        if 'atol' in limits:
            result['outside_tolerance'] += int((delta.abs() > limits['atol'] + limits['rtol'] * b.abs()).sum())
    result['relative_l2'] = (math.sqrt(result['error_sq'] / result['reference_sq'])
                             if result['reference_sq'] else None)
    denominator = math.sqrt(result['actual_sq'] * result['reference_sq'])
    result['cosine'] = result['dot'] / denominator if denominator else None
    result['passed'] = accepts(result, dtype, kind)
    return result


def exact_tensor(actual, expected, chunk_elements=1 << 20):
    import torch
    if actual.shape != expected.shape:
        raise ValueError('weight shape mismatch')
    a, b = actual.detach().reshape(-1), expected.detach().reshape(-1)
    equal = True
    for start in range(0, a.numel(), chunk_elements):
        left = a[start:start + chunk_elements].cpu().contiguous()
        right = b[start:start + chunk_elements].to(device='cpu', dtype=a.dtype).contiguous()
        equal &= torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    return {'elements': a.numel(), 'bitwise_equal': equal}
