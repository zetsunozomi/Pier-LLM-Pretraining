"""Explicit normalization for Local-SGD's inner gradient collective."""


def inner_gradient_scale(full_size, inner_size, *, inner_average, collective_average):
    if full_size < 1 or inner_size < 1 or full_size % inner_size:
        raise ValueError('inner group must evenly partition the full DP group')
    denominator = inner_size if inner_average else full_size
    return (inner_size if collective_average else 1) / denominator


def warmup_rescale(full_size, inner_size, *, inner_average):
    # A full-DP average of inner-group averages is already the global average.
    inner_gradient_scale(full_size, inner_size, inner_average=inner_average,
                         collective_average=False)
    return 1 if inner_average else full_size // inner_size
