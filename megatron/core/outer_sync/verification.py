"""Exact tensor comparison used only by explicit correctness gates."""

import torch


def same(actual, expected, label):
    actual = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    expected = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    if not torch.equal(actual, expected):
        raise AssertionError(f'{label}: bit mismatch')
