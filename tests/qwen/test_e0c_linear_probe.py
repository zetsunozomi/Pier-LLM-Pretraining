"""CPU checks for diagnostic hooks and a known linear-gradient decomposition."""

import copy
import json
import unittest

import torch

from experiments.qwen.e0c_linear_probe import LinearTrace, analyze_linear
from experiments.qwen.e0c_metrics import compare_tensor


def operands(x, g):
    return {'input': torch.tensor(x, dtype=torch.float32).reshape(1, -1, 1),
            'grad_output': torch.tensor(g, dtype=torch.float32).reshape(1, -1, 1)}


def comparison(actual, reference):
    return compare_tensor(torch.tensor([[actual]], dtype=torch.float32),
                          torch.tensor([[reference]], dtype=torch.float32), 'fp32', 'gradient')


class LinearProbeTests(unittest.TestCase):
    def test_capture_preserves_forward_backward_and_canonical_layout(self):
        torch.manual_seed(421)
        plain = torch.nn.Linear(5, 3, bias=False)
        traced = copy.deepcopy(plain)
        x = torch.randn(2, 7, 5, requires_grad=True)
        tx = x.detach().clone().requires_grad_()
        upstream = torch.randn(2, 7, 3)
        expected = plain(x)
        expected.backward(upstream)
        with LinearTrace(traced) as trace:
            actual = traced(tx)
            actual.backward(upstream)
        for a, b in ((actual, expected), (traced.weight.grad, plain.weight.grad), (tx.grad, x.grad)):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(trace.complete()['input'], x))
        self.assertTrue(torch.equal(trace.complete()['grad_output'], upstream))
        self.assertEqual(len(traced._forward_hooks), 0)
        # Saved values must not alias the original input or incoming gradient.
        with torch.no_grad():
            tx.zero_()
            upstream.zero_()
        self.assertTrue(torch.equal(trace.complete()['input'], x))
        self.assertGreater(float(trace.complete()['grad_output'].abs().sum()), 0)

        class SequenceLinear(torch.nn.Linear):
            def forward(self, inputs):
                return super().forward(inputs), None

        native = SequenceLinear(5, 3, bias=False)
        native.load_state_dict(plain.state_dict())
        sx = x.detach().transpose(0, 1).contiguous().requires_grad_()
        with LinearTrace(native, sequence_first=True) as native_trace:
            result, _ = native(sx)
            result.backward(trace.complete()['grad_output'].transpose(0, 1).contiguous())
        for key in ('input', 'grad_output'):
            self.assertTrue(torch.equal(trace.complete()[key], native_trace.complete()[key]))

    def test_cancellation_is_separated_from_operand_changes(self):
        # FP32 reduction can lose the unit between large opposing terms.
        values = operands([1., 1., 1.], [1e8, 1., -1e8])
        result = analyze_linear(values, values, comparison(0., 1.))
        point = result['points'][0]
        self.assertEqual(point['hf_dot_fp64'], 1.)
        self.assertEqual(point['native_dot_fp64'], 1.)
        self.assertEqual(point['native_actual_minus_dot_fp64'], -1.)
        self.assertEqual(point['hf_actual_minus_dot_fp64'], 0.)
        self.assertEqual(point['operands_delta_fp64'], 0.)
        self.assertEqual(point['activation_contribution_fp64'], 0.)
        self.assertEqual(point['incoming_gradient_contribution_fp64'], 0.)
        self.assertEqual(point['decomposition_residual'], 0.)
        self.assertEqual(point['hf_cancellation_ratio'], 200000001.)
        self.assertFalse(result['acceptance_override'])
        self.assertFalse(result['performance_result'])
        self.assertEqual(len(result['points']), 1)  # worst and violation deduplicated
        json.dumps(result, allow_nan=False)

    def test_activation_and_incoming_gradient_contributions(self):
        reference = operands([1., 2.], [3., 4.])  # dW = 11
        for actual, weight, activation, incoming in (
                (operands([2., 2.], [3., 4.]), 14., 3., 0.),
                (operands([1., 2.], [5., 4.]), 13., 0., 2.),
                (operands([2., 2.], [5., 4.]), 18., 3., 4.)):
            with self.subTest(activation=activation, incoming=incoming):
                point = analyze_linear(reference, actual, comparison(weight, 11.))['points'][0]
                self.assertEqual(point['native_actual_minus_dot_fp64'], 0.)
                self.assertEqual(point['hf_actual_minus_dot_fp64'], 0.)
                self.assertEqual(point['activation_contribution_fp64'], activation)
                self.assertEqual(point['incoming_gradient_contribution_fp64'], incoming)
                self.assertEqual(point['operands_delta_fp64'], activation + incoming)
                self.assertEqual(point['decomposition_residual'], 0.)
                self.assertEqual(point['operand_split_residual'], 0.)

    def test_trace_and_operand_guards(self):
        linear = torch.nn.Linear(1, 1, bias=False)
        with LinearTrace(linear) as trace:
            linear(torch.ones(1, 2, 1))
        with self.assertRaisesRegex(ValueError, 'missing'):
            trace.complete()
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            with LinearTrace(linear):
                linear(torch.ones(1, 2, 1))
                linear(torch.ones(1, 2, 1))
        self.assertEqual(len(linear._forward_hooks), 0)
        valid = operands([1., 2.], [3., 4.])
        for invalid in (operands([float('nan'), 2.], [3., 4.]),
                        operands([1.], [3.]),
                        {k: v.double() for k, v in valid.items()}):
            with self.assertRaises(ValueError):
                analyze_linear(valid, invalid, comparison(11., 11.))
        with self.assertRaisesRegex(ValueError, 'token dimensions'):
            invalid = operands([1., 2.], [3.])
            analyze_linear(invalid, invalid, comparison(11., 11.))
        with self.assertRaisesRegex(ValueError, 'weight shape'):
            analyze_linear(valid, valid, dict(comparison(11., 11.), shape=[2, 1]))


if __name__ == '__main__':
    unittest.main()
