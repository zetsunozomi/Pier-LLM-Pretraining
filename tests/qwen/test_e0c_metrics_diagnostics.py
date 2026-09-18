"""Bounded numerical diagnostics; these CPU tests are not E0c GPU evidence."""

import json
import unittest

import torch

from experiments.qwen.e0c_metrics import OUTLIER_SAMPLE_LIMIT, accepts, compare_tensor


class E0cDiagnosticTests(unittest.TestCase):
    def test_coordinates_and_worst_tolerance_ratio_across_chunks(self):
        expected = torch.zeros(3, 5, dtype=torch.float64)
        expected[0, 0] = 10.
        actual = expected.clone()
        actual[0, 0] += .001  # Largest absolute error, but within its tolerance.
        actual[0, 1], actual[1, 3], actual[2, 4] = 2.1e-5, -3e-5, 1e-4
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                a, b = (actual.T, expected.T) if transpose else (actual, expected)
                stats = compare_tensor(a, b, 'fp32', 'gradient', chunk_elements=4)
                self.assertFalse(stats['passed'])
                self.assertEqual(stats['outside_tolerance'], 3)
                self.assertEqual(stats['shape'], list(a.shape))
                coordinates = [[1, 0], [3, 1], [4, 2]] if transpose else [[0, 1], [1, 3], [2, 4]]
                self.assertEqual([x['index'] for x in stats['outside_tolerance_samples']], coordinates)
                worst = stats['worst_element']
                self.assertEqual(worst['index'], coordinates[-1])
                self.assertEqual(worst['flat_index'], 14)
                self.assertAlmostEqual(worst['ratio'], 5.)
                self.assertAlmostEqual(worst['actual'], 1e-4)
                self.assertEqual(worst['reference'], 0.)
                self.assertGreater(stats['max_abs'], worst['absolute_error'])

    def test_samples_are_bounded_without_losing_counts_or_worst_point(self):
        actual = torch.arange(1, 17, dtype=torch.float64).reshape(4, 4) * 5e-5
        expected = torch.zeros_like(actual)
        for chunk in (3, 5, 100):
            with self.subTest(chunk=chunk):
                stats = compare_tensor(actual, expected, 'fp32', 'gradient', chunk)
                self.assertEqual(stats['outside_tolerance'], 16)
                self.assertFalse(accepts(stats, 'fp32', 'gradient'))
                samples = stats['outside_tolerance_samples']
                self.assertEqual(len(samples), OUTLIER_SAMPLE_LIMIT)
                self.assertEqual([x['flat_index'] for x in samples], list(range(OUTLIER_SAMPLE_LIMIT)))
                self.assertEqual(stats['worst_element']['flat_index'], 15)
                self.assertEqual(stats['worst_element']['index'], [3, 3])

    def test_nonfinite_filter_preserves_original_coordinates_and_valid_json(self):
        actual = torch.zeros(3, 4, dtype=torch.float64)
        expected = actual.clone()
        actual[0, 0], actual[1, 0], actual[1, 3] = float('nan'), float('inf'), 1e-4
        stats = compare_tensor(actual, expected, 'fp32', 'gradient', chunk_elements=5)
        self.assertEqual(stats['nonfinite'], 2)
        self.assertFalse(stats['passed'])
        self.assertEqual(stats['outside_tolerance'], 1)
        self.assertEqual(stats['worst_element']['flat_index'], 7)
        self.assertEqual(stats['worst_element']['index'], [1, 3])
        self.assertEqual(stats['outside_tolerance_samples'][0]['index'], [1, 3])
        json.dumps(stats, allow_nan=False)
        invalid = compare_tensor(torch.tensor([float('nan')]), torch.zeros(1), 'fp32', 'gradient')
        self.assertIsNone(invalid['worst_element'])
        self.assertEqual(invalid['outside_tolerance_samples'], [])
        self.assertFalse(invalid['passed'])
        json.dumps(invalid, allow_nan=False)

    def test_pass_scalar_and_bf16_acceptance_are_unchanged(self):
        scalar = torch.tensor(1.)
        exact = compare_tensor(scalar, scalar.clone(), 'fp32', 'loss')
        self.assertTrue(exact['passed'])
        self.assertEqual(exact['shape'], [])
        self.assertEqual(exact['worst_element']['index'], [])
        self.assertEqual(exact['worst_element']['ratio'], 0.)
        self.assertEqual(exact['outside_tolerance_samples'], [])
        failed = compare_tensor(scalar + .1, scalar, 'fp32', 'loss')
        self.assertFalse(failed['passed'])
        self.assertEqual(failed['outside_tolerance_samples'][0]['index'], [])
        bf16 = compare_tensor(scalar * 1.01, scalar, 'bf16', 'gradient')
        self.assertTrue(bf16['passed'])
        self.assertNotIn('worst_element', bf16)


if __name__ == '__main__':
    unittest.main()
