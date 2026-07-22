"""Focused regression tests for deterministic post-DDIM constraints."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from trajectory.constraints import _deterministic_cumsum_1d


class DeterministicConstraintTests(unittest.TestCase):
    def test_cumsum_runs_on_cpu_and_preserves_source_contract(self):
        values = torch.tensor(
            [0.25, 1.5, 0.75, 2.0], dtype=torch.float32, requires_grad=True
        )
        expected = torch.tensor([0.25, 1.75, 2.5, 4.5], dtype=torch.float32)

        original_cumsum = torch.cumsum
        with mock.patch(
            "trajectory.constraints.torch.cumsum", wraps=original_cumsum
        ) as cumsum:
            observed = _deterministic_cumsum_1d(values)

        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        self.assertEqual(cumsum.call_count, 1)
        self.assertEqual(cumsum.call_args.args[0].device.type, "cpu")
        self.assertFalse(cumsum.call_args.args[0].requires_grad)
        self.assertEqual(observed.device, values.device)
        self.assertEqual(observed.dtype, values.dtype)
        self.assertFalse(observed.requires_grad)

    def test_cumsum_rejects_non_vector_input(self):
        with self.assertRaisesRegex(ValueError, "expects a 1-D tensor"):
            _deterministic_cumsum_1d(torch.ones(2, 2, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
