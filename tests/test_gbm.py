"""Tests for the GBM numerical example."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gbm"))

from gbm_simulation import euler_maruyama, exact_gbm, terminal_moments


class GBMTests(unittest.TestCase):
    def test_exact_solution_starts_at_s0(self):
        rng = np.random.default_rng(1)
        t, _, W = euler_maruyama(100.0, 0.05, 0.2, 1.0, 0.01, rng)
        exact = exact_gbm(100.0, 0.05, 0.2, t, W)
        self.assertAlmostEqual(exact[0], 100.0)

    def test_zero_volatility(self):
        rng = np.random.default_rng(2)
        t, approx, W = euler_maruyama(100.0, 0.05, 0.0, 1.0, 0.01, rng)
        exact = exact_gbm(100.0, 0.05, 0.0, t, W)
        self.assertAlmostEqual(exact[-1], 100.0 * np.exp(0.05), places=12)
        self.assertGreater(approx[-1], 100.0)

    def test_terminal_moments(self):
        mean, variance = terminal_moments(100.0, 0.05, 0.2, 1.0)
        self.assertAlmostEqual(mean, 100.0 * np.exp(0.05), places=12)
        self.assertGreater(variance, 0.0)


if __name__ == "__main__":
    unittest.main()
