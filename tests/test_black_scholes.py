"""Tests for Black--Scholes and Monte Carlo pricing."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "black_scholes"))

from option_pricing import black_scholes_price, monte_carlo_price


class BlackScholesTests(unittest.TestCase):
    def test_put_call_parity(self):
        s0, strike, r, sigma, T = 100.0, 105.0, 0.03, 0.2, 1.0
        call = black_scholes_price(s0, strike, r, sigma, T, "call")
        put = black_scholes_price(s0, strike, r, sigma, T, "put")
        rhs = s0 - strike * np.exp(-r * T)
        self.assertAlmostEqual(call - put, rhs, places=10)

    def test_zero_maturity(self):
        self.assertEqual(
            black_scholes_price(100.0, 110.0, 0.05, 0.2, 0.0, "call"),
            0.0,
        )
        self.assertEqual(
            black_scholes_price(100.0, 110.0, 0.05, 0.2, 0.0, "put"),
            10.0,
        )

    def test_mc_is_consistent_with_black_scholes(self):
        exact = black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0, "call")
        estimate, se, _ = monte_carlo_price(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            "call",
            100_000,
            np.random.default_rng(2026),
        )
        self.assertLess(abs(estimate - exact), 4.0 * se)


if __name__ == "__main__":
    unittest.main()
