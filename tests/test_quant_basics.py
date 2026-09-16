"""Offline numerical sanity checks; no financial-data downloads."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'projects' / 'quant_finance'))
from project1_sde_simulator import euler_maruyama, exact_gbm
from project2_montecarlo_eu_option_pricing import black_scholes_call_price, black_scholes_put_price, monte_carlo_option_pricing
from project3_heston_model import simulate_heston
from project5_GARCH import garch_variance


class NumericalChecks(unittest.TestCase):
    def test_gbm_same_brownian_path(self):
        np.random.seed(6)
        t, approximated, brownian = euler_maruyama(100, 0.05, 0.20, 0.10, 0.01)
        exact = exact_gbm(100, 0.05, 0.20, t, brownian)
        self.assertEqual(approximated.shape, exact.shape)
        self.assertAlmostEqual(exact[0], 100)
        self.assertTrue(np.isfinite(exact).all())

    def test_black_scholes_put_call_parity(self):
        s0, strike, r, sigma, maturity = 100, 105, 0.03, 0.2, 1
        call = black_scholes_call_price(s0, strike, r, sigma, maturity)
        put = black_scholes_put_price(s0, strike, r, sigma, maturity)
        self.assertAlmostEqual(call - put, s0 - strike * np.exp(-r * maturity), places=10)

    def test_monte_carlo_converges_to_black_scholes_in_example(self):
        np.random.seed(42)
        estimate = monte_carlo_option_pricing(100, 100, 0.05, 0.2, 1, 'call', 25000)
        benchmark = black_scholes_call_price(100, 100, 0.05, 0.2, 1)
        self.assertLess(abs(estimate - benchmark), 0.35)

    def test_heston_constant_variance_when_vol_of_vol_zero(self):
        np.random.seed(9)
        time, stock, variance = simulate_heston(100, 0.04, 0.03, 2, 0.04, 0, 0, 0.1, 0.01)
        np.testing.assert_allclose(variance, 0.04)
        self.assertEqual(len(stock), len(time))

    def test_garch_recursion(self):
        returns = np.array([0.01, -0.02, 0.03])
        variance = garch_variance(returns, 0.0001, 0.1, 0.85, 0)
        self.assertAlmostEqual(variance[1], 0.0001 + 0.1 * returns[0] ** 2 + 0.85 * variance[0])


if __name__ == '__main__':
    unittest.main()
