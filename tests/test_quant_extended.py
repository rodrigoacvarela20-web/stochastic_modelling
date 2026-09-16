"""Additional offline checks for quantitative examples; no market-data downloads."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'projects' / 'quant_finance'))
from project1_sde_simulator import euler_maruyama, exact_gbm
from project2_montecarlo_eu_option_pricing import black_scholes_call_price, black_scholes_put_price, monte_carlo_option_pricing
from project3_heston_model import simulate_heston, monte_carlo_heston_option_pricing


class ExtendedNumericalChecks(unittest.TestCase):
    def test_gbm_zero_volatility_is_deterministic(self):
        t, approximated, brownian = euler_maruyama(100, .05, 0, 1, .01)
        self.assertAlmostEqual(approximated[-1], 100 * (1 + .05 * .01)**100, places=10)
        self.assertAlmostEqual(exact_gbm(100, .05, 0, t, brownian)[-1], 100 * np.exp(.05), places=10)

    def test_gbm_refinement_same_brownian_path(self):
        increments = np.random.default_rng(123).standard_normal(100)
        coarse = increments.reshape(10, 10).sum(axis=1) / np.sqrt(10)
        with patch('numpy.random.normal', side_effect=coarse):
            _, coarse_x, coarse_w = euler_maruyama(100, .05, .2, 1, .1)
        with patch('numpy.random.normal', side_effect=increments):
            t, fine_x, fine_w = euler_maruyama(100, .05, .2, 1, .01)
        self.assertAlmostEqual(coarse_w[-1], fine_w[-1], places=12)
        exact = exact_gbm(100, .05, .2, t, fine_w)[-1]
        self.assertLess(abs(fine_x[-1] - exact), abs(coarse_x[-1] - exact))

    def test_zero_strike_put_and_call(self):
        np.random.seed(12)
        self.assertEqual(monte_carlo_option_pricing(100, 0, .05, .2, 1, 'put', 500), 0)
        np.random.seed(12)
        call = monte_carlo_option_pricing(100, 0, .05, .2, 1, 'call', 16000)
        self.assertLess(abs(call - 100), 2.0)

    def test_black_scholes_strike_monotonicity(self):
        strikes = [80, 100, 120]
        calls = [black_scholes_call_price(100, k, .03, .2, 1) for k in strikes]
        puts = [black_scholes_put_price(100, k, .03, .2, 1) for k in strikes]
        self.assertTrue(all(calls[i] > calls[i+1] and puts[i] < puts[i+1] for i in range(2)))

    def test_heston_constant_variance(self):
        np.random.seed(31)
        t, stock, variance = simulate_heston(100, .04, .04, 2, .04, 0, -.7, .5, .01)
        np.testing.assert_allclose(variance, .04, atol=1e-13)
        self.assertEqual(len(t), len(stock))

    def test_heston_call_decreases_with_strike_on_common_paths(self):
        params = (100, .04, .02, 2, .04, .3, -.6, .2, .01)
        np.random.seed(16)
        lower = monte_carlo_heston_option_pricing(*params, 90, 'call', 300)
        np.random.seed(16)
        higher = monte_carlo_heston_option_pricing(*params, 110, 'call', 300)
        self.assertGreaterEqual(lower, higher)


if __name__ == '__main__':
    unittest.main()
