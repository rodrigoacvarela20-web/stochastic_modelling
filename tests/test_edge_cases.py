"""Boundary conditions and estimation guards for financial examples."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'projects'/'quant_finance'))
from project2_montecarlo_eu_option_pricing import black_scholes_call_price, black_scholes_put_price, monte_carlo_option_pricing
from project5_GARCH import estimate_garch, negative_log_likelihood, garch_variance


class FinancialBoundaryChecks(unittest.TestCase):
    def test_zero_maturity_equals_intrinsic_value(self):
        self.assertEqual(black_scholes_call_price(100, 110, .05, .2, 0), 0)
        self.assertEqual(black_scholes_put_price(100, 110, .05, .2, 0), 10)
        self.assertEqual(monte_carlo_option_pricing(100, 110, .05, .2, 0, 'put', 5), 10)

    def test_zero_volatility_is_deterministic_discounted_payoff(self):
        answer = 100-90*np.exp(-.04)
        self.assertAlmostEqual(black_scholes_call_price(100, 90, .04, 0, 1), answer)
        self.assertAlmostEqual(monte_carlo_option_pricing(100, 90, .04, 0, 1, 'call', 100), answer)
        self.assertEqual(black_scholes_put_price(100, 90, .04, 0, 1), 0)

    def test_zero_strike_call_is_spot_and_put_zero(self):
        self.assertAlmostEqual(black_scholes_call_price(100, 0, .04, .2, 1), 100)
        self.assertEqual(black_scholes_put_price(100, 0, .04, .2, 1), 0)

    def test_invalid_garch_parameters_rejected(self):
        with self.assertRaises(ValueError):
            garch_variance([.01, -.01], .0001, .6, .5, 0)
        self.assertTrue(np.isinf(negative_log_likelihood([.0001, .6, .5], np.array([.01, -.01]), 0)))
        with self.assertRaises(ValueError):
            estimate_garch(np.ones(25), 1)


if __name__ == '__main__':
    unittest.main()
