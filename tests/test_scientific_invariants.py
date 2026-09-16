"""Independent mathematical controls for finance demonstrations; no live data."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'projects' / 'quant_finance'))
from project1_sde_simulator import euler_maruyama, exact_gbm
from project2_montecarlo_eu_option_pricing import black_scholes_call_price, black_scholes_put_price
from project3_heston_model import simulate_heston, simulate_terminal_heston
from project5_GARCH import garch_variance


class ScientificInvariants(unittest.TestCase):
    def test_gbm_grid_and_brownian_path_use_same_dt(self):
        t, x, w = euler_maruyama(100, .1, 0., 1., .3, np.random.default_rng(1))
        np.testing.assert_allclose(np.diff(t), .25)
        self.assertAlmostEqual(x[-1], 100*(1 + .1*.25)**4, places=12)
        self.assertAlmostEqual(w[0], 0.)
        self.assertAlmostEqual(exact_gbm(100, .1, 0., t, w)[-1], 100*np.exp(.1), places=12)

    def test_heston_deterministic_grid_and_price_positivity(self):
        t, s, v = simulate_heston(100, 0., .2, 0., 0., 0., 0., .25, .1, np.random.default_rng(4))
        self.assertAlmostEqual(t[-1], .25)
        self.assertAlmostEqual(s[-1], 100*np.exp(.2*.25), places=12)
        np.testing.assert_array_equal(v, np.zeros(len(v)))
        self.assertGreater(np.min(s), 0.)

    def test_heston_terminal_matches_full_path_with_same_shocks(self):
        params = (100., .04, .05, 2., .04, .3, -.7, .25, .1)
        path = simulate_heston(*params, rng=np.random.default_rng(13))[1][-1]
        terminal = simulate_terminal_heston(*params, rng=np.random.default_rng(13))
        self.assertAlmostEqual(path, terminal, places=12)

    def test_zero_vol_of_vol_sample_mean_matches_gbm_with_sampling_error(self):
        values = np.array([simulate_terminal_heston(100., .04, .04, 2., .04, 0., -.5, 1., .2, np.random.default_rng(seed)) for seed in range(3000)])
        se = values.std(ddof=1)/np.sqrt(len(values))
        self.assertLess(abs(values.mean() - 100*np.exp(.04)), 4.5*se)

    def test_garch_recursion_nonnegative(self):
        r = np.array([.01, -.025, .018, .002])
        v = garch_variance(r, 1e-5, .1, .85, 0.)
        self.assertTrue(np.all(v > 0))
        self.assertAlmostEqual(v[1], 1e-5 + .1*r[0]**2 + .85*v[0])

    def test_european_put_call_parity(self):
        c = black_scholes_call_price(100, 110, .04, .25, 1.5)
        p = black_scholes_put_price(100, 110, .04, .25, 1.5)
        self.assertAlmostEqual(c-p, 100-110*np.exp(-.04*1.5), places=10)


if __name__ == '__main__':
    unittest.main()
