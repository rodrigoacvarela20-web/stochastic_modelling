"""Offline quote checks; NOT tests of QuantLib pricing or live TSLA calibration."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'projects' / 'quant_finance'))
from project4_heston_calibration import filter_option_quotes


class QuoteChecks(unittest.TestCase):
    def test_filter_drops_crossed_nonfinite_zero_bid_and_wide_quotes(self):
        quotes = pd.DataFrame({
            'strike': [100, 100, 100, 100, 100, 100, 100, 100],
            'bid': [10, 11, 0, np.nan, 5, 3, np.inf, 10],
            'ask': [11, 10, 2, 5, 10, 3.1, 11, 10.01],
            'openInterest': [30, 30, 30, 30, 30, 30, 30, 2],
        })
        clean = filter_option_quotes(quotes, 100)
        self.assertEqual(len(clean), 2)
        np.testing.assert_allclose(clean['bid'].to_numpy(), [10, 3])
        self.assertTrue(np.all(clean['spread'] > 0))

    def test_missing_fields_and_invalid_spot_raise(self):
        with self.assertRaises(ValueError):
            filter_option_quotes(pd.DataFrame({'bid': [1]}), 100)
        with self.assertRaises(ValueError):
            filter_option_quotes(pd.DataFrame(), 0)


if __name__ == '__main__':
    unittest.main()
