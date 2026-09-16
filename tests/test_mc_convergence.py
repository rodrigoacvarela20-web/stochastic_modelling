"""Offline checks for the replicated Monte Carlo experiment."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from mc_convergence import run_study, save_plot


class ReplicatedMonteCarloChecks(unittest.TestCase):
    def test_fixed_seed_and_error_statistics(self):
        benchmark, results = run_study(sample_sizes=(100, 400), repetitions=12, seed=42)
        benchmark_again, results_again = run_study(sample_sizes=(100, 400), repetitions=12, seed=42)
        self.assertEqual((benchmark, results), (benchmark_again, results_again))
        self.assertEqual([row[0] for row in results], [100, 400])
        for _, bias, rmse, spread in results:
            self.assertGreaterEqual(rmse, abs(bias))
            self.assertGreater(spread, 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'figure.png'
            save_plot(results, output)
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
