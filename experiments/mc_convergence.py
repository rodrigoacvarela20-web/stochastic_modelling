"""Replicated, offline Monte Carlo convergence study for European call prices."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'projects' / 'quant_finance'))
from project2_montecarlo_eu_option_pricing import black_scholes_call_price, monte_carlo_option_pricing


def run_study(sample_sizes=(100, 400, 1600, 6400), repetitions=200, seed=2026):
    if repetitions < 2 or not all(isinstance(n, int) and n > 1 for n in sample_sizes):
        raise ValueError('Use at least two repetitions and positive integer sample sizes greater than one.')
    s0, strike, rate, volatility, maturity = 100, 100, 0.05, 0.20, 1
    benchmark = black_scholes_call_price(s0, strike, rate, volatility, maturity)
    rng = np.random.default_rng(seed)
    results = []
    for n in sample_sizes:
        estimates = np.array([monte_carlo_option_pricing(s0, strike, rate, volatility, maturity, 'call', n, rng) for _ in range(repetitions)])
        errors = estimates - benchmark
        results.append((n, float(np.mean(errors)), float(np.sqrt(np.mean(errors**2))), float(np.std(estimates, ddof=1))))
    return benchmark, results


def save_plot(results, output):
    counts, _, rmse, _ = map(np.asarray, zip(*results))
    reference = rmse[0] * np.sqrt(counts[0] / counts)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(counts, rmse, 'o-', label='Replicated empirical RMSE')
    ax.loglog(counts, reference, '--', label=r'$N^{-1/2}$ guide (anchored at first point)')
    ax.set(xlabel='Simulations per estimate', ylabel='RMSE of discounted call price', title='European call: Monte Carlo sampling error')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', type=Path, default=Path('assets/monte_carlo_convergence.png'))
    args = parser.parse_args()
    benchmark, results = run_study()
    print(f'Black-Scholes benchmark: {benchmark:.6f}')
    print('N       Bias          RMSE        Across-trial SD')
    for n, bias, rmse, spread in results:
        print(f'{n:<7} {bias:>+10.6f} {rmse:>12.6f} {spread:>17.6f}')
    save_plot(results, args.plot)
    print(f'Figure saved to {args.plot}')


if __name__ == '__main__':
    main()
