# Reproducibility and scope

This repository contains learning and research prototypes, not validated trading strategies or measured trading performance.

## Offline checks

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use fixed random seeds. The original demonstration scripts do **not** set seeds, so their estimates and plots can vary between runs. The Monte Carlo option price is a point estimate without an explicit confidence interval in the demonstration.

## Market data

Install optional dependencies with `python -m pip install -r requirements-market.txt`. Projects 04 and 05 retrieve live Yahoo Finance data, which can be unavailable or change between runs. The repository has no archived option-chain snapshot, so historic calibration outputs cannot be recreated exactly. The Heston calibration fits **European-call model prices** to TSLA quotes; listed US equity options may have American exercise features. Risk-free and dividend rates are flat assumptions, not independently fitted market curves. No market-data integration tests are included.

## Numerical assumptions

- Euler–Maruyama and the full-truncation-style Heston discretisation incur timestep bias, while Monte Carlo estimates incur sampling error.
- The Heston variance is clipped at zero, an approximation that can affect its numerical distribution. The simulation and calibration use different numerical pricing methods.
- GARCH(1,1) uses a Gaussian likelihood and a fixed sample mean. An optimiser success flag does not establish parameter stability or out-of-sample forecasting performance.
- These projects make no claim of live trading profitability, market forecasting advantage, or production-grade calibration.

The uploaded `monte_carlo` executable is a compiled Linux binary, not the corresponding C++ source; it is deliberately excluded from publication.
