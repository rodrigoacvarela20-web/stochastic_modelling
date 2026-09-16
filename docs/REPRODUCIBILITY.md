# Reproducibility and scope

This repository contains four educational stochastic-modelling and pricing prototypes, not validated trading strategies or measured trading performance.

## Offline checks

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Tests use fixed random seeds where needed. Individual Monte Carlo estimates have sampling uncertainty; plotted absolute errors from a single run should not be interpreted as confidence intervals or monotonic convergence guarantees.

## Market data

Install optional dependencies with `python -m pip install -r requirements-market.txt`. Project 04 retrieves live Yahoo Finance data, which can be unavailable or change between runs. There is no archived synchronized option-chain snapshot, so historic calibration outputs cannot be recreated exactly. The Heston calibration fits **European-call model prices** to TSLA quotes; listed US equity options may have American exercise features. Risk-free and dividend rates are flat assumptions, not independently fitted market curves. There is no full market-data integration test.

## Numerical assumptions

- Euler–Maruyama and the Heston variance discretisation incur timestep bias, while Monte Carlo incurs sampling error.
- Variance truncation is an approximation that can affect its numerical distribution. Heston simulation and calibration use different numerical pricing methods.
- Neither the code nor the tests establish a profitable trading strategy, market-forecasting advantage or production-grade calibration.

The uploaded `monte_carlo` executable from the original source materials was a compiled Linux binary, not corresponding C++ source; it was excluded from publication.
