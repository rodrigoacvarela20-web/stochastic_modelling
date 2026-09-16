# Stochastic Modelling & Quantitative Methods

**Numerical experiments in stochastic differential equations, option pricing, and volatility modelling.**

A collection of five Python projects exploring stochastic simulation, European option pricing, Heston stochastic volatility, calibration and GARCH(1,1). These are educational and exploratory mathematical models, **not validated trading strategies or investment advice**.

## Projects

| Project | Question | Source |
|:--|:--|:--|
| 01 · Geometric Brownian Motion | How does Euler–Maruyama compare with an exact solution on the same Brownian path? | [SDE simulator](projects/quant_finance/project1_sde_simulator.py) |
| 02 · Monte Carlo Option Pricing | How does sampling error affect European call and put estimates relative to Black–Scholes? | [European options](projects/quant_finance/project2_montecarlo_eu_option_pricing.py) |
| 03 · Heston Stochastic Volatility | How do correlated price and variance shocks change paths and option prices? | [Heston simulation](projects/quant_finance/project3_heston_model.py) |
| 04 · Heston Calibration | How closely can European-call Heston model prices fit a filtered live TSLA options chain? | [Calibration](projects/quant_finance/project4_heston_calibration.py) |
| 05 · GARCH(1,1) | How can conditional variance be estimated from historical log returns? | [GARCH](projects/quant_finance/project5_GARCH.py) |

## Mathematical foundations

Geometric Brownian motion and its exact solution:

$$
dS_t=\mu S_t\,dt+\sigma S_t\,dW_t,\qquad S_t=S_0\exp\!\left[(\mu-\tfrac12\sigma^2)t+\sigma W_t\right].
$$

GARCH(1,1) conditional variance:

$$
h_t=\omega+\alpha(r_{t-1}-\mu)^2+\beta h_{t-1}.
$$

## Running the examples

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python projects/quant_finance/project1_sde_simulator.py
python projects/quant_finance/project2_montecarlo_eu_option_pricing.py
python projects/quant_finance/project3_heston_model.py
```

For the scripts requiring live market data and optional QuantLib:

```bash
python -m pip install -r requirements-market.txt
python projects/quant_finance/project4_heston_calibration.py
python projects/quant_finance/project5_GARCH.py
```

The demonstrations display interactive Matplotlib figures. For assumptions and reproducibility limitations, see [Reproducibility](docs/REPRODUCIBILITY.md).

## Scope and limitations

- Simulated prices have discretisation and Monte Carlo sampling error; the original demonstration scripts are not seeded.
- The calibration uses live Yahoo Finance quotes and cannot be reconstructed exactly without a historical option-chain snapshot. The listed TSLA options may have American exercise features, whereas the model prices European calls.
- No trading performance, profitable strategy, or empirical predictive advantage is claimed.

**Author:** [Rodrigo Varela](https://github.com/rodrigoacvarela20-web) · Engineering Physics, Instituto Superior Técnico · Incoming MSc student, Imperial College London