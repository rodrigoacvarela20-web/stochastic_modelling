<div align="center">

![Stochastic modelling — research portfolio banner](assets/portfolio-banner.svg)

# Stochastic Modelling & Quantitative Methods

**Four Python studies · stochastic differential equations, Monte Carlo and option pricing**

</div>

## Overview

Four educational numerical projects, from a geometric Brownian motion benchmark to exploratory Heston calibration. The purpose is to understand the mathematics, implementations and modelling assumptions; no predictive trading performance or production-grade pricing is claimed.

| Study | Question | Code |
| :-- | :-- | :-- |
| **01 · Geometric Brownian motion** | How does Euler–Maruyama compare with the exact process driven by the same Brownian path? | [Python](projects/quant_finance/project1_sde_simulator.py) |
| **02 · European option pricing** | How does Monte Carlo sampling compare with Black–Scholes? | [Python](projects/quant_finance/project2_montecarlo_eu_option_pricing.py) |
| **03 · Heston simulation** | How do correlated asset and variance shocks affect paths and European payoffs? | [Python](projects/quant_finance/project3_heston_model.py) |
| **04 · Exploratory Heston calibration** | How do European Heston prices compare with filtered TSLA call quotes? | [Python](projects/quant_finance/project4_heston_calibration.py) |

## Mathematical starting point

The first study solves the same geometric Brownian motion using Euler–Maruyama and its exact solution:

$$dS_t=\mu S_t\,dt+\sigma S_t\,dW_t,\quad S_t=S_0\exp\!\left((\mu-\tfrac12\sigma^2)t+\sigma W_t\right).$$

## Run the examples

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python projects/quant_finance/project1_sde_simulator.py
python projects/quant_finance/project2_montecarlo_eu_option_pricing.py
python projects/quant_finance/project3_heston_model.py
```

Project 04 also needs `python -m pip install -r requirements-market.txt`, market-data access and QuantLib; see [data requirements](docs/HESTON_TSLA_CALIBRATION_DATA_STATUS_2026-09-16.md). A past market fit cannot be reproduced from a changing live chain without a timestamped archived snapshot.

## Scope and limitations

Numerical SDE discretisations have timestep bias and Monte Carlo has sampling uncertainty. The calibration applies a European exercise model to listed equity options, which may have American exercise rights, and assumes simplified interest/dividend curves. No historical synchronized options dataset or out-of-sample pricing assessment is supplied. See the [reproducibility notes](docs/REPRODUCIBILITY.md) and [technical audit](docs/SCIENTIFIC_AUDIT_2026-09-16.md).

**Author:** [Rodrigo Varela](https://github.com/rodrigoacvarela20-web).
