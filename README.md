# Stochastic Finance Models

A compact numerical portfolio in stochastic modelling and derivative pricing.

The repository currently contains two focused projects:

1. **Geometric Brownian Motion & SDE Simulation**
   - Euler--Maruyama discretisation of
     \(dS_t = \mu S_t\,dt + \sigma S_t\,dW_t\)
   - exact GBM solution evaluated on the same Brownian path
   - absolute, relative and RMSE path-error analysis
   - Monte Carlo validation of terminal mean and variance

2. **Monte Carlo Option Pricing & Black--Scholes Validation**
   - risk-neutral simulation of terminal prices
   - European call and put pricing
   - closed-form Black--Scholes benchmark
   - Monte Carlo standard errors and 95% confidence intervals
   - convergence study against the analytical price, with the
     \(N^{-1/2}\) Monte Carlo reference rate

## Repository structure

```text
stochastic_modelling/
├── gbm/
│   └── gbm_simulation.py
├── black_scholes/
│   └── option_pricing.py
├── tests/
│   ├── test_gbm.py
│   └── test_black_scholes.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
python -m pip install -r requirements.txt

python gbm/gbm_simulation.py
python black_scholes/option_pricing.py
python -m unittest discover -s tests -v
```

## Mathematical focus

### Geometric Brownian motion

The model is

\[
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,
\]

with exact solution

\[
S_t = S_0
\exp\!\left[
\left(\mu-\frac{1}{2}\sigma^2\right)t
+\sigma W_t
\right].
\]

The numerical experiment compares Euler--Maruyama with this exact solution
along the same Brownian path, then checks empirical terminal moments against
their theoretical values.

### European option pricing

Under the Black--Scholes assumptions, terminal prices are simulated under the
risk-neutral measure and discounted option payoffs estimate

\[
V_0=e^{-rT}\mathbb{E}^{\mathbb{Q}}[\Phi(S_T)].
\]

Monte Carlo estimates are compared with the closed-form Black--Scholes prices.
The experiment reports sampling uncertainty and illustrates the characteristic
Monte Carlo convergence rate of order \(N^{-1/2}\).

## Scope

These are educational numerical projects designed to demonstrate stochastic
modelling, numerical simulation, statistical validation and analytical
benchmarking. They are not presented as production pricing libraries or
trading strategies.
