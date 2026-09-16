# Scientific and numerical audit — 16 September 2026

**Scope:** five educational Python projects. This is an internal numerical/code review, not an independent research replication, a validated trading strategy or a calibration against a preserved market-data snapshot.

## Code corrections

- **GBM (project 1):** previously `N=int(T/dt)` produced a displayed grid spacing `T/N` but Euler drift and Brownian increments still used `dt`. For non-dividing steps, this mixed two clocks. The revised code uses `N=ceil(T/dt)` and the consistent step `h=T/N`. A seeded demo improves repeatability. A counterexample `T=1,dt=0.3,sigma=0` now yields four deterministic steps of 0.25.
- **European option Monte Carlo (project 2):** the Black–Scholes expressions used to divide by `sigma*sqrt(T)` even for zero volatility or maturity. Those limits and zero strike are now treated explicitly, with finite input validation and optional seeded generators. A plot of absolute MC errors from one run should not be read as a monotonic convergence guarantee or error bar.
- **Heston simulation (project 3):** the Euler update for the price could create negative prices; the revised scheme evolves log price while retaining the explicit full-truncation variance scheme. Its Brownian increments use the same step as the output grid, and reproducible generators are supported. This is not an exact-in-law Heston sampler; variance discretisation bias persists, especially near the origin. Strike curves in the exploratory demo use separate draws and therefore can fluctuate non-monotonically due to sampling noise.

## Independent offline checks

Eleven tests passed locally against the audited copies of the five-project source bundle and the above revised code: five original tests plus six new tests covering GBM clocks, the deterministic Heston limit, agreement of full and terminal path routines with identical shocks, a sampled constant-variance martingale expectation, GARCH recursion, and Black–Scholes put–call parity. The public repository also contains a separately published extended suite; **it was not downloaded and rerun in this audit**, so eleven is the count claimed here. The changes and new tests have been uploaded to GitHub, but no GitHub Actions run is claimed.

## Unresolved validation

**Project 4, live TSLA option-chain calibration:** QuantLib and yfinance and a historical, time-aligned quote snapshot were unavailable in this environment. Its optimizer output, fit quality and identifiable Heston parameters therefore have not been independently verified. The code uses European call prices and fixed interest/dividend curves; OCC specifies standard US equity options as American-exercise contracts (https://www.theocc.com/clearance-and-settlement/clearing/equity-options-product-specifications). For non-dividend-paying calls under suitable assumptions early exercise need not change their values, but those assumptions and the actual contract data still require checking. Bid/ask filtering does not guarantee synchronized executable quotes or exclude every crossed market. Do not present the calibration as validated pricing or a profitable signal.

**Project 5, GARCH:** its recursion and one synthetic fitting example were previously checked, but this audit did not independently establish parameter recovery, robustness to heavy tails, performance on current TSLA returns, or out-of-sample volatility forecasting. Gaussian innovations, fixed mean, initialization and stationarity constraints are assumptions. A failed optimizer should not be interpreted as a calibrated parameter estimate.

**Next checks:** build a timestamped options dataset and compare European Heston prices with a second independent implementation; perform time-step/MC confidence-interval studies; test GARCH forecasts on held-out returns and alternative innovations; preserve software versions and random seeds.