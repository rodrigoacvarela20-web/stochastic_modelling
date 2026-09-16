# Scientific and numerical audit — 16 September 2026

**Scope:** five educational Python projects. This is an internal numerical/code review, not independent research replication, a validated trading strategy or a calibration against a preserved market-data snapshot.

## Corrections to all five projects

1. **GBM:** `N=int(T/dt)` previously mixed the displayed grid step `T/N` with Euler increments of length `dt`. Use `N=ceil(T/dt)`, then `h=T/N` consistently. The demonstration now uses a fixed seed.
2. **European option Monte Carlo:** Black–Scholes had division by zero at zero maturity or volatility. Analytic limits, zero strike, input checks and seeded draws are supported. A single absolute-error curve does not establish monotonic Monte Carlo convergence.
3. **Heston path simulation:** direct price Euler could yield negative simulated prices. Log-price updates maintain positivity while variance uses full-truncation Euler. The simulation clock now matches the actual Brownian time step. Discretisation bias remains, especially for variance near zero; independently sampled strike curves may still look nonmonotonic.
4. **Heston calibration:** the previous filter accepted some crossed markets (`ask < bid`), because negative relative spreads passed the upper-bound filter. It now rejects crossed, locked, zero-bid, nonfinite and overly wide quotes. The quote filtering can now be tested offline without importing QuantLib or yfinance. A failed calibration optimizer now raises an error instead of displaying parameters as calibrated.
5. **GARCH:** require finite inputs, nonnegative coefficients and a stationary parameter region; refuse degenerate constant-return fitting and report optimization failures rather than presenting failed results as fitted parameters. These guards do not establish parameter identification or out-of-sample performance.

## Independent offline checks

**17 tests passed** locally using the audited finance-source bundle with the corrected project files: five original baseline tests, six numerical-invariant tests, two synthetic quote-filter tests and four boundary/invalid-input tests. The additional extended suite already present in the public repository was **not downloaded and rerun in this audit**. The tests and corrections have been uploaded to GitHub; no GitHub Actions result is claimed.

## Unresolved scientific validation

**Project 4:** QuantLib, yfinance and a historical, time-aligned option-chain snapshot were unavailable, so the full live TSLA calibration and identifiability were **not** independently verified. Quotes can remain stale or unsynchronized even after filtering. The code uses European Heston call values and fixed flat interest/dividend assumptions. Standard US equity options have American-style exercise according to the Options Clearing Corporation (https://www.theocc.com/clearance-and-settlement/clearing/equity-options-product-specifications). Under suitable non-dividend and rate assumptions, early exercise may not alter call value, but this must be checked against actual contracts.

**Project 5:** Gaussian GARCH recursion and example synthetic fits do not demonstrate stable parameter recovery, heavy-tail robustness, current TSLA fit, or out-of-sample volatility forecasts. Data adjustments, mean modelling, initialization and stationarity constraints matter.

**Next checks:** preserve timestamped live option quotations, compare European Heston values against an independent pricing implementation, quantify Monte Carlo and time-step error, and evaluate GARCH on held-out market data. The code is exploratory research, not verified investment performance.