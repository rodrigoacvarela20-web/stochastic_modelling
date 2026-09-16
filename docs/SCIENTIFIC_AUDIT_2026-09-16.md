# Scientific and numerical audit — 16 September 2026

**Scope: four retained educational Python studies.** This is an internal numerical/code review, not independent research replication, a validated trading strategy or a calibration against a preserved market-data snapshot. This note retains the findings relevant to the four studies remaining in the public portfolio.

## Corrections

1. **GBM:** `N=int(T/dt)` previously mixed the displayed grid step `T/N` with Euler increments of length `dt`. The corrected code uses `N=ceil(T/dt)` and `h=T/N` consistently.
2. **European option Monte Carlo:** Black–Scholes previously divided by zero at zero maturity or volatility. Analytic limits, zero strike, input checks and seeded draws are now supported. A single absolute-error curve does not establish monotonic Monte Carlo convergence.
3. **Heston simulation:** direct price Euler could yield negative simulated prices. Log-price updates maintain positivity while variance uses full-truncation Euler, and Brownian increments agree with the output grid. Discretisation bias remains.
4. **Heston calibration:** the previous filter could accept crossed quotes (`ask < bid`) because negative relative spreads passed the upper-bound filter. It now rejects crossed, locked, zero-bid, nonfinite and overly wide quotes. Quote filtering can be checked offline. A failed calibration optimizer raises an error rather than reporting parameters as calibrated.

## Verification status

The initial local audit exercised numerical invariants and synthetic quote filters. The test suite has subsequently been adjusted to the four retained studies. **The post-removal suite has not been rerun in this audit**, so the earlier test count should not be interpreted as a verified result for the current tree. No GitHub Actions result is claimed.

## Outstanding validation

QuantLib, a verified timestamped TSLA option-chain snapshot and associated market inputs were unavailable for the full Heston calibration. The fitted parameters, their stability and pricing performance therefore remain unverified. Yahoo Finance quotes can be stale or unsynchronized even after filtering. The code uses European Heston call values and fixed flat rates, while listed equity options may offer American exercise rights.

For further work, preserve synchronized historical quotes, compare prices with an independent implementation, and measure numerical discretisation and Monte Carlo error. None of these studies establishes trading profitability.
