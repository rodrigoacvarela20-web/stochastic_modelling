# TSLA Heston calibration: empirical verification status (16 September 2026)

**Not completed.** The research project uses an online Yahoo option chain and QuantLib European vanilla pricing, but the audit environment has neither QuantLib/yfinance nor a verifiable archived TSLA chain. Searching public sources did not yield an accessible, complete, synchronized snapshot with sufficient live bid/ask quotes and corresponding spot, expiration, timestamp and contract details. A collection of stock closing prices is **not** an option chain. Synthetic prices are not a valid substitute for empirical calibration. Accordingly there are **no verified TSLA parameter estimates, pricing-error statistics, independent model comparisons or market-data results to report**.

## Required preserved snapshot

One immutable file should contain at least `quote_timestamp_utc, symbol, contract_symbol, option_type, exercise_style, expiration, strike, bid, ask, bid_size, ask_size, open_interest, underlying_quote_timestamp_utc, underlying_bid, underlying_ask`, plus source/vendor, trading calendar, contract multiplier, corporate-action status, and provenance/hash. Attach an aligned risk-free curve and dividend/borrow assumptions for that **historical** date. Call or put prices must be filtered with explicit rules for crossed/locked/stale/zero/wide quotes, bid/ask consistency, time-to-expiry and moneyness. Options with different timestamps should not be treated as a synchronous arbitrage-free surface without further diagnostics.

## Validation plan after data acquisition

1. Preserve raw input and a cleaning/rejection log; check strike-monotonicity and vertical-spread/convexity constraints where applicable.
2. Reprice the retained contracts with an independent Heston implementation or reliable characteristic-function quadrature and a documented numerical tolerance. Distinguish American-style US equity options from European model prices; non-dividend call equivalence holds only under appropriate contract/rate assumptions.
3. Calibrate using spread-weighted price errors and fixed deterministic initial guesses; separately report training errors and held-out expiration/strike errors with timestamps and confidence/parameter-identifiability diagnostics.
4. Compare to an implied-volatility surface and Black–Scholes baseline on exactly the same options; perturb initial guesses and curve assumptions and estimate stability.
5. Report *non*-convergence and rejected quotes. This is a model-fit study, **not** a demonstration of market predictability or executable arbitrage.

The project's existing offline synthetic quote-filter checks **do not** constitute these empirical tests. If the author supplies a broker/vendor export, the above can be run against a documented snapshot without inventing observations.