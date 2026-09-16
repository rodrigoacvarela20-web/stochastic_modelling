# Replicated Monte Carlo convergence: European calls

This **offline, educational experiment** measures sampling error rather than presenting an option-pricing strategy. It extends [Study 02](../projects/quant_finance/project2_montecarlo_eu_option_pricing.py), whose original graph shows one draw at each sample size.

## Mathematical question

For independent risk-neutral terminal prices under geometric Brownian motion, write

\[
\widehat C_N=e^{-rT}\frac1N\sum_{i=1}^N(S_T^{(i)}-K)^+, \qquad C_{\mathrm{BS}}=\mathbb E[\widehat C_N].
\]

When the payoff has finite variance, independent sampling gives

\[
\operatorname{Var}(\widehat C_N)=\frac{\operatorname{Var}(e^{-rT}(S_T-K)^+)}{N}.
\]

Thus the theoretical root-mean-square error around the Black–Scholes value scales as \(N^{-1/2}\). One noisy sequence of absolute errors does **not** demonstrate this rate, so we repeat each sample size independently and calculate the empirical RMSE.

## Reproduce from the repository root

```bash
python -m pip install -r requirements.txt
python experiments/mc_convergence.py --plot assets/monte_carlo_convergence.png
python -m unittest discover -s tests -v
```

The experiment uses \(S_0=K=100\), \(r=0.05\), \(\sigma=0.20\), \(T=1\), sample sizes 100, 400, 1,600 and 6,400, 200 independent pricing estimates at each sample size, and a fixed NumPy generator seed of 2026. No market data or QuantLib is required. The figure is **generated locally** by the script; it is not a checked-in image.

## Representative seeded output

Black–Scholes call value: **10.450584** (six decimal places).

| Samples per estimate | Mean error (bias estimate) | Empirical RMSE | SD across estimates |
| ---: | ---: | ---: | ---: |
| 100 | +0.087602 | 1.437845 | 1.438776 |
| 400 | −0.049414 | 0.722713 | 0.722831 |
| 1,600 | +0.017176 | 0.376801 | 0.377354 |
| 6,400 | −0.001676 | 0.186129 | 0.186588 |

The plotted \(N^{-1/2}\) guide is **anchored at the first empirical RMSE** to compare the shape; it is not an independently estimated theoretical error constant. The SD across estimates is not the standard error of their mean. RMSE and bias themselves have finite-repetition uncertainty. These results illustrate the expected sampling-error behavior for this specified model and seed, not a universal numerical guarantee, an empirical market fit or an out-of-sample trading result.
