"""Heston variance full-truncation Euler and positivity-preserving log-price steps.

The variance scheme incurs discretisation bias, particularly near zero;
these Monte Carlo estimates are not exact Heston prices.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def _steps(s0, v0, kappa, theta, sigma_v, rho, T, dt):
    inputs = [s0, v0, kappa, theta, sigma_v, rho, T, dt]
    if not np.all(np.isfinite(inputs)) or s0 <= 0 or v0 < 0 or kappa < 0 or theta < 0 or sigma_v < 0 or abs(rho) > 1 or T <= 0 or dt <= 0:
        raise ValueError('Invalid Heston parameters, maturity or time step.')
    N = max(1, int(np.ceil(T / dt)))
    return N, T / N


def simulate_heston(s0, v0, r, kappa, theta, sigma_v, rho, T, dt, rng=None):
    N, h = _steps(s0, v0, kappa, theta, sigma_v, rho, T, dt)
    if not np.isfinite(r):
        raise ValueError('Interest rate must be finite.')
    t = np.linspace(0, T, N + 1)
    S, v = np.zeros(N + 1), np.zeros(N + 1)
    S[0], v[0] = s0, v0
    normal = np.random.normal if rng is None else rng.normal
    for i in range(N):
        z1, z = normal(), normal()
        z2 = rho * z1 + np.sqrt(max(0.0, 1 - rho**2)) * z
        variance = max(v[i], 0.0)
        S[i + 1] = S[i] * np.exp((r - 0.5 * variance) * h + np.sqrt(variance * h) * z1)
        v[i + 1] = max(0.0, v[i] + kappa * (theta - variance) * h + sigma_v * np.sqrt(variance * h) * z2)
    return t, S, v


def simulate_terminal_heston(s0, v0, r, kappa, theta, sigma_v, rho, T, dt, rng=None):
    N, h = _steps(s0, v0, kappa, theta, sigma_v, rho, T, dt)
    if not np.isfinite(r):
        raise ValueError('Interest rate must be finite.')
    S, v = s0, v0
    normal = np.random.normal if rng is None else rng.normal
    for _ in range(N):
        z1, z = normal(), normal()
        z2 = rho * z1 + np.sqrt(max(0.0, 1 - rho**2)) * z
        variance = max(v, 0.0)
        S *= np.exp((r - 0.5 * variance) * h + np.sqrt(variance * h) * z1)
        v = max(0.0, v + kappa * (theta - variance) * h + sigma_v * np.sqrt(variance * h) * z2)
    return S


def monte_carlo_heston_option_pricing(s0, v0, r, kappa, theta, sigma_v, rho, T, dt, K, option_type='call', num_simulations=10000, rng=None):
    if option_type not in ('call', 'put') or K <= 0 or not isinstance(num_simulations, (int, np.integer)) or num_simulations < 1:
        raise ValueError('Require call/put, positive strike and positive simulation count.')
    payoffs = np.zeros(num_simulations)
    for i in range(num_simulations):
        terminal = simulate_terminal_heston(s0, v0, r, kappa, theta, sigma_v, rho, T, dt, rng)
        payoffs[i] = max(terminal - K, 0) if option_type == 'call' else max(K - terminal, 0)
    return float(np.exp(-r * T) * payoffs.mean())


def black_scholes_call_price(s0, K, r, sgm, T):
    if s0 <= 0 or K <= 0 or sgm < 0 or T < 0:
        raise ValueError('Invalid Black–Scholes inputs.')
    if T == 0:
        return max(s0 - K, 0)
    if sgm == 0:
        return max(s0 - K * np.exp(-r * T), 0)
    d1 = (np.log(s0 / K) + (r + 0.5 * sgm**2) * T) / (sgm * np.sqrt(T))
    d2 = d1 - sgm * np.sqrt(T)
    return s0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put_price(s0, K, r, sgm, T):
    if s0 <= 0 or K <= 0 or sgm < 0 or T < 0:
        raise ValueError('Invalid Black–Scholes inputs.')
    if T == 0:
        return max(K - s0, 0)
    if sgm == 0:
        return max(K * np.exp(-r * T) - s0, 0)
    d1 = (np.log(s0 / K) + (r + 0.5 * sgm**2) * T) / (sgm * np.sqrt(T))
    d2 = d1 - sgm * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1)


def main():
    s0, v0, r, kappa, theta, sigma_v, rho, T, dt, K = 100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7, 1, 0.005, 100
    rng = np.random.default_rng(2026)
    t, S, v = simulate_heston(s0, v0, r, kappa, theta, sigma_v, rho, T, dt, rng)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(t, S); axes[0].set(xlabel='Time', ylabel='Stock Price', title='Heston Model — Stock Price')
    axes[1].plot(t, np.sqrt(v)); axes[1].set(xlabel='Time', ylabel='Volatility', title='Heston Model — Volatility')
    plt.tight_layout(); plt.show()
    for kind in ('call', 'put'):
        heston = monte_carlo_heston_option_pricing(s0, v0, r, kappa, theta, sigma_v, rho, T, dt, K, kind, 1000, rng)
        bs = black_scholes_call_price(s0, K, r, np.sqrt(v0), T) if kind == 'call' else black_scholes_put_price(s0, K, r, np.sqrt(v0), T)
        print(f'{kind.capitalize()}: Heston={heston:.6f}; Black–Scholes={bs:.6f}; difference={abs(heston - bs):.6f}')
    strikes = [70, 80, 90, 100, 110, 120, 130]
    heston_prices = [monte_carlo_heston_option_pricing(s0, v0, r, kappa, theta, sigma_v, rho, T, dt, strike, 'call', 3000, rng) for strike in strikes]
    bs_prices = [black_scholes_call_price(s0, strike, r, np.sqrt(v0), T) for strike in strikes]
    plt.plot(strikes, heston_prices, marker='o', label='Heston')
    plt.plot(strikes, bs_prices, marker='o', label='Black–Scholes')
    plt.xlabel('Strike Price'); plt.ylabel('Call Option Price'); plt.title('Heston vs Black–Scholes'); plt.legend(); plt.show()


if __name__ == '__main__':
    main()
