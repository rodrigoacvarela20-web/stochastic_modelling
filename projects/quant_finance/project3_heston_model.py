"""Heston paths and European option Monte Carlo, from the original project."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def simulate_heston(s0, v0, r, kappa, theta, sigma_v, rho, T, dt):
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    S = np.zeros(N + 1)
    v = np.zeros(N + 1)
    S[0], v[0] = s0, v0
    for i in range(N):
        Z1, Z = np.random.normal(0, 1), np.random.normal(0, 1)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z
        dW1, dW2 = np.sqrt(dt) * Z1, np.sqrt(dt) * Z2
        sqrt_v = np.sqrt(max(v[i], 0))
        S[i + 1] = S[i] + r * S[i] * dt + sqrt_v * S[i] * dW1
        v[i + 1] = v[i] + kappa * (theta - v[i]) * dt + sigma_v * sqrt_v * dW2
        v[i + 1] = max(v[i + 1], 0)
    return t, S, v


def simulate_terminal_heston(s0, v0, r, kappa, theta, sigma_v, rho, T, dt):
    N = int(T / dt)
    S, v = s0, v0
    for i in range(N):
        Z1, Z = np.random.normal(0, 1), np.random.normal(0, 1)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z
        dW1, dW2 = np.sqrt(dt) * Z1, np.sqrt(dt) * Z2
        sqrt_v = np.sqrt(max(v, 0))
        S = S + r * S * dt + sqrt_v * S * dW1
        v = v + kappa * (theta - v) * dt + sigma_v * sqrt_v * dW2
        v = max(v, 0)
    return S


def monte_carlo_heston_option_pricing(s0, v0, r, kappa, theta, sigma_v, rho, T, dt, K, option_type='call', num_simulations=10000):
    if option_type not in ('call', 'put'):
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    payoffs = np.zeros(num_simulations)
    for i in range(num_simulations):
        S = simulate_terminal_heston(s0, v0, r, kappa, theta, sigma_v, rho, T, dt)
        payoffs[i] = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    return np.exp(-r * T) * np.mean(payoffs)


def black_scholes_call_price(s0, K, r, sgm, T):
    d1 = (np.log(s0 / K) + (r + 0.5 * sgm**2) * T) / (sgm * np.sqrt(T))
    d2 = d1 - sgm * np.sqrt(T)
    return s0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put_price(s0, K, r, sgm, T):
    d1 = (np.log(s0 / K) + (r + 0.5 * sgm**2) * T) / (sgm * np.sqrt(T))
    d2 = d1 - sgm * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1)


def main():
    s0, v0, r, kappa, theta, sigma_v, rho, T, dt, K = 100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7, 1, 0.005, 100
    t, S, v = simulate_heston(s0, v0, r, kappa, theta, sigma_v, rho, T, dt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(t, S); axes[0].set(xlabel='Time', ylabel='Stock Price', title='Heston Model — Stock Price')
    axes[1].plot(t, np.sqrt(v)); axes[1].set(xlabel='Time', ylabel='Volatility', title='Heston Model — Volatility')
    plt.tight_layout(); plt.show()
    for kind in ('call', 'put'):
        heston = monte_carlo_heston_option_pricing(s0, v0, r, kappa, theta, sigma_v, rho, T, dt, K, kind, 1000)
        bs = black_scholes_call_price(s0, K, r, np.sqrt(v0), T) if kind == 'call' else black_scholes_put_price(s0, K, r, np.sqrt(v0), T)
        print(f'{kind.capitalize()}: Heston={heston:.6f}; Black–Scholes={bs:.6f}; difference={abs(heston - bs):.6f}')
    strikes = [70, 80, 90, 100, 110, 120, 130]
    heston_prices = [monte_carlo_heston_option_pricing(s0, v0, r, kappa, theta, sigma_v, rho, T, dt, strike, 'call', 3000) for strike in strikes]
    bs_prices = [black_scholes_call_price(s0, strike, r, np.sqrt(v0), T) for strike in strikes]
    plt.plot(strikes, heston_prices, marker='o', label='Heston')
    plt.plot(strikes, bs_prices, marker='o', label='Black–Scholes')
    plt.xlabel('Strike Price'); plt.ylabel('Call Option Price'); plt.title('Heston vs Black–Scholes'); plt.legend(); plt.show()


if __name__ == '__main__':
    main()
