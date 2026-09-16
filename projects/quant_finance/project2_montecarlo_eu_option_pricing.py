"""European call/put Monte Carlo and Black–Scholes for a non-dividend asset."""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def _validate(s0, K, r, sgm, T):
    if not np.all(np.isfinite([s0, K, r, sgm, T])) or s0 <= 0 or K < 0 or sgm < 0 or T < 0:
        raise ValueError('Require finite S0>0, K>=0, volatility>=0, maturity>=0 and finite rate.')


def simulate_terminal_price(s0, r, sgm, T, rng=None):
    _validate(s0, 0, r, sgm, T)
    if T == 0:
        return float(s0)
    normal = np.random.normal if rng is None else rng.normal
    return float(s0 * np.exp((r - sgm**2/2)*T + sgm*np.sqrt(T)*normal()))


def call_payoff(S, K):
    return max(S-K, 0)


def put_payoff(S, K):
    return max(K-S, 0)


def monte_carlo_option_pricing(s0, K, r, sgm, T, option_type='call', num_simulations=10000, rng=None):
    _validate(s0, K, r, sgm, T)
    if option_type not in ('call', 'put') or not isinstance(num_simulations, (int, np.integer)) or num_simulations < 1:
        raise ValueError('Require call/put and a positive integer simulation count.')
    if T == 0:
        return float(call_payoff(s0, K) if option_type == 'call' else put_payoff(s0, K))
    if sgm == 0:
        terminal = s0*np.exp(r*T)
        return float(np.exp(-r*T)*(call_payoff(terminal, K) if option_type == 'call' else put_payoff(terminal, K)))
    normal = np.random.normal if rng is None else rng.normal
    z = normal(size=num_simulations)
    terminal = s0*np.exp((r - sgm**2/2)*T + sgm*np.sqrt(T)*z)
    payoff = np.maximum(terminal-K, 0) if option_type == 'call' else np.maximum(K-terminal, 0)
    return float(np.exp(-r*T)*np.mean(payoff))


def black_scholes_call_price(s0, K, r, sgm, T):
    _validate(s0, K, r, sgm, T)
    if T == 0:
        return float(max(s0-K, 0))
    if sgm == 0 or K == 0:
        return float(max(s0-K*np.exp(-r*T), 0))
    d1 = (np.log(s0/K)+(r+0.5*sgm**2)*T)/(sgm*np.sqrt(T))
    d2 = d1-sgm*np.sqrt(T)
    return float(s0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2))


def black_scholes_put_price(s0, K, r, sgm, T):
    _validate(s0, K, r, sgm, T)
    if T == 0:
        return float(max(K-s0, 0))
    if sgm == 0 or K == 0:
        return float(max(K*np.exp(-r*T)-s0, 0))
    d1 = (np.log(s0/K)+(r+0.5*sgm**2)*T)/(sgm*np.sqrt(T))
    d2 = d1-sgm*np.sqrt(T)
    return float(K*np.exp(-r*T)*norm.cdf(-d2)-s0*norm.cdf(-d1))


def main():
    s0, K, r, sgm, T = 100, 100, 0.05, 0.2, 1
    simulation_numbers = [100, 1000, 10000, 100000]
    exact_call = black_scholes_call_price(s0, K, r, sgm, T)
    exact_put = black_scholes_put_price(s0, K, r, sgm, T)
    call_error, put_error = [], []
    rng = np.random.default_rng(2026)
    print('Black–Scholes Call:', exact_call)
    print('Black–Scholes Put:', exact_put)
    for n in simulation_numbers:
        call = monte_carlo_option_pricing(s0, K, r, sgm, T, 'call', n, rng)
        put = monte_carlo_option_pricing(s0, K, r, sgm, T, 'put', n, rng)
        call_error.append(abs(call-exact_call)); put_error.append(abs(put-exact_put))
        print(f'N={n}: MC call={call:.6f}, call error={call_error[-1]:.6f}; MC put={put:.6f}, put error={put_error[-1]:.6f}')
    plt.plot(simulation_numbers, call_error, marker='o', label='Call Error')
    plt.plot(simulation_numbers, put_error, marker='o', label='Put Error')
    plt.xscale('log'); plt.xlabel('Number of Simulations'); plt.ylabel('Absolute Error')
    plt.title('MC absolute error (one random run per sample size)'); plt.legend(); plt.show()


if __name__ == '__main__':
    main()
