"""European call/put pricing: Monte Carlo versus Black–Scholes."""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def simulate_terminal_price(s0, r, sgm, T):
    Z = np.random.normal(0, 1)
    return s0 * np.exp((r - sgm**2 / 2) * T + sgm * np.sqrt(T) * Z)


def call_payoff(S, K):
    return max(S - K, 0)


def put_payoff(S, K):
    return max(K - S, 0)


def monte_carlo_option_pricing(s0, K, r, sgm, T, option_type='call', num_simulations=10000):
    if option_type not in ('call', 'put'):
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    payoffs = np.zeros(num_simulations)
    for i in range(num_simulations):
        S = simulate_terminal_price(s0, r, sgm, T)
        payoffs[i] = call_payoff(S, K) if option_type == 'call' else put_payoff(S, K)
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
    s0, K, r, sgm, T = 100, 100, 0.05, 0.2, 1
    simulation_numbers = [100, 1000, 10000, 100000]
    exact_call = black_scholes_call_price(s0, K, r, sgm, T)
    exact_put = black_scholes_put_price(s0, K, r, sgm, T)
    call_error, put_error = [], []
    print('Black–Scholes Call:', exact_call)
    print('Black–Scholes Put:', exact_put)
    for n in simulation_numbers:
        call = monte_carlo_option_pricing(s0, K, r, sgm, T, 'call', n)
        put = monte_carlo_option_pricing(s0, K, r, sgm, T, 'put', n)
        call_error.append(abs(call - exact_call)); put_error.append(abs(put - exact_put))
        print(f'N={n}: MC call={call:.6f}, call error={call_error[-1]:.6f}; MC put={put:.6f}, put error={put_error[-1]:.6f}')
    plt.plot(simulation_numbers, call_error, marker='o', label='Call Error')
    plt.plot(simulation_numbers, put_error, marker='o', label='Put Error')
    plt.xscale('log'); plt.xlabel('Number of Simulations'); plt.ylabel('Absolute Error')
    plt.title('Monte Carlo Error vs Number of Simulations'); plt.legend(); plt.show()


if __name__ == '__main__':
    main()
