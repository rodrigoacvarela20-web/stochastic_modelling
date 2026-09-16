"""Fit a Gaussian GARCH(1,1) model to historical TSLA returns.

The numerical functions are importable offline; only get_returns() needs Yahoo Finance.
This is a research demonstration, not a validated volatility forecast.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def get_returns(symbol, period):
    import yfinance as yf

    data = yf.Ticker(symbol).history(period=period)
    if data.empty or 'Close' not in data:
        raise ValueError('Could not retrieve usable historical market data.')
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    if log_returns.empty:
        raise ValueError('Insufficient historical returns.')
    return data, log_returns


def garch_variance(returns, omega, alpha, beta, mu):
    N = len(returns)
    variance = np.zeros(N)
    if N == 0:
        return variance
    variance[0] = np.var(returns)
    for i in range(1, N):
        epsilon_previous = returns[i - 1] - mu
        variance[i] = omega + alpha * epsilon_previous**2 + beta * variance[i - 1]
    return variance


def negative_log_likelihood(parameters, returns, mu):
    omega, alpha, beta = parameters
    variance = garch_variance(returns, omega, alpha, beta, mu)
    if len(returns) == 0 or not np.all(np.isfinite(variance)) or not np.all(variance > 0):
        return np.inf
    epsilon = returns - mu
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(variance) + epsilon**2 / variance)
    return -log_likelihood


def stationarity_constraint(parameters):
    return 0.999 - parameters[1] - parameters[2]


def estimate_garch(returns, mu):
    initial_guess = [0.00001, 0.1, 0.85]
    bounds = [(1e-8, 0.01), (0.0001, 0.999), (0.0001, 0.999)]
    constraint = {'type': 'ineq', 'fun': stationarity_constraint}
    return minimize(negative_log_likelihood, initial_guess, args=(returns, mu), bounds=bounds, constraints=constraint, method='SLSQP')


def main():
    _, log_returns = get_returns('TSLA', '5y')
    returns = log_returns.to_numpy()
    mu = np.mean(returns)
    historical_variance = np.var(returns)
    historical_volatility = np.std(returns)
    print('Mean daily return:', mu)
    print('Daily variance:', historical_variance)
    print('Daily volatility:', historical_volatility)
    print('Annualized volatility:', historical_volatility * np.sqrt(252))
    result = estimate_garch(returns, mu)
    if not result.success:
        print('Warning: optimization did not report success:', result.message)
    omega, alpha, beta = result.x
    print('Estimated GARCH parameters:', dict(omega=omega, alpha=alpha, beta=beta))
    print('alpha + beta:', alpha + beta)
    fitted_volatility = np.sqrt(garch_variance(returns, omega, alpha, beta, mu))
    plt.figure(figsize=(12, 6)); plt.plot(log_returns.index, log_returns)
    plt.xlabel('Date'); plt.ylabel('Log Return'); plt.title('TSLA Daily Log Returns'); plt.show()
    plt.figure(figsize=(12, 6)); plt.plot(log_returns.index, np.abs(log_returns))
    plt.xlabel('Date'); plt.ylabel('Absolute Log Return'); plt.title('TSLA Absolute Daily Returns'); plt.show()
    plt.figure(figsize=(12, 6)); plt.plot(log_returns.index, fitted_volatility)
    plt.xlabel('Date'); plt.ylabel('Daily Volatility'); plt.title('TSLA GARCH(1,1) Estimated Volatility'); plt.show()


if __name__ == '__main__':
    main()
