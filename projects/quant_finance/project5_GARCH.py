"""Gaussian GARCH(1,1) quasi-MLE for daily historical log returns.

A demonstrator, not a validated volatility forecast. Daily observations are
assumed to be consistently adjusted; mean and innovations are simplified.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def get_returns(symbol, period):
    import yfinance as yf
    data = yf.Ticker(symbol).history(period=period)
    if data.empty or 'Close' not in data:
        raise ValueError('Could not retrieve usable historical market data.')
    prices = data['Close'].dropna()
    if len(prices) < 6 or not np.all(np.isfinite(prices)) or not np.all(prices > 0):
        raise ValueError('Require at least six finite positive prices.')
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return data, log_returns


def garch_variance(returns, omega, alpha, beta, mu):
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or not np.all(np.isfinite(returns)) or not np.all(np.isfinite([omega, alpha, beta, mu])):
        raise ValueError('Require one-dimensional finite returns and parameters.')
    if omega <= 0 or alpha < 0 or beta < 0 or alpha+beta >= 1:
        raise ValueError('Require omega>0, alpha,beta>=0 and alpha+beta<1.')
    N = len(returns)
    variance = np.zeros(N)
    if N == 0:
        return variance
    variance[0] = np.var(returns)
    for i in range(1, N):
        previous = returns[i-1]-mu
        variance[i] = omega + alpha*previous**2 + beta*variance[i-1]
    return variance


def negative_log_likelihood(parameters, returns, mu):
    try:
        omega, alpha, beta = parameters
        returns = np.asarray(returns, dtype=float)
        variance = garch_variance(returns, omega, alpha, beta, mu)
    except (ValueError, TypeError, OverflowError):
        return np.inf
    if len(returns) == 0 or not np.all(np.isfinite(variance)) or not np.all(variance > 0):
        return np.inf
    residual = returns-mu
    log_likelihood = -.5*np.sum(np.log(2*np.pi)+np.log(variance)+residual**2/variance)
    return float(-log_likelihood) if np.isfinite(log_likelihood) else np.inf


def stationarity_constraint(parameters):
    return .999-parameters[1]-parameters[2]


def estimate_garch(returns, mu):
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or len(returns) < 20 or not np.all(np.isfinite(returns)) or not np.isfinite(mu):
        raise ValueError('Require at least 20 finite daily returns and finite mean for fitting.')
    if np.var(returns) == 0:
        raise ValueError('Constant returns do not identify positive Gaussian GARCH variance.')
    initial_guess = [1e-5, .1, .85]
    bounds = [(1e-8, .01), (.0001, .999), (.0001, .999)]
    constraint = {'type': 'ineq', 'fun': stationarity_constraint}
    return minimize(negative_log_likelihood, initial_guess, args=(returns, mu), bounds=bounds, constraints=constraint, method='SLSQP')


def main():
    _, log_returns = get_returns('TSLA', '5y')
    returns = log_returns.to_numpy()
    mu = np.mean(returns)
    historical_volatility = np.std(returns)
    print('Mean daily return:', mu)
    print('Daily variance:', np.var(returns))
    print('Daily volatility:', historical_volatility)
    print('Annualized volatility:', historical_volatility*np.sqrt(252))
    result = estimate_garch(returns, mu)
    if not result.success:
        raise RuntimeError('GARCH optimization failed: '+str(result.message))
    omega, alpha, beta = result.x
    print('Estimated GARCH parameters:', dict(omega=omega, alpha=alpha, beta=beta))
    print('alpha + beta:', alpha+beta)
    fitted_volatility = np.sqrt(garch_variance(returns, omega, alpha, beta, mu))
    plt.figure(figsize=(12, 6)); plt.plot(log_returns.index, log_returns)
    plt.xlabel('Date'); plt.ylabel('Log Return'); plt.title('TSLA Daily Log Returns'); plt.show()
    plt.figure(figsize=(12, 6)); plt.plot(log_returns.index, np.abs(log_returns))
    plt.xlabel('Date'); plt.ylabel('Absolute Log Return'); plt.title('TSLA Absolute Daily Returns'); plt.show()
    plt.figure(figsize=(12, 6)); plt.plot(log_returns.index, fitted_volatility)
    plt.xlabel('Date'); plt.ylabel('Daily Volatility'); plt.title('TSLA GARCH(1,1) Estimated Volatility'); plt.show()


if __name__ == '__main__':
    main()
