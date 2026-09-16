"""Euler–Maruyama simulation of geometric Brownian motion."""
import numpy as np
import matplotlib.pyplot as plt


def euler_maruyama(x0, mu, sigma, T, dt):
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    X = np.zeros(N + 1)
    W = np.zeros(N + 1)
    X[0] = x0
    for i in range(N):
        Z = np.random.normal(0, 1)
        dW = np.sqrt(dt) * Z
        W[i + 1] = W[i] + dW
        X[i + 1] = X[i] + mu * X[i] * dt + sigma * X[i] * dW
    return t, X, W


def exact_gbm(x0, mu, sigma, t, W):
    return x0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)


def main():
    x0, mu, sigma, T, dt = 100, 0.05, 0.20, 1, 0.001
    t, X, W = euler_maruyama(x0, mu, sigma, T, dt)
    X_exact = exact_gbm(x0, mu, sigma, t, W)
    absolute_error = np.abs(X - X_exact)
    print('Euler–Maruyama final value:', X[-1])
    print('Exact final value:', X_exact[-1])
    print('Final absolute error:', absolute_error[-1])
    print('Final relative error:', absolute_error[-1] / abs(X_exact[-1]))
    print('RMSE:', np.sqrt(np.mean((X - X_exact)**2)))
    plt.plot(t, X, label='Euler–Maruyama')
    plt.plot(t, X_exact, label='Exact GBM')
    plt.xlabel('Time'); plt.ylabel('X'); plt.title('Euler–Maruyama vs Exact GBM'); plt.legend(); plt.show()
    plt.plot(t, absolute_error); plt.xlabel('Time'); plt.ylabel('Absolute Error'); plt.title('Euler–Maruyama Error'); plt.show()
    n_simulations = 1000
    final_values = np.array([euler_maruyama(x0, mu, sigma, T, dt)[1][-1] for _ in range(n_simulations)])
    print('Monte Carlo simulations:', n_simulations)
    print('Simulated mean:', np.mean(final_values)); print('Theoretical mean:', x0 * np.exp(mu * T))
    print('Simulated variance:', np.var(final_values))
    print('Theoretical variance:', x0**2 * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1))
    plt.hist(final_values, bins=40); plt.xlabel('X(T)'); plt.ylabel('Frequency'); plt.title('Monte Carlo Distribution of X(T)'); plt.show()


if __name__ == '__main__':
    main()
