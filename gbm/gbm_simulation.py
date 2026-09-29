"""Geometric Brownian motion: Euler--Maruyama, exact solution and Monte Carlo validation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def euler_maruyama(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    rng: np.random.Generator,
):
    """Simulate one GBM path with Euler--Maruyama."""
    if s0 <= 0 or sigma < 0 or T <= 0 or dt <= 0:
        raise ValueError("Require s0>0, sigma>=0, T>0 and dt>0.")

    n_steps = int(np.ceil(T / dt))
    h = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    dW = np.sqrt(h) * rng.standard_normal(n_steps)
    W = np.concatenate(([0.0], np.cumsum(dW)))

    S = np.empty(n_steps + 1)
    S[0] = s0
    for i in range(n_steps):
        S[i + 1] = S[i] + mu * S[i] * h + sigma * S[i] * dW[i]

    return t, S, W


def exact_gbm(
    s0: float,
    mu: float,
    sigma: float,
    t: np.ndarray,
    W: np.ndarray,
):
    """Evaluate the exact GBM solution on a supplied Brownian path."""
    return s0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)


def terminal_moments(s0: float, mu: float, sigma: float, T: float):
    """Theoretical mean and variance of S_T."""
    mean = s0 * np.exp(mu * T)
    variance = (
        s0**2
        * np.exp(2.0 * mu * T)
        * (np.exp(sigma**2 * T) - 1.0)
    )
    return mean, variance


def monte_carlo_terminal_values(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    n_paths: int,
    rng: np.random.Generator,
):
    """Generate terminal values using Euler--Maruyama."""
    if n_paths < 1:
        raise ValueError("n_paths must be positive.")

    terminal = np.empty(n_paths)
    for j in range(n_paths):
        _, path, _ = euler_maruyama(s0, mu, sigma, T, dt, rng)
        terminal[j] = path[-1]
    return terminal


def main():
    s0, mu, sigma, T, dt = 100.0, 0.05, 0.20, 1.0, 1e-3
    seed = 2026

    rng_path = np.random.default_rng(seed)
    t, approx, W = euler_maruyama(s0, mu, sigma, T, dt, rng_path)
    exact = exact_gbm(s0, mu, sigma, t, W)

    abs_error = np.abs(approx - exact)
    rel_error = abs_error / np.abs(exact)
    rmse = np.sqrt(np.mean((approx - exact) ** 2))

    print(f"Euler--Maruyama final value: {approx[-1]:.6f}")
    print(f"Exact final value:          {exact[-1]:.6f}")
    print(f"Final absolute error:       {abs_error[-1]:.6f}")
    print(f"Final relative error:       {rel_error[-1]:.6%}")
    print(f"Path RMSE:                  {rmse:.6f}")

    plt.figure()
    plt.plot(t, approx, label="Euler--Maruyama")
    plt.plot(t, exact, label="Exact GBM")
    plt.xlabel("Time")
    plt.ylabel("S(t)")
    plt.title("Euler--Maruyama vs exact GBM")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(t, abs_error)
    plt.xlabel("Time")
    plt.ylabel("Absolute error")
    plt.title("Euler--Maruyama path error")
    plt.tight_layout()
    plt.show()

    rng_mc = np.random.default_rng(seed + 1)
    n_paths = 1_000
    terminal = monte_carlo_terminal_values(
        s0, mu, sigma, T, dt, n_paths, rng_mc
    )

    theoretical_mean, theoretical_var = terminal_moments(
        s0, mu, sigma, T
    )

    print(f"\nMonte Carlo paths:          {n_paths:,}")
    print(f"Simulated mean:             {terminal.mean():.6f}")
    print(f"Theoretical mean:           {theoretical_mean:.6f}")
    print(f"Simulated variance:         {terminal.var(ddof=1):.6f}")
    print(f"Theoretical variance:       {theoretical_var:.6f}")

    plt.figure()
    plt.hist(terminal, bins=40)
    plt.xlabel("S(T)")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo distribution of S(T)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
