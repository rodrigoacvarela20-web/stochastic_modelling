"""European option pricing with Black--Scholes and Monte Carlo simulation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def _validate(s0: float, strike: float, r: float, sigma: float, T: float):
    values = np.array([s0, strike, r, sigma, T], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Inputs must be finite.")
    if s0 <= 0 or strike < 0 or sigma < 0 or T < 0:
        raise ValueError("Require s0>0, strike>=0, sigma>=0 and T>=0.")


def black_scholes_price(
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
):
    """Closed-form Black--Scholes price for a non-dividend-paying asset."""
    _validate(s0, strike, r, sigma, T)

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    if T == 0:
        return max(s0 - strike, 0.0) if option_type == "call" else max(strike - s0, 0.0)

    if sigma == 0 or strike == 0:
        discounted_strike = strike * np.exp(-r * T)
        return (
            max(s0 - discounted_strike, 0.0)
            if option_type == "call"
            else max(discounted_strike - s0, 0.0)
        )

    d1 = (
        np.log(s0 / strike)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(
            s0 * norm.cdf(d1)
            - strike * np.exp(-r * T) * norm.cdf(d2)
        )

    return float(
        strike * np.exp(-r * T) * norm.cdf(-d2)
        - s0 * norm.cdf(-d1)
    )


def monte_carlo_price(
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
    n_paths: int,
    rng: np.random.Generator,
):
    """Price a European option and return estimate, standard error and 95% CI."""
    _validate(s0, strike, r, sigma, T)

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2.")

    if T == 0:
        intrinsic = (
            max(s0 - strike, 0.0)
            if option_type == "call"
            else max(strike - s0, 0.0)
        )
        return intrinsic, 0.0, (intrinsic, intrinsic)

    z = rng.standard_normal(n_paths)
    terminal = s0 * np.exp(
        (r - 0.5 * sigma**2) * T
        + sigma * np.sqrt(T) * z
    )

    if option_type == "call":
        payoff = np.maximum(terminal - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal, 0.0)

    discounted = np.exp(-r * T) * payoff
    estimate = float(discounted.mean())
    standard_error = float(discounted.std(ddof=1) / np.sqrt(n_paths))
    ci = (
        estimate - 1.96 * standard_error,
        estimate + 1.96 * standard_error,
    )

    return estimate, standard_error, ci


def main():
    s0, strike, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    sample_sizes = np.array([100, 1_000, 10_000, 100_000])
    seed = 2026

    exact_call = black_scholes_price(
        s0, strike, r, sigma, T, "call"
    )
    exact_put = black_scholes_price(
        s0, strike, r, sigma, T, "put"
    )

    print(f"Black--Scholes call: {exact_call:.6f}")
    print(f"Black--Scholes put:  {exact_put:.6f}\n")

    call_errors = []
    put_errors = []

    for i, n_paths in enumerate(sample_sizes):
        call_rng = np.random.default_rng(seed + 2 * i)
        put_rng = np.random.default_rng(seed + 2 * i + 1)

        call, call_se, call_ci = monte_carlo_price(
            s0, strike, r, sigma, T, "call", int(n_paths), call_rng
        )
        put, put_se, put_ci = monte_carlo_price(
            s0, strike, r, sigma, T, "put", int(n_paths), put_rng
        )

        call_error = abs(call - exact_call)
        put_error = abs(put - exact_put)
        call_errors.append(call_error)
        put_errors.append(put_error)

        print(
            f"N={n_paths:>6,} | "
            f"call={call:.6f}, SE={call_se:.6f}, "
            f"95% CI=({call_ci[0]:.6f}, {call_ci[1]:.6f}), "
            f"|error|={call_error:.6f}"
        )
        print(
            f"           | "
            f"put ={put:.6f}, SE={put_se:.6f}, "
            f"95% CI=({put_ci[0]:.6f}, {put_ci[1]:.6f}), "
            f"|error|={put_error:.6f}"
        )

    plt.figure()
    plt.loglog(sample_sizes, call_errors, marker="o", label="Call error")
    plt.loglog(sample_sizes, put_errors, marker="o", label="Put error")

    reference = call_errors[0] * np.sqrt(sample_sizes[0] / sample_sizes)
    plt.loglog(sample_sizes, reference, linestyle="--", label=r"$N^{-1/2}$ reference")

    plt.xlabel("Number of Monte Carlo paths")
    plt.ylabel("Absolute pricing error")
    plt.title("Monte Carlo convergence to Black--Scholes")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
