"""Calibrate European Heston call prices to a filtered live TSLA options chain.

Exploratory model only: US-listed equity options may have American exercise
features. Live Yahoo quotes and the flat interest/dividend assumptions make
results date-dependent and unsuitable as a trading recommendation.
"""
from datetime import datetime, date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import QuantLib as ql
from scipy.optimize import least_squares
from project3_heston_model import simulate_heston


def get_option_data(symbol, min_days=30, max_days=240, max_expirations=4):
    ticker = yf.Ticker(symbol)
    history = ticker.history(period='5d')
    if history.empty:
        raise ValueError('Could not download stock price data.')
    s0 = float(history['Close'].dropna().iloc[-1])
    today = date.today()
    selected_expirations = []
    for expiration in ticker.options:
        days = (datetime.strptime(expiration, '%Y-%m-%d').date() - today).days
        if min_days <= days <= max_days:
            selected_expirations.append((expiration, days))
    selected_expirations = selected_expirations[:max_expirations]
    if not selected_expirations:
        raise ValueError('No suitable expirations found.')
    market_data = []
    for expiration, days in selected_expirations:
        calls = ticker.option_chain(expiration).calls.copy()
        calls['openInterest'] = calls['openInterest'].fillna(0)
        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)].copy()
        calls['mid'] = (calls['bid'] + calls['ask']) / 2
        calls['spread'] = calls['ask'] - calls['bid']
        calls['relative_spread'] = calls['spread'] / calls['mid']
        calls = calls[(calls['relative_spread'] < 0.20) & (calls['openInterest'] > 20)]
        calls = calls[(calls['strike'] > 0.8 * s0) & (calls['strike'] < 1.2 * s0)].copy()
        if calls.empty:
            continue
        calls['expiration'], calls['days'] = expiration, days
        market_data.append(calls)
    if not market_data:
        raise ValueError('No usable option data found.')
    return s0, pd.concat(market_data, ignore_index=True)


def create_market_environment(s0, r, q):
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    risk_free_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    dividend_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
    spot = ql.QuoteHandle(ql.SimpleQuote(s0))
    return spot, risk_free_curve, dividend_curve


def heston_prices(parameters, market_data, spot, risk_free_curve, dividend_curve):
    v0, kappa, theta, sigma_v, rho = parameters
    process = ql.HestonProcess(risk_free_curve, dividend_curve, spot, v0, kappa, theta, sigma_v, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    prices = np.zeros(len(market_data))
    for i in range(len(market_data)):
        row = market_data.iloc[i]
        expiration = datetime.strptime(row['expiration'], '%Y-%m-%d')
        maturity = ql.Date(expiration.day, expiration.month, expiration.year)
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(row['strike']))
        option = ql.VanillaOption(payoff, ql.EuropeanExercise(maturity))
        option.setPricingEngine(engine)
        prices[i] = option.NPV()
    return prices


def calibration_residuals(parameters, market_data, spot, risk_free_curve, dividend_curve):
    try:
        model_prices = heston_prices(parameters, market_data, spot, risk_free_curve, dividend_curve)
    except Exception:
        return np.ones(len(market_data)) * 1000
    market_prices = market_data['mid'].to_numpy()
    spreads = market_data['spread'].to_numpy()
    return (model_prices - market_prices) / np.maximum(spreads, 0.50)


def calibrate_heston(market_data, spot, risk_free_curve, dividend_curve):
    initial_guess = [0.18, 5.0, 0.20, 1.0, -0.3]
    lower_bounds = [0.001, 0.01, 0.001, 0.01, -0.99]
    upper_bounds = [1.0, 20.0, 1.0, 5.0, 0.99]
    return least_squares(calibration_residuals, initial_guess, bounds=(lower_bounds, upper_bounds), args=(market_data, spot, risk_free_curve, dividend_curve), max_nfev=300, verbose=1)


def calculate_results(parameters, market_data, spot, risk_free_curve, dividend_curve):
    results = market_data.copy()
    results['heston_price'] = heston_prices(parameters, market_data, spot, risk_free_curve, dividend_curve)
    results['price_error'] = results['heston_price'] - results['mid']
    results['absolute_error'] = np.abs(results['price_error'])
    results['inside_spread'] = (results['heston_price'] >= results['bid']) & (results['heston_price'] <= results['ask'])
    return results


def main():
    symbol, r, q = 'TSLA', 0.04, 0.0
    print('Downloading market data...')
    s0, market_data = get_option_data(symbol)
    print('Stock:', symbol, 'Spot price:', s0, 'Number of options:', len(market_data))
    print(market_data[['expiration', 'days', 'strike', 'bid', 'ask', 'mid', 'spread']])
    spot, risk_free_curve, dividend_curve = create_market_environment(s0, r, q)
    print('Calibrating Heston model...')
    calibration = calibrate_heston(market_data, spot, risk_free_curve, dividend_curve)
    v0, kappa, theta, sigma_v, rho = calibration.x
    print('Optimizer success:', calibration.success, calibration.message)
    print('Calibrated Heston parameters:', dict(zip(('v0', 'kappa', 'theta', 'sigma_v', 'rho'), calibration.x)))
    print('Initial volatility:', np.sqrt(v0), 'Long-run volatility:', np.sqrt(theta))
    print('Feller condition satisfied:', 2 * kappa * theta > sigma_v**2)
    results = calculate_results(calibration.x, market_data, spot, risk_free_curve, dividend_curve)
    print('Price RMSE:', np.sqrt(np.mean(results['price_error']**2)))
    print('Price MAE:', np.mean(results['absolute_error']))
    print('Options inside bid-ask spread (%):', 100 * np.mean(results['inside_spread']))
    print(results[['expiration', 'strike', 'bid', 'ask', 'mid', 'heston_price', 'price_error']])
    for expiration in results['expiration'].unique():
        data = results[results['expiration'] == expiration].sort_values('strike')
        plt.figure()
        plt.plot(data['strike'], data['mid'], marker='o', label='Market Mid')
        plt.plot(data['strike'], data['heston_price'], marker='o', label='Heston')
        plt.xlabel('Strike'); plt.ylabel('Call Option Price')
        plt.title('Heston vs Market — ' + symbol + ' — ' + expiration); plt.legend(); plt.show()
    for expiration in results['expiration'].unique():
        data = results[results['expiration'] == expiration]
        plt.scatter(data['strike'], data['price_error'], label=expiration)
    plt.axhline(0, linestyle='--'); plt.xlabel('Strike'); plt.ylabel('Heston Price − Market Mid')
    plt.title('Heston Calibration Residuals'); plt.legend(); plt.show()
    plt.scatter(results['mid'], results['heston_price'])
    minimum = min(results['mid'].min(), results['heston_price'].min())
    maximum = max(results['mid'].max(), results['heston_price'].max())
    plt.plot([minimum, maximum], [minimum, maximum], linestyle='--')
    plt.xlabel('Market Mid Price'); plt.ylabel('Heston Price'); plt.title('Model Price vs Market Price'); plt.show()
    t, S, v = simulate_heston(s0, v0, r, kappa, theta, sigma_v, rho, 1, 0.001)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(t, S); axes[0].set(xlabel='Time', ylabel='Stock Price', title=symbol + ' — Calibrated Heston Simulation')
    axes[1].plot(t, np.sqrt(v)); axes[1].set(xlabel='Time', ylabel='Volatility', title='Calibrated Stochastic Volatility')
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    main()
