"""Exploratory calibration of European Heston calls to live TSLA quotes.

US equity options are American-exercise contracts. A live option-chain snapshot
may mix timestamps with the spot close. No fit is a trading recommendation.
"""
from datetime import datetime, date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from project3_heston_model import simulate_heston


def filter_option_quotes(calls, s0):
    """Reject non-finite, crossed, zero-bid and excessively wide quotes.

    This does not establish quote simultaneity, liquidity or executability.
    """
    if not np.isfinite(s0) or s0 <= 0:
        raise ValueError('Spot must be positive and finite.')
    required = ('strike', 'bid', 'ask', 'openInterest')
    if any(field not in calls.columns for field in required):
        raise ValueError('Option data must contain strike, bid, ask and openInterest.')
    data = calls.copy()
    for field in required:
        data[field] = pd.to_numeric(data[field], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    data = data[(data['strike'] > 0.8*s0) & (data['strike'] < 1.2*s0)]
    data = data[(data['strike'] > 0) & (data['bid'] > 0) & (data['ask'] > data['bid']) & (data['openInterest'] > 20)].copy()
    data['mid'] = (data['bid'] + data['ask'])/2
    data['spread'] = data['ask']-data['bid']
    data['relative_spread'] = data['spread']/data['mid']
    return data[data['relative_spread'] < .20].copy()


def get_option_data(symbol, min_days=30, max_days=240, max_expirations=4):
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    history = ticker.history(period='5d')
    if history.empty or 'Close' not in history or history['Close'].dropna().empty:
        raise ValueError('Could not download stock price data.')
    s0 = float(history['Close'].dropna().iloc[-1])
    today = date.today()
    selected_expirations = []
    for expiration in ticker.options:
        days = (datetime.strptime(expiration, '%Y-%m-%d').date()-today).days
        if min_days <= days <= max_days:
            selected_expirations.append((expiration, days))
    if not selected_expirations:
        raise ValueError('No suitable expirations found.')
    market_data = []
    for expiration, days in selected_expirations[:max_expirations]:
        calls = filter_option_quotes(ticker.option_chain(expiration).calls, s0)
        if calls.empty:
            continue
        calls['expiration'], calls['days'] = expiration, days
        market_data.append(calls)
    if not market_data:
        raise ValueError('No usable option quotes after filtering.')
    return s0, pd.concat(market_data, ignore_index=True)


def create_market_environment(s0, r, q):
    import QuantLib as ql
    if not np.all(np.isfinite([s0, r, q])) or s0 <= 0:
        raise ValueError('Market environment must be finite with positive spot.')
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    risk_free_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    dividend_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
    spot = ql.QuoteHandle(ql.SimpleQuote(s0))
    return spot, risk_free_curve, dividend_curve


def heston_prices(parameters, market_data, spot, risk_free_curve, dividend_curve):
    import QuantLib as ql
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
        return np.ones(len(market_data))*1000
    market_prices = market_data['mid'].to_numpy()
    spreads = market_data['spread'].to_numpy()
    return (model_prices-market_prices)/np.maximum(spreads, .50)


def calibrate_heston(market_data, spot, risk_free_curve, dividend_curve):
    if market_data.empty:
        raise ValueError('Cannot calibrate without option quotes.')
    initial_guess = [.18, 5.0, .20, 1.0, -.3]
    lower_bounds = [.001, .01, .001, .01, -.99]
    upper_bounds = [1.0, 20.0, 1.0, 5.0, .99]
    return least_squares(calibration_residuals, initial_guess, bounds=(lower_bounds, upper_bounds), args=(market_data, spot, risk_free_curve, dividend_curve), max_nfev=300, verbose=1)


def calculate_results(parameters, market_data, spot, risk_free_curve, dividend_curve):
    results = market_data.copy()
    results['heston_price'] = heston_prices(parameters, market_data, spot, risk_free_curve, dividend_curve)
    results['price_error'] = results['heston_price']-results['mid']
    results['absolute_error'] = np.abs(results['price_error'])
    results['inside_spread'] = (results['heston_price'] >= results['bid']) & (results['heston_price'] <= results['ask'])
    return results


def main():
    symbol, r, q = 'TSLA', .04, 0.0
    print('Downloading market data...')
    s0, market_data = get_option_data(symbol)
    print('Stock:', symbol, 'Spot price:', s0, 'Number of options:', len(market_data))
    print(market_data[['expiration', 'days', 'strike', 'bid', 'ask', 'mid', 'spread']])
    spot, risk_free_curve, dividend_curve = create_market_environment(s0, r, q)
    print('Calibrating Heston model...')
    calibration = calibrate_heston(market_data, spot, risk_free_curve, dividend_curve)
    if not calibration.success:
        raise RuntimeError('Heston optimizer failed: '+str(calibration.message))
    v0, kappa, theta, sigma_v, rho = calibration.x
    print('Calibrated Heston parameters:', dict(zip(('v0', 'kappa', 'theta', 'sigma_v', 'rho'), calibration.x)))
    print('Initial volatility:', np.sqrt(v0), 'Long-run volatility:', np.sqrt(theta))
    print('Feller condition satisfied:', 2*kappa*theta > sigma_v**2)
    results = calculate_results(calibration.x, market_data, spot, risk_free_curve, dividend_curve)
    print('Price RMSE:', np.sqrt(np.mean(results['price_error']**2)))
    print('Price MAE:', np.mean(results['absolute_error']))
    print('Options inside bid-ask spread (%):', 100*np.mean(results['inside_spread']))
    print(results[['expiration', 'strike', 'bid', 'ask', 'mid', 'heston_price', 'price_error']])
    for expiration in results['expiration'].unique():
        data = results[results['expiration'] == expiration].sort_values('strike')
        plt.figure()
        plt.plot(data['strike'], data['mid'], marker='o', label='Market Mid')
        plt.plot(data['strike'], data['heston_price'], marker='o', label='Heston')
        plt.xlabel('Strike'); plt.ylabel('Call Option Price')
        plt.title('Heston vs Market — '+symbol+' — '+expiration); plt.legend(); plt.show()
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
    t, S, v = simulate_heston(s0, v0, r, kappa, theta, sigma_v, rho, 1, .001)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(t, S); axes[0].set(xlabel='Time', ylabel='Stock Price', title=symbol+' — Calibrated Heston Simulation')
    axes[1].plot(t, np.sqrt(v)); axes[1].set(xlabel='Time', ylabel='Volatility', title='Calibrated Stochastic Volatility')
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    main()
