import numpy as np
import pandas as pd
import configparser
import datetime
import logging
from scipy.optimize import minimize
import random

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configurations
config = configparser.ConfigParser()
config.read('data/config.ini')
lookback_period = config.getint('parameter', 'lookback_period_for_price_volatility')
trading_capital = config.getint('parameter', 'trading_capital')
buy_price_denominator = config.getint('parameter', 'buy_vol_fraction')

# Load and prepare data
file_path = 'data/predictions/predictions.pkl'
combined_df = pd.read_pickle(file_path)
combined_df['date'] = pd.to_datetime(combined_df['date'])

# Select the latest data
latest_date = combined_df['date'].max()
filtered_df = combined_df[combined_df['date'] == latest_date]
sorted_df = filtered_df.sort_values(by='signal', ascending=False)
final_ticker_list = ['ANANDRATHI','GODFRYPHLP','NESTLEIND']


# Function to calculate price volatility
def calculate_price_volatility(close_prices):
    if len(close_prices) < lookback_period:
        logging.error('Insufficient data for the lookback period')
        return np.nan
    returns = np.diff(close_prices) / close_prices[:-1]
    lamda = 2 / (1 + lookback_period)
    ewma_variance = np.nanvar(returns)  # Initialize with sample variance
    for r in returns[1:]:
        ewma_variance = lamda * (r ** 2) + (1 - lamda) * ewma_variance
    return np.sqrt(ewma_variance)

# Function to calculate position size
def calculate_position_size(sharpe_ratio, price_volatility, current_price, forecast):
    annualised_target = trading_capital * (sharpe_ratio / 2) # 25%
    daily_target = annualised_target / 16
    instrument_volatility = current_price * price_volatility
    volatility_scalar = daily_target / instrument_volatility
    subsystem_position = (forecast * volatility_scalar) / 10
    return round(subsystem_position)

# Bootstrap optimization function
def bootstrap_portfolio(returns, monte_carlo=200, monte_length=250):
    weightlist = []
    for _ in range(monte_carlo):
        bs_idx = [int(random.uniform(0, 1) * len(returns)) for _ in range(monte_length)]
        returns_sample = returns.iloc[bs_idx, :]
        weights = markowitz_optimization(returns_sample)
        weightlist.append(weights)
    return np.mean(weightlist, axis=0)

def markowitz_optimization(returns):
    cov_matrix = returns.cov().values
    mean_returns = returns.mean().values
    num_assets = len(mean_returns)
    args = (cov_matrix, mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def neg_sharpe_ratio(weights, cov_matrix, mean_returns, risk_free_rate=0):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return - (portfolio_return - risk_free_rate) / portfolio_volatility


# Calculate the diversification multiplier

def calculate_diversification_multiplier(weights, correlation_matrix):
    correlation_matrix[correlation_matrix < 0] = 0  # Floor negative correlations at zero
    weighted_correlation_matrix = np.dot(np.dot(weights.T, correlation_matrix), weights)
    diversification_multiplier = 1 / np.sqrt(weighted_correlation_matrix)
    return diversification_multiplier

# Calculate and log position sizes for each ticker
returns_data = {}

for ticker in final_ticker_list:
    stock_data = combined_df[combined_df['symbol'] == ticker]
    stock_data = stock_data.set_index('date')
    returns_data[ticker] = stock_data['close'].pct_change().dropna()

returns_df = pd.DataFrame(returns_data).dropna()

try:
    weights = bootstrap_portfolio(returns_df)
    # Calculate the correlation matrix
    correlation_matrix = returns_df.corr().values
    diversification_multiplier = calculate_diversification_multiplier(weights, correlation_matrix)
    logging.info(f'Diversification Multiplier: {diversification_multiplier:.4f}')
    
    total_positions = []
    for weight, ticker in zip(weights, final_ticker_list):
        stock_data = combined_df[combined_df['symbol'] == ticker]
        forecast = min(20,stock_data['signal'].iloc[-1])
        current_price = stock_data['close'].iloc[-1]
        predicted_price = stock_data['Predicted_Price'].iloc[-1]
        close_prices = stock_data['close'].to_numpy()
        
        price_volatility = calculate_price_volatility(close_prices)
        size = calculate_position_size(0.5, price_volatility, current_price, forecast=forecast)
        investment = current_price * round(size*weight*diversification_multiplier,0)
        logging.info(f'''
                     Ticker: {ticker}, 
                     Weight: {weight:.2%}, 
                     Position Size: {round(size*weight*diversification_multiplier,0):.0f}, 
                     Investment: {investment:.2f}, 
                     Forecast: {forecast:.2f}, 
                     Current_Price: {current_price:.2f}, 
                     Predicted_Price: {predicted_price:.2f},
                     Price_Volatility: {current_price*price_volatility/buy_price_denominator:.2f},
                     Buy_at: {current_price *(1-price_volatility/buy_price_denominator):.2f}
                    ''')
        total_positions.append((ticker, weight, size, investment, forecast))


except Exception as e:
    logging.error(f'Error during bootstrapping: {e}')
