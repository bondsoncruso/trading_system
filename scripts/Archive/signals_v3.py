import numpy as np
import pandas as pd
import configparser
import datetime
import logging
from sklearn.utils import resample

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configurations
config = configparser.ConfigParser()
config.read('data/config.ini')
lookback_period = config.getint('parameter', 'lookback_period_for_price_volatility')
trading_capital = config.getint('parameter', 'trading_capital')

# Load and prepare data
file_path = 'data/predictions/predictions.pkl'
combined_df = pd.read_pickle(file_path)
combined_df['date'] = pd.to_datetime(combined_df['date'])

# Calculate returns if not present
if 'returns' not in combined_df.columns:
    combined_df['returns'] = combined_df.groupby('symbol')['close'].pct_change()

# Select the latest data
latest_date = combined_df['date'].max()
filtered_df = combined_df[combined_df['date'] == latest_date]
sorted_df = filtered_df.sort_values(by='signal', ascending=False)
final_ticker_list = sorted_df[sorted_df['volume'] >= 400000]['symbol'].head(5)

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

# Function to bootstrap and calculate weights
def bootstrap_weights(returns, num_iterations=100, sample_size=0.1):
    weights_list = []
    num_samples = int(sample_size * len(returns))
    
    for _ in range(num_iterations):
        sample = resample(returns, n_samples=num_samples, replace=True)
        logging.debug(f'Bootstrap sample shape: {sample.shape}')
        if sample.shape[1] < 2:  # Ensure the sample is at least two-dimensional
            continue
        
        mean_returns = sample.mean(axis=0)
        covariance_matrix = np.cov(sample, rowvar=False)
        
        if covariance_matrix.shape[0] < 2:  # Ensure covariance matrix is at least 2x2
            logging.warning('Covariance matrix is too small, skipping this sample')
            continue
        
        inv_cov_matrix = np.linalg.pinv(covariance_matrix)
        ones = np.ones(len(mean_returns))
        
        weights = np.dot(inv_cov_matrix, ones) / np.dot(ones, np.dot(inv_cov_matrix, ones))
        weights_list.append(weights)
        
    if weights_list:
        return np.mean(weights_list, axis=0)
    else:
        logging.warning('Insufficient data to calculate weights, returning default weights')
        return np.ones(returns.shape[1])  # Default to equal weights if insufficient data

# Function to calculate position size using weighted forecasts
def calculate_position_size(sharpe_ratio, price_volatility, current_price, forecast, weight):
    annualised_target = trading_capital * (sharpe_ratio / 2)  # 25%
    daily_target = annualised_target / 16
    instrument_volatility = current_price * price_volatility
    volatility_scalar = daily_target / instrument_volatility
    subsystem_position = (forecast * volatility_scalar * weight) / 10
    return round(subsystem_position)

# Calculate and log position sizes for each ticker
for ticker in final_ticker_list:
    stock_data = combined_df[combined_df['symbol'] == ticker]
    forecast = stock_data['signal'].iloc[-1]
    current_price = stock_data['close'].iloc[-1]
    close_prices = stock_data['close'].to_numpy()
    
    try:
        price_volatility = calculate_price_volatility(close_prices)
        
        # Bootstrapping to calculate weights
        returns_data = stock_data['returns'].dropna().to_frame()  # Ensure it's a DataFrame
        logging.debug(f'Returns data shape for {ticker}: {returns_data.shape}')
        logging.debug(f'Returns data type for {ticker}: {type(returns_data)}')
        
        if returns_data.shape[0] > 1:  # Ensure there are enough data points
            returns_values = returns_data.values  # Convert to numpy array for bootstrapping
            logging.debug(f'Returns values shape for {ticker}: {returns_values.shape}')
            weights = bootstrap_weights(returns_values)
            weight = weights[0] if len(weights) > 0 else 1  # Use the first weight if it's a single column
        else:
            logging.warning(f'Insufficient returns data for {ticker}, using default weight')
            weight = 1  # Default weight if insufficient data
        
        size = calculate_position_size(0.5, price_volatility, current_price, forecast=forecast, weight=weight)
        logging.info(f'Ticker: {ticker}, Position Size: {size}, Investment: {current_price * size}, Forecast: {forecast}')
    except Exception as e:
        logging.error(f'Error processing {ticker}: {e}')
