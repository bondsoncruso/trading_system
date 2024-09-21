import numpy as np
import pandas as pd
import configparser
import datetime
import logging

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

# Select the latest data
latest_date = combined_df['date'].max()
filtered_df = combined_df[combined_df['date'] == latest_date]
sorted_df = filtered_df.sort_values(by='signal', ascending=False)
# final_ticker_list = sorted_df[sorted_df['volume'] >= 400000]['symbol'].head(10)
final_ticker_list = ['PARAS','BEL', 'HUDCO']

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
    annualised_target = trading_capital * (sharpe_ratio / 2) #25%
    daily_target = annualised_target / 16
    instrument_volatility = current_price * price_volatility
    volatility_scalar = daily_target / instrument_volatility
    subsystem_position = (forecast * volatility_scalar) / 10
    return round(subsystem_position)

# Calculate and log position sizes for each ticker
for ticker in final_ticker_list:
    stock_data = combined_df[combined_df['symbol'] == ticker]
    forecast = stock_data['signal'].iloc[-1]
    current_price = stock_data['close'].iloc[-1]
    close_prices = stock_data['close'].to_numpy()

    try:
        price_volatility = calculate_price_volatility(close_prices)
        size = calculate_position_size(0.5, price_volatility, current_price, forecast=forecast)
        logging.info(f'Ticker: {ticker}, Position Size: {size}, Investment: {current_price*size}, Forecast: {forecast}')
    except Exception as e:
        logging.error(f'Error processing {ticker}: {e}')

