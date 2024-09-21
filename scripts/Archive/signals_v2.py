import numpy as np
import pandas as pd
import configparser
import datetime
import logging
from scipy.optimize import minimize
import warnings

# To ignore a specific category of warning
warnings.filterwarnings("ignore", category=FutureWarning)



# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configurations
config = configparser.ConfigParser()
config.read('data/config.ini')
lookback_period = config.getint('parameter', 'lookback_period_for_price_volatility')
trading_capital = config.getint('parameter', 'trading_capital')
num_iterations = 100  # Number of bootstrap iterations

# Load and prepare data
file_path = 'data/predictions/predictions.pkl'
combined_df = pd.read_pickle(file_path)
combined_df['date'] = pd.to_datetime(combined_df['date'])

# Select the latest data
latest_date = combined_df['date'].max()
filtered_df = combined_df[combined_df['date'] == latest_date]
sorted_df = filtered_df.sort_values(by='signal', ascending=False)
final_ticker_list = ['PARAS', 'MOTHERSON']

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
    annualised_target = trading_capital * (sharpe_ratio / 2)  # 25%
    daily_target = annualised_target / 16
    instrument_volatility = current_price * price_volatility
    volatility_scalar = daily_target / instrument_volatility
    subsystem_position = (forecast * volatility_scalar) / 10
    return round(subsystem_position)

# Function to perform bootstrapping
def optimize_weights(correlation_matrix, num_assets):
    # Objective function: maximize the diversification multiplier
    def objective(weights):
        # Calculate the diversification multiplier
        weighted_correlation = np.dot(np.dot(weights, correlation_matrix), weights.T)
        multiplier = 1 / np.sqrt(weighted_correlation)
        return -multiplier  # Minimize negative to maximize positive
    
    # Constraints: sum of weights must be 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # Bounds for weights: each weight must be between 0 and 1
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # Initial guess: equally distributed weights
    initial_weights = np.array(num_assets * [1. / num_assets])
    
    # Optimization
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x  # Optimized weights
    else:
        raise ValueError("Optimization failed:", result.message)

def bootstrap_weights(df, num_iterations, final_ticker_list):
    weights = {ticker: [] for ticker in final_ticker_list}
    
    for _ in range(num_iterations):
        sample_df = df.sample(frac=1, replace=True)
        # Ensure date and symbol are the appropriate type; convert if necessary
        sample_df['date'] = pd.to_datetime(sample_df['date'])
        sample_df['symbol'] = sample_df['symbol'].astype('category')

        # Modify the groupby operation
        sample_df = sample_df.groupby(['date', 'symbol'], observed=True).mean().reset_index()
        pivot_df = sample_df.pivot_table(index='date', columns='symbol', values='signal', aggfunc='mean')
        correlation_matrix = pivot_df.corr().values
        np.fill_diagonal(correlation_matrix, 1)  # Ensure diagonal is 1
        correlation_matrix[correlation_matrix < 0] = 0  # Floor negative correlations at zero
        
        # Get optimized weights
        try:
            optimized_weights = optimize_weights(correlation_matrix, len(final_ticker_list))
            for i, ticker in enumerate(final_ticker_list):
                weights[ticker].append(optimized_weights[i])
        except Exception as e:
            logging.error(f"Optimization error: {e}")
    
    # Average the weights
    avg_weights = {ticker: np.mean(weights[ticker]) for ticker in weights}
    return avg_weights

# Filter the dataframe to include only the final ticker list
filtered_combined_df = combined_df[combined_df['symbol'].isin(final_ticker_list)]

# Perform bootstrapping to get weights
weights = bootstrap_weights(filtered_combined_df, num_iterations,final_ticker_list)
print(weights)

# Calculate and log position sizes for each ticker
for ticker in final_ticker_list:
    stock_data = combined_df[combined_df['symbol'] == ticker]
    forecast = stock_data['signal'].iloc[-1]
    current_price = stock_data['close'].iloc[-1]
    close_prices = stock_data['close'].to_numpy()

    try:
        price_volatility = calculate_price_volatility(close_prices)
        base_size = calculate_position_size(0.5, price_volatility, current_price, forecast=forecast)
        weighted_size = base_size * weights[ticker]
        logging.info(f'Ticker: {ticker}, Position Size: {weighted_size}, Investment: {current_price * weighted_size}, Forecast: {forecast}')
    except Exception as e:
        logging.error(f'Error processing {ticker}: {e}')
