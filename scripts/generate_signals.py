import configparser
config = configparser.ConfigParser()
config.read('D:/code/trading_system/data/config.ini')
feature_columns_str = config.get('parameter', "feature_columns")
feature_columns = feature_columns_str.strip().split(",")

# Define the default columns list
default_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']

feature_columns_plus_default_cols = feature_columns + default_cols

import pandas as pd
import pickle


data_path = 'D:/code/trading_system/data/processed/UpstoxDataWithIndicators.pkl'
data_df = pd.read_pickle(data_path)
predictions_df = pd.read_pickle('D:/code/trading_system/data/predictions/raw_predictions.pkl')
predictions_df = predictions_df.dropna()
# predictions_df['Predicted_Price'] = predictions
# Merge predictions with data
merged_df = pd.merge(data_df, predictions_df, on=['date', 'symbol'], how='inner')

import numpy as np
from scipy.stats import zscore

def calculate_continuous_signals(close_prices, predicted_prices, sensitivity, include_rate_of_change=False, momentum_weight=0.2):
    if len(close_prices) < 2:
        return np.zeros_like(close_prices)  # Return zero signals if not enough data

    # Calculate log returns and volatility
    log_returns = np.log(close_prices[1:] / close_prices[:-1])
    if np.std(log_returns) == 0:
        volatilities = np.zeros_like(close_prices)
    else:
        volatilities = np.std(log_returns) * np.sqrt(252)  # Annualizing

    # Extend volatilities array to match the length of close_prices
    volatilities = np.full_like(close_prices, volatilities)

    # Calculate percentage differences and standardize by volatility
    percentage_differences = (predicted_prices - close_prices) / close_prices * 100
    standardized_differences = percentage_differences / volatilities

    # Handle cases where volatilities might be zero
    standardized_differences[volatilities == 0] = 0

    # Calculate z-scores to dynamically determine thresholds
    z_scores = zscore(standardized_differences, nan_policy='omit')

    # Continuous signals based on z-scores and sensitivity
    signals = z_scores / sensitivity

    if include_rate_of_change:
        # Calculate rate of change (momentum) of signals
        momentum = np.diff(signals, prepend=signals[0])

        # Combine the original signals and their momentum
        signals = (1 - momentum_weight) * signals + momentum_weight * momentum

    return signals


def apply_signals_to_group(df_group, sensitivity=1, include_rate_of_change=False, momentum_weight=0.5):
    close_prices = df_group['close'].values
    predicted_prices = df_group['Predicted_Price'].values
    df_group['signal'] = calculate_continuous_signals(close_prices, predicted_prices, sensitivity, include_rate_of_change, momentum_weight)
    return df_group

# Example usage
signal_df = merged_df
sensitivity = 0.5
include_rate_of_change = True  # Set this to False if you don't want to include the rate of change
momentum_weight = 0.5  # Weight for the rate of change (momentum)

# Apply the function to each group using groupby and apply
signal_df = signal_df.groupby('symbol').apply(apply_signals_to_group, sensitivity=sensitivity, include_rate_of_change=include_rate_of_change, momentum_weight=momentum_weight).reset_index(drop=True)

signal_df.to_pickle('D:/code/trading_system/data/predictions/predictions.pkl')
import pandas as pd
import datetime

# Step 1: Convert 'date' column to datetime
signal_df['date'] = pd.to_datetime(signal_df['date'],errors='coerce')

# Step 2: Determine the date 20 days ago
twenty_days_ago = datetime.datetime.now() - datetime.timedelta(days=20)

# Step 3: Filter rows within the last twenty days
filtered_df = signal_df[signal_df['date'] >= twenty_days_ago]

# Step 4: Sort by 'signal'
sorted_df = filtered_df.sort_values(by='signal', ascending=False)

# Step 5: Get the maximum date in the DataFrame
max_date = sorted_df['date'].max().strftime('%Y-%m-%d')

print(f"Maximum date: {max_date}")

sorted_df.to_excel(f'D:/code/trading_system/data/predictions/excel_output/{max_date}.xlsx', index=False)