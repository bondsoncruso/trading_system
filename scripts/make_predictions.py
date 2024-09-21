import configparser
import pandas as pd
from joblib import load
from sklearnex import patch_sklearn
import pickle
import os

# Initialize sklearn optimizations
patch_sklearn()

# Load configuration
config = configparser.ConfigParser()
config.read('D:/code/trading_system/data/config.ini')

# Extract the list of selected models
model_selected_str = config.get('parameter', 'model_selected')
model_selected_list = model_selected_str.strip().split(',')
print(f"Selected models: {model_selected_list}")

# Path for data
data_base_path = 'D:/code/trading_system/data/processed/'

# Directory to save predictions
predictions_dir = 'D:/code/trading_system/data/predictions'
os.makedirs(predictions_dir, exist_ok=True)

# Initialize a list to store combined predictions with symbol and date
combined_results = []

# Iterate over each selected model
for model_selected in model_selected_list:
    model_selected = model_selected.strip()  # Remove any leading/trailing whitespace
    
    # Load feature columns for the selected model
    feature_columns_str = config.get(model_selected, "feature_columns")
    feature_columns = feature_columns_str.strip().split(",")
    
    # Extract condition from the model name (assumes format: <model>_<condition>_<timestamp>)
    try:
        model_name, condition_with_timestamp = model_selected.rsplit('_', 1)
        condition = model_name.split('_', 1)[1]
    except ValueError:
        print(f"Unexpected model format: {model_selected}, skipping...")
        continue

    # Define data path based on model name (assumes specific data per model)
    data_path = f'{data_base_path}{condition}_with_indicators.pkl'
    if not os.path.exists(data_path):
        print(f"Data file not found for condition {condition}, skipping...")
        continue

    # Load the dataset for predictions
    data_df = pd.read_pickle(data_path)
    data_df = data_df[feature_columns + ['symbol', 'date']].dropna()

    # Load the pre-trained model
    model_path = f'D:/code/trading_system/data/models/{model_selected}.joblib'
    model = load(model_path, mmap_mode='r+')

    # Prepare data for predictions
    X = data_df[feature_columns]

    # Make predictions
    predictions = model.predict(X)

    # Store predictions with symbol and date in the combined_results list
    results = pd.DataFrame({
        'symbol': data_df['symbol'],
        'date': data_df['date'],
        'Predicted_Price': predictions
    })
    results['condition'] = condition  # Add condition for identification
    combined_results.append(results)

    print(f"Predictions made for model: {model_selected}")

# Concatenate all results into a single DataFrame
final_results_df = pd.concat(combined_results, ignore_index=True)

# Define save path for combined predictions DataFrame
combined_save_path = os.path.join(predictions_dir, 'raw_predictions.pkl')

# Save the combined predictions DataFrame
final_results_df.to_pickle(combined_save_path)
print(final_results_df)
print(f"Combined predictions saved at {combined_save_path}")
