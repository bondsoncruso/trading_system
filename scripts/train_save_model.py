import pandas as pd
import talib
from tqdm import tqdm
import configparser
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearnex import patch_sklearn
from sklearn import set_config
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_regression, mutual_info_classif
import datetime
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
import os
# Initialize configparser
config = configparser.ConfigParser()
config.read('data/config.ini')
# Conditions to model mappings
conditions = [
    'bullish_largecap', 'bullish_midcap', 'bullish_smallcap',
    'bearish_largecap', 'bearish_midcap', 'bearish_smallcap'
]

#Inputs
training_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
user_input = input("Enter name of the model: ")
prediction_days_ahead = 5
rf_estimator = 1

# Get the feature columns as a string
feature_columns_str = config.get('parameter', 'feature_columns')

# Split the string into a list
feature_columns = feature_columns_str.split(',')
print(feature_columns)



patch_sklearn()
set_config(transform_output="pandas")


# Define all columns
all_columns = [
    'SMA', 'EMA', 'SAR', 'ADX', 'ADXR', 'APO', 'AROON_DOWN', 'AROON_UP', 'AROONOSC', 'ATR', 'AVGPRICE',
    'Bollinger_Upper', 'Bollinger_Lower', 'BETA', 'BOP', 'CCI', 'RSI', 'MACD', 'MACD_signal', 'ROC', 'CMO',
    'MFI', 'MOM', 'PPO', 'ROCP', 'ROCR', 'ROCR100', 'TRIX', 'ULTOSC', 'WILLR', 'NATR', 'TRANGE', 'OBV', 'ADL',
    'ADOSC', 'AD', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR_inphase',
    'HT_PHASOR_quadrature', 'HT_SINE', 'HT_LEADSINE', 'HT_TRENDLINE', 'HT_TRENDMODE', 'CDL2CROWS', 'CDL3BLACKCROWS',
    'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK',
    'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER',
    'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE',
    'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD',
    'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
    'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
    'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
    'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR',
    'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
]

# Define all categorical columns
all_categorical_columns = [
    'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS',
    'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
    'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING',
    'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
    'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
    'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI',
    'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
    'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR',
    'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
    'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
]

# Determine categorical columns from feature_columns
categorical_columns = [col for col in feature_columns if col in all_categorical_columns]

# Determine numerical columns from feature_columns
numerical_columns = [col for col in feature_columns if col not in categorical_columns]


# Preprocessing for numerical data
numerical_pipeline = Pipeline([
    ('robust_scaler', RobustScaler()),
    ('minmax_scaler', MinMaxScaler())
    # ('selectkbest', SelectKBest(score_func=mutual_info_regression, k=10,))
])

# Preprocessing for categorical data
categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ('selectkbest', SelectKBest(score_func=mutual_info_classif, k=10))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_columns),
    ('cat', categorical_pipeline, categorical_columns)
])


# Assuming logging is desired
# logging.basicConfig(level=logging.INFO)


# Define the regressors
# Define the regressors
regressors = {
    'catboost': CatBoostRegressor(iterations=100, learning_rate=0.1, depth=10, thread_count=-1, verbose=100, early_stopping_rounds=50),
    'xgboost': XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=10, 
        n_jobs=-1, 
        verbosity=1
        ),
    'lightgbm': LGBMRegressor(
        n_estimators=1000, 
        learning_rate=0.1, 
        max_depth=10, 
        n_jobs=-1,
        verbose=10
        )
}
# Define the stacking regressor
elasticnet_cv = ElasticNetCV(alphas=[0.1, 1.0, 10.0], l1_ratio=[0.1, 0.5, 0.9], cv=5)
stacking_regressor = StackingRegressor(
    estimators=[('catboost', regressors['catboost']), 
                ('lightgbm', regressors['lightgbm'])],
    final_estimator=elasticnet_cv,
    n_jobs=-1,
    verbose=10
)
# Iterate over each condition
for condition_selected in conditions:
    # Load the dataset for the current condition
    file_path = f'data/processed/{condition_selected}_with_indicators.pkl'
    if not os.path.exists(file_path):
        print(f"File not found for condition {condition_selected}, skipping...")
        continue

    combined_df = pd.read_pickle(file_path)

    # Set the model name based on the condition
    model_name = f'{user_input}_{condition_selected}_{training_time}'

    # Filter and process the data
    result_pd = combined_df.dropna()
    x_pd = result_pd[feature_columns]

    X = x_pd
    y = result_pd['y']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Define the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressors['lightgbm'])  # You can change to other regressors as needed
    ])

    # Train the model on the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"Condition: {condition_selected}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

    # Save the model and parameters
    params = {
        'feature_columns': ','.join(feature_columns),  # Join list as a string
        'n_estimators': str(rf_estimator),
        'prediction_days_ahead': str(prediction_days_ahead),
        'mae': str(mae),
        'mse': str(mse),
        'r2': str(r2),
        'rmse': str(rmse),
        'mape': str(mape),
        'model_filename': model_name
    }

    # Add a section for the model and save parameters
    config[model_name] = params

    # Write to config file
    config_filename = 'data/config.ini'
    with open(config_filename, 'w') as configfile:
        config.write(configfile)

    # Save the trained model
    model_filename = f'data/models/{model_name}.joblib'
    dump(pipeline, model_filename)

    # Plot the actual vs predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual', color='blue', marker='o', markersize=4, linestyle='dashed')
    plt.plot(y_pred, label='Predicted', color='orange', alpha=0.7, marker='x', markersize=4)
    plt.title(f'Actual vs Predicted Prices - {condition_selected}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()