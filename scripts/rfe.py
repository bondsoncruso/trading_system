from sklearn.feature_selection import RFE
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

# Separate feature selection step
def run_rfe(x_pd, y, estimator, n_features_to_select):
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1, verbose=2)
    rfe.fit(x_pd, y)
    return rfe.support_

# Load data
result_pd = pd.read_pickle('data/processed/UpstoxDataWithIndicators.pkl').dropna()

feature_columns = [
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

x_pd = result_pd[feature_columns]
y = result_pd['y']  # Ensure you have the target variable defined

# Run RFE with CatBoostRegressor (or any other regressor of your choice)
selected_features_mask = run_rfe(x_pd, y, CatBoostRegressor(iterations=100, learning_rate=0.3, depth=2, thread_count=-1, verbose=2), n_features_to_select=30)

# Get the selected feature names
selected_features = x_pd.columns[selected_features_mask]

print("Selected features:", selected_features)
