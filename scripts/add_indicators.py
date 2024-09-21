import pandas as pd
import talib
import os
from tqdm import tqdm

file_path = 'D:/code/trading_system/data/processed/UpstoxData.pkl'

combined_df = pd.read_pickle(file_path)

# Identify unique symbols where the volume is less than zero
symbols_to_remove = combined_df[combined_df['volume'] < 0]['symbol'].unique()

# Filter the original DataFrame to exclude the rows with these symbols
combined_df = combined_df[~combined_df['symbol'].isin(symbols_to_remove)]
# manual_symbols = [
#     'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
#     'HDFC', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'BAJFINANCE',
#     'BHARTIARTL', 'ITC', 'ASIANPAINT', 'LT', 'WIPRO',
#     'AXISBANK', 'MARUTI', 'ONGC', 'SUNPHARMA', 'HCLTECH',
#     # 'HINDALCO', 'SHREECEM', 'TECHM', 'NTPC', 'TITAN',
#     # 'HEROMOTOCO', 'COALINDIA', 'BRITANNIA', 'DRREDDY',
#     'BAJAJFINSV', 'BAJAJHLDNG', 'CIPLA', 'EICHERMOT',
#     'ZOMATO'
#     ]

# combined_df = combined_df[combined_df['symbol'].isin(manual_symbols)]
# Sort the DataFrame first by 'symbol' and then by 'date', both in ascending order
combined_df = combined_df.sort_values(by=['symbol', 'date'], ascending=[True, True]).reset_index(drop=True)

market_cap_csv_path = 'D:/code/trading_system/data/tickertape/mkt_cap.csv'
nifty500_csv_path = input("Enter the path to the NIFTY 500 CSV file: ")
nifty500df = pd.read_csv(nifty500_csv_path)
market_cap_df = pd.read_csv(market_cap_csv_path)
market_cap_df = market_cap_df[['Ticker', 'Market Cap']]
combined_df = combined_df.merge(market_cap_df, left_on='symbol', right_on='Ticker', how='inner')
combined_df.drop(columns=['Ticker'], inplace=True)
nifty500df = nifty500df[['Date', 'Close']]
nifty500df = nifty500df.rename(columns={'Close': 'NIFTY 500', 'Date': 'Nifty 500 Date'})
combined_df['date'] = pd.to_datetime(combined_df['date'])
nifty500df['Nifty 500 Date'] = pd.to_datetime(nifty500df['Nifty 500 Date'], format='%d %b %Y')
combined_df = combined_df.merge(nifty500df, left_on='date', right_on='Nifty 500 Date', how='inner')
combined_df.drop(columns=['Nifty 500 Date'], inplace=True)
print(combined_df)
# Function to apply TA-Lib indicators
def apply_talib_indicators(group_df):
    if group_df.empty:
        print("Received an empty DataFrame.")
        return pd.DataFrame()

    # Prices and volume
    close = group_df['close'].astype(float).to_numpy()
    high = group_df['high'].astype(float).to_numpy()
    low = group_df['low'].astype(float).to_numpy()
    volume = group_df['volume'].astype(float).to_numpy()
    open = group_df['open'].astype(float).to_numpy()

    indicators = {}

    # Trend Indicators
    indicators['SMA'] = talib.SMA(close, timeperiod=10)
    indicators['EMA'] = talib.EMA(close, timeperiod=20)
    indicators['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
    indicators['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    indicators['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
    indicators['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    aroondown, aroonup = talib.AROON(high, low, timeperiod=14)
    indicators['AROON_DOWN'] = aroondown
    indicators['AROON_UP'] = aroonup
    indicators['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
    indicators['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    indicators['AVGPRICE'] = talib.AVGPRICE(open, high, low, close)
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    indicators['Bollinger_Upper'] = upperband
    indicators['Bollinger_Lower'] = lowerband
    indicators['BETA'] = talib.BETA(high, low, timeperiod=5)
    indicators['BOP'] = talib.BOP(open, high, low, close)
    indicators['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    
    # Candlestick Patterns
    patterns = {
        'CDL2CROWS': talib.CDL2CROWS,
        'CDL3BLACKCROWS': talib.CDL3BLACKCROWS,
        'CDL3INSIDE': talib.CDL3INSIDE,
        'CDL3LINESTRIKE': talib.CDL3LINESTRIKE,
        'CDL3STARSINSOUTH': talib.CDL3STARSINSOUTH,
        'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS,
        'CDLABANDONEDBABY': talib.CDLABANDONEDBABY,
        'CDLADVANCEBLOCK': talib.CDLADVANCEBLOCK,
        'CDLBELTHOLD': talib.CDLBELTHOLD,
        'CDLBREAKAWAY': talib.CDLBREAKAWAY,
        'CDLCLOSINGMARUBOZU': talib.CDLCLOSINGMARUBOZU,
        'CDLCONCEALBABYSWALL': talib.CDLCONCEALBABYSWALL,
        'CDLCOUNTERATTACK': talib.CDLCOUNTERATTACK,
        'CDLDARKCLOUDCOVER': talib.CDLDARKCLOUDCOVER,
        'CDLDOJI': talib.CDLDOJI,
        'CDLDOJISTAR': talib.CDLDOJISTAR,
        'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI,
        'CDLENGULFING': talib.CDLENGULFING,
        'CDLEVENINGDOJISTAR': talib.CDLEVENINGDOJISTAR,
        'CDLEVENINGSTAR': talib.CDLEVENINGSTAR,
        'CDLGAPSIDESIDEWHITE': talib.CDLGAPSIDESIDEWHITE,
        'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI,
        'CDLHAMMER': talib.CDLHAMMER,
        'CDLHANGINGMAN': talib.CDLHANGINGMAN,
        'CDLHARAMI': talib.CDLHARAMI,
        'CDLHARAMICROSS': talib.CDLHARAMICROSS,
        'CDLHIGHWAVE': talib.CDLHIGHWAVE,
        'CDLHIKKAKE': talib.CDLHIKKAKE,
        'CDLHIKKAKEMOD': talib.CDLHIKKAKEMOD,
        'CDLHOMINGPIGEON': talib.CDLHOMINGPIGEON,
        'CDLIDENTICAL3CROWS': talib.CDLIDENTICAL3CROWS,
        'CDLINNECK': talib.CDLINNECK,
        'CDLINVERTEDHAMMER': talib.CDLINVERTEDHAMMER,
        'CDLKICKING': talib.CDLKICKING,
        'CDLKICKINGBYLENGTH': talib.CDLKICKINGBYLENGTH,
        'CDLLADDERBOTTOM': talib.CDLLADDERBOTTOM,
        'CDLLONGLEGGEDDOJI': talib.CDLLONGLEGGEDDOJI,
        'CDLLONGLINE': talib.CDLLONGLINE,
        'CDLMARUBOZU': talib.CDLMARUBOZU,
        'CDLMATCHINGLOW': talib.CDLMATCHINGLOW,
        'CDLMATHOLD': talib.CDLMATHOLD,
        'CDLMORNINGDOJISTAR': talib.CDLMORNINGDOJISTAR,
        'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
        'CDLONNECK': talib.CDLONNECK,
        'CDLPIERCING': talib.CDLPIERCING,
        'CDLRICKSHAWMAN': talib.CDLRICKSHAWMAN,
        'CDLRISEFALL3METHODS': talib.CDLRISEFALL3METHODS,
        'CDLSEPARATINGLINES': talib.CDLSEPARATINGLINES,
        'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
        'CDLSHORTLINE': talib.CDLSHORTLINE,
        'CDLSPINNINGTOP': talib.CDLSPINNINGTOP,
        'CDLSTALLEDPATTERN': talib.CDLSTALLEDPATTERN,
        'CDLSTICKSANDWICH': talib.CDLSTICKSANDWICH,
        'CDLTAKURI': talib.CDLTAKURI,
        'CDLTASUKIGAP': talib.CDLTASUKIGAP,
        'CDLTHRUSTING': talib.CDLTHRUSTING,
        'CDLTRISTAR': talib.CDLTRISTAR,
        'CDLUNIQUE3RIVER': talib.CDLUNIQUE3RIVER,
        'CDLUPSIDEGAP2CROWS': talib.CDLUPSIDEGAP2CROWS,
        'CDLXSIDEGAP3METHODS': talib.CDLXSIDEGAP3METHODS
    }

    for pattern_name, pattern_func in patterns.items():
        indicators[pattern_name] = pattern_func(open, high, low, close)
    
    # Momentum Indicators
    indicators['RSI'] = talib.RSI(close, timeperiod=14)
    macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['MACD'] = macd
    indicators['MACD_signal'] = macd_signal
    indicators['ROC'] = talib.ROC(close, timeperiod=10)
    indicators['CMO'] = talib.CMO(close, timeperiod=14)
    indicators['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
    indicators['MOM'] = talib.MOM(close, timeperiod=10)
    indicators['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    indicators['ROCP'] = talib.ROCP(close, timeperiod=10)
    indicators['ROCR'] = talib.ROCR(close, timeperiod=10)
    indicators['ROCR100'] = talib.ROCR100(close, timeperiod=10)
    indicators['TRIX'] = talib.TRIX(close, timeperiod=30)
    indicators['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    indicators['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

    # Volatility Indicators
    indicators['NATR'] = talib.NATR(high, low, close, timeperiod=14)
    indicators['TRANGE'] = talib.TRANGE(high, low, close)

    # Volume Indicators
    indicators['OBV'] = talib.OBV(close, volume)
    indicators['ADL'] = talib.AD(high, low, close, volume)
    indicators['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    indicators['AD'] = talib.AD(high, low, close, volume)

    # Price Transform
    indicators['AVGPRICE'] = talib.AVGPRICE(open, high, low, close)
    indicators['MEDPRICE'] = talib.MEDPRICE(high, low)
    indicators['TYPPRICE'] = talib.TYPPRICE(high, low, close)
    indicators['WCLPRICE'] = talib.WCLPRICE(high, low, close)

    # Cycle Indicators
    indicators['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    indicators['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    inphase, quadrature = talib.HT_PHASOR(close)
    indicators['HT_PHASOR_inphase'] = inphase
    indicators['HT_PHASOR_quadrature'] = quadrature
    sine, leadsine = talib.HT_SINE(close)
    indicators['HT_SINE'] = sine
    indicators['HT_LEADSINE'] = leadsine
    indicators['HT_TRENDLINE'] = talib.HT_TRENDLINE(close)
    indicators['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

    # Create DataFrame from indicators
    indicators_df = pd.DataFrame(indicators, index=group_df.index)

    # Concatenate the original DataFrame with the indicators DataFrame
    group_df = pd.concat([group_df, indicators_df], axis=1)
    for period in [5, 15]:
        group_df[f'Nifty500_{period}EMA'] = talib.EMA(group_df['NIFTY 500'].astype(float), timeperiod=period)

    # Create target variable
    group_df['y'] = group_df['close'].shift(-1)  # Calculate percentage change and shift to align with the next day

    return group_df

# Apply the TA-Lib indicators function to each group and concatenate the results with progress bar
result_dfs = []
for symbol, group_df in tqdm(combined_df.groupby('symbol'), desc="Processing tickers"):
    result_dfs.append(apply_talib_indicators(group_df))

# Concatenate the results
result_df = pd.concat(result_dfs)

result_pd = result_df.reset_index(drop=True)
conditions = {
    'bullish_largecap': (result_pd['Nifty500_5EMA'] > result_pd['Nifty500_15EMA']) & (result_pd['Market Cap'] > 90556.12),
    'bullish_midcap': (result_pd['Nifty500_5EMA'] > result_pd['Nifty500_15EMA']) & (result_pd['Market Cap'].between(29547.26, 90556.12)),
    'bullish_smallcap': (result_pd['Nifty500_5EMA'] > result_pd['Nifty500_15EMA']) & (result_pd['Market Cap'] <= 29547.26),
    'bearish_largecap': (result_pd['Nifty500_5EMA'] < result_pd['Nifty500_15EMA']) & (result_pd['Market Cap'] > 90556.12),
    'bearish_midcap': (result_pd['Nifty500_5EMA'] < result_pd['Nifty500_15EMA']) & (result_pd['Market Cap'].between(29547.26, 90556.12)),
    'bearish_smallcap': (result_pd['Nifty500_5EMA'] < result_pd['Nifty500_15EMA']) & (result_pd['Market Cap'] <= 29547.26),
}
output_dir = 'D:/code/trading_system/data/processed/'
# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)
    # Save each filtered DataFrame as a pickle file
for name, condition in conditions.items():
    filtered_df = result_pd[condition]
    pickle_file_path = os.path.join(output_dir, f"{name}_with_indicators.pkl")
    print(f"Saving {name} DataFrame as a pickle file")
    filtered_df.to_pickle(pickle_file_path)

print("All DataFrames processed and saved to pickle files.")
result_pd.to_pickle('D:/code/trading_system/data/processed/UpstoxDataWithIndicators.pkl')