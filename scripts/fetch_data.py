import requests
import pandas as pd
import gzip
import json
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import sqlite3
import os
import datetime

speed = 4

def fetch_data(url):
    response = requests.get(url)
    response.raise_for_status()
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
        json_bytes = f.read()
    json_str = json_bytes.decode('utf-8')
    data = json.loads(json_str)
    return pd.DataFrame(data)

def fetch_unique_trading_symbols(url):
    df = fetch_data(url)
    df_filtered = df[(df['instrument_type'].isin(["EQ", "BE", "SM"])) & (~df['isin'].astype(str).str.startswith("INF") | df['isin'].isna())]
    return df_filtered['trading_symbol'].unique().tolist()

def get_instrument_key(trading_symbol, df):
    result = df[df['trading_symbol'] == trading_symbol]
    if not result.empty:
        return result['instrument_key'].values[0]
    return None

def generate_url(instrument_key, start_date, end_date):
    base_url = 'https://api.upstox.com/v2/historical-candle'
    return f"{base_url}/{instrument_key}/day/{end_date}/{start_date}"

def fetch_candle_data(url):
    headers = {'Accept': 'application/json'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data['data']['candles'], columns=['date', 'open', 'high', 'low', 'close', 'volume', 'unknown'])
    df.drop(columns=['unknown'], inplace=True)
    return df

def fetch_candle_data_for_symbol(args):
    symbol, df, start_date, end_date = args
    instrument_key = get_instrument_key(symbol, df)
    if instrument_key:
        try:
            candle_url = generate_url(instrument_key, start_date, end_date)
            candle_data = fetch_candle_data(candle_url)
            candle_data['symbol'] = symbol
            return candle_data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    else:
        print(f"Instrument key for {symbol} not found.")
    return None

def save_to_sqlite(df, db_conn):
    df.to_sql('candle_data', db_conn, if_exists='append', index=False)

def get_combined_candle_data(trading_symbols, start_date, end_date):
    instrument_url = 'https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz'
    df = fetch_data(instrument_url)

    # Create SQLite database connection
    db_path = 'D:/code/trading_system/data/raw/UpstoxData.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Remove the existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)

    db_conn = sqlite3.connect(db_path)

    # Create new table
    db_conn.execute('''
    CREATE TABLE candle_data (
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        symbol TEXT
    )
    ''')

    with ThreadPoolExecutor(max_workers=cpu_count() * speed) as executor:
        args_list = [(symbol, df, start_date, end_date) for symbol in trading_symbols]
        future_to_symbol = {executor.submit(fetch_candle_data_for_symbol, args): args[0] for args in args_list}

        for future in tqdm(as_completed(future_to_symbol), total=len(trading_symbols), desc="Fetching candle data"):
            result = future.result()
            if result is not None:
                save_to_sqlite(result, db_conn)

    db_conn.close()

# URL of the gzipped JSON file
url = 'https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz'

# Fetch unique trading symbols
unique_symbols = fetch_unique_trading_symbols(url)
print(f"Unique trading symbols: {len(unique_symbols)}")

trading_symbols = unique_symbols
# trading_symbols = ['TCS']
start_date = '2007-01-01'
end_date = datetime.datetime.now().strftime('%Y-%m-%d')

get_combined_candle_data(trading_symbols, start_date, end_date)

print("Data saved to SQLite database")
