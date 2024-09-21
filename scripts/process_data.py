import sqlite3
import pandas as pd
import os

# Define the SQLite database path
db_path = 'D:/code/trading_system/data/raw/UpstoxData.db'

# Define the output pickle file path
pickle_file_path = 'D:/code/trading_system/data/processed/UpstoxData.pkl'

print("Connecting to the SQLite database")
conn = sqlite3.connect(db_path)

# Read the data from the SQLite database into a Pandas DataFrame
print("Reading data from the SQLite database") 
query = "SELECT * FROM candle_data"
combined_df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()


# Perform the necessary data transformations using Polars
print("Performing necessary data transformations")
combined_df['date'] = pd.to_datetime(combined_df['date']).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
# Set data types
combined_df = combined_df.astype({
    'open': 'float',
    'high': 'float',
    'low': 'float',
    'close': 'float',
    'volume': 'int',
    'symbol': 'category'
})
# Sort by date
combined_df = combined_df.sort_values('date')

# Remove duplicate rows based on 'symbol' and 'date'
combined_df = combined_df.drop_duplicates(subset=['symbol', 'date'],keep='last')

# Create the output directory if it does not exist
os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)

# Save the DataFrame as a pickle file
print("Saving the DataFrame as a pickle file")
combined_df.to_pickle(pickle_file_path)

print("Data processed and saved to pickle file")

print(combined_df)
