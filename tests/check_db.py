import sqlite3, pandas as pd
c = sqlite3.connect('..\data\processed\..\data\processed\nowcast_data.db')
df = pd.read_sql_query('SELECT timestamp, region, message, status FROM alerts WHERE region LIKE "%Noida%" ORDER BY timestamp DESC LIMIT 5', c)
for _, row in df.iterrows():
    print(f"{row['timestamp']} | {row['status']}")
