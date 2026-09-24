import sqlite3, pandas as pd
c = sqlite3.connect('..\data\processed\..\data\processed\nowcast_data.db')
df = pd.read_sql_query('SELECT timestamp, region, message, severity FROM alerts ORDER BY timestamp ASC', c)
for region in df['region'].unique():
    print(f'\n--- {region} ---')
    region_df = df[df['region'] == region]
    for _, row in region_df.tail(10).iterrows():
        msg = row['message']
        if 'ETA: ' in msg:
            eta = msg.split('ETA: ')[1].split(' ')[0]
        elif 'expected in ' in msg:
            eta = msg.split('expected in ')[1].split(' mins')[0]
        else:
            eta = '0'
            
        if 'Max Rain Rate: ' in msg:
            rain = msg.split('Max Rain Rate: ')[1].split(' mm/h')[0]
        else:
            rain = 'N/A'
        print(f"Time: {row['timestamp']} | ETA: {eta} mins | Rain: {rain} mm/h")
