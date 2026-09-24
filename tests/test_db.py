import sqlite3
import pandas as pd

conn = sqlite3.connect("..\data\processed\..\data\processed\nowcast_data.db")

print("=== ALERTS ===")
print(pd.read_sql_query("SELECT timestamp, region, status, severity FROM alerts LIMIT 5", conn))

print("\n=== STORM CELLS ===")
print(pd.read_sql_query("SELECT timestamp, lead_time_min, cell_id, geom_wkt, hazard_type FROM storm_cells LIMIT 5", conn))

print("\n=== RADAR GRIDS ===")
print(pd.read_sql_query("SELECT timestamp, lead_time_min, length(grid_blob) as blob_size FROM radar_grids LIMIT 5", conn))
conn.close()
