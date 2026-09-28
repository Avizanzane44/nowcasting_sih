import sqlite3
import json
import numpy as np
import io
import time

import os
import datetime
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "nowcast_data.db")

def get_connection(retries=5, delay=0.5):
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=20.0)
            # Use TRUNCATE or MEMORY journal to avoid WSL/NTFS file locking errors with WAL or DELETE
            conn.execute("PRAGMA journal_mode=TRUNCATE;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            return conn
        except sqlite3.OperationalError as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

def execute_with_retry(func):
    def wrapper(*args, **kwargs):
        retries = 5
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "disk I/O error" in str(e) or "database is locked" in str(e) or "readonly database" in str(e):
                    if attempt < retries - 1:
                        time.sleep(0.5)
                        continue
                raise e
    return wrapper

@execute_with_retry
def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS radar_grids (
        timestamp TEXT,
        lead_time_min INTEGER,
        grid_blob BLOB,
        PRIMARY KEY(timestamp, lead_time_min)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        region TEXT,
        message TEXT,
        status TEXT,
        severity TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS storm_cells (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        lead_time_min INTEGER,
        cell_id INTEGER,
        geom_wkt TEXT,
        hazard_type TEXT,
        severity TEXT,
        intensity REAL
    )
    ''')

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_storm_cells_ts ON storm_cells (timestamp);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts (timestamp);")

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY,
        metrics_json TEXT
    )
    ''')

    conn.commit()
    conn.close()

@execute_with_retry
def save_metrics_to_db(metrics_dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO metrics (id, metrics_json) VALUES (1, ?)", (json.dumps(metrics_dict),))
    conn.commit()
    conn.close()

@execute_with_retry
def save_grid_to_db(timestamp, lead_time_min, grid_array):
    conn = get_connection()
    cursor = conn.cursor()
    
    out = io.BytesIO()
    np.save(out, grid_array)
    blob = out.getvalue()
    
    cursor.execute(
        "INSERT OR REPLACE INTO radar_grids (timestamp, lead_time_min, grid_blob) VALUES (?, ?, ?)",
        (timestamp, lead_time_min, blob)
    )
    conn.commit()
    conn.close()

@execute_with_retry
@execute_with_retry
def save_alerts_to_db(timestamp, alerts):
    conn = get_connection()
    cursor = conn.cursor()
    for alert in alerts:
        cursor.execute(
            "INSERT INTO alerts (timestamp, region, message, status, severity) VALUES (?, ?, ?, ?, ?)",
            (timestamp, alert["location"], alert["alert_message"], alert["status"], alert["severity"])
        )
    conn.commit()
    conn.close()

@execute_with_retry
def save_storm_cells_to_db(timestamp, all_cells):
    conn = get_connection()
    cursor = conn.cursor()
    for lead_time_min, cells in all_cells.items():
        for cell in cells:
            bbox = cell["bbox"]
            min_x, min_y = bbox["min_x"], bbox["min_y"]
            max_x, max_y = bbox["max_x"], bbox["max_y"]
            wkt = f"POLYGON (({min_x} {min_y}, {max_x} {min_y}, {max_x} {max_y}, {min_x} {max_y}, {min_x} {min_y}))"
            
            cursor.execute(
                "INSERT INTO storm_cells (timestamp, lead_time_min, cell_id, geom_wkt, hazard_type, severity, intensity) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (timestamp, lead_time_min, cell["cell_id"], wkt, cell["hazard_type"], cell["severity"], cell["max_intensity_mmh"])
            )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")

@execute_with_retry
def prune_old_data(keep_frames=10):
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 5 * 1024**3:
        print(f"\n[WARNING] DB is {os.path.getsize(DB_PATH) / 1024**3:.2f} GB! Delete data/processed/nowcast_data.db* and restart. Skipping prune.\n")
        return
        
    conn = get_connection()
    cursor = conn.cursor()
    
    # Keep only most recent N distinct timestamps in radar_grids
    cursor.execute("SELECT DISTINCT timestamp FROM radar_grids ORDER BY timestamp DESC LIMIT ?", (keep_frames,))
    rows = cursor.fetchall()
    if rows:
        oldest_keep_ts = rows[-1][0]
        cursor.execute("DELETE FROM radar_grids WHERE timestamp < ?", (oldest_keep_ts,))
        cursor.execute("DELETE FROM storm_cells WHERE timestamp < ?", (oldest_keep_ts,))
        
    # Delete alerts older than 1 day
    one_day_ago = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).isoformat()
    cursor.execute("DELETE FROM alerts WHERE timestamp < ?", (one_day_ago,))
    
    conn.commit()
    conn.close()
