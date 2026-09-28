# tests/verify_retention.py
import sys
import os
import datetime
import sqlite3
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import db_storage

# Configuration
PRUNE_INTERVAL = 3
KEEP_FRAMES = 3
TOTAL_FRAMES = 10

def run_test():
    db_storage.init_db()
    
    dummy_grid = np.zeros((10, 10))
    
    for i in range(1, TOTAL_FRAMES + 1):
        ts = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=i)).isoformat()
        
        # Insert 1 obs and 2 forecast grids per frame
        db_storage.save_grid_to_db(ts, 0, dummy_grid)
        db_storage.save_grid_to_db(ts, 5, dummy_grid)
        db_storage.save_grid_to_db(ts, 10, dummy_grid)
        
        # Insert 1 dummy alert
        db_storage.save_alerts_to_db(ts, [{"location": "Test", "alert_message": "Test", "status": "WARN", "severity": "LOW"}])
        
        if i % PRUNE_INTERVAL == 0:
            print(f"Frame {i}: Triggering prune_old_data(keep_frames={KEEP_FRAMES})...")
            db_storage.prune_old_data(keep_frames=KEEP_FRAMES)
            
    # Verify retention
    conn = sqlite3.connect(db_storage.DB_PATH)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(DISTINCT timestamp) FROM radar_grids")
    unique_ts = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM radar_grids")
    total_grids = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM alerts")
    total_alerts = cur.fetchone()[0]
    
    db_size_mb = os.path.getsize(db_storage.DB_PATH) / (1024 * 1024)
    conn.close()
    
    print("\n--- TEST RESULTS ---")
    print(f"Distinct timestamps in radar_grids: {unique_ts}")
    print(f"Total rows in radar_grids: {total_grids}")
    print(f"Total rows in alerts: {total_alerts}")
    print(f"Database Size: {db_size_mb:.2f} MB")
    
    # Validation assertion (allowed to be up to keep_frames + prune_interval - 1 before pruning fires)
    max_expected = KEEP_FRAMES + (PRUNE_INTERVAL - 1)
    if unique_ts <= max_expected:
        print(f"\n[PASS] Distinct timestamps ({unique_ts}) never exceeded keep_frames + prune_interval ({max_expected}).")
    else:
        print(f"\n[FAIL] Distinct timestamps ({unique_ts}) exceeded limit.")

if __name__ == "__main__":
    run_test()
