"""
FastAPI Server for Convective Scale Nowcasting & Multi-Source GIS Dashboard
- Serves real pysteps radar forecasts (historical replay)
- Simulates Satellite/Lightning data based on actual radar
"""

import os
import io
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from fastapi import FastAPI, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import alert_dispatcher
from data_fusion import generate_insat_satellite_ctt, generate_lightning_density_grid, compute_fused_convective_hazard_index

app = FastAPI(title="India Convective Nowcasting Multi-Region API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
if os.path.exists(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

@app.get("/")
def read_root():
    html_path = os.path.join(dashboard_dir, "dashboard.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "Nowcasting API Online"}

RADAR_COLORS = [
    (0.00, (0.0, 0.0, 0.0, 0.0)),       
    (0.08, (0.12, 0.85, 0.95, 0.75)),   
    (0.20, (0.15, 0.88, 0.35, 0.85)),   
    (0.38, (0.98, 0.92, 0.15, 0.92)),   
    (0.55, (1.00, 0.50, 0.05, 0.96)),   
    (0.72, (0.95, 0.12, 0.15, 0.98)),   
    (0.88, (0.88, 0.15, 0.95, 1.00)),   
    (1.00, (1.00, 1.00, 1.00, 1.00))    
]
IMD_VIVID_CMAP = LinearSegmentedColormap.from_list("imd_vivid", [(pos, col) for pos, col in RADAR_COLORS], N=512)

REGIONS = {
    "delhi_ncr": {
        "id": "delhi_ncr",
        "name": "Delhi-NCR & Northern Plains",
        "radar": "IMD Palam & Mausam Bhavan C-Band DWR",
        "bounds": [[27.8, 76.2], [29.3, 78.2]],
        "center": [28.55, 77.2],
        "locations": [
            {"name": "Delhi IGI Airport (Aviation Hub)", "type": "airport", "lat": 28.556, "lng": 77.100, "color": "#ef4444"},
            {"name": "Gurugram CyberCity (Tech Hub)",     "type": "city",    "lat": 28.495, "lng": 77.089, "color": "#f59e0b"},
            {"name": "Noida Sector 62 (Industrial Zone)",      "type": "city",    "lat": 28.628, "lng": 77.365, "color": "#38bdf8"},
            {"name": "Faridabad Agri-Belt (Rural)",    "type": "agri",    "lat": 28.408, "lng": 77.317, "color": "#10b981"}
        ]
    }
}

import sqlite3

def get_radar_data(lead_time_min: int):
    """Loads actual radar grid from database."""
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "nowcast_data.db"))
    cursor = conn.cursor()
    # Get the latest timestamp
    cursor.execute("SELECT MAX(timestamp) FROM radar_grids")
    res = cursor.fetchone()
    if res and res[0]:
        latest_ts = res[0]
        cursor.execute("SELECT grid_blob FROM radar_grids WHERE timestamp = ? AND lead_time_min = ?", (latest_ts, lead_time_min))
        blob_res = cursor.fetchone()
        if blob_res and blob_res[0]:
            out = io.BytesIO(blob_res[0])
            grid = np.load(out)
            conn.close()
            return grid
    conn.close()
    return np.zeros((600, 600))


@app.get("/api/regions")
def get_regions():
    return {"status": "success", "regions": REGIONS}


@app.get("/api/alerts")
def get_alerts(region: str = Query("delhi_ncr")):
    reg = REGIONS.get(region, REGIONS["delhi_ncr"])
    
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "nowcast_data.db"))
    cursor = conn.cursor()
    
    # Get latest timestamp
    cursor.execute("SELECT MAX(timestamp) FROM alerts")
    res = cursor.fetchone()
    alerts = []
    latest_ts = None
    if res and res[0]:
        latest_ts = res[0]
        cursor.execute("SELECT region, message, status, severity FROM alerts WHERE timestamp = ?", (latest_ts,))
        rows = cursor.fetchall()
        for row in rows:
            alerts.append({
                "location": row[0],
                "alert_message": row[1],
                "status": row[2],
                "severity": row[3]
            })
    
    conn.close()
            
    # Load replay state metadata
    replay_meta = {}
    if os.path.exists("replay_state.json"):
        with open("replay_state.json", "r") as f:
            replay_meta = json.load(f)
            
    return {
        "status": "success", 
        "region_id": reg["id"], 
        "region_name": reg["name"],
        "data_source": replay_meta.get("data_source", "historical_replay"),
        "historical_file": replay_meta.get("historical_file", "unknown"),
        "alerts": alerts
    }


@app.get("/api/dispatches")
def get_dispatches():
    dispatches = []
    if os.path.exists("dispatched_alerts_log.json"):
        with open("dispatched_alerts_log.json", "r", encoding="utf-8") as f:
            dispatches = json.load(f)
    return {"status": "success", "dispatches": dispatches}


@app.get("/api/metrics")
def get_metrics():
    # Provide REAL metrics calculated by evaluate_metrics.py from DB
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "nowcast_data.db"))
    cursor = conn.cursor()
    cursor.execute("SELECT metrics_json FROM metrics WHERE id = 1")
    res = cursor.fetchone()
    conn.close()
    
    if res and res[0]:
        metrics = json.loads(res[0])
    else:
        metrics = {}
        
    return {"status": "success", "metrics": metrics}


@app.get("/api/layer")
@app.get("/api/layer/{layer_name}")
@app.get("/api/layer/{layer_name}/{lead_time_min}")
def get_layer_frame(layer_name: str = "radar", lead_time_min: int = 0):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    
    # Load actual radar grid from disk (historical replay)
    radar_grid = get_radar_data(lead_time_min)
    
    # Mask to circular region to match original dashboard UI look
    H, W = radar_grid.shape
    yy, xx = np.mgrid[:H, :W]
    cx, cy = W // 2, H // 2
    dist_from_radar = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    radar_horizon_mask = dist_from_radar <= 270
    radar_grid[~radar_horizon_mask] = 0.0

    if layer_name == "radar":
        masked = np.ma.masked_where(radar_grid < 2.0, radar_grid)
        ax.imshow(masked, cmap=IMD_VIVID_CMAP, vmin=2.0, vmax=80.0, alpha=0.92)

    elif layer_name == "satellite":
        sat_grid = generate_insat_satellite_ctt(radar_grid)
        sat_grid[~radar_horizon_mask] = 25.0
        masked = np.ma.masked_where(sat_grid > -15.0, sat_grid)
        ax.imshow(masked, cmap="coolwarm_r", vmin=-75, vmax=-15, alpha=0.88)

    elif layer_name == "lightning":
        lgt_grid = generate_lightning_density_grid(radar_grid)
        lgt_grid[~radar_horizon_mask] = 0.0
        masked = np.ma.masked_where(lgt_grid < 0.2, lgt_grid)
        ax.imshow(masked, cmap="plasma", vmin=0.2, vmax=5.0, alpha=0.95)

    elif layer_name == "fused":
        sat_grid = generate_insat_satellite_ctt(radar_grid)
        lgt_grid = generate_lightning_density_grid(radar_grid)
        fused_grid, _ = compute_fused_convective_hazard_index(radar_grid, sat_grid, lgt_grid)
        fused_grid[~radar_horizon_mask] = 0.0
        masked = np.ma.masked_where(fused_grid < 15.0, fused_grid)
        ax.imshow(masked, cmap="turbo", vmin=15.0, vmax=100.0, alpha=0.90)

    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    # Label headers for transparency
    headers = {
        "X-Data-Source": "historical_replay_pysteps"
    }
    if layer_name in ["satellite", "lightning", "fused"]:
        headers["X-Data-Simulation"] = "physically-informed simulation derived from real radar data"
        
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)