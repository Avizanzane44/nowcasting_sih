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

import sys, os
import threading
import time
import urllib.request
import logging

rainviewer_cache = {"path": None}

def poll_rainviewer():
    while True:
        try:
            req = urllib.request.Request("https://api.rainviewer.com/public/weather-maps.json")
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read())
                past = data.get("radar", {}).get("past", [])
                if past:
                    rainviewer_cache["path"] = past[-1]["path"]
        except Exception as e:
            logging.warning(f"Failed to fetch RainViewer API: {e}")
        time.sleep(300)

# Old threading removed

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "config"))
import config

import alert_dispatcher
from data_fusion import generate_insat_satellite_ctt, generate_lightning_density_grid, compute_fused_convective_hazard_index

app = FastAPI(title="India Convective Nowcasting Multi-Region API", version="4.0.0")

import time
import traceback
from run_nowcast import main as run_nowcast_main
from src.hazard_tracker import run_hazard_tracker_loop

def resilient_nowcast_thread():
    while True:
        try:
            print("[DAEMON] Starting run_nowcast...")
            run_nowcast_main()
        except Exception as e:
            print(f"[DAEMON ERROR] run_nowcast crashed: {e}")
            traceback.print_exc()
            time.sleep(30)

def resilient_hazard_tracker_thread():
    while True:
        try:
            print("[DAEMON] Starting hazard_tracker...")
            run_hazard_tracker_loop()
        except Exception as e:
            print(f"[DAEMON ERROR] hazard_tracker crashed: {e}")
            traceback.print_exc()
            time.sleep(30)

def resilient_rainviewer_poll():
    while True:
        try:
            poll_rainviewer()
            time.sleep(60)
        except Exception as e:
            print(f"[DAEMON ERROR] poll_rainviewer crashed: {e}")
            traceback.print_exc()
            time.sleep(30)

@app.on_event("startup")
def start_background_daemons():
    print("[API] Starting background daemons...")
    threading.Thread(target=resilient_nowcast_thread, daemon=True).start()
    threading.Thread(target=resilient_hazard_tracker_thread, daemon=True).start()
    threading.Thread(target=resilient_rainviewer_poll, daemon=True).start()

import psutil

@app.middleware("http")
async def memory_logging_middleware(request, call_next):
    response = await call_next(request)
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEMORY] API {request.url.path} RSS: {mem_mb:.1f} MB")
    return response

import json

@app.get("/api/status")
def get_zone_status(region: str):
    path = os.path.join(os.path.dirname(__file__), "..", f"{region}_status.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"coherence_passed": False, "active_rain_present": False, "bounds": None}

@app.get("/api/layer-rainviewer/{region}/{lead_time_min}")
def get_rainviewer_layer(region: str, lead_time_min: int):
    # Depending on the region, we fetch the corresponding cache dir
    cache_map = {
        "mumbai_ghats": "mumbai_cache",
        "himalayan_belt": "himalayan_cache",
        "northeast_bengal": "northeast_cache",
        "delhi_ncr": "delhi_cache"
    }
    cache_dir = cache_map.get(region)
    if not cache_dir:
        return Response(content=b"", media_type="image/png")
        
    npy_path = os.path.join(os.path.dirname(__file__), "..", "tests", cache_dir, f"forecast_grid_{lead_time_min}.npy")
    if not os.path.exists(npy_path):
        return Response(content=b"", media_type="image/png")
        
    grid = np.load(npy_path)
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    # We do NOT use a binary NumPy mask because masked_where creates sharp, blocky, 
    # rectangular cutoffs at the edge of the 256x256 grid cells.
    # Instead, we rely on the colormap (where 0.0 maps to (0,0,0,0) transparent)
    # and interpolation to create smooth, organic gradients at the edges.
    smoothed = gaussian_filter(grid, sigma=1.5)
    ax.imshow(smoothed, cmap=IMD_VIVID_CMAP, vmin=0.0, vmax=80.0, alpha=0.92, interpolation='bilinear')
    
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


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

REGIONS = config.REGIONS


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



@app.get("/api/radar-tiles")
def get_radar_tiles():
    if not rainviewer_cache["path"]:
        return {"status": "error", "message": "Radar path not cached yet"}
    return {
        "status": "success",
        "path": rainviewer_cache["path"],
        "template": "https://tilecache.rainviewer.com{path}/256/{z}/{x}/{y}/2/1_1.png"
    }

@app.get("/api/regions")
def get_regions():
    return {"status": "success", "regions": REGIONS}


@app.get("/api/alerts")
def get_alerts(region: str = Query("delhi_ncr")):
    if region != "delhi_ncr":
        json_path = os.path.join(os.path.dirname(__file__), "..", f"{region}_hazard_alerts.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                alerts_data = json.load(f)
            return {"data_source": "live_rainviewer", "alerts": alerts_data}
        return {"data_source": "live_rainviewer", "alerts": []}

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
        # Filter alerts to only include those relevant to the requested region
        location_names = [loc["name"] for loc in reg.get("locations", [])]
        for row in rows:
            if row[0] in location_names:
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
def get_dispatches(region: str = "delhi_ncr"):
    dispatches = []
    if region == "delhi_ncr" and os.path.exists("dispatched_alerts_log.json"):
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
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
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
        # Use colormap transparency instead of binary masking for smooth edges
        smoothed = gaussian_filter(radar_grid, sigma=1.5)
        ax.imshow(smoothed, cmap=IMD_VIVID_CMAP, vmin=0.0, vmax=80.0, alpha=0.92, interpolation='bilinear')

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
    plt.savefig(buf, format="png", transparent=True)
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