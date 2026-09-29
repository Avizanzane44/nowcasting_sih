"""
Hazard Extraction & Storm Tracking Engine - India Region
- Configured for IMD Doppler Weather Radar Grid
- Real Indian Hubs (Airports, Tech Corridors, Agricultural Sectors)
"""

import os
import json
import datetime
import math
import cv2
import numpy as np
import requests
from scipy import ndimage
from scipy.stats import circmean, circstd
from scipy.ndimage import median_filter, gaussian_filter
from PIL import Image

# ==========================================
# 1. HAZARD THRESHOLD DEFINITIONS (IMD Standards)
# ==========================================
# Cloudburst: >= 100 mm/h (IMD standard definition) or >= 50 mm/h localized extreme
CLOUDBURST_THRESHOLD_MMH = 50.0  
# Severe Hail & Downburst Squall: >= 30 mm/h (reflectivity > 45 dBZ)
HAIL_THRESHOLD_MMH = 30.0        
# Moderate Convective Thunderstorm: >= 15 mm/h (reflectivity > 38 dBZ)
THUNDERSTORM_THRESHOLD_MMH = 15.0 

MIN_CELL_PIXELS = 150 


def classify_cell(max_intensity, area_pixels):
    """Classifies hazard category according to IMD severe weather protocols."""
    if max_intensity >= CLOUDBURST_THRESHOLD_MMH:
        return "CLOUDBURST_FLASH_FLOOD", "CRITICAL"
    elif max_intensity >= HAIL_THRESHOLD_MMH:
        return "SEVERE_HAIL_DOWNBURST", "HIGH"
    elif max_intensity >= THUNDERSTORM_THRESHOLD_MMH:
        return "THUNDERSTORM_GUSTY_WIND", "MODERATE"
    return "CONVECTIVE_RAIN", "LOW"


def extract_storm_cells(rain_grid, timestep_minutes, base_time):
    binary_mask = (rain_grid >= THUNDERSTORM_THRESHOLD_MMH).astype(np.uint8)
    labeled_array, num_features = ndimage.label(binary_mask)
    
    cells = []
    for cell_id in range(1, num_features + 1):
        cell_mask = (labeled_array == cell_id)
        cell_size = np.sum(cell_mask)
        
        if cell_size < MIN_CELL_PIXELS:
            continue
            
        cell_rain_values = rain_grid[cell_mask]
        max_rain = float(np.max(cell_rain_values))
        mean_rain = float(np.mean(cell_rain_values))
        
        cy, cx = ndimage.center_of_mass(cell_mask)
        y_indices, x_indices = np.where(cell_mask)
        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
        
        hazard_type, severity = classify_cell(max_rain, cell_size)
        
        cells.append({
            "cell_id": cell_id,
            "lead_time_min": timestep_minutes,
            "forecast_time": (base_time + datetime.timedelta(minutes=timestep_minutes)).isoformat(),
            "centroid": {"x": round(cx, 1), "y": round(cy, 1)},
            "bbox": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
            "area_pixels": int(cell_size),
            "max_intensity_mmh": round(max_rain, 1),
            "mean_intensity_mmh": round(mean_rain, 1),
            "hazard_type": hazard_type,
            "severity": severity
        })
    return cells


def compute_arrival_countdowns(forecast_frames_cells, monitored_locations, coherence_passed=True, active_rain_present=True):
    alerts = []
    for loc in monitored_locations:
        loc_x, loc_y = loc.get("x", 0), loc.get("y", 0)
        loc_name = loc["name"]
        
        hit_found = False
        for lead_time_min in sorted(forecast_frames_cells.keys()):
            cells_at_time = forecast_frames_cells[lead_time_min]
            for cell in cells_at_time:
                bbox = cell["bbox"]
                if (bbox["min_x"] - 3 <= loc_x <= bbox["max_x"] + 3) and \
                   (bbox["min_y"] - 3 <= loc_y <= bbox["max_y"] + 3):
                    
                    if not coherence_passed and active_rain_present:
                        alerts.append({
                            "location": loc_name,
                            "status": "NO_COHERENT_ADVECTION",
                            "eta_minutes": None,
                            "hazard_type": cell["hazard_type"],
                            "severity": cell["severity"],
                            "expected_peak_rain_mmh": cell["max_intensity_mmh"],
                            "forecast_arrival": None,
                            "alert_message": f"[INFO] {loc_name}: Storm is stationary/bubbling in place. No reliable bulk advection ETA."
                        })
                    else:
                        alerts.append({
                            "location": loc_name,
                            "status": "IMMINENT_HAZARD",
                            "eta_minutes": lead_time_min,
                            "hazard_type": cell["hazard_type"],
                            "severity": cell["severity"],
                            "expected_peak_rain_mmh": cell["max_intensity_mmh"],
                            "forecast_arrival": cell["forecast_time"],
                            "alert_message": f"[WARN] {cell['severity']} Early Warning for {loc_name}: {cell['hazard_type']} arriving in {lead_time_min} mins!"
                        })
                    hit_found = True
                    break
            if hit_found:
                break
                
        if not hit_found:
            if not active_rain_present:
                status = "CLEAR"
                msg = f"[OK] {loc_name}: No active rain detected in the zone at all."
            elif not coherence_passed:
                status = "NO_COHERENT_ADVECTION"
                msg = f"[INFO] {loc_name}: Active rain present in zone, but no coherent advection (stationary/bubbling)."
            else:
                status = "CLEAR"
                msg = f"[OK] {loc_name}: No convective hazards approaching in next 60 minutes."
                
            alerts.append({
                "location": loc_name,
                "status": status,
                "eta_minutes": None,
                "hazard_type": "NONE",
                "severity": "NORMAL",
                "alert_message": msg
            })
    return alerts



# ==========================================
# RAINVIEWER LIVE RADAR NOWCASTING ENGINE
# ==========================================

# Data Provenance: RainViewer redistributed IMD Doppler Radar composite tiles (mausam.imd.gov.in).
# Note: Intensity values are reverse-engineered from PNG color palette steps, 
# not calibrated raw Level-1/2 reflectivity arrays.

RAINVIEWER_COLOR_SCALE_DBZ = {
    (0, 0, 0, 0): 0.0,
    (0, 78, 120): 10.0,
    (0, 91, 142): 15.0,
    (0, 105, 156): 20.0,
    (0, 127, 180): 25.0,
    (0, 154, 213): 30.0,
    (0, 163, 224): 35.0,
    (0, 255, 0): 40.0,
    (173, 255, 47): 42.0,
    (255, 255, 0): 45.0,
    (255, 238, 0): 48.0,
    (255, 210, 0): 50.0,
    (255, 139, 0): 55.0,
    (255, 0, 0): 60.0,
    (200, 0, 0): 65.0,
    (255, 0, 255): 70.0,
}

rv_colors = np.array([list(k[:3]) for k in RAINVIEWER_COLOR_SCALE_DBZ.keys() if k != (0,0,0,0)])
rv_intensities = np.array([v for k, v in RAINVIEWER_COLOR_SCALE_DBZ.items() if k != (0,0,0,0)])

def _color_to_dbz(rgb):
    r, g, b, a = rgb
    if a == 0:
        return 0.0
    t = (r, g, b)
    if t in RAINVIEWER_COLOR_SCALE_DBZ:
        return RAINVIEWER_COLOR_SCALE_DBZ[t]
    dists = np.sum((rv_colors - [r, g, b]) ** 2, axis=1)
    best_idx = np.argmin(dists)
    if dists[best_idx] > 20000:
        return 0.0
    return rv_intensities[best_idx]

def decode_rainviewer_png(img_path):
    """Decodes a RainViewer PNG tile into a 2D numpy reflectivity grid (dBZ)."""
    img = Image.open(img_path).convert("RGBA")
    arr = np.array(img)
    h, w, c = arr.shape
    flat_pixels = arr.reshape(-1, 4)
    flat_decoded = np.zeros(flat_pixels.shape[0], dtype=np.float32)
    for i, p in enumerate(flat_pixels):
        flat_decoded[i] = _color_to_dbz(p)
    return flat_decoded.reshape((h, w))


def get_tile_bounds(zoom, x, y):
    n = 2.0 ** zoom
    nw_lon = x / n * 360.0 - 180.0
    nw_lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    nw_lat = math.degrees(nw_lat_rad)
    
    se_lon = (x + 1) / n * 360.0 - 180.0
    se_lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    se_lat = math.degrees(se_lat_rad)
    
    return [[se_lat, nw_lon], [nw_lat, se_lon]]

def latlng_to_tile_pixel(lat, lon, zoom, tile_x, tile_y):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    global_x = (lon + 180.0) / 360.0 * (n * 256)
    global_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (n * 256)
    pixel_x = int(global_x - tile_x * 256)
    pixel_y = int(global_y - tile_y * 256)
    return pixel_x, pixel_y

def run_rainviewer_nowcast_for_zone(zone_id, zone_name, cache_dir, zoom, tile_x, tile_y, locations, max_angular_std_dev=30.0):
    """
    Runs the complete RainViewer live nowcasting engine for a given zone.
    Includes: Fetch -> Decode -> Farneback Motion -> Spatial Filtering -> Angular Coherence Gate -> Extrapolation -> Alert Dispatch.
    """
    print("\n=======================================================")
    print(f"RUNNING LIVE RAINVIEWER NOWCAST: {zone_name.upper()}")
    print("=======================================================")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    res = requests.get('https://api.rainviewer.com/public/weather-maps.json')
    past_frames = res.json()['radar']['past'][-4:]
    
    frames = []
    timestamps = []
    

    
    for frame in past_frames:
        path = frame['path']
        ts = frame['time']
        timestamps.append(ts)
        url = f"https://tilecache.rainviewer.com{path}/256/{zoom}/{tile_x}/{tile_y}/2/0_0.png"
        img_path = os.path.join(cache_dir, f"tile_{ts}.png")
        
        if not os.path.exists(img_path):
            with open(img_path, 'wb') as f:
                f.write(requests.get(url).content)
                
        grid = decode_rainviewer_png(img_path)
        frames.append(grid)
        
    print(f"Loaded {len(frames)} frames for {zone_name} (Timestamps: {timestamps})")
    
    img1 = np.clip(frames[-2] * (255.0 / 50.0), 0, 255).astype(np.uint8)
    img2 = np.clip(frames[-1] * (255.0 / 50.0), 0, 255).astype(np.uint8)
    
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 
                                        pyr_scale=0.5, levels=3, winsize=15, 
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    u_raw = flow[:, :, 0]
    v_raw = flow[:, :, 1]
    
    u_filt = gaussian_filter(median_filter(u_raw, size=3), sigma=1)
    v_filt = gaussian_filter(median_filter(v_raw, size=3), sigma=1)
    
    # Angular Coherence Gate Check
    active_mask = frames[-2] > 0
    u_active = u_filt[active_mask]
    v_active = v_filt[active_mask]
    mag = np.sqrt(u_active**2 + v_active**2)
    valid = mag > 0.1
    
    if np.sum(valid) < 100:
        print("[ANGULAR COHERENCE GATE] Insufficient active storm vectors. Setting advection to stationary.")
        u_final = np.zeros_like(u_filt)
        v_final = np.zeros_like(v_filt)
        coherence_passed = False
    else:
        angles = np.arctan2(v_active[valid], u_active[valid])
        c_mean = np.degrees(circmean(angles)) % 360
        c_std = np.degrees(circstd(angles))
        
        print(f"[ANGULAR COHERENCE STATS] Mean Angle: {c_mean:.1f} deg, Angular StdDev: {c_std:.1f} deg")
        
        if c_std > max_angular_std_dev:
            print(f"[WARN] [ANGULAR COHERENCE GATE TRIPPED] StdDev {c_std:.1f} deg > {max_angular_std_dev} deg. Stationary convection detected. Zeroing flow field.")
            u_final = np.zeros_like(u_filt)
            v_final = np.zeros_like(v_filt)
            coherence_passed = False
        else:
            print(f"[OK] [ANGULAR COHERENCE GATE PASSED] Coherent advection locked. Moving at {c_mean:.1f} deg.")
            mean_u = np.mean(u_active[valid])
            mean_v = np.mean(v_active[valid])
            u_final = np.full_like(u_filt, mean_u)
            v_final = np.full_like(v_filt, mean_v)
            coherence_passed = True


            
    
    
    num_lead_times = 4 # for 15, 30, 45, 60
    forecast_grids = [frames[-1]]
    h, w = frames[-1].shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    for step in range(1, num_lead_times + 1):
        # u_final is per 10 mins. For 15 min steps, multiply by 1.5
        map_x = np.float32(grid_x - u_final * (step * 1.5))
        map_y = np.float32(grid_y - v_final * (step * 1.5))
        forecast = cv2.remap(frames[-1], map_x, map_y, 
                             interpolation=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=0)
        forecast_grids.append(forecast)
        
    for loc in locations:
        px, py = latlng_to_tile_pixel(loc["lat"], loc["lng"], zoom, tile_x, tile_y)
        loc["x"] = px
        loc["y"] = py
        
    base_time = datetime.datetime.now(datetime.timezone.utc)
    forecast_cells = {}
    
    for idx, grid in enumerate(forecast_grids):
        lead_min = idx * 10
        forecast_cells[lead_min] = extract_storm_cells(grid, lead_min, base_time)
        
    has_rain = bool(np.sum(active_mask) > MIN_CELL_PIXELS)
    alerts = compute_arrival_countdowns(forecast_cells, locations, coherence_passed=coherence_passed, active_rain_present=has_rain)
    
    # Save status
    status_data = {
        "coherence_passed": coherence_passed,
        "active_rain_present": has_rain,
        "bounds": get_tile_bounds(zoom, tile_x, tile_y)
    }
    with open(f"{zone_id}_status.json", "w") as f:
        json.dump(status_data, f, indent=2)

    # Save grids as .npy
    for idx, grid in enumerate(forecast_grids):
        np.save(os.path.join(cache_dir, f"forecast_grid_{idx*15}.npy"), grid)

    with open(f"{zone_id}_hazard_alerts.json", "w") as f:
        json.dump(alerts, f, indent=2)
        
    with open(f"{zone_id}_forecast_cells.json", "w") as f:
        json.dump(forecast_cells, f, indent=2)
        
    print(f"Successfully generated {zone_name} hazard alerts: {len(alerts)} locations monitored.")
    return alerts, forecast_cells, coherence_passed

if __name__ == "__main__":
    # Run Delhi-NCR
    delhi_locations = [
        {"name": "Delhi IGI Airport (Aviation Hub)", "lat": 28.556, "lng": 77.100},
        {"name": "Gurugram CyberCity (Tech Hub)", "lat": 28.495, "lng": 77.089},
        {"name": "Noida Sector 62 (Industrial Zone)", "lat": 28.628, "lng": 77.365},
        {"name": "Faridabad Agri-Belt (Rural)", "lat": 28.408, "lng": 77.317}
    ]
    run_rainviewer_nowcast_for_zone("delhi_ncr", "Delhi-NCR", "tests/delhi_cache", 7, 91, 53, delhi_locations)
    
    # Run Mumbai
    mumbai_locations = [
        {"name": "CSIA Mumbai", "lat": 19.089, "lng": 72.865},
        {"name": "Navi Mumbai", "lat": 19.033, "lng": 73.029},
        {"name": "Lonavala Ghats", "lat": 18.748, "lng": 73.405}
    ]
    run_rainviewer_nowcast_for_zone("mumbai_ghats", "Mumbai & Western Ghats", "tests/mumbai_cache", 7, 89, 57, mumbai_locations)
    
    # Run Himalayan Belt
    himalaya_locations = [
        {"name": "Dehradun Airport", "lat": 30.189, "lng": 78.180},
        {"name": "Shimla Tourist Hub", "lat": 31.104, "lng": 77.173},
        {"name": "Uttarkashi Agri", "lat": 30.726, "lng": 78.435}
    ]
    run_rainviewer_nowcast_for_zone("himalayan_belt", "Himalayan Belt", "tests/himalayan_cache", 7, 91, 52, himalaya_locations)
    
    # Run Northeast & Meghalaya
    northeast_locations = [
        {"name": "Guwahati Airport", "lat": 26.106, "lng": 91.585},
        {"name": "Shillong City", "lat": 25.578, "lng": 91.893},
        {"name": "Cherrapunji Agri", "lat": 25.270, "lng": 91.732}
    ]
    run_rainviewer_nowcast_for_zone("northeast_bengal", "Northeast & Meghalaya", "tests/northeast_cache", 7, 96, 54, northeast_locations)


