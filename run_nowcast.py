"""
Real-time Replay Daemon for Convective Nowcasting System
- Steps through historical radar frames to simulate a live feed
- Computes Optical Flow motion vectors & advection forecasts
- Extracts storm cells and issues hazard alerts
- Saves .npy and .json for the FastAPI server to serve
"""

import os
import sys
import glob
import gzip
import json
import argparse
import datetime
import time
import numpy as np
from scipy import ndimage
import torch

try:
    import pysteps
    from pysteps import io, motion, nowcasts
    from pysteps.utils import conversion, transformation
    HAS_PYSTEPS = True
except ImportError:
    HAS_PYSTEPS = False

sys.path.append("src")
sys.path.append("src")
import alert_dispatcher

# --- HAZARD THRESHOLDS ---
CLOUDBURST_THRESHOLD_MMH = 50.0  
HAIL_THRESHOLD_MMH = 30.0        
THUNDERSTORM_THRESHOLD_MMH = 15.0 
MIN_CELL_PIXELS = 15 

MONITORED_CITIES = [
    {"name": "Delhi IGI Airport (Aviation Hub)",  "x": 410, "y": 640, "lat": 28.556, "lng": 77.100},
    {"name": "Gurugram CyberCity (Tech Hub)",      "x": 390, "y": 680, "lat": 28.495, "lng": 77.089},
    {"name": "Noida Sector 62 (Industrial Zone)", "x": 480, "y": 620, "lat": 28.628, "lng": 77.365},
    {"name": "Faridabad Agri-Belt (Rural)",        "x": 440, "y": 760, "lat": 28.408, "lng": 77.317},
]

def classify_cell(max_intensity, area_pixels):
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
        if cell_size < MIN_CELL_PIXELS: continue
        
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
            "centroid": {"x": round(float(cx), 1), "y": round(float(cy), 1)},
            "bbox": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
            "area_pixels": int(cell_size),
            "max_intensity_mmh": round(max_rain, 1),
            "mean_intensity_mmh": round(mean_rain, 1),
            "hazard_type": hazard_type,
            "severity": severity
        })
    return cells

def compute_arrival_countdowns(obs_grid, R_forecast, all_forecast_cells, monitored_locations, base_time):
    alerts = []
    
    # Pre-build a dictionary of grids for easy lookup by lead time
    grids_by_time = {0: obs_grid}
    for step_idx in range(len(R_forecast)):
        grids_by_time[(step_idx + 1) * 5] = R_forecast[step_idx]

    for loc in monitored_locations:
        loc_x, loc_y = loc["x"], loc["y"]
        loc_name = loc["name"]
        
        hit_found = False
        
        # Check chronologically from T=0 to T=60
        for lead_time_min in sorted(grids_by_time.keys()):
            grid = grids_by_time[lead_time_min]
            
            # Check a 10x10 km neighborhood around the city (e.g. 5x5 pixels)
            min_y, max_y = max(0, loc_y - 2), min(grid.shape[0], loc_y + 3)
            min_x, max_x = max(0, loc_x - 2), min(grid.shape[1], loc_x + 3)
            
            local_neighborhood = grid[min_y:max_y, min_x:max_x]
            local_max_rain = float(np.max(local_neighborhood))
            
            if local_max_rain >= THUNDERSTORM_THRESHOLD_MMH:
                hazard_type, severity = classify_cell(local_max_rain, 25) # Mock area since it's a neighborhood
                
                alerts.append({
                    "location": loc_name,
                    "status": "IMMINENT_HAZARD",
                    "eta_minutes": int(lead_time_min),
                    "hazard_type": hazard_type,
                    "severity": severity,
                    "expected_peak_rain_mmh": round(local_max_rain, 1),
                    "forecast_arrival": (base_time + datetime.timedelta(minutes=lead_time_min)).isoformat(),
                    "alert_message": f"⚠️ {severity} Early Warning for {loc_name}: {hazard_type} arriving in {lead_time_min} mins!"
                })
                hit_found = True
                break
                
        if not hit_found:
            alerts.append({
                "location": loc_name,
                "status": "CLEAR",
                "eta_minutes": None,
                "hazard_type": "NONE",
                "severity": "NORMAL",
                "alert_message": f"✅ {loc_name}: No convective hazards detected in next 60 minutes."
            })
    return alerts

def main():
    parser = argparse.ArgumentParser(description="Real-Time Historical Replay Daemon")
    parser.add_argument("--model", type=str, choices=["optical_flow", "convlstm"], default="optical_flow", 
                        help="Nowcasting model to use")
    parser.add_argument("--frames", type=int, default=None, 
                        help="Number of frames to process before exiting (default: infinite loop)")
    args = parser.parse_args()

    print("Initializing Real-Time Historical Replay Daemon...")
    print(f"Using Model: {args.model.upper()}")

    data_dir = os.path.abspath("./data/raw/radar/pysteps_data")
    gz_files = sorted(glob.glob(os.path.join(data_dir, "**", "*20160928*.pgm.gz"), recursive=True))
    if not gz_files:
        print("No .pgm.gz files found.")
        return

    sys.path.append('models/convlstm')
    from train import read_pgm_gz

    sys.path.append('src')
    import db_storage
    db_storage.init_db()

    if args.model == "optical_flow":
        sys.path.append('src')
        from opencv_optical_flow import extrapolate_opencv
    else:
        from model import Seq2SeqConvLSTM
        import cv2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        convlstm_model = Seq2SeqConvLSTM(in_channels=1, hidden_channels=8, out_channels=1, kernel_size=(3, 3)).to(device)
        weights_path = "models/convlstm/convlstm_weights.pth"
        if os.path.exists(weights_path):
            convlstm_model.load_state_dict(torch.load(weights_path, map_location=device))
        convlstm_model.eval()

    n_leadtimes = 12
    print(f"Found {len(gz_files)} historical radar frames. Starting replay...")

    # Loop through historical data, simulating a live feed
    i = 2
    frames_processed = 0
    while True:
        if i >= len(gz_files):
            i = 2 # loop back
            
        if args.frames is not None and frames_processed >= args.frames:
            print(f"Completed {args.frames} frames. Exiting for testing.")
            break


        current_files_gz = gz_files[i-2 : i+1]
        selected_files = []
        for gz_path in current_files_gz:
            pgm_path = gz_path[:-3]
            if not os.path.exists(pgm_path):
                with gzip.open(gz_path, "rb") as f_in, open(pgm_path, "wb") as f_out:
                    f_out.write(f_in.read())
            selected_files.append((pgm_path, None))
        
        print(f"[{datetime.datetime.now().isoformat()}] Processing frame index {i} -> {os.path.basename(selected_files[-1][0])}")

        if args.model == "optical_flow":
            # Load frames manually using our PGM reader
            frames = []
            for gz_path in current_files_gz:
                frames.append(read_pgm_gz(gz_path))
            
            # Predict
            R_forecast = extrapolate_opencv(frames, n_leadtimes)
            R_forecast = R_forecast * 50.0  # Scale back to mm/h
            
            obs_grid = frames[-1] * 50.0
            
        else: # ConvLSTM
            input_frames = []
            original_shape = None
            for p in current_files_gz:
                img = read_pgm_gz(p)
                if original_shape is None: original_shape = img.shape
                img = img[::4, ::4]
                input_frames.append(img)
            
            input_seq = np.array(input_frames)
            input_seq = np.expand_dims(input_seq, axis=(0, 1))
            input_seq = np.swapaxes(input_seq, 1, 2)
            
            x = torch.tensor(input_seq, dtype=torch.float32).to(device)
            with torch.no_grad():
                preds = convlstm_model(x, future_seq_len=n_leadtimes)
            
            preds = preds.squeeze().cpu().numpy()
            upsampled_preds = []
            for j in range(preds.shape[0]):
                resized = cv2.resize(preds[j], (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
                resized = resized * 50.0 
                upsampled_preds.append(resized)
            R_forecast = np.array(upsampled_preds)
            obs_grid = read_pgm_gz(current_files_gz[-1]) * 50.0

        # Database writing
        sys.path.append('src')
        import db_storage
        current_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Save grids to DB
        db_storage.save_grid_to_db(current_iso, 0, obs_grid)
        for step_idx in range(n_leadtimes):
            lead_min = (step_idx + 1) * 5
            db_storage.save_grid_to_db(current_iso, lead_min, R_forecast[step_idx])

        # Hazard Tracking
        base_time = datetime.datetime.fromisoformat(current_iso)
        all_forecast_cells = {0: extract_storm_cells(obs_grid, 0, base_time)}
        for step_idx in range(n_leadtimes):
            lead_min = (step_idx + 1) * 5
            all_forecast_cells[lead_min] = extract_storm_cells(R_forecast[step_idx], lead_min, base_time)

        countdowns = compute_arrival_countdowns(obs_grid, R_forecast, all_forecast_cells, MONITORED_CITIES, base_time)

        # Write to Database
        db_storage.save_alerts_to_db(current_iso, countdowns)
        db_storage.save_storm_cells_to_db(current_iso, all_forecast_cells)
        # Save metrics to DB
        try:
            with open("verification_metrics.json", "r") as mf:
                metrics_data = json.load(mf)
            db_storage.save_metrics_to_db(metrics_data)
        except Exception as e:
            print("Could not save metrics:", e)


        # Fallback flat-files for any other legacy scripts
        np.save("radar_obs.npy", obs_grid)
        np.save("radar_forecast.npy", R_forecast)
        with open("live_hazard_alerts.json", "w") as f:
            json.dump(countdowns, f, indent=2)
        with open("forecast_storm_cells.json", "w") as f:
            json.dump(all_forecast_cells, f, indent=2)
        with open("replay_state.json", "w") as f:
            json.dump({
                "data_source": "historical_replay",
                "historical_file": os.path.basename(selected_files[-1][0]),
                "updated_at": current_iso
            }, f, indent=2)

        # Trigger dispatcher
        alert_dispatcher.run_alert_dispatch()

        print(f"[{current_iso}] Database updated for frame {i}. Sleeping for 1 second...")
        time.sleep(1)
        i += 1
        frames_processed += 1

if __name__ == "__main__":
    main()