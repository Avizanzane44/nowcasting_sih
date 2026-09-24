import glob
import numpy as np
import cv2
import gzip
import os

import sys
sys.path.append('models/convlstm')
from train import read_pgm_gz

def extrapolate_opencv(frames, num_lead_times):
    img1 = np.clip(frames[-2] * (255.0 / 50.0), 0, 255).astype(np.uint8)
    img2 = np.clip(frames[-1] * (255.0 / 50.0), 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 
                                        pyr_scale=0.5, levels=3, winsize=15, 
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    h, w = frames[-1].shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    forecasts = []
    for step in range(1, num_lead_times + 1):
        map_x = np.float32(grid_x - flow[:, :, 0] * step)
        map_y = np.float32(grid_y - flow[:, :, 1] * step)
        forecast = cv2.remap(frames[-1], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        forecasts.append(forecast)
    return np.array(forecasts)

data_dir = os.path.abspath("./pysteps_data")
gz_files = sorted(glob.glob(os.path.join(data_dir, "**", "*20160928*.pgm.gz"), recursive=True))

MONITORED_CITIES = [
    {"name": "Delhi",  "x": 410, "y": 640},
    {"name": "Gurugram",      "x": 390, "y": 680},
    {"name": "Noida", "x": 480, "y": 620},
    {"name": "Faridabad",        "x": 440, "y": 760},
]

print("Scanning 40 frames...")

frames = []
for idx, gz_path in enumerate(gz_files[:40]):
    frames.append(read_pgm_gz(gz_path))
    if len(frames) > 2:
        frames.pop(0)
    
    if len(frames) == 2:
        obs_grid = frames[-1] * 50.0
        R_forecast = extrapolate_opencv(frames, 12) * 50.0
        
        print(f"Frame {idx}:")
        for city in MONITORED_CITIES:
            x, y = city["x"], city["y"]
            min_y, max_y = max(0, y - 2), min(obs_grid.shape[0], y + 3)
            min_x, max_x = max(0, x - 2), min(obs_grid.shape[1], x + 3)
            
            # Check obs_grid
            if np.max(obs_grid[min_y:max_y, min_x:max_x]) >= 15.0:
                print(f"  {city['name']}: 0 mins")
                continue
                
            found = False
            for step_idx in range(12):
                if np.max(R_forecast[step_idx, min_y:max_y, min_x:max_x]) >= 15.0:
                    print(f"  {city['name']}: {(step_idx+1)*5} mins")
                    found = True
                    break
            
            if not found:
                print(f"  {city['name']}: CLEAR")
