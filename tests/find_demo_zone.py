import glob
import numpy as np
import cv2
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

print("Loading frames...")
frames_raw = []
for f in gz_files[:40]:
    frames_raw.append(read_pgm_gz(f))

# Compute ETA map for each frame (start at index 1 because we need 2 frames for optical flow)
eta_maps = []
for i in range(1, len(frames_raw)):
    f_input = frames_raw[i-1:i+1]
    obs = f_input[-1] * 50.0
    R_forecast = extrapolate_opencv(f_input, 12) * 50.0
    
    # eta_map initializes to 999
    eta_map = np.full(obs.shape, 999, dtype=np.int32)
    
    # Check backwards from 60 mins down to 5 mins
    for step_idx in range(11, -1, -1):
        lead_min = (step_idx + 1) * 5
        mask = R_forecast[step_idx] >= 15.0
        eta_map[mask] = lead_min
        
    # Lead time 0
    mask_0 = obs >= 15.0
    eta_map[mask_0] = 0
    eta_maps.append(eta_map)

print(f"Computed {len(eta_maps)} ETA maps.")
eta_maps = np.stack(eta_maps, axis=0) # shape: (39, 1226, 760)

# We want to find a sequence like [..., 25, 20, 15, 10, 5, 0, ...]
# Let's count consecutive decreases in ETA for each pixel.
max_decreases = np.zeros((eta_maps.shape[1], eta_maps.shape[2]), dtype=np.int32)

for t in range(1, len(eta_maps)):
    prev_eta = eta_maps[t-1]
    curr_eta = eta_maps[t]
    
    # A valid decrease is when prev_eta != 999 and curr_eta < prev_eta and curr_eta != 999
    # Actually, a perfect countdown is prev_eta - 5 == curr_eta
    valid_decrease = (prev_eta != 999) & (curr_eta != 999) & (curr_eta == prev_eta - 5)
    
    # Accumulate streaks
    max_decreases = np.where(valid_decrease, max_decreases + 1, max_decreases)

# Find coordinates with the longest countdown streak
best_y, best_x = np.unravel_index(np.argmax(max_decreases), max_decreases.shape)
best_streak = max_decreases[best_y, best_x]

print(f"Best streak length: {best_streak}")
if best_streak >= 3:
    print(f"Found a good test zone at (y={best_y}, x={best_x}) with {best_streak} consecutive countdown steps.")
    print("ETA sequence for this pixel over time:")
    for t in range(len(eta_maps)):
        eta = eta_maps[t, best_y, best_x]
        print(f"Frame {t+2}: {'CLEAR' if eta==999 else eta}")
else:
    print("No convincing countdown sequence found anywhere in the entire grid.")
