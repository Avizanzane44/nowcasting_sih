import os
import glob
import gzip
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    import pysteps
    from pysteps import io, motion, nowcasts
    from pysteps.utils import conversion, transformation
    HAS_PYSTEPS = False # Force OpenCV baseline
except ImportError:
    HAS_PYSTEPS = False

import sys
sys.path.append('src')
sys.path.append('models/convlstm')
from train import read_pgm_gz
from opencv_optical_flow import extrapolate_opencv

# 1. Load radar sequence (using held-out TEST set: last 15 frames)
data_dir = os.path.abspath("data/raw/radar/pysteps_data")
gz_files = sorted(glob.glob(os.path.join(data_dir, "**", "*20160928*.pgm.gz"), recursive=True))
test_files = gz_files[-15:]

# Always use custom reader for ground truth and OpenCV baseline when PySteps fails
R_rain = []
for gz in test_files[:7]: # Need 3 inputs + 4 future (T+15, T+30, T+45, T+60 -> indices 3, 4, 5, 6)
    img = read_pgm_gz(gz)
    img = img * 50.0  # Scale up for evaluation
    R_rain.append(img)

R_rain = np.array(R_rain)
input_frames = R_rain[:3]
observed_future = R_rain[3:7]

if HAS_PYSTEPS:
    print("Using PySteps for Optical Flow...")
    # Actually run PySteps!
    R_log, metadata = transformation.dB_transform(R_rain, threshold=0.1, zerovalue=-15.0)
    zeroval = -15.0
    R_log = np.nan_to_num(R_log, nan=zeroval, posinf=zeroval, neginf=zeroval)
    oflow = motion.get_method("lucaskanade")
    velocity = oflow(R_log[:3])
    extrapolate = nowcasts.get_method("extrapolation")
    R_forecast_log = extrapolate(R_log[2], velocity, 4)
    R_forecast_log = np.nan_to_num(R_forecast_log, nan=zeroval)
    R_forecast, _ = transformation.dB_transform(R_forecast_log, threshold=-10.0, inverse=True)
    R_forecast = np.nan_to_num(R_forecast, nan=0.0)
else:
    print("PySteps missing. Using OpenCV Farneback Optical Flow Baseline...")
    # The extrapolate function returns `num_lead_times` steps. We need 12 steps (60 mins), 
    # but we only evaluate at indices 2, 5, 8, 11 (T+15, T+30, T+45, T+60).
    R_forecast_full = extrapolate_opencv(input_frames, num_lead_times=12)
    R_forecast = [R_forecast_full[2], R_forecast_full[5], R_forecast_full[8], R_forecast_full[11]]

# ==========================================
# 2. METRIC CALCULATION FUNCTIONS
# ==========================================
def calculate_contingency_scores(forecast, observed, threshold):
    """Calculates Hits, Misses, False Alarms, CSI, POD, FAR, HSS."""
    f_bin = (forecast >= threshold)
    o_bin = (observed >= threshold)
    
    hits = int(np.sum(f_bin & o_bin))
    misses = int(np.sum((~f_bin) & o_bin))
    false_alarms = int(np.sum(f_bin & (~o_bin)))
    correct_negs = int(np.sum((~f_bin) & (~o_bin)))
    
    # POD (Probability of Detection)
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    # FAR (False Alarm Ratio)
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    # CSI (Critical Success Index / Threat Score)
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
    
    # HSS (Heidke Skill Score)
    total = hits + misses + false_alarms + correct_negs
    expected_hits = ((hits + misses) * (hits + false_alarms) + (correct_negs + misses) * (correct_negs + false_alarms)) / total
    hss_denom = total - expected_hits
    hss = (hits + correct_negs - expected_hits) / hss_denom if hss_denom > 0 else 0.0
    
    return {
        "hits": hits, "misses": misses, "false_alarms": false_alarms, "correct_negs": correct_negs,
        "CSI": round(csi, 3), "POD": round(pod, 3), "FAR": round(far, 3), "HSS": round(hss, 3)
    }


# Evaluate metrics across lead times for Storm threshold (15 mm/h)
THRESHOLDS = [5.0, 15.0, 30.0]
results = {"optical_flow": {}, "convlstm": {}}

# Load ConvLSTM Forecast if available
has_convlstm = os.path.exists("convlstm_forecast.npy")
if has_convlstm:
    R_forecast_dl = np.load("convlstm_forecast.npy")

print("--- [1] Meteorological Skill Score Evaluation ---")
for t in THRESHOLDS:
    t_name = f"{int(t)} mm/h Threshold"
    results["optical_flow"][t_name] = []
    if has_convlstm:
        results["convlstm"][t_name] = []
    
    for i in range(min(4, len(observed_future))):
        lead_time = (i + 1) * 15  # 15m, 30m, 45m, 60m
        
        # Optical Flow
        scores_of = calculate_contingency_scores(R_forecast[i], observed_future[i], t)
        scores_of["lead_time_min"] = lead_time
        results["optical_flow"][t_name].append(scores_of)
        print(f"[OptFlow] T+{lead_time}m ({t_name}): CSI = {scores_of['CSI']:.3f} | POD = {scores_of['POD']:.3f}")
        
        # ConvLSTM
        if has_convlstm:
            scores_dl = calculate_contingency_scores(R_forecast_dl[i], observed_future[i], t)
            scores_dl["lead_time_min"] = lead_time
            results["convlstm"][t_name].append(scores_dl)
            print(f"[ConvLSTM] T+{lead_time}m ({t_name}): CSI = {scores_dl['CSI']:.3f} | POD = {scores_dl['POD']:.3f}")

# Save JSON for Frontend Dashboard Modal (Saving only optical flow to top level for backward compat, plus new structures)
export_results = {k: v for k, v in results["optical_flow"].items()}
export_results["models"] = results

# Write to DB
import sys
sys.path.append('src')
import db_storage
db_storage.save_metrics_to_db(export_results)

# Fallback JSON
with open("verification_metrics.json", "w") as f:
    json.dump(export_results, f, indent=2)

print("\nSaved metric scores to 'verification_metrics.json'.")

# ==========================================
# 3. GENERATE SKILL SCORE PLOT
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot CSI across lead times
ax0 = axes[0]
lead_times = [15, 30, 45, 60][:len(observed_future)]
for t in THRESHOLDS:
    t_name = f"{int(t)} mm/h Threshold"
    if True:
        csi_vals_of = [r["CSI"] for r in results["optical_flow"][t_name]]
        ax0.plot(lead_times, csi_vals_of, marker="o", linewidth=2.5, label=f"OF: Rain ≥ {int(t)}")
    if has_convlstm:
        csi_vals_dl = [r["CSI"] for r in results["convlstm"][t_name]]
        ax0.plot(lead_times, csi_vals_dl, marker="s", linestyle="--", linewidth=2.5, label=f"DL: Rain ≥ {int(t)}")

ax0.set_title("Critical Success Index (CSI / Threat Score)", fontsize=12, fontweight="bold")
ax0.set_xlabel("Lead Time (Minutes)", fontsize=10)
ax0.set_ylabel("CSI Score (0.0 to 1.0)", fontsize=10)
ax0.set_ylim(0.0, 1.05)
ax0.grid(True, linestyle="--", alpha=0.5)
ax0.legend()

# Plot POD vs FAR for Severe Storms (15 mm/h)
ax1 = axes[1]
storm_key = "15 mm/h Threshold"
if True:
    pod_vals_of = [r["POD"] for r in results["optical_flow"][storm_key]]
    far_vals_of = [r["FAR"] for r in results["optical_flow"][storm_key]]
    ax1.plot(lead_times, pod_vals_of, "g-o", linewidth=2.5, label="OF POD")
    ax1.plot(lead_times, far_vals_of, "r--o", linewidth=2.5, label="OF FAR")
    
if has_convlstm:
    pod_vals_dl = [r["POD"] for r in results["convlstm"][storm_key]]
    far_vals_dl = [r["FAR"] for r in results["convlstm"][storm_key]]
    ax1.plot(lead_times, pod_vals_dl, "g-s", linestyle="--", linewidth=2.5, label="DL POD")
    ax1.plot(lead_times, far_vals_dl, "r--s", linestyle=":", linewidth=2.5, label="DL FAR")

ax1.set_title("Detection vs False Alarms (15 mm/h Storms)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Lead Time (Minutes)", fontsize=10)
ax1.set_ylabel("Score Ratio", fontsize=10)
ax1.set_ylim(0.0, 1.05)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

plt.tight_layout()
plt.savefig("verification_metrics_plot.png", dpi=200, bbox_inches="tight")
print("[OK] Generated 'verification_metrics_plot.png'.")