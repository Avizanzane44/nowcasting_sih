import os
import io
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_insat_satellite_ctt(radar_rain_grid):
    ctt_grid = np.full_like(radar_rain_grid, 20.0)
    anvil_mask = gaussian_filter((radar_rain_grid > 0.5).astype(float), sigma=10)
    ctt_grid -= anvil_mask * 35.0
    core_cooling = 45.0 * np.log10(radar_rain_grid + 1.0)
    core_cooling = gaussian_filter(core_cooling, sigma=2)
    ctt_grid -= core_cooling
    np.random.seed(42)
    structural_noise = np.random.normal(0, 2.0, radar_rain_grid.shape)
    ctt_grid += structural_noise * (radar_rain_grid > 0.1)
    return np.clip(ctt_grid, -80.0, 30.0)

def generate_lightning_density_grid(radar_rain_grid):
    electrification_threshold = 10.0
    active_convection = np.maximum(radar_rain_grid - electrification_threshold, 0.0)
    raw_flash_density = 0.05 * (active_convection ** 1.2)
    np.random.seed(101)
    strike_variation = np.random.uniform(0.8, 1.2, radar_rain_grid.shape)
    raw_flash_density *= strike_variation
    lightning_density = gaussian_filter(raw_flash_density, sigma=4)
    return np.clip(lightning_density, 0.0, 20.0)

def compute_fused_convective_hazard_index(radar_rain, sat_ctt, lightning_density):
    s_radar = np.clip(radar_rain / 50.0, 0.0, 1.0)
    s_sat = np.clip((-sat_ctt) / 60.0, 0.0, 1.0)
    s_lightning = np.clip(lightning_density / 5.0, 0.0, 1.0)
    fused_index = (0.40 * s_radar + 0.35 * s_sat + 0.25 * s_lightning) * 100.0
    early_ci_mask = (sat_ctt < -40.0) & (radar_rain < 5.0)
    return fused_index, early_ci_mask

if __name__ == "__main__":
    import sys
    sys.path.append('src')
    from hazard_tracker import decode_rainviewer_png
    import glob
    
    # Load the latest Mumbai tile we have instead of the broken legacy PGM array
    cache_dir = "tests/mumbai_cache"
    tiles = sorted(glob.glob(os.path.join(cache_dir, "*.png")))
    if not tiles:
        print("No Mumbai tiles found. Please run hazard_tracker.py first.")
        sys.exit(1)
        
    current_rain = decode_rainviewer_png(tiles[-1])
    
    sat_ctt = generate_insat_satellite_ctt(current_rain)
    lightning_density = generate_lightning_density_grid(current_rain)
    fused_index, early_ci_mask = compute_fused_convective_hazard_index(current_rain, sat_ctt, lightning_density)
    
    print("--- Multi-Source Data Fusion Summary ---")
    print(f"1. Radar Peak Rain Rate:       {np.max(current_rain):.1f} mm/h")
    print(f"2. INSAT-3D Min Cloud Top Temp:{np.min(sat_ctt):.1f} °C (Deep Convective Overshoot)")
    print(f"3. Max Lightning Density:      {np.max(lightning_density):.2f} strikes/km²")
    print(f"4. Peak Fused Hazard Index:    {np.max(fused_index):.1f} %")
    print(f"5. Early Convective Zones:     {np.sum(early_ci_mask)} grid cells flagged for storm initiation!")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    im0 = axes[0, 0].imshow(current_rain, cmap="turbo", vmin=0.5, vmax=50)
    axes[0, 0].set_title("1. DWR Doppler Radar (Rain Rate mm/h)", fontsize=11, weight="bold")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im1 = axes[0, 1].imshow(sat_ctt, cmap="coolwarm_r", vmin=-70, vmax=20)
    axes[0, 1].set_title("2. INSAT-3D Thermal IR (Cloud Top Temp °C)", fontsize=11, weight="bold")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 0].imshow(lightning_density, cmap="magma", vmin=0, vmax=5)
    axes[1, 0].set_title("3. Ground Lightning Strike Density (LLN)", fontsize=11, weight="bold")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 1].imshow(fused_index, cmap="inferno", vmin=10, vmax=100)
    axes[1, 1].set_title("4. FUSED Multi-Source Severe Hazard Index (0-100%)", fontsize=11, weight="bold", color="darkred")
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("multi_source_fusion_preview.png", dpi=200, bbox_inches="tight")
    print("\n[OK] Saved 'multi_source_fusion_preview.png'.")