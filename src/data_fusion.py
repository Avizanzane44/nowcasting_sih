"""
Multi-Source Data Fusion Engine (Convective Scale Nowcasting)
- Ingests & aligns Doppler Weather Radar (DWR)
- Simulates & processes INSAT-3D/3DR Satellite Cloud Top Temperature (CTT)
- Generates 2D Gaussian Lightning Flash Density Grids (IITM/Damini style)
- Computes Unified Convective Hazard Index (0-100%) and Early Initiation Flags
"""

import os
import io
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def generate_insat_satellite_ctt(radar_rain_grid):
    """
    Simulates INSAT-3D TIR-1 Cloud Top Temperature (CTT in Celsius) using a physically-informed
    derivation from real radar data, rather than independent randomness.
    
    Meteorological Relationship:
    Based on empirical inverse log relationships between radar-inferred rainfall rate 
    and IR cloud top temperatures via updraft strength (Vicente et al., 1998, 'The Auto-Estimator').
    Stronger updrafts (higher rain rates) push the cloud top higher into the troposphere,
    reaching colder temperatures (down to -75°C).
    """
    # Base ground temperature (+20°C)
    ctt_grid = np.full_like(radar_rain_grid, 20.0)
    
    # 1. Broad cloud anvil (smooth, low-intensity spread)
    anvil_mask = gaussian_filter((radar_rain_grid > 0.5).astype(float), sigma=10)
    ctt_grid -= anvil_mask * 35.0  # Cool down to ~ -15°C for general anvil
    
    # 2. Convective cores (exponential cooling linked to rain intensity)
    # CTT drops logarithmically with increasing rain rate: CTT = 15 - 45 * log10(RR + 1)
    core_cooling = 45.0 * np.log10(radar_rain_grid + 1.0)
    
    # Apply a slight spatial smoothing to account for sensor resolution differences
    core_cooling = gaussian_filter(core_cooling, sigma=2)
    ctt_grid -= core_cooling
    
    # 3. Add small high-frequency structural noise for realism (dependent on rain presence)
    np.random.seed(42) # Fixed seed for deterministic noise texture
    structural_noise = np.random.normal(0, 2.0, radar_rain_grid.shape)
    ctt_grid += structural_noise * (radar_rain_grid > 0.1)

    return np.clip(ctt_grid, -80.0, 30.0)


def generate_lightning_density_grid(radar_rain_grid):
    """
    Simulates lightning strike density (flashes/km^2) using a physically-informed 
    power-law derivation from real radar data.
    
    Meteorological Relationship:
    Based on the robust power-law correlation between convective precipitation intensity 
    and lightning flash rate (Petersen and Rutledge, 1992; Boccippio et al., 2001).
    Flash Rate Density ~ a * (RainRate)^b. Lightning rarely occurs below 10-15 mm/h.
    """
    # Threshold for electrification (typically 35-40 dBZ / ~10 mm/h rain rate)
    electrification_threshold = 10.0
    active_convection = np.maximum(radar_rain_grid - electrification_threshold, 0.0)
    
    # Power-law relationship: Density = a * (RR_excess)^1.2
    # This ensures lightning scales non-linearly with severe cores
    raw_flash_density = 0.05 * (active_convection ** 1.2)
    
    # Add minor high-frequency variation to simulate discrete strikes
    np.random.seed(101)
    strike_variation = np.random.uniform(0.8, 1.2, radar_rain_grid.shape)
    raw_flash_density *= strike_variation
    
    # Lightning channels span horizontally (3-10km), so apply Gaussian smoothing
    lightning_density = gaussian_filter(raw_flash_density, sigma=4)
    
    return np.clip(lightning_density, 0.0, 20.0)


def compute_fused_convective_hazard_index(radar_rain, sat_ctt, lightning_density):
    """
    Multi-Source Fusion Algorithm:
    Normalizes and weights:
    - Radar Precipitation Severity (40%)
    - INSAT-3D Cold Cloud Top Temperature (35%)
    - Lightning Strike Density (25%)
    
    Returns: Fused Hazard Index (0.0 to 100.0%)
    """
    # 1. Radar Score: 0 to 1.0 (50 mm/h = 1.0)
    s_radar = np.clip(radar_rain / 50.0, 0.0, 1.0)
    
    # 2. Satellite CTT Score: 0 to 1.0 (0°C = 0.0, -60°C = 1.0)
    s_sat = np.clip((-sat_ctt) / 60.0, 0.0, 1.0)
    
    # 3. Lightning Score: 0 to 1.0 (5.0 density = 1.0)
    s_lightning = np.clip(lightning_density / 5.0, 0.0, 1.0)
    
    # Weighted Data Fusion
    fused_index = (0.40 * s_radar + 0.35 * s_sat + 0.25 * s_lightning) * 100.0
    
    # Early Convective Initiation Flag: Cold tops (< -40°C) with low current surface rain (< 5 mm/h)
    early_ci_mask = (sat_ctt < -40.0) & (radar_rain < 5.0)
    
    return fused_index, early_ci_mask


if __name__ == "__main__":
    import hazard_tracker
    
    # Execute fusion on current radar frame
    current_rain = hazard_tracker.R_rain[-1]
    sat_ctt = generate_insat_satellite_ctt(current_rain)
    lightning_density = generate_lightning_density_grid(current_rain)
    fused_index, early_ci_mask = compute_fused_convective_hazard_index(current_rain, sat_ctt, lightning_density)
    
    print("--- Multi-Source Data Fusion Summary ---")
    print(f"1. Radar Peak Rain Rate:       {np.max(current_rain):.1f} mm/h")
    print(f"2. INSAT-3D Min Cloud Top Temp:{np.min(sat_ctt):.1f} °C (Deep Convective Overshoot)")
    print(f"3. Max Lightning Density:      {np.max(lightning_density):.2f} strikes/km²")
    print(f"4. Peak Fused Hazard Index:    {np.max(fused_index):.1f} %")
    print(f"5. Early Convective Zones:     {np.sum(early_ci_mask)} grid cells flagged for storm initiation!")
    
    # Generate Multi-Panel Fusion Comparison Graphic
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Doppler Radar
    im0 = axes[0, 0].imshow(current_rain, cmap="turbo", vmin=0.5, vmax=50)
    axes[0, 0].set_title("1. DWR Doppler Radar (Rain Rate mm/h)", fontsize=11, weight="bold")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 2. INSAT-3D CTT
    im1 = axes[0, 1].imshow(sat_ctt, cmap="coolwarm_r", vmin=-70, vmax=20)
    axes[0, 1].set_title("2. INSAT-3D Thermal IR (Cloud Top Temp °C)", fontsize=11, weight="bold")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Lightning Density
    im2 = axes[1, 0].imshow(lightning_density, cmap="magma", vmin=0, vmax=5)
    axes[1, 0].set_title("3. Ground Lightning Strike Density (LLN)", fontsize=11, weight="bold")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 4. Fused Convective Hazard Index
    im3 = axes[1, 1].imshow(fused_index, cmap="inferno", vmin=10, vmax=100)
    axes[1, 1].set_title("4. FUSED Multi-Source Severe Hazard Index (0-100%)", fontsize=11, weight="bold", color="darkred")
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("multi_source_fusion_preview.png", dpi=200, bbox_inches="tight")
    print("\n🎉 Saved 'multi_source_fusion_preview.png'.")