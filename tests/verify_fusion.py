import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Load the fusion functions
from data_fusion import generate_insat_satellite_ctt, generate_lightning_density_grid

def verify():
    # Load actual radar observation
    if not os.path.exists("radar_obs.npy"):
        print("radar_obs.npy not found. Run the pipeline first.")
        return
        
    radar_rain = np.load("radar_obs.npy")
    
    # Generate physically-informed simulations
    sat_ctt = generate_insat_satellite_ctt(radar_rain)
    lightning = generate_lightning_density_grid(radar_rain)
    
    # Flatten arrays for scatter plotting
    radar_flat = radar_rain.flatten()
    ctt_flat = sat_ctt.flatten()
    lightning_flat = lightning.flatten()
    
    # Filter out empty space (0 mm/h) for cleaner plotting, keeping only active weather pixels
    mask = radar_flat > 0.5
    radar_active = radar_flat[mask]
    ctt_active = ctt_flat[mask]
    lightning_active = lightning_flat[mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Radar vs CTT
    ax1.scatter(radar_active, ctt_active, alpha=0.1, color='blue', s=2)
    ax1.set_title("Correlation: Radar Rain Rate vs Simulated CTT\n(Physically-informed: Vicente et al., 1998)")
    ax1.set_xlabel("Real Radar Rain Rate (mm/h)")
    ax1.set_ylabel("Simulated Cloud Top Temperature (°C)")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Radar vs Lightning
    ax2.scatter(radar_active, lightning_active, alpha=0.1, color='orange', s=2)
    ax2.set_title("Correlation: Radar Rain Rate vs Simulated Lightning\n(Physically-informed: Petersen & Rutledge, 1992)")
    ax2.set_xlabel("Real Radar Rain Rate (mm/h)")
    ax2.set_ylabel("Simulated Lightning Density (flashes/km²/min)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("fusion_correlation_check.png", dpi=150)
    print("Saved 'fusion_correlation_check.png'")

if __name__ == "__main__":
    verify()
