# India Convective Nowcasting System - Progress Track

## 1. Pipeline Stages Status (Verified in Code)

### Stages 1-3: Intact and Operational
The core hazard_tracker.py pipeline (decoding RainViewer PNGs to dBZ, Farneback optical flow, extrapolation, and hazard JSON generation) remains fully intact and operational for the Mumbai & Western Ghats zone, as well as the others.

### Stage 4: Multi-Zone Extension
All four zones are now explicitly configured in `src/hazard_tracker.py` with distinct and correct Leaflet tile coordinates:
- **Delhi-NCR:** zoom=7, x=91, y=53
- **Mumbai & Western Ghats:** zoom=7, x=89, y=57
- **Himalayan Belt:** zoom=7, x=91, y=52
- **Northeast & Meghalaya:** zoom=7, x=96, y=54

## 2. Three-State Status Model
The `CLEAR` / `NO_COHERENT_ADVECTION` / `IMMINENT_HAZARD` UI state model is correctly implemented in `src/hazard_tracker.py` and `dashboard.html`, and is consistently applied across all four zones. 

## 3. Verified Fixes from Recent Sessions

- **Stray Marker Artifacts (CONFIRMED FIXED):** The unstyled green/yellow squares (Mumbai) and checkered artifacts (Northeast) were NOT Leaflet markers. They were isolated 1-2 pixel blobs (e.g. marine clutter) physically present in the raw RainViewer `0_0.png` data. They appeared as blocky UI artifacts because of a rigid NumPy alpha mask (`masked_where(grid < 2.0)`). 
- **Bicubic Ringing Artifacts & Mask Cutoff (CONFIRMED FIXED):** 
  - To solve the artificial jagged edges across all storm borders (caused by the NumPy boolean mask), the `masked_where` logic was completely deleted from `api/server.py`. 
  - The interpolation was changed to `bilinear`.
  - The Gaussian blur (`sigma`) was increased to `1.5`. 
  - This allows the colormap (`vmin=0.0`) to handle the alpha gradient naturally, successfully smoothing out the rigid physical rectangular anomalies found in the raw IMD data into organic, soft-edged blobs that match the native RainViewer aesthetic.
- **Mumbai `active_rain_present` Logic Bug (CONFIRMED FIXED):** The tracker previously evaluated `active_rain_present = bool(np.sum(valid) > 0)`, where `valid = mag > 0.1`. Because the heavy median filter crushed the flow magnitude of bubbling storms below 0.1, the tracker wrongly declared the zone completely dry. This was fixed by rewriting it to check reflectivity directly: `has_rain = bool(np.sum(active_mask) > MIN_CELL_PIXELS)`. The tracker now correctly enters the `NO_COHERENT_ADVECTION` state instead of `CLEAR` when rain is present but uncooperative.
- **Silent `L.imageOverlay` Fallback Discovery:** It was discovered that the non-Delhi zones were previously failing to load the backend-generated `.npy` overlays entirely because `hazard_tracker.py` was saving files as `1.npy` instead of `forecast_grid_15.npy`. Because of this failure, the dashboard silently fell back to showing the native `rainViewerLayer` (the heavily smoothed `1_1.png` tiles). This meant earlier "visual verifications" of the non-Delhi overlays were actually just verifying RainViewer's native tiles, not our pipeline's output. This has since been fixed.

## 4. Current Top Open Investigation (UNRESOLVED)

**Systemic Mis-calibration of the Coherence Gate:** 
Currently, the Himalayan Belt, Mumbai, and Northeast zones are almost constantly triggering the `NO_COHERENT_ADVECTION` state (or `CLEAR` when dry). It is an **explicitly flagged open question** whether this reflects genuine uncooperative weather (bubbling convection with no bulk advection) OR if it is caused by the extremely aggressive smoothing in `hazard_tracker.py` (`median_filter size=21` + `gaussian_filter sigma=5`). This smoothing was previously proven to crush real flow signal by 6-10x, but has not yet been conclusively tuned or resolved for the general case.

## 5. Delhi-NCR Legacy Pipeline (Intentional Design Decision)
**Delhi-NCR Pipeline Dual-Architecture:** 
The platform intentionally runs a dual-pipeline architecture. 
- **Delhi-NCR** continues to run the original, extensively-tested legacy pysteps/historical-replay pipeline (
un_nowcast.py), which persists data to SQLite (
owcast_data.db) and serves radar via the /api/layer/ endpoint.
- **Mumbai, Himalayan Belt, and Northeast** run the newer live RainViewer multi-source pipeline (hazard_tracker.py), saving to JSON and .npy cache files. 

Both pipelines are fully operational, real, and functional. They serve different zones by design, and the frontend intelligently routes API calls (such as /api/alerts and /api/layer) based on the selected region to hit the appropriate backend architecture.
