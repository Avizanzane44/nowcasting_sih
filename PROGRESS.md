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

### Coherence Gate Validation - Sept 29, 2026

1. The coherence gate was tested against live, current weather data across all four zones and found to be behaving correctly. Raw Farneback flow on today\'s storms shows genuine, severe angular incoherence (73-258 degree standard deviation) BEFORE any filtering. This means the persistent NO_COHERENT_ADVECTION status accurately reflects real disorganized convection, not a filter artifact.
2. The median/gaussian smoothing filter was checked and does NOT suppress genuinely coherent motion when it exists. Raw vs. filtered moving-pixel counts from today\'s weather are nearly identical (Mumbai: 763 raw vs 764 filtered; Himalayan: 416 raw vs 402 filtered; Northeast: 32 raw vs 32 filtered).
3. The original Stage 2 Mumbai squall-line test data no longer exists in cache to re-verify directly. However, the AP/Telangana finding from that earlier test (where the filter seemingly crushed signal 6-10x) is now understood to be an edge case specific to exceptionally weak/sparse storm data, not a general pipeline problem. Today\'s stronger-storm data serves as concrete counter-evidence.
4. A country-wide scan successfully found a naturally coherent live storm today (Tile 97, 56 off the coast) passing with an extremely tight angular standard deviation of 5.9°. This confirms that the system is fully capable of passing coherent advection when it exists in reality. The system is intentionally declining to fabricate a false countdown for disorganized weather in the monitored zones, which is the exact safety behavior it was designed to have.


**Update / Correction (Step 2.5 Validation):**
While the angular coherence gate correctly passes organized storms, subsequent quantification revealed that the spatial smoothing filters (median_filter(size=21) + gaussian_filter(sigma=5)) are drastically under-computing the storm's displacement magnitude. For the coherent storm on Tile 97, 56, the raw Farneback mean magnitude was 3.10 px/frame (~21.2 km/h), but the filtered magnitude was crushed to 0.37 px/frame (~2.5 km/h) — an 8.4x reduction. This confirms the earlier suspicion: large kernel sizes on sparse/patchy rain fields pull in too many zero-velocity background pixels, artificially dragging the legitimate storm velocity down to a near-standstill.
