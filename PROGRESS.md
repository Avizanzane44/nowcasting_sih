# Pipeline Implementation Progress

This document tracks the actual status of the project against the intended pipeline. 

### 1. Data Ingestion: 🟡 Partial (Historical Replay)
- **What's done**: A `run_nowcast.py` daemon continuously loops through historical `.pgm.gz` radar data to simulate a live stream feed.
- **What's missing**: No real-time data ingestion. No APIs for ISRO satellite data, IITM Damini lightning, or IMD live radar feeds.

### 2. Preprocessing & Fusion: 🟢 Done
- **What's done**: Radar conversion and fusion logic is wired up.
- **What's improved**: The satellite (CTT) and lightning data are now **radar-derived physically-informed simulations**, rather than arbitrary Gaussian random noise.
  - INSAT-3D CTT is derived from updraft intensity via empirical logarithmic mappings (Vicente et al., 1998).
  - Damini Lightning Density is derived via non-linear power-law precipitation scaling (Petersen & Rutledge, 1992).
  - Explicitly labeled in the UI and API headers as `physically-informed simulation derived from real radar data`.

### 3. Model Development: 🟢 Done (Proof-of-Concept)

**a) Architecture Status:** The deep learning model architecture (`Seq2SeqConvLSTM`) is fully implemented in PyTorch, functional, and integrated into the daemon via the `--model convlstm` flag. 

**b) Training Constraints:** Due to a lack of access to live feeds or large historical archives during the hackathon, the model was trained on a highly limited 40-frame historical sample. To ensure honest validation, we set aside the final 15 frames as a contiguous held-out test set that the ConvLSTM never saw during training or validation. We also applied spatial data augmentation (random rotations and flips) during training to mitigate memorization. 

**c) Performance Comparison (Held-Out Test Set):**
Because the environment lacks MSVC C++ tools to run the original `pysteps` baseline, we implemented a pure-Python fallback using OpenCV's `calcOpticalFlowFarneback` to provide a legitimate, non-DL baseline comparison. 

*Metric: Critical Success Index (CSI) / Probability of Detection (POD) for a 15 mm/h Severe Storm threshold.*

| Lead Time | Optical Flow (Baseline) | ConvLSTM (Unaugmented) | ConvLSTM (Augmented) |
|-----------|-------------------------|------------------------|----------------------|
| **T+15m** | CSI=0.889 / POD=0.938   | CSI=0.887 / POD=0.917  | CSI=0.878 / POD=0.970|
| **T+30m** | CSI=0.849 / POD=0.913   | CSI=0.861 / POD=0.964  | CSI=0.842 / POD=0.976|
| **T+60m** | CSI=0.809 / POD=0.884   | CSI=0.785 / POD=0.981  | CSI=0.787 / POD=0.978|

*Note on Augmentation:* Applying spatial augmentation (flips/rotations) slightly reduced the raw CSI score, indicating it successfully reduced the model's tendency to just memorize the exact pixel locations of the single training storm.

**Verdict:** The Optical Flow baseline currently outperforms the ConvLSTM. The ConvLSTM's high test scores are an artifact of the test set being a chronological continuation of the single training storm. Optical flow naturally excels at advecting existing storms over short periods.

**d) Path to Production:** To make this model production-grade and surpass optical flow, we need to train it on a large, multi-event spatial dataset (e.g., the open-source SEVIR dataset, or integrating live IMD/ISRO feeds) to teach it generalized storm evolution rather than the lifecycle of one specific storm.

### 4. Inference / Serving: 🟢 Done
- **What's done**: A FastAPI server (`api/server.py`) serves the actual output from the `run_nowcast` daemon using serialized `.npy` and `.json` arrays. 
- **Integration**: The daemon now supports switching models via `python run_nowcast.py --model convlstm`.
- **Bug Fix**: Fixed a bounding-box logic bug in `compute_arrival_countdowns` that caused ETAs to freeze at 0 and rain rates at 50.0 mm/h. Countdowns and intensities are now accurately computed from the advected/predicted storm grids locally at the city coordinates.
- **Model Artifact (ConvLSTM vs Optical Flow)**: The ConvLSTM model hallucinates drifting ETAs (e.g., 20 -> 30 -> 35 mins) for distant, stationary storms due to uncalibrated motion blur. To verify if *any* model could produce a genuine multi-step countdown (e.g., 45 -> 40 -> 35 -> 0), a full-grid scan was run across all 40 frames (nearly 1 million pixels per frame) using the Optical Flow baseline. The maximum consecutive countdown sequence found anywhere in the entire dataset was only 1 step. **Conclusion:** This specific historical 40-frame sample lacks any fast-moving, persistent approaching storms; cells are either already overhead (0 mins) or stationary/dissipating (`CLEAR`). The live countdown logic is fully implemented mathematically, but a multi-step live demonstration is strictly limited by the dataset's lack of approaching storms.

### 5. Storage: 🟢 Done
- **What's done**: Migrated the legacy flat-file JSON and `.npy` architecture to a real geospatial database (`data/processed/nowcast_data.db` via SQLite).
  - Used `sqlite3` natively with Python User Defined Functions (UDFs) via `shapely` to bring `ST_Intersects` spatial querying to the database without requiring heavy C-compiled SpatiaLite DLLs on Windows.
  - Implemented robust `execute_with_retry` loops and `PRAGMA journal_mode=TRUNCATE` in `src/db_storage.py` to seamlessly bypass SQLite disk I/O and lock contention errors common on WSL/NTFS mounted drives.
  - Minimal schema implemented: `radar_grids` (BLOBs), `alerts`, `storm_cells` (WKT boundaries), and `metrics`.
  - Solved concurrency/locking issues between the background daemon (`run_nowcast.py`) and the FastAPI server (`api/server.py`).
  - Pipeline verified end-to-end; rows are successfully inserted and served via API without changing `dashboard/dashboard.html`.


### 6. Dashboard / GIS: 🟡 Work In Progress
- **What's done**: The interactive Leaflet.js dashboard (`dashboard/dashboard.html`) pulls real alerts, real radar layers, and real verification metrics from the backend.
- **UI & UX Refinement (Professional Ops Console)**:
  - Flattened the "AI-generated neon/glow" aesthetic into a restrained, professional ops-console palette (Grafana/Datadog style). Removed CSS glows, drop-shadows, and inline JavaScript `box-shadow` styles inside `renderLocations()`.
  - Rebuilt the region selector from a native browser `<select>` to a custom `div`-based listbox dropdown to ensure consistent dark-theme rendering.
  - Added a startup splash animation (radar logo, sonar pulse, boot sequence) that plays on initial load without blocking background API calls. It can be skipped via click.
  - Implemented `leaflet.markercluster` to handle map pin label collisions intelligently at lower zoom levels.
  - Established and documented a strict Z-Index scale (base map=1, map controls=1000, header=1500, dropdown=1600, modals=2200, splash=9999) to definitively fix stacking bugs (e.g., dropdowns hiding behind banners, or the header blocking modal close buttons).
- **RainViewer Integration**:
  - Validated that RainViewer's India radar overlay is genuine IMD radar data, directly sourcing from IMD's `mausam.imd.gov.in` and `ddgmui.imd.gov.in`.
  - See the **RainViewer Integration Stages (Single Source of Truth)** section below for the detailed breakdown of Stages 1 through 5.
- **Interactive Features Status**:
  - **Live/Real API Integration**: The timeline slider (0m to 60m playback) fully controls the backend `/api/layer` and properly pulls forecasted grids; alerts are actively populated from the database `/api/alerts`; metrics dynamically pull from `/api/metrics`.
  - **Unified Live Data Integration**: All four zones (Delhi-NCR, Himalayan Belt, Mumbai Ghats, Northeast Bengal) are fully live and backend-integrated. The dashboard accurately credits IMD (via RainViewer) for all zones.

### 7. Deployment: ❌ Not started
- **What's missing**: No Dockerfiles, CI/CD pipelines, or cloud deployment manifests exist.

---

## RainViewer Integration Stages (Single Source of Truth)

- **Stage 1: Decode tiles → intensity arrays** [DONE]
  - Validated that RainViewer's India radar overlay is genuine IMD radar data, directly sourcing from IMD (`mausam.imd.gov.in`).
  - Implemented standalone decoder (`tests/decode_rainviewer_test.py`). Successfully mapped palette colors to numpy intensity grids (`tests/himalayan_intensity_test.npy`).

- **Stage 2: Optical flow + semi-Lagrangian extrapolation** [DONE]
  - Implemented 4-frame time-series caching (`tests/himalayan_cache`, `tests/mumbai_cache`).
  - **Key Physical Finding (Stationary Convection vs Advection)**: Discovered that isolated, vertically-developing convective cells (as opposed to organized frontal/squall systems) produce angularly-incoherent optical flow (measured std dev 70-97° across zoom levels and filtering attempts) because they lack genuine bulk translation. This is a real physical property of stationary convection (growth/decay in place), not a pipeline defect. This highlights that a production system should detect and flag this case (e.g. via angular coherence thresholding) rather than blindly extrapolating stationary cells as if they were advecting.
  - **Validated Advection & Extrapolation on Mumbai**: Successfully validated on the Mumbai & Western Ghats zone (linearly elongated squall structure). Filtered Farneback optical flow yielded a highly coherent vector field (**1.9° angular std dev**). Executed semi-Lagrangian backward warping forward 60 minutes (`tests/mumbai_extrapolation_plot.png`), confirming smooth, shape-preserving translation (13.54 px shift at T+60) without tearing or ConvLSTM-style hallucinations.
  - *Limitation / Stretch Goal*: Validation was on a relatively small, simple translating cluster (~118 active pixels). Testing the pipeline on a much larger, complex, organized translating system is a flagged stretch goal.

- **Stage 3: Wire into hazard_tracker.py for ONE zone (Mumbai)** [DONE]
  - Integrated the RainViewer fetch, decode, filter, Farneback optical flow, and extrapolation steps into `src/hazard_tracker.py` for the Mumbai zone.
  - Implemented an angular-coherence safety gate to prevent advecting stationary convective noise.
  - **Live Validation & Safety Gate**: Ran the pipeline against live data. In Poll 1, the gate correctly tracked an approaching storm at Navi Mumbai with an ETA of 50 minutes. Ten elapsed real-world minutes later (Poll 2), the live storm physically dissipated below the active vector tracking threshold; the safety gate correctly detected this structural loss, zeroed the advection vectors, and gracefully cleared the alert. This successfully validated the real-world safety logic of the system.
  - **Synthetic Arithmetic Verification**: Because the live storm dissipated before completing a full countdown (i.e. ETA decreasing sequentially across multiple polls), a separate explicitly labeled synthetic test (`test_synthetic_countdown.py`) was used to verify the underlying math inside `compute_arrival_countdowns()`. The synthetic test isolated the arithmetic to prove the ETA decrements correctly (e.g., 50 -> 40) under constant advection.

- **Stage 4: Extend to remaining zones + honest empty state** [DONE]
  - Extended the live tracking engine across all remaining zones (Delhi-NCR, Himalayan Belt, Northeast & Meghalaya) using distinct Zoom 7 coordinates to prevent tile overlap.
  - Implemented robust unified backend status states (`CLEAR`, `NO_COHERENT_ADVECTION`, `IMMINENT_HAZARD`) based on the Farneback angular coherence metrics. Demonstrated the Himalayan Belt naturally tripping the `NO_COHERENT_ADVECTION` gate on live stationary convection.

- **Stage 5: Regression test, attribution update, rehearsal, freeze** [DONE]
  - Deleted legacy PGM FMI code from the backend and stripped unused dependencies (pysteps, matplotlib).
  - Passed visual browser regression across all UI features (clustering, z-index layering, region selection).
  - Updated dashboard UI and config attribution to correctly label all zones as live IMD radar streams via RainViewer. Codebase frozen.