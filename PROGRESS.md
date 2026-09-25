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

### 6. Dashboard / GIS: 🟢 Done
- **What's done**: The interactive Leaflet.js dashboard (`dashboard/dashboard.html`) pulls real alerts, real radar layers, and real verification metrics from the backend.
- **Interactive Features Status**:
  - **Live/Real API Integration**: The timeline slider (0m to 60m playback) fully controls the backend `/api/layer` and properly pulls forecasted grids; alerts are actively populated from the database `/api/alerts`; metrics dynamically pull from `/api/metrics`.
  - **Demo/Mock Data**: Several selectable zones in the top dropdown (Himalayan Belt, Mumbai Ghats, Northeast Bengal) are **hardcoded frontend presets** with no live tracking backend support. They have been relabeled as `[Demo Preset - No Live Data]` in the UI to prevent deceiving end-users. The only fully live, backend-integrated zone is Delhi-NCR.

### 7. Deployment: ❌ Not started
- **What's missing**: No Dockerfiles, CI/CD pipelines, or cloud deployment manifests exist.

---

## Short-Term Next Steps (Top 3)
1. **Source SEVIR Dataset**: Deep learning requires a massive spatial dataset. We must acquire SEVIR or similar for robust ConvLSTM training.
2. **Source Real Live Feeds**: Replace the historical replay with live IMD/ISRO data ingestion APIs.
3. **Containerize**: Create a Dockerfile to manage the `pysteps` C-dependencies and Python environment securely.