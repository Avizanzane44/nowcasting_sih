# Current System Architecture

This document describes the *actual* implemented state of the codebase. The system functions as a decoupled background processing daemon and API server working on historical data.

## 1. High-Level Pipeline Overview

The pipeline currently has the following implemented stages:
- **Ingestion**: A historical replay daemon reads sample `.pgm.gz` radar files from local disk continuously.
- **Model / Nowcasting**: Supports two models via configuration flag:
  1. `optical_flow`: Lucas-Kanade extrapolation via `pysteps` (baseline).
  2. `convlstm`: A Seq2Seq PyTorch ConvLSTM deep learning model.
- **Storage/IPC**: Forecasts, hazard alerts, storm cells, and metrics are saved to a SQLite database (`data/processed/nowcast_data.db`) using Python UDFs to simulate SpatiaLite (`ST_Intersects`) for spatial queries. This replaces the legacy JSON/`.npy` flat-file passing, enabling historical queries and resolving concurrency issues between daemon and server.
- **Preprocessing/Fusion**: The server dynamically combines the real radar data with *physically-informed simulated* satellite and lightning data, derived directly from the real radar reflectivity field using empirical meteorological relationships (e.g., Petersen & Rutledge 1992, Vicente et al. 1998).
- **Inference/Serving**: A FastAPI server loads the serialized arrays and serves them as map tile images and JSON APIs.
- **Dashboard**: A frontend UI that visualizes the historical replay and fetches real evaluation metrics.

## 2. Pipeline Stages & Implementation (Dual Architecture)

The system currently runs two parallel, distinct pipelines serving different regions by intentional design.

### Architecture A: Live RainViewer Pipeline (Mumbai, Himalayan Belt, Northeast)
- **Files**: src/hazard_tracker.py
- **Logic**: Acts as a live pipeline polling RainViewer's API every 10 minutes.
  - **Stage 1 (Decode)**: Downloads composite .png tiles for a region and decodes them into quantitative dBZ NumPy grids.
  - **Stage 2 (Optical Flow)**: Computes dense optical flow (Farneback) on the decoded reflectivity grids to extract motion vectors.
  - **Stage 3 (Extrapolate)**: Advects the current grid forward in time using OpenCV 
emap to generate a 60-min forecast.
  - **Stage 4 (Hazard Generation)**: Employs a three-state model (CLEAR / NO_COHERENT_ADVECTION / IMMINENT_HAZARD). Saves outputs as .npy cache files and JSON alerts.

### Architecture B: Legacy Historical Replay Pipeline (Delhi-NCR)
- **Files**: 
un_nowcast.py
- **Logic**: The original, extensively-tested baseline pipeline. Acts as a continuous background daemon reading historical .pgm.gz radar files. Uses pysteps for Lucas-Kanade optical flow extrapolation.
- **Storage/IPC**: Forecasts, hazard alerts, storm cells, and metrics are saved to a SQLite database (data/processed/nowcast_data.db) using Python UDFs to simulate SpatiaLite (ST_Intersects). 

### Inference / Serving
- **Files**: pi/server.py
- **Logic**: A FastAPI server that exposes /api/layer, /api/layer-rainviewer, /api/alerts, and /api/status. The server intelligently routes API requests to the appropriate backend architecture based on the region. It dynamically applies Gaussian smoothing (sigma=1.5) and bilinear interpolation to the raw RainViewer grids to cleanly map alpha gradients and hide raw data anomalies.

### Dashboard
- **Files**: dashboard/dashboard.html
- **Logic**: A Leaflet.js based web UI that seamlessly integrates both pipelines. Delhi-NCR hits the legacy API paths (/api/layer), while the other three zones dynamically hit the live paths (/api/layer-rainviewer). Both architectures flow into the same UI components.

### Alert Dispatch
- **Files**: `src/alert_dispatcher.py`
- **Logic**: Reads `live_hazard_alerts.json` and formats them as strings (Aviation SIGMET, DDMA Civil, Farmer SMS). Can ping a Telegram webhook.

## 3. Data Flow Diagram

`mermaid
flowchart TD
    subgraph Architecture_B_Legacy
        A[data/raw/radar/pysteps_data / .pgm.gz] --> B(run_nowcast.py Daemon)
        B --> C[(nowcast_data.db SQLite)]
        B --> D[.npy grids]
    end

    subgraph Architecture_A_Live
        R1[RainViewer API .png] --> H(src/hazard_tracker.py)
        H --> H1[Decode dBZ] --> H2[Farneback Flow] --> H3[Extrapolate]
        H3 --> J1[.npy grids]
        H3 --> J2[.json status/alerts]
    end

    subgraph API_Serving
        C --> E(api/server.py)
        D --> E
        J1 --> E
        J2 --> E
    end

    subgraph Web_Dashboard
        E --> I[dashboard.html UI]
        I -.->|Delhi-NCR| Architecture_B_Legacy
        I -.->|Other Zones| Architecture_A_Live
    end
`

## 4. Actual External Dependencies

The `requirements.txt` file is accurate and uses the following:

- `pysteps` - Core library for reading radar data, optical flow, and advection extrapolation.
- `numpy`, `scipy` - Array manipulation, mathematical processing, and saving grid arrays.
- `matplotlib` - Generating image layers (colormaps and byte buffering) for the Leaflet maps.
- `fastapi`, `uvicorn` - Web server and API routing.
- `opencv-python` (`cv2`) - Computer vision ops.
