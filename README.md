# India Convective Nowcasting Multi-Region API

This project is a Convective Scale Nowcasting System that predicts severe weather hazards (such as cloudbursts, severe hail, and thunderstorms) at a 5-60 minute lead time using radar data. 

## SIH Problem Statement
This solution is designed to address the problem of short-term, localized severe weather forecasting. Current NWP (Numerical Weather Prediction) models operate on a timescale of hours to days, which is often too slow to accurately warn against rapidly developing convective storms (like cloudbursts). This system uses real-time radar data to advect (push forward) existing storms and predict their paths, providing actionable, hyper-local warnings (0-60 mins) to aviation hubs, disaster management authorities, and farmers.

---

## Team Onboarding & Quick Start

Welcome to the team! Before diving in, please read this quick-start guide covering environment setup, running the pipeline, and known project constraints.

### 1. Environment Setup (Windows vs. WSL)
The core library, `pysteps`, fails to build on Windows without MSVC C++ Build Tools. 
* **Plain Windows:** You can run the project natively on Windows *without* `pysteps`. The `run_nowcast.py` daemon is programmed to safely fall back to OpenCV (`calcOpticalFlowFarneback`) for optical flow tracking.
* **WSL (Linux):** If you want to use the full `pysteps` feature set (like stochastic ensembles) or run the `ConvLSTM` deep learning path, you must use WSL. 
  ```bash
  # WSL Quick Setup
  cd /mnt/c/Users/avish/OneDrive/Desktop/CNN_project
  python3.11 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

### 2. Radar Dataset (Included)
The standard PySteps example dataset is over 500MB, which exceeds standard GitHub push limits. To keep this repository lightweight and hackathon-friendly, **a curated 40-frame, 9.7MB subset of the 2016 FMI radar dataset is included directly in this repository** at `data/raw/radar/pysteps_data/`. 
Any new team member who clones the repo will have the data immediately available to run the daemon out-of-the-box, without needing to download external data or configure Git LFS.

### 3. Running the Pipeline End-to-End
The system is decoupled into a backend processing daemon and a frontend serving API.
* **Terminal 1 (Daemon):** 
  ```bash
  python run_nowcast.py
  ```
  *Note:* The daemon runs continuously by default, indefinitely looping the historical 40-frame dataset to simulate a live stream. Use `python run_nowcast.py --frames N` for quick finite testing.
* **Terminal 2 (Server):** 
  ```bash
  python -m uvicorn api.server:app --reload
  ```
* **Dashboard:** Simply double-click `dashboard/dashboard.html` in your file explorer to open it directly in a browser. It is hardcoded to fetch live data from the local `127.0.0.1:8000` API.

### 4. Folder Structure Quick Reference
Post-reorganization, the codebase is structured as follows:
* `api/` - `server.py` (FastAPI serving layer)
* `src/` - Core logic (`data_fusion.py`, `alert_dispatcher.py`, `hazard_tracker.py`, `db_storage.py`)
* `data/` - Contains `raw/radar/pysteps_data/` (historical `.pgm.gz`) and `processed/nowcast_data.db` (SQLite Database)
* `dashboard/` - Frontend Leaflet.js UI
* `models/convlstm/` - PyTorch model definitions and `.pth` weights
* `tests/` - Standalone evaluation and testing scripts

### 5. Known Limitations & Project Status
Be aware of the following hackathon constraints before modifying code:
* **Simulated Data:** Satellite (CTT) and Lightning (Density) layers are *not* real-time APIs; they are physically-informed simulations derived mathematically from the real radar data.
* **ConvLSTM Proof-of-Concept:** The deep learning model is a PoC trained on only 40 frames. It suffers from ETA drift and hallucination on distant storms. The Optical Flow fallback is currently more robust.
* **Live Multi-Zone Tracking:** All four geographic zones (Delhi-NCR, Mumbai & Western Ghats, Himalayan Belt, Northeast & Meghalaya) are actively tracked using live IMD radar streams via RainViewer. The dashboard correctly discriminates between advecting hazards, stationary convection (`NO_COHERENT_ADVECTION`), and empty skies.
* **Countdown Demonstration:** The live countdown logic is fully implemented, but a multi-step continuous live countdown (e.g., 45 -> 40 -> 35 mins) has never been demonstrated on a genuinely approaching storm because this specific historical dataset lacks one.

**Important:** Before making structural changes or questioning the data flow, please read `ARCHITECTURE.md` and `PROGRESS.md` in full. They contain the complete, honest history of what is real vs. simulated vs. PoC in this pipeline.
