# 🌡️ Heatwave AI — Backend

A physics-informed deep learning backend for heatwave prediction over Thailand, powered by **ConvLSTM** and **ERA5 reanalysis data**.

## Overview

This project trains a spatiotemporal deep learning model to forecast heatwave events across Thailand using meteorological variables from the [ERA5 reanalysis dataset](https://cds.climate.copernicus.eu/). It serves predictions through a Flask REST API consumed by the [Heatwave Webapp](../Heatwave-Webapp/my-app/) frontend.

## Architecture

```
ERA5 Data (NetCDF)  →  Data Loader  →  ConvLSTM Model  →  Flask API  →  Frontend App
```

### Key Components

| File | Description |
|---|---|
| `download_era5.py` | Downloads ERA5 surface & upper-air data (2000–2023) for the Thailand region via the CDS API |
| `data_loader.py` | Loads, crops, merges, and normalizes ERA5 `.nc` files into model-ready tensors `(Time, Channels, H, W)` |
| `heatwave_model.py` | Defines the **Encoder-Decoder ConvLSTM** model and the **Physics-Informed Loss** function |
| `Train_Ai.py` | Training pipeline with train/val split, versioned checkpoint saving |
| `api_server.py` | Flask REST API serving predictions, forecasts, and GeoJSON map data |
| `main.py` | Entry point / orchestration script |

### Model Details

- **Architecture**: Encoder-Decoder ConvLSTM (2 layers, 16 hidden channels)
- **Input**: 5-day sequences of 3 channels — **Z500** (Geopotential), **T2m** (2m Temperature), **Soil Moisture**
- **Output**: 2-day ahead temperature forecast grids
- **Loss**: MSE + Physics constraint (Adiabatic Compression/Expansion penalty)
- **Region**: Thailand (Lat 5°–21°N, Lon 97°–106°E)

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, not required)
- [CDS API key](https://cds.climate.copernicus.eu/) for data download

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 1. Download ERA5 Data

> **Note:** Requires a CDS API key configured in `~/.cdsapirc`.

```bash
python download_era5.py
```

Data will be saved to the `era5_data/` directory as NetCDF files.

### 2. Train the Model

```bash
python Train_Ai.py
```

Checkpoints are saved to `models/` with automatic versioning (e.g., `heatwave_model_checkpoint_v1.pth`).

### 3. Start the API Server

```bash
python api_server.py
```

The server starts on `http://localhost:5000`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check for frontend connectivity |
| `GET` | `/predict/summary` | Dashboard summary statistics |
| `GET` | `/predict/map` | GeoJSON heatmap features for the map layer |
| `GET` | `/forecast/summary` | Multi-day forecast data |

## Project Structure

```
NewHeart/
├── api_server.py          # Flask REST API
├── data_loader.py         # ERA5 data loading & preprocessing
├── download_era5.py       # ERA5 data downloader (CDS API)
├── heatwave_model.py      # ConvLSTM model & physics loss
├── Train_Ai.py            # Training pipeline
├── main.py                # Entry point
├── requirements.txt       # Python dependencies
├── era5_data/             # Downloaded ERA5 NetCDF files
├── models/                # Saved model checkpoints
└── output/                # Prediction outputs
```

## Tech Stack

- **PyTorch** — Model definition & training
- **xarray / NetCDF4** — Climate data I/O
- **Flask + Flask-CORS** — REST API
- **NumPy** — Numerical computation
- **CDS API** — ERA5 data retrieval
