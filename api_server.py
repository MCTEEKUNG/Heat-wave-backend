from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import numpy as np
import os
import glob
from data_loader import load_era5_data, create_sequences
from heatwave_model import HeatwaveConvLSTM

app = Flask(__name__)
CORS(app) # Enable CORS

DATA_DIR = "era5_data"
MODELS_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables
model = None
X_test = None
Y_test = None
lats = None
lons = None
mean = None
std = None

def get_latest_model(model_dir):
    if not os.path.exists(model_dir): return None
    files = glob.glob(os.path.join(model_dir, "heatwave_model_checkpoint_v*.pth"))
    if not files: return None
    def get_version(f):
        try: return int(f.split("_v")[-1].split(".")[0])
        except: return 0
    return max(files, key=get_version)

def load_resources():
    global model, X_test, Y_test, lats, lons, mean, std
    
    print("Loading data...")
    try:
        data, lats, lons, mean, std = load_era5_data(DATA_DIR)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create sequences
    # We take the last part for testing/demo
    X, Y = create_sequences(data, seq_len=5, future_seq=2)
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]
    
    print(f"Loaded {len(X_test)} test samples.")
    
    # Load Model
    checkpoint_path = get_latest_model(MODELS_DIR)
    if not checkpoint_path:
        print(f"No model found in {MODELS_DIR}")
        return
        
    print(f"Loading model from {checkpoint_path}...")
    channels = X.shape[2]
    
    model = HeatwaveConvLSTM(input_dim=channels,
                             hidden_dim=[16, 16], 
                             kernel_size=[(3, 3), (3, 3)],
                             num_layers=2).to(DEVICE)
                             
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

def get_prediction_data(sample_idx=-1):
    """Run model inference and return denormalized temperature grid (Day T+1)"""
    # Input shape: (1, Seq, C, H, W)
    x_in = torch.FloatTensor(X_test[sample_idx]).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred = model(x_in, future_seq=2)
        pred_np = pred.squeeze(0).cpu().numpy()
        
    # Extract Temperature (Channel 1)
    temp_norm = pred_np[0, 1, :, :]
    
    # Denormalize
    temp_mean = mean[0, 1, 0, 0]
    temp_std = std[0, 1, 0, 0]
    
    temp_celsius = (temp_norm * (temp_std + 1e-6)) + temp_mean
    
    # Check if Kelvin
    if np.nanmean(temp_celsius) > 200:
        temp_celsius -= 273.15
        
    return temp_celsius

@app.route('/api/predict', methods=['GET'])
def predict_summary():
    """Returns summary statistics for the dashboard."""
    if model is None: return jsonify({"error": "Model not loaded"}), 500
    
    try:
        temp_grid = get_prediction_data()
        max_temp = float(np.max(temp_grid))
        mean_temp = float(np.mean(temp_grid))
        
        # Risk Logic
        risk_level = "LOW"
        if max_temp >= 35: risk_level = "MEDIUM"
        if max_temp >= 38: risk_level = "HIGH"
        if max_temp >= 41: risk_level = "CRITICAL"
        
        probability = 0.0
        if risk_level == "MEDIUM": probability = 0.5
        elif risk_level == "HIGH": probability = 0.8
        elif risk_level == "CRITICAL": probability = 0.95
        else: probability = 0.1
        
        return jsonify({
            "status": "ok",
            "date": "2000-XX-XX", # Mock date for now since we run on 2000 data
            "risk_level": risk_level,
            "probability": probability,
            "advice": f"Max temp reaching {max_temp:.1f}°C. {risk_level} precautions advised.",
            "model_type": "ConvLSTM",
            "weather": {
                "T2M_MAX": max_temp,
                "T2M": mean_temp,
                "T2M_MIN": float(np.min(temp_grid)),
                # Fill others with null/dummy for now
                "RH2M": 60,
                "WS10M": 2.5
            },
            "anomaly": {
                "is_anomaly": risk_level in ["HIGH", "CRITICAL"],
                "severity": risk_level.lower(),
                "n_triggers": 0,
                "triggers": []
            },
            "bbox": {
                "north": float(np.max(lats)),
                "south": float(np.min(lats)),
                "east": float(np.max(lons)),
                "west": float(np.min(lons))
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast', methods=['GET'])
def forecast_summary():
    """Returns a simple forecast (mocked for now using simple repetition)."""
    # In real world, we would predict T+1, T+2, ...
    # For now, let's just use the single prediction we have and shift it slightly
    if model is None: return jsonify({"error": "Model not loaded"}), 500
    
    try:
        temp_grid = get_prediction_data()
        mean_temp = float(np.mean(temp_grid))
        
        days = []
        import datetime
        base_date = datetime.datetime.now()
        
        for i in range(7):
            date = (base_date + datetime.timedelta(days=i+1)).strftime("%Y-%m-%d")
            # accurate prediction for day 1, others are just dummy variations
            t_max = float(np.max(temp_grid)) if i == 0 else float(np.max(temp_grid)) + np.random.uniform(-1, 1)
            
            days.append({
                "day": i+1,
                "date": date,
                "day_name": (base_date + datetime.timedelta(days=i+1)).strftime("%a"),
                "probability": 0.5, 
                "risk_level": "MEDIUM" if t_max > 35 else "LOW",
                "risk_label": "MEDIUM" if t_max > 35 else "LOW",
                "advice": "Generated from ConvLSTM",
                "weather": {
                    "T2M": mean_temp,
                    "T2M_MAX": t_max,
                    "T2M_MIN": float(np.min(temp_grid)),
                    "PRECTOTCORR": 0, "WS10M": 2, "RH2M": 60, "NDVI": 0.5
                }
            })
            
        return jsonify({
            "status": "ok",
            "model_type": "ConvLSTM",
            "days": 7,
            "generated_at": base_date.isoformat(),
            "forecasts": days
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/map', methods=['GET'])
def predict_map():
    """Returns GeoJSON features for the map."""
    if model is None: return jsonify({"error": "Model not loaded"}), 500
    
    try:
        temp_grid = get_prediction_data()
        
        features = []
        rows, cols = temp_grid.shape
        
        if len(lats) > 1: dlat = abs(lats[1] - lats[0])
        else: dlat = 0.25
        
        if len(lons) > 1: dlon = abs(lons[1] - lons[0])
        else: dlon = 0.25
        
        for r in range(rows):
            for c in range(cols):
                val = float(temp_grid[r, c])
                
                # Optimization: Only send polygons if temp > 30 to save bandwidth?
                # or send all for heatmap? Let's send all for now but maybe skip very low ones if needed.
                
                lat = float(lats[r])
                lon = float(lons[c])
                
                risk = 0
                if val >= 35: risk = 1
                if val >= 38: risk = 2
                if val >= 41: risk = 3
                
                min_lon, max_lon = lon - dlon/2, lon + dlon/2
                min_lat, max_lat = lat - dlat/2, lat + dlat/2
                
                poly_coords = [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat]
                ]
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [poly_coords]
                    },
                    "properties": {
                        "temperature": round(val, 2),
                        "risk_level": risk
                    }
                }
                features.append(feature)
        
        return jsonify({
            "type": "FeatureCollection",
            "features": features
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check for frontend to detect live backend."""
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == '__main__':
    load_resources()
    print("Starting Flask API on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
