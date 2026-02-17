import os
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import โมเดลและ data loader
from heatwave_model import HeatwaveConvLSTM
from data_loader import load_era5_data, create_sequences

app = Flask(__name__)
CORS(app)  # อนุญาตให้ Frontend (Webapp) เรียก API ได้

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = "heatwave_model_checkpoint_v3.pth"
DATA_DIR = "era5_data"
SEQ_LEN = 5       # จำนวน time steps ที่ใช้เป็น input
FUTURE_SEQ = 2    # จำนวน time steps ที่พยากรณ์

# Hyperparameters ต้องตรงกับตอน train
CHANNELS = 3
HIDDEN_DIM = [16, 16]
KERNEL_SIZE = [(3, 3), (3, 3)]
NUM_LAYERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# โหลดโมเดลตอนเริ่ม Server
# ==========================================
model = HeatwaveConvLSTM(
    input_dim=CHANNELS,
    hidden_dim=HIDDEN_DIM,
    kernel_size=KERNEL_SIZE,
    num_layers=NUM_LAYERS
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

# รองรับทั้ง checkpoint ที่มี key 'model_state_dict' และไม่มี
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# ==========================================
# โหลดข้อมูล ERA5 และเตรียม sequences
# ==========================================
data_norm, lats, lons, mean, std = load_era5_data(DATA_DIR)
X, Y = create_sequences(data_norm, seq_len=SEQ_LEN, future_seq=FUTURE_SEQ)
print(f"✅ ERA5 Data loaded: {X.shape[0]} sequences available")

# ==========================================
# ENDPOINTS
# ==========================================

@app.route("/", methods=["GET"])
def root():
    """Health check"""
    return jsonify({
        "status": "ok",
        "message": "Heat-wave API is running 🌡️",
        "sequences_available": len(X)
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    รับ index ของ sequence แล้วพยากรณ์ล่วงหน้า FUTURE_SEQ steps
    Body: { "index": 0 }
    """
    body = request.get_json()
    idx = body.get("index", -1)  # default ใช้ sequence ล่าสุด

    # ดึง sequence ที่ต้องการ
    x_input = X[idx]  # (SEQ_LEN, C, H, W)
    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)
    # shape: (1, SEQ_LEN, C, H, W)

    with torch.no_grad():
        output = model(x_tensor, future_seq=FUTURE_SEQ)
    # output shape: (1, FUTURE_SEQ, C, H, W)

    # Denormalize กลับเป็นค่าจริง
    output_np = output.squeeze(0).cpu().numpy()  # (FUTURE_SEQ, C, H, W)
    output_denorm = output_np * std[:, :, 0, 0] + mean[:, :, 0, 0]

    # แยกแต่ละ channel
    # Channel 0 = Z (Geopotential), 1 = T2M (Temperature), 2 = SWVL1 (Soil Moisture)
    result = []
    for t in range(FUTURE_SEQ):
        result.append({
            "step": t + 1,
            "z500": output_denorm[t, 0].tolist(),      # Geopotential (H x W)
            "t2m": output_denorm[t, 1].tolist(),        # Temperature 2m (H x W)
            "swvl1": output_denorm[t, 2].tolist(),      # Soil Moisture (H x W)
        })

    return jsonify({
        "status": "success",
        "index": idx,
        "future_steps": FUTURE_SEQ,
        "lat_range": [float(lats.min()), float(lats.max())],
        "lon_range": [float(lons.min()), float(lons.max())],
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "predictions": result
    })


@app.route("/latest", methods=["GET"])
def latest():
    """
    พยากรณ์จาก sequence ล่าสุดในข้อมูล ERA5 โดยไม่ต้องส่ง body
    """
    x_input = X[-1]
    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x_tensor, future_seq=FUTURE_SEQ)

    output_np = output.squeeze(0).cpu().numpy()
    output_denorm = output_np * std[:, :, 0, 0] + mean[:, :, 0, 0]

    result = []
    for t in range(FUTURE_SEQ):
        result.append({
            "step": t + 1,
            "z500": output_denorm[t, 0].tolist(),
            "t2m": output_denorm[t, 1].tolist(),
            "swvl1": output_denorm[t, 2].tolist(),
        })

    return jsonify({
        "status": "success",
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "predictions": result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)