import torch
import numpy as np
import matplotlib.pyplot as plt
from heatwave_model import HeatwaveConvLSTM
from data_loader import load_era5_data, create_sequences
import os
import glob
import sys

# Configuration
DATA_DIR = "era5_data"
MODELS_DIR = "models"
OUTPUT_DIR = "output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latest_model(model_dir, base_name="heatwave_model_checkpoint"):
    """
    Finds the latest version of the model checkpoint.
    """
    if not os.path.exists(model_dir):
        return None
        
    files = glob.glob(os.path.join(model_dir, f"{base_name}_v*.pth"))
    if not files:
        return None
        
    # Sort key: version number
    # "models/checkpoint_v1.pth" -> 1
    def get_version(f):
        try:
            return int(f.split("_v")[-1].split(".")[0])
        except:
            return 0
            
    latest_file = max(files, key=get_version)
    return latest_file

def plot_prediction(input_frame, target_frame, pred_frame, sample_idx=0):
    """
    Plots the Temperature channel (index 1) for Input, Target, and Prediction.
    """
    t_input = input_frame[1]
    t_target = target_frame[1]
    t_pred = pred_frame[1]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Input
    im0 = axes[0].imshow(t_input, cmap='hot')
    axes[0].set_title("Input (Day T)")
    plt.colorbar(im0, ax=axes[0])
    
    # Plot Target
    im1 = axes[1].imshow(t_target, cmap='hot')
    axes[1].set_title("Target (Day T+1)")
    plt.colorbar(im1, ax=axes[1])
    
    # Plot Prediction
    im2 = axes[2].imshow(t_pred, cmap='hot')
    axes[2].set_title("Forecast (Model)")
    plt.colorbar(im2, ax=axes[2])
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    save_path = os.path.join(OUTPUT_DIR, f"forecast_sample_{sample_idx}.png")
    plt.suptitle(f"Heatwave Forecast (Sample {sample_idx})")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

def run_inference():
    print(f"Loading data from {DATA_DIR}...")
    try:
        data, lats, lons, mean, std = load_era5_data(DATA_DIR)
    except Exception as e:
        print(f"Data loading error: {e}")
        return

    X, Y = create_sequences(data, seq_len=5, future_seq=2)
    
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]
    
    print(f"Test samples: {len(X_test)}")
    
    # Find Model
    checkpoint_path = get_latest_model(MODELS_DIR)
    if not checkpoint_path:
        # Fallback to old path if models dir empty
        if os.path.exists("heatwave_model_checkpoint.pth"):
             checkpoint_path = "heatwave_model_checkpoint.pth"
        else:
            print(f"Error: No model checkpoints found in {MODELS_DIR}!")
            return
            
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    channels = X.shape[2]
    model = HeatwaveConvLSTM(input_dim=channels,
                             hidden_dim=[16, 16],
                             kernel_size=[(3, 3), (3, 3)],
                             num_layers=2).to(DEVICE)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    print("Running inference...")
    with torch.no_grad():
        indices = np.random.choice(len(X_test), 3, replace=False)
        
        for idx in indices:
            x_in = torch.FloatTensor(X_test[idx]).unsqueeze(0).to(DEVICE)
            y_true = Y_test[idx]
            
            pred = model(x_in, future_seq=2)
            pred_np = pred.squeeze(0).cpu().numpy()
            
            input_last = X_test[idx][-1] 
            target_first = y_true[0]
            pred_first = pred_np[0]
            
            plot_prediction(input_last, target_first, pred_first, sample_idx=idx)
            
    print("Inference completed.")

if __name__ == "__main__":
    run_inference()
