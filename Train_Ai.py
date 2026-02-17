import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import glob
import time

# Import Custom Modules
from data_loader import load_era5_data, create_sequences
from heatwave_model import HeatwaveConvLSTM, PhysicsInformedLoss

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "era5_data"
MODELS_DIR = "models"

# Hyperparameters
BATCH_SIZE = 4
SEQ_LEN = 5        # Input days
FUTURE_SEQ = 2     # Predict days
EPOCHS = 20        # Increased epochs for better results
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_next_version(model_dir: str, base_name: str = "heatwave_model_checkpoint") -> int:
    """Helper to determine the next model version number."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return 1
    
    files = glob.glob(os.path.join(model_dir, f"{base_name}_v*.pth"))
    if not files: return 1
        
    versions = []
    for f in files:
        try:
            part = f.split("_v")[-1]
            ver = int(part.split(".")[0])
            versions.append(ver)
        except: continue
            
    return max(versions) + 1 if versions else 1

def main():
    print(f"🚀 Starting Heatwave AI Training on {DEVICE}...")
    
    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    print("\n[1/4] Loading ERA5 Data...")
    try:
        # Load and normalize data
        # data shape: (Time, Channels, Height, Width)
        data, lats, lons, mean, std = load_era5_data(DATA_DIR)
        print(f"      Data Shape: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # ---------------------------------------------------------
    # 2. PREPARE SEQUENCES
    # ---------------------------------------------------------
    print("\n[2/4] Creating Sequences...")
    X, Y = create_sequences(data, SEQ_LEN, FUTURE_SEQ)
    
    # Train/Val Split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    
    # Convert to Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"      Training Samples: {len(X_train)}")
    print(f"      Validation Samples: {len(X_val)}")

    # ---------------------------------------------------------
    # 3. INITIALIZE MODEL
    # ---------------------------------------------------------
    print("\n[3/4] Initializing Model...")
    channels = X.shape[2] # (Batch, Time, Channels, H, W) -> index 2
    
    model = HeatwaveConvLSTM(
        input_dim=channels,
        hidden_dim=[16, 16],        # Hidden layers
        kernel_size=[(3, 3), (3, 3)],
        num_layers=2
    ).to(DEVICE)
    
    # Custom Loss: MSE + Physics Constraint
    criterion = PhysicsInformedLoss(lambda_phy=0.1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------------------------------------------------------
    # 4. TRAINING LOOP
    # ---------------------------------------------------------
    print("\n[4/4] Training Loop...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Pass
            output = model(x_batch, future_seq=FUTURE_SEQ)
            
            # Loss Calculation
            loss, mse_val, phy_val = criterion(output, y_batch)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # Process validation in batches to avoid OOM if val set is large
            # But for simplicity here we just do full batch if it fits, or manual batching
            # Let's use manual full batch for small data
            x_val_t = torch.FloatTensor(X_val).to(DEVICE)
            y_val_t = torch.FloatTensor(Y_val).to(DEVICE)
            
            val_out = model(x_val_t, future_seq=FUTURE_SEQ)
            v_loss, _, _ = criterion(val_out, y_val_t)
            val_loss = v_loss.item()
            
        print(f"      Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    duration = time.time() - start_time
    print(f"\n✅ Training Completed in {duration:.2f} seconds.")

    # ---------------------------------------------------------
    # SAVE CHECKPOINT
    # ---------------------------------------------------------
    version = get_next_version(MODELS_DIR)
    save_filename = f"heatwave_model_checkpoint_v{version}.pth"
    save_path = os.path.join(MODELS_DIR, save_filename)
    
    # Save state dict
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to: {save_path}")

if __name__ == "__main__":
    main()
