import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import load_era5_data, create_sequences
from heatwave_model import HeatwaveConvLSTM, PhysicsInformedLoss
import numpy as np
import os
import glob

# Configuration
DATA_DIR = "era5_data"
MODELS_DIR = "models"
BATCH_SIZE = 4
SEQ_LEN = 5
FUTURE_SEQ = 2
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_next_version(model_dir, base_name="heatwave_model_checkpoint"):
    """
    Determines the next version number based on existing files.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return 1
    
    # List existing files
    files = glob.glob(os.path.join(model_dir, f"{base_name}_v*.pth"))
    if not files:
        return 1
        
    # Extract version numbers
    versions = []
    for f in files:
        try:
            # f is like "models/heatwave_model_checkpoint_v1.pth"
            part = f.split("_v")[-1] # "1.pth"
            ver = int(part.split(".")[0])
            versions.append(ver)
        except:
            continue
            
    if not versions:
        return 1
        
    return max(versions) + 1

def train():
    print(f"Using device: {DEVICE}")
    
    # Ensure models directory exists
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    # 1. Load Data
    print("Loading data...")
    try:
        data, _, _, _, _ = load_era5_data(DATA_DIR)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Create Sequences
    X, Y = create_sequences(data, SEQ_LEN, FUTURE_SEQ)
    
    # Train/Test Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    # Convert to PyTorch Tensors
    X_train = torch.FloatTensor(X_train).to(DEVICE)
    Y_train = torch.FloatTensor(Y_train).to(DEVICE)
    X_test = torch.FloatTensor(X_test).to(DEVICE)
    Y_test = torch.FloatTensor(Y_test).to(DEVICE)
    
    print(f"Training Samples: {len(X_train)}")
    print(f"Test Samples: {len(X_test)}")
    
    # 2. Initialize Model
    # Input shape: (Batch, Time, Channels, Height, Width)
    channels = X_train.shape[2]
    
    model = HeatwaveConvLSTM(input_dim=channels,
                             hidden_dim=[16, 16],
                             kernel_size=[(3, 3), (3, 3)],
                             num_layers=2).to(DEVICE)
    
    criterion = PhysicsInformedLoss(lambda_phy=0.1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    print("\nStarting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Batching
        num_batches = len(X_train) // BATCH_SIZE
        
        for i in range(0, len(X_train), BATCH_SIZE):
            x_batch = X_train[i : i + BATCH_SIZE]
            y_batch = Y_train[i : i + BATCH_SIZE]
            
            # Forward
            optimizer.zero_grad()
            output = model(x_batch, future_seq=FUTURE_SEQ)
            
            # Loss
            loss, mse, phy = criterion(output, y_batch)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / (num_batches + 1)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_output = model(X_test, future_seq=FUTURE_SEQ)
            v_loss, v_mse, v_phy = criterion(val_output, Y_test)
            val_loss = v_loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
    print("Training Completed.")
    
    # Save Model with Versioning
    base_name = "heatwave_model_checkpoint"
    version = get_next_version(MODELS_DIR, base_name)
    save_path = os.path.join(MODELS_DIR, f"{base_name}_v{version}.pth")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
