import os
import xarray as xr
import torch
import numpy as np

def load_era5_data(data_dir, year=2000):
    """
    Loads ERA5 data from NetCDF files in the specified directory.
    Finds files containing 't2m', 'swvl1', and 'z' (geopotential).
    Merges them into a single dataset, crops to a 64x64 region, and normalizes.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith(".nc")]
    
    ds_list = []
    found_vars = set()
    
    print(f"Scanning {len(files)} files in {data_dir}...")
    
    for f in files:
        path = os.path.join(data_dir, f)
        try:
            ds = xr.open_dataset(path)
            # Standardize variable names if necessary (ERA5 sometimes uses different names)
            # t2m, swvl1, z
            
            # Check for variables
            for var in ['t2m', 'swvl1', 'z']:
                if var in ds.data_vars:
                    found_vars.add(var)
                    ds_list.append(ds)
                    print(f"  Found {var} in {f}")
        except Exception as e:
            print(f"  Could not read {f}: {e}")

    if not ds_list:
        raise ValueError("No valid ERA5 NetCDF files found.")

    # Merge datasets based on time and coordinates
    # We might need to look out for 'expver' coordinate which indicates preliminary data in ERA5
    try:
        full_ds = xr.merge(ds_list)
    except Exception as e:
        print(f"Merge failed: {e}. Trying to dropping expver if present.")
        # Sometimes expver causes issues. 
        clean_ds_list = []
        for ds in ds_list:
            if 'expver' in ds.coords:
                # Combine expver (take the one that is not NaN) usually ERA5T vs ERA5
                # For simplicity, just drop it or select the first one
                ds = ds.drop_vars('expver', errors='ignore')
            clean_ds_list.append(ds)
    # Merge all datasets
    # compat='override' handles conflicting attrs/values by taking the first one
    try:
        full_ds = xr.merge(clean_ds_list, compat='override')
    except Exception as e:
        print(f"Merge failed: {e}")
        # Fallback: try to just use the first file that has everything?
        # Or just raise
        raise e

    print("merged dataset:\n", full_ds)
    
    # Select variables
    req_vars = ['z', 't2m', 'swvl1']
    for v in req_vars:
        if v not in full_ds:
            print(f"Warning: Variable {v} missing in dataset!")
            # Filling with zeros for prototype if missing (strictly for testing pipeline)
            full_ds[v] = xr.full_like(full_ds[list(full_ds.data_vars)[0]], 0)

    # Crop to 64x64 spatial region to fit the model architecture
    # The inspected data was huge (813x105). 
    # Let's take a center crop or just the first 64x64.
    
    lat_size = full_ds.dims['latitude']
    lon_size = full_ds.dims['longitude']
    
    # Crop to Thailand Region (Lat 5-21, Lon 97-106)
    # The downloaded data might be broader than expected.
    # We use .sel(method='nearest') or slice to get the correct area.
    # Note: ERA5 latitudes are often descending (90 to -90), so slice might need (21, 5).
    
    print("Cropping to Thailand region (Lat: 21-5, Lon: 97-106)...")
    try:
        # Check if we need to sort or handle descending coords
        if full_ds.latitude[0] < full_ds.latitude[-1]:
            lat_slice = slice(5, 21)
        else:
            lat_slice = slice(21, 5) # Descending
            
        full_ds = full_ds.sel(latitude=lat_slice, longitude=slice(97, 106))
        
        # Ensure we have a consistent size (64x64) for the model?
        # If the selected region is smaller/larger, we might need to resize or pad.
        # 21-5 = 16 deg. 16/0.25 = 64 points.
        # 106-97 = 9 deg. 9/0.25 = 36 points.
        
        # We need 64x64 for the current model architecture.
        # Let's pad or resize if needed, OR just take a 64x64 chunks starting from there.
        # Actually, let's just resize/interpolate to 64x64 or change model input size effectively?
        # The model is ConvLSTM, it can handle variable sizes if fully convolutional?
        # No, the linear layers or flat/init_hidden might depend on size.
        # The provided model `heatwave_model.py` uses `init_hidden` which takes `image_size`.
        # So it SHOULD be flexible.
        
    except Exception as e:
        print(f"Cropping failed: {e}. Fallback to index slicing.")
        full_ds = full_ds.isel(latitude=slice(0, 64), longitude=slice(0, 64))

    print(f"Cropped dataset dimensions: {full_ds.dims}")

    # Convert to Numpy and Normalize
    # We need (Time, Channels, Height, Width)
    # Channels: z, t2m, swvl1
    
    z = full_ds['z'].values
    t2m = full_ds['t2m'].values
    swvl1 = full_ds['swvl1'].values
    
    print(f"Shapes before processing: Z={z.shape}, T2M={t2m.shape}, SWVL1={swvl1.shape}")

    # Squeeze Z if it has an extra dimension (pressure levels)
    if z.ndim == 4 and z.shape[1] == 1:
        z = z.squeeze(axis=1)
        print(f"Squeezed Z shape: {z.shape}")
        
    # Ensure all have same shape
    if z.shape != t2m.shape:
        print("Shape mismatch detected! Attempting to trim/broadcast.")
        # If time dimension differs slightly
        min_time = min(z.shape[0], t2m.shape[0], swvl1.shape[0])
        z = z[:min_time]
        t2m = t2m[:min_time]
        swvl1 = swvl1[:min_time]
    
    # Handle NaN values
    z = np.nan_to_num(z)
    t2m = np.nan_to_num(t2m)
    swvl1 = np.nan_to_num(swvl1)

    # Stack into (Time, Channels, H, W)
    # shapes are (Time, H, W)
    data = np.stack([z, t2m, swvl1], axis=1) # Axis 1 becomes channels
    
    # Normalize (Standard Scaling)
    mean = data.mean(axis=(0, 2, 3), keepdims=True)
    std = data.std(axis=(0, 2, 3), keepdims=True)
    
    data_norm = (data - mean) / (std + 1e-6)
    
    print(f"Final Data Shape: {data_norm.shape}")
    
    # Return coordinates and scaler statistics for API/Mapping
    lats = full_ds['latitude'].values
    lons = full_ds['longitude'].values
    
    return data_norm, lats, lons, mean, std

def create_sequences(data, seq_len=5, future_seq=2):
    """
    Creates (Input, Target) sequences.
    Input: (Batch, seq_len, C, H, W)
    Target: (Batch, future_seq, C, H, W)
    """
    num_samples = len(data) - seq_len - future_seq + 1
    
    X = []
    Y = []
    
    for i in range(num_samples):
        x_window = data[i : i + seq_len]
        y_window = data[i + seq_len : i + seq_len + future_seq]
        X.append(x_window)
        Y.append(y_window)
        
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    # Test
    d, lats, lons, m, s = load_era5_data("era5_data")
    x, y = create_sequences(d)
    print("X shape:", x.shape)
    print("Y shape:", y.shape)
    print("Lat range:", lats.min(), lats.max())
    print("Lon range:", lons.min(), lons.max())
