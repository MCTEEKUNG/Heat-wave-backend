import os
import xarray as xr
import torch
import numpy as np

def load_era5_data(data_dir, year=None):
    """
    Loads ERA5 data, crops to Thailand region IMMEDIATELY to save memory,
    then merges and normalizes.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith(".nc")]
    files.sort() # Ensure time order
    
    print(f"Scanning {len(files)} files in {data_dir}...")
    
    # Thailand Region Bounds
    # Lat: 5 to 21, Lon: 97 to 106
    LAT_MIN, LAT_MAX = 5, 21
    LON_MIN, LON_MAX = 97, 106
    
    processed_data_list = []
    
    # Global lat/lon arrays (to be captured from the first valid file)
    final_lats = None
    final_lons = None
    
    for f in files:
        path = os.path.join(data_dir, f)
        try:
            # open_dataset with chunks={} to use dask (lazy loading) if possible, 
            # but here we want to slice immediately.
            ds = xr.open_dataset(path)
            
            # 1. Drop unnecessary variables immediately
            # Keep only: z, t2m, swvl1
            vars_to_keep = [v for v in ['z', 't2m', 'swvl1'] if v in ds]
            if not vars_to_keep:
                continue
            ds = ds[vars_to_keep]
            
            # 2. Handle 'expver' (ERA5T vs ERA5)
            if 'expver' in ds.dims:
                # Combine expver (take the one that is not NaN) usually ERA5T vs ERA5
                # Safer: combine_first or reduce
                # print(f"  Collapsing expver dimension in {f}...")
                ds = ds.reduce(np.nanmax, dim='expver')
            elif 'expver' in ds.coords:
                ds = ds.drop_vars('expver')
                
            # 3. Rename generic names if needed (not implementing here unless known)
            
            # 4. CROP SPATIALLY
            # Determine slice direction for latitude
            if ds.latitude[0] < ds.latitude[-1]:
                lat_slice = slice(LAT_MIN, LAT_MAX)
            else:
                lat_slice = slice(LAT_MAX, LAT_MIN)
                
            ds_cropped = ds.sel(latitude=lat_slice, longitude=slice(LON_MIN, LON_MAX))
            
            # Check if empty (e.g. file covers different area)
            if ds_cropped.latitude.size == 0 or ds_cropped.longitude.size == 0:
                continue
                
            # 5. Load into memory (small chunk now!)
            # We want to standardize dimensions to (Time, Lat, Lon)
            # Ensure no extra dims like 'pressure_level'
            if 'pressure_level' in ds_cropped.coords:
                ds_cropped = ds_cropped.squeeze('pressure_level', drop=True)
                
            # Save coordinates from first file
            if final_lats is None:
                final_lats = ds_cropped.latitude.values
                final_lons = ds_cropped.longitude.values
                
            # Verify coordinates match (simple check)
            if ds_cropped.latitude.size != final_lats.size:
                # Interpolate or skip? For now, skip if mismatch to avoid error
                print(f"  Skipping {f} due to shape mismatch after crop.")
                continue
            
            # Ensure all required vars exist, fill with 0 if missing (e.g. swvl1 might be in separate file)
            # This is tricky if files are split by variable. 
            # If files are split by variable (e.g. z_2000.nc, t2m_2000.nc), we CANNOT crop and append yet.
            # We must merge by TIME first.
            
            # Assumption based on error logs: "Found z in era5_upper... Found t2m in ..."
            # It seems files are split by VARIABLE and YEAR. 
            # Merging by variable first, then cropping is memory intensive.
            # We should crop EACH variable file, then merge.
            
            processed_data_list.append(ds_cropped)
            print(f"  Loaded & Cropped: {f} {ds_cropped.dims}")
            
        except Exception as e:
            print(f"  Error processing {f}: {e}")
            continue

    if not processed_data_list:
        raise ValueError("No data loaded successfully.")

    print("Merging cropped datasets...")
    # Now merge the SMALL cropped datasets
    # compat='override' is safe now because they should be same grid
    try:
        full_ds = xr.merge(processed_data_list, compat='override', join='outer')
    except Exception as e:
        print(f"Merge failed: {e}")
        raise e

    print(f"Merged Data Shape: {full_ds.dims}")
    
    # Check for missing variables and fill
    for var in ['z', 't2m', 'swvl1']:
        if var not in full_ds:
             print(f"  Warning: {var} missing, filling with zeros")
             full_ds[var] = (('valid_time', 'latitude', 'longitude'), np.zeros((full_ds.dims['valid_time'], len(final_lats), len(final_lons))))

    # Interpolate NaNs over time
    # full_ds = full_ds.interpolate_na(dim='valid_time', method='linear', fill_value="extrapolate")
    
    # Normalize
    z = full_ds['z'].values
    t2m = full_ds['t2m'].values
    swvl1 = full_ds['swvl1'].values
    
    # Handle NaNs (from join='outer' or missing data)
    z = np.nan_to_num(z)
    t2m = np.nan_to_num(t2m)
    swvl1 = np.nan_to_num(swvl1)
    
    # Stack: (Time, Channels, H, W)
    data = np.stack([z, t2m, swvl1], axis=1)
    
    # Normalize
    mean = data.mean(axis=(0, 2, 3), keepdims=True)
    std = data.std(axis=(0, 2, 3), keepdims=True)
    data_norm = (data - mean) / (std + 1e-6)
    
    return data_norm, full_ds['latitude'].values, full_ds['longitude'].values, mean, std

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
