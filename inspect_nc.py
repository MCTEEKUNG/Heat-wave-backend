import xarray as xr
import os

data_dir = r"d:\Heart\NewHeart\era5_data"
files = [f for f in os.listdir(data_dir) if f.endswith(".nc")]

for f in files:
    path = os.path.join(data_dir, f)
    print(f"\n--- Inspecting {f} ---")
    try:
        ds = xr.open_dataset(path)
        print(ds)
        print("\nVariables:", list(ds.data_vars))
        ds.close()
    except Exception as e:
        print(f"Error reading {f}: {e}")
