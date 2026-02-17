import cdsapi
import os

# ==========================================
# ERA5 Data Downloader for Heatwave Analysis
# ==========================================

# PREREQUISITES:
# 1. Install the CDS API client:
#    pip install cdsapi
#
# 2. Setup your API Key:
#    - Register/Login at https://cds.climate.copernicus.eu/
#    - Go to your User Profile to find your UID and API Key.
#    - Create a file named '.cdsapirc' in your home directory (e.g., C:\Users\Username\.cdsapirc).
#    - Add the following content to the file:
#      url: https://cds.climate.copernicus.eu/api/v2
#      key: 10aa1926-af5c-49c0-847b-d86976c2b9ae
#
#    Alternatively, you can pass the 'url' and 'key' arguments directly to cdsapi.Client(),
#    but using the config file is recommended for security.

def download_era5_data():
    # Initialize the Client
    # This will look for the .cdsapirc file in your home directory.
    c = cdsapi.Client()

    # Define the Geographic Area (Bounding Box)
    # Format: [North, West, South, East]
    # Area: Thailand/Southeast Asia
    AREA = [21, 97, 5, 106]

    # Timeframe: 2000 to 2023
    start_year = 2000
    end_year = 2023
    years = range(start_year, end_year + 1)

    # Output directory
    output_dir = "era5_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting download for years {start_year}-{end_year}...")
    print(f"Region: {AREA}")
    print(f"Output Directory: {output_dir}")

    for year in years:
        year_str = str(year)
        print(f"\nProcessing Year: {year_str}")

        # ---------------------------------------------------------
        # 1. Surface Data (Single Levels)
        # Variables: 2m Temperature, Volumetric Soil Water Layer 1
        # ---------------------------------------------------------
        surface_filename = os.path.join(output_dir, f"era5_surface_{year_str}.nc")
        
        if os.path.exists(surface_filename):
            print(f"File {surface_filename} already exists. Skipping.")
        else:
            print(f"Downloading Surface Data for {year_str}...")
            try:
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': [
                            '2m_temperature',
                            'volumetric_soil_water_layer_1',
                        ],
                        'year': year_str,
                        'month': [
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                        ],
                        'day': [
                            '01', '02', '03', '04', '05', '06',
                            '07', '08', '09', '10', '11', '12',
                            '13', '14', '15', '16', '17', '18',
                            '19', '20', '21', '22', '23', '24',
                            '25', '26', '27', '28', '29', '30',
                            '31',
                        ],
                        'time': [
                            '12:00', # As requested. 
                            # Note: For more temporal resolution, add more hours here.
                        ],
                        'area': AREA,
                    },
                    surface_filename)
                print(f"Saved: {surface_filename}")
            except Exception as e:
                print(f"Error downloading Surface Data for {year_str}: {e}")

        # ---------------------------------------------------------
        # 2. Upper Air Data (Pressure Levels)
        # Variable: Geopotential at 500hPa (Z500)
        # ---------------------------------------------------------
        upper_filename = os.path.join(output_dir, f"era5_upper_{year_str}.nc")

        if os.path.exists(upper_filename):
            print(f"File {upper_filename} already exists. Skipping.")
        else:
            print(f"Downloading Upper Air Data (500hPa) for {year_str}...")
            try:
                c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': 'geopotential',
                        'pressure_level': '500',
                        'year': year_str,
                        'month': [
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                        ],
                        'day': [
                            '01', '02', '03', '04', '05', '06',
                            '07', '08', '09', '10', '11', '12',
                            '13', '14', '15', '16', '17', '18',
                            '19', '20', '21', '22', '23', '24',
                            '25', '26', '27', '28', '29', '30',
                            '31',
                        ],
                        'time': [
                            '12:00',
                        ],
                        'area': AREA,
                    },
                    upper_filename)
                print(f"Saved: {upper_filename}")
            except Exception as e:
                print(f"Error downloading Upper Air Data for {year_str}: {e}")

    print("\nDownload process completed.")

if __name__ == '__main__':
    download_era5_data()
