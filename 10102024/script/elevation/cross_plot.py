import rasterio
import numpy as np
import pandas as pd
from pyproj import Transformer

# Input raster file
input_raster = "/content/TEXAS_basement_depth_km_AOI.tif"
output_csv = "/content/extracted_elevation.csv"

# Open the raster file
with rasterio.open(input_raster) as src:
    transform = src.transform  # Affine transformation
    data = src.read(1)  # Read raster data
    nodata_value = src.nodata  # NoData value
    crs_utm = src.crs  # Get the CRS (should be UTM)
    
    # Transformer to convert from UTM (EPSG:32614) to WGS 84 (EPSG:4326)
    transformer = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

    # Get raster dimensions
    height, width = data.shape

    # Create lists to store extracted values
    latitudes, longitudes, elevations = [], [], []

    # Loop over each pixel in the raster
    for row in range(height):
        for col in range(width):
            # Convert pixel coordinates to UTM
            utm_x, utm_y = transform * (col, row)

            # Convert UTM to Latitude/Longitude
            lon, lat = transformer.transform(utm_x, utm_y)

            # Get the elevation value
            elevation = data[row, col]

            # Skip NoData values
            if elevation == nodata_value:
                continue

            # Append to lists
            latitudes.append(lat)
            longitudes.append(lon)
            elevations.append(elevation)

# Create a DataFrame
df = pd.DataFrame({"Latitude": latitudes, "Longitude": longitudes, "Elevation": elevations})

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"CSV file saved: {output_csv}")
print(df.head())  # Print first few rows