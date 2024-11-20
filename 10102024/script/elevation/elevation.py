import rasterio
import numpy as np

# Open the raster file
file_path = "/content/TEXAS_base_6_20_1km.tif"


# File paths
input_raster = "/content/TEXAS_base_6_20_1km.tif"
output_raster = "/content/TEXAS_base_6_20_1km_in_km.tif"

# Conversion factor: 1 foot = 0.0003048 kilometers
ft_to_km = 0.0003048

# Open the input raster
with rasterio.open(input_raster) as src:
    # Read the metadata
    metadata = src.meta
    
    # Read the raster data
    data = src.read(1)  # Assuming single-band raster

    # Get the NoData value
    nodata_value = metadata.get("nodata", None)

    # Ensure we handle NoData correctly
    if nodata_value is not None:
        # Apply the operation only to valid data (exclude NoData)
        positive_data = np.where(data != nodata_value, -1 * data, nodata_value)
    else:
        # If no NoData value is defined, just negate all values
        positive_data = -1 * data

    # Update metadata if necessary
    metadata.update(dtype="float32")  # Change to float32 if necessary for consistency

    # Save the output raster
    with rasterio.open(output_raster, "w", **metadata) as dst:
        dst.write(positive_data.astype("float32"), 1)