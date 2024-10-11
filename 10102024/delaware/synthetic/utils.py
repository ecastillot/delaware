import os
import pandas as pd
from delaware.vel.montecarlo import VelModel
from delaware.vel.pykonal import get_xyz_velocity_model
from delaware.core.eqviewer_utils import single_latlon2yx_in_km

def get_stations_info_from_inventory(inventory):
    """
    Extract station information from an inventory object.

    Args:
        inventory: ObsPy inventory object containing network and station information.

    Returns:
        pd.DataFrame: DataFrame containing station information.
    """
    # Initialize an empty list to store station information
    data = []
    
    # Iterate over networks in the inventory
    for network in inventory:
        net_code = network.code
        
        # Iterate over stations in the network
        for station in network:
            # Extract station attributes
            sta_code = station.code
            sta_lat = station.latitude
            sta_lon = station.longitude
            sta_elv = station.elevation / 1e3  # Convert elevation to kilometers
            
            # Append station information to the list
            data.append([net_code, sta_code, sta_lon, sta_lat, sta_elv])
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=["network", "station", "longitude", "latitude", "elevation"])
    
    # Add a station index column
    df["station_index"] = df.index
    
    # Reorder columns with station index as the first column
    df = df[df.columns[-1:].tolist() + df.columns[:-1].tolist()] 
    
    return df

def change_file_extension(input_path, new_extension='.csv'):
    """
    Change the file extension of the given input path to the specified new extension.

    Args:
        input_path (str): The input file path.
        new_extension (str, optional): The new file extension to be applied. Defaults to '.csv'.

    Returns:
        str: The modified file path with the new extension.
    """
    # Split the file path into its directory and file name parts
    directory, filename = os.path.split(input_path)
    
    # Split the filename into its name and extension parts
    filename_without_extension, _ = os.path.splitext(filename)
    
    # Concatenate the directory, filename without extension, and new extension
    output_path = os.path.join(directory, filename_without_extension + new_extension)
    
    return output_path

def prepare_db1d_syn_vel_model(lons, lats, z, vel_path, proj):
    """
    Generates 3D DB pykonal velocity models for P-wave and S-wave phases using a 1D velocity profile.
    
    Parameters:
    lons (tuple): Longitude range in degrees (min, max).
    lats (tuple): Latitude range in degrees (min, max).
    z (array): Depth levels for the model.
    nx (int): Number of grid points in x-direction.
    ny (int): Number of grid points in y-direction.
    nz (int): Number of grid points in z-direction.
    vel_path (str): Path to the 1D velocity model file. Columns (Depth (km),VP (km/s),VS (km/s))
    proj (int): EPSG code for the projection.

    Returns:
    x,y,z in plane coords and profiles dict
    """
    
    # Read the 1D velocity model from the provided file path
    db1d = pd.read_csv(vel_path)
    
    # Create a DBVelModel object with the velocity data and set the topographic datum (z[0])
    db1d = DBVelModel(db1d, name="db1d", dtm=z[0])
    # Convert the 1D velocity profile into synthetic profiles for P and S waves
    profiles = db1d.to_synthetics()
    
    # Convert longitude and latitude from degrees to kilometers using the projection
    ymin, xmin = single_latlon2yx_in_km(lats[0], lons[0], proj)
    ymax, xmax = single_latlon2yx_in_km(lats[1], lons[1], proj)
    
    # Update x and y ranges in kilometers
    x = (xmin, xmax)
    y = (ymin, ymax)
    
    return x,y,z, profiles

def get_db1d_syn_vel_model(x, y, z, nx, ny, nz, vel_path, proj):
    """
    Generates 3D DB pykonal velocity models for P-wave and S-wave phases using a 1D velocity profile.
    
    Parameters:
    x (tuple): Longitude range in degrees (min, max).
    y (tuple): Latitude range in degrees (min, max).
    z (array): Depth levels for the model.
    nx (int): Number of grid points in x-direction.
    ny (int): Number of grid points in y-direction.
    nz (int): Number of grid points in z-direction.
    vel_path (str): Path to the 1D velocity model file. Columns (Depth (km),VP (km/s),VS (km/s))
    proj (int): EPSG code for the projection.

    Returns:
    dict: Dictionary containing 3D P-wave and S-wave velocity models.
    """
    
    x,y,z,profiles = prepare_db1d_syn_vel_model(x,y,z,vel_path,proj)
    
    
    # Generate the 3D P-wave velocity model
    p_model = get_xyz_velocity_model(
        x, y, z, nx, ny, nz, phase="P",
        xy_epsg=proj, profile=profiles["P"], layer=True
    )
    
    # Generate the 3D S-wave velocity model
    s_model = get_xyz_velocity_model(
        x, y, z, nx, ny, nz, phase="S",
        xy_epsg=proj, profile=profiles["S"], layer=True
    )
    
    # Return a dictionary with both P-wave and S-wave models
    return {"P": p_model, "S": s_model}

class DBVelModel(VelModel):
    def __init__(self, data, name, dtm=None) -> None:
        super().__init__(data, name, dtm)
        
    def to_synthetics(self):
        data = self.data.copy()
        p_data = data[["Depth (km)","VP (km/s)"]]
        p_data=p_data.rename(columns={"Depth (km)":"depth",
                               "VP (km/s)":"vel"})
        s_data = data[["Depth (km)","VS (km/s)"]]
        s_data=s_data.rename(columns={"Depth (km)":"depth",
                               "VS (km/s)":"vel"})
        
        vel_data = {"P":p_data.to_dict(orient="list"),
                    "S":s_data.to_dict(orient="list")}
        
        return vel_data

