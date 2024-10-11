import pandas as pd
import pandas as pd
from delaware.vel.velocity import VelModel

from delaware.synthetic.tt_utils import get_xyz_velocity_model,single_latlon2yx_in_km

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

