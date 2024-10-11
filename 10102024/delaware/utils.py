# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-24 21:02:12
#  * @modify date 2024-09-24 21:02:12
#  * @desc [description]
#  */
import pandas as pd
from datetime import datetime
from delaware.eqviewer.eqviewer import Catalog
from delaware.vel.velocity import VelModel
import pandas as pd
from delaware.synthetic.tt_utils import Stations,get_xyz_velocity_model,single_latlon2yx_in_km

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

def get_db_syn_stations(x, y, stations_path,proj):
    """Retrieve and filter station data within a specified polygon.

    Args:
        x (list): List of x-coordinates defining the polygon boundaries.
        y (list): List of y-coordinates defining the polygon boundaries.
        stations_path (str): Path to the CSV file containing station data.
        columns(network,station,latitude,longitude,elevation,)

    Returns:
        Stations: An instance of the Stations class containing filtered stations.
    """
    # Prepare stations data by reading from the CSV file
    stations_data = pd.read_csv(stations_path)
    
    
    # Add an index column for station identification
    stations_data["station_index"] = stations_data.index
    
    # Convert elevation from meters to kilometers
    stations_data["elevation"] = stations_data["elevation"] / 1e3
    
    # Define the polygon coordinates for the region of interest
    dw_w_pol = [
        (x[1], y[0]),
        (x[1], y[1]),
        (x[0], y[1]),
        (x[0], y[0]),
        (x[1], y[0])  # Closing the polygon
    ]
    
    # Create an instance of the Stations class with the prepared data
    stations = Stations(stations_data, xy_epsg=proj)
    
    # Filter the stations to include only those within the defined polygon
    stations.filter_region(polygon=dw_w_pol)
    
    return stations

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
        # for key,val in vel_data.copy().items():
        #     print(val)
            # data.rename()
            # vel_data[key]
            
        # print(vel_data)
        
        # print("hola")
        # p_vel
        

# class DB1D():
#     def __init__(self,data,dtm=-0.750) -> None:
#         data["Depth (km)"].replace("DTM", self.dtm, inplace=True)
#         data = data.astype(float)
        
#         self.data = data
        
        
#     def to_synthetics(self):
#         p_vel = 
        

def get_texnet_high_resolution_catalog(path):
    df = pd.read_csv(path)
    # Function to create datetime
    def create_datetime(row):
        try:
            return datetime(year=row['yr'], month=row['mon'], day=row['day'],
                            hour=row['hr'], minute=row['min_'], second=int(row['sec']),
                            microsecond=int((row['sec'] % 1) * 1e6))
        except ValueError:
            return pd.NaT  # or use `None` if you want to see errors

    # Apply function to create 'origin_time'
    df['origin_time'] = df.apply(create_datetime, axis=1)
    df = df.rename(columns={"latR":"latitude",
                            "lonR":"longitude",
                            "depR":"depth",
                            "mag":"magnitude",
                            "EventId":"id"})
    
    catalog = Catalog(df)
    return catalog
    
    # df.to_csv(outpath,index=False)
    
