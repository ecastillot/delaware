import numpy as np
import pandas as pd
from pyproj import Transformer
from obspy.geodetics.base import gps2dist_azimuth

def get_distance_in_dataframe(data: pd.DataFrame, lat1_name: str, lon1_name: str,
                              lat2_name: str, lon2_name: str,columns=None):
    """
    Compute distances between two sets of latitude and longitude coordinates in a DataFrame.

    Args:
    - data (pd.DataFrame): Input DataFrame containing the latitude and longitude columns.
    - lat1_name (str): Name of the column containing the first set of latitudes.
    - lon1_name (str): Name of the column containing the first set of longitudes.
    - lat2_name (str): Name of the column containing the second set of latitudes.
    - lon2_name (str): Name of the column containing the second set of longitudes.
    - columns (list): Default:None means 'r','az','baz'. 3d List representing distance, azimuth y back azimuth

    Returns:
    - pd.DataFrame: DataFrame with additional columns 'r', 'az', 'baz' representing distance (in km),
      azimuth (degrees clockwise from north), and back azimuth (degrees clockwise from south), respectively.
    """
    if data.empty:
        return data
    
    data = data.reset_index(drop=True)
    computing_r = lambda x: gps2dist_azimuth(x[lat1_name], x[lon1_name],
                                             x[lat2_name], x[lon2_name])
    r = data.apply(computing_r, axis=1)
    
    if columns is None:
        columns = ["r", "az", "baz"]
        
    data[columns] = pd.DataFrame(r.tolist(), index=data.index)
    data[columns[0]] = data[columns[0]] / 1e3
    return data

def single_latlon2yx_in_km(lat:float,lon:float, 
                    xy_epsg:str="EPSG:3116"):
    """Latitude and longitude coordinates to xy plane coordinates
    based on the EPSG defined.

    Args:
        lat (float): latitude
        lon (float): longitude
        xy_epsg (str, optional): EPSG. Defaults to "EPSG:3116".

    Returns:
        coords (tuple): y,x coordinates
    """
    transformer = Transformer.from_crs("EPSG:4326", xy_epsg)
    x,y = transformer.transform(lat,lon)
    coords = y/1e3,x/1e3
    return coords

def single_yx_in_km2latlon(y: float, x: float, xy_epsg: str = "EPSG:3116"):
    """
    Convert x and y coordinates in kilometers to latitude and longitude coordinates.

    Parameters:
    - y (float): y coordinate in kilometers.
    - x (float): x coordinate in kilometers.
    - xy_epsg (str): EPSG code specifying the coordinate reference system for x and y coordinates.
                     Default is EPSG:3116.

    Returns:
    - tuple: Tuple containing latitude and longitude coordinates.
    """
    transformer = Transformer.from_crs(xy_epsg, "EPSG:4326")  # Creating a Transformer object
    lon, lat = transformer.transform(x * 1e3, y * 1e3)  # Converting x and y from km to meters and transforming to latlon
    return lon,lat

def single_latlon2yx_in_km(lat:float,lon:float, 
                    xy_epsg:str="EPSG:3116"):
    """Latitude and longitude coordinates to xy plane coordinates
    based on the EPSG defined.

    Args:
        lat (float): latitude
        lon (float): longitude
        xy_epsg (str, optional): EPSG. Defaults to "EPSG:3116".

    Returns:
        coords (tuple): y,x coordinates
    """
    transformer = Transformer.from_crs("EPSG:4326", xy_epsg)
    x,y = transformer.transform(lat,lon)
    coords = y/1e3,x/1e3
    return coords

def latlon2yx_in_km(stations: pd.DataFrame, epsg: str):
    """
    Convert latitude and longitude coordinates to x and y coordinates in kilometers.

    Parameters:
    - stations (pd.DataFrame): DataFrame containing latitude and longitude coordinates.
    - epsg (str): EPSG code specifying the coordinate reference system.

    Returns:
    - pd.DataFrame: DataFrame with added columns 'x[km]' and 'y[km]' containing x and y coordinates in kilometers.
    """
    
    def get_xy(row):
        """
        Helper function to convert latitude and longitude to x and y coordinates in kilometers.

        Parameters:
        - row (pd.Series): A row of the DataFrame containing latitude and longitude.

        Returns:
        - pd.Series: Series containing 'x[km]' and 'y[km]' with converted coordinates.
        """
        y,x = single_latlon2yx_in_km(row.latitude, row.longitude, epsg)
        return pd.Series({'x[km]': x, 'y[km]': y})

    # Applying the get_xy function to each row of the DataFrame
    stations[['x[km]', 'y[km]']] = stations.apply(get_xy, axis=1)
    return stations

def get_minmax_coords_from_points(stations: pd.DataFrame, epsg: str, padding: list = 5):
    """
    Get the minimum and maximum coordinates from a DataFrame of points.

    Parameters:
    - stations (pd.DataFrame): DataFrame containing coordinates.
    - epsg (str): EPSG code specifying the coordinate reference system.
    - padding (list): Padding values to extend the bounding box. Default is 5.

    Returns:
    - tuple: Tuple containing minimum and maximum coordinates.
    """
    # Convert latitude, longitude coordinates to x, y, z coordinates in kilometers
    stations = latlon2yx_in_km(stations, epsg)

    # Get minimum and maximum coordinates in x, y, z dimensions
    min_coords = stations[["x[km]", "y[km]", "z[km]"]].min().values
    max_coords = stations[["x[km]", "y[km]", "z[km]"]].max().values

    # Apply padding to the minimum and maximum coordinates
    min_coords = min_coords - padding
    max_coords = max_coords + padding

    # Round and convert the coordinates to integers
    return np.round(min_coords).astype(int), np.round(max_coords).astype(int)

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

def inside_the_polygon(p,pol_points):
    """
    Parameters:
    -----------
    p: tuple
        Point of the event. (lon,lat)
    pol_points: list of tuples
        Each tuple indicates one polygon point (lon,lat).
    Returns: 
    --------
    True inside 
    """
    V = pol_points

    cn = 0  
    V = tuple(V[:])+(V[0],)
    for i in range(len(V)-1): 
        if ((V[i][1] <= p[1] and V[i+1][1] > p[1])   
            or (V[i][1] > p[1] and V[i+1][1] <= p[1])): 
            vt = (p[1] - V[i][1]) / float(V[i+1][1] - V[i][1])
            if p[0] < V[i][0] + vt * (V[i+1][0] - V[i][0]): 
                cn += 1  
    condition= cn % 2  
    
    if condition== 1:   
        return True
    else:
        return False