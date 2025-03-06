import numpy as np
import pandas as pd
import os
import glob
import math
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import string
from obspy.geodetics.base import gps2dist_azimuth
from pyproj import Transformer


#### Transformation
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

def latlon2yx_in_km(data: pd.DataFrame, epsg: str):
    """
    Convert latitude and longitude coordinates to x and y coordinates in kilometers.

    Parameters:
    - data (pd.DataFrame): DataFrame containing latitude and longitude coordinates.
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
    data[['x[km]', 'y[km]']] = data.apply(get_xy, axis=1)
    return data