# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-02-09 17:37:34
#  * @modify date 2024-02-09 17:37:34
#  * @desc [description]
#  */

import numpy as np
import h5py
import matplotlib
from itertools import product
import os
import random
import pandas as pd
import pykonal
from pykonal.transformations import sph2geo,geo2sph
from scipy.interpolate import interp1d
import datetime as dt
import pyproj
from pyproj import Transformer
from scipy.optimize import minimize
import concurrent.futures as cf
import time
from obspy.geodetics.base import gps2dist_azimuth
import itertools
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl.patch import geos_to_path
from matplotlib.collections import LineCollection
from pykonal.fields import ScalarField3D

import matplotlib.pyplot as plt
import concurrent.futures as cf
from mpl_toolkits.mplot3d import Axes3D



### Others

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

#### Earthquakes

class Source(object):
    def __init__(self, latitude: float, longitude: float, depth: float,
                 xy_epsg: str, origin_time: dt.datetime = None) -> None:
        """
        Initialize the Source object.

        Parameters:
        - latitude (float): Latitude of the source.
        - longitude (float): Longitude of the source.
        - depth (float): Depth of the source.
        - xy_epsg (str): EPSG code specifying the coordinate reference system for x and y coordinates.
        - origin_time (dt.datetime): Origin time of the source. Default is None.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.origin_time = origin_time
        self.lonlat_epsg = "EPSG:4326"
        self.xy_epsg = xy_epsg

        # Convert latitude and longitude to x and y coordinates in kilometers
        y,x = single_latlon2yx_in_km(self.latitude, self.longitude, xy_epsg=xy_epsg)
        
        self.x = x
        self.y = y
        self.z = depth

    def __str__(self) -> str:
        """
        Return a string representation of the Source object.

        Returns:
        - str: String representation of the Source object.
        """
        msg1 = f"Source [{self.longitude},{self.latitude},{self.depth},{self.origin_time}]"
        msg2 = f"       ({self.xy_epsg}:km) -> [{self.x},{self.y},{self.z},{self.origin_time}]"
        msg = msg1 + "\n" + msg2
        return msg

class Earthquakes(object):
    def __init__(self, data: pd.DataFrame, xy_epsg: str) -> None:
        """
        Initialize the Earthquakes object.

        Parameters:
        - data (pd.DataFrame): DataFrame containing earthquake data.
                            columns = ['latitude', 'longitude', 'depth', "origin_time"]
                            origin_time could be NaN values if you are not worry about the time
        - xy_epsg (str): EPSG code specifying the coordinate reference system for x and y coordinates.
        """
        self.lonlat_epsg = "EPSG:4326"
        self.xy_epsg = xy_epsg

        # Define mandatory columns
        self._mandatory_columns = ['latitude', 'longitude', 'depth', "origin_time"]
        
        # Check if all mandatory columns are present in the DataFrame
        check = all(item in data.columns.to_list() for item in self._mandatory_columns)
        if not check:
            raise Exception("There is not the mandatory columns for the data in Earthquakes object." \
                            + f"->{self._mandatory_columns}")

        # Convert latitude and longitude to x and y coordinates in kilometers
        data = latlon2yx_in_km(data, xy_epsg)
        
        # Copy depth to z column (assuming depth is already in kilometers)
        data["z[km]"] = data["depth"]
        
        # Set the processed data
        self.data = data

    def __str__(self) -> str:
        """
        Return a string representation of the Earthquakes object.

        Returns:
        - str: String representation of the Earthquakes object.
        """
        msg = f"Earthquakes ({len(self.data)})"
        return msg

    def get_minmax_coords(self, padding: list = [5, 5, 1]):
        """
        Get the minimum and maximum coordinates from the earthquake data.

        Parameters:
        - padding (list): Padding values to extend the bounding box. Default is [5, 5, 1].

        Returns:
        - tuple: Tuple containing minimum and maximum coordinates.
        """
        minmax_coords = get_minmax_coords_from_points(self.data, self.xy_epsg, padding)
        return minmax_coords

#### Stations

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

class Stations(object):
    def __init__(self, data: pd.DataFrame, xy_epsg: str) -> None:
        """
        Initialize the Stations object.

        Parameters:
        - data (pd.DataFrame): DataFrame containing station data.
                                columns = []'station_index', 'network', 'station',
                                            'latitude', 'longitude', 'elevation']
        - xy_epsg (str): EPSG code specifying the coordinate reference system for x and y coordinates.
        """
        self.lonlat_epsg = "EPSG:4326"
        self.xy_epsg = xy_epsg

        # Define mandatory columns
        self._mandatory_columns = ['station_index', 'network', 'station', 
                                   'latitude', 'longitude', 'elevation']
        
        # Check if all mandatory columns are present in the DataFrame
        check = all(item in data.columns.to_list() for item in self._mandatory_columns)
        if not check:
            raise Exception("There is not the mandatory columns for the data in Station object." \
                            + f"->{self._mandatory_columns}")

        # Convert latitude and longitude to x and y coordinates in kilometers
        data = latlon2yx_in_km(data, xy_epsg)
        
        # Convert elevation to z coordinates in kilometers
        data["z[km]"] = data["elevation"] * -1
        
        # Set the processed data
        self.data = data

    def __str__(self) -> str:
        """
        Return a string representation of the Stations object.

        Returns:
        - str: String representation of the Stations object.
        """
        msg = f"Stations ({len(self.data)})"
        return msg

    def get_minmax_coords(self, padding: list = [5, 5, 1]):
        """
        Get the minimum and maximum coordinates from the station data.

        Parameters:
        - padding (list): Padding values to extend the bounding box. Default is [5, 5, 1].

        Returns:
        - tuple: Tuple containing minimum and maximum coordinates.
        """
        minmax_coords = get_minmax_coords_from_points(self.data, self.xy_epsg, padding)
        return minmax_coords

    def sort_data_by_source(self, source: Source,ascending:bool=False):
        """
        Sorts data by distance from a specified source location.

        Parameters:
        - source (Source): The source location used for sorting.
        - ascending (bool,False): Sort ascending vs. descending. Specify list for multiple sort orders. 
                If this is a list of bools, must match the length of the by.

        Returns:
        - pd.DataFrame: DataFrame sorted by distance from the source.
        """

        # Extract data from the object
        stations = self.data

        if stations.empty:
            raise Exception("Stations Object can not be sorted because its data attribute is empty")

        # Define a distance function using the haversine formula
        distance_func = lambda y: gps2dist_azimuth(y.latitude, y.longitude,
                                                source.latitude, source.longitude)[0]/1e3

        # Compute distances and add a new 'sort_by_r' column to the DataFrame
        stations["sort_by_r"] = stations.apply(distance_func, axis=1)

        # Sort the DataFrame by the 'sort_by_r' column
        stations = stations.sort_values("sort_by_r",ascending=ascending, ignore_index=True)

        return stations
    
    def filter_region(self,polygon):
        """
        Filter the region of the data.

        Parameters:
        -----------
        polygon: list of tuples
            Each tuple is consider a point (lon,lat).
            The first point must be equal to the last point in the polygon.
        
        """
        if polygon[0] != polygon[-1]:
            raise Exception("The first point must be equal to the last point in the polygon.")

        is_in_polygon = lambda x: inside_the_polygon((x.longitude,x.latitude),polygon)
        mask = self.data[["longitude","latitude"]].apply(is_in_polygon,axis=1)
        
        self.data = self.data[mask]
        return self

#### Velocity Model

class VelModel(ScalarField3D):
    def __init__(self, phase: str, xy_epsg: str,
                 km2degree: float = 111, **kwargs):
        """
        Initialize the VelModel object.

        Args:
            phase (str): Phase type.
            xy_epsg (str): EPSG code.
            km2degree (float): Conversion factor from kilometers to degrees. Default is 111.
            **kwargs: Arguments corresponding to ScalarField3D.
        """
        self.phase = phase
        self.xy_epsg = xy_epsg
        self.km2degree = km2degree
        super().__init__(**kwargs)


    @classmethod
    def load_npz(cls, path, phase, xy_epsg, km2degree=111):
        """
        Load a VelModel object from an .npz file.

        Args:
            path (str): Path to the .npz file.
            phase (str): Phase type.
            xy_epsg (str): EPSG code.
            km2degree (float): Conversion factor from kilometers to degrees. Default is 111.

        Returns:
            VelModel: A new VelModel object loaded from the .npz file.
        """
        scalar_field3d = pykonal.fields.load(path)
        mycls = cls(phase, xy_epsg, km2degree)
        mycls.min_coords = scalar_field3d.min_coords
        mycls.npts = scalar_field3d.npts
        mycls.node_intervals = scalar_field3d.node_intervals
        mycls.values = scalar_field3d.values
        return mycls

    def __str__(self) -> str:
        """
        Return a string representation of the VelModel object.

        Returns:
            str: String representation of the VelModel object.
        """
        msg = f"Pykonal3DField | VelModel1D | Phase: {self.phase}"
        return msg
    
    @property
    def geo_min_coords(self):
        """
        Get the minimum geographic coordinates.

        Returns:
            tuple: Minimum geographic coordinates (longitude, latitude, depth).
        """
        min_x, min_y, min_z = self.min_coords
        min_lat, min_lon = single_yx_in_km2latlon(min_y, min_x,
                                                   xy_epsg=self.xy_epsg)
        geo_min_coords = round(min_lon,4), round(min_lat,4), round(min_z,4)
        return geo_min_coords
    
    @property
    def delta_in_km(self):
        """_summary_

        Get the distance in km between max and min coords

        Returns:
            tuple: distance in km in x,y,z
        """
        delta = self.max_coords - self.min_coords
        return round(delta,4)
    
    @property
    def geo_max_coords(self):
        """
        Get the maximum geographic coordinates.

        Returns:
            tuple: Maximum geographic coordinates (longitude, latitude, depth).
        """
        max_x, max_y, max_z = self.max_coords
        max_lat, max_lon = single_yx_in_km2latlon(max_y, max_x,
                                                   xy_epsg=self.xy_epsg)
        geo_max_coords = round(max_lon,4), round(max_lat,4), round(max_z,4)
        return geo_max_coords
    
    @property
    def approx_geo_node_intervals(self):
        """
        Get the approximate geographic node intervals.

        Returns:
            tuple: Approximate geographic node intervals (latitude, longitude, depth).
        """
        x_interval, y_interval, z_interval = self.node_intervals
        lat_interval = x_interval / self.km2degree
        lon_interval = y_interval / self.km2degree
        geo_intervals = lat_interval, lon_interval, z_interval
        return geo_intervals
    
    @property
    def geo_coords(self):
        """
        Get the geographic coordinates.

        Returns:
            tuple: Geographic coordinates (longitude, latitude, depth).
        """
        min_lon, min_lat, min_z = self.geo_min_coords
        max_lon, max_lat, max_z = self.geo_max_coords
        
        x_pts, y_pts, z_pts = self.npts
        lat = np.linspace(min_lat, max_lat, y_pts)
        lon = np.linspace(min_lon, max_lon, x_pts)
        z = np.linspace(min_z, max_z, z_pts)
        
        return np.round(lon,4), np.round(lat,4), np.round(z,4)
    
    @property
    def approx_geo_coords(self):
        """
        Get the approximate geographic coordinates.

        Returns:
            tuple: Approximate geographic coordinates (longitude, latitude, depth).
        """
        min_lat, min_lon, min_z = self.geo_min_coords
        lat_interval, lon_interval, z_interval = self.approx_geo_node_intervals
        
        lat_npts = np.arange(0, self.npts[1], 1)
        lon_npts = np.arange(0, self.npts[0], 1)
        z_npts = np.arange(0, self.npts[2], 1)
        
        lon = min_lon + lon_npts * lon_interval
        lat = min_lat + lat_npts * lat_interval
        z = min_z + z_npts * z_interval
        
        return lon, lat, z
      
    def filter_geo_coords(self, polygons: list):
        """
        Filters geographical coordinates based on specified polygons.

        Parameters:
        - polygons (list): List of polygons to check if points are inside.

        Returns:
        - pd.DataFrame: DataFrame containing geographical coordinates and
                        information about whether each point is inside any polygon.
        """

        # Extract lon, lat, and z from self.geo_coords
        lon, lat, z = self.geo_coords
        

        # Get all possible combinations of indices
        indices = list(product(range(lon.shape[0]), range(lat.shape[0]), range(z.shape[0])))

        # Create DataFrame with combinations
        df = pd.DataFrame(indices, columns=["src_lon", "src_lat", "src_z[km]"])

        # Convert source coordinates to geographical coordinates
        df["src_lon"] = df["src_lon"].apply(lambda x: lon[int(x)])
        df["src_lat"] = df["src_lat"].apply(lambda x: lat[int(x)])
        df["src_z[km]"] = df["src_z[km]"].apply(lambda x: z[int(x)])
        df["inside_pol"] = True

        if polygons:
            # Define a function to check if a point is inside any of the polygons
            def is_in_polygon(row):
                for polygon in polygons:
                    is_in_polygon = inside_the_polygon((row["src_lon"], row["src_lat"]), polygon)
                    if is_in_polygon:
                        return True
                return False

            # Create a new column indicating whether the point is inside any polygon
            df["inside_pol"] = df[["src_lon", "src_lat"]].apply(is_in_polygon, axis=1)

        return df
        
    
    def export_npz(self, output):
        """
        Export the object to an NPZ file.

        Args:
            output (str): The output file path for the NPZ file.

        Returns:
            bool: True if export is successful, False otherwise.
        """
        if isinstance(output, str):
            output_dir = os.path.dirname(output)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            self.savez(output)
            return True  # Return True to indicate successful export
        return False  # Return False if output is not a string
            
    def plot_profile(self, zmin: float = None, zmax: float = None,
                     show: bool = True, savefig:str=None):
        """
        Plot velocity profile.

        Args:
            zmin (float): Minimum depth to plot. Default is None.
            zmax (float): Maximum depth to plot. Default is None.
            show (bool): Whether to show the plot. Default is True.
            savefig: File path to save the plot. Default is None.

        Returns:
            Figure: Matplotlib figure object.
        """
        fig = plt.figure()
        rect = fig.patch
        rect.set_facecolor('w')

        _, _, z = self.geo_coords
        vel_z = self.values[0, 0, :]

        y, vp = z, vel_z

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.step(vp, y, 'c', linewidth=2.5, linestyle='-', label=f'Vel-{self.phase}')
        ax1.legend(loc='lower left')

        ax1.set_xlabel('Velocity [km/s]')
        ax1.set_ylabel('Depth [km]')
        ax1.set_ylim(ymin=zmin, ymax=zmax)
        ax1.scale_x = 1
        ax1.scale_y = 1
        ax1.invert_yaxis()
        ax1.grid()
        
        if savefig is not None:
            fig.savefig(savefig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_velocity_model(self, coords="geo", 
                            stations = None,
                            show: bool = True,
                            view_init_args=None,
                            savefig:str=None):
        """
        Plot velocity model.

        Args:
            coords (str): Coordinate system for plotting. Options are 'geo' or 'npts'. Default is 'geo'.
            show (bool): Whether to show the plot. Default is True.
            savefig: File path to save the plot. Default is None.

        Returns:
            Figure: Matplotlib figure object.
            ax: axes
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        jet_cmap = matplotlib.cm.jet
        
        if coords == "geo":
            lon, lat, z = self.geo_coords
            x_grid, y_grid, z_grid = np.meshgrid(lon, lat, z, indexing='xy')
            
            xlabel = "Lon [°]"
            ylabel = "Lat [°]"
            zlabel = "Depth [km]"
            
        elif coords == "npts":
            x, y, z = self.npts
            x_grid, y_grid, z_grid = np.mgrid[0:x:1, 0:y:1, 0:z:1]
            xlabel = "X nodes"
            ylabel = "Y nodes"
            zlabel = "Depth nodes"
            ax.set_aspect('equal')
        else:
            raise Exception("coords only can be 'geo' or 'npts'")
        
        values = self.values
        
        vel = ax.scatter(x_grid, y_grid, z_grid,  
                         c=values, 
                         cmap=jet_cmap.reversed())
        ax.invert_zaxis()
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        
        if coords == "geo":
            if stations is not None:
                stations_data = stations.data.copy()
                stations_data["z"] = z[0]-1
                
                ax.scatter(stations_data["longitude"],
                           stations_data["latitude"],
                           stations_data["z"],
                           color="black",
                           marker='^', 
                           alpha=1,
                           s=200
                           )
                print(z[0])
        # if coords == "geo":
        #     #fancy
        #     target_projection = ccrs.PlateCarree()

        #     # Add coast borders
        #     feature = cartopy.feature.COASTLINE
        #     geoms = feature.geometries()
        #     geoms = [target_projection.project_geometry(geom, feature.crs)
        #             for geom in geoms]

        #     paths = list(itertools.chain.from_iterable(geos_to_path(geom) for geom in geoms))

        #     segments = []
        #     for path in paths:
        #         vertices = [vertex for vertex, _ in path.iter_segments()]
        #         vertices = np.asarray(vertices)
        #         segments.append(vertices)

        #     lc = LineCollection(segments, color='yellow')
        #     ax.add_collection3d(lc)

        #     # Add country borders
        #     feature_borders = cartopy.feature.BORDERS
        #     geoms_borders = feature_borders.geometries()
        #     geoms_borders = [target_projection.project_geometry(geom, feature_borders.crs)
        #                     for geom in geoms_borders]
        #     paths_borders = list(itertools.chain.from_iterable(geos_to_path(geom) for geom in geoms_borders))
            
        #     segments_borders = []
        #     for path in paths_borders:
        #         vertices = [vertex for vertex, _ in path.iter_segments()]
        #         vertices = np.asarray(vertices)
        #         segments_borders.append(vertices)
        #     lc_borders = LineCollection(segments_borders, color='yellow')
        #     ax.add_collection3d(lc_borders)
        
        fig.colorbar(vel, ax=ax, fraction=0.03, pad=0.15, label=f"Vel-{self.phase}")
        
        if view_init_args is not None:
            ax.view_init(**view_init_args)  # Adjust elevation and azimuth as needed
        
        
        if savefig is not None:
            fig.savefig(savefig)
        
        if show:
            plt.show()
        
        return fig,ax

def get_xyz_layer_velocity_model(profile:dict={"depth":[-5,4,25,32,40,100],
                                        "vel":[4.8,6.6,7,8,8.1,8.2]}):
    """The interfaces in the 1d velocity model will be treated
    as a layer, it means it will maintain the abrupt change. 

    Args:
        profile (dict, optional): keys: depth,vel
                                    values: list of float
                                    Defaults to {"depth":[-10,4,25,32,40,100,200], 
                                                   "vel":[4.8,6.6,7,8,8.1,8.2,8.2]}.
    Raises:
        Exception: depth and vel values must have the same length

    Returns:
        velocity_profile: pd.DataFrame
    """

    if len(profile["depth"]) != len(profile["vel"]):
        raise Exception("The length of depth and vel lists must be the same")

    velocity_profile = {"depth":[profile["depth"][0]],
                        "vel":[profile["vel"][0]]}
    depths = profile["depth"]
    for i in range(1,len(depths)):
        velocity_profile["depth"].append(profile["depth"][i]-1e-5)
        velocity_profile["depth"].append(profile["depth"][i])
        velocity_profile["vel"].append(profile["vel"][i-1])
        velocity_profile["vel"].append(profile["vel"][i])
        
    velocity_profile = pd.DataFrame.from_dict(velocity_profile)
    return velocity_profile

def get_xyz_velocity_model(x:tuple,y:tuple,z:tuple,
                           nx:int,ny:int,nz:int,
                        xy_epsg:str,
                        phase:str,
                        profile={"depth":[-10,4,25,32,40,100,200],
                                "vel":[4.8,6.6,7,8,8.1,8.2,8.2]},
                        layer=True):
    """A 3D grid with a 1D velocity model. It will create
    a pykonal.fields.ScalarField3D based on the information
    defined in the parameters.

    Args:
        x (tuple): min_x,max_x
        y (tuple): min_y,max_y
        z (tuple): min_z,max_z
        nx (int): Number of points in x-axis
        ny (int): Number of points in y-axis
        nz (int): Number of points in z-axis
        xy_epsg (str): EPSG code
        phase (str): phase type
        vel1d (dict, optional): keys: depth,vel
                                values: list of float
                                Defaults to {"depth":[-10,4,25,32,40,100,200], 
                                                "vel":[4.8,6.6,7,8,8.1,8.2,8.2]}.
        layer (bool, optional): The interfaces in the velocity model could be
                                modeled linearly, if you select layer = True it means
                                you prefer to mantain the abrupt change. Defaults to True.
    
    Returns:
        field: pykonal.fields.ScalarField3D
    """

    if len(profile["depth"]) != len(profile["vel"]) :
        raise Exception("The length of depth and vel lists must be the same")

    if layer:
        profile = get_xyz_layer_velocity_model(profile)
    else:
        profile = pd.DataFrame.from_dict(profile)

    field = VelModel(phase=phase,xy_epsg=xy_epsg,coord_sys='cartesian')
    field.min_coords = np.array([x[0],y[0],z[0]])
    max_coords = np.array([x[1],y[1],z[1]])
    field.npts = np.array([nx,ny,nz])
    field.node_intervals = (max_coords - field.min_coords)  / (field.npts-1)

    values = np.zeros(field.npts)
    depth2interp = np.linspace(z[0],z[1],nz+1)
    interp = interp1d(profile["depth"].tolist(),profile["vel"].tolist())
    interp_vel = interp(depth2interp)
    for iz in range(field.npts[-1]):
        values[:,:,iz] = interp_vel[iz]
        
    field.values = values

    return field

if __name__ == "__main__":
    # test stations
    mainshock = Source(latitude=4.4160,
                    longitude=-73.8894,
                    depth=20,xy_epsg="EPSG:3116")
    
    stations_path = pd.read_csv("/home/emmanuel/ecastillo/dev/associator/GPA/data/CM_stations.csv")
    stations = Stations(stations_path,"EPSG:3116")
    stations.sort_data_by_source(mainshock)
    
    # ###test
    # ymin, xmin = single_latlon2yx_in_km(-1.403,-81.819,"EPSG:3116")
    # ymax, xmax= single_latlon2yx_in_km(14.571,-68.161,"EPSG:3116")
    # x = (xmin,xmax)
    # y = (ymin,ymax)
    # z = (-5,200)
    # P_profile = {"depth":[-10,4,25,32,40,100,200],
    #        "vel":[4.8,6.6,7,8,8.1,8.2,8.2]}
    # model = get_xyz_velocity_model(x,y,z,50,60,80,phase="P",
    #                 xy_epsg="EPSG:3116",
    #                 profile=P_profile,layer=True)
    # pol1 = [(-80,0),(-71,0),
    #         (-71,14),(-80,14),
    #         (-80,0)]
    # polygons = [pol1]
    # model.filter_geo_coords(polygons=polygons)
    
    
    # print(geo_coords)
    # model.plot_velocity_model()
    # print(geo_coords)
    # approx_geo_grid = model.approx_geo_grid
    # print(model.values.shape)
    # print(model.values[0,0,:].shape)
    # print(model.values[0,0,:])
    # model.plot_profile()
    # print(len(geo_grid[0]),len(geo_grid[1]),len(geo_grid[2]))
    # print(len(approx_geo_grid[0]),len(approx_geo_grid[1]),len(approx_geo_grid[2]))