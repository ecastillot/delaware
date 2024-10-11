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

from delaware.core.eqviewer_utils import inside_the_polygon,single_yx_in_km2latlon

#### Velocity Model

class PykonalVelModel(ScalarField3D):
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

    field = PykonalVelModel(phase=phase,xy_epsg=xy_epsg,coord_sys='cartesian')
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