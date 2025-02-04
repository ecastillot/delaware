# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2025-01-24 13:56:16
#  * @modify date 2025-01-24 13:56:16
#  * @desc [description]
#  */
import pandas as pd
from operator import add
import datetime as dt
from obspy.geodetics.base import gps2dist_azimuth

from .data import QuakeDataFrame
from . import utils as ut

class SinglePoint(object):
    def __init__(self, latitude: float, longitude: float, depth: float,
                 xy_epsg: str, origin_time: dt.datetime = None) -> None:
        """
        Initialize the Point object.

        Parameters:
        - latitude (float): Latitude of the Point.
        - longitude (float): Longitude of the Point.
        - depth (float): Depth of the Point.
        - xy_epsg (str): EPSG code specifying the coordinate reference system for x and y coordinates.
        - origin_time (dt.datetime): Origin time of the Point. Default is None.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.origin_time = origin_time
        self.lonlat_epsg = "EPSG:4326"
        self.xy_epsg = xy_epsg

        # Convert latitude and longitude to x and y coordinates in kilometers
        y,x = ut.single_latlon2yx_in_km(self.latitude, self.longitude, xy_epsg=xy_epsg)
        
        self.x = x
        self.y = y
        self.z = depth

    def __str__(self) -> str:
        """
        Return a string representation of the Point object.

        Returns:
        - str: String representation of the Point object.
        """
        msg1 = f"Point [{self.longitude},{self.latitude},{self.depth},{self.origin_time}]"
        msg2 = f"       ({self.xy_epsg}:km) -> [{self.x},{self.y},{self.z},{self.origin_time}]"
        msg = msg1 + "\n" + msg2
        return msg

class Points(QuakeDataFrame):
    def __init__(self, data, xy_epsg, author,required_columns=None,**kwargs ) -> None:
        
        col_id = None
        cols = ["ev_id","sta_id"]
        for key_id in cols:
            if key_id in data.columns.to_list():
                col_id = key_id
                break
        
        if col_id is None:
            raise Exception (f" None of these columns were found {cols}")
        else:
            self.key_id = col_id
           
        if required_columns is None:
            required_columns = ['latitude', 'longitude']
            
        self.lonlat_epsg = "EPSG:4326"
        self.xy_epsg = xy_epsg
        data = ut.latlon2yx_in_km(data, xy_epsg)
        super().__init__(data=data, 
                         required_columns=required_columns,
                         author=author,
                         **kwargs)
        
    def get_region(self,padding=[]):
        """
        It gets the region according to the limits in the coords
        Parameters:
        -----------
        padding: 4D-list or float or int
            list: Padding on each side of the region [lonw,lonw,lats,latn] in degrees.
            float or int: padding amount on each side of the region from 0 to 1,
                        where 1 is considered the distance on each side of the region.
        """
        lonw,lone = self.data.longitude.min(),self.data.longitude.max()
        lats,latn = self.data.latitude.min(),self.data.latitude.max()
        region = [lonw, lone, lats, latn]
        region = list(map(lambda x:round(x,2),region))

        if isinstance(padding,list):
            if region[0] == region[1]:
                region[0] = region[0] - 0.01
                region[1] = region[1] + 0.01
            if region[2] == region[3]:
                region[2] = region[2] - 0.01
                region[3] = region[3] + 0.01
            
            
            if padding:
                if len(padding) != 4:
                    raise Exception("Padding parameter must be 4D")
                else:
                    padding = [-padding[0],padding[1],-padding[2],padding[3]]
                    region = list( map(add, region, padding) )
        elif isinstance(padding,float) or isinstance(padding,int):
            if region[0] == region[1]:
                region[0] = region[0] - 0.01
                region[1] = region[1] + 0.01
            if region[2] == region[3]:
                region[2] = region[2] - 0.01
                region[3] = region[3] + 0.01
            
            lon_distance = abs(region[1]-region[0])
            lat_distance = abs(region[3]-region[2])
            adding4lon = lon_distance*padding
            adding4lat = lat_distance*padding
            padding = [-adding4lon, adding4lon, -adding4lat, adding4lat]
            region = list( map(add, region, padding) )
        return region
    
    def __str__(self,extended=False) -> str:
        msg = f"Points | {self.__len__()} items"
        if extended:
            region = list(map(lambda x: round(x,2),self.get_region()))
            msg += f"\n\tregion: {region}"
        else:
            pass
        return msg
    
    def sort_data_by_source(self, source: SinglePoint,ascending:bool=False):
        """
        Sorts data by distance from a specified source location.

        Parameters:
        - source (Point): The source location used for sorting.
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
    
    def filter_general_region(self,polygon):
        """
        Filter the region of the catalog.

        Parameters:
        -----------
        polygon: list of tuples
            Each tuple is consider a point (lon,lat).
            The first point must be equal to the last point in the polygon.
        
        """
        if polygon[0] != polygon[-1]:
            raise Exception("The first point must be equal to the last point in the polygon.")

        is_in_polygon = lambda x: ut.inside_the_polygon((x.longitude,x.latitude),polygon)
        mask = self.data[["longitude","latitude"]].apply(is_in_polygon,axis=1)
        self.data = self.data[mask]
        return self
    
    def filter_rectangular_region(self,region_lims):
        """
        Filter the region of the catalog.

        Parameters:
        -----------
        region_lims: list of 4 elements
            lonw,lone,lats,latw
        
        """
        polygon = [(region_lims[0],region_lims[2]),
                (region_lims[0],region_lims[3]),
                (region_lims[1],region_lims[3]),
                (region_lims[1],region_lims[2]),
                (region_lims[0],region_lims[2])
                ]
        if polygon[0] != polygon[-1]:
            raise Exception("The first point must be equal to the last point in the polygon.")

        is_in_polygon = lambda x: ut.inside_the_polygon((x.longitude,x.latitude),polygon)
        mask = self.data[["longitude","latitude"]].apply(is_in_polygon,axis=1)
        self.data = self.data[mask]
        return self
    
    def filter_by_r_az(self, latitude, longitude, r, az=None):
        """
        Filter data points based on distance (r) and optionally azimuth (az).

        Parameters:
        ----------
        latitude : float
            Latitude of the reference point.
        longitude : float
            Longitude of the reference point.
        r : float
            Maximum distance in kilometers to filter data points.
        az : float, optional
            Maximum azimuth in degrees to filter data points (default is None).
        
        Returns:
        -------
        self : object
            The object with updated data after filtering.
        """
        if self.empty:
            return self
        
        # Calculate distance, azimuth, and back-azimuth from the reference point
        # to each data point (latitude, longitude).
        is_in_polygon = lambda x: gps2dist_azimuth(
            latitude, longitude, x.latitude, x.longitude
        )
        data = self.data.copy()
        data.reset_index(drop=True,inplace=True)
        
        # Apply the 'is_in_polygon' function to each row in the DataFrame.
        # This results in a Series of tuples (r, az, baz) for each data point.
        mask = data[["longitude", "latitude"]].apply(is_in_polygon, axis=1)
        
        # Convert the Series of tuples into a DataFrame with columns 'r' (distance), 
        # 'az' (azimuth), and 'baz' (back-azimuth).
        mask = pd.DataFrame(mask.tolist(), columns=["r", "az", "baz"])
        
        # Convert distance 'r' from meters to kilometers.
        mask.loc[:, "r"] /= 1e3
        
        
        data[mask.columns.to_list()] = mask
        
        data = data[data["r"] < r]
        
        if az is not None:
            data = data[data["az"] < az]
        
        self.data = data
        # print(len(data))
        self.data.reset_index(drop=True,inplace=True)
        
        # Return the updated object to allow method chaining.
        return self
    
    def get_minmax_coords(self, padding: list = [5, 5, 1]):
        """
        Get the minimum and maximum coordinates from the station data.

        Parameters:
        - padding (list): Padding values to extend the bounding box. Default is [5, 5, 1].

        Returns:
        - tuple: Tuple containing minimum and maximum coordinates.
        """
        minmax_coords = ut.get_minmax_coords_from_points(self.data, 
                                                         self.xy_epsg, padding)
        return minmax_coords
    