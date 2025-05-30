# /**
#  * @author [Emmanuel Castillo]
#  * @email [ecastillot@unal.edu.co]
#  * @create date 2022-11-03 12:07:04
#  * @modify date 2022-11-03 12:07:04
#  * @desc [description]
#  */
from operator import add
import copy
import types
import os
import numpy as np
import pygmt
import pandas as pd
import geopandas as gpd
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from obspy.geodetics.base import gps2dist_azimuth
from scipy import interpolate
from . import eqviewer_utils as ut
from tqdm import tqdm
from delaware.core.database import save_dataframe_to_sqlite, load_dataframe_from_sqlite
pygmt.config(FORMAT_GEO_MAP="ddd.xx")

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def args_cleaner(args,rm_args=[]):
    info = args
    for key in list(info.keys()):
        if key in rm_args:
            info.pop(key,None)

    return info

def update_kwargs(args1,args2):
    """
    keys in args1 will update to arg2
    """
    for key,value in args1.items():
        if key in list(args2.keys()):
            args2[key] = value
    return args2

class BaseMeca():
    def __init__(self,scale="1.0c",color="red",cmap=False,
                transparency=None) -> None:
        """
        color: str or None
            Color from pygmt color gallery. 
            It is not considered when cbar=True
        cmap: bool
            Use Colorbar (the specifications of the colorbar are located in FocalMechanisms object).
        scale: float
            default:  1cm -> M5
            Adjusts the scaling of the radius of the beachball, 
            which is proportional to the magnitude. 
            Scale defines the size for magnitude = 5.
        """
        self.scale = scale
        self.color = color
        self.cmap = cmap
        self.transparency = transparency
    
    def get_info2pygmt(self):
        args = self.__dict__.copy()
        return args

class BaseText():
    def __init__(self,**text_kwargs) -> None:
        """
        text_kwargs: pygmt.text arguments
        """
        self.text_kwargs = text_kwargs

    def get_info2pygmt(self):
        rm_args = ["text_kwargs"]
        args = self.__dict__.copy()
        args = args_cleaner(args,rm_args)
        return args

class BasePlot():
    def __init__(self,
            size=None,
            style=None,
            cmap=False,
            color="lightblue",
            label=None,
            transparency=0,
            pen=None,
            **plot_kwargs) -> None:
        """
        Parameters:
        -----------
        size: None or lambda function
            Equation for the size. 
            lambda x: 0.1 * np.sqrt(1.5 ** (x.magnitude*1.5))
            in this case size is assigned to the magnitude.
            
            If size is defined as lambda function,
            you must use 'style':"cc"
        style: str
            style of you data. 
            First letter assing the symbol.
            Second letter assign the measure distance
            If there is a number between the letter, means the
            size for every point.
            
            For instance, use c0.2c circle,0.2 centimeters for all data
            use cc, for circle and centimeters (this time size must be specified)
        cmap: bool
            Use Colorbar (the specifications of the colorbar always are located in '.plot' functions). 
        color: str or None
            Fill color from pygmt color gallery. 
            It is not considered when cmap=True
        transparency: float
            transparency of your plots
        pen : str or None
            color and size of the symbol border
        plot_kwargs: other pygmt.plot arguments
        """
        self.size = size
        self.color = color
        self.label = label
        self.style = style
        self.cmap = cmap
        self.transparency = transparency
        self.pen = pen
        self.plot_kwargs = plot_kwargs

    def get_info2pygmt(self,data=pd.DataFrame):
        rm_args = ["plot_kwargs"]
        args = self.__dict__.copy()
        
        if type(self.size) is types.LambdaType:
            if not data.empty:
                args["size"] = data.apply(args["size"],axis=1)
            else:
                raise Exception("No data to apply size")

        kwargs = update_kwargs(args,self.plot_kwargs)
        args.update(kwargs)
        args = args_cleaner(args,rm_args)
        return args

class BaseProfile():
    def __init__(self,
            projection,
            depth_lims,
            output_unit="m",
            grid = None,
            reverse_xy=False,
            ) -> None:
        """
        Parameters:
        -----------
        output_unit: str
            "m" or "km"
        grid: tuple
            xy grid in the same units as output units
        reverse_xy: bool
            Reverse the figure
        
        panel: bool or int or list
            Select a specific subplot panel. Only allowed when in subplot mode. 
            Use panel=True to advance to the next panel in the selected order.
            Instead of row,col you may also give a scalar value index which depends 
            on the order you set via autolabel when the subplot was defined. 
            Note: row, col, and index all start at 0.
        """
        self.projection = projection
        self.output_unit = output_unit
        self.depth_lims = depth_lims
        self.grid = grid 
        self.reverse_xy = reverse_xy

    def get_basemap_args(self,max_distance):
        if self.output_unit == "km":
            distance_lims = [0,max_distance/1e3]
        else:
            distance_lims = [0,max_distance]


        if self.reverse_xy:
            region = self.depth_lims + distance_lims
            # projection="x?/?"
            y_label = "Distance"
            x_label = "Depth"
            f_label= "ESnw"
        else:
            region = distance_lims + self.depth_lims
            # projection="x?/-?"
            # projection="X10/-10"
            x_label = "Distance"
            y_label = "Depth"
            f_label= "WSrt"

        if self.grid == None:
            grid1 = region[:2]
            grid1 = round((grid1[1] -grid1[0])/5,2)
            grid2 = region[2:]
            grid2 = round((grid2[1] -grid2[0])/5,2)
            grid = [grid1,grid2]
        else:
            grid = self.grid

        frame=[
                f'xafg{grid[0]}+l"{x_label} ({self.output_unit})"', 
                f'yafg{grid[1]}+l"{y_label} ({self.output_unit})"',
                f_label
                ]
        return {"region":region,"projection":self.projection,
                "frame":frame}

class CPT():
   def __init__(self,color_target,label,**makecpt_kwargs):
        """
        Parameters:
        -----------
        color_target: "str"
            Target name to apply the colorbar.
        label: "str"
            Label to show in the colorbar legend
        makecpt_kwargs:
            Args from pygmt.makecpt
        """
        self.color_target = color_target
        self.label = label
        self.makecpt_kwargs = makecpt_kwargs

class Picks():
    def __init__(self,
            data,
            author,
            p_color="blue",
            s_color="red",
            ) -> None:

        """
        Parameters:
        -----------
        data: pd.DataFrame 
            ev_id,arrival_time,phase_hint,station
        """
        self.author = author
        
        self._mandatory_columns = ['ev_id','network','station','arrival_time','phase_hint']
        
        # Check if all mandatory columns are present in the DataFrame
        check = all(item in data.columns.to_list() for item in self._mandatory_columns)
        if not check:
            raise Exception("There is not the mandatory columns for the data in Picks object." \
                            + f"->{self._mandatory_columns}")
        
        self.data = data
        self.p_color = p_color
        self.s_color = s_color
    
    @property
    def empty(self):
        return self.data.empty
    
    @property
    def events(self):
        return list(set(self.data["ev_id"]))

    def __len__(self):
        return len(self.data)

    def __str__(self) -> str:
        
        msg = f"Picks | {len(self.events)} events, {self.__len__()} picks  "
        return msg
    
    def __getitem__(self, key):
        
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def get_lead_station(self):
        min_idx = self.data['arrival_time'].idxmin()  # Get the index of the minimum arrival time
        row = self.data.loc[min_idx, ['arrival_time', 'station']]  # Retrieve the station at that index
        return row.station,row.arrival_time
    
    def get_station_ids(self):
        data = self.data.copy()
        data = data.drop_duplicates(subset=["network","station"],ignore_index=True)
        
        data["station_ids"] = data.apply(lambda x: ".".join((x.network,
                                                             x.station)) ,axis=1)
        return data["station_ids"].to_list()

    def filter(self,key,start=None,end=None):
        """
        Filter data of the catalog.

        Parameters:
        -----------
        key: str
            Name of the column to filter
        start: int or float or datetime.datetime
            must be the same type as data[key] does
        end: int or float or datetime.datetime
            must be the same type as data[key] does
        
        """
        if (start is not None) and (len(self) !=0):
            self.data = self.data[self.data[key]>=start]
        if (end is not None) and (len(self) !=0):
            self.data = self.data[self.data[key]<=end]
            
        self.data.reset_index(drop=True,inplace=True)
        return self
    
    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)
    
    def sort_values(self,**args):
        """
        Sort values. Take in mind that it could affect the order of the events plotted
        args: The parameters are the pd.DataFrame.sort_values parameters
        """
        self.data = self.data.sort_values(**args)
        self.data.reset_index(drop=True,inplace=True)
        return self
    
    def remove_data(self, rowval):
        """
        remove rows to the data.

        Parameters:
        -----------
        rowval : dict
            key: 
                column name
            value: list
                values specified to remove
        """
        if not isinstance(rowval,dict):
            raise Exception("rowval must be a dictionary")
        
        mask = self.data.isin(rowval)
        mask = mask.any(axis='columns')
        self.data = self.data[~mask]
        self.data.reset_index(drop=True,inplace=True)
        return self
    
    def select_data(self, rowval):
        """
        select rows to the data.

        Parameters:
        -----------
        rowval : dict
            key: 
                column name
            value: list
                values specified to select
        """
        if not isinstance(rowval,dict):
            raise Exception("rowval must be a dictionary")
        if self.empty:
            return self
        mask = self.data.isin(rowval)
        mask = mask.any(axis='columns')
        self.data = self.data[mask]
        
        return self
    
    def filter_requiring_ps_phases_in_station(self):
        
        if self.data.empty:
            return self
        
        data = self.data.copy()
        picks = []
        for x,df in data.groupby(["ev_id","station"]):
            df = df.drop_duplicates(["phase_hint"])
            if len(df) == 2: #p and s phases
                picks.append(df)
        
        if not picks:
            picks = pd.DataFrame()
        else:
            picks = pd.concat(picks,axis=0)
                
        self.data = picks
        
        return self

class MulPicks():
    def __init__(self,picks=[]):
        """
        Parameters:
        -----------
        picks: list
            list of Catalog objects
        """
        self.picks = picks
        
    def __iter__(self):
        return list(self.picks).__iter__()

    def __nonzero__(self):
        return bool(len(self.picks))

    def __len__(self):
        return len(self.picks)

    def __str__(self,extended=False) -> str:
        msg = f"MulPicks ({self.__len__()} Picks objects)\n"
        msg += "-"*len(msg) 

        submsgs = []
        for i,pick in enumerate(self.__iter__(),1):
            submsg = f"{i}. "+pick.__str__()
            submsgs.append(submsg)
                
        if len(self.picks)<=20 or extended is True:
            submsgs = "\n".join(submsgs)
        else:
            three_first_submsgs = submsgs[0:3]
            last_two_subsgs = submsgs[-2:]
            len_others = len(self.picks) -len(three_first_submsgs) - len(last_two_subsgs)
            submsgs = "\n".join(three_first_submsgs+\
                                [f"...{len_others} other picks..."]+\
                                last_two_subsgs)

        return msg+ "\n" +submsgs

    def __setitem__(self, index, trace):
        self.picks.__setitem__(index, trace)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(picks=self.picks.__getitem__(index))
        else:
            return self.picks.__getitem__(index)

    def __delitem__(self, index):
        return self.picks.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        return self.__class__(picks=self.picks[max(0, i):max(0, j):k])

    def get_lead_station(self,preferred_author=None):
        lead_station = []
        for pick in self.picks:
            if preferred_author is not None:
                if pick.author != preferred_author:
                    continue
            lead_station.append(pick.get_lead_station())
            
        lead_station = pd.DataFrame(lead_station,columns=["station","arrival_time"])
        min_idx = lead_station['arrival_time'].idxmin()  # Get the index of the minimum arrival time
        row = lead_station.loc[min_idx, ['arrival_time', 'station']]  # Retrieve the station at that index
        return row.station,row.arrival_time
    
    def compare(self,author1,author2,
                sr_columns = None
                ):
        
        list2compare = [author1,author2]
        
        if sr_columns is None:
            sr_columns = ["sr_r [km]","sr_az","sr_baz"]
        
        # print(list2compare)
        data2compare = []
        for pick in self.__iter__():
            if pick.author in list2compare:
                data = pick.data.copy()
                _mandatory_columns = pick._mandatory_columns
                
                _optional_columns = [v for v in sr_columns\
                            if v in data.columns.to_list()]
                
                if _optional_columns:
                    cols = _mandatory_columns + _optional_columns
                else:
                    cols = _mandatory_columns
                
                data = data[cols]
                data2compare.append(data)
                
        key = "arrival_time"
        cols2merge =  _mandatory_columns.copy() 
        cols2merge.remove(key)
        suffixes = list(map(lambda x: "_"+x,list2compare))
        data = pd.merge(left=data2compare[0], 
                        right=data2compare[1],
                        on=cols2merge,
                        suffixes=suffixes )
        
        data.loc[:,"delta_arrival_time"] = data[key+suffixes[0]] - data[key+suffixes[1]]
        data['delta_arrival_time'] = data['delta_arrival_time'].dt.total_seconds()
        
        return data
            

    def remove_data(self,rowval):
        """
        remove rows of each data.

        Parameters:
        -----------
        rowval : dict
            key:  
                column name
            value: 
                One or more values specified to remove
        """
        picks = []
        for pick in self.picks:
            picks.append(pick.remove_data(rowval))
        self.picks = picks
        return self

    def select_data(self,rowval):
        """
        select rows of each data.

        Parameters:
        -----------
        rowval : dict
            key:  
                column name
            value: 
                One or more values specified to select
        """
        picks = []
        for pick in self.picks:
            picks.append(pick.select_data(rowval))
        self.picks = picks
        return self

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)

    def sort_values(self,**args):
        """
        Sort values. Take in mind that it could affect the order of the events plotted
        args: The parameters are the pd.DataFrame.sort_values parameters
        """
        picks = []
        for pick in self.picks:
            picks.append(pick.sort_values(**args))
        self.picks = picks
        return self
    
    def filter(self,key,start=None,end=None):
        """
        Filter data of the pick.

        Parameters:
        -----------
        key: str
            Name of the column to filter
        start: int or float or datetime.datetime
            must be the same type as data[key] does
        end: int or float or datetime.datetime
            must be the same type as data[key] does
        
        """
        picks = []
        for pick in self.picks:
            picks.append(pick.filter(key,start,end))
        self.picks = picks
        return self

    def get_station_ids(self,preferred_author=None):
        station_ids = []
        for pick in self.picks:
            
            if preferred_author is not None:
                if pick.author != preferred_author:
                    continue
            station_ids += pick.get_station_ids()
        
        if not station_ids:
            raise Exception("Empty stations ids. Review your preferred author in case you are using it")
        
        station_ids = list(set(station_ids))
        return station_ids

class Catalog():
    def __init__(self,
            data,
            xy_epsg: str,
            baseplot = BasePlot(size=None,
                        style="c0.2c",
                        cmap=False,
                        color="lightblue",
                        label="data",
                        transparency=0,
                        pen=None)
            ) -> None:

        """
        Parameters:
        -----------
        data: pd.DataFrame 
            Dataframe with the next mandatory columns:
            'origin_time','latitude','longitude','depth','magnitude'
        baseplot: BasePlot
            Control plot args
        """
        self.lonlat_epsg = "EPSG:4326"
        self.xy_epsg = xy_epsg
        
        self._mandatory_columns = ['origin_time','latitude','longitude','depth','magnitude']
        # Check if all mandatory columns are present in the DataFrame
        check = all(item in data.columns.to_list() for item in self._mandatory_columns)
        if not check:
            raise Exception("There is not the mandatory columns for the data in Earthquakes object." \
                            + f"->{self._mandatory_columns}")

        # Convert latitude and longitude to x and y coordinates in kilometers
        data = ut.latlon2yx_in_km(data, xy_epsg)
        
        # Copy depth to z column (assuming depth is already in kilometers)
        data["z[km]"] = data["depth"]
        
        data = data.drop_duplicates(subset=self._mandatory_columns,ignore_index=True)
        # pd.to_datetime(data.loc[:,"origin_time"],format="mixed").dt.tz_localize(None)
        data["origin_time"] = pd.to_datetime(data["origin_time"],errors='coerce').dt.tz_localize(None)
        
        # self.data = data[self.columns]
        self.data = data
        if self.empty:
            raise Exception("No data in the catalog")

        self.baseplot = baseplot
        self.profiles = []

    @property
    def empty(self):
        return self.data.empty

    def __len__(self):
        return len(self.data)

    def __str__(self,extended=False) -> str:
        if extended:
            timefmt = "%Y%m%dT%H:%M:%S"
            start=  self.data.origin_time.min()
            end = self.data.origin_time.max()
            region = list(map(lambda x: round(x,2),self.get_region()))
            msg = f"Catalog | {self.__len__()} events "\
                    +f"\n\tperiod: [{start.strftime(timefmt)} - {end.strftime(timefmt)}]"\
                    +f"\n\tdepth : {[round(self.data.depth.min(),2),round(self.data.depth.max(),2)]}"\
                    +f"\n\tmagnitude : {[round(self.data.magnitude.min(),2),round(self.data.magnitude.max(),2)]}"\
                    +f"\n\tregion: {region}"
        else:
            msg = f"Catalog | {self.__len__()} events "

        return msg

    def append(self, data):
        """
        append data
        """
        if isinstance(data, pd.DataFrame):

            check =  all(item in data.columns.to_list() for item in self.columns)
            if not check:
                raise Exception("There is not the mandatory columns for the data in Catalog object."\
                                +"->'origin_time','latitude','longitude','depth','magnitude'")

            data = data.drop_duplicates(subset=self.columns,ignore_index=True)
            pd.to_datetime(data.loc[:,"origin_time"]).dt.tz_localize(None)

            self.data = pd.concat([self.data,data])
        else:
            msg = 'Append only supports a single Dataframe object as an argument.'
            raise TypeError(msg)
        return self

    def remove_data(self, rowval):
        """
        remove rows to the data.

        Parameters:
        -----------
        rowval : dict
            key: 
                column name
            value: list
                values specified to remove
        """
        if not isinstance(rowval,dict):
            raise Exception("rowval must be a dictionary")
        
        mask = self.data.isin(rowval)
        mask = mask.any(axis='columns')
        self.data = self.data[~mask]
        self.data.reset_index(drop=True,inplace=True)
        return self
    
    def select_data(self, rowval):
        """
        select rows to the data.

        Parameters:
        -----------
        rowval : dict
            key: 
                column name
            value: list
                values specified to select
        """
        if not isinstance(rowval,dict):
            raise Exception("rowval must be a dictionary")
        if self.empty:
            return self
        mask = self.data.isin(rowval)
        mask = mask.any(axis='columns')
        self.data = self.data[mask]
        self.data.reset_index(drop=True,inplace=True)
        return self

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)

    def sort_values(self,**args):
        """
        Sort values. Take in mind that it could affect the order of the events plotted
        args: The parameters are the pd.DataFrame.sort_values parameters
        """
        self.data = self.data.sort_values(**args)
        self.data.reset_index(drop=True,inplace=True)
        return self

    def filter(self,key,start=None,end=None):
        """
        Filter data of the catalog.

        Parameters:
        -----------
        key: str
            Name of the column to filter
        start: int or float or datetime.datetime
            must be the same type as data[key] does
        end: int or float or datetime.datetime
            must be the same type as data[key] does
        
        """
        if (start is not None) and (len(self) !=0):
            self.data = self.data[self.data[key]>=start]
        if (end is not None) and (len(self) !=0):
            self.data = self.data[self.data[key]<=end]
            
        self.data.reset_index(drop=True,inplace=True)
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
        self.data.reset_index(drop=True,inplace=True)
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
        self.data.reset_index(drop=True,inplace=True)
        return self

    def get_minmax_coords(self, padding: list = [5, 5, 1]):
        """
        Get the minimum and maximum coordinates from the earthquake data.

        Parameters:
        - padding (list): Padding values to extend the bounding box. Default is [5, 5, 1].

        Returns:
        - tuple: Tuple containing minimum and maximum coordinates.
        """
        minmax_coords = ut.get_minmax_coords_from_points(self.data, self.xy_epsg, padding)
        return minmax_coords

    def get_region(self,padding=[]):
        """
        It gets the region according to the limits in the catalog
        
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

    def query(self,starttime=None,
                endtime=None,
                ev_ids=None,agencies=None,
                mag_lims=None,region_lims=None,
                general_region=None,
                region_from_src=None):
        
        self.filter("origin_time",starttime,endtime)
        
        if (ev_ids is not None) and (len(self) !=0):
            self.select_data({"ev_id":ev_ids})
            
        if (agencies is not None) and (len(self) !=0):
            self.select_data({"agency":agencies}) #agencies is a list
        
        if (mag_lims is not None) and (len(self) !=0):
            self.filter("magnitude",start=mag_lims[0],
                               end=mag_lims[1])
        
        if (region_lims is not None) and (len(self) !=0):
            self.filter_rectangular_region(region_lims)
            
        if (general_region is not None) and (len(self) !=0):
            self.filter_general_region(general_region)
            
        if (region_from_src is not None) and (len(self) !=0):
            lat,lon, r_max, az_max =  region_from_src
            self.filter_by_r_az(latitude=lat,
                            longitude=lon,
                            r=r_max,
                            az=az_max)
        
        return self    
        
    def get_picks(self, picks_path, ev_ids=None,
               starttime=None,endtime=None,
               mag_lims = None,
               general_region=None,
               region_lims=None, agencies=None,
               region_from_src = None,
               author=None,
               stations=None
               ):
        self.query(starttime=starttime,
                    endtime=endtime,
                    ev_ids=ev_ids,
                    mag_lims=mag_lims,
                    agencies=agencies,
                    region_lims=region_lims,
                    general_region=general_region,
                    region_from_src=region_from_src)
        
            
        if len(self) != 0:
            ev_ids = self.data["ev_id"].to_list()
            # print(picks_path,ev_ids)
            print(ev_ids)
            picks = load_dataframe_from_sqlite(db_name=picks_path,
                                      tables=ev_ids,debug=True
                                      )
            # print(picks)
            if picks.empty:
                picks = pd.DataFrame(columns=['ev_id', 'network', 
                                          'station', 'arrival_time',
                                          'phase_hint'])
            else:
                if "arrival_time" not in picks.columns.to_list():
                    picks["arrival_time"] = pd.to_datetime(picks["time"]) #due to the database 
                else:
                    picks["arrival_time"] = pd.to_datetime(picks["arrival_time"])
        else :
            picks = pd.DataFrame(columns=['ev_id', 'network', 
                                          'station', 'arrival_time',
                                          'phase_hint'])
        
        if stations is not None:
            stations_data = stations.data.copy()
            renaming = {"latitude":"station_latitude",
                        "longitude":"station_longitude",
                        "elevation":"station_elevation"}
            to_rename = {k: v for k, v in renaming.items() \
                                    if v not in stations_data.columns}
            stations_data = stations_data.rename(columns=to_rename)
            pick_columns = picks.columns.to_list()
            for key in renaming.values():
                if key in pick_columns:
                    picks.drop(key,axis=1,inplace=True)
            
            picks = pd.merge(picks,stations_data,
                            on=["network","station"],
                            )
            
            # exit()
        
        picks = Picks(picks,author=author)
        
        
        return  picks

    def project(self,startpoint,endpoint,
                width,verbose=True):
        """
        Project data onto a line

        Parameters:
        -----------
        startpoint: tuple
            (lon,lat)
        endpoint: tuple
            (lon,lat)
        width: tuple
            (w_left,w_right)
        
        """
        data = self.data
        data = data[["longitude","latitude","depth",
                    "origin_time","magnitude"]]
        
        data = data.drop_duplicates(subset=["latitude","longitude","depth"])
        data = data.dropna(subset=["latitude","longitude","depth"])

        try:
            projection = pygmt.project(
                                data=data,
                                unit=True,
                                center=startpoint,
                                endpoint=endpoint,
                                convention="pz",
                                width=width,
                                verbose=verbose
                                    )
            projection = projection.rename(columns={0:"distance",
                                    1:"depth",
                                    2:"origin_time",
                                    3:"magnitude"})
        except:
            projection = pd.DataFrame(columns =["distance","depth"])
        return projection

    def plot_map(self,
            cpt=None,
            show_cpt=True,
            profiles=[],
            fig=None):
        """
        Plot the catalog.

        Parameters:
        -----------
        fig: None or pygmt.Figure
            Basemap figure
        cpt: None or CPT
            color palette table applied to the catalog
        show_cpt: bool
            Show the color palette table
        """
        data = self.data

        if fig == None:
            fig = pygmt.Figure() 
            fig.basemap(region=self.get_region(padding=0.1),
                        projection="M12c", 
                        frame=["af","WNse"])
            fig.coast(
                    shorelines=True,
                    borders='1/1p,black',
                    frame=["afg","WNse"],
                    )
        
        info2pygmt = self.baseplot.get_info2pygmt(data)
        if self.baseplot.cmap:
            if cpt == None:
                zmin = data.depth.min()
                zmax = data.depth.max()
                cpt = CPT(color_target="depth",
                            label="depth",
                            cmap="rainbow",
                            series=[zmin,zmax],
                            reverse=True,
                            overrule_bg=True)

            info2pygmt["color"] = data[cpt.color_target]
            pygmt.makecpt(**cpt.makecpt_kwargs)
            
            if show_cpt:
                fig.colorbar(frame=f'af+l"{cpt.label}"',
                        position="JBC+e")
        fig.plot(
            x=data.longitude,
            y=data.latitude,
            **info2pygmt
        )

        for profile in profiles:
            C,D = profile.coords
            ul = ut.points_parallel_to_line(profile.coords,
                                            abs(profile.width[0]))
            cul,dul = ul
            bl = ut.points_parallel_to_line(profile.coords,
                                            abs(profile.width[1]),
                                            upper_line=False)
            cbl,dbl = bl
            
            fig.plot(x=[cbl[0], dbl[0],dul[0],cul[0],cbl[0]], 
                    y=[cbl[1], dbl[1],dul[1],cul[1],cbl[1]],
                    projection="M", pen=f"1.5p,{profile.colorline},4_2:2p")
            ln,rn = profile.name
            fig.text(x=C[0], y=C[1], text=f"{ln}",
                    font=f"10p,Helvetica,black",
                    pen="black",
                    fill="white",offset="-0.05c/0.05c")
                    # fill=None,offset="-0.05c/0.05c")
            fig.text(x=D[0], y=D[1], text=f"{rn}",
                    font=f"10p,Helvetica,black",
                    pen="black",
                    fill="white",offset="0.05c/0.05c")
                    # fill=None,offset="0.05c/0.05c")

        return fig

    def plot_profile(self,profile,
                    depth_unit,
                    cpt=None,
                    show_cpt=True,
                    fig=None,
                    verbose=True):


        startpoint = profile.coords[0]
        endpoint = profile.coords[1]
        baseprofile = profile.baseprofile
        projection = self.project(startpoint,endpoint,
                                profile.width,verbose)

        if fig == None:
            C,D = profile.coords
            max_distance,a,ba = gps2dist_azimuth(C[1],C[0],D[1],D[0])
            basemap_args = baseprofile.get_basemap_args(max_distance)

            fig = pygmt.Figure()
            fig.basemap(**basemap_args)

        if depth_unit == "m":
            projection["depth"] = projection["depth"]/1e3
        if baseprofile.output_unit == "m":
            projection["depth"] = projection["depth"]*1e3
            projection["distance"] = projection["distance"]*1e3

        info2pygmt = self.baseplot.get_info2pygmt(projection)
        if self.baseplot.cmap:
            if cpt == None:
                zmin = projection.depth.min()
                zmax = projection.depth.max()
                cpt = CPT(color_target="depth",
                            label="depth",
                            cmap="rainbow",
                            series=[zmin,zmax],
                            reverse=True,
                            overrule_bg=True)

            info2pygmt["color"] = projection[cpt.color_target]
            pygmt.makecpt(**cpt.makecpt_kwargs)
            
            if show_cpt:
                fig.colorbar(frame=f'af+l"{cpt.label}"',
                        position="JBC+e")
        fig.plot(
            x=projection.distance,
            y=projection.depth,
            **info2pygmt
        )

        return fig

    def matplot(self,color_target="depth",
            s=8,cpt="viridis",show_cpt=True,
            ax=None):
        """
        Quickly matplotlib figure

        Parameters:
        color_target: str
            target to apply cbar
        s: float
            marker size
        cpt: str
            Name of the colorbar
        show_cpt: bool
            Show color palette table.
        ax: axis
            existent axis
        """

        if ax == None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
        ax.set_aspect("equal")


        if color_target == "origin_time":
            cb = ax.scatter(self.data.longitude, self.data.latitude,
                    c=mdates.date2num(self.data[color_target]), s=s, cmap=cpt)
            
            if show_cpt:
                cpt = fig.colorbar(cb)
                cpt.ax.set_ylim(cpt.ax.get_ylim()[::-1])
                cpt.set_label(f"{color_target}")
                
                loc = mdates.AutoDateLocator()
                cpt.ax.yaxis.set_major_locator(loc)
                cpt.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        else:
            cb = ax.scatter(self.data.longitude, self.data.latitude,
                    c=self.data[color_target], s=s, cmap=cpt)
            if show_cpt:
                cpt = fig.colorbar(cb)
                cpt.ax.set_ylim(cpt.ax.get_ylim()[::-1])
                cpt.set_label(f"{color_target}")

        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        return ax

class MulCatalog():
    def __init__(self,catalogs=[],cpt=None,show_cpt=True):
        """
        Parameters:
        -----------
        catalogs: list
            list of Catalog objects
        cpt: None or CPT
            color palette table applied to the catalog
        show_cpt: bool
            Show color palette table.
        """
        self.catalogs = catalogs
        self.cpt = cpt
        self.show_cpt = show_cpt

    def __iter__(self):
        return list(self.catalogs).__iter__()

    def __nonzero__(self):
        return bool(len(self.catalogs))

    def __len__(self):
        return len(self.catalogs)
    
    def __str__(self,extended=False) -> str:
        msg = f"Catalogs ({self.__len__()} catalogs)\n"
        msg += "-"*len(msg) 

        submsgs = []
        for i,catalog in enumerate(self.__iter__(),1):
            submsg = f"{i}. "+catalog.__str__(extended=extended)
            submsgs.append(submsg)
                
        if len(self.catalogs)<=20 or extended is True:
            submsgs = "\n".join(submsgs)
        else:
            three_first_submsgs = submsgs[0:3]
            last_two_subsgs = submsgs[-2:]
            len_others = len(self.catalogs) -len(three_first_submsgs) - len(last_two_subsgs)
            submsgs = "\n".join(three_first_submsgs+\
                                [f"...{len_others} other catalogs..."]+\
                                last_two_subsgs)

        return msg+ "\n" +submsgs
        
    def __setitem__(self, index, trace):
        self.catalogs.__setitem__(index, trace)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(catalogs=self.catalogs.__getitem__(index))
        else:
            return self.catalogs.__getitem__(index)

    def __delitem__(self, index):
        return self.catalogs.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        return self.__class__(catalogs=self.catalogs[max(0, i):max(0, j):k])

    def append(self, catalog):
        """
        append a catalog
        """
        if isinstance(catalog, Catalog):
            self.catalogs.append(catalog)
        else:
            msg = 'Append only supports a single Catalog object as an argument.'
            raise TypeError(msg)
        return self

    def remove_data(self,rowval):
        """
        remove rows of each data.

        Parameters:
        -----------
        rowval : dict
            key:  
                column name
            value: 
                One or more values specified to remove
        """
        catalogs = []
        for catalog in self.catalogs:
            catalogs.append(catalog.remove_data(rowval))
        self.catalogs = catalogs
        return self

    def select_data(self,rowval):
        """
        select rows of each data.

        Parameters:
        -----------
        rowval : dict
            key:  
                column name
            value: 
                One or more values specified to select
        """
        catalogs = []
        for catalog in self.catalogs:
            catalogs.append(catalog.select_data(rowval))
        self.catalogs = catalogs
        return self

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)

    def project(self,startpoint,endpoint,
                width,verbose=True):
        projections = []
        for catalog in self.catalogs:
            projection = catalog.project(startpoint,endpoint,
                                            width,verbose)
            projections.append(projection)
        return projections

    def sort_values(self,**args):
        """
        Sort values. Take in mind that it could affect the order of the events plotted
        args: The parameters are the pd.DataFrame.sort_values parameters
        """
        catalogs = []
        for catalog in self.catalogs:
            catalogs.append(catalog.sort_values(**args))
        self.catalogs = catalogs
        return self
    
    def filter(self,key,start=None,end=None):
        """
        Filter data of the catalog.

        Parameters:
        -----------
        key: str
            Name of the column to filter
        start: int or float or datetime.datetime
            must be the same type as data[key] does
        end: int or float or datetime.datetime
            must be the same type as data[key] does
        
        """
        catalogs = []
        for catalog in self.catalogs:
            catalogs.append(catalog.filter(key,start,end))
        self.catalogs = catalogs
        return self

    def filter_general_region(self,polygon):
        """
        Filter the region of the catalog.

        Parameters:
        -----------
        polygon: list of tuples
            Each tuple is consider a point (lon,lat).
            The first point must be equal to the last point in the polygon.
        
        """
        catalogs = []
        for catalog in self.catalogs:
            catalogs.append(catalog.filter_general_region(polygon))
        self.catalogs = catalogs
        return self

    def filter_rectangular_region(self,region_lims):
        """
        Filter the region of the catalog.

        Parameters:
        -----------
        region_lims: list of 4 elements
            lonw,lone,lats,latw
        
        """
        catalogs = []
        for catalog in self.catalogs:
            catalogs.append(catalog.filter_rectangular_region(region_lims))
        self.catalogs = catalogs
        return self

    def get_region(self,padding=[]):
        """
        It gets the region according to the limits of all events as a whole

        Parameters:
        -----------
        padding: 4D-list or float or int
            list: Padding on each side of the region [lonw,lonw,lats,latn] in degrees.
            float or int: padding amount on each side of the region from 0 to 1,
                        where 1 is considered the distance on each side of the region.
        """
        lons,lats = [],[]
        for catalog in self.catalogs:
            region = catalog.get_region()
            lons.append(region[0:2])
            lats.append(region[2:])
        lons = [ x for sublist in lons for x in sublist]
        lats = [ x for sublist in lats for x in sublist]
        region = [min(lons),max(lons),min(lats),max(lats)]


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

    def plot_map(self,fig=None):

        """
        Plot the catalog.

        Parameters:
        -----------
        fig: None or pygmt.Figure
            Basemap figure
        """

        if fig == None:
            fig = pygmt.Figure() 
            fig.basemap(region=self.get_region(padding=0.1),
                        projection="M12c", 
                        frame=["af","WNse"])
            fig.coast(
                    shorelines=True,
                    borders='1/1p,black',
                    frame=["afg","WNse"],
                    )
        if self.cpt == None:
            data = []
            for catalog in self.catalogs:
                if catalog.baseplot.cmap:
                    data.append(catalog.data)
            if data:
                data = pd.concat(data)
                zmin = data.depth.min()
                zmax = data.depth.max()
                self.cpt = CPT(color_target="depth",
                            label="depth",
                            cmap="rainbow",
                            series=[zmin,zmax],
                            reverse=True,
                            overrule_bg=True)

        show_catalog_cpt = []
        for catalog in self.catalogs:
            if catalog.baseplot.cmap:
                catalog.plot_map(fig=fig,cpt=self.cpt,show_cpt=False)
                _show_cpt = True
            else:
                catalog.plot_map(fig=fig,cpt=None,show_cpt=False)
                _show_cpt = False
            show_catalog_cpt.append(_show_cpt)

        if any(show_catalog_cpt):
            if self.show_cpt:
                fig.colorbar(frame=f'af+l"{self.cpt.label}"',
                        position="JBC+e")

        return fig

    def plot_profile(self,profile,
                    depth_unit,
                    fig=None,
                    verbose=True):

        """
        Plot the catalog.

        Parameters:
        -----------
        fig: None or pygmt.Figure
            Basemap figure
        """
        baseprofile = profile.baseprofile

        if fig == None:
            C,D = profile.coords
            max_distance,a,ba = gps2dist_azimuth(C[1],C[0],D[1],D[0])
            basemap_args = baseprofile.get_basemap_args(max_distance)

            fig = pygmt.Figure()
            fig.basemap(**basemap_args)

        if self.cpt == None:
            data = []
            for catalog in self.catalogs:
                if catalog.baseplot.cmap:
                    data.append(catalog.data)
            if data:
                data = pd.concat(data)
                zmin = data.depth.min()
                zmax = data.depth.max()
                self.cpt = CPT(color_target="depth",
                            label="depth",
                            cmap="rainbow",
                            series=[zmin,zmax],
                            reverse=True,
                            overrule_bg=True)

        show_catalog_cpt = []
        for catalog in self.catalogs:
            if catalog.baseplot.cmap:
                catalog.plot_profile(fig=fig,
                                profile=profile,
                                depth_unit=depth_unit,
                                cpt=self.cpt,
                                show_cpt=False,
                                verbose=verbose)
                _show_cpt = True
            else:
                catalog.plot_profile(fig=fig,
                            profile=profile,
                            depth_unit=depth_unit,
                            cpt=None,show_cpt=False,
                            verbose=verbose)
                _show_cpt = False
            show_catalog_cpt.append(_show_cpt)

        if any(show_catalog_cpt):
            if self.show_cpt:
                fig.colorbar(frame=f'af+l"{self.cpt.label}"',
                        position="JBC+e")

        return fig

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
        y,x = ut.single_latlon2yx_in_km(self.latitude, self.longitude, xy_epsg=xy_epsg)
        
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
    
class Stations():
    def __init__(self,data,
                 xy_epsg,
                baseplot=BasePlot(color="black",
                                label="stations",
                                transparency = 0,
                                style="i0.3c",
                                pen="black"),
                basetext = BaseText(font="10p,Helvetica,black",
                                    fill=None,
                                    offset="-0.05c/0.15c")
                ) -> None:

        """
        data: DataFrame
            Dataframe with the next mandatory columns:
            'station','latitude','longitude'
        baseplot: None or BasePlot
            Control plot args
        basetext: None or BaseText
            Control text args
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
        # self.data = data[columns]
        
        # Convert latitude and longitude to x and y coordinates in kilometers
        data = ut.latlon2yx_in_km(data, xy_epsg)
        
        # Convert elevation to z coordinates in kilometers
        data["z[km]"] = data["elevation"] * -1
        
        
        self.data = data
        self.baseplot = baseplot
        self.basetext = basetext

    @property
    def empty(self):
        return self.data.empty
    
    def get_region(self,padding=[]):
        """
        It gets the region according to the limits in the catalog
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

    def __len__(self):
        return len(self.data)

    def __str__(self,extended=False) -> str:
        msg = f"Station | {self.__len__()} stations"
        if extended:
            region = list(map(lambda x: round(x,2),self.get_region()))
            msg += f"\n\tregion: {region}"
        else:
            pass
        return msg

    def remove_data(self, rowval):
        """
        remove rows to the data.

        Parameters:
        -----------
        rowval : dict
            key: 
                column name
            value: list
                values specified to remove
        """
        if not isinstance(rowval,dict):
            raise Exception("rowval must be a dictionary")
        
        mask = self.data.isin(rowval)
        mask = mask.any(axis='columns')
        self.data = self.data[~mask]
        return self

    def select_data(self, rowval):
        """
        select rows to the data.

        Parameters:
        -----------
        rowval : dict
            key: 
                column name
            value: list
                values specified to select
        """
        if not isinstance(rowval,dict):
            raise Exception("rowval must be a dictionary")
        mask = self.data.isin(rowval)
        mask = mask.any(axis='columns')
        self.data = self.data[mask]
        return self

    def append(self, data):
        """
        append data
        """
        if isinstance(data, pd.DataFrame):

            check =  all(item in data.columns.to_list() for item in self.columns)
            if not check:
                raise Exception("There is not the mandatory columns for the data in Catalog object."\
                                +"->'origin_time','latitude','longitude','depth','magnitude'")

            data = data.drop_duplicates(subset=self.columns,ignore_index=True)
            self.data = pd.concat([self.data,data])
        else:
            msg = 'Append only supports a single Dataframe object as an argument.'
            raise TypeError(msg)
        return self

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)

    def sort_values(self,**args):
        """
        Sort values. Take in mind that it could affect the order of the stations plotted
        args: The parameters are the pd.DataFrame.sort_values parameters
        """
        self.data = self.data.sort_values(**args)
        return self

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

    def get_events_by_sp(self,catalog,rmax,zmin=None,
                         picks_path=None,
                         output_folder=None):
        
        if output_folder is not None:
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
        
        sta_data = self.data.drop_duplicates("station",ignore_index=True)
        
        # events = {}
        all_events = dict((sta,[]) for sta in sta_data["station"].to_list())
        all_picks = dict((sta,[]) for sta in sta_data["station"].to_list())
        
        # Wrap the iterrows() with tqdm for progress bar
        for i, row in tqdm(sta_data.iterrows(), total=len(sta_data),
                           desc="Processing stations"):
            new_catalog = catalog.copy()
            if zmin is not None:
                new_catalog.filter("depth", start=zmin)
            new_catalog.filter_by_r_az(latitude=row.latitude,
                                    longitude=row.longitude,
                                    r=rmax)
            if new_catalog.empty:
                # print("bad",i,new_catalog)
                continue
            
            if picks_path is not None:
                picks = new_catalog.get_picks(picks_path)
                picks.select_data({"station":[row.station]})
                picks.filter_requiring_ps_phases_in_station()
                
                if picks.empty:
                    continue
                
                ev_ids = picks.data["ev_id"].to_list()
                new_catalog.select_data(rowval={"ev_id":ev_ids})
                if new_catalog.empty:
                    continue
                
                
            if output_folder is not None:
                
                sp_cat_path = os.path.join(output_folder,
                                            "catalog_sp_method.db")
                sp_picks_path = os.path.join(output_folder,
                                                "picks_sp_method.db")
                
                save_dataframe_to_sqlite(new_catalog.data,
                                            db_name=sp_cat_path,
                                        table_name=row.station)
                
                if picks_path is not None:
                    save_dataframe_to_sqlite(picks.data,
                                            db_name=sp_picks_path,
                                        table_name=row.station)
            # new_catalog = Catalog(data=new_catalog,xy_epsg=self.xy_epsg)        
            all_events[row.station].append(new_catalog)
            
            if picks_path is not None:
                all_picks[row.station].append(picks)
            
        
        return all_events,all_picks

    def plot_map(self,fig=None):
        """
        Plot the stations

        Parameters:
        -----------
        fig: None or pygmt.Figure
            Basemap figure
        """
        data = self.data

        if fig == None:
            fig = pygmt.Figure() 
            fig.basemap(region=self.get_region(padding=0.1),
                        projection="M12c", 
                        frame=["af","WNse"])
            fig.coast(
                    shorelines=True,
                    borders='1/1p,black',
                    frame=["afg","WNse"],
                    )
        
        plot_info2pygmt = self.baseplot.get_info2pygmt()
        fig.plot(
                x=data["longitude"],
                y=data["latitude"],
                **plot_info2pygmt
            )
        if self.basetext != None:
            text_info2pygmt = self.basetext.get_info2pygmt()
            fig.text(x=data["longitude"], 
                    y=data["latitude"], 
                    text=data["station"],
                    **text_info2pygmt)
        return fig

    def matplot(self,ax=None):
        """
        Quickly matplotlib figure
        """
        if ax == None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.plot(self.data.longitude, self.data.latitude, "r^", 
                ms=10, mew=1, mec="k")
        for i,row in self.data.iterrows():
            ax.text(row.longitude, row.latitude, row.station, fontsize=12)
        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        return ax

class MulStations():
    def __init__(self,stations=[]):
        """
        Parameters:
        -----------
        stations: list
            list of Station objects
        """
        self.stations = stations

    def __iter__(self):
        return list(self.stations).__iter__()

    def __nonzero__(self):
        return bool(len(self.stations))

    def __len__(self):
        return len(self.stations)
    
    def __str__(self,extended=False) -> str:
        msg = f"Stations ({self.__len__()} stations)\n"
        msg += "-"*len(msg) 

        submsgs = []
        for i,station in enumerate(self.__iter__(),1):
            submsg = f"{i}. "+station.__str__(extended=extended)
            submsgs.append(submsg)
                
        if len(self.stations)<=20 or extended is True:
            submsgs = "\n".join(submsgs)
        else:
            three_first_submsgs = submsgs[0:3]
            last_two_subsgs = submsgs[-2:]
            len_others = len(self.stations) -len(three_first_submsgs) - len(last_two_subsgs)
            submsgs = "\n".join(three_first_submsgs+\
                                [f"...{len_others} other catalogs..."]+\
                                last_two_subsgs)

        return msg+ "\n" +submsgs

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(stations=self.stations.__getitem__(index))
        else:
            return self.stations.__getitem__(index)

    def __delitem__(self, index):
        return self.stations.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        return self.__class__(stations=self.stations[max(0, i):max(0, j):k])

    def get_region(self,padding=[]):
        """
        It gets the region according to the limits of all stations as a whole

        Parameters:
        -----------
        padding: 4D-list or float or int
            list: Padding on each side of the region [lonw,lonw,lats,latn] in degrees.
            float or int: padding amount on each side of the region from 0 to 1,
                        where 1 is considered the distance on each side of the region.
        """
        lons,lats = [],[]
        for station in self.stations:
            region = station.get_region()
            lons.append(region[0:2])
            lats.append(region[2:])
        lons = [ x for sublist in lons for x in sublist]
        lats = [ x for sublist in lats for x in sublist]
        region = [min(lons),max(lons),min(lats),max(lats)]


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
    
    def append(self, stations):
        """
        append a stations
        """
        if isinstance(stations, Station):
            self.stations.append(stations)
        else:
            msg = 'Append only supports a single Stations object as an argument.'
            raise TypeError(msg)
        return self

    def remove_data(self,rowval):
        """
        remove rows of each data.

        Parameters:
        -----------
        rowval : dict
            key:  
                column name
            value: 
                One or more values specified to remove
        """
        stations = []
        for station in self.stations:
            stations.append(station.remove_data(rowval))
        self.stations = stations
        return self

    def select_data(self,rowval):
        """
        select rows of each data.

        Parameters:
        -----------
        rowval : dict
            key:  
                column name
            value: 
                One or more values specified to select
        """
        stations = []
        for station in self.stations:
            stations.append(station.select_data(rowval))
        self.stations = stations
        return self

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)    

    def sort_values(self,**args):
        """
        Sort values. Take in mind that it could affect the order of the stations plotted
        args: The parameters are the pd.DataFrame.sort_values parameters
        """
        stations = []
        for station in self.stations:
            stations.append(station.sort_values(**args))
        self.stations = stations
        return self

    def filter_general_region(self,polygon):
        """
        Filter the region of the catalog.

        Parameters:
        -----------
        polygon: list of tuples
            Each tuple is consider a point (lon,lat).
            The first point must be equal to the last point in the polygon.
        
        """
        stations = []
        for station in self.stations:
            stations.append(station.filter_general_region(polygon))
        self.stations = stations
        return self

    def filter_rectangular_region(self,region_lims):
        """
        Filter the region of the catalog.

        Parameters:
        -----------
        region_lims: list of 4 elements
            lonw,lone,lats,latw
        
        """
        stations = []
        for station in self.stations:
            stations.append(station.filter_rectangular_region(region_lims))
        self.stations = stations
        return self

    def plot_map(self,fig=None):
        """
        Plot the stations.

        Parameters:
        -----------
        fig: None or pygmt.Figure
            Basemap figure
        """

        if fig == None:
            fig = pygmt.Figure() 
            fig.basemap(region=self.get_region(padding=0.1),
                        projection="M12c", 
                        frame=["af","WNse"])
            fig.coast(
                    shorelines=True,
                    borders='1/1p,black',
                    frame=["afg","WNse"],
                    )

        for station in self.stations:
            station.plot_map(fig=fig)
        return fig

class Shape():
    def __init__(self,data,projection,
            baseplot=BasePlot(color=None, 
                            pen=["0.02c,black,-"]
                            )):
        """
        data: GeoDataFrame
            Data of the shape file
        projection: str
            EPSG projection. Example 'EPSG:4326'
        baseplot: BasePlot
            Control plot args
        """
        self.projection =  projection
        self.data = data.to_crs(projection)
        self.baseplot = baseplot

    @property
    def empty(self):
        return self.data.empty

    def __len__(self):
        return len(self.data)

    def __str__(self,extended=False) -> str:
        msg = f"Shape | {self.__len__()} geometries"
        if extended:
            region = list(map(lambda x: round(x,2),self.get_region()))
            msg += f"\n\tprojection: {self.data.geometry.crs}"
            msg += f"\n\tregion: {region}"
            for i,row in enumerate(self.data.geometry,1):
                msg += f"\n\tgeometry type #{i}: {row.geom_type}"
        else:
            pass
        return msg

    def get_region(self,padding=[]):
        """
        It gets the region according to the limits in the shape
        
        Parameters:
        -----------
        padding: 4D-list or float or int
            list: Padding on each side of the region [lonw,lonw,lats,latn] in degrees.
            float or int: padding amount on each side of the region from 0 to 1,
                        where 1 is considered the distance on each side of the region.
        """
        data = self.data
        df = data["geometry"].apply(lambda x: x.bounds)
        data = pd.DataFrame(df.tolist(),columns=["min_x","min_y","max_x","max_y"])                                    

        lonw,lone = data.min_x.min(),data.max_x.max()
        lats,latn = data.min_y.min(),data.max_y.max()
        
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

    def to_crs(self,projection):
        """
        projection: str
            EPSG projection. Example 'EPSG:4326'
        """
        self.data.to_crs(projection)

    def append(self, data):
        """
        append data
        """
        if isinstance(data, gpd.GeoDataFrame):
            data = data.drop_duplicates(subset=self.columns,ignore_index=True)
            self.data = pd.concat([self.data,data])
        else:
            msg = 'Append only supports a single GeoDataframe object as an argument.'
            raise TypeError(msg)
        return self

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)

    def sort_values(self,**args):
        """
        Sort values. Take in mind that it could affect the order of the geometry plotted
        args: The parameters are the pd.DataFrame.sort_values parameters
        """
        self.data = self.data.sort_values(**args)
        return self

    def remove_data(self, rowval):
        """
        remove rows to the data.

        Parameters:
        -----------
        rowval : dict
            key: 
                column name
            value: list
                values specified to remove
        """
        if not isinstance(rowval,dict):
            raise Exception("rowval must be a dictionary")
        
        mask = self.data.isin(rowval)
        mask = mask.any(axis='columns')
        self.data = self.data[~mask]
        return self
    
    def select_data(self, rowval):
        """
        select rows to the data.

        Parameters:
        -----------
        rowval : dict
            key: 
                column name
            value: list
                values specified to select
        """
        if not isinstance(rowval,dict):
            raise Exception("rowval must be a dictionary")
        mask = self.data.isin(rowval)
        mask = mask.any(axis='columns')
        self.data = self.data[mask]
        return self

    def matplot(self,**args):
        """
        args: See GeoDataFrame args.
        """
        self.data.plot(**args)

    def plot_map(self,fig=None):
        """
        Plot the stations

        Parameters:
        -----------
        fig: None or pygmt.Figure
            Basemap figure
        """

        if fig == None:
            fig = pygmt.Figure() 
            fig.basemap(region=self.get_region(padding=0.1),
                        projection="M12c", 
                        frame=["af","WNse"])
            fig.coast(
                    shorelines=True,
                    borders='1/1p,black',
                    frame=["afg","WNse"],
                    )
        
        info2pygmt = self.baseplot.get_info2pygmt()
        fig.plot(self.data,**info2pygmt)
        return fig

class MulShape():
    def __init__(self,shapes=[]):
        """
        Parameters:
        -----------
        shapes: list
            list of Shape objects
        """
        self.shapes = shapes
    
    def __iter__(self):
        return list(self.shapes).__iter__()

    def __nonzero__(self):
        return bool(len(self.shapes))

    def __len__(self):
        return len(self.shapes)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(shapes=self.shapes.__getitem__(index))
        else:
            return self.shapes.__getitem__(index)

    def __delitem__(self, index):
        return self.shapes.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        return self.__class__(shapes=self.shapes[max(0, i):max(0, j):k])
    
    def __str__(self,extended=False) -> str:
        msg = f"Shapes ({self.__len__()} shapes)\n"
        msg += "-"*len(msg) 

        submsgs = []
        for i,shape in enumerate(self.__iter__(),1):
            submsg = f"{i}. "+shape.__str__(extended=extended)
            submsgs.append(submsg)
                
        if len(self.shapes)<=20 or extended is True:
            submsgs = "\n".join(submsgs)
        else:
            three_first_submsgs = submsgs[0:3]
            last_two_subsgs = submsgs[-2:]
            len_others = len(self.shapes) -len(three_first_submsgs) - len(last_two_subsgs)
            submsgs = "\n".join(three_first_submsgs+\
                                [f"...{len_others} other shapes..."]+\
                                last_two_subsgs)

        return msg+ "\n" +submsgs

    def get_region(self,padding=[]):
        """
        It gets the region according to the limits of all shapes as a whole

        Parameters:
        -----------
        padding: 4D-list or float or int
            list: Padding on each side of the region [lonw,lonw,lats,latn] in degrees.
            float or int: padding amount on each side of the region from 0 to 1,
                        where 1 is considered the distance on each side of the region.
        """
        lons,lats = [],[]
        for shape in self.shapes:
            region = shape.get_region()
            lons.append(region[0:2])
            lats.append(region[2:])
        lons = [ x for sublist in lons for x in sublist]
        lats = [ x for sublist in lats for x in sublist]
        region = [min(lons),max(lons),min(lats),max(lats)]


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

    def append(self, shapes):
        """
        append a shapes
        """
        if isinstance(shapes, Shape):
            self.shapes.append(shapes)
        else:
            msg = 'Append only supports a single Shape object as an argument.'
            raise TypeError(msg)
        return self
    
    def to_crs(self,projection):
        """
        projection: str
            EPSG projection. Example 'EPSG:4326'
        """
        shapes = []
        for shape in self.shapes:
            shapes.append(shape.to_crs(projection))
        self.shapes = shapes
        return self

    def remove_data(self,rowval):
        """
        remove rows of each data.

        Parameters:
        -----------
        rowval : dict
            key:  
                column name
            value: 
                One or more values specified to remove
        """
        shapes = []
        for shape in self.shapes:
            shapes.append(shape.remove_data(rowval))
        self.shapes = shapes
        return self

    def select_data(self,rowval):
        """
        select rows of each data.

        Parameters:
        -----------
        rowval : dict
            key:  
                column name
            value: 
                One or more values specified to select
        """
        shapes = []
        for shape in self.shapes:
            shapes.append(shape.select_data(rowval))
        self.shapes = shapes
        return self

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)  
    
    def sort_values(self,**args):
        """
        Sort values. Take in mind that it could affect the order of the shapes plotted
        args: The parameters are the pd.DataFrame.sort_values parameters
        """
        shapes = []
        for shape in self.shapes:
            shapes.append(shape.sort_values(**args))
        self.shapes = shapes
        return self

    def plot_map(self,fig=None):
        """
        Plot the stations.

        Parameters:
        -----------
        fig: None or pygmt.Figure
            Basemap figure
        """

        if fig == None:
            fig = pygmt.Figure() 
            fig.basemap(region=self.get_region(padding=0.1),
                        projection="M12c", 
                        frame=["af","WNse"])
            fig.coast(
                    shorelines=True,
                    borders='1/1p,black',
                    frame=["afg","WNse"],
                    )

        for shape in self.shapes:
            shape.plot_map(fig=fig)
        return fig

class FM():
    def __init__(self,data,
                basemeca=BaseMeca(),
                ):
        """
        Parameters:
        -----------
        data: pd.DataFrame 
            Dataframe with the next mandatory columns:
            'latitude','longitude','depth','magnitude',
            'strike','dip','rake'.
            WARNING: depth must be in km
            Optionals:
                'event_name' to write text in each beachball
                'plot_latitude','plot_longitude' at which to place beachball
        
        """

        self.columns = ['longitude','latitude',
                        'depth',
                        'strike','dip','rake','magnitude',
                        ]
        self.optional_columns1 = ['longitude','latitude',
                        'depth',
                        'strike','dip','rake','magnitude',
                        'plot_longitude',
                        'plot_latitude'
                        ]
        self.optional_columns2 = ['longitude','latitude',
                        'depth',
                        'strike','dip','rake','magnitude',
                        'plot_longitude',
                        'plot_latitude',
                        'event_name']
        check =  all(item in data.columns.to_list() for item in self.columns)
        if not check:
            raise Exception("There is not the mandatory columns for the data in Catalog object."\
                            +"->'latitude','longitude','depth','magnitude'")

        data = data.drop_duplicates(subset=self.columns,ignore_index=True)

        self.data = data
        self.basemeca = basemeca

    @property
    def empty(self):
        return self.data.empty

    def __len__(self):
        return len(self.data)

    def get_region(self,padding=[]):
        """
        It gets the region according to the limits in the catalog
        
        Parameters:
        -----------
        padding: 4D-list or float or int
            list: Padding on each side of the region [lonw,lonw,lats,latn] in degrees.
            float or int: padding amount on each side of the region from 0 to 1,
                        where 1 is considered the distance on each side of the region.
        """
        lonw,lone = self.data.longitude.min(),self.data.longitude.max()
        lats,latn = self.data.latitude.min(),self.data.latitude.max()
        
        if "plot_longitude" in self.data.columns.tolist():
            lonw = min(lonw,self.data.plot_longitude.min())
            lone = max(lone,self.data.plot_longitude.max())
        if "plot_latitude" in self.data.columns.tolist():
            lats = min(lats,self.data.plot_latitude.min())
            latn = max(latn,self.data.plot_latitude.max())


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
        if extended:
            region = list(map(lambda x: round(x,2),self.get_region()))
            msg = f"Catalog | {self.__len__()} focal mechanisms "\
                    +f"\n\tdepth : {[round(self.data.depth.min(),2),round(self.data.depth.max(),2)]}"\
                    +f"\n\tmagnitude : {[round(self.data.magnitude.min(),2),round(self.data.magnitude.max(),2)]}"\
                    +f"\n\tregion: {region}"
        else:
            msg = f"Catalog | {self.__len__()} focal mechanisms "

        return msg

    def append(self, data):
        """
        append data
        """
        if isinstance(data, pd.DataFrame):

            check =  all(item in data.columns.to_list() for item in self.columns)
            if not check:
                raise Exception("There is not the mandatory columns for the data in Catalog object."\
                                +"->'origin_time','latitude','longitude','depth','magnitude'")

            data = data.drop_duplicates(subset=self.columns,ignore_index=True)
            pd.to_datetime(data.loc[:,"origin_time"]).dt.tz_localize(None)

            self.data = pd.concat([self.data,data])
        else:
            msg = 'Append only supports a single Dataframe object as an argument.'
            raise TypeError(msg)
        return self

    def remove_data(self, rowval):
        """
        remove rows to the data.

        Parameters:
        -----------
        rowval : dict
            key: 
                column name
            value: list
                values specified to remove
        """
        if not isinstance(rowval,dict):
            raise Exception("rowval must be a dictionary")
        
        mask = self.data.isin(rowval)
        mask = mask.any(axis='columns')
        self.data = self.data[~mask]
        return self
    
    def select_data(self, rowval):
        """
        select rows to the data.

        Parameters:
        -----------
        rowval : dict
            key: 
                column name
            value: list
                values specified to select
        """
        if not isinstance(rowval,dict):
            raise Exception("rowval must be a dictionary")
        mask = self.data.isin(rowval)
        mask = mask.any(axis='columns')
        self.data = self.data[mask]
        return self

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)

    def sort_values(self,**args):
        """
        Sort values. Take in mind that it could affect the order of the fms plotted
        args: The parameters are the pd.DataFrame.sort_values parameters
        """
        self.data = self.data.sort_values(**args)
        return self

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

    def project(self,startpoint,endpoint,
                width,verbose=True):
        """
        Project data onto a line

        Parameters:
        -----------
        startpoint: tuple
            (lon,lat)
        endpoint: tuple
            (lon,lat)
        width: tuple
            (w_left,w_right)
        
        """
        data = self.data
        
        data = data[["longitude","latitude","depth",
                    "magnitude","strike","dip","rake"]]
        data = data.drop_duplicates(subset=["latitude","longitude","depth"])
        data = data.dropna(subset=["latitude","longitude","depth"])
        try:
            projection = pygmt.project(
                                data=data,
                                unit=True,
                                center=startpoint,
                                endpoint=endpoint,
                                convention="pz",
                                width=width,
                                verbose=verbose
                                    )
            # n_columns = range(0,len(columns))
            # renaming = dict(zip(n_columns,columns))
            # projection = projection.rename(columns=renaming)
            projection = projection.rename(columns={0:"distance",
                                    1:"depth",
                                    2:"magnitude",
                                    3:"strike",
                                    4:"dip",
                                    5:"rake"})
        except:
            projection = pd.DataFrame(columns =["distance","depth"])
        return projection

    def plot_map(self,fig=None,
            cpt=None,
            show_cpt=True):
        """
        Plot the catalog.

        Parameters:
        -----------
        fig: None or pygmt.Figure
            Basemap figure
        cpt: None or CPT
            color palette table applied to the catalog
        show_cpt: bool
            Show the color palette table
        """
        data = self.data

        if fig == None:
            fig = pygmt.Figure() 
            fig.basemap(region=self.get_region(padding=0.1),
                        projection="M12c", 
                        frame=["af","WNse"])
            fig.coast(
                    shorelines=True,
                    borders='1/1p,black',
                    frame=["afg","WNse"],
                    )
        
        info2pygmt = self.basemeca.get_info2pygmt()
        
        try:
            data = data[self.optional_columns2]
        except:
            try:
                data = data[self.optional_columns1]
            except:
                data = data[self.columns]

        data.to_csv("./tmp_meca.txt",sep="\t",index=False,header=False)
        if self.basemeca.cmap:
            if cpt == None:
                zmin = data.depth.min()
                zmax = data.depth.max()
                cpt = CPT(color_target="depth",
                            label="depth",
                            cmap="rainbow",
                            series=[zmin,zmax],
                            reverse=True,
                            overrule_bg=True)

            info2pygmt["color"] = data[cpt.color_target]
            pygmt.makecpt(**cpt.makecpt_kwargs)
            
            if show_cpt:
                fig.colorbar(frame=f'af+l"{cpt.label}"',
                        position="JBC+e")
        
            fig.meca(
                spec="./tmp_meca.txt",
                convention="aki",
                scale = str(info2pygmt["scale"]),
                C = info2pygmt["cmap"],
                offset=True,
                transparency= info2pygmt["transparency"]
                )
        else:
            fig.meca(
                spec="./tmp_meca.txt",
                convention="aki",
                scale = str(info2pygmt["scale"]),
                G = info2pygmt["color"],
                offset=True,
                transparency= info2pygmt["transparency"],
                )
        os.remove("./tmp_meca.txt")
            
        

        return fig

class MulFM():
    def __init__(self,fms=[],cpt=None,show_cpt=True):
        """
        Parameters:
        -----------
        fms: list
            list of fm objects
        cpt: None or CPT
            color palette table applied to the fm
        """
        self.fms = fms
        self.cpt = cpt
        self.show_cpt = show_cpt

    def __iter__(self):
        return list(self.fms).__iter__()

    def __nonzero__(self):
        return bool(len(self.fms))

    def __len__(self):
        return len(self.fms)
    
    def __str__(self,extended=False) -> str:
        msg = f"fms ({self.__len__()} fms)\n"
        msg += "-"*len(msg) 

        submsgs = []
        for i,fm in enumerate(self.__iter__(),1):
            submsg = f"{i}. "+fm.__str__(extended=extended)
            submsgs.append(submsg)
                
        if len(self.fms)<=20 or extended is True:
            submsgs = "\n".join(submsgs)
        else:
            three_first_submsgs = submsgs[0:3]
            last_two_subsgs = submsgs[-2:]
            len_others = len(self.fms) -len(three_first_submsgs) - len(last_two_subsgs)
            submsgs = "\n".join(three_first_submsgs+\
                                [f"...{len_others} other fms..."]+\
                                last_two_subsgs)

        return msg+ "\n" +submsgs
        
    def __setitem__(self, index, trace):
        self.fms.__setitem__(index, trace)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(fms=self.fms.__getitem__(index))
        else:
            return self.fms.__getitem__(index)

    def __delitem__(self, index):
        return self.fms.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        return self.__class__(fms=self.fms[max(0, i):max(0, j):k])

    def append(self, fm):
        """
        append a fm
        """
        if isinstance(fm, FM):
            self.fms.append(fm)
        else:
            msg = 'Append only supports a single FM object as an argument.'
            raise TypeError(msg)
        return self

    def remove_data(self,rowval):
        """
        remove rows of each data.

        Parameters:
        -----------
        rowval : dict
            key:  
                column name
            value: 
                One or more values specified to remove
        """
        fms = []
        for fm in self.fms:
            fms.append(fm.remove_data(rowval))
        self.fms = fms
        return self

    def select_data(self,rowval):
        """
        select rows of each data.

        Parameters:
        -----------
        rowval : dict
            key:  
                column name
            value: 
                One or more values specified to select
        """
        fms = []
        for fm in self.fms:
            fms.append(fm.select_data(rowval))
        self.fms = fms
        return self

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)

    def project(self,startpoint,endpoint,
                width,verbose=True):
        projections = []
        for fm in self.fms:
            projection = fm.project(startpoint,endpoint,
                                            width,verbose)
            projections.append(projection)
        return projections

    def sort_values(self,**args):
        """
        Sort values. Take in mind that it could affect the order of the events plotted
        args: The parameters are the pd.DataFrame.sort_values parameters
        """
        fms = []
        for fm in self.fms:
            fms.append(fm.sort_values(**args))
        self.fms = fms
        return self
    
    def filter_datetime(self,starttime=None,endtime=None):
        """
        Filter the period of the fm.

        Parameters:
        -----------
        starttime: datetime.datetime
            start time
        endtime: datetime.datetime
            end time
        
        """
        fms = []
        for fm in self.fms:
            fms.append(fm.filter_datetime(starttime,endtime))
        self.fms = fms
        return self

    def filter_general_region(self,polygon):
        """
        Filter the region of the catalog.

        Parameters:
        -----------
        polygon: list of tuples
            Each tuple is consider a point (lon,lat).
            The first point must be equal to the last point in the polygon.
        
        """
        fms = []
        for fm in self.fms:
            fms.append(fm.filter_general_region(polygon))
        self.fms = fms
        return self


    def filter_rectangular_region(self,region_lims):
        """
        Filter the region of the catalog.

        Parameters:
        -----------
        region_lims: list of 4 elements
            lonw,lone,lats,latw
        
        """
        fms = []
        for fm in self.fms:
            fms.append(fm.filter_rectangular_region(region_lims))
        self.fms = fms
        return self

    def get_region(self,padding=[]):
        """
        It gets the region according to the limits of all events as a whole

        Parameters:
        -----------
        padding: 4D-list or float or int
            list: Padding on each side of the region [lonw,lonw,lats,latn] in degrees.
            float or int: padding amount on each side of the region from 0 to 1,
                        where 1 is considered the distance on each side of the region.
        """
        lons,lats = [],[]
        for fm in self.fms:
            region = fm.get_region()
            lons.append(region[0:2])
            lats.append(region[2:])
        lons = [ x for sublist in lons for x in sublist]
        lats = [ x for sublist in lats for x in sublist]
        region = [min(lons),max(lons),min(lats),max(lats)]


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

    def plot_map(self,fig=None):

        """
        Plot the fm.

        Parameters:
        -----------
        fig: None or pygmt.Figure
            Basemap figure
        """

        if fig == None:
            fig = pygmt.Figure() 
            fig.basemap(region=self.get_region(padding=0.1),
                        projection="M12c", 
                        frame=["af","WNse"])
            fig.coast(
                    shorelines=True,
                    borders='1/1p,black',
                    frame=["afg","WNse"],
                    )
        if self.cpt == None:
            data = []
            for fm in self.fms:
                if fm.basemeca.cmap:
                    data.append(fm.data)
            if data:
                data = pd.concat(data)
                zmin = data.depth.min()
                zmax = data.depth.max()
                self.cpt = CPT(color_target="depth",
                            label="depth",
                            cmap="rainbow",
                            series=[zmin,zmax],
                            reverse=True,
                            overrule_bg=True)

        show_fm_cpt = []
        for fm in self.fms:
            if fm.basemeca.cmap:
                fm.plot_map(fig=fig,cpt=self.cpt,show_cpt=False)
                _show_cpt = True
            else:
                fm.plot_map(fig=fig,cpt=None,show_cpt=False)
                _show_cpt = False
            show_fm_cpt.append(_show_cpt)

        if any(show_fm_cpt):
            if self.show_cpt:
                fig.colorbar(frame=f'af+l"{self.cpt.label}"',
                        position="JBC+e")

        return fig

class Injection():
    """
    https://blog.finxter.com/scipy-interpolate-1d-2d-and-3d/
    """
    def __init__(self,data,depth_type,
                baseplot=BasePlot(color="blue",
                            cmap=False, 
                            pen=None)):
        """
        Parameters:
        -----------
        data: pd.DataFrame
            Dataframe with the next mandatory columns:
            'min_depth','max_depth',
            optional:
            'measurement'
        depth_type: str
            'depth','TVD','MD'
        """
        self.columns = ['min_depth','max_depth']
        check =  all(item in data.columns.to_list() for item in self.columns)
        if not check:
            raise Exception("There is not the mandatory columns for the data in Injection object."\
                            +"->'min_depth','max_depth'")
        self.data = data
        self.depth_type = depth_type
        self.baseplot = baseplot
    
    @property
    def empty(self):
        return self.data.empty

    def _get_injection_trajectories(self,trajectory):
        """
        Parameters:
        -----------
        trajectory: pd.DataFrame 
            Dataframe with the next mandatory columns:
            'latitude','longitude','depth','TVD','MD'
        """
        if self.depth_type=="depth":
            depth = pd.DataFrame({"depth":trajectory.depth})
        else:
            columns = ['latitude','longitude','depth','TVD','MD']
            check =  all(item in trajectory.columns.to_list() for item in columns)
            if not check:
                raise Exception("There is not the mandatory columns for the trajectory "\
                                +"->'latitude','longitude','depth','TVD','MD'")
            
            trajectory = trajectory.drop_duplicates(subset=[self.depth_type,"depth"])
            f = interpolate.interp1d(trajectory[self.depth_type].to_numpy(), 
                                            trajectory["depth"].to_numpy(),
                                            kind="linear",
                                            fill_value="extrapolate")

            z = np.linspace(trajectory[self.depth_type].min(),
                            trajectory[self.depth_type].max(),
                            num=int(abs(trajectory[self.depth_type].max()-\
                                        trajectory[self.depth_type].min())),
                            endpoint=True)
            f_z =  f(z)
            depth = pd.DataFrame({str(self.depth_type):z,
                                "depth":f_z})


        depths = []
        for i,row in self.data.iterrows():
            _depth = depth[(depth[self.depth_type]>=row.min_depth) &\
                           (depth[self.depth_type]<=row.max_depth) ]
            if "measurement" in self.data.columns.to_list():
                _depth = _depth.assign(measurement=row.measurement)
            depths.append(_depth)
        return depths

class Well():
    def __init__(self,data,name,
                baseplot = BasePlot(
                        # size=None,
                        style="g0.3",
                        cmap=True,
                        color="black",
                        label="data",
                        transparency=0,
                        pen=f"+0.0001p+i"),
                injection = None,
                ) -> None:
        """
        Parameters:
        -----------
        data: pd.DataFrame 
            Dataframe with the next mandatory columns:
            'latitude','longitude','depth','TVD','MD'
        name: str
            Name of the well
        baseplot: BasePlot
            Control survey plot args
        injection: Injection
            injection measurement
        """
        self.columns = ['longitude','latitude','depth','TVD','MD']
        check =  all(item in data.columns.to_list() for item in self.columns)
        if not check:
            raise Exception("There is not the mandatory columns for the data in Well object."\
                            +"->'longitude','latitude','depth','TVD','MD'")
        self.data = data.sort_values("depth")
        self.name = name
        self.baseplot = baseplot
        self.injection = injection
        
    @property
    def empty(self):
        return self.data.empty

    def __len__(self):
        return len(self.data)

    def __str__(self,extended=False) -> str:
        start = (round(self.data.iloc[0].longitude,2),
                round(self.data.iloc[0].latitude,2))
                    
        if extended:
            region = list(map(lambda x: round(x,2),self.get_region()))
            msg = f"Well | starting in (lon,lat): {start} "\
                    +f"\n\tdepth : {[round(self.data.depth.min(),2),round(self.data.depth.max(),2)]}"\
                    +f"\n\tregion: {region}"
        else:
            msg = f"Well | starting in (lon,lat): {start} "

        return msg

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)

    def get_region(self,padding=[]):
        """
        It gets the region according to the limits in the well
        
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

    def sort_values(self,**args):
        """
        Sort values. Take in mind that it could affect the order of the events plotted
        args: The parameters are the pd.DataFrame.sort_values parameters
        """
        wells = []
        for well in self.wells:
            wells.append(well.sort_values(**args))
        self.wells = wells
        return self
    
    def project(self,startpoint,endpoint,
                width,with_injection=True,verbose=True):
        """
        Project data onto a line

        Parameters:
        -----------
        startpoint: tuple
            (lon,lat)
        endpoint: tuple
            (lon,lat)
        width: tuple
            (w_left,w_right)
        with_injection: bool
            True if want to get the multiple projected segments with injection
        Returns:
            if with_injection==False -> pd.Dataframe related to the projected trajectory
            else: tuple -> (pd.dataframe,list of pd.dataframes)
                the first dataframe is related to the projected trajectory
                the list of dataframes is related to the multiple projected segments with injection
        """
        data = self.data
        try:
            columns = self.columns+["water_flow"]
            data = data[columns]
        except:
            columns = self.columns
            data = data[self.columns]
        
        data = data.drop_duplicates(subset=["latitude","longitude","depth"])
        data = data.dropna(subset=["latitude","longitude","depth"])

        try:
            projection = pygmt.project(
                                data=data,
                                unit=True,
                                center=startpoint,
                                endpoint=endpoint,
                                convention="pz",
                                width=width,
                                verbose=verbose
                                    )
            n_columns = range(1,len(columns)-1)
            renaming = dict(zip(n_columns,columns[2:]))
            renaming[0] = "distance"
            projection = projection.rename(columns=renaming)
        except:
            projection = pd.DataFrame(columns =["distance","depth"])

        if with_injection:
            if not self.injection:
                return projection
            elif len(projection) <2:
                return (projection,[])

            else:
                min_distance = projection["distance"].min()
                max_distance = projection["distance"].max()
                min_depth = projection["depth"].min()
                max_depth = projection["depth"].max()

                depth = projection["depth"].to_numpy()
                distance = projection["distance"].to_numpy()
                f_depth2distance= interpolate.interp1d(depth, distance,
                                                        kind="linear",
                                                        fill_value="extrapolate")

                injection_trajectories = self.injection._get_injection_trajectories(data)
                inj_projections = []
                for inj_trajectory in injection_trajectories:
                    measurement = inj_trajectory["measurement"].to_numpy()
                    inj_depth = inj_trajectory["depth"].to_numpy()
                    inj_distance = f_depth2distance(inj_depth)
                    inj_projection = pd.DataFrame({"distance":inj_distance,
                                                    "depth":inj_depth,
                                                    "measurement":measurement})

                    inj_projection = inj_projection[ (inj_projection["distance"] >= min_distance) &
                                                    (inj_projection["distance"] <= max_distance) &  
                                                    (inj_projection["depth"] >= min_depth) & 
                                                    (inj_projection["depth"] <= max_depth) ]
                    if inj_projection.empty:
                        continue
                    inj_projections.append(inj_projection)

                    
                return (projection,inj_projections)

    def plot_map(self,fig=None,
            survey_cpt=None,
            show_survey_cpt=True,
            with_injection=True,
            injection_cpt=None,
            show_injection_cpt=True):
        """
        Plot the catalog.

        Parameters:
        -----------
        fig: None or pygmt.Figure
            Basemap figure
        cpt: None or CPT
            color palette table applied to the catalog
        show_cpt: bool
            Show the color palette table
        """
        data = self.data

        if fig == None:
            fig = pygmt.Figure() 
            fig.basemap(region=self.get_region(padding=0.1),
                        projection="M12c", 
                        frame=["af","WNse"])
            fig.coast(
                    shorelines=True,
                    borders='1/1p,black',
                    frame=["afg","WNse"],
                    )
        
        survey_info2pygmt = self.baseplot.get_info2pygmt(data)
        if self.baseplot.cmap:
            if survey_cpt == None:
                zmin = data.depth.min()
                zmax = data.depth.max()
                survey_cpt = CPT(color_target="depth",
                            label="depth",
                            cmap="roma",
                            series=[zmin,zmax],
                            reverse=True,
                            overrule_bg=True)

            survey_info2pygmt["color"] = data[survey_cpt.color_target]
            pygmt.makecpt(**survey_cpt.makecpt_kwargs)
            
            if show_survey_cpt:
                fig.colorbar(frame=f'af+l"{survey_cpt.label}"',
                        position="JBC+e")
        fig.plot(
            x=data.longitude,
            y=data.latitude,
            **survey_info2pygmt
        )

        if (self.injection != None) and (with_injection == True):

            injection_trajectories = self.injection._get_injection_trajectories(data)
            
            all_injection = pd.concat(injection_trajectories)
            injection_info2pygmt = self.injection.baseplot.get_info2pygmt(data)
            if self.injection.baseplot.cmap:

                if (injection_cpt == None) and \
                    ("measurement" in all_injection.columns.to_list()):
                    zmin = all_injection.measurement.min()
                    zmax = all_injection.measurement.max()
                    injection_cpt = CPT(color_target="measurement",
                                label="measurement",
                                cmap="cool",
                                series=[zmin,zmax],
                                reverse=True,
                                overrule_bg=True)

                injection_info2pygmt["color"] = all_injection[injection_cpt.color_target].to_numpy()
                pygmt.makecpt(**injection_cpt.makecpt_kwargs)
        
                if show_injection_cpt:
                    with pygmt.config(FORMAT_FLOAT_MAP="%.1e"):
                        fig.colorbar(frame=["af",f'y+l{injection_cpt.label}'],
                        position='JMR+o1c/0c+e')
           
            data = self.data.drop_duplicates(subset=["latitude","depth"])
            data = data.drop_duplicates(subset=["longitude","depth"])
            f_lat= interpolate.interp1d(data.depth.to_numpy(), 
                                            data.latitude.to_numpy(),
                                            kind="linear",
                                            fill_value="extrapolate")
            f_lon= interpolate.interp1d(data.depth.to_numpy(), 
                                            data.longitude.to_numpy(),
                                            kind="linear",
                                            fill_value="extrapolate")
            for injection in injection_trajectories:
                lat = f_lat(injection["depth"].to_numpy())
                lon = f_lon(injection["depth"].to_numpy())

                if self.injection.baseplot.cmap:
                    injection_info2pygmt["color"] = injection["measurement"].to_numpy()
                    injection_info2pygmt["cmap"] = True
                fig.plot(
                        x=lon,
                        y=lat,
                        **injection_info2pygmt
                        )
        return fig

    def matplot(self,ax=None):
        if ax == None:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

        ax.plot3D(self.data.longitude, self.data.latitude,
                    self.data.z, 'gray')
        ax.invert_zaxis()

class Profile():
    def __init__(self,name,coords,width,
                baseprofile) -> None:
        """
        name: str
            Name of the profile.
        coords: 2d-tuple
            2d-Tuple of two 2d-tuples.
            ((ini_lon,ini_lat),(end_lon,end_lat))
        width: 2d-tuple
            (left_width,right_width) in km.
        baseprofile: BaseProfile
            To control profile axis    
        """
        
        self.name = name
        self.coords = coords
        self.startpoint = coords[0]
        self.endpoint = coords[1]
        self.width = width
        self.baseprofile = baseprofile
        self.mulobjects = {}

    def __str__(self,extended=False) -> str:
        name = "-".join(self.name)
        if extended:
            msg = f"Profile  {name} "\
                    +f"\n\t{self.name[0]}: {self.coords[0]}"\
                    +f"\n\t{self.name[1]}: {self.coords[1]}"\
                    +f"\n\twidth: {self.width} m"\
                    +f"\n\toutput_unit: {self.baseprofile.output_unit}"
        else:
            msg = f"Profile  {name}"

        return msg

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)

    def _add_mulwell(self,mulwell,depth_unit,verbose=True):
        mulwell_name = mulwell.__class__.__name__
        projected_data = mulwell.project(startpoint=self.startpoint,
                                    endpoint=self.endpoint,
                                    width=self.width,
                                    with_injection=True,
                                    verbose=verbose)

        trajectories = []
        injections = []
        survey_baseplots = [ ]
        inj_baseplots = [ ]
        for n_well,survey_data in enumerate(projected_data):
            if survey_data.empty:
                continue

            if isinstance(survey_data,pd.DataFrame):
                trajectory,injection_segments = survey_data, []
            elif isinstance(survey_data,tuple):
                trajectory,injection_segments = survey_data
            else:
                raise Exception("Error with projected data in mullwell.project")

            well = mulwell[n_well]
            inj = well.injection

            if depth_unit == "m":
                trajectory["depth"] = trajectory["depth"]/1e3
            if self.baseprofile.output_unit == "m":
                trajectory["depth"] = trajectory["depth"]*1e3
                trajectory["distance"] = trajectory["distance"]*1e3

            for injection in injection_segments:
                if injection.empty:
                    continue

                if depth_unit == "m":
                    injection["depth"] = injection["depth"]/1e3
                if self.baseprofile.output_unit == "m":
                    injection["depth"] = injection["depth"]*1e3
                    injection["distance"] = injection["distance"]*1e3

                inj_baseplots.append(inj.baseplot)
                injections.append(injection)

            survey_baseplots.append(well.baseplot)
            trajectories.append(trajectory)

        survey_info = {"projections":trajectories,"baseplots":survey_baseplots,
                "cpt":mulwell.survey_cpt,
                "show_cpt":mulwell.show_survey_cpt}
        injection_info = {"projections":injections,"baseplots":inj_baseplots,
                "cpt":mulwell.injection_cpt,
                "show_cpt":mulwell.show_injection_cpt}

        self.mulobjects[mulwell_name] = survey_info
        self.mulobjects["Injection"] = injection_info

    def _add_mulcatalog(self,mulcatalog,depth_unit,
                            verbose=True ):
        mulcatalog_name = mulcatalog.__class__.__name__

        projections = mulcatalog.project(self.startpoint,self.endpoint,
                                    self.width,verbose=verbose)

        for projection in projections:
            if depth_unit == "m":
                projection["depth"] = projection["depth"]/1e3
            if self.baseprofile.output_unit == "m":
                projection["depth"] = projection["depth"]*1e3
                projection["distance"] = projection["distance"]*1e3
        
        baseplots = [ x.baseplot for x in mulcatalog]
        
        info = {"projections":projections,"baseplots":baseplots,"cpt":mulcatalog.cpt,
                "show_cpt":mulcatalog.show_cpt}


        self.mulobjects[mulcatalog_name] = info

    def _add_mulfm(self,mulfm,depth_unit,
                            verbose=True ):
        mulcatalog_name = mulfm.__class__.__name__

        projections = mulfm.project(self.startpoint,self.endpoint,
                                    self.width,verbose=verbose)
        for projection in projections:
            if depth_unit == "m":
                projection["depth"] = projection["depth"]/1e3
            if self.baseprofile.output_unit == "m":
                projection["depth"] = projection["depth"]*1e3
                projection["distance"] = projection["distance"]*1e3
        
        baseplots = [ x.basemeca for x in mulfm]
        
        info = {"projections":projections,"baseplots":baseplots,"cpt":mulfm.cpt,
                "show_cpt":mulfm.show_cpt}


        self.mulobjects[mulcatalog_name] = info

    def add_mulobject(self,mulobject,depth_unit,verbose=True):
        mulcatalog_name = mulobject.__class__.__name__
        if mulcatalog_name == "MulWell":
            self._add_mulwell(mulobject,depth_unit,verbose)
        elif mulcatalog_name == "MulCatalog":
            self._add_mulcatalog(mulobject,depth_unit,verbose)
        elif mulcatalog_name == "MulFM":
            self._add_mulfm(mulobject,depth_unit,verbose)

    def plot_in_map(self,fig,colorline="magenta",rescale=False):
        """
        add profiles in figure
        """
        ul = ut.points_parallel_to_line(self.coords,
                                        abs(self.width[0]))
        cul,dul = ul
        bl = ut.points_parallel_to_line(self.coords,
                                        abs(self.width[1]),
                                        upper_line=False)
        cbl,dbl = bl
        x = [cbl[0], dbl[0],dul[0],cul[0],cbl[0]]
        y = [cbl[1], dbl[1],dul[1],cul[1],cbl[1]]

        if rescale:
            region = [min(x),max(x),min(y),max(y)]
        else:
            region =None

        fig.plot(x=[cbl[0], dbl[0],dul[0],cul[0],cbl[0]], 
                y=[cbl[1], dbl[1],dul[1],cul[1],cbl[1]],
                pen=f"1.5p,{colorline},4_2:2p",region=region)
        ln,rn = self.name
        fig.text(x=self.startpoint[0], y=self.startpoint[1], text=f"{ln}",
                font=f"10p,Helvetica,black",
                pen="black",
                fill="white",offset="-0.05c/0.05c")
                # fill=None,offset="-0.05c/0.05c")
        fig.text(x=self.endpoint[0], y=self.endpoint[1], text=f"{rn}",
                font=f"10p,Helvetica,black",
                pen="black",
                fill="white",offset="0.05c/0.05c")
                # fill=None,offset="0.05c/0.05c")
        return fig

    def plot(self,fig=None):

        max_distance,a,ba = gps2dist_azimuth(self.startpoint[1],self.startpoint[0],
                                            self.endpoint[1],self.endpoint[0])
        basemap_args = self.baseprofile.get_basemap_args(max_distance)

        if fig == None:
            fig = pygmt.Figure()
        else:
            if self.baseprofile.reverse_xy:
                basemap_args["projection"] = "x?/?"
            else:
                basemap_args["projection"] = "x?/-?"
                
        fig.basemap(**basemap_args)
        mulobjects = self.mulobjects.items()

        n_showed_cpt = 0
        for mulobject_name, info in mulobjects:
            if not info["cpt"]:
                if mulobject_name == "Injection":
                    zmin = [x.measurement.min() for x in info["projections"]]
                    zmax = [x.measurement.max() for x in info["projections"]]
                    if (not zmin) or (not zmax):
                        continue
                    zmin = min(zmin)
                    zmax = max(zmax)
                    if zmin == zmax:
                        zmin = zmin-0.01
                        zmax = zmax+0.01
                    cpt = CPT(color_target="measurement",
                                label="measurement",
                                cmap="cool",
                                series=[zmin,zmax],
                                reverse=True,
                                overrule_bg=True)
                else:
                    zmin = [x.depth.min() for x in info["projections"]]
                    zmax = [x.depth.max() for x in info["projections"]]
                    if (not zmin) or (not zmax):
                        continue
                    zmin = min(zmin)
                    zmax = max(zmax)
                    if zmin == zmax:
                        zmin = zmin-0.01
                        zmax = zmax+0.01
                    cpt = CPT(color_target="depth",
                                label="Depth",
                                cmap="rainbow",
                                series=[zmin,zmax],
                                reverse=True,
                                overrule_bg=True)
            else:
                cpt = info["cpt"]

            show_cpt = False
            for i,data in enumerate(info["projections"]):
                if data.empty:
                    continue
                if mulobject_name == "MulFM":
                    info2pygmt = info["baseplots"][i].get_info2pygmt()
                    pygmt.makecpt(**cpt.makecpt_kwargs)
                    data = data.rename(columns={"distance":"longitude",
                                                "depth":"latitude"})
                    data["depth"] = data["latitude"]
                    data = data[['longitude','latitude',
                                'depth',
                                'strike','dip','rake','magnitude',
                                ]]
                    data.to_csv("./tmp_meca.txt",sep="\t",index=False,header=False)
                    if info2pygmt["cmap"] == True:
                        show_cpt = True
                        fig.meca(
                            spec="./tmp_meca.txt",
                            convention="aki",
                            scale = str(info2pygmt["scale"]),
                            C = info2pygmt["cmap"],
                            offset=True,
                            transparency= info2pygmt["transparency"]
                            )
                    else:
                        fig.meca(
                            spec="./tmp_meca.txt",
                            convention="aki",
                            scale = str(info2pygmt["scale"]),
                            G = info2pygmt["color"],
                            offset=True,
                            transparency= info2pygmt["transparency"]
                            )
                    os.remove("./tmp_meca.txt")
                else:
                    info2pygmt = info["baseplots"][i].get_info2pygmt(data)
                    if info2pygmt["cmap"] == True:
                        show_cpt = True
                        info2pygmt["color"] = data[cpt.color_target]
                    pygmt.makecpt(**cpt.makecpt_kwargs)
                    fig.plot(x=data.distance,
                        y=data.depth,
                        **info2pygmt)
                    
                
            if show_cpt:
                n_showed_cpt += 1
                if n_showed_cpt == 1: 
                    if cpt.makecpt_kwargs["series"][1]/1e4 < 1:
                        fig.colorbar(frame=f'af+l"{cpt.label}"',
                                    position="JBC+e")
                    else:
                        with pygmt.config(FORMAT_FLOAT_MAP="%.1e"):
                            fig.colorbar(frame=f'af+l"{cpt.label}"',
                                        position="JBC+e")

                elif n_showed_cpt == 2: 
                    if cpt.makecpt_kwargs["series"][1]/1e4 < 1:
                        fig.colorbar(frame=["af",f'y+l{cpt.label}'],
                                    position='JMR+o1c/0c+e')
                    else:
                        with pygmt.config(FORMAT_FLOAT_MAP="%.1e"):
                            fig.colorbar(frame=["af",f'y+l{cpt.label}'],
                                        position='JMR+o1c/0c+e')

        return fig

class MulWell():
    def __init__(self,wells=[],
                survey_cpt=None,
                show_survey_cpt=True,
                injection_cpt=None,
                show_injection_cpt=True):
        """
        Parameters:
        -----------
        wells: list
            list of Well objects
        cpt: None or CPT
            color palette table applied to the catalog
        show_cpt: bool
            Show color palette table.
        """
        self.wells = wells
        self.survey_cpt=survey_cpt
        self.show_survey_cpt=show_survey_cpt
        self.injection_cpt=injection_cpt
        self.show_injection_cpt=show_injection_cpt

    def __iter__(self):
        return list(self.wells).__iter__()

    def __nonzero__(self):
        return bool(len(self.wells))

    def __len__(self):
        return len(self.wells)

    def __str__(self,extended=False) -> str:
        msg = f"MulWell ({self.__len__()} wells)\n"
        msg += "-"*len(msg) 

        submsgs = []
        for i,well in enumerate(self.__iter__(),1):
            submsg = f"{i}. "+well.__str__(extended=extended)
            submsgs.append(submsg)
                
        if len(self.wells)<=20 or extended is True:
            submsgs = "\n".join(submsgs)
        else:
            three_first_submsgs = submsgs[0:3]
            last_two_subsgs = submsgs[-2:]
            len_others = len(self.wells) -len(three_first_submsgs) - len(last_two_subsgs)
            submsgs = "\n".join(three_first_submsgs+\
                                [f"...{len_others} other wells..."]+\
                                last_two_subsgs)

        return msg+ "\n" +submsgs
    
    def __setitem__(self, index, trace):
        self.wells.__setitem__(index, trace)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(wells=self.wells.__getitem__(index))
        else:
            return self.wells.__getitem__(index)

    def __delitem__(self, index):
        return self.wells.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        return self.__class__(wells=self.wells[max(0, i):max(0, j):k])

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)

    def append(self, well):
        """
        append a well
        """
        if isinstance(well, Well):
            self.wells.append(well)
        else:
            msg = 'Append only supports a single Catalog object as an argument.'
            raise TypeError(msg)
        return self

    def project(self,startpoint,endpoint,
                width,with_injection=True,verbose=True):
        projections = []
        for well in self.wells:
            projection = well.project(startpoint,endpoint,
                                    width,with_injection,verbose)
            projections.append(projection)
        return projections

    def get_region(self,padding=[]):
        """
        It gets the region according to the limits of all events as a whole

        Parameters:
        -----------
        padding: 4D-list or float or int
            list: Padding on each side of the region [lonw,lonw,lats,latn] in degrees.
            float or int: padding amount on each side of the region from 0 to 1,
                        where 1 is considered the distance on each side of the region.
        """
        lons,lats = [],[]
        for well in self.wells:
            region = well.get_region()
            lons.append(region[0:2])
            lats.append(region[2:])
        lons = [ x for sublist in lons for x in sublist]
        lats = [ x for sublist in lats for x in sublist]
        region = [min(lons),max(lons),min(lats),max(lats)]


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

    def plot_map(self,fig=None,with_injection=True):

        """
        Plot the catalog.

        Parameters:
        -----------
        fig: None or pygmt.Figure
            Basemap figure
        """

        if fig == None:
            fig = pygmt.Figure() 
            fig.basemap(region=self.get_region(padding=0.1),
                        projection="M12c", 
                        frame=["af","WNse"])
            fig.coast(
                    shorelines=True,
                    borders='1/1p,black',
                    frame=["afg","WNse"],
                    )

        if self.survey_cpt == None:
            data = []
            for well in self.wells:
                if well.baseplot.cmap:
                    data.append(well.data)
            if data:
                data = pd.concat(data)
                zmin = data.depth.min()
                zmax = data.depth.max()
                self.survey_cpt = CPT(color_target="depth",
                            label="depth",
                            cmap="rainbow",
                            series=[zmin,zmax],
                            reverse=True,
                            overrule_bg=True)
        if self.injection_cpt == None:
            data = []
            for well in self.wells:
                if well.injection != None:
                    if well.injection.baseplot.cmap:
                        data.append(well.injection.data)
            if data:
                data = pd.concat(data)
                zmin = data.measurement.min()
                zmax = data.measurement.max()
                self.injection_cpt = CPT(color_target="measurement",
                            label="measurement",
                            cmap="cool",
                            series=[zmin,zmax],
                            reverse=True,
                            overrule_bg=True)

        show_survey_cpt = []
        show_injection_cpt = []
        for well in self.wells:
            if well.baseplot.cmap:
                if well.injection != None:
                    if well.injection.baseplot.cmap:
                        well.plot_map(fig=fig,survey_cpt=self.survey_cpt,show_survey_cpt=False,
                                    with_injection=with_injection,injection_cpt=self.injection_cpt,
                                    show_injection_cpt=False)
                        _show_injection_cpt = True
                    else:
                        well.plot_map(fig=fig,survey_cpt=self.survey_cpt,show_survey_cpt=False,
                                    with_injection=with_injection,injection_cpt=None,
                                    show_injection_cpt=False)
                        _show_injection_cpt = False
                else:
                    _show_injection_cpt = False

                _show_survey_cpt = True
            else:
                well.plot_map(fig=fig,survey_cpt=None,show_survey_cpt=False,
                            with_injection=with_injection)
                if well.injection != None:
                    if well.injection.baseplot.cmap:
                        well.plot_map(fig=fig,survey_cpt=None,show_survey_cpt=False,
                                    with_injection=with_injection,injection_cpt=self.injection_cpt,
                                    show_injection_cpt=False)
                        _show_injection_cpt = True
                    else:
                        well.plot_map(fig=fig,survey_cpt=None,show_survey_cpt=False,
                                    with_injection=with_injection,injection_cpt=None,
                                    show_injection_cpt=False)
                        _show_injection_cpt = False
                else:
                    _show_injection_cpt = False

                _show_survey_cpt = False
            show_survey_cpt.append(_show_survey_cpt)
            show_injection_cpt.append(_show_injection_cpt)

        if any(show_survey_cpt):
            if self.show_survey_cpt:
                pygmt.makecpt(**self.survey_cpt.makecpt_kwargs)
                fig.colorbar(frame=f'af+l"{self.survey_cpt.label}"',
                        position="JBC+e")
        if any(show_injection_cpt):
            if self.show_injection_cpt:
                pygmt.makecpt(**self.injection_cpt.makecpt_kwargs)
                with pygmt.config(FORMAT_FLOAT_MAP="%.1e"):
                    fig.colorbar(frame=["af",f'y+l{self.injection_cpt.label}'],
                    position='JMR+o1c/0c+e')
        return fig

class MulProfile():
    def __init__(self,profiles=[]):
        """
        Parameters:
        -----------
        profiles: list
            list of Shape objects
        """
        self.profiles = profiles
        
    def __iter__(self):
        return list(self.profiles).__iter__()

    def __nonzero__(self):
        return bool(len(self.profiles))

    def __len__(self):
        return len(self.profiles)
    
    def __str__(self,extended=False) -> str:
        msg = f"Profiles ({self.__len__()} profiles)\n"
        msg += "-"*len(msg) 

        submsgs = []
        for i,profile in enumerate(self.__iter__(),1):
            submsg = f"{i}. "+profile.__str__(extended=extended)
            submsgs.append(submsg)
                
        if len(self.profiles)<=20 or extended is True:
            submsgs = "\n".join(submsgs)
        else:
            three_first_submsgs = submsgs[0:3]
            last_two_subsgs = submsgs[-2:]
            len_others = len(self.profiles) -len(three_first_submsgs) - len(last_two_subsgs)
            submsgs = "\n".join(three_first_submsgs+\
                                [f"...{len_others} other profiles..."]+\
                                last_two_subsgs)

        return msg+ "\n" +submsgs

    def __setitem__(self, index, trace):
        self.profiles.__setitem__(index, trace)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(profiles=self.profiles.__getitem__(index))
        else:
            return self.profiles.__getitem__(index)

    def __delitem__(self, index):
        return self.profiles.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        return self.__class__(profiles=self.profiles[max(0, i):max(0, j):k])

    def append(self, profile):
        """
        append a profile
        """
        if isinstance(profile, profile):
            self.profiles.append(profile)
        else:
            msg = 'Append only supports a single Profile object as an argument.'
            raise TypeError(msg)
        return self

    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)

    def add_mulobject(self,mulobject,depth_unit,verbose=True):
        for profile in self.profiles:
            profile.add_mulobject(mulobject,depth_unit,verbose)

    def plot_in_map(self,fig,colorline="magenta",rescale=False):
        """
        add profiles in figure
        """
        for profile in self.profiles:
            fig = profile.plot_in_map(fig,colorline,rescale)
        return fig

    def plot(self,nrows=None,ncols=None,
                frame="WSrt",
                subsize = ("12c", "12c"),
                figsize=None,
                margins=["1c","1c"],**kwargs):
        """
        kwargs: pygmt subplot kwargs
        """

        if (nrows != None) and (ncols!=None):
            if nrows*ncols < len(self.profiles):
                raise Exception(f"nrows + ncols must be equal to {len(self.profiles)}")
        elif (nrows != None) or (ncols!=None):
            if nrows:
                ncols = len(self.profiles) - nrows
            else:
                nrows = len(self.profiles) - ncols
        else:
            n = len(self.profiles)
            n_square = np.sqrt(n)
            ncols = int(n_square )
            nrows = int(n_square )
            if n ==2:
                ncols = 1
                nrows = 2
            elif n%n_square != 0:
                ncols += 1
                nrows += 1
        
        fig = pygmt.Figure()
        with pygmt.clib.Session() as session:
            session.call_module('gmtset', 'FONT 10p')
            subplot =   fig.subplot(
                        nrows=nrows, ncols=ncols, 
                        subsize=subsize, 
                        figsize=figsize,
                        frame=frame,
                        margins=margins
                        ) 
        with subplot:
            for f,_profile in enumerate(self.profiles):
                print(f+1,"->",_profile.name)
                profile = _profile.copy()
                with fig.set_panel(panel=f):
                    x = profile.plot(fig=fig)
                    x.text(
                            position="BR",
                            text=f"{profile.name[0]}-{profile.name[1]}",
                            font="13p,Helvetica-Bold,red",
                        )
                    x.plot(x=0,y=0,pen=None,transparency=100,label=".") #only to remove the error caused by subplot legend
                    x.legend(transparency=100) #only to remove the error caused by subplot legend
        return fig

if __name__ == "__main__":
    cat = Catalog(data="hola")
    print(cat.to_dict())